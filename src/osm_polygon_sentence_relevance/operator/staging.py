"""Remote checkout and immutable artifact staging command generation."""

from __future__ import annotations

import json
import os
import shlex
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from osm_polygon_sentence_relevance.operator.config import (
    V2_LOGIT_PROMPT_VERSION,
    OperatorConfig,
    Stage,
)
from osm_polygon_sentence_relevance.operator.relay_transport import RemoteTransfer
from osm_polygon_sentence_relevance.operator.ssh import SshClient
from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout


@dataclass(frozen=True, slots=True)
class StagingResult:
    """Validated reusable remote layout."""

    layout: RemoteLayout
    reused: bool


@dataclass(frozen=True, slots=True)
class LabelAssets:
    """Pinned labeling artifacts staged below the managed run root."""

    input_parquet: PurePosixPath
    model_file: PurePosixPath
    tokenizer_dir: PurePosixPath
    llama_server_ready: bool


def _q(value: object) -> str:
    return shlex.quote(str(value))


class Stager:
    """Prepare one clean detached checkout on persistent remote storage."""

    def __init__(self, ssh: SshClient) -> None:
        self._ssh = ssh

    def prepare(self, config: OperatorConfig, layout: RemoteLayout) -> StagingResult:
        """Create/reuse a locked checkout without executing model inference."""

        extras = "--extra hub"
        if config.stage in {Stage.SPLIT, Stage.ALL} and not (
            config.prompt_version == V2_LOGIT_PROMPT_VERSION
            and config.stage is Stage.ALL
        ):
            extras += " --extra segmentation"
        if config.stage in {Stage.LABEL, Stage.ALL}:
            # Label selection uses H3 and the operator's runtime support.
            extras += " --extra operator"
        repo_url = (
            "https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance.git"
        )
        marker = json.dumps(
            {"schema_version": 1, "run_id": config.run_id, "status": "active"},
            sort_keys=True,
            separators=(",", ":"),
        )
        # ``source_commit`` is the immutable data/checkpoint identity.  The
        # executable checkout may be a newer behavior-preserving revision
        # when resuming a run, so keep the two contracts separate here.
        checkout_commit = config.execution_commit or config.source_commit
        script = f"""
set -euo pipefail
umask 077
root={_q(layout.root)}
repo={_q(layout.repo)}
mkdir -p -m 0700 "$root" {_q(layout.hf_home)} {_q(layout.logs)}
printf '%s\n' \
  {_q(marker)} \
  >"$root/.operator-managed.json"
chmod 0600 "$root/.operator-managed.json"
reused=false
if [ ! -d "$repo/.git" ]; then
  git clone --no-tags {_q(repo_url)} "$repo"
else
  reused=true
fi
git -C "$repo" fetch --no-tags origin main
git -C "$repo" cat-file -e {_q(checkout_commit)}^{{commit}}
git -C "$repo" checkout --detach {_q(checkout_commit)}
test -z "$(git -C "$repo" status --porcelain)"
UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ]; then UV_BIN="$HOME/.local/bin/uv"; fi
test -x "$UV_BIN"
UV_CACHE_DIR="$root/uv-cache"
export UV_CACHE_DIR
cleanup_uv_cache() {{
  [ "$UV_CACHE_DIR" = "$root/uv-cache" ] || exit 70
  [ ! -L "$UV_CACHE_DIR" ] || exit 70
  rm -rf -- "$UV_CACHE_DIR"
}}
trap cleanup_uv_cache EXIT
"$UV_BIN" sync --locked --no-dev {extras} --project "$repo"
egg_info="$repo/src/osm_polygon_sentence_relevance.egg-info"
[ ! -L "$egg_info" ]
if [ -d "$egg_info" ]; then
  rm -rf -- "$egg_info"
elif [ -e "$egg_info" ]; then
  exit 70
fi
cleanup_uv_cache
trap - EXIT
printf 'STAGING_OK reused=%s\\n' "$reused"
""".strip()
        result = self._ssh.run(script)
        if "STAGING_OK reused=" not in result.stdout:
            raise RuntimeError("remote staging did not return its success marker")
        return StagingResult(layout, "reused=true" in result.stdout)

    def clean_generated_python_caches(self, layout: RemoteLayout) -> None:
        """Remove only Python caches before a guarded job submission.

        The checkout guard rejects unexpected ignored entries. Python imports
        can recreate ``__pycache__`` after staging, so cleanup must happen
        immediately before every allocation. The command is confined to the
        managed checkout's ``src`` and ``scripts`` trees and refuses a
        symlinked repository root.
        """

        repo = _q(layout.repo)
        script = f"""
set -euo pipefail
repo={repo}
[ -d "$repo/.git" ]
[ ! -L "$repo" ]
for tree in "$repo/src" "$repo/scripts"; do
  if [ -d "$tree" ] && [ ! -L "$tree" ]; then
    find -P "$tree" -type d -name __pycache__ -prune -exec rm -rf -- {{}} +
  fi
done
printf 'PYTHON_CACHES_CLEAN\\n'
""".strip()
        result = self._ssh.run(script)
        if "PYTHON_CACHES_CLEAN" not in result.stdout:
            raise RuntimeError("remote Python cache cleanup failed")

    def prepare_label_assets(
        self,
        config: OperatorConfig,
        layout: RemoteLayout,
        *,
        download_input: bool = True,
    ) -> LabelAssets:
        """Download/reuse pinned Hub files and verify the model digest."""

        if config.input_dataset_revision is None:
            raise ValueError("immutable input revision is required")
        input_dir = layout.root / "input"
        model_dir = layout.root / "model"
        tokenizer_dir = layout.root / "tokenizer"
        input_file = input_dir / "sentences.parquet"
        model_file = model_dir / config.label_model_file
        input_download = (
            """
input_target_path = pathlib.Path(input_target)
if not (
    input_target_path.is_file()
    and input_target_path.stat().st_size > 0
):
    input_source = hf_hub_download(
        repo_id=dataset, repo_type="dataset", revision=input_revision,
        filename="sentences.parquet",
    )
    if not input_target_path.exists():
        shutil.copyfile(input_source, input_target_path)
"""
            if download_input
            else "pathlib.Path(input_target).touch(exist_ok=True)\n"
        )
        script = f"""
set -euo pipefail
umask 077
mkdir -p -m 0700 {_q(input_dir)} {_q(model_dir)} {_q(tokenizer_dir)}
token_file={_q(layout.hf_token)}
[ -f "$token_file" ] && [ ! -L "$token_file" ] && [ "$(stat -c %a -- "$token_file")" = 600 ]
export HF_TOKEN="$(cat -- "$token_file")"
[ -n "$HF_TOKEN" ]
python={_q(layout.repo / ".venv/bin/python")}
"$python" - {_q(config.output_dataset_id)} {_q(config.input_dataset_revision)} \
  {_q(input_file)} {_q(config.label_model_repo_id)} \
  {_q(config.label_model_revision)} {_q(config.label_model_file)} \
  {_q(model_dir)} {_q(config.tokenizer_repo_id)} \
  {_q(config.tokenizer_revision)} {_q(tokenizer_dir)} <<'PY'
import pathlib
import shutil
import sys
from huggingface_hub import hf_hub_download, snapshot_download

(dataset, input_revision, input_target, model_repo, model_revision,
 model_name, model_dir, tokenizer_repo, tokenizer_revision,
 tokenizer_dir) = sys.argv[1:]
{input_download}
hf_hub_download(
    repo_id=model_repo, revision=model_revision, filename=model_name,
    local_dir=model_dir,
)
snapshot_download(
    repo_id=tokenizer_repo, revision=tokenizer_revision,
    local_dir=tokenizer_dir,
    allow_patterns=["*.json", "*.txt", "*.jinja", "*.model"],
)
PY
test "$(sha256sum {_q(model_file)} | awk '{{print $1}}')" = \
  {_q(config.label_model_file_sha256)}
test -f {_q(tokenizer_dir / "tokenizer.json")}
test -f {_q(tokenizer_dir / "tokenizer_config.json")}
if test -x {_q(layout.root / "llama-server-bin/llama-server")} && \
   test -f {_q(layout.root / "llama-server-bin/libllama-server-impl.so")}; then
  llama_ready=true
else
  llama_ready=false
fi
printf 'LABEL_ASSETS_OK llama_ready=%s\\n' "$llama_ready"
""".strip()
        result = self._ssh.run(script)
        if "LABEL_ASSETS_OK" not in result.stdout:
            raise RuntimeError("remote label assets failed validation")
        return LabelAssets(
            input_file,
            model_file,
            tokenizer_dir,
            "llama_ready=true" in result.stdout,
        )

    def stage_hf_token(
        self,
        layout: RemoteLayout,
        *,
        token_path: Path | None = None,
    ) -> None:
        """Stage the local Hugging Face login for remote Hub writes.

        The token is transferred as a mode-0600 file, never as a command
        argument.  The remote payloads read the file into ``HF_TOKEN`` only
        inside the allocation; the managed-root cleanup removes it with the
        rest of the run.
        """

        source, temporary = _local_hf_token_file(token_path)
        try:
            RemoteTransfer(ssh_target=self._ssh.target).push(
                source, str(layout.hf_token)
            )
        finally:
            if temporary:
                source.unlink(missing_ok=True)
        result = self._ssh.run(
            "set -euo pipefail; "
            f"token={_q(layout.hf_token)}; "
            '[ -f "$token" ] && [ ! -L "$token" ] && '
            'chmod 0600 "$token" && [ -s "$token" ]; '
            "printf 'HF_TOKEN_OK\\n'"
        )
        if "HF_TOKEN_OK" not in result.stdout:
            raise RuntimeError("remote Hugging Face credential staging failed")

    def prepare_v2_input(
        self,
        config: OperatorConfig,
        layout: RemoteLayout,
        source_parquet: PurePosixPath,
    ) -> PurePosixPath:
        """Enrich split output with pinned upstream polygon metadata."""

        if config.input_dataset_revision is None:
            raise ValueError("immutable input revision is required")
        output = layout.v2_input
        script = f"""
set -euo pipefail
umask 077
if test -s {_q(output)} && test ! -L {_q(output)}; then
  printf 'V2_INPUT_OK reused=true\\n'
  exit 0
fi
python={_q(layout.repo / ".venv/bin/python")}
token_file={_q(layout.hf_token)}
[ -f "$token_file" ] && [ ! -L "$token_file" ] && [ "$(stat -c %a -- "$token_file")" = 600 ]
export HF_TOKEN="$(cat -- "$token_file")"
[ -n "$HF_TOKEN" ]
mkdir -p -m 0700 {_q(layout.input_dir)} {_q(layout.hf_home)}
"$python" -m osm_polygon_sentence_relevance.labeling.v2_input \\
  --source {_q(source_parquet)} \\
  --output {_q(output)} \\
  --dataset-id {_q(config.input_dataset_id)} \\
  --revision {_q(config.input_dataset_revision)} \\
  --cache-dir {_q(layout.hf_home)}
test -f {_q(output)}
cache={_q(layout.hf_home)}
[ -d "$cache" ] && [ ! -L "$cache" ] || exit 70
find -P "$cache" -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +
printf 'V2_INPUT_OK\\n'
""".strip()
        result = self._ssh.run(script)
        if "V2_INPUT_OK" not in result.stdout:
            raise RuntimeError("remote V2 input preparation failed")
        return output


__all__ = ["LabelAssets", "Stager", "StagingResult"]


def _local_hf_token_file(token_path: Path | None) -> tuple[Path, bool]:
    """Return a private local token file and whether it is temporary."""

    candidate = token_path or _discover_hf_token_path()
    if candidate is not None:
        try:
            mode = candidate.stat().st_mode
        except OSError as exc:
            raise RuntimeError("local Hugging Face login file is unavailable") from exc
        if not stat.S_ISREG(mode) or candidate.is_symlink() or mode & 0o077:
            raise RuntimeError(
                "local Hugging Face login file must be a private regular file"
            )
        try:
            token = candidate.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError("local Hugging Face login file is unreadable") from exc
        if not token or any(character.isspace() for character in token):
            raise RuntimeError("local Hugging Face login file is invalid")
        return candidate, False

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token or any(character.isspace() for character in token):
        raise RuntimeError("no local Hugging Face login is available")
    descriptor, name = tempfile.mkstemp(prefix=".hf-token.")
    path = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with stream:
            stream.write(token)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return path, True


def _discover_hf_token_path() -> Path | None:
    """Find the standard local Hugging Face token file without reading it."""

    candidates: list[Path] = []
    configured_home = os.environ.get("HF_HOME")
    if configured_home:
        candidates.append(Path(configured_home) / "token")
    candidates.extend(
        (
            Path.home() / ".cache" / "huggingface" / "token",
            Path.home() / ".huggingface" / "token",
        )
    )
    return next(
        (
            candidate
            for candidate in candidates
            if candidate.exists() and not candidate.is_symlink()
        ),
        None,
    )
