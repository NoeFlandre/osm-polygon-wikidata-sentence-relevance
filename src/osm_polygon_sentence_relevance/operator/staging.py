"""Remote checkout and immutable artifact staging command generation."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

from osm_polygon_sentence_relevance.operator.config import OperatorConfig, Stage
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
        if config.stage in {Stage.SPLIT, Stage.ALL}:
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
git -C "$repo" cat-file -e {_q(config.source_commit)}^{{commit}}
git -C "$repo" checkout --detach {_q(config.source_commit)}
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
input_source = hf_hub_download(
    repo_id=dataset, repo_type="dataset", revision=input_revision,
    filename="sentences.parquet",
)
target = pathlib.Path(input_target)
if not target.exists():
    shutil.copyfile(input_source, target)
"""
            if download_input
            else "pathlib.Path(input_target).touch(exist_ok=True)\n"
        )
        script = f"""
set -euo pipefail
umask 077
mkdir -p -m 0700 {_q(input_dir)} {_q(model_dir)} {_q(tokenizer_dir)}
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
if test -x {_q(layout.root / "llama-server-bin/llama-server")}; then
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


__all__ = ["LabelAssets", "Stager", "StagingResult"]
