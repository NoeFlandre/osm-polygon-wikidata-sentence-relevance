"""Remote checkout and immutable artifact staging command generation."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

from osm_polygon_sentence_relevance.operator.config import OperatorConfig
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
test "$(git -C "$repo" rev-parse origin/main)" = {_q(config.source_commit)}
git -C "$repo" checkout --detach {_q(config.source_commit)}
test -z "$(git -C "$repo" status --porcelain)"
uv sync --locked --extra hub --extra segmentation -C "$repo"
printf 'STAGING_OK reused=%s\\n' "$reused"
""".strip()
        result = self._ssh.run(script)
        if "STAGING_OK reused=" not in result.stdout:
            raise RuntimeError("remote staging did not return its success marker")
        return StagingResult(layout, "reused=true" in result.stdout)

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
)
PY
test "$(sha256sum {_q(model_file)} | awk '{{print $1}}')" = \
  {_q(config.label_model_file_sha256)}
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
