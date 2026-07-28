"""Remote staging contracts without network access."""

from __future__ import annotations

from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator.config import OperatorConfig
from osm_polygon_sentence_relevance.operator.staging import Stager
from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout


def _config() -> OperatorConfig:
    return OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )


class RecordingSsh:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.commands: list[str] = []

    def run(self, command: str) -> SimpleNamespace:
        self.commands.append(command)
        return SimpleNamespace(stdout=self.outputs.pop(0))


def test_prepare_builds_clean_pinned_checkout() -> None:
    ssh = RecordingSsh(["STAGING_OK reused=false\n"])
    layout = RemoteLayout(PurePosixPath("/home/user/operator/run"))
    result = Stager(ssh).prepare(_config(), layout)  # type: ignore[arg-type]
    assert not result.reused
    command = ssh.commands[0]
    assert "git clone --no-tags" in command
    assert "checkout --detach" in command
    assert 'UV_BIN="$(command -v uv || true)"' in command
    assert 'UV_BIN="$HOME/.local/bin/uv"' in command
    assert '"$UV_BIN" sync --locked --extra hub --extra segmentation' in command
    assert '"status":"active"' in command


def test_prepare_reports_reuse_and_rejects_missing_marker() -> None:
    layout = RemoteLayout(PurePosixPath("/home/user/operator/run"))
    reused = RecordingSsh(["STAGING_OK reused=true\n"])
    assert Stager(reused).prepare(_config(), layout).reused  # type: ignore[arg-type]
    missing = RecordingSsh(["unexpected\n"])
    with pytest.raises(RuntimeError, match="success marker"):
        Stager(missing).prepare(_config(), layout)  # type: ignore[arg-type]


def test_prepare_label_assets_downloads_and_validates_pins() -> None:
    ssh = RecordingSsh(["LABEL_ASSETS_OK llama_ready=true\n"])
    layout = RemoteLayout(PurePosixPath("/home/user/operator/run"))
    assets = Stager(ssh).prepare_label_assets(_config(), layout)  # type: ignore[arg-type]
    assert assets.llama_server_ready
    assert assets.input_parquet == layout.root / "input/sentences.parquet"
    command = ssh.commands[0]
    assert "hf_hub_download" in command
    assert "snapshot_download" in command
    assert "sha256sum" in command


def test_prepare_label_assets_can_reuse_split_output_without_download() -> None:
    ssh = RecordingSsh(["LABEL_ASSETS_OK llama_ready=false\n"])
    layout = RemoteLayout(PurePosixPath("/home/user/operator/run"))
    assets = Stager(ssh).prepare_label_assets(
        _config(),
        layout,
        download_input=False,  # type: ignore[arg-type]
    )
    assert not assets.llama_server_ready
    assert "touch(exist_ok=True)" in ssh.commands[0]


def test_prepare_label_assets_requires_revision_and_marker() -> None:
    layout = RemoteLayout(PurePosixPath("/home/user/operator/run"))
    config = _config()
    object.__setattr__(config, "input_dataset_revision", None)
    with pytest.raises(ValueError, match="revision"):
        Stager(RecordingSsh([])).prepare_label_assets(  # type: ignore[arg-type]
            config, layout
        )
    with pytest.raises(RuntimeError, match="failed validation"):
        Stager(RecordingSsh(["bad\n"])).prepare_label_assets(  # type: ignore[arg-type]
            _config(), layout
        )
