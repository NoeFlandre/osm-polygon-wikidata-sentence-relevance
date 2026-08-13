"""Remote staging contracts without network access."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator import staging
from osm_polygon_sentence_relevance.operator.config import (
    V2_LOGIT_PROMPT_VERSION,
    OperatorConfig,
)
from osm_polygon_sentence_relevance.operator.staging import Stager
from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout


def _config(*, stage: str = "label") -> OperatorConfig:
    return OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage=stage,
        source_commit="a" * 40,
        input_revision="b" * 40,
    )


class RecordingSsh:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.commands: list[str] = []
        self.target = "grenoble"

    def run(self, command: str) -> SimpleNamespace:
        self.commands.append(command)
        return SimpleNamespace(stdout=self.outputs.pop(0))


class RecordingTransfer:
    calls: list[tuple[Path, str]] = []

    def __init__(self, *, ssh_target: str) -> None:
        self.ssh_target = ssh_target

    def push(self, local_path: Path, remote_path: str) -> None:
        self.calls.append((local_path, remote_path))


def test_prepare_builds_clean_pinned_checkout() -> None:
    ssh = RecordingSsh(["STAGING_OK reused=false\n"])
    layout = RemoteLayout(PurePosixPath("/home/user/operator/run"))
    result = Stager(ssh).prepare(_config(), layout)  # type: ignore[arg-type]
    assert not result.reused
    command = ssh.commands[0]
    assert "git clone --no-tags" in command
    assert "checkout --detach" in command
    assert "cat-file -e" in command
    assert "rev-parse origin/main" not in command
    assert 'UV_BIN="$(command -v uv || true)"' in command
    assert 'UV_BIN="$HOME/.local/bin/uv"' in command
    assert '"$UV_BIN" sync --locked --no-dev --extra hub --extra operator' in command
    assert "--extra segmentation" not in command
    assert '--project "$repo"' in command
    assert '"$UV_BIN" sync --locked --no-dev --extra hub -C' not in command
    assert 'UV_CACHE_DIR="$root/uv-cache"' in command
    assert "export UV_CACHE_DIR" in command
    assert "trap cleanup_uv_cache EXIT" in command
    assert 'rm -rf -- "$UV_CACHE_DIR"' in command
    assert 'egg_info="$repo/src/osm_polygon_sentence_relevance.egg-info"' in command
    assert '[ ! -L "$egg_info" ]' in command
    assert 'rm -rf -- "$egg_info"' in command
    assert command.index('"$UV_BIN" sync') < command.index('rm -rf -- "$egg_info"')
    assert '"status":"active"' in command


def test_stage_hf_token_pushes_login_file_without_putting_secret_in_ssh_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("hf_test_secret\n", encoding="utf-8")
    token_path.chmod(0o600)
    RecordingTransfer.calls = []
    monkeypatch.setattr(
        "osm_polygon_sentence_relevance.operator.staging.RemoteTransfer",
        RecordingTransfer,
    )
    ssh = RecordingSsh(["HF_TOKEN_OK\n"])
    layout = RemoteLayout(PurePosixPath("/home/user/operator/run"))

    Stager(ssh).stage_hf_token(layout, token_path=token_path)  # type: ignore[arg-type]

    assert RecordingTransfer.calls == [(token_path, str(layout.hf_token))]
    assert len(ssh.commands) == 1
    assert "hf_test_secret" not in ssh.commands[0]
    assert "chmod 0600" in ssh.commands[0]


def test_stage_hf_token_cleans_environment_fallback_after_remote_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pushed: list[Path] = []

    class _Transfer:
        def __init__(self, *, ssh_target: str) -> None:
            del ssh_target

        def push(self, local_path: Path, _remote_path: str) -> None:
            pushed.append(local_path)

    monkeypatch.setattr(staging, "RemoteTransfer", _Transfer)
    monkeypatch.setattr(staging, "_discover_hf_token_path", lambda: None)
    monkeypatch.setenv("HF_TOKEN", "hf_environment_secret")
    ssh = RecordingSsh(["invalid\n"])

    with pytest.raises(RuntimeError, match="staging failed"):
        Stager(ssh).stage_hf_token(  # type: ignore[arg-type]
            RemoteLayout(PurePosixPath("/home/user/operator/run"))
        )

    assert len(pushed) == 1
    assert not pushed[0].exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "invalid"),
        ("has whitespace\n", "invalid"),
    ],
)
def test_local_hf_token_file_rejects_invalid_content(
    tmp_path: Path, content: str, message: str
) -> None:
    token_path = tmp_path / "token"
    token_path.write_text(content, encoding="utf-8")
    token_path.chmod(0o600)
    with pytest.raises(RuntimeError, match=message):
        staging._local_hf_token_file(token_path)


def test_local_hf_token_file_rejects_non_private_file(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("hf_token", encoding="utf-8")
    token_path.chmod(0o644)
    with pytest.raises(RuntimeError, match="private regular"):
        staging._local_hf_token_file(token_path)


def test_local_hf_token_file_wraps_read_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("hf_token", encoding="utf-8")
    token_path.chmod(0o600)
    original = Path.read_text

    def fail_read(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path == token_path:
            raise OSError("read failed")
        return original(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", fail_read)
    with pytest.raises(RuntimeError, match="unreadable"):
        staging._local_hf_token_file(token_path)


def test_local_hf_token_file_uses_hf_home_and_environment_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    token_path = hf_home / "token"
    token_path.write_text("hf_from_file\n", encoding="utf-8")
    token_path.chmod(0o600)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    path, temporary = staging._local_hf_token_file(None)
    assert path == token_path
    assert not temporary

    token_path.unlink()
    monkeypatch.setattr(staging, "_discover_hf_token_path", lambda: None)
    monkeypatch.setenv("HF_TOKEN", "hf_from_environment")
    path, temporary = staging._local_hf_token_file(None)
    try:
        assert temporary
        assert path.read_text(encoding="utf-8") == "hf_from_environment"
        assert path.stat().st_mode & 0o777 == 0o600
    finally:
        path.unlink(missing_ok=True)


def test_local_hf_token_file_requires_a_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(staging, "_discover_hf_token_path", lambda: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="no local Hugging Face login"):
        staging._local_hf_token_file(None)


def test_prepare_includes_segmentation_dependencies_only_when_needed() -> None:
    ssh = RecordingSsh(["STAGING_OK reused=false\n"])
    layout = RemoteLayout(PurePosixPath("/home/user/operator/run"))
    Stager(ssh).prepare(_config(stage="split"), layout)  # type: ignore[arg-type]
    command = ssh.commands[0]
    assert (
        '"$UV_BIN" sync --locked --no-dev --extra hub --extra segmentation' in command
    )


def test_prepare_v2_all_skips_segmentation_dependencies() -> None:
    config = OperatorConfig.build(
        scope="all",
        region=None,
        stage="all",
        source_commit="a" * 40,
        input_revision="b" * 40,
        prompt_version=V2_LOGIT_PROMPT_VERSION,
    )
    ssh = RecordingSsh(["STAGING_OK reused=false\n"])
    Stager(ssh).prepare(config, RemoteLayout(PurePosixPath("/run")))  # type: ignore[arg-type]
    command = ssh.commands[0]
    assert '"$UV_BIN" sync --locked --no-dev --extra hub --extra operator' in command
    assert "--extra segmentation" not in command


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
    assert "allow_patterns=" in command
    assert '"*.safetensors"' not in command
    assert '"*.bin"' not in command
    assert "test -f /home/user/operator/run/tokenizer/" in command
    assert "/tokenizer.json" in command
    assert "/tokenizer_config.json" in command
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


def test_prepare_label_assets_reuses_existing_input_before_hub_download() -> None:
    ssh = RecordingSsh(["LABEL_ASSETS_OK llama_ready=false\n"])
    layout = RemoteLayout(PurePosixPath("/home/user/operator/run"))

    Stager(ssh).prepare_label_assets(_config(), layout)  # type: ignore[arg-type]

    command = ssh.commands[0]
    assert "input_target_path.is_file()" in command
    assert "input_target_path.stat().st_size > 0" in command
    assert command.index("input_target_path.is_file()") < command.index(
        "hf_hub_download("
    )


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


def test_prepare_v2_input_enriches_split_output_from_pinned_polygon_files() -> None:
    ssh = RecordingSsh(["V2_INPUT_OK\n"])
    layout = RemoteLayout(PurePosixPath("/home/user/operator/run"))
    source = layout.logs / "42/output/sentences.parquet"
    output = Stager(ssh).prepare_v2_input(_config(stage="all"), layout, source)  # type: ignore[arg-type]
    assert output == layout.root / "input/v2-sentences.parquet"
    command = ssh.commands[0]
    assert '"$python" -m osm_polygon_sentence_relevance.labeling.v2_input' in command
    assert "--source" in command
    assert str(source) in command
    assert "--dataset-id" in command
    assert "--revision" in command
    assert "--cache-dir" in command
    assert "cache=/home/user/operator/run/hf_home" in command
    assert "if test -s /home/user/operator/run/input/v2-sentences.parquet" in command
    assert "rm -rf -- {}" in command
    assert "V2_INPUT_OK" in command


def test_prepare_v2_input_requires_immutable_revision() -> None:
    config = _config(stage="all")
    object.__setattr__(config, "input_dataset_revision", None)
    with pytest.raises(ValueError, match="revision"):
        Stager(RecordingSsh([])).prepare_v2_input(
            config,
            RemoteLayout(PurePosixPath("/home/user/operator/run")),
            PurePosixPath("/home/user/sentences.parquet"),
        )  # type: ignore[arg-type]


def test_prepare_v2_input_requires_success_marker() -> None:
    layout = RemoteLayout(PurePosixPath("/home/user/operator/run"))
    with pytest.raises(RuntimeError, match="preparation failed"):
        Stager(RecordingSsh(["unexpected\n"])).prepare_v2_input(
            _config(stage="all"),
            layout,
            layout.logs / "42/output/sentences.parquet",
        )  # type: ignore[arg-type]


def test_clean_generated_python_caches_is_scoped_to_runtime_sources() -> None:
    ssh = RecordingSsh(["PYTHON_CACHES_CLEAN\n"])
    layout = RemoteLayout(PurePosixPath("/home/user/operator/run"))
    Stager(ssh).clean_generated_python_caches(layout)  # type: ignore[arg-type]
    command = ssh.commands[0]
    assert "repo=/home/user/operator/run/repo" in command
    assert 'for tree in "$repo/src" "$repo/scripts"' in command
    assert 'find -P "$tree"' in command
    assert "rm -rf --" in command
    assert "PYTHON_CACHES_CLEAN" in command


def test_clean_generated_python_caches_requires_success_marker() -> None:
    layout = RemoteLayout(PurePosixPath("/home/user/operator/run"))
    with pytest.raises(RuntimeError, match="cache cleanup"):
        Stager(RecordingSsh(["unexpected\n"])).clean_generated_python_caches(  # type: ignore[arg-type]
            layout
        )
