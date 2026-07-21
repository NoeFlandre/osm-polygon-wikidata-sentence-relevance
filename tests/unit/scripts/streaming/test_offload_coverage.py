"""Focused defensive branches for verified remote checkpoints."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest
from scripts.streaming.offload import (
    CheckpointOffloader,
    CheckpointOffloadError,
    OffloadHandle,
    _download_file,
    _entry_lfs_sha,
    _handle_from_files,
    _list_files,
    _validate_metadata,
    _validate_run_id,
    _validate_shard_key,
    discover_run,
)

from osm_polygon_sentence_relevance.application.checkpoint import (
    segmented_schema_sha256,
)


def _metadata(shard: str = "a-latest") -> dict[str, object]:
    return {
        "shard_key": shard,
        "segmented_table_sha256": "0" * 64,
        "segmented_table_bytes": 1,
        "segmented_schema_sha256": segmented_schema_sha256(),
        "source_commit": "1" * 40,
        "input_dataset_revision": "2" * 40,
        "pipeline_version": "0.1.0",
        "model_name": "sat-3l-sm",
        "batch_size": 128,
    }


def _offloader(tmp_path: Path, hub: object | None = None) -> CheckpointOffloader:
    return CheckpointOffloader(
        hub_api=hub or mock.MagicMock(),
        repo_id="owner/repo",
        staging_revision="checkpoints/run-1",
        run_id="run-1",
        local_cache_dir=tmp_path,
    )


def test_constructor_rejects_invalid_repo(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repo_id"):
        CheckpointOffloader(
            hub_api=mock.Mock(),
            repo_id="invalid",
            staging_revision="branch",
            run_id="run",
            local_cache_dir=tmp_path,
        )


def test_constructor_rejects_blank_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="staging_revision"):
        CheckpointOffloader(
            hub_api=mock.Mock(),
            repo_id="owner/repo",
            staging_revision=" ",
            run_id="run",
            local_cache_dir=tmp_path,
        )


def test_ensure_branch_tolerates_duplicate(tmp_path: Path) -> None:
    hub = mock.MagicMock()
    hub.create_branch.side_effect = RuntimeError("409: branch already exists")
    _offloader(tmp_path, hub)._ensure_branch()


def test_ensure_branch_wraps_network_failure(tmp_path: Path) -> None:
    hub = mock.MagicMock()
    hub.create_branch.side_effect = OSError("network unavailable")
    with pytest.raises(CheckpointOffloadError, match="ensure staging branch") as err:
        _offloader(tmp_path, hub)._ensure_branch()
    assert isinstance(err.value.__cause__, OSError)


def test_list_files_treats_missing_branch_as_empty() -> None:
    hub = mock.MagicMock()
    hub.list_repo_tree.side_effect = RuntimeError("404 revision not found")
    assert (
        _list_files(
            hub_api=hub,
            repo_id="owner/repo",
            revision="missing",
            folder_path="checkpoints/run",
        )
        == {}
    )


def test_list_files_wraps_unexpected_failure() -> None:
    hub = mock.MagicMock()
    hub.list_repo_tree.side_effect = OSError("transport")
    with pytest.raises(CheckpointOffloadError, match="inspect staging") as err:
        _list_files(
            hub_api=hub,
            repo_id="owner/repo",
            revision="branch",
            folder_path="checkpoints/run",
        )
    assert isinstance(err.value.__cause__, OSError)


def test_handle_is_frozen() -> None:
    handle = OffloadHandle(
        repo_id="owner/repo",
        run_id="run",
        shard_key="shard",
        staging_revision="branch",
        folder_path="checkpoints/run/shard",
        expected_table_sha256="0" * 64,
        computed_table_sha256="0" * 64,
        table_bytes=1,
        metadata={},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        handle.shard_key = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("value", [None, "", "Bad", "a/b", "a b"])
def test_shard_key_validation_rejects_invalid(value: object) -> None:
    with pytest.raises(ValueError, match="shard_key"):
        _validate_shard_key(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, "", "a/b", "a b"])
def test_run_id_validation_rejects_invalid(value: object) -> None:
    with pytest.raises(ValueError, match="run_id"):
        _validate_run_id(value)  # type: ignore[arg-type]


def test_metadata_rejects_non_object() -> None:
    with pytest.raises(CheckpointOffloadError, match="must be an object"):
        _validate_metadata([], shard_key="a-latest")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shard_key", "wrong", "shard_key mismatch"),
        ("segmented_table_sha256", "bad", "invalid segmented_table_sha256"),
        ("segmented_table_bytes", True, "invalid segmented_table_bytes"),
        ("segmented_table_bytes", 0, "invalid segmented_table_bytes"),
        ("segmented_schema_sha256", "bad", "schema fingerprint mismatch"),
    ],
)
def test_metadata_rejects_invalid_fields(
    field: str, value: object, message: str
) -> None:
    payload = _metadata()
    payload[field] = value
    with pytest.raises(CheckpointOffloadError, match=message):
        _validate_metadata(payload, shard_key="a-latest")


def test_metadata_rejects_missing_field() -> None:
    payload = _metadata()
    del payload["model_name"]
    with pytest.raises(CheckpointOffloadError, match="missing"):
        _validate_metadata(payload, shard_key="a-latest")


@pytest.mark.parametrize(
    "field",
    [
        "source_commit",
        "input_dataset_revision",
        "pipeline_version",
        "model_name",
        "batch_size",
    ],
)
def test_metadata_rejects_identity_mismatch(field: str) -> None:
    payload = _metadata()
    expected = {
        key: payload[key]
        for key in (
            "source_commit",
            "input_dataset_revision",
            "pipeline_version",
            "model_name",
            "batch_size",
        )
    }
    expected[field] = "different"
    with pytest.raises(CheckpointOffloadError, match=field):
        _validate_metadata(payload, shard_key="a-latest", expected_identity=expected)


def test_entry_lfs_sha_supports_mapping_and_object() -> None:
    digest = "a" * 64
    assert _entry_lfs_sha(mock.Mock(lfs={"sha256": digest})) == digest
    assert _entry_lfs_sha(mock.Mock(lfs=mock.Mock(sha256=digest))) == digest
    assert _entry_lfs_sha(mock.Mock(lfs={"sha256": "bad"})) is None


def test_download_file_wraps_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.streaming.offload._lazy_hf_hub_download",
        lambda: mock.Mock(side_effect=OSError("network")),
    )
    with pytest.raises(CheckpointOffloadError, match="readback failed") as error:
        _download_file(
            repo_id="o/r",
            revision="r",
            filename="x",
            local_dir=tmp_path,
        )
    assert isinstance(error.value.__cause__, OSError)


def test_download_file_requires_created_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.streaming.offload._lazy_hf_hub_download",
        lambda: mock.Mock(return_value=str(tmp_path / "missing")),
    )
    with pytest.raises(CheckpointOffloadError, match="did not create"):
        _download_file(
            repo_id="o/r",
            revision="r",
            filename="x",
            local_dir=tmp_path,
        )


class _Entry:
    def __init__(self, *, size: int | None = None, sha: str | None = None) -> None:
        self.size = size
        self.lfs = {"sha256": sha} if sha is not None else None


def test_handle_rejects_unexpected_entries(tmp_path: Path) -> None:
    with pytest.raises(CheckpointOffloadError, match="entries mismatch"):
        _handle_from_files(
            hub_api=mock.Mock(),
            repo_id="o/r",
            staging_revision="branch",
            run_id="run",
            shard_key="a-latest",
            files={"metadata.json": _Entry()},
            local_cache_dir=tmp_path,
        )


def test_handle_rejects_bad_json(tmp_path: Path, monkeypatch) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not-json", encoding="utf-8")
    monkeypatch.setattr("scripts.streaming.offload._download_file", lambda **_: bad)
    with pytest.raises(CheckpointOffloadError, match="invalid metadata JSON"):
        _handle_from_files(
            hub_api=mock.Mock(),
            repo_id="o/r",
            staging_revision="branch",
            run_id="run",
            shard_key="a-latest",
            files={"metadata.json": _Entry(), "segmented.parquet": _Entry()},
            local_cache_dir=tmp_path,
        )


def test_handle_rejects_remote_size_mismatch(tmp_path: Path, monkeypatch) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(_metadata()), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.streaming.offload._download_file", lambda **_: metadata
    )
    with pytest.raises(CheckpointOffloadError, match="remote byte size"):
        _handle_from_files(
            hub_api=mock.Mock(),
            repo_id="o/r",
            staging_revision="branch",
            run_id="run",
            shard_key="a-latest",
            files={
                "metadata.json": _Entry(),
                "segmented.parquet": _Entry(size=2, sha="0" * 64),
            },
            local_cache_dir=tmp_path,
        )


def test_handle_rejects_lfs_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(_metadata()), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.streaming.offload._download_file", lambda **_: metadata
    )
    with pytest.raises(CheckpointOffloadError, match="Hub LFS SHA"):
        _handle_from_files(
            hub_api=mock.Mock(),
            repo_id="o/r",
            staging_revision="branch",
            run_id="run",
            shard_key="a-latest",
            files={
                "metadata.json": _Entry(),
                "segmented.parquet": _Entry(size=1, sha="f" * 64),
            },
            local_cache_dir=tmp_path,
        )


def test_handle_downloads_when_lfs_hash_absent_then_evicts(
    tmp_path: Path, monkeypatch
) -> None:
    table = tmp_path / "segmented.parquet"
    table.write_bytes(b"x")
    metadata_payload = _metadata()
    metadata_payload["segmented_table_sha256"] = hashlib.sha256(b"x").hexdigest()
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(metadata_payload), encoding="utf-8")

    def download(*, filename: str, **_: object) -> Path:
        return metadata if filename.endswith("metadata.json") else table

    monkeypatch.setattr("scripts.streaming.offload._download_file", download)
    result = _handle_from_files(
        hub_api=mock.Mock(),
        repo_id="o/r",
        staging_revision="branch",
        run_id="run",
        shard_key="a-latest",
        files={"metadata.json": _Entry(), "segmented.parquet": _Entry(size=1)},
        local_cache_dir=tmp_path,
    )
    assert result.local_table_path is None
    assert not table.exists()


def test_discover_run_empty_and_unexpected_entry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.streaming.offload._list_files", lambda **_: {})
    assert (
        discover_run(
            hub_api=mock.Mock(), repo_id="o/r", run_id="run", local_cache_dir=tmp_path
        )
        == []
    )
    # A single-segment relative path represents a folder entry
    # (``expand=True``), which discover_run now skips silently.
    monkeypatch.setattr(
        "scripts.streaming.offload._list_files", lambda **_: {"orphan": _Entry()}
    )
    assert (
        discover_run(
            hub_api=mock.Mock(), repo_id="o/r", run_id="run", local_cache_dir=tmp_path
        )
        == []
    )
    # A two-segment relative path with an empty second segment is
    # structurally malformed and must raise.
    monkeypatch.setattr(
        "scripts.streaming.offload._list_files",
        lambda **_: {"shard/": _Entry()},
    )
    with pytest.raises(CheckpointOffloadError):
        discover_run(
            hub_api=mock.Mock(), repo_id="o/r", run_id="run", local_cache_dir=tmp_path
        )
