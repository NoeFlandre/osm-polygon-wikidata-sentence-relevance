from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.streaming.finalized_offload import (
    FinalizedArtifactOffloader,
    FinalizedOffloadError,
    FinalizedShardHandle,
    _handle_from_files,
    _validate_metadata,
    schema_sha256,
)


def _artifact(tmp_path: Path) -> tuple[Path, dict[str, object], pa.Schema]:
    table = pa.table({"sentence_id": ["s1"], "polygon_id": ["p1"]})
    path = tmp_path / "finalized.parquet"
    pq.write_table(table, path)
    schema = table.schema
    metadata = {
        "schema_version": 1,
        "shard_key": "a-latest",
        "table_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "table_bytes": path.stat().st_size,
        "row_count": 1,
        "schema_sha256": schema_sha256(schema),
        "source_commit": "a" * 40,
    }
    active = tmp_path / "active"
    active.mkdir()
    (active / "finalized.parquet").write_bytes(path.read_bytes())
    (active / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return active, metadata, schema


def test_finalized_offloader_rejects_metadata_identity_mismatch(
    tmp_path: Path,
) -> None:
    active, metadata, schema = _artifact(tmp_path)
    metadata["source_commit"] = "b" * 40
    (active / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    offloader = FinalizedArtifactOffloader(
        hub_api=mock.Mock(),
        repo_id="owner/output",
        staging_revision="checkpoints/run",
        run_id="run",
        local_cache_dir=tmp_path / "cache",
        schema=schema,
        expected_identity={"source_commit": "a" * 40},
    )
    with pytest.raises(FinalizedOffloadError, match="source_commit mismatch"):
        offloader.upload_and_verify(
            shard_key="a-latest", active_dir=active, metadata=metadata
        )


def test_finalized_handle_has_durable_identity_contract(tmp_path: Path) -> None:
    _, metadata, schema = _artifact(tmp_path)
    handle = FinalizedShardHandle(
        repo_id="owner/output",
        run_id="run",
        shard_key="a-latest",
        staging_revision="checkpoints/run",
        folder_path="finalized/run/a-latest",
        table_sha256=str(metadata["table_sha256"]),
        table_bytes=int(metadata["table_bytes"]),
        row_count=1,
        schema_sha256=schema_sha256(schema),
        metadata=metadata,
    )
    assert handle.row_count == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "identity mismatch"),
        ("shard_key", "other-latest", "identity mismatch"),
        ("table_sha256", "bad", "SHA-256 is invalid"),
        ("table_bytes", 0, "byte count is invalid"),
        ("table_bytes", True, "byte count is invalid"),
        ("row_count", -1, "row count is invalid"),
        ("row_count", True, "row count is invalid"),
        ("schema_sha256", "bad", "fingerprint mismatch"),
    ],
)
def test_validate_metadata_rejects_corrupt_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    _, metadata, schema = _artifact(tmp_path)
    metadata[field] = value
    with pytest.raises(FinalizedOffloadError, match=message):
        _validate_metadata(
            metadata,
            shard_key="a-latest",
            schema=schema,
            expected_identity=None,
        )


def test_validate_metadata_rejects_non_object_and_missing_fields(
    tmp_path: Path,
) -> None:
    _, metadata, schema = _artifact(tmp_path)
    with pytest.raises(FinalizedOffloadError, match="must be an object"):
        _validate_metadata(
            [], shard_key="a-latest", schema=schema, expected_identity=None
        )
    missing = dict(metadata)
    del missing["row_count"]
    with pytest.raises(FinalizedOffloadError, match="incomplete"):
        _validate_metadata(
            missing, shard_key="a-latest", schema=schema, expected_identity=None
        )


def test_validate_metadata_checks_expected_identity(tmp_path: Path) -> None:
    _, metadata, schema = _artifact(tmp_path)
    with pytest.raises(FinalizedOffloadError, match="source_commit mismatch"):
        _validate_metadata(
            metadata,
            shard_key="a-latest",
            schema=schema,
            expected_identity={"source_commit": "b" * 40},
        )


class _Entry:
    def __init__(self, path: Path, *, sha: str | None = None) -> None:
        self.size = path.stat().st_size
        self.lfs = {"sha256": sha} if sha is not None else None


def test_handle_from_files_materializes_and_verifies_remote_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active, metadata, schema = _artifact(tmp_path)
    table = active / "finalized.parquet"
    cache = tmp_path / "cache"
    downloads = {
        "metadata.json": active / "metadata.json",
        "finalized.parquet": table,
    }

    def download(**kwargs: object) -> Path:
        filename = Path(str(kwargs["filename"])).name
        target = Path(str(kwargs["local_dir"])) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(downloads[filename].read_bytes())
        return target

    monkeypatch.setattr("scripts.streaming.finalized_offload._download_file", download)
    files = {
        "metadata.json": _Entry(active / "metadata.json"),
        "finalized.parquet": _Entry(table, sha=str(metadata["table_sha256"])),
    }
    handle = _handle_from_files(
        hub_api=mock.Mock(),
        repo_id="owner/output",
        staging_revision="checkpoints/run",
        run_id="run",
        shard_key="a-latest",
        files=files,
        local_cache_dir=cache,
        schema=schema,
        expected_identity={"source_commit": "a" * 40},
        materialize=True,
    )
    assert handle.local_table_path is not None
    assert handle.local_table_path.exists()
    assert handle.row_count == 1


def test_handle_from_files_rejects_incomplete_or_mismatched_remote_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active, metadata, schema = _artifact(tmp_path)
    table = active / "finalized.parquet"
    monkeypatch.setattr(
        "scripts.streaming.finalized_offload._download_file",
        lambda **kwargs: active / Path(str(kwargs["filename"])).name,
    )
    files = {
        "metadata.json": _Entry(active / "metadata.json"),
        "finalized.parquet": _Entry(table, sha="b" * 64),
    }
    with pytest.raises(FinalizedOffloadError, match="remote SHA-256"):
        _handle_from_files(
            hub_api=mock.Mock(),
            repo_id="owner/output",
            staging_revision="checkpoints/run",
            run_id="run",
            shard_key="a-latest",
            files=files,
            local_cache_dir=tmp_path / "cache",
            schema=schema,
            expected_identity=None,
            materialize=False,
        )


@pytest.mark.parametrize(
    "failure", ["metadata", "size", "bytes", "hash", "schema", "rows"]
)
def test_handle_from_files_fails_closed_on_readback_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    active, metadata, schema = _artifact(tmp_path)
    table = active / "finalized.parquet"
    if failure == "metadata":
        (active / "metadata.json").write_text("not-json", encoding="utf-8")
    elif failure == "size":
        metadata["table_bytes"] = int(metadata["table_bytes"]) + 1
        (active / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    elif failure == "bytes" or failure == "hash":
        metadata["table_sha256"] = "a" * 64
        (active / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    elif failure == "schema":
        other = pa.table({"different": [1]})
        pq.write_table(other, table)
        metadata["table_sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
        metadata["table_bytes"] = table.stat().st_size
        metadata["schema_sha256"] = schema_sha256(schema)
        (active / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    else:
        metadata["row_count"] = 2
        (active / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def download(**kwargs: object) -> Path:
        filename = Path(str(kwargs["filename"])).name
        target = tmp_path / "download" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((active / filename).read_bytes())
        if failure == "bytes" and filename == "finalized.parquet":
            target.write_bytes(b"x" * int(metadata["table_bytes"]))
        return target

    monkeypatch.setattr("scripts.streaming.finalized_offload._download_file", download)
    files = {
        "metadata.json": _Entry(active / "metadata.json"),
        "finalized.parquet": _Entry(table),
    }
    message = {
        "metadata": "metadata JSON",
        "size": "remote byte size",
        "bytes": "readback SHA",
        "hash": "readback SHA",
        "schema": "Parquet schema",
        "rows": "row count",
    }[failure]
    if failure == "size":
        files["finalized.parquet"] = _Entry(table)
        files["finalized.parquet"].size = int(metadata["table_bytes"]) - 1
    with pytest.raises(FinalizedOffloadError, match=message):
        _handle_from_files(
            hub_api=mock.Mock(),
            repo_id="owner/output",
            staging_revision="run",
            run_id="run",
            shard_key="a-latest",
            files=files,
            local_cache_dir=tmp_path / "cache",
            schema=schema,
            expected_identity=None,
            materialize=True,
        )
    with pytest.raises(FinalizedOffloadError, match="entries are incomplete"):
        _handle_from_files(
            hub_api=mock.Mock(),
            repo_id="owner/output",
            staging_revision="checkpoints/run",
            run_id="run",
            shard_key="a-latest",
            files={"metadata.json": files["metadata.json"]},
            local_cache_dir=tmp_path / "cache",
            schema=schema,
            expected_identity=None,
            materialize=False,
        )


def test_offloader_inspect_returns_none_for_missing_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, schema = _artifact(tmp_path)
    offloader = FinalizedArtifactOffloader(
        hub_api=mock.Mock(),
        repo_id="owner/output",
        staging_revision="checkpoints/run",
        run_id="run",
        local_cache_dir=tmp_path / "cache",
        schema=schema,
    )
    monkeypatch.setattr(
        "scripts.streaming.finalized_offload._list_files", lambda **_: {}
    )
    assert offloader.inspect("a-latest", materialize=False) is None


def test_offloader_inspect_delegates_verified_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, schema = _artifact(tmp_path)
    expected = mock.Mock(spec=FinalizedShardHandle)
    monkeypatch.setattr(
        "scripts.streaming.finalized_offload._list_files",
        lambda **_: {"finalized.parquet": object(), "metadata.json": object()},
    )
    called: dict[str, object] = {}

    def handle(**kwargs: object) -> FinalizedShardHandle:
        called.update(kwargs)
        return expected

    monkeypatch.setattr(
        "scripts.streaming.finalized_offload._handle_from_files", handle
    )
    offloader = FinalizedArtifactOffloader(
        hub_api=mock.Mock(),
        repo_id="owner/output",
        staging_revision="checkpoints/run",
        run_id="run",
        local_cache_dir=tmp_path / "cache",
        schema=schema,
    )
    assert offloader.inspect("a-latest", materialize=True) is expected
    assert called["shard_key"] == "a-latest"
    assert called["materialize"] is True


def test_offloader_rejects_invalid_repo_id(tmp_path: Path) -> None:
    _, _, schema = _artifact(tmp_path)
    with pytest.raises(ValueError, match="owner/name"):
        FinalizedArtifactOffloader(
            hub_api=mock.Mock(),
            repo_id="invalid",
            staging_revision="run",
            run_id="run",
            local_cache_dir=tmp_path / "cache",
            schema=schema,
        )


def test_offloader_branch_creation_redacts_unexpected_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, schema = _artifact(tmp_path)
    hub_api = mock.Mock()
    hub_api.create_branch.side_effect = RuntimeError("permission denied")
    offloader = FinalizedArtifactOffloader(
        hub_api=hub_api,
        repo_id="owner/output",
        staging_revision="run",
        run_id="run",
        local_cache_dir=tmp_path / "cache",
        schema=schema,
    )
    with pytest.raises(FinalizedOffloadError, match="ensure staging branch"):
        offloader._ensure_branch()
    hub_api.create_branch.side_effect = RuntimeError("409 already exists")
    offloader._ensure_branch()


def test_offloader_upload_is_idempotent_when_remote_artifact_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active, metadata, schema = _artifact(tmp_path)
    offloader = FinalizedArtifactOffloader(
        hub_api=mock.Mock(),
        repo_id="owner/output",
        staging_revision="run",
        run_id="run",
        local_cache_dir=tmp_path / "cache",
        schema=schema,
        expected_identity={"source_commit": "a" * 40},
    )
    existing = FinalizedShardHandle(
        repo_id="owner/output",
        run_id="run",
        shard_key="a-latest",
        staging_revision="run",
        folder_path="finalized/run/a-latest",
        table_sha256=str(metadata["table_sha256"]),
        table_bytes=int(metadata["table_bytes"]),
        row_count=1,
        schema_sha256=schema_sha256(schema),
        metadata=metadata,
    )
    monkeypatch.setattr(offloader, "_ensure_branch", lambda: None)
    monkeypatch.setattr(offloader, "inspect", lambda *_args, **_kwargs: existing)
    offloader.upload_and_verify(
        shard_key="a-latest", active_dir=active, metadata=metadata
    )
    offloader.hub_api.upload_folder.assert_not_called()


def test_offloader_upload_verifies_local_metadata_and_uploads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active, metadata, schema = _artifact(tmp_path)
    offloader = FinalizedArtifactOffloader(
        hub_api=mock.Mock(),
        repo_id="owner/output",
        staging_revision="run",
        run_id="run",
        local_cache_dir=tmp_path / "cache",
        schema=schema,
        expected_identity={"source_commit": "a" * 40},
    )
    readback = tmp_path / "readback.parquet"
    readback.write_bytes((active / "finalized.parquet").read_bytes())
    verified = FinalizedShardHandle(
        repo_id="owner/output",
        run_id="run",
        shard_key="a-latest",
        staging_revision="run",
        folder_path="finalized/run/a-latest",
        table_sha256=str(metadata["table_sha256"]),
        table_bytes=int(metadata["table_bytes"]),
        row_count=1,
        schema_sha256=schema_sha256(schema),
        metadata=metadata,
        local_table_path=readback,
    )
    monkeypatch.setattr(offloader, "_ensure_branch", lambda: None)
    monkeypatch.setattr(offloader, "inspect", mock.Mock(side_effect=[None, verified]))
    offloader.upload_and_verify(
        shard_key="a-latest", active_dir=active, metadata=metadata
    )
    offloader.hub_api.upload_folder.assert_called_once()
    assert not readback.exists()


def test_offloader_upload_rejects_metadata_file_mismatch(tmp_path: Path) -> None:
    active, metadata, schema = _artifact(tmp_path)
    changed = dict(metadata)
    changed["extra"] = "not-on-disk"
    offloader = FinalizedArtifactOffloader(
        hub_api=mock.Mock(),
        repo_id="owner/output",
        staging_revision="run",
        run_id="run",
        local_cache_dir=tmp_path / "cache",
        schema=schema,
        expected_identity=None,
    )
    with pytest.raises(FinalizedOffloadError, match="does not match payload"):
        offloader.upload_and_verify(
            shard_key="a-latest", active_dir=active, metadata=changed
        )


@pytest.mark.parametrize("failure", ["missing", "hash", "size", "schema", "rows"])
def test_offloader_upload_validates_local_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    active, metadata, schema = _artifact(tmp_path)
    if failure == "missing":
        (active / "metadata.json").unlink()
    elif failure == "hash":
        metadata["table_sha256"] = "a" * 64
    elif failure == "size":
        metadata["table_bytes"] = int(metadata["table_bytes"]) + 1
    elif failure == "schema":
        metadata["schema_sha256"] = schema_sha256(
            pa.schema([pa.field("other", pa.int64())])
        )
    elif failure == "rows":
        metadata["row_count"] = 2
    if failure != "missing":
        (active / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    offloader = FinalizedArtifactOffloader(
        hub_api=mock.Mock(),
        repo_id="owner/output",
        staging_revision="run",
        run_id="run",
        local_cache_dir=tmp_path / "cache",
        schema=schema,
    )
    with pytest.raises(FinalizedOffloadError):
        offloader.upload_and_verify(
            shard_key="a-latest", active_dir=active, metadata=metadata
        )


def test_offloader_upload_rejects_missing_or_failed_remote_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active, metadata, schema = _artifact(tmp_path)
    offloader = FinalizedArtifactOffloader(
        hub_api=mock.Mock(),
        repo_id="owner/output",
        staging_revision="run",
        run_id="run",
        local_cache_dir=tmp_path / "cache",
        schema=schema,
    )
    monkeypatch.setattr(offloader, "_ensure_branch", lambda: None)
    monkeypatch.setattr(offloader, "inspect", mock.Mock(side_effect=[None, None]))
    offloader.hub_api.upload_folder.side_effect = RuntimeError("upload failed")
    with pytest.raises(FinalizedOffloadError, match="upload failed"):
        offloader.upload_and_verify(
            shard_key="a-latest", active_dir=active, metadata=metadata
        )
    offloader.hub_api.upload_folder.side_effect = None
    offloader.hub_api.upload_folder.reset_mock()
    monkeypatch.setattr(offloader, "inspect", mock.Mock(side_effect=[None, None]))
    with pytest.raises(FinalizedOffloadError, match="not visible"):
        offloader.upload_and_verify(
            shard_key="a-latest", active_dir=active, metadata=metadata
        )
