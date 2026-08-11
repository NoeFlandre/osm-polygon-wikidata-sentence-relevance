"""Verified HF staging for per-shard finalized V2 tables.

Finalized artifacts are internal inputs to the sampler, not public dataset
files.  They live beside the segmented checkpoints on the run's staging
branch and are reusable after an allocation disappears.  Every artifact is
validated by identity, byte size, schema fingerprint, row count, and SHA-256
before it is accepted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .offload import (
    _HEX64,
    CheckpointOffloadError,
    _download_file,
    _entry_lfs_sha,
    _list_files,
    _validate_run_id,
    _validate_shard_key,
)


class FinalizedOffloadError(CheckpointOffloadError):
    """A finalized V2 shard could not be safely staged or reused."""


@dataclass(frozen=True, slots=True)
class FinalizedShardHandle:
    """Verified descriptor for one durable finalized shard."""

    repo_id: str
    run_id: str
    shard_key: str
    staging_revision: str
    folder_path: str
    table_sha256: str
    table_bytes: int
    row_count: int
    schema_sha256: str
    metadata: Mapping[str, Any]
    local_table_path: Path | None = None


def schema_sha256(schema: pa.Schema) -> str:
    """Return a stable fingerprint for an Arrow schema."""

    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_metadata(
    payload: Any,
    *,
    shard_key: str,
    schema: pa.Schema,
    expected_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise FinalizedOffloadError("finalized metadata must be an object")
    required = (
        "schema_version",
        "shard_key",
        "table_sha256",
        "table_bytes",
        "row_count",
        "schema_sha256",
    )
    if any(field not in payload for field in required):
        raise FinalizedOffloadError("finalized metadata is incomplete")
    if payload["schema_version"] != 1 or payload["shard_key"] != shard_key:
        raise FinalizedOffloadError("finalized metadata identity mismatch")
    if not isinstance(payload["table_sha256"], str) or not _HEX64.fullmatch(
        payload["table_sha256"]
    ):
        raise FinalizedOffloadError("finalized table SHA-256 is invalid")
    if (
        isinstance(payload["table_bytes"], bool)
        or not isinstance(payload["table_bytes"], int)
        or payload["table_bytes"] <= 0
    ):
        raise FinalizedOffloadError("finalized table byte count is invalid")
    if (
        isinstance(payload["row_count"], bool)
        or not isinstance(payload["row_count"], int)
        or payload["row_count"] < 0
    ):
        raise FinalizedOffloadError("finalized row count is invalid")
    expected_schema = schema_sha256(schema)
    if payload["schema_sha256"] != expected_schema:
        raise FinalizedOffloadError("finalized schema fingerprint mismatch")
    if expected_identity is not None:
        for key, expected in expected_identity.items():
            if payload.get(key) != expected:
                raise FinalizedOffloadError(f"{key} mismatch in finalized metadata")
    return dict(payload)


def _handle_from_files(
    *,
    hub_api: Any,
    repo_id: str,
    staging_revision: str,
    run_id: str,
    shard_key: str,
    files: Mapping[str, Any],
    local_cache_dir: Path,
    schema: pa.Schema,
    expected_identity: Mapping[str, Any] | None,
    materialize: bool,
) -> FinalizedShardHandle:
    if set(files) != {"finalized.parquet", "metadata.json"}:
        raise FinalizedOffloadError("finalized artifact entries are incomplete")
    folder = f"finalized/{run_id}/{shard_key}"
    cache = local_cache_dir / run_id / shard_key
    metadata_path = _download_file(
        repo_id=repo_id,
        revision=staging_revision,
        filename=f"{folder}/metadata.json",
        local_dir=cache,
    )
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalizedOffloadError("finalized metadata JSON is invalid") from exc
    metadata = _validate_metadata(
        payload,
        shard_key=shard_key,
        schema=schema,
        expected_identity=expected_identity,
    )
    entry = files["finalized.parquet"]
    remote_size = getattr(entry, "size", None)
    if isinstance(remote_size, int) and remote_size != metadata["table_bytes"]:
        raise FinalizedOffloadError("finalized remote byte size mismatch")
    remote_sha = _entry_lfs_sha(entry)
    if remote_sha is not None and remote_sha != metadata["table_sha256"]:
        raise FinalizedOffloadError("finalized remote SHA-256 mismatch")
    local_table: Path | None = None
    if materialize or remote_sha is None:
        local_table = _download_file(
            repo_id=repo_id,
            revision=staging_revision,
            filename=f"{folder}/finalized.parquet",
            local_dir=cache,
        )
        if local_table.stat().st_size != metadata["table_bytes"]:
            raise FinalizedOffloadError("finalized downloaded byte size mismatch")
        if _sha256_file(local_table) != metadata["table_sha256"]:
            raise FinalizedOffloadError("finalized readback SHA-256 mismatch")
        parquet = pq.ParquetFile(local_table)
        if not parquet.schema_arrow.equals(schema):
            raise FinalizedOffloadError("finalized Parquet schema mismatch")
        if (
            parquet.metadata is None
            or parquet.metadata.num_rows != metadata["row_count"]
        ):
            raise FinalizedOffloadError("finalized row count mismatch")
        if not materialize:
            local_table.unlink()
            local_table = None
    return FinalizedShardHandle(
        repo_id=repo_id,
        run_id=run_id,
        shard_key=shard_key,
        staging_revision=staging_revision,
        folder_path=folder,
        table_sha256=metadata["table_sha256"],
        table_bytes=metadata["table_bytes"],
        row_count=metadata["row_count"],
        schema_sha256=metadata["schema_sha256"],
        metadata=metadata,
        local_table_path=local_table,
    )


class FinalizedArtifactOffloader:
    """Upload and independently verify one finalized V2 shard."""

    def __init__(
        self,
        *,
        hub_api: Any,
        repo_id: str,
        staging_revision: str,
        run_id: str,
        local_cache_dir: Path,
        schema: pa.Schema,
        expected_identity: Mapping[str, Any] | None = None,
    ) -> None:
        _validate_run_id(run_id)
        if not isinstance(repo_id, str) or "/" not in repo_id:
            raise ValueError("repo_id must be owner/name")
        self.hub_api = hub_api
        self.repo_id = repo_id
        self.staging_revision = staging_revision
        self.run_id = run_id
        self.local_cache_dir = Path(local_cache_dir)
        self.schema = schema
        self.expected_identity = dict(expected_identity or {})

    def _ensure_branch(self) -> None:
        try:
            self.hub_api.create_branch(
                repo_id=self.repo_id,
                branch=self.staging_revision,
                revision="main",
                repo_type="dataset",
                exist_ok=True,
            )
        except Exception as exc:
            text = str(exc).lower()
            if "already exists" not in text and "409" not in text:
                raise FinalizedOffloadError("could not ensure staging branch") from exc

    def inspect(
        self, shard_key: str, *, materialize: bool
    ) -> FinalizedShardHandle | None:
        _validate_shard_key(shard_key)
        folder = f"finalized/{self.run_id}/{shard_key}"
        files = _list_files(
            hub_api=self.hub_api,
            repo_id=self.repo_id,
            revision=self.staging_revision,
            folder_path=folder,
        )
        if not files:
            return None
        return _handle_from_files(
            hub_api=self.hub_api,
            repo_id=self.repo_id,
            staging_revision=self.staging_revision,
            run_id=self.run_id,
            shard_key=shard_key,
            files=files,
            local_cache_dir=self.local_cache_dir,
            schema=self.schema,
            expected_identity=self.expected_identity,
            materialize=materialize,
        )

    def upload_and_verify(
        self, *, shard_key: str, active_dir: Path, metadata: Mapping[str, Any]
    ) -> FinalizedShardHandle:
        _validate_shard_key(shard_key)
        table = Path(active_dir) / "finalized.parquet"
        metadata_path = Path(active_dir) / "metadata.json"
        if (
            not table.is_file()
            or table.is_symlink()
            or not metadata_path.is_file()
            or metadata_path.is_symlink()
        ):
            raise FinalizedOffloadError("local finalized artifact is incomplete")
        payload = _validate_metadata(
            dict(metadata),
            shard_key=shard_key,
            schema=self.schema,
            expected_identity=self.expected_identity,
        )
        try:
            on_disk = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FinalizedOffloadError(
                "local finalized metadata JSON is invalid"
            ) from exc
        if on_disk != payload:
            raise FinalizedOffloadError(
                "local finalized metadata does not match payload"
            )
        if _sha256_file(table) != payload["table_sha256"]:
            raise FinalizedOffloadError("local finalized SHA-256 mismatch")
        if table.stat().st_size != payload["table_bytes"]:
            raise FinalizedOffloadError("local finalized byte size mismatch")
        parquet = pq.ParquetFile(table)
        if not parquet.schema_arrow.equals(self.schema):
            raise FinalizedOffloadError("local finalized schema mismatch")
        if (
            parquet.metadata is None
            or parquet.metadata.num_rows != payload["row_count"]
        ):
            raise FinalizedOffloadError("local finalized row count mismatch")
        self._ensure_branch()
        existing = self.inspect(shard_key, materialize=False)
        if existing is not None:
            return existing
        try:
            self.hub_api.upload_folder(
                repo_id=self.repo_id,
                folder_path=str(active_dir),
                path_in_repo=f"finalized/{self.run_id}/{shard_key}",
                revision=self.staging_revision,
                commit_message=f"Add finalized V2 shard {shard_key}",
                repo_type="dataset",
            )
        except Exception as exc:
            raise FinalizedOffloadError("finalized artifact upload failed") from exc
        verified = self.inspect(shard_key, materialize=True)
        if verified is None or verified.local_table_path is None:
            raise FinalizedOffloadError("uploaded finalized artifact is not visible")
        verified.local_table_path.unlink(missing_ok=True)
        return replace(verified, local_table_path=None)


__all__ = [
    "FinalizedArtifactOffloader",
    "FinalizedOffloadError",
    "FinalizedShardHandle",
    "schema_sha256",
]
