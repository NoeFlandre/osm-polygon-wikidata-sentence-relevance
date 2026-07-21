"""Verified Hugging Face staging for streamed shard checkpoints.

Each checkpoint is committed atomically to a dedicated branch in the
existing output dataset.  Remote metadata is authoritative.  A checkpoint
is reusable only after its identity, byte size, schema fingerprint and
content hash have been verified.  Large Parquet files are downloaded only
when the Hub LFS SHA-256 is unavailable or when a caller explicitly
materialises a checkpoint for finalisation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_sentence_relevance.application.checkpoint import (
    segmented_schema_sha256,
)
from osm_polygon_sentence_relevance.contracts.schemas import (
    SEGMENTED_SENTENCES_SCHEMA,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_FIELDS = (
    "source_commit",
    "input_dataset_revision",
    "pipeline_version",
    "model_name",
    "batch_size",
)


class CheckpointOffloadError(RuntimeError):
    """A staging checkpoint could not be safely uploaded or verified."""


@dataclass(frozen=True, slots=True)
class OffloadHandle:
    """Verified remote checkpoint descriptor."""

    repo_id: str
    run_id: str
    shard_key: str
    staging_revision: str
    folder_path: str
    expected_table_sha256: str
    computed_table_sha256: str
    table_bytes: int
    metadata: Mapping[str, Any]
    local_table_path: Path | None = None
    local_metadata_path: Path | None = None


def _lazy_hf_hub_download() -> Callable[..., str]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise CheckpointOffloadError(
            "huggingface_hub is required for checkpoint staging"
        ) from exc
    return hf_hub_download


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phys(path: Path) -> Path:
    return Path(os.path.realpath(str(path)))


def _validate_shard_key(shard_key: str) -> None:
    if not isinstance(shard_key, str) or not shard_key:
        raise ValueError("shard_key must be a non-blank string")
    if any(
        not ((c.isascii() and c.isalnum() and c == c.lower()) or c in "-_.")
        for c in shard_key
    ):
        raise ValueError(f"invalid shard_key: {shard_key!r}")


def _validate_run_id(run_id: str) -> None:
    if (
        not isinstance(run_id, str)
        or not run_id
        or "/" in run_id
        or any(c.isspace() for c in run_id)
    ):
        raise ValueError("run_id must be non-blank and contain no slash or whitespace")


def _validate_metadata(
    payload: Any,
    *,
    shard_key: str,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CheckpointOffloadError("remote checkpoint metadata must be an object")
    required = (
        "shard_key",
        "segmented_table_sha256",
        "segmented_table_bytes",
        "segmented_schema_sha256",
        *_IDENTITY_FIELDS,
    )
    for field in required:
        if field not in payload:
            raise CheckpointOffloadError(
                f"checkpoint {shard_key!r} metadata missing {field!r}"
            )
    if payload["shard_key"] != shard_key:
        raise CheckpointOffloadError(
            f"checkpoint shard_key mismatch: expected {shard_key!r}, "
            f"got {payload['shard_key']!r}"
        )
    sha = payload["segmented_table_sha256"]
    if not isinstance(sha, str) or not _HEX64.fullmatch(sha):
        raise CheckpointOffloadError(
            f"checkpoint {shard_key!r} has invalid segmented_table_sha256"
        )
    size = payload["segmented_table_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise CheckpointOffloadError(
            f"checkpoint {shard_key!r} has invalid segmented_table_bytes"
        )
    if payload["segmented_schema_sha256"] != segmented_schema_sha256():
        raise CheckpointOffloadError(
            f"checkpoint {shard_key!r} segmented schema fingerprint mismatch"
        )
    if expected_identity is not None:
        for field in _IDENTITY_FIELDS:
            if payload[field] != expected_identity.get(field):
                raise CheckpointOffloadError(
                    f"checkpoint {shard_key!r} {field} mismatch: "
                    f"expected {expected_identity.get(field)!r}, got {payload[field]!r}"
                )
    return dict(payload)


def _entry_lfs_sha(entry: Any) -> str | None:
    lfs = getattr(entry, "lfs", None)
    value = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    return value if isinstance(value, str) and _HEX64.fullmatch(value) else None


def _list_files(
    *, hub_api: Any, repo_id: str, revision: str, folder_path: str
) -> dict[str, Any]:
    try:
        entries = hub_api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            path_in_repo=folder_path,
            recursive=True,
            expand=True,
        )
        return {
            (getattr(entry, "path", "") or "").removeprefix(folder_path + "/"): entry
            for entry in entries
            if (getattr(entry, "path", "") or "").startswith(folder_path + "/")
        }
    except Exception as exc:
        text = str(exc).lower()
        if "404" in text or "revision not found" in text or "not found" in text:
            return {}
        raise CheckpointOffloadError(
            f"could not inspect staging checkpoint {folder_path!r}: {exc}"
        ) from exc


def _download_file(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    local_dir: Path,
) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = _lazy_hf_hub_download()(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="dataset",
            local_dir=str(local_dir),
            force_download=True,
        )
    except Exception as exc:
        raise CheckpointOffloadError(
            f"readback failed for {filename!r}: {exc}"
        ) from exc
    path = _phys(Path(result))
    if not path.is_file():
        raise CheckpointOffloadError(f"readback did not create {filename!r}")
    return path


def _handle_from_files(
    *,
    hub_api: Any,
    repo_id: str,
    staging_revision: str,
    run_id: str,
    shard_key: str,
    files: Mapping[str, Any],
    local_cache_dir: Path,
    expected_identity: Mapping[str, Any] | None = None,
    materialize: bool = False,
) -> OffloadHandle:
    expected_names = {"segmented.parquet", "metadata.json"}
    actual_names = set(files)
    if actual_names != expected_names:
        raise CheckpointOffloadError(
            f"checkpoint {shard_key!r} entries mismatch: "
            f"expected {sorted(expected_names)!r}, got {sorted(actual_names)!r}"
        )
    folder = f"checkpoints/{run_id}/{shard_key}"
    cache = _phys(local_cache_dir) / run_id / shard_key
    meta_path = _download_file(
        repo_id=repo_id,
        revision=staging_revision,
        filename=f"{folder}/metadata.json",
        local_dir=cache,
    )
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointOffloadError(
            f"checkpoint {shard_key!r} has invalid metadata JSON"
        ) from exc
    metadata = _validate_metadata(
        payload, shard_key=shard_key, expected_identity=expected_identity
    )
    table_entry = files["segmented.parquet"]
    remote_size = getattr(table_entry, "size", None)
    if (
        isinstance(remote_size, int)
        and remote_size != metadata["segmented_table_bytes"]
    ):
        raise CheckpointOffloadError(
            f"checkpoint {shard_key!r} remote byte size mismatch"
        )

    local_table: Path | None = None
    remote_lfs_sha = _entry_lfs_sha(table_entry)
    if materialize or remote_lfs_sha is None:
        local_table = _download_file(
            repo_id=repo_id,
            revision=staging_revision,
            filename=f"{folder}/segmented.parquet",
            local_dir=cache,
        )
        if local_table.stat().st_size != metadata["segmented_table_bytes"]:
            raise CheckpointOffloadError(
                f"checkpoint {shard_key!r} downloaded byte size mismatch"
            )
        computed = _sha256_file(local_table)
        if computed != metadata["segmented_table_sha256"]:
            raise CheckpointOffloadError(
                f"checkpoint {shard_key!r} readback SHA-256 mismatch"
            )
        if materialize:
            import pyarrow.parquet as pq

            if not pq.read_schema(local_table).equals(SEGMENTED_SENTENCES_SCHEMA):
                raise CheckpointOffloadError(
                    f"checkpoint {shard_key!r} Parquet schema mismatch"
                )
        else:
            local_table.unlink()
            local_table = None
    elif remote_lfs_sha != metadata["segmented_table_sha256"]:
        raise CheckpointOffloadError(
            f"checkpoint {shard_key!r} Hub LFS SHA-256 mismatch"
        )

    return OffloadHandle(
        repo_id=repo_id,
        run_id=run_id,
        shard_key=shard_key,
        staging_revision=staging_revision,
        folder_path=folder,
        expected_table_sha256=metadata["segmented_table_sha256"],
        computed_table_sha256=metadata["segmented_table_sha256"],
        table_bytes=metadata["segmented_table_bytes"],
        metadata=metadata,
        local_table_path=local_table,
        local_metadata_path=meta_path,
    )


def inspect_remote_checkpoint(
    *,
    hub_api: Any,
    repo_id: str,
    staging_revision: str,
    run_id: str,
    shard_key: str,
    local_cache_dir: Path,
    expected_identity: Mapping[str, Any] | None = None,
    materialize: bool = False,
) -> OffloadHandle | None:
    """Return one verified remote checkpoint, or ``None`` if absent."""

    _validate_run_id(run_id)
    _validate_shard_key(shard_key)
    folder = f"checkpoints/{run_id}/{shard_key}"
    files = _list_files(
        hub_api=hub_api,
        repo_id=repo_id,
        revision=staging_revision,
        folder_path=folder,
    )
    if not files:
        return None
    return _handle_from_files(
        hub_api=hub_api,
        repo_id=repo_id,
        staging_revision=staging_revision,
        run_id=run_id,
        shard_key=shard_key,
        files=files,
        local_cache_dir=local_cache_dir,
        expected_identity=expected_identity,
        materialize=materialize,
    )


def materialize_checkpoint(
    handle: OffloadHandle, *, hub_api: Any, local_cache_dir: Path
) -> OffloadHandle:
    """Download and revalidate the Parquet bytes for ``handle``."""

    result = inspect_remote_checkpoint(
        hub_api=hub_api,
        repo_id=handle.repo_id,
        staging_revision=handle.staging_revision,
        run_id=handle.run_id,
        shard_key=handle.shard_key,
        local_cache_dir=local_cache_dir,
        expected_identity=handle.metadata,
        materialize=True,
    )
    if result is None:  # pragma: no cover - concurrent remote deletion
        raise CheckpointOffloadError(
            f"checkpoint {handle.shard_key!r} disappeared during materialisation"
        )
    return result


class CheckpointOffloader:
    """Atomic upload and independent readback verifier."""

    def __init__(
        self,
        *,
        hub_api: Any,
        repo_id: str,
        staging_revision: str,
        run_id: str,
        local_cache_dir: Path,
    ) -> None:
        if not isinstance(repo_id, str) or "/" not in repo_id or not repo_id.strip():
            raise ValueError("repo_id must be a non-blank owner/name string")
        if not isinstance(staging_revision, str) or not staging_revision.strip():
            raise ValueError("staging_revision must be non-blank")
        _validate_run_id(run_id)
        self.hub_api = hub_api
        self.repo_id = repo_id
        self.staging_revision = staging_revision
        self.run_id = run_id
        self.local_cache_dir = _phys(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

    def _folder_path(self, shard_key: str) -> str:
        return f"checkpoints/{self.run_id}/{shard_key}"

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
            message = str(exc).lower()
            if "already exists" in message or "409" in message:
                return
            raise CheckpointOffloadError(
                f"could not ensure staging branch {self.staging_revision!r}: {exc}"
            ) from exc

    def upload_and_verify(
        self, *, shard_key: str, active_dir: Path, metadata: dict[str, Any]
    ) -> OffloadHandle:
        _validate_shard_key(shard_key)
        active = _phys(active_dir)
        table = active / "segmented.parquet"
        meta_path = active / "metadata.json"
        if not active.is_dir() or not table.is_file() or not meta_path.is_file():
            raise CheckpointOffloadError(
                f"local checkpoint {shard_key!r} is incomplete"
            )
        validated = _validate_metadata(metadata, shard_key=shard_key)
        computed = _sha256_file(table)
        if computed != validated["segmented_table_sha256"]:
            raise CheckpointOffloadError(f"local SHA-256 mismatch for {shard_key!r}")
        if table.stat().st_size != validated["segmented_table_bytes"]:
            raise CheckpointOffloadError(f"local byte size mismatch for {shard_key!r}")

        self._ensure_branch()
        existing = inspect_remote_checkpoint(
            hub_api=self.hub_api,
            repo_id=self.repo_id,
            staging_revision=self.staging_revision,
            run_id=self.run_id,
            shard_key=shard_key,
            local_cache_dir=self.local_cache_dir,
            expected_identity=validated,
            materialize=False,
        )
        if existing is not None:
            return existing

        folder = self._folder_path(shard_key)
        try:
            self.hub_api.upload_folder(
                repo_id=self.repo_id,
                folder_path=str(active),
                path_in_repo=folder,
                revision=self.staging_revision,
                commit_message=f"Add streamed checkpoint {shard_key}",
                repo_type="dataset",
            )
        except Exception as exc:
            raise CheckpointOffloadError(
                f"checkpoint upload failed for {shard_key!r}: {exc}"
            ) from exc

        verified = inspect_remote_checkpoint(
            hub_api=self.hub_api,
            repo_id=self.repo_id,
            staging_revision=self.staging_revision,
            run_id=self.run_id,
            shard_key=shard_key,
            local_cache_dir=self.local_cache_dir,
            expected_identity=validated,
            materialize=True,
        )
        if verified is None:  # pragma: no cover - impossible successful commit
            raise CheckpointOffloadError(
                f"uploaded checkpoint {shard_key!r} is not visible"
            )
        if verified.local_table_path is not None:
            verified.local_table_path.unlink()
        return OffloadHandle(
            repo_id=verified.repo_id,
            run_id=verified.run_id,
            shard_key=verified.shard_key,
            staging_revision=verified.staging_revision,
            folder_path=verified.folder_path,
            expected_table_sha256=verified.expected_table_sha256,
            computed_table_sha256=verified.computed_table_sha256,
            table_bytes=verified.table_bytes,
            metadata=verified.metadata,
            local_metadata_path=verified.local_metadata_path,
        )


def discover_run(
    *,
    hub_api: Any,
    repo_id: str,
    run_id: str,
    local_cache_dir: Path,
    staging_revision: str | None = None,
    expected_identity: Mapping[str, Any] | None = None,
) -> list[OffloadHandle]:
    """Discover all authoritative checkpoints without mirroring Parquet files."""

    _validate_run_id(run_id)
    revision = staging_revision or run_id
    root = f"checkpoints/{run_id}"
    files = _list_files(
        hub_api=hub_api,
        repo_id=repo_id,
        revision=revision,
        folder_path=root,
    )
    if not files:
        return []
    grouped: dict[str, dict[str, Any]] = {}
    for relative, entry in files.items():
        # ``_list_files`` uses ``expand=True``, which means folder
        # entries appear alongside files.  Folder entries have no
        # ``/`` in their relative path; the discovery invariant is
        # one folder per shard_key plus the two expected files, so
        # single-segment relative paths are ignored here.
        if "/" not in relative:
            continue
        parts = relative.split("/", 1)
        grouped.setdefault(parts[0], {})[parts[1]] = entry
    if not grouped:
        return []
    return [
        _handle_from_files(
            hub_api=hub_api,
            repo_id=repo_id,
            staging_revision=revision,
            run_id=run_id,
            shard_key=shard_key,
            files=grouped[shard_key],
            local_cache_dir=local_cache_dir,
            expected_identity=expected_identity,
            materialize=False,
        )
        for shard_key in sorted(grouped)
    ]


__all__ = [
    "CheckpointOffloadError",
    "CheckpointOffloader",
    "OffloadHandle",
    "discover_run",
    "inspect_remote_checkpoint",
    "materialize_checkpoint",
]
