"""Atomic checkpoint publication, recovery, loading, and progress writes."""

from __future__ import annotations

import json
import os
import stat
import time
import uuid
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.contracts.schemas import SEGMENTED_SENTENCES_SCHEMA
from osm_polygon_sentence_relevance.ingestion.discovery import RegionShardSet
from osm_polygon_sentence_relevance.output.checksum import sha256_file
from osm_polygon_sentence_relevance.sentences.segmentation import SegmentationReport

from .common import (
    _DIR_MODE,
    _FILE_MODE,
    _LOWER_HEX_40,
    _METADATA_SCHEMA_VERSION,
    HEARTBEAT_NAME,
    SHARD_METADATA_NAME,
    SHARD_PARQUET_NAME,
    SHARDS_ACTIVE_DIRNAME,
    SHARDS_QUARANTINE_DIRNAME,
    SHARDS_STAGING_PREFIX,
    CheckpointPublicationError,
    CheckpointValidationError,
    _valid_shard_key,
    segmented_schema_sha256,
)
from .inventory import SourceFileEntry, compute_shard_source_manifest
from .io import _atomic_write_bytes, _atomic_write_parquet, _fsync_dir_strict
from .validation import (
    _ensure_safe_active_dir,
    _segmentation_report_from_dict,
    _segmentation_report_to_dict,
    validate_checkpoint_metadata,
)


def _make_staging_dirname(shard_key: str) -> str:
    suffix = uuid.uuid4().hex[:8]
    return f"{SHARDS_STAGING_PREFIX}{shard_key}.{int(time.time())}.{suffix}"


def _verify_pre_publish_manifest(
    shard: RegionShardSet,
    *,
    initial_manifest: list[SourceFileEntry],
    input_root: Path,
) -> list[SourceFileEntry]:
    """Re-hash the source files immediately before publication and
    abort if the manifest drifted since inventory construction.

    Returns the freshly-computed manifest so the caller can reuse it
    for the metadata payload without hashing again.
    """
    current_manifest = compute_shard_source_manifest(shard, input_root=input_root)
    if current_manifest != initial_manifest:
        raise CheckpointValidationError(
            f"source manifest drift for shard {shard.shard_key!r}: "
            "input files changed during segmentation; aborting publication"
        )
    return current_manifest


def _validate_staged_checkpoint(
    *,
    staging_dir: Path,
    staging_parquet: Path,
    staging_metadata: Path,
    shard_key: str,
    meta_bytes: bytes,
    recorded_sha: str,
) -> None:
    """Fully validate the just-written staged checkpoint in place.

    The staging directory must contain exactly two entries, both
    regular files, both mode ``0o600``. The Parquet file must open
    and its schema must equal :data:`SEGMENTED_SENTENCES_SCHEMA`. The
    Parquet's SHA-256 must equal the recorded digest. The metadata
    JSON must parse and pass the strict validator. Any failure
    raises :class:`CheckpointValidationError`; the staging directory
    is preserved as evidence and the active slot is not touched.
    """
    # 1. Layout: exactly two regular files, no symlinks, no extras.
    try:
        lst_dir = os.lstat(staging_dir)
    except OSError as exc:  # pragma: no cover (filesystem race)
        raise CheckpointValidationError(
            f"could not lstat staging directory {staging_dir}: {exc}"
        ) from exc
    if stat.S_ISLNK(lst_dir.st_mode) or not stat.S_ISDIR(
        lst_dir.st_mode
    ):  # pragma: no cover (defensive)
        raise CheckpointValidationError(
            f"staging directory {staging_dir} is not a regular directory"
        )
    dir_mode = lst_dir.st_mode & 0o777
    if dir_mode != _DIR_MODE:  # pragma: no cover (defensive)
        raise CheckpointValidationError(
            f"staging directory {staging_dir} has mode {dir_mode:o}, "
            f"expected {_DIR_MODE:o}"
        )
    try:
        names = sorted(p.name for p in staging_dir.iterdir())
    except OSError as exc:  # pragma: no cover (filesystem race)
        raise CheckpointValidationError(
            f"could not iter staging directory {staging_dir}: {exc}"
        ) from exc
    expected = sorted([SHARD_PARQUET_NAME, SHARD_METADATA_NAME])
    if names != expected:
        raise CheckpointValidationError(
            f"staging directory {staging_dir} has unexpected entries: {names}"
        )
    for p in (staging_parquet, staging_metadata):
        lst = os.lstat(p)
        if stat.S_ISLNK(lst.st_mode):
            raise CheckpointValidationError(f"staging entry {p} is a symlink; refusing")
        if not stat.S_ISREG(lst.st_mode):  # pragma: no cover (defensive)
            raise CheckpointValidationError(f"staging entry {p} is not a regular file")
        mode = lst.st_mode & 0o777
        if mode != _FILE_MODE:
            raise CheckpointValidationError(
                f"staging entry {p} has mode {mode:o}, expected {_FILE_MODE:o}"
            )

    # 2. Parquet: re-read, verify schema, verify SHA-256 against bytes.
    try:
        table = pq.read_table(staging_parquet)
    except Exception as exc:
        raise CheckpointValidationError(
            f"staged parquet for shard {shard_key!r} cannot be read: {exc}"
        ) from exc
    if not table.schema.equals(SEGMENTED_SENTENCES_SCHEMA):
        raise CheckpointValidationError(
            f"staged parquet for shard {shard_key!r} has wrong schema"
        )
    actual_sha = sha256_file(staging_parquet)
    if actual_sha != recorded_sha.lower():
        raise CheckpointValidationError(
            f"staged parquet SHA-256 mismatch for shard {shard_key!r}: "
            f"recorded {recorded_sha}, got {actual_sha}"
        )

    # 3. Metadata: parse + strict validation.
    try:
        meta_for_validation = json.loads(meta_bytes.decode("utf-8"))
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:  # pragma: no cover (defensive)
        raise CheckpointValidationError(
            f"staged metadata for shard {shard_key!r} is malformed: {exc}"
        ) from exc
    validate_checkpoint_metadata(
        meta_for_validation,
        shard_key=shard_key,
        expect_active=False,
    )


def publish_shard_checkpoint(
    *,
    work_dir: Path,
    shard: RegionShardSet,
    input_root: Path,
    table: Any,
    report: SegmentationReport,
    input_dataset_revision: str,
    pipeline_version: str,
    source_commit: str,
    model_name: str,
    batch_size: int,
    verified_manifest: list[SourceFileEntry],
) -> Path:
    """Publish a shard checkpoint as a whole-directory atomic rename.

    Workflow:

    1. Create a unique staging sibling under ``${work_dir}/shards/``.
    2. Write ``segmented.parquet`` and ``metadata.json`` into staging,
       each with mode 0o600, fsync each file and the staging directory.
    3. Validate the just-written contents in place using the strict
       metadata validator.
    4. **Refuse** to publish if ``active/<shard_key>`` already exists.
    5. Atomically ``os.rename`` staging -> active.
    6. fsync the parent directory that contains the renamed entry.

    ``verified_manifest`` is the manifest produced by
    :func:`_verify_pre_publish_manifest`; it is the one recorded in
    metadata. The pipeline must hash shard sources once at inventory
    construction and once again here; this function does NOT hash
    source files itself.

    On any failure inside steps 1-4, the staging directory is preserved
    on disk as evidence and the active slot is not touched. On failure
    of step 5, the staging directory is also preserved.
    """
    if not _valid_shard_key(shard.shard_key):
        raise CheckpointValidationError(f"invalid shard_key: {shard.shard_key!r}")
    # Validate identity upfront.
    for field_name, value in (
        ("input_dataset_revision", input_dataset_revision),
        ("pipeline_version", pipeline_version),
        ("source_commit", source_commit),
        ("model_name", model_name),
    ):
        if not isinstance(value, str) or not value.strip():
            raise CheckpointValidationError(f"{field_name} must be a non-blank string")
    if not _LOWER_HEX_40.match(source_commit):
        raise CheckpointValidationError(
            "source_commit must be lowercase 40-character hex"
        )
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise CheckpointValidationError("batch_size must be a positive integer")

    shards_root = work_dir / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    os.chmod(shards_root, _DIR_MODE)
    _fsync_dir_strict(work_dir)

    active_root = shards_root / SHARDS_ACTIVE_DIRNAME
    active_root.mkdir(parents=True, exist_ok=True)
    os.chmod(active_root, _DIR_MODE)
    _fsync_dir_strict(shards_root)

    active_target = active_root / shard.shard_key
    if active_target.exists():
        raise CheckpointPublicationError(
            f"active checkpoint already exists at {active_target}; refusing to overwrite"
        )

    staging_dir = shards_root / _make_staging_dirname(shard.shard_key)
    staging_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(staging_dir, _DIR_MODE)
    _fsync_dir_strict(shards_root)

    staging_parquet = staging_dir / SHARD_PARQUET_NAME
    staging_metadata = staging_dir / SHARD_METADATA_NAME

    try:
        _atomic_write_parquet(table, staging_parquet)

        sha = sha256_file(staging_parquet)
        metadata_payload = {
            "schema_version": _METADATA_SCHEMA_VERSION,
            "shard_key": shard.shard_key,
            "input_dataset_revision": input_dataset_revision,
            "pipeline_version": pipeline_version,
            "source_commit": source_commit,
            "model_name": model_name,
            "batch_size": int(batch_size),
            "input_root": str(Path(input_root).expanduser().resolve(strict=False)),
            "source_files": [e.to_dict() for e in verified_manifest],
            "segmentation_report": _segmentation_report_to_dict(report),
            "segmented_table_sha256": sha,
            "segmented_table_bytes": staging_parquet.stat().st_size,
            "segmented_schema_sha256": segmented_schema_sha256(),
            "completed_at_unix": int(time.time()),
        }
        meta_bytes = json.dumps(
            metadata_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        _atomic_write_bytes(staging_metadata, meta_bytes)
        _fsync_dir_strict(staging_dir)

        # Full staged-checkpoint validation. Anything wrong here
        # aborts publication without ever renaming staging to active.
        _validate_staged_checkpoint(
            staging_dir=staging_dir,
            staging_parquet=staging_parquet,
            staging_metadata=staging_metadata,
            shard_key=shard.shard_key,
            meta_bytes=meta_bytes,
            recorded_sha=sha,
        )
    except Exception:
        # Leave staging_dir intact as evidence; active is untouched.
        raise

    # Refuse to overwrite active if it materialized between the pre-check
    # above and the rename below. If it does, the staging directory
    # remains as evidence and the function raises.
    if active_target.exists():
        raise CheckpointPublicationError(
            f"active checkpoint materialized before rename at {active_target}; "
            f"staging preserved at {staging_dir}"
        )
    os.rename(staging_dir, active_target)
    _fsync_dir_strict(shards_root)
    return active_target


def _make_quarantine_dirname(shard_key: str, reason: str) -> str:
    suffix = uuid.uuid4().hex[:8]
    return f"{shard_key}.{int(time.time())}.{suffix}"


def quarantine_shard_checkpoint(
    *,
    work_dir: Path,
    shard_key: str,
    reason: str,
) -> Path | None:
    """Move ``active/<shard_key>`` into ``quarantine/`` preserving bytes.

    Returns the new quarantine directory path, or ``None`` if no active
    checkpoint exists. Raises on failure (including ``EXDEV``); the
    active directory is **never** touched on failure.
    """
    if not _valid_shard_key(shard_key):
        raise CheckpointValidationError(f"invalid shard_key: {shard_key!r}")
    shards_root = work_dir / "shards"
    active_target = shards_root / SHARDS_ACTIVE_DIRNAME / shard_key
    if not active_target.exists():
        return None

    quarantine_root = shards_root / SHARDS_QUARANTINE_DIRNAME
    quarantine_root.mkdir(parents=True, exist_ok=True)
    os.chmod(quarantine_root, _DIR_MODE)
    _fsync_dir_strict(shards_root)

    base_name = _make_quarantine_dirname(shard_key, reason)
    candidate = quarantine_root / base_name
    for _ in range(8):
        if not candidate.exists():
            break
        candidate = quarantine_root / f"{base_name}.{uuid.uuid4().hex[:4]}"
    else:  # pragma: no cover (practically unreachable)
        raise CheckpointPublicationError(
            f"could not allocate unique quarantine name under {quarantine_root}"
        )

    # Same-filesystem rename only. Any failure (incl. EXDEV) leaves the
    # active directory untouched and propagates to the caller.
    os.rename(active_target, candidate)
    _fsync_dir_strict(shards_root)
    return candidate


def _read_persisted_metadata(metadata_path: Path) -> dict[str, Any]:
    return json.loads(metadata_path.read_bytes().decode("utf-8"))


def load_shard_checkpoint(
    work_dir: Path,
    shard_key: str,
    *,
    input_dataset_revision: str,
    pipeline_version: str,
    source_commit: str,
    model_name: str,
    batch_size: int,
    input_root: Path,
    current_manifest: list[SourceFileEntry] | None = None,
) -> tuple[Any, SegmentationReport, dict[str, Any]]:
    """Return ``(table, report, metadata)`` iff the active checkpoint is
    valid for the current invocation. Raises
    :class:`CheckpointValidationError` on any mismatch.

    Performs filesystem-level validation first (no symlinks, no
    non-regular-files, mode 0o700 on the directory, mode 0o600 on the
    files), then runs the strict metadata validator.
    """
    if not _valid_shard_key(shard_key):
        raise CheckpointValidationError(f"invalid shard_key: {shard_key!r}")

    active_target = work_dir / "shards" / SHARDS_ACTIVE_DIRNAME / shard_key
    try:
        if not active_target.exists():
            raise CheckpointValidationError(
                f"no active checkpoint for shard {shard_key!r}"
            )
    except OSError as exc:
        raise CheckpointValidationError(
            f"could not stat active checkpoint for shard {shard_key!r}: {exc}"
        ) from exc

    parquet_path, metadata_path = _ensure_safe_active_dir(
        active_target, shard_key=shard_key
    )

    try:
        meta = _read_persisted_metadata(metadata_path)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise CheckpointValidationError(
            f"checkpoint metadata for shard {shard_key!r} is malformed: {exc}"
        ) from exc

    # Strict validation: schema_version, identity, source_files sort
    # + relative-posix paths + lowercase SHA-256, segmentation report.
    expected_input_root = str(Path(input_root).expanduser().resolve(strict=False))
    # Patch expected identity values into a synthetic copy so we don't
    # mutate the persisted payload.
    candidate = dict(meta)
    candidate["shard_key"] = shard_key
    candidate["input_dataset_revision"] = input_dataset_revision
    candidate["pipeline_version"] = pipeline_version
    candidate["source_commit"] = source_commit
    candidate["model_name"] = model_name
    candidate["batch_size"] = int(batch_size)
    candidate["input_root"] = expected_input_root
    validate_checkpoint_metadata(
        candidate,
        shard_key=shard_key,
        expect_active=True,
    )

    # Identity-value match (strict validator confirms types but does
    # not enforce a value match — that is checked here against the
    # current invocation's arguments).
    identity_checks = (
        ("input_dataset_revision", input_dataset_revision),
        ("pipeline_version", pipeline_version),
        ("source_commit", source_commit),
        ("model_name", model_name),
        ("batch_size", int(batch_size)),
        ("input_root", expected_input_root),
    )
    for field, expected_value in identity_checks:
        actual_value = meta.get(field)
        if actual_value != expected_value:
            raise CheckpointValidationError(
                f"checkpoint identity mismatch on {field!r}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )

    actual_sha = sha256_file(parquet_path)
    if str(meta.get("segmented_table_sha256", "")).lower() != actual_sha:
        raise CheckpointValidationError(
            f"checkpoint SHA-256 mismatch for shard {shard_key!r}: got {actual_sha!r}"
        )

    table = pq.read_table(parquet_path)
    if not table.schema.equals(SEGMENTED_SENTENCES_SCHEMA):
        raise CheckpointValidationError(
            f"checkpoint for shard {shard_key!r} has wrong schema"
        )

    # Source-file manifest binding: skip if not provided.
    if current_manifest is not None:
        recorded = meta.get("source_files", [])
        try:
            recorded_entries = sorted(
                [
                    SourceFileEntry(
                        path=str(e["path"]),
                        size=int(e["size"]),
                        sha256=str(e["sha256"]).lower(),
                    )
                    for e in recorded
                ],
                key=lambda x: x.path,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointValidationError(
                f"checkpoint source_files field is malformed: {exc}"
            ) from exc
        if recorded_entries != current_manifest:
            raise CheckpointValidationError(
                f"checkpoint source_files mismatch for shard {shard_key!r}"
            )

    report = _segmentation_report_from_dict(meta["segmentation_report"])
    return table, report, meta


def write_heartbeat(
    work_dir: Path,
    *,
    stage: str,
    total_shards: int,
    completed_shards: int,
    current_shard_key: str | None,
    retained_sentence_occurrence_count: int,
    dropped_empty_raw_count: int,
    dropped_empty_normalized_count: int,
    elapsed_seconds: float,
    input_dataset_revision: str,
    source_commit: str,
) -> Path:
    """Atomically write the heartbeat JSON. Raises on ``OSError``; the
    caller is responsible for preserving any successfully published
    checkpoint bytes regardless of whether this raises.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(work_dir, _DIR_MODE)
    payload = {
        "stage": stage,
        "total_shards": int(total_shards),
        "completed_shards": int(completed_shards),
        "current_shard_key": current_shard_key,
        "retained_sentence_occurrence_count": int(retained_sentence_occurrence_count),
        "dropped_empty_raw_count": int(dropped_empty_raw_count),
        "dropped_empty_normalized_count": int(dropped_empty_normalized_count),
        "elapsed_seconds": float(elapsed_seconds),
        "input_dataset_revision": input_dataset_revision,
        "source_commit": source_commit,
    }
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    path = work_dir / HEARTBEAT_NAME
    _atomic_write_bytes(path, data)
    return path
