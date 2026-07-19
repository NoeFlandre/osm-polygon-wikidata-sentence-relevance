"""Phase 9L-A Amendment: quarantine-in-place checkpoint + heartbeat layer.

This module provides the smallest possible restartability layer on top
of the existing sorted-shard loop in ``run_pipeline``. It is a
**single-writer** layer: at most one ``run_pipeline`` invocation may
own a given ``work_dir`` at a time.

The contract is "never lose previously completed work":

* A non-blocking exclusive flock on ``${work_dir}/shards/.lock`` is
  acquired before any side-effecting I/O (discovery, hashing, joins,
  model use). The lock is always released through ``finally``; the
  lock file is **never** unlinked by the lock holder (so a stale lock
  is detectable and another process cannot accidentally remove
  someone else's evidence).
* Publishing a checkpoint is a whole-directory ``os.rename`` from a
  unique staging directory to the per-shard ``active/`` slot. On any
  failure mid-publish, the staging directory is preserved as evidence
  and the ``active/`` slot is **never** modified in place.
* An invalid, mismatched or stale checkpoint is **quarantined**: the
  entire active directory is renamed (same-filesystem ``os.rename``
  only, no fallback) into ``${WORK_DIR}/shards/quarantine/`` with a
  UUID-suffixed unique name. The original bytes are preserved exactly.
  If the rename fails (e.g. cross-filesystem ``EXDEV``), the call
  raises and the active directory is left **untouched**.
* Re-segmentation only happens after quarantine succeeds.
* Each checkpoint binds to actual source-file bytes via a
  ``RegionShardSet``-derived manifest. Source files are SHA-256
  fingerprinted once at inventory construction and re-fingerprinted
  once immediately before publication; any drift aborts the run
  before the rename. Each shard manifest is computed **once** at
  inventory construction and re-verified **once** immediately before
  publication; the verified manifest is what is recorded.
* A run-level inventory (``shards/inventory.json``) snapshots the
  discovered shard keys and a single canonical manifest per shard.
  Inventory reconciliation is **per shard**: added shards process
  alone, removed shards quarantine only their own orphaned checkpoint,
  changed shards quarantine only their own active directory,
  unchanged shards reuse the cached checkpoint. Inventory is versioned
  (``schema_version >= 2``); malformed prior inventory is atomically
  moved into ``shards/quarantine/inventory/<utcts>.<hex8>`` so its
  bytes are preserved exactly.
* Heartbeat failures propagate visibly. They never undo a successfully
  published checkpoint.

The pipeline uses ``fsync`` on the parent directory after every rename
to ensure crash-consistent directory entry persistence. ``fsync`` and
``chmod`` failures **propagate**: there is no ``suppress`` of
durability failures on publication paths.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import os
import posixpath
import re
import stat
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.contracts.errors import CheckpointError
from osm_polygon_sentence_relevance.contracts.schemas import (
    SEGMENTED_SENTENCES_SCHEMA,
)
from osm_polygon_sentence_relevance.ingestion.discovery import RegionShardSet
from osm_polygon_sentence_relevance.output.checksum import sha256_file
from osm_polygon_sentence_relevance.sentences.segmentation import (
    SegmentationReport,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Name of the segmented table file inside one checkpoint directory.
SHARD_PARQUET_NAME = "segmented.parquet"

#: Name of the metadata file inside one checkpoint directory.
SHARD_METADATA_NAME = "metadata.json"

#: Directory under ``${WORK_DIR}/shards`` holding live checkpoints.
SHARDS_ACTIVE_DIRNAME = "active"

#: Directory under ``${WORK_DIR}/shards`` holding quarantined checkpoints.
SHARDS_QUARANTINE_DIRNAME = "quarantine"

#: Prefix used for unique staging directories during publication.
SHARDS_STAGING_PREFIX = ".staging."

#: Name of the heartbeat file at the root of the work directory.
HEARTBEAT_NAME = "heartbeat.json"

#: File name of the work-dir exclusive lock file.
WORK_DIR_LOCK_NAME = ".lock"

#: Sub-directory of ``shards/quarantine`` that preserves malformed
#: ``inventory.json`` files. The directory name is fixed so the safe
#: loader can sweep known-bad inventory quickly.
INVENTORY_QUARANTINE_DIR = "inventory"

#: Mode for all checkpoint directories.
_DIR_MODE = 0o700

#: Mode for all checkpoint files.
_FILE_MODE = 0o600

#: Current metadata schema version. Bumped from 1 to 2 in Phase 9L-A
#: Amendment to introduce source-file manifests and refined identity
#: normalization. Older versions are quarantined, not silently loaded.
_METADATA_SCHEMA_VERSION = 2

#: Regex for an explicit 40-char lowercase hex source commit.
_LOWER_HEX_40 = re.compile(r"^[0-9a-f]{40}$")

#: Regex for a lowercase 64-hex SHA-256 digest.
_LOWER_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CheckpointValidationError(CheckpointError):
    """Raised when a checkpoint on disk is malformed, mismatched, or partial."""


class CheckpointPublicationError(CheckpointError):
    """Raised when a publish attempt fails. Active bytes are untouched."""


# ---------------------------------------------------------------------------
# Single-writer work-dir lock
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkDirLock:
    """An open file descriptor on the work-dir lock file. The lock is
    held until :func:`release_work_dir_lock` is called; the file is
    not unlinked by the holder."""

    fd: int
    path: Path


def acquire_work_dir_lock(work_dir: Path) -> WorkDirLock:
    """Acquire a non-blocking exclusive ``flock`` on
    ``${work_dir}/shards/.lock``.

    The lock file is hardened before flock is taken:

    * it is opened with ``O_NOFOLLOW`` so a symlink at ``.lock`` can
      never be followed;
    * the ``fstat`` of the resulting descriptor must describe a
      regular file owned by the current user with mode ``0o600``;
    * any permissive mode or non-regular entry is rejected with
      :class:`CheckpointValidationError`.

    Raises :class:`CheckpointValidationError` if the lock is already
    held by another process or the lock file fails hardening. On
    POSIX systems the lock is held until the file descriptor is
    closed; this function never unlinks the lock file.
    """
    work_dir = work_dir.expanduser().resolve(strict=False)
    shards_root = work_dir / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):  # pragma: no cover (read-only work_dir)
        os.chmod(shards_root, _DIR_MODE)
    lock_path = shards_root / WORK_DIR_LOCK_NAME
    open_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(lock_path), open_flags, _FILE_MODE)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise CheckpointValidationError(
                f"work_dir lock file {lock_path} is a symlink; refusing"
            ) from exc
        raise
    try:
        st = os.fstat(fd)
    except OSError:
        os.close(fd)
        raise
    mode_bits = st.st_mode & 0o777
    if (
        not stat.S_ISREG(st.st_mode)
        or st.st_uid != os.getuid()
        or mode_bits != _FILE_MODE
    ):
        os.close(fd)
        raise CheckpointValidationError(
            f"work_dir lock file {lock_path} has unexpected "
            f"type/owner/mode: regular={stat.S_ISREG(st.st_mode)} "
            f"uid={st.st_uid} (expected {os.getuid()}) mode={mode_bits:o} "
            f"(expected {_FILE_MODE:o})"
        )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            raise CheckpointValidationError(
                f"work_dir {work_dir} is already locked by another run"
            ) from exc
        raise
    return WorkDirLock(fd=fd, path=lock_path)


def release_work_dir_lock(ctx: WorkDirLock) -> None:
    """Release the lock acquired by :func:`acquire_work_dir_lock`.

    The lock file is **not** unlinked: ownership of the evidence file
    remains with the work_dir and a stale lock check can still detect
    a process that died while holding it.
    """
    try:
        fcntl.flock(ctx.fd, fcntl.LOCK_UN)
    finally:
        os.close(ctx.fd)


# ---------------------------------------------------------------------------
# Source-file manifest helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceFileEntry:
    """A single source-file binding inside a checkpoint manifest."""

    path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


def _relative_to_input_root(path: Path, input_root: Path) -> str:
    """Return ``path`` as a forward-slash relative path to ``input_root``."""
    rel = path.relative_to(input_root)
    return rel.as_posix()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def compute_shard_source_manifest(
    shard: RegionShardSet, *, input_root: Path
) -> list[SourceFileEntry]:
    """Return a sorted list of source-file bindings for ``shard``.

    The manifest is derived from the explicit per-shard file references
    held by the :class:`RegionShardSet` (four core Parquet files plus
    the optional Wikivoyage pair). Entries are sorted by the canonical
    forward-slash relative path.
    """
    input_root = input_root.expanduser().resolve(strict=False)
    candidates: list[Path] = [
        shard.polygons,
        shard.polygon_articles,
        shard.wikipedia_documents,
        shard.wikipedia_sections,
    ]
    if shard.wikivoyage_documents is not None:
        candidates.append(shard.wikivoyage_documents)
    if shard.wikivoyage_sections is not None:
        candidates.append(shard.wikivoyage_sections)

    entries: list[SourceFileEntry] = []
    for fpath in candidates:
        rel = _relative_to_input_root(fpath, input_root)
        entries.append(
            SourceFileEntry(
                path=rel,
                size=fpath.stat().st_size,
                sha256=_hash_file(fpath),
            )
        )
    entries.sort(key=lambda e: e.path)
    return entries


# ---------------------------------------------------------------------------
# Run inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunInventory:
    """A run-level snapshot of the discovered shard set and per-shard
    source manifests. Built once at the start of each ``run_pipeline``
    invocation and used to reconcile against any prior inventory.
    """

    schema_version: int
    discovered_at_unix: int
    input_dataset_revision: str
    source_commit: str
    pipeline_version: str
    model_name: str
    batch_size: int
    shards: dict[str, list[SourceFileEntry]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "discovered_at_unix": int(self.discovered_at_unix),
            "input_dataset_revision": self.input_dataset_revision,
            "source_commit": self.source_commit,
            "pipeline_version": self.pipeline_version,
            "model_name": self.model_name,
            "batch_size": int(self.batch_size),
            "shards": {
                shard_key: [e.to_dict() for e in entries]
                for shard_key, entries in sorted(self.shards.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunInventory:
        shards_data = data.get("shards", {})
        shards: dict[str, list[SourceFileEntry]] = {}
        for shard_key, entries in shards_data.items():
            shards[shard_key] = [
                SourceFileEntry(
                    path=str(e["path"]),
                    size=int(e["size"]),
                    sha256=str(e["sha256"]).lower(),
                )
                for e in entries
            ]
        return cls(
            schema_version=int(data.get("schema_version", -1)),
            discovered_at_unix=int(data.get("discovered_at_unix", 0)),
            input_dataset_revision=str(data.get("input_dataset_revision", "")),
            source_commit=str(data.get("source_commit", "")),
            pipeline_version=str(data.get("pipeline_version", "")),
            model_name=str(data.get("model_name", "")),
            batch_size=int(data.get("batch_size", 0)),
            shards=shards,
        )


def compute_run_inventory(
    shards: tuple[RegionShardSet, ...],
    *,
    input_root: Path,
    input_dataset_revision: str,
    pipeline_version: str,
    source_commit: str,
    model_name: str,
    batch_size: int,
) -> RunInventory:
    """Build a :class:`RunInventory` from the discovered shards.

    Each shard's source manifest is computed **once** here. Downstream
    code reuses this snapshot rather than hashing again.
    """
    inventory_shards: dict[str, list[SourceFileEntry]] = {}
    for shard in shards:
        inventory_shards[shard.shard_key] = compute_shard_source_manifest(
            shard, input_root=input_root
        )
    return RunInventory(
        schema_version=_METADATA_SCHEMA_VERSION,
        discovered_at_unix=int(time.time()),
        input_dataset_revision=input_dataset_revision,
        source_commit=source_commit,
        pipeline_version=pipeline_version,
        model_name=model_name,
        batch_size=int(batch_size),
        shards=inventory_shards,
    )


def _inventory_manifests_equal(
    prior: list[SourceFileEntry] | None, current: list[SourceFileEntry]
) -> bool:
    return prior == current


def reconcile_inventory(
    prior: dict[str, list[SourceFileEntry]] | None,
    current: dict[str, list[SourceFileEntry]],
) -> dict[str, set[str]]:
    """Return a ``{added, removed, changed, unchanged}`` decision for
    each shard key, comparing two per-shard manifests.

    * ``added`` — present only in ``current``.
    * ``removed`` — present only in ``prior``.
    * ``changed`` — present in both, but with a different manifest.
    * ``unchanged`` — present in both, with identical manifest.

    ``prior`` may be ``None`` for the first run; in that case every
    shard in ``current`` is reported as ``added``.
    """
    added: set[str] = set()
    removed: set[str] = set()
    changed: set[str] = set()
    unchanged: set[str] = set()

    prior_keys = set(prior.keys()) if prior else set()
    current_keys = set(current.keys())

    for key in sorted(current_keys - prior_keys):
        added.add(key)
    for key in sorted(prior_keys - current_keys):
        removed.add(key)
    for key in sorted(prior_keys & current_keys):
        prior_manifest = (prior or {}).get(key)
        current_manifest = current.get(key)
        if prior_manifest is None or current_manifest is None:
            # ``key`` is in both ``prior`` and ``current`` so neither
            # lookup can return ``None`` by construction. The runtime
            # assertion is here only to satisfy the type checker; on
            # the Python level this branch is dead.
            raise CheckpointValidationError(
                f"reconcile_inventory: missing manifest for shared key {key!r}"
            )
        if _inventory_manifests_equal(prior_manifest, current_manifest):
            unchanged.add(key)
        else:
            changed.add(key)
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": unchanged,
    }


# ---------------------------------------------------------------------------
# Atomic primitives (durability-failure-propagating)
# ---------------------------------------------------------------------------


def _fsync_dir_strict(path: Path) -> None:
    """``fsync`` a directory entry; ``OSError`` propagates."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically inside ``path.parent``.

    Durability invariants:

    * data is ``flush()``-ed and the file descriptor is ``fsync``-ed
      *before* the rename — a crash between the rename and the parent
      ``fsync`` cannot lose the file;
    * the file mode is set to ``0o600`` *before* the rename;
    * the parent directory is ``fsync``-ed after the rename so the
      directory entry is durable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_name, _FILE_MODE)
        os.replace(tmp_name, path)
        os.chmod(path, _FILE_MODE)
    except Exception:
        with contextlib.suppress(OSError):  # pragma: no cover (best-effort cleanup)
            os.unlink(tmp_name)
        raise
    _fsync_dir_strict(path.parent)


def _atomic_write_parquet(table: Any, path: Path) -> None:
    """Write a PyArrow ``table`` to ``path`` atomically inside ``path.parent``.

    Same durability invariants as :func:`_atomic_write_bytes`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        pq.write_table(table, tmp_name)
        os.chmod(tmp_name, _FILE_MODE)
        # Force a fsync on the file so that the bytes hit disk before
        # the rename; this matches the bytes-then-rename-then-fsync-dir
        # pattern documented above.
        with open(tmp_name, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
        os.chmod(path, _FILE_MODE)
    except Exception:
        with contextlib.suppress(OSError):  # pragma: no cover (best-effort cleanup)
            os.unlink(tmp_name)
        raise
    _fsync_dir_strict(path.parent)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _valid_shard_key(shard_key: str) -> bool:
    return (
        isinstance(shard_key, str)
        and bool(shard_key)
        and shard_key == shard_key.strip()
        and "/" not in shard_key
        and "\0" not in shard_key
        and ".." not in shard_key
    )


def _segmentation_report_to_dict(report: SegmentationReport) -> dict[str, int]:
    return {
        "input_section_occurrence_count": report.input_section_occurrence_count,
        "emitted_segment_count": report.emitted_segment_count,
        "retained_sentence_occurrence_count": report.retained_sentence_occurrence_count,
        "dropped_empty_raw_count": report.dropped_empty_raw_count,
        "dropped_empty_normalized_count": report.dropped_empty_normalized_count,
        "wikipedia_sentence_occurrence_count": (
            report.wikipedia_sentence_occurrence_count
        ),
        "wikivoyage_sentence_occurrence_count": (
            report.wikivoyage_sentence_occurrence_count
        ),
    }


def _segmentation_report_from_dict(data: dict[str, Any]) -> SegmentationReport:
    required = {
        "input_section_occurrence_count",
        "emitted_segment_count",
        "retained_sentence_occurrence_count",
        "dropped_empty_raw_count",
        "dropped_empty_normalized_count",
        "wikipedia_sentence_occurrence_count",
        "wikivoyage_sentence_occurrence_count",
    }
    if not isinstance(data, dict):
        raise CheckpointValidationError(
            "checkpoint metadata: segmentation_report must be a mapping"
        )
    missing = required - set(data)
    if missing:
        raise CheckpointValidationError(
            f"checkpoint metadata: missing report field(s): {sorted(missing)}"
        )
    try:
        return SegmentationReport(
            input_section_occurrence_count=int(data["input_section_occurrence_count"]),
            emitted_segment_count=int(data["emitted_segment_count"]),
            retained_sentence_occurrence_count=int(
                data["retained_sentence_occurrence_count"]
            ),
            dropped_empty_raw_count=int(data["dropped_empty_raw_count"]),
            dropped_empty_normalized_count=int(data["dropped_empty_normalized_count"]),
            wikipedia_sentence_occurrence_count=int(
                data["wikipedia_sentence_occurrence_count"]
            ),
            wikivoyage_sentence_occurrence_count=int(
                data["wikivoyage_sentence_occurrence_count"]
            ),
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(
            f"checkpoint metadata: segmentation_report has invalid field: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Strict metadata validator (single source of truth)
# ---------------------------------------------------------------------------


def validate_checkpoint_metadata(
    metadata: dict[str, Any],
    *,
    shard_key: str,
    expect_active: bool,
) -> dict[str, Any]:
    """Strict validator for a checkpoint metadata payload.

    The same validator is used by both :func:`publish_shard_checkpoint`
    (post-write, against staged contents) and :func:`load_shard_checkpoint`
    (post-read, before use). It enforces:

    * schema_version == 2
    * all required identity fields present and of the correct type
    * source_files is a sorted list of dicts with relative posix
      paths, integer sizes, and lowercase 64-hex SHA-256 digests
    * segmentation_report is a complete valid report
    * segmented_table_sha256 is a lowercase 64-hex string
    * completed_at_unix is an int
    """
    if not isinstance(metadata, dict):
        raise CheckpointValidationError(
            "checkpoint metadata: payload must be a mapping"
        )

    if int(metadata.get("schema_version", -1)) != _METADATA_SCHEMA_VERSION:
        raise CheckpointValidationError(
            f"checkpoint schema_version mismatch: expected "
            f"{_METADATA_SCHEMA_VERSION}, got {metadata.get('schema_version')!r}"
        )

    identity_fields = (
        "shard_key",
        "input_dataset_revision",
        "pipeline_version",
        "source_commit",
        "model_name",
        "batch_size",
        "input_root",
        "segmented_table_sha256",
        "completed_at_unix",
    )
    for field in identity_fields:
        if field not in metadata:
            raise CheckpointValidationError(
                f"checkpoint metadata: missing required field {field!r}"
            )

    str_fields = (
        "shard_key",
        "input_dataset_revision",
        "pipeline_version",
        "source_commit",
        "model_name",
        "input_root",
    )
    for field in str_fields:
        value = metadata[field]
        if not isinstance(value, str):
            raise CheckpointValidationError(
                f"checkpoint metadata: field {field!r} must be a string"
            )
        if field == "source_commit" and not _LOWER_HEX_40.match(value):
            raise CheckpointValidationError(
                "checkpoint metadata: source_commit must be lowercase 40-character hex"
            )
        if field == "shard_key":
            if not _valid_shard_key(value):
                raise CheckpointValidationError(
                    f"checkpoint metadata: invalid shard_key {value!r}"
                )
            if value != shard_key:
                raise CheckpointValidationError(
                    f"checkpoint metadata: shard_key mismatch: "
                    f"expected {shard_key!r}, got {value!r}"
                )
        if field == "input_dataset_revision" and not value.strip():
            raise CheckpointValidationError(
                "checkpoint metadata: input_dataset_revision must be non-blank"
            )

    batch_size = metadata["batch_size"]
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise CheckpointValidationError(
            "checkpoint metadata: batch_size must be a positive integer"
        )

    sha = metadata["segmented_table_sha256"]
    if not isinstance(sha, str) or not _LOWER_HEX_64.match(sha):
        raise CheckpointValidationError(
            "checkpoint metadata: segmented_table_sha256 must be lowercase "
            "64-character hex"
        )

    completed_at_unix = metadata["completed_at_unix"]
    if not isinstance(completed_at_unix, int):
        raise CheckpointValidationError(
            "checkpoint metadata: completed_at_unix must be an integer"
        )

    source_files = metadata.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise CheckpointValidationError(
            "checkpoint metadata: source_files must be a non-empty list"
        )
    entries: list[SourceFileEntry] = []
    for raw in source_files:
        if not isinstance(raw, dict):
            raise CheckpointValidationError(
                "checkpoint metadata: source_files entry must be a mapping"
            )
        try:
            path = raw["path"]
            size = raw["size"]
            digest = raw["sha256"]
        except KeyError as exc:
            raise CheckpointValidationError(
                f"checkpoint metadata: source_files entry missing key {exc.args[0]!r}"
            ) from exc
        if not isinstance(path, str):
            raise CheckpointValidationError(
                "checkpoint metadata: source_files entry path must be a string"
            )
        if path != path.strip() or path == "":
            raise CheckpointValidationError(
                "checkpoint metadata: source_files entry path must be a non-blank string"
            )
        # posix-relative check: must not be absolute, must not start
        # with a drive letter, must not contain ".." segments, and
        # ``posixpath`` must not normalize outside the relative shape.
        if posixpath.isabs(path) or re.match(r"^[A-Za-z]:", path):
            raise CheckpointValidationError(
                "checkpoint metadata: source_files entry path must be a "
                "relative POSIX path"
            )
        if path.startswith(("/", "\\")):
            raise CheckpointValidationError(
                "checkpoint metadata: source_files entry path must be a "
                "relative POSIX path"
            )
        normalized = posixpath.normpath(path)
        if normalized.startswith("..") or "/.." in ("/" + normalized):
            raise CheckpointValidationError(
                "checkpoint metadata: source_files entry path must not "
                "contain parent-directory segments"
            )
        if not isinstance(size, int) or size < 0:
            raise CheckpointValidationError(
                "checkpoint metadata: source_files entry size must be a non-negative integer"
            )
        if not isinstance(digest, str) or not _LOWER_HEX_64.match(digest):
            raise CheckpointValidationError(
                "checkpoint metadata: source_files entry sha256 must be "
                "lowercase 64-character hex"
            )
        entries.append(SourceFileEntry(path=path, size=size, sha256=digest))

    sorted_entries = sorted(entries, key=lambda e: e.path)
    if [e.path for e in sorted_entries] != [e.path for e in entries]:
        raise CheckpointValidationError(
            "checkpoint metadata: source_files must be sorted by path"
        )

    # Replace the source_files list with validated entries (so callers
    # that consume the metadata can rely on lowercase digests).
    metadata["source_files"] = [e.to_dict() for e in sorted_entries]

    report_dict = metadata.get("segmentation_report")
    if report_dict is None:
        raise CheckpointValidationError(
            "checkpoint metadata: segmentation_report is required"
        )
    _segmentation_report_from_dict(report_dict)

    return metadata


# ---------------------------------------------------------------------------
# Filesystem entry inspection (symlink / mode checks)
# ---------------------------------------------------------------------------


def _ensure_safe_active_dir(
    active_target: Path,
    *,
    shard_key: str,
) -> tuple[Path, Path]:
    """Inspect ``active_target`` and refuse symlinks, broken symlinks,
    non-regular-files, or unexpected directory modes.

    Returns the resolved ``(parquet_path, metadata_path)``.
    """
    # Use ``lstat`` so we don't follow symlinks.
    lst = os.lstat(active_target)
    if stat.S_ISLNK(lst.st_mode):
        raise CheckpointValidationError(
            f"active checkpoint for shard {shard_key!r} is a symlink "
            f"at {active_target}; refusing"
        )
    if not stat.S_ISDIR(lst.st_mode):
        raise CheckpointValidationError(
            f"active checkpoint for shard {shard_key!r} is not a regular directory "
            f"at {active_target}"
        )
    dir_mode = lst.st_mode & 0o777
    if dir_mode != _DIR_MODE:
        raise CheckpointValidationError(
            f"active checkpoint directory for shard {shard_key!r} "
            f"has mode {dir_mode:o}, expected {_DIR_MODE:o}"
        )

    entries = sorted(p.name for p in active_target.iterdir())
    expected = sorted([SHARD_PARQUET_NAME, SHARD_METADATA_NAME])
    if entries != expected:
        raise CheckpointValidationError(
            f"checkpoint directory {active_target} has unexpected entries: {entries}"
        )

    parquet_path = active_target / SHARD_PARQUET_NAME
    metadata_path = active_target / SHARD_METADATA_NAME
    for p in (parquet_path, metadata_path):
        lst = os.lstat(p)
        if stat.S_ISLNK(lst.st_mode):
            raise CheckpointValidationError(
                f"checkpoint entry {p} is a symlink; refusing"
            )
        if not stat.S_ISREG(lst.st_mode):
            raise CheckpointValidationError(
                f"checkpoint entry {p} is not a regular file"
            )
        mode = lst.st_mode & 0o777
        if mode != _FILE_MODE:
            raise CheckpointValidationError(
                f"checkpoint entry {p} has mode {mode:o}, expected {_FILE_MODE:o}"
            )
    return parquet_path, metadata_path


def scan_active_directory(work_dir: Path) -> None:
    """Defensively scan ``${work_dir}/shards/active/``.

    Every entry is classified with ``os.lstat`` (never ``stat``):

    * regular directories with valid ``shard_key`` names are left in
      place — the per-shard load path will validate their contents;
    * files, broken symlinks, symlinks to files/directories, FIFO /
      socket / device nodes, directories with invalid shard keys, and
      any other unexpected entry type cause an immediate
      :class:`CheckpointValidationError`. The entries are never
      silently dropped or followed.
    """
    active_root = work_dir / "shards" / SHARDS_ACTIVE_DIRNAME
    if not active_root.exists():
        return
    for entry in active_root.iterdir():
        try:
            lst = os.lstat(entry)
        except OSError as exc:
            raise CheckpointValidationError(
                f"could not lstat active entry {entry}: {exc}"
            ) from exc
        if stat.S_ISLNK(lst.st_mode):
            raise CheckpointValidationError(f"unexpected symlink at {entry}; refusing")
        if not stat.S_ISDIR(lst.st_mode):
            raise CheckpointValidationError(
                f"unexpected non-directory entry at {entry} "
                f"(mode={oct(lst.st_mode)}); refusing"
            )
        if not _valid_shard_key(entry.name):
            raise CheckpointValidationError(
                f"active directory {entry} has invalid shard_key "
                f"{entry.name!r}; refusing"
            )
        mode = lst.st_mode & 0o777
        if mode != _DIR_MODE:
            raise CheckpointValidationError(
                f"active directory {entry} has mode {mode:o}, expected {_DIR_MODE:o}"
            )


# ---------------------------------------------------------------------------
# Inventory write/load (versioned + safe loader)
# ---------------------------------------------------------------------------


def _inventory_path(work_dir: Path) -> Path:
    return work_dir / "shards" / "inventory.json"


def _inventory_quarantine_path(work_dir: Path) -> Path:
    base = work_dir / "shards" / SHARDS_QUARANTINE_DIRNAME / INVENTORY_QUARANTINE_DIR
    base.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(base, _DIR_MODE)
    return base


def write_run_inventory(work_dir: Path, inventory: RunInventory) -> Path:
    """Atomically write the run-level inventory under ``work_dir``."""
    shards_root = work_dir / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    os.chmod(shards_root, _DIR_MODE)
    data = json.dumps(
        inventory.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    target = _inventory_path(work_dir)
    _atomic_write_bytes(target, data)
    return target


def _parse_inventory_payload(raw: bytes, *, source_path: Path) -> RunInventory:
    """Parse ``raw`` bytes as a JSON inventory payload."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(
            f"inventory at {source_path} is malformed JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise CheckpointValidationError(
            f"inventory at {source_path} is malformed: payload must be a mapping"
        )
    schema_version = int(data.get("schema_version", -1))
    if schema_version != _METADATA_SCHEMA_VERSION:
        raise CheckpointValidationError(
            f"inventory at {source_path} has schema_version={schema_version!r}, "
            f"expected {_METADATA_SCHEMA_VERSION}"
        )
    required = (
        "input_dataset_revision",
        "source_commit",
        "pipeline_version",
        "model_name",
        "batch_size",
        "discovered_at_unix",
        "shards",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise CheckpointValidationError(
            f"inventory at {source_path} is malformed: missing fields {missing}"
        )
    if not isinstance(data["shards"], dict):
        raise CheckpointValidationError(
            f"inventory at {source_path} is malformed: shards must be a mapping"
        )
    for key, entries in data["shards"].items():
        if not isinstance(key, str) or not _valid_shard_key(key):
            raise CheckpointValidationError(
                f"inventory at {source_path} is malformed: invalid shard_key {key!r}"
            )
        if not isinstance(entries, list):
            raise CheckpointValidationError(
                f"inventory at {source_path} is malformed: shards[{key!r}] "
                f"must be a list"
            )
    return RunInventory.from_dict(data)


def load_run_inventory(work_dir: Path) -> RunInventory | None:
    """Load the persisted inventory at ``work_dir`` if present.

    Returns ``None`` when no inventory exists. Malformed JSON, wrong
    schema, missing identity fields, or wrong shard-key shape raise
    :class:`CheckpointValidationError`. The corrupt file is **not**
    modified by this call.
    """
    path = _inventory_path(work_dir)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CheckpointValidationError(
            f"inventory at {path} could not be read: {exc}"
        ) from exc
    return _parse_inventory_payload(raw, source_path=path)


def load_run_inventory_quarantine_first(work_dir: Path) -> RunInventory | None:
    """Load the persisted inventory at ``work_dir``.

    If the inventory exists and parses cleanly, return it. If it is
    malformed, atomically move the bytes into
    ``shards/quarantine/inventory/`` (preserving them as evidence)
    and return ``None``. This lets the calling pipeline fall back to
    orphan-active recovery and complete the run in the same
    invocation, without a manual second run.

    If the rename itself fails (e.g. cross-filesystem ``EXDEV``), the
    file is left untouched and the underlying parse error is
    re-raised so the caller can decide what to do.
    """
    path = _inventory_path(work_dir)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CheckpointValidationError(
            f"inventory at {path} could not be read: {exc}"
        ) from exc
    try:
        return _parse_inventory_payload(raw, source_path=path)
    except CheckpointValidationError as parse_exc:
        # Move the malformed bytes aside so a fresh
        # ``write_run_inventory`` doesn't silently overwrite evidence.
        qdir = _inventory_quarantine_path(work_dir)
        ts = int(time.time())
        target = qdir / f"inventory.{ts}.{uuid.uuid4().hex[:8]}"
        try:
            os.rename(path, target)
        except OSError:
            # Could not move it (e.g. EXDEV). Re-raise the original
            # parse error so the caller sees the malformed state.
            raise parse_exc from None
        _fsync_dir_strict(qdir)
        # Quarantined successfully; signal "no prior inventory" to
        # the caller so it can continue via orphan-active recovery.
        return None


# ---------------------------------------------------------------------------
# Publication (whole-directory atomic rename)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Quarantine (same-filesystem rename only)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Load (read-only inspection; never mutates)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# work_dir validation
# ---------------------------------------------------------------------------


def validate_work_dir(work_dir: str | Path | None) -> Path | None:
    """Validate ``work_dir`` and return the canonical absolute path."""
    if work_dir is None:
        return None
    if not isinstance(work_dir, (str, Path)):
        raise ValueError("work_dir must be a string or Path when provided")
    if isinstance(work_dir, str):
        if not work_dir.strip():
            raise ValueError("work_dir must not be blank")
        candidate = Path(work_dir)
    else:
        candidate = work_dir
    abs_path = candidate.expanduser().resolve(strict=False)
    return abs_path


def validate_source_commit(value: str | None) -> str:
    """Validate that ``value`` is a lowercase 40-hex string.

    Required only when ``--work-dir`` is used. The CLI enforces this
    rule before ``run_pipeline`` is ever invoked.
    """
    if not isinstance(value, str) or not value:
        raise ValueError("source-commit must be a non-empty string")
    if not _LOWER_HEX_40.match(value):
        raise ValueError(
            "source-commit must be lowercase 40-character hex (a Git commit SHA)"
        )
    return value


__all__ = [
    "CheckpointPublicationError",
    "CheckpointValidationError",
    "HEARTBEAT_NAME",
    "INVENTORY_QUARANTINE_DIR",
    "RunInventory",
    "SHARDS_ACTIVE_DIRNAME",
    "SHARDS_QUARANTINE_DIRNAME",
    "SHARDS_STAGING_PREFIX",
    "SHARD_METADATA_NAME",
    "SHARD_PARQUET_NAME",
    "SourceFileEntry",
    "WorkDirLock",
    "WORK_DIR_LOCK_NAME",
    "acquire_work_dir_lock",
    "compute_run_inventory",
    "compute_shard_source_manifest",
    "load_run_inventory",
    "load_run_inventory_quarantine_first",
    "load_shard_checkpoint",
    "publish_shard_checkpoint",
    "quarantine_shard_checkpoint",
    "reconcile_inventory",
    "release_work_dir_lock",
    "validate_checkpoint_metadata",
    "validate_source_commit",
    "validate_work_dir",
    "write_heartbeat",
    "write_run_inventory",
]
