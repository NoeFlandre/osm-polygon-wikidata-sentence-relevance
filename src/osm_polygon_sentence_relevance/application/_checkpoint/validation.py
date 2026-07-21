"""Checkpoint metadata, schema, and filesystem validation."""

from __future__ import annotations

import os
import posixpath
import re
import stat
from pathlib import Path
from typing import Any

from osm_polygon_sentence_relevance.sentences.segmentation import SegmentationReport

from .common import (
    _DIR_MODE,
    _FILE_MODE,
    _LOWER_HEX_40,
    _LOWER_HEX_64,
    _METADATA_SCHEMA_VERSION,
    SHARD_METADATA_NAME,
    SHARD_PARQUET_NAME,
    SHARDS_ACTIVE_DIRNAME,
    CheckpointValidationError,
    _valid_shard_key,
)
from .inventory import SourceFileEntry


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
