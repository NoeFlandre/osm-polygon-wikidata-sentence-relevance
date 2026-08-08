"""Crash-safe intra-shard segmentation progress.

Partial batches are local recovery state, not published checkpoints.  Each
batch is written and hashed before the small progress manifest advances, so a
walltime interruption can resume at the next section without repeating prior
segmenter calls.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.contracts.schemas import SEGMENTED_SENTENCES_SCHEMA
from osm_polygon_sentence_relevance.output.checksum import sha256_file
from osm_polygon_sentence_relevance.sentences.segmentation import SegmentationReport
from osm_polygon_sentence_relevance.sentences.table import SegmentedBatch

from .common import (
    _DIR_MODE,
    _FILE_MODE,
    CheckpointValidationError,
    _valid_shard_key,
)
from .inventory import SourceFileEntry
from .io import _atomic_write_bytes, _atomic_write_parquet, _fsync_dir_strict
from .validation import _segmentation_report_from_dict, _segmentation_report_to_dict

PARTIAL_DIRNAME = "partial"
PARTIAL_PROGRESS_NAME = "progress.json"
_PARTIAL_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PartialBatch:
    """One durably written section batch."""

    start_index: int
    end_index: int
    filename: str
    sha256: str
    rows: int
    report: SegmentationReport


@dataclass(frozen=True, slots=True)
class PartialShardState:
    """Validated intra-shard progress and its referenced batch files."""

    directory: Path
    shard_key: str
    source_commit: str
    input_dataset_revision: str
    pipeline_version: str
    model_name: str
    batch_size: int
    input_root: str
    source_files: tuple[SourceFileEntry, ...]
    total_sections: int
    next_section_index: int
    batches: tuple[PartialBatch, ...]


def partial_shard_path(work_dir: Path, shard_key: str) -> Path:
    """Return the guarded local directory for one partial shard."""
    if not _valid_shard_key(shard_key):
        raise CheckpointValidationError(f"invalid shard_key: {shard_key!r}")
    return work_dir / "shards" / PARTIAL_DIRNAME / shard_key


def load_partial_state(
    work_dir: Path,
    *,
    shard_key: str,
    source_commit: str,
    input_dataset_revision: str,
    pipeline_version: str,
    model_name: str,
    batch_size: int,
    input_root: Path,
    source_files: list[SourceFileEntry],
    total_sections: int,
) -> PartialShardState | None:
    """Load and strictly validate partial progress, if present."""
    directory = partial_shard_path(work_dir, shard_key)
    if not directory.exists():
        return None
    _ensure_directory(directory)
    progress_path = directory / PARTIAL_PROGRESS_NAME
    _ensure_regular(progress_path, _FILE_MODE)
    try:
        payload = json.loads(progress_path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(
            f"partial progress for {shard_key!r} is malformed"
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointValidationError("partial progress must be a JSON object")

    expected = {
        "schema_version": _PARTIAL_SCHEMA_VERSION,
        "shard_key": shard_key,
        "source_commit": source_commit,
        "input_dataset_revision": input_dataset_revision,
        "pipeline_version": pipeline_version,
        "model_name": model_name,
        "batch_size": int(batch_size),
        "input_root": str(Path(input_root).expanduser().resolve(strict=False)),
        "total_sections": int(total_sections),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise CheckpointValidationError(
                f"partial progress {key!r} mismatch for {shard_key!r}"
            )
    encoded_sources = [entry.to_dict() for entry in source_files]
    if payload.get("source_files") != encoded_sources:
        raise CheckpointValidationError(
            f"partial progress source manifest mismatch for {shard_key!r}"
        )

    batches = _parse_batches(payload.get("batches"), shard_key)
    next_index = payload.get("next_section_index")
    if isinstance(next_index, bool) or not isinstance(next_index, int):
        raise CheckpointValidationError("partial next_section_index must be an integer")
    _validate_batch_sequence(batches, next_index, total_sections)

    _remove_interrupted_atomic_temporary_files(directory)
    expected_names = {PARTIAL_PROGRESS_NAME} | {batch.filename for batch in batches}
    names = {path.name for path in directory.iterdir()}
    if names != expected_names:
        raise CheckpointValidationError(
            f"partial directory has unexpected entries: {sorted(names)}"
        )
    for batch in batches:
        path = directory / batch.filename
        _ensure_regular(path, _FILE_MODE)
        try:
            if sha256_file(path) != batch.sha256:
                raise CheckpointValidationError(
                    f"partial batch hash mismatch: {batch.filename}"
                )
            table = pq.read_table(path)
        except CheckpointValidationError:
            raise
        except Exception as exc:
            raise CheckpointValidationError(
                f"partial batch cannot be read: {batch.filename}"
            ) from exc
        if not table.schema.equals(SEGMENTED_SENTENCES_SCHEMA):
            raise CheckpointValidationError(
                f"partial batch has wrong schema: {batch.filename}"
            )
        if table.num_rows != batch.rows:
            raise CheckpointValidationError(
                f"partial batch row count mismatch: {batch.filename}"
            )

    return PartialShardState(
        directory=directory,
        shard_key=shard_key,
        source_commit=source_commit,
        input_dataset_revision=input_dataset_revision,
        pipeline_version=pipeline_version,
        model_name=model_name,
        batch_size=int(batch_size),
        input_root=expected["input_root"],
        source_files=tuple(source_files),
        total_sections=total_sections,
        next_section_index=next_index,
        batches=tuple(batches),
    )


def create_partial_state(
    work_dir: Path,
    *,
    shard_key: str,
    source_commit: str,
    input_dataset_revision: str,
    pipeline_version: str,
    model_name: str,
    batch_size: int,
    input_root: Path,
    source_files: list[SourceFileEntry],
    total_sections: int,
) -> PartialShardState:
    """Create an empty progress manifest after validating no stale state exists."""
    directory = partial_shard_path(work_dir, shard_key)
    if directory.exists():
        raise CheckpointValidationError(
            f"partial state already exists for {shard_key!r}; load it first"
        )
    parent = directory.parent
    if parent.exists():
        _ensure_directory(parent)
    else:
        parent.mkdir(parents=True, mode=_DIR_MODE)
        os.chmod(parent, _DIR_MODE)
    directory.mkdir(mode=_DIR_MODE)
    os.chmod(directory, _DIR_MODE)
    state = PartialShardState(
        directory=directory,
        shard_key=shard_key,
        source_commit=source_commit,
        input_dataset_revision=input_dataset_revision,
        pipeline_version=pipeline_version,
        model_name=model_name,
        batch_size=int(batch_size),
        input_root=str(Path(input_root).expanduser().resolve(strict=False)),
        source_files=tuple(source_files),
        total_sections=int(total_sections),
        next_section_index=0,
        batches=(),
    )
    _write_progress(state)
    return state


def append_partial_batch(
    state: PartialShardState, batch: SegmentedBatch
) -> PartialShardState:
    """Atomically persist one next batch, then advance progress."""
    if batch.start_index != state.next_section_index:
        raise CheckpointValidationError("partial batch is not the next expected batch")
    if not batch.table.schema.equals(SEGMENTED_SENTENCES_SCHEMA):
        raise CheckpointValidationError("partial batch has wrong schema")
    if batch.end_index <= batch.start_index or batch.end_index > state.total_sections:
        raise CheckpointValidationError("partial batch indexes are out of bounds")
    filename = f"batch-{batch.start_index:09d}-{batch.end_index:09d}.parquet"
    path = state.directory / filename
    if path.exists():
        raise CheckpointValidationError(f"partial batch already exists: {filename}")
    _atomic_write_parquet(batch.table, path)
    digest = sha256_file(path)
    item = PartialBatch(
        start_index=batch.start_index,
        end_index=batch.end_index,
        filename=filename,
        sha256=digest,
        rows=batch.table.num_rows,
        report=batch.report,
    )
    updated = PartialShardState(
        directory=state.directory,
        shard_key=state.shard_key,
        source_commit=state.source_commit,
        input_dataset_revision=state.input_dataset_revision,
        pipeline_version=state.pipeline_version,
        model_name=state.model_name,
        batch_size=state.batch_size,
        input_root=state.input_root,
        source_files=state.source_files,
        total_sections=state.total_sections,
        next_section_index=batch.end_index,
        batches=(*state.batches, item),
    )
    try:
        _write_progress(updated)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return updated


def read_partial_table(state: PartialShardState) -> pa.Table:
    """Read validated partial batches in deterministic order."""
    tables = [
        pq.read_table(state.directory / batch.filename) for batch in state.batches
    ]
    if not tables:
        return SEGMENTED_SENTENCES_SCHEMA.empty_table()
    return pa.concat_tables(tables)


def merge_partial_reports(state: PartialShardState) -> SegmentationReport:
    """Aggregate reports from validated partial batches."""
    reports = [batch.report for batch in state.batches]
    return SegmentationReport(
        input_section_occurrence_count=sum(
            r.input_section_occurrence_count for r in reports
        ),
        emitted_segment_count=sum(r.emitted_segment_count for r in reports),
        retained_sentence_occurrence_count=sum(
            r.retained_sentence_occurrence_count for r in reports
        ),
        dropped_empty_raw_count=sum(r.dropped_empty_raw_count for r in reports),
        dropped_empty_normalized_count=sum(
            r.dropped_empty_normalized_count for r in reports
        ),
        wikipedia_sentence_occurrence_count=sum(
            r.wikipedia_sentence_occurrence_count for r in reports
        ),
        wikivoyage_sentence_occurrence_count=sum(
            r.wikivoyage_sentence_occurrence_count for r in reports
        ),
    )


def discard_partial_state(work_dir: Path, shard_key: str) -> None:
    """Remove a completed partial shard after its active checkpoint is durable."""
    directory = partial_shard_path(work_dir, shard_key)
    if not directory.exists():
        return
    _ensure_directory(directory)
    shutil.rmtree(directory)
    _fsync_dir_strict(directory.parent)


def _write_progress(state: PartialShardState) -> None:
    payload = {
        "schema_version": _PARTIAL_SCHEMA_VERSION,
        "shard_key": state.shard_key,
        "source_commit": state.source_commit,
        "input_dataset_revision": state.input_dataset_revision,
        "pipeline_version": state.pipeline_version,
        "model_name": state.model_name,
        "batch_size": state.batch_size,
        "input_root": state.input_root,
        "source_files": [entry.to_dict() for entry in state.source_files],
        "total_sections": state.total_sections,
        "next_section_index": state.next_section_index,
        "batches": [
            {
                "start_index": batch.start_index,
                "end_index": batch.end_index,
                "filename": batch.filename,
                "sha256": batch.sha256,
                "rows": batch.rows,
                "report": _segmentation_report_to_dict(batch.report),
            }
            for batch in state.batches
        ],
    }
    _atomic_write_bytes(
        state.directory / PARTIAL_PROGRESS_NAME,
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )


def _parse_batches(value: object, shard_key: str) -> list[PartialBatch]:
    if not isinstance(value, list):
        raise CheckpointValidationError(
            f"partial batches must be a list for {shard_key!r}"
        )
    result: list[PartialBatch] = []
    for item in value:
        if not isinstance(item, dict):
            raise CheckpointValidationError("partial batch entry must be an object")
        filename = item.get("filename")
        if (
            not isinstance(filename, str)
            or not filename.startswith("batch-")
            or not filename.endswith(".parquet")
            or "/" in filename
            or ".." in filename
        ):
            raise CheckpointValidationError("partial batch filename is unsafe")
        start = item.get("start_index")
        end = item.get("end_index")
        rows = item.get("rows")
        if any(
            isinstance(v, bool) or not isinstance(v, int) for v in (start, end, rows)
        ):
            raise CheckpointValidationError("partial batch indexes must be integers")
        start = cast(int, start)
        end = cast(int, end)
        rows = cast(int, rows)
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise CheckpointValidationError("partial batch hash is invalid")
        report_data = item.get("report")
        if not isinstance(report_data, dict):
            raise CheckpointValidationError("partial batch report is invalid")
        try:
            report = _segmentation_report_from_dict(cast(dict[str, Any], report_data))
        except Exception as exc:
            raise CheckpointValidationError("partial batch report is invalid") from exc
        result.append(PartialBatch(start, end, filename, digest, rows, report))
    return result


def _validate_batch_sequence(
    batches: list[PartialBatch], next_index: int, total_sections: int
) -> None:
    if next_index < 0 or next_index > total_sections:
        raise CheckpointValidationError("partial next index is out of bounds")
    expected = 0
    for batch in batches:
        if batch.start_index != expected or batch.end_index <= batch.start_index:
            raise CheckpointValidationError("partial batches are not contiguous")
        if batch.rows < 0:
            raise CheckpointValidationError("partial batch row count is negative")
        expected = batch.end_index
    if expected != next_index:
        raise CheckpointValidationError("partial next index does not match batches")


def _ensure_directory(path: Path) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise CheckpointValidationError(
            f"partial directory is inaccessible: {path}"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CheckpointValidationError(f"partial path is not a directory: {path}")
    if info.st_mode & 0o777 != _DIR_MODE:
        raise CheckpointValidationError(f"partial directory has unsafe mode: {path}")


def _ensure_regular(path: Path, expected_mode: int | None) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise CheckpointValidationError(
            f"partial file is inaccessible: {path}"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CheckpointValidationError(f"partial file is not regular: {path}")
    if expected_mode is not None and info.st_mode & 0o777 != expected_mode:
        raise CheckpointValidationError(f"partial file has unsafe mode: {path}")


def _remove_interrupted_atomic_temporary_files(directory: Path) -> None:
    """Remove only our incomplete atomic-write files after a hard stop.

    A scheduler can kill a process between ``mkstemp`` and ``os.replace``.
    The durable manifest and batch files remain authoritative, so these
    temporary files are safe to discard before strict directory validation.
    """
    removed = False
    for path in directory.iterdir():
        name = path.name
        is_progress_temp = name.startswith(
            f".{PARTIAL_PROGRESS_NAME}."
        ) and name.endswith(".tmp")
        is_batch_temp = (
            name.startswith(".batch-") and ".parquet." in name and name.endswith(".tmp")
        )
        if not (is_progress_temp or is_batch_temp):
            continue
        # An interrupted temp file is disposable.  Validate its type and
        # reject symlinks, but do not let a stale pre-crash mode block recovery.
        _ensure_regular(path, None)
        path.unlink()
        removed = True
    if removed:
        _fsync_dir_strict(directory)


__all__ = [
    "PARTIAL_DIRNAME",
    "PARTIAL_PROGRESS_NAME",
    "PartialBatch",
    "PartialShardState",
    "append_partial_batch",
    "create_partial_state",
    "discard_partial_state",
    "load_partial_state",
    "merge_partial_reports",
    "partial_shard_path",
    "read_partial_table",
]
