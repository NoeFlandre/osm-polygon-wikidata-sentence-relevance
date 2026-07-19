from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from osm_polygon_sentence_relevance.application import checkpoint as _checkpoint
from osm_polygon_sentence_relevance.application.checkpoint import (
    SHARDS_ACTIVE_DIRNAME,
    CheckpointValidationError,
    SourceFileEntry,
    WorkDirLock,
    _verify_pre_publish_manifest,
    acquire_work_dir_lock,
    compute_run_inventory,
    load_run_inventory_quarantine_first,
    load_shard_checkpoint,
    publish_shard_checkpoint,
    quarantine_shard_checkpoint,
    reconcile_inventory,
    release_work_dir_lock,
    scan_active_directory,
    write_run_inventory,
)
from osm_polygon_sentence_relevance.contracts.schemas import SEGMENTED_SENTENCES_SCHEMA
from osm_polygon_sentence_relevance.ingestion.discovery import discover_shards
from osm_polygon_sentence_relevance.joins import build_region_section_occurrences
from osm_polygon_sentence_relevance.output.exporter import (
    ExportResult,
    export_finalized_dataset,
)
from osm_polygon_sentence_relevance.sentences.finalization import (
    FinalizationReport,
    finalize_sentence_dataset,
)
from osm_polygon_sentence_relevance.sentences.segmentation import (
    SegmentationReport,
    SentenceSegmenter,
)
from osm_polygon_sentence_relevance.sentences.table import segment_joined_sections


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """The result of running the injection-based pipeline."""

    export_result: ExportResult
    processed_regions_count: int
    total_joined_section_occurrences: int
    segmentation_report: SegmentationReport
    finalization_report: FinalizationReport


def run_pipeline(
    input_root: str | Path,
    output_dir: str | Path,
    segmenter: SentenceSegmenter,
    *,
    input_dataset_revision: str,
    pipeline_version: str,
    batch_size: int = 128,
    overwrite: bool = False,
    input_dataset_id: str | None = None,
    work_dir: str | Path | None = None,
    source_commit: str | None = None,
    model_name: str | None = None,
) -> PipelineResult:
    """Orchestrate regional shard discovery, occurrences join, segmentation, finalization, and export.

    Parameters
    ----------
    work_dir : str | Path | None, optional
        Persistent work directory for shard-level checkpoints and a
        factual progress heartbeat. ``None`` (default) preserves the
        legacy behaviour: no checkpoint layer, no resume on interruption.
        When supplied, completed shards are persisted under
        ``${WORK_DIR}/shards/active/<shard_key>/``. Invalid or stale
        checkpoints are moved to ``${WORK_DIR}/shards/quarantine/`` with
        a UUID-suffixed unique name; their original bytes are preserved
        for inspection.

        The work directory is **single-writer**: a non-blocking
        exclusive lock on ``${WORK_DIR}/shards/.lock`` is acquired
        before any side-effecting I/O and released through a
        ``finally`` block. A second concurrent invocation is rejected
        with :class:`CheckpointValidationError`.

        Orphan-quarantine failures (e.g. cross-filesystem rename) and
        publication failures are **never** swallowed and never retried
        automatically; the run aborts with active and staging bytes
        left untouched.

    source_commit : str | None, optional
        40-character lowercase hex commit SHA binding the checkpoints
        to a specific code revision. Required when ``work_dir`` is set.
        The CLI rejects non-conforming values before this function is
        called.
    model_name : str | None, optional
        Model name recorded into each shard checkpoint. Required when
        ``work_dir`` is set.
    """
    # 1. Validate the segmenter and identity arguments before any I/O.
    if not isinstance(segmenter, SentenceSegmenter):
        raise TypeError("segmenter must implement the SentenceSegmenter protocol")

    in_path = Path(input_root).expanduser().resolve(strict=False)
    out_path = Path(output_dir).expanduser().resolve(strict=False)
    if in_path == out_path:
        raise ValueError("Input root and output directory cannot be the same path")
    if in_path in out_path.parents:
        raise ValueError(
            "Input root cannot be an ancestor of output directory (paths overlap)"
        )
    if out_path in in_path.parents:
        raise ValueError(
            "Output directory cannot be an ancestor of input root (paths overlap)"
        )

    work_path = _checkpoint.validate_work_dir(work_dir)
    if work_path is not None:
        if work_path == in_path:
            raise ValueError("work_dir and input root cannot be the same path")
        if work_path == out_path:
            raise ValueError("work_dir and output directory cannot be the same path")
        for ancestor, child, label in (
            (in_path, work_path, "input root"),
            (out_path, work_path, "output directory"),
            (work_path, in_path, "work_dir"),
            (work_path, out_path, "work_dir"),
        ):
            if ancestor in child.parents:
                raise ValueError(f"{label} paths overlap")

        if source_commit is not None:
            source_commit = _checkpoint.validate_source_commit(source_commit)
        else:
            raise ValueError("source_commit must be supplied when work_dir is set")
        if (
            model_name is None
            or not isinstance(model_name, str)
            or not model_name.strip()
        ):
            raise ValueError(
                "model_name must be a non-blank string when work_dir is set"
            )

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")

    if (
        not isinstance(input_dataset_revision, str)
        or not input_dataset_revision.strip()
    ):
        raise ValueError("input_dataset_revision must be a non-blank string")

    if not isinstance(pipeline_version, str) or not pipeline_version.strip():
        raise ValueError("pipeline_version must be a non-blank string")

    if input_dataset_id is not None and (
        not isinstance(input_dataset_id, str) or not input_dataset_id.strip()
    ):
        raise ValueError("input_dataset_id must be a non-blank string when provided")
    if (
        input_dataset_id is not None
        and isinstance(input_dataset_id, str)
        and input_dataset_id != input_dataset_id.strip()
    ):
        raise ValueError(
            "input_dataset_id has surrounding whitespace; surrounding "
            "whitespace is rejected, not silently normalized"
        )

    # 2. Acquire the single-writer work-dir lock (if applicable).
    lock_ctx: WorkDirLock | None = None
    if work_path is not None:
        lock_ctx = acquire_work_dir_lock(work_path)

    try:
        return _run_pipeline_locked(
            in_path=in_path,
            out_path=out_path,
            work_path=work_path,
            segmenter=segmenter,
            input_dataset_revision=input_dataset_revision,
            pipeline_version=pipeline_version,
            batch_size=batch_size,
            input_dataset_id=input_dataset_id,
            source_commit=source_commit,
            model_name=model_name,
            overwrite=overwrite,
        )
    finally:
        if lock_ctx is not None:
            release_work_dir_lock(lock_ctx)


def _run_pipeline_locked(
    *,
    in_path: Path,
    out_path: Path,
    work_path: Path | None,
    segmenter: SentenceSegmenter,
    input_dataset_revision: str,
    pipeline_version: str,
    batch_size: int,
    input_dataset_id: str | None,
    source_commit: str | None,
    model_name: str | None,
    overwrite: bool,
) -> PipelineResult:
    """Pipeline body executed under the work-dir lock (if any)."""

    # 3. Discover shards (always; no model construction).
    shards = sorted(discover_shards(in_path), key=lambda s: s.shard_key)

    # 4. Build the current inventory ONCE (each shard manifest hashed
    #    exactly once here).
    current_inventory: _checkpoint.RunInventory | None = None
    if work_path is not None:
        current_inventory = compute_run_inventory(
            tuple(shards),
            input_root=in_path,
            input_dataset_revision=input_dataset_revision,
            pipeline_version=pipeline_version,
            source_commit=source_commit or "",
            model_name=model_name or "",
            batch_size=batch_size,
        )

    # 5. Reconcile inventory against the prior run, if any.
    prior_inventory = (
        load_run_inventory_quarantine_first(work_path)
        if work_path is not None
        else None
    )
    decisions: dict[str, set[str]] = {
        "added": set(),
        "removed": set(),
        "changed": set(),
        "unchanged": set(),
    }
    if current_inventory is not None:
        decisions = reconcile_inventory(
            prior_inventory.shards if prior_inventory is not None else None,
            current_inventory.shards,
        )

    # 5a. Defensive scan of the active directory. Any unexpected
    #     entry (file, broken symlink, symlink, non-directory, invalid
    #     shard_key, wrong mode) causes the run to abort visibly.
    if work_path is not None:
        scan_active_directory(work_path)

    # 5b. Recover from prior inventory loss: a prior run that crashed
    #     *after* publishing an active checkpoint but *before* writing
    #     ``inventory.json`` leaves behind an orphan active entry. If
    #     the entry's manifest matches the current run's manifest,
    #     treat it as ``unchanged`` (resume). Otherwise quarantine it
    #     so the new run can publish a fresh checkpoint. If
    #     quarantine itself fails (e.g. cross-filesystem), the run
    #     aborts with the orphan directory untouched.
    if work_path is not None and current_inventory is not None:
        active_root = work_path / "shards" / SHARDS_ACTIVE_DIRNAME
        if active_root.is_dir():
            for entry in sorted(active_root.iterdir()):
                if not entry.is_dir():
                    continue
                # Recovered keys are not part of the explicit decision
                # set; only inspect those that the reconciler marked
                # as added (i.e. no prior record).
                key = entry.name
                if key not in decisions["added"]:
                    continue
                # Try to load the active and compare its manifest to
                # the current manifest for the same shard.
                try:
                    _, _, meta = load_shard_checkpoint(
                        work_path,
                        key,
                        input_dataset_revision=input_dataset_revision,
                        pipeline_version=pipeline_version,
                        source_commit=source_commit or "",
                        model_name=model_name or "",
                        batch_size=batch_size,
                        input_root=in_path,
                        current_manifest=current_inventory.shards.get(key),
                    )
                    # Manifest matches: move from ``added`` to
                    # ``unchanged`` so the loop reuses the cache.
                    decisions["added"].discard(key)
                    decisions["unchanged"].add(key)
                except CheckpointValidationError:
                    quarantine_shard_checkpoint(
                        work_dir=work_path,
                        shard_key=key,
                        reason="orphan active from prior interrupted run",
                    )
                    # The entry is now gone; remove it from ``added``
                    # so the loop treats it as a clean add.
                    decisions["added"].discard(key)

    # 5b. Quarantine orphaned active checkpoints directly. Failure
    #     propagates: the caller will see an OSError (e.g. EXDEV) and
    #     the orphaned active directory is left untouched.
    for prior_key in sorted(decisions["removed"]):
        quarantine_shard_checkpoint(
            work_dir=work_path,  # type: ignore[arg-type]
            shard_key=prior_key,
            reason="removed from current input",
        )

    # 6. Process regions.
    processed_regions_count = len(shards)
    total_joined_section_occurrences = 0
    segmented_tables: list[Any] = []
    segmentation_reports: list[SegmentationReport] = []

    start_time = time.monotonic()
    completed_shards = 0

    if work_path is not None:
        _write_heartbeat_or_propagate(
            work_path,
            stage="processing",
            total_shards=processed_regions_count,
            completed_shards=0,
            current_shard_key=None,
            retained_sentence_occurrence_count=0,
            dropped_empty_raw_count=0,
            dropped_empty_normalized_count=0,
            elapsed_seconds=0.0,
            input_dataset_revision=input_dataset_revision,
            source_commit=source_commit or "",
        )

    for shard in shards:
        # The current manifest for this shard was computed once at
        # inventory construction; reuse it for the load check.
        current_manifest: list[SourceFileEntry] | None = (
            current_inventory.shards.get(shard.shard_key)
            if current_inventory is not None
            else None
        )

        # 6a. unchanged: load the active checkpoint. If strict load
        #     validation fails (corrupt metadata, identity mismatch,
        #     SHA mismatch, schema drift, wrong mode, symlink,
        #     non-regular file, etc.), atomically quarantine the
        #     active bytes and fall through to a fresh re-segment.
        #     If quarantine itself fails, the run aborts with the
        #     active bytes left untouched (no silent fallback).
        if work_path is not None and shard.shard_key in decisions["unchanged"]:
            try:
                res_table, res_report, _ = load_shard_checkpoint(
                    work_path,
                    shard.shard_key,
                    input_dataset_revision=input_dataset_revision,
                    pipeline_version=pipeline_version,
                    source_commit=source_commit or "",
                    model_name=model_name or "",
                    batch_size=batch_size,
                    input_root=in_path,
                    current_manifest=current_manifest,
                )
            except CheckpointValidationError:
                quarantine_shard_checkpoint(
                    work_dir=work_path,
                    shard_key=shard.shard_key,
                    reason="unchanged checkpoint failed strict validation",
                )
                # Active bytes have been moved aside; fall through to
                # the re-segment branch below.
                decisions["unchanged"].discard(shard.shard_key)
                decisions["added"].add(shard.shard_key)
            else:
                # Reuse the cached joined-occurrence total to
                # preserve uninterrupted-run accounting parity.
                cached_total = _cached_joined_total(res_report)
                total_joined_section_occurrences += cached_total
                segmented_tables.append(res_table)
                segmentation_reports.append(res_report)
                completed_shards += 1
                _write_progress_heartbeat(
                    work_path,
                    processed_regions_count=processed_regions_count,
                    completed_shards=completed_shards,
                    segmentation_reports=segmentation_reports,
                    start_time=start_time,
                    input_dataset_revision=input_dataset_revision,
                    source_commit=source_commit or "",
                )
                continue

        # 6b. changed: quarantine the existing active directory (no
        #     ``suppress``) and re-segment.
        if work_path is not None and shard.shard_key in decisions["changed"]:
            quarantine_shard_checkpoint(
                work_dir=work_path,
                shard_key=shard.shard_key,
                reason="source manifest drift",
            )

        # 6c. Re-segment. Compute the joined section occurrences,
        #     accumulate its total, run the segmenter once.
        joined = build_region_section_occurrences(shard)
        total_joined_section_occurrences += joined.report.total_occurrence_count
        segmented_res = segment_joined_sections(
            joined.table, segmenter, batch_size=batch_size
        )
        segmented_tables.append(segmented_res.table)
        segmentation_reports.append(segmented_res.report)

        if work_path is not None:
            # Verify the source manifest has not drifted since
            # inventory construction. This is the second and final
            # time the source files are hashed for this shard. On
            # drift the run aborts with active and staging untouched.
            verified_manifest = _verify_pre_publish_manifest(
                shard,
                initial_manifest=current_manifest or [],
                input_root=in_path,
            )

            # Publish as a whole-directory atomic rename. Any failure
            # leaves the active slot untouched. There is no automatic
            # retry: CheckpointPublicationError aborts.
            publish_shard_checkpoint(
                work_dir=work_path,
                shard=shard,
                input_root=in_path,
                table=segmented_res.table,
                report=segmented_res.report,
                input_dataset_revision=input_dataset_revision,
                pipeline_version=pipeline_version,
                source_commit=source_commit or "",
                model_name=model_name or "",
                batch_size=batch_size,
                verified_manifest=verified_manifest,
            )
            completed_shards += 1
            _write_progress_heartbeat(
                work_path,
                processed_regions_count=processed_regions_count,
                completed_shards=completed_shards,
                segmentation_reports=segmentation_reports,
                start_time=start_time,
                input_dataset_revision=input_dataset_revision,
                source_commit=source_commit or "",
            )

    # 7. Persist inventory AFTER every shard has been processed (or
    #    reused). This way the inventory always reflects the as-built
    #    state at the moment the run produced its output.
    if work_path is not None and current_inventory is not None:
        write_run_inventory(work_path, current_inventory)

    # 8. Concatenate and attach metadata.
    if segmented_tables:
        concat_table = pa.concat_tables(segmented_tables)
    else:
        concat_table = SEGMENTED_SENTENCES_SCHEMA.empty_table()

    metadata = {
        b"input_dataset_revision": input_dataset_revision.encode("utf-8"),
        b"pipeline_version": pipeline_version.encode("utf-8"),
    }
    concat_table = concat_table.replace_schema_metadata(metadata)

    # 9. Aggregate segmentation reports.
    input_sec_occ = sum(r.input_section_occurrence_count for r in segmentation_reports)
    emitted_seg = sum(r.emitted_segment_count for r in segmentation_reports)
    retained_sent = sum(
        r.retained_sentence_occurrence_count for r in segmentation_reports
    )
    dropped_raw = sum(r.dropped_empty_raw_count for r in segmentation_reports)
    dropped_norm = sum(r.dropped_empty_normalized_count for r in segmentation_reports)
    wp_sent = sum(r.wikipedia_sentence_occurrence_count for r in segmentation_reports)
    wv_sent = sum(r.wikivoyage_sentence_occurrence_count for r in segmentation_reports)

    agg_seg_report = SegmentationReport(
        input_section_occurrence_count=input_sec_occ,
        emitted_segment_count=emitted_seg,
        retained_sentence_occurrence_count=retained_sent,
        dropped_empty_raw_count=dropped_raw,
        dropped_empty_normalized_count=dropped_norm,
        wikipedia_sentence_occurrence_count=wp_sent,
        wikivoyage_sentence_occurrence_count=wv_sent,
    )

    # 10. Finalize globally.
    finalized = finalize_sentence_dataset(
        concat_table,
        input_dataset_revision=input_dataset_revision,
        pipeline_version=pipeline_version,
        input_dataset_id=input_dataset_id,
    )

    # 11. Export atomically.
    export_res = export_finalized_dataset(
        finalized,
        out_path,
        overwrite=overwrite,
    )

    if work_path is not None:
        _write_heartbeat_or_propagate(
            work_path,
            stage="completed",
            total_shards=processed_regions_count,
            completed_shards=completed_shards,
            current_shard_key=None,
            retained_sentence_occurrence_count=retained_sent,
            dropped_empty_raw_count=dropped_raw,
            dropped_empty_normalized_count=dropped_norm,
            elapsed_seconds=time.monotonic() - start_time,
            input_dataset_revision=input_dataset_revision,
            source_commit=source_commit or "",
        )

    return PipelineResult(
        export_result=export_res,
        processed_regions_count=processed_regions_count,
        total_joined_section_occurrences=total_joined_section_occurrences,
        segmentation_report=agg_seg_report,
        finalization_report=finalized.report,
    )


def _cached_joined_total(report: SegmentationReport) -> int:
    """Return the joined-occurrence total to credit for a cached
    checkpoint.

    The cached ``report`` is produced by :func:`segment_joined_sections`;
    the ``input_section_occurrence_count`` is the count of joined
    sections that flowed into segmentation and is the same value that
    ``joined.report.total_occurrence_count`` recorded before
    segmentation. We use it here so resumed runs preserve
    ``total_joined_section_occurrences`` parity with uninterrupted
    runs without re-running the join layer.
    """
    return int(report.input_section_occurrence_count)


def _write_progress_heartbeat(
    work_path: Path,
    *,
    processed_regions_count: int,
    completed_shards: int,
    segmentation_reports: list[SegmentationReport],
    start_time: float,
    input_dataset_revision: str,
    source_commit: str,
) -> None:
    _write_heartbeat_or_propagate(
        work_path,
        stage="processing",
        total_shards=processed_regions_count,
        completed_shards=completed_shards,
        current_shard_key=None,
        retained_sentence_occurrence_count=sum(
            r.retained_sentence_occurrence_count for r in segmentation_reports
        ),
        dropped_empty_raw_count=sum(
            r.dropped_empty_raw_count for r in segmentation_reports
        ),
        dropped_empty_normalized_count=sum(
            r.dropped_empty_normalized_count for r in segmentation_reports
        ),
        elapsed_seconds=time.monotonic() - start_time,
        input_dataset_revision=input_dataset_revision,
        source_commit=source_commit,
    )


def _write_heartbeat_or_propagate(work_path: Path, **kwargs: Any) -> Path:
    """Call ``write_heartbeat`` and let any error propagate. The
    pipeline never silently swallows heartbeat failures: any successfully
    published checkpoint bytes are preserved exactly, regardless of
    whether this raises.
    """
    return _checkpoint.write_heartbeat(work_path, **kwargs)


__all__ = ["PipelineResult", "run_pipeline"]
