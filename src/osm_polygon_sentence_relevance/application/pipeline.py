from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

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
) -> PipelineResult:
    """Orchestrate regional shard discovery, occurrences join, segmentation, finalization, and export."""
    # 1. Validate configuration and segmenter
    if not isinstance(segmenter, SentenceSegmenter):
        raise TypeError("segmenter must implement the SentenceSegmenter protocol")

    in_path = Path(input_root).resolve()
    out_path = Path(output_dir).resolve()
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

    # 2. Discover shards and explicitly sort them by shard_key ascending
    shards = sorted(discover_shards(Path(input_root)), key=lambda s: s.shard_key)

    # 3. Process regions
    processed_regions_count = len(shards)
    total_joined_section_occurrences = 0

    segmented_tables = []
    segmentation_reports = []

    for shard in shards:
        joined = build_region_section_occurrences(shard)
        total_joined_section_occurrences += joined.report.total_occurrence_count

        segmented_res = segment_joined_sections(
            joined.table, segmenter, batch_size=batch_size
        )
        segmented_tables.append(segmented_res.table)
        segmentation_reports.append(segmented_res.report)

    # 4. Concatenate and attach metadata
    if segmented_tables:
        concat_table = pa.concat_tables(segmented_tables)
    else:
        concat_table = SEGMENTED_SENTENCES_SCHEMA.empty_table()

    metadata = {
        b"input_dataset_revision": input_dataset_revision.encode("utf-8"),
        b"pipeline_version": pipeline_version.encode("utf-8"),
    }
    concat_table = concat_table.replace_schema_metadata(metadata)

    # 5. Aggregate segmentation reports before export so contract failures preserve output
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

    # 6. Finalize globally
    finalized = finalize_sentence_dataset(
        concat_table,
        input_dataset_revision=input_dataset_revision,
        pipeline_version=pipeline_version,
    )

    # 7. Export atomically
    export_res = export_finalized_dataset(
        finalized,
        output_dir,
        overwrite=overwrite,
    )

    return PipelineResult(
        export_result=export_res,
        processed_regions_count=processed_regions_count,
        total_joined_section_occurrences=total_joined_section_occurrences,
        segmentation_report=agg_seg_report,
        finalization_report=finalized.report,
    )
