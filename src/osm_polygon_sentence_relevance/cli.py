from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from osm_polygon_sentence_relevance.sat_adapter import SaTSentenceSegmenter
from osm_polygon_sentence_relevance.pipeline import run_pipeline


def main(args: list[str] | None = None, *, model_factory=None) -> int:
    """CLI entry point to run the sentence relevance pipeline."""
    parser = argparse.ArgumentParser(
        description="Deterministic OSM Polygon Sentence Relevance Dataset Orchestrator"
    )
    parser.add_argument("--input-root", required=True, help="Input root directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--input-dataset-revision", required=True, help="Input dataset revision"
    )
    parser.add_argument(
        "--pipeline-version", required=True, help="Pipeline version"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for segmenter"
    )
    parser.add_argument(
        "--sat-model", default="sat-3l-sm", help="wtpsplit SaT model name"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as e:
        return e.code

    try:
        # Validate batch size here before model construction
        batch_size = parsed_args.batch_size
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if not parsed_args.input_root.strip():
            raise ValueError("input_root cannot be blank")

        if not parsed_args.output_dir.strip():
            raise ValueError("output_dir cannot be blank")

        if not parsed_args.sat_model.strip():
            raise ValueError("sat_model cannot be blank")

        if not parsed_args.input_dataset_revision.strip():
            raise ValueError("input_dataset_revision cannot be blank")

        if not parsed_args.pipeline_version.strip():
            raise ValueError("pipeline_version cannot be blank")

        # Construct segmenter
        segmenter = SaTSentenceSegmenter(
            model_name=parsed_args.sat_model,
            model_factory=model_factory,
        )

        # Run pipeline
        res = run_pipeline(
            input_root=Path(parsed_args.input_root),
            output_dir=Path(parsed_args.output_dir),
            segmenter=segmenter,
            input_dataset_revision=parsed_args.input_dataset_revision,
            pipeline_version=parsed_args.pipeline_version,
            batch_size=parsed_args.batch_size,
            overwrite=parsed_args.overwrite,
        )

        # Print stable JSON summary to stdout
        summary = {
            "parquet_path": str(res.export_result.parquet_path),
            "manifest_path": str(res.export_result.manifest_path),
            "processed_regions_count": res.processed_regions_count,
            "total_joined_section_occurrences": res.total_joined_section_occurrences,
            "segmentation_report": {
                "input_section_occurrence_count": res.segmentation_report.input_section_occurrence_count,
                "emitted_segment_count": res.segmentation_report.emitted_segment_count,
                "retained_sentence_occurrence_count": res.segmentation_report.retained_sentence_occurrence_count,
                "dropped_empty_raw_count": res.segmentation_report.dropped_empty_raw_count,
                "dropped_empty_normalized_count": res.segmentation_report.dropped_empty_normalized_count,
                "wikipedia_sentence_occurrence_count": res.segmentation_report.wikipedia_sentence_occurrence_count,
                "wikivoyage_sentence_occurrence_count": res.segmentation_report.wikivoyage_sentence_occurrence_count,
            },
            "finalization_report": {
                "input_sentence_occurrence_count": res.finalization_report.input_sentence_occurrence_count,
                "output_sentence_count": res.finalization_report.output_sentence_count,
                "duplicate_occurrence_count_removed": res.finalization_report.duplicate_occurrence_count_removed,
                "cross_source_duplicate_group_count": res.finalization_report.cross_source_duplicate_group_count,
            },
        }

        print(
            json.dumps(
                summary,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
