from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from osm_polygon_sentence_relevance import acquisition
from osm_polygon_sentence_relevance.application.pipeline import (
    PipelineResult,
    run_pipeline,
)
from osm_polygon_sentence_relevance.ingestion.acquisition import AcquisitionResult
from osm_polygon_sentence_relevance.publishing import (
    PublicationResult,
    publish_export_directory,
)
from osm_polygon_sentence_relevance.sentences.sat import SaTSentenceSegmenter


@dataclass(frozen=True, slots=True)
class _ResolvedInput:
    """Resolved input metadata for a single CLI invocation."""

    mode: str  # "local" or "huggingface"
    dataset_id: str | None
    requested_revision: str
    resolved_revision: str
    snapshot_path: str


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with mutually exclusive input modes."""
    parser = argparse.ArgumentParser(
        description="Deterministic OSM Polygon Sentence Relevance Dataset Orchestrator"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-root", help="Existing local input snapshot root directory"
    )
    input_group.add_argument(
        "--input-dataset-id",
        help="Upstream Hugging Face dataset ID to acquire read-only snapshot from",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--input-dataset-revision", required=True, help="Input dataset revision"
    )
    parser.add_argument("--pipeline-version", required=True, help="Pipeline version")
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
    parser.add_argument(
        "--publish-dataset-id",
        help="Optional Hugging Face dataset ID to publish the export to "
        "(after a successful build). The target repository must already "
        "exist. No repository is created.",
    )
    parser.add_argument(
        "--publish-revision",
        default=None,
        help="Target Hugging Face dataset revision for publishing "
        "(default: main). Only used with --publish-dataset-id.",
    )
    parser.add_argument(
        "--publish-commit-message",
        help="Optional commit message for the publishing commit. Only "
        "used with --publish-dataset-id.",
    )
    return parser


def _validate_args(parsed: argparse.Namespace) -> None:
    """Validate parsed arguments before any acquisition or model construction."""
    batch_size = parsed.batch_size
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")

    if not parsed.output_dir.strip():
        raise ValueError("output_dir cannot be blank")

    if not parsed.sat_model.strip():
        raise ValueError("sat_model cannot be blank")

    requested_revision = parsed.input_dataset_revision
    if not requested_revision.strip():
        raise ValueError("input_dataset_revision cannot be blank")

    if not parsed.pipeline_version.strip():
        raise ValueError("pipeline_version cannot be blank")

    if parsed.input_root is not None:
        if not parsed.input_root.strip():
            raise ValueError("input_root cannot be blank")
    else:
        if not parsed.input_dataset_id.strip():
            raise ValueError("input_dataset_id cannot be blank")
        if parsed.input_dataset_id != parsed.input_dataset_id.strip():
            raise ValueError(
                "input_dataset_id has surrounding whitespace; surrounding "
                "whitespace is rejected, not silently normalized"
            )

    # Publishing is optional and strictly post-build. Validate all
    # publishing relationships before acquisition or model construction.
    publish_dataset_id = parsed.publish_dataset_id
    publish_revision = parsed.publish_revision
    publish_commit_message = parsed.publish_commit_message

    if publish_dataset_id is None:
        # A revision or commit message without a dataset id is invalid.
        if publish_revision is not None:
            raise ValueError("publish_revision requires --publish-dataset-id")
        if publish_commit_message is not None:
            raise ValueError("publish_commit_message requires --publish-dataset-id")
    else:
        if not publish_dataset_id.strip():
            raise ValueError("publish_dataset_id cannot be blank")
        # Resolve the effective revision (default "main") after confirming
        # a dataset id is present, then reject a blank explicit value.
        effective_revision = (
            publish_revision if publish_revision is not None else "main"
        )
        if not effective_revision.strip():
            raise ValueError("publish_revision cannot be blank")
        if publish_commit_message is not None and not publish_commit_message.strip():
            raise ValueError("publish_commit_message cannot be blank")


def _resolve_input(
    parsed: argparse.Namespace,
    *,
    acquisition_fn: Callable[..., AcquisitionResult] | None,
) -> _ResolvedInput:
    """Resolve the input mode into a concrete root path and revision."""
    requested_revision = parsed.input_dataset_revision

    if parsed.input_root is not None:
        input_root = Path(parsed.input_root)
        return _ResolvedInput(
            mode="local",
            dataset_id=None,
            requested_revision=requested_revision,
            resolved_revision=requested_revision,
            snapshot_path=str(input_root),
        )

    dataset_id = parsed.input_dataset_id
    # Acquire before constructing the segmenter so acquisition failures
    # do not download model weights.
    snapshot = (acquisition_fn or acquisition.acquire_dataset_snapshot)(
        dataset_id,
        requested_revision,
    )
    return _ResolvedInput(
        mode="huggingface",
        dataset_id=dataset_id,
        requested_revision=requested_revision,
        resolved_revision=snapshot.resolved_sha,
        snapshot_path=str(snapshot.snapshot_path),
    )


def _serialize_summary(res: PipelineResult, resolved: _ResolvedInput) -> str:
    """Serialize a PipelineResult plus resolved input into stable JSON."""
    summary = {
        "parquet_path": str(res.export_result.parquet_path),
        "manifest_path": str(res.export_result.manifest_path),
        "card_path": str(res.export_result.card_path),
        "processed_regions_count": res.processed_regions_count,
        "total_joined_section_occurrences": res.total_joined_section_occurrences,
        "input": {
            "mode": resolved.mode,
            "dataset_id": resolved.dataset_id,
            "requested_revision": resolved.requested_revision,
            "resolved_revision": resolved.resolved_revision,
            "snapshot_path": resolved.snapshot_path,
        },
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
    return json.dumps(
        summary,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def main(
    args: list[str] | None = None,
    *,
    model_factory: Callable[..., object] | None = None,
    acquisition_fn: Callable[..., AcquisitionResult] | None = None,
    publishing_fn: Callable[..., PublicationResult] | None = None,
) -> int:
    """CLI entry point to run the sentence relevance pipeline.

    Publishing is strictly optional and strictly post-build: it runs only
    when ``--publish-dataset-id`` is supplied, and only after the pipeline
    has succeeded. ``publishing_fn`` is injectable; it defaults lazily to
    ``publish_export_directory``. ``PublicationError`` is intentionally
    not caught here so the existing failure boundary yields exit code 1.
    """
    parser = _build_parser()
    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as e:
        code = e.code
        return code if isinstance(code, int) else 2

    try:
        # Validate arguments before acquisition or model construction.
        _validate_args(parsed_args)

        # Resolve input (local path or Hub snapshot) before model construction.
        resolved = _resolve_input(parsed_args, acquisition_fn=acquisition_fn)

        # Construct segmenter only after acquisition has succeeded.
        segmenter = SaTSentenceSegmenter(
            model_name=parsed_args.sat_model,
            model_factory=model_factory,
        )

        # Run pipeline with the resolved input root and immutable revision.
        res = run_pipeline(
            input_root=Path(resolved.snapshot_path),
            output_dir=Path(parsed_args.output_dir),
            segmenter=segmenter,
            input_dataset_revision=resolved.resolved_revision,
            pipeline_version=parsed_args.pipeline_version,
            batch_size=parsed_args.batch_size,
            overwrite=parsed_args.overwrite,
            # Hub mode threads the exact CLI dataset ID into the export
            # chain (Parquet metadata -> manifest -> statistics -> card);
            # local mode omits it (``None``).
            input_dataset_id=resolved.dataset_id,
        )

        summary = json.loads(_serialize_summary(res, resolved))

        # Optional, post-build, single-commit publishing to an existing
        # Hugging Face dataset repository. Never runs without a dataset id,
        # and never before the export is successfully produced.
        publish_dataset_id = parsed_args.publish_dataset_id
        if publish_dataset_id is not None:
            publisher = publishing_fn or publish_export_directory
            target_revision = (
                parsed_args.publish_revision
                if parsed_args.publish_revision is not None
                else "main"
            )
            publication = publisher(
                res.export_result.parquet_path.parent,
                publish_dataset_id,
                target_revision=target_revision,
                commit_message=parsed_args.publish_commit_message,
            )
            summary["publication"] = {
                "dataset_id": publication.dataset_id,
                "target_revision": publication.target_revision,
                "commit_id": publication.commit_id,
                "commit_url": publication.commit_url,
                "row_count": publication.row_count,
                "sha256": publication.sha256,
            }

        print(
            json.dumps(
                summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        )
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
