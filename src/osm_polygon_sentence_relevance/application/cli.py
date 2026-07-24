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
from osm_polygon_sentence_relevance.contracts._exception_chain import (
    format_exception_chain,
)
from osm_polygon_sentence_relevance.contracts.errors import SegmentationError
from osm_polygon_sentence_relevance.ingestion.acquisition import AcquisitionResult
from osm_polygon_sentence_relevance.publishing import (
    PublicationResult,
    publish_export_directory,
)
from osm_polygon_sentence_relevance.sentences.device import (
    PUBLIC_DEVICE_VALUES,
    default_caps,
    resolve_device,
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
        "--sat-model", default="sat-12l-sm", help="wtpsplit SaT model name"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=sorted(PUBLIC_DEVICE_VALUES),
        help=(
            "Accelerator for SaT inference. ``auto`` (default) prefers "
            "CUDA, then MPS, then CPU. Explicit ``cuda``/``mps`` fail "
            "when the backend is unavailable."
        ),
    )
    parser.add_argument(
        "--input-source-dataset-id",
        default=None,
        help=(
            "Optional Hugging Face dataset ID of the upstream source for "
            "a local input snapshot. Only valid with --input-root; it "
            "populates the source provenance recorded in the manifest and "
            "dataset card without triggering any network request."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help=(
            "Optional persistent work directory for shard-level "
            "checkpoints and a factual progress heartbeat. When "
            "supplied, the pipeline publishes one checkpoint per "
            "shard after segmentation, written under "
            "${work_dir}/shards/active/<shard_key>/, and a "
            "heartbeat.json updated at shard boundaries. A "
            "subsequent invocation with the same work_dir resumes "
            "from the last valid checkpoint; invalid or mismatched "
            "checkpoints are moved into "
            "${work_dir}/shards/quarantine/ with a UUID-suffixed "
            "unique name and their original bytes are preserved. "
            "Cannot overlap with --input-root or --output-dir. "
            "Ignored when omitted (legacy no-work-directory mode)."
        ),
    )
    parser.add_argument(
        "--source-commit",
        default=None,
        help=(
            "Source commit SHA (40 lowercase hex characters) to bind "
            "each checkpoint and the heartbeat to a specific code "
            "revision. Required when --work-dir is supplied; "
            "ignored otherwise. The value is validated as a 40-char "
            "lowercase hex string and is recorded verbatim into "
            "every shard checkpoint."
        ),
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

    # Validate the requested inference device eagerly. The CLI performs
    # both the syntactic / public-set check and a hardware-availability
    # probe (via the production capability snapshot) so explicit
    # unavailable accelerators fail before acquisition or model
    # construction. ``--help`` is served by argparse before this function
    # is reached, so no Torch import occurs for the help path.
    device_value = parsed.device
    if not isinstance(device_value, str) or not device_value.strip():
        raise ValueError("device cannot be blank")
    if device_value not in PUBLIC_DEVICE_VALUES:
        raise ValueError(
            f"device must be one of {sorted(PUBLIC_DEVICE_VALUES)}; "
            f"got {device_value!r}"
        )
    try:
        resolve_device(device_value, caps=default_caps())
    except SegmentationError as exc:
        raise ValueError(str(exc)) from exc

    # ``--input-source-dataset-id`` is only meaningful with ``--input-root``.
    # Hub acquisition already carries the dataset ID; supplying both would
    # create ambiguity.
    source_dataset_id = parsed.input_source_dataset_id
    if source_dataset_id is not None:
        if parsed.input_root is None:
            raise ValueError("input-source-dataset-id is only valid with --input-root")
        if not isinstance(source_dataset_id, str) or not source_dataset_id.strip():
            raise ValueError("input-source-dataset-id cannot be blank")
        if source_dataset_id != source_dataset_id.strip():
            raise ValueError(
                "input-source-dataset-id has surrounding whitespace; "
                "surrounding whitespace is rejected, not silently normalized"
            )

    # Publishing is optional and strictly post-build. Validate all
    # publishing relationships before acquisition or model construction.
    publish_dataset_id = parsed.publish_dataset_id
    publish_revision = parsed.publish_revision
    publish_commit_message = parsed.publish_commit_message

    # ``--work-dir`` is optional and orthogonal to all publishing flags.
    # We only validate the syntactic form here; overlap with input /
    # output is checked by the pipeline itself (after argument
    # validation), once the canonical paths are resolved.
    work_dir = parsed.work_dir
    if work_dir is not None:
        if not isinstance(work_dir, str) or not work_dir.strip():
            raise ValueError("work_dir must be a non-blank string when provided")
        if work_dir != work_dir.strip():
            raise ValueError(
                "work_dir has surrounding whitespace; surrounding "
                "whitespace is rejected, not silently normalized"
            )
        # ``--source-commit`` must be supplied when ``--work-dir`` is.
        source_commit = parsed.source_commit
        if source_commit is None:
            raise ValueError(
                "--source-commit is required when --work-dir is set; "
                "provide a 40-char lowercase hex Git commit SHA"
            )
        if (
            not isinstance(source_commit, str)
            or not source_commit.strip()
            or not __import__("re").match(r"^[0-9a-f]{40}$", source_commit.strip())
        ):
            raise ValueError(
                "--source-commit must be a 40-character lowercase hex string"
            )

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
            # ``dataset_id`` is the single provenance value: Hub
            # acquisitions supply the upstream ID; local snapshots
            # may carry the source dataset ID via
            # ``--input-source-dataset-id``; otherwise ``None``.
            dataset_id=parsed.input_source_dataset_id,
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
            device=parsed_args.device,
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
            work_dir=parsed_args.work_dir,
            source_commit=parsed_args.source_commit,
            model_name=parsed_args.sat_model,
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
        # Render the bounded exception chain so the build log
        # surfaces the actual underlying cause (e.g. CUDA OOM,
        # allocator failure, weight-load mismatch) instead of just
        # the top-level message. No traceback frames, file paths,
        # or local variable bindings are emitted.
        print(format_exception_chain(e), file=sys.stderr)
        return 1
