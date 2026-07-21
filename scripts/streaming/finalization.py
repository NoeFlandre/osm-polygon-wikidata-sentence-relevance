"""Bounded finalization of authoritative streamed shard checkpoints.

The expensive CUDA sentence segmentation is checkpointed on a Hugging Face
staging branch.  This module materializes one checkpoint at a time, applies
the canonical production finalizer, writes its rows directly to the final
Parquet stream, and evicts the local checkpoint before continuing.

Global correctness follows from the input contract: every ``polygon_id`` is
prefixed by its shard key.  The context and deduplication keys both include
``polygon_id``; consequently no group can cross a shard boundary.  We verify
that prefix and the global output ordering while writing.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from scripts.streaming.driver import list_remote_shard_keys
from scripts.streaming.offload import (
    OffloadHandle,
    discover_run,
    materialize_checkpoint,
)

from osm_polygon_sentence_relevance.contracts.schemas import OUTPUT_SENTENCE_SCHEMA
from osm_polygon_sentence_relevance.output.atomic import (
    cleanup_on_failure,
    install_atomic,
    remove_backup,
)
from osm_polygon_sentence_relevance.output.checksum import sha256_file
from osm_polygon_sentence_relevance.output.dataset_card import (
    compute_parquet_statistics,
    render_dataset_card,
)
from osm_polygon_sentence_relevance.output.manifest import (
    build_manifest_data_from_statistics,
    write_manifest,
)
from osm_polygon_sentence_relevance.sentences.finalization import (
    FinalizationReport,
    finalize_sentence_dataset,
)


class StreamingFinalizationError(RuntimeError):
    """The authoritative checkpoints could not be finalized safely."""


def _identity(
    *,
    source_commit: str,
    input_dataset_revision: str,
    pipeline_version: str,
    model_name: str,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "source_commit": source_commit,
        "input_dataset_revision": input_dataset_revision,
        "pipeline_version": pipeline_version,
        "model_name": model_name,
        "batch_size": batch_size,
    }


def _validate_inventory(
    handles: Sequence[OffloadHandle], expected_shard_keys: Iterable[str]
) -> list[OffloadHandle]:
    expected = set(expected_shard_keys)
    actual = {handle.shard_key for handle in handles}
    if len(actual) != len(handles):
        raise StreamingFinalizationError("staging run contains duplicate shard keys")
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise StreamingFinalizationError(
            "staging checkpoint inventory is incomplete or unexpected: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return sorted(handles, key=lambda handle: handle.shard_key)


def _verify_shard_namespace(table: pa.Table, shard_key: str) -> None:
    prefix = f"{shard_key}:"
    for polygon_id in table.column("polygon_id").to_pylist():
        if not polygon_id.startswith(prefix):
            raise StreamingFinalizationError(
                f"checkpoint {shard_key!r} contains a polygon from another shard"
            )


def _aggregate_reports(reports: Iterable[FinalizationReport]) -> FinalizationReport:
    reports = tuple(reports)
    return FinalizationReport(
        input_sentence_occurrence_count=sum(
            item.input_sentence_occurrence_count for item in reports
        ),
        output_sentence_count=sum(item.output_sentence_count for item in reports),
        duplicate_occurrence_count_removed=sum(
            item.duplicate_occurrence_count_removed for item in reports
        ),
        cross_source_duplicate_group_count=sum(
            item.cross_source_duplicate_group_count for item in reports
        ),
    )


def _evict_materialized(handle: OffloadHandle, cache_root: Path) -> None:
    """Remove only the materialized files belonging to one verified handle."""

    root = Path(os.path.realpath(cache_root))
    candidates = (handle.local_table_path, handle.local_metadata_path)
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        physical = Path(os.path.realpath(candidate))
        try:
            physical.relative_to(root)
        except ValueError as error:
            raise StreamingFinalizationError(
                "refusing to evict a checkpoint outside the finalization cache"
            ) from error
        physical.unlink()
    for parent in sorted(
        {candidate.parent for candidate in candidates if candidate is not None},
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        current = parent
        while current != root and current.is_relative_to(root):
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


def finalize_streamed_run(
    *,
    hub_api: Any,
    repo_id: str,
    upstream_repo_id: str,
    run_id: str,
    staging_revision: str,
    source_commit: str,
    input_dataset_revision: str,
    pipeline_version: str,
    model_name: str,
    batch_size: int,
    local_cache_dir: Path,
    scratch_dir: Path,
    output_dir: Path,
    expected_shard_keys: Sequence[str] | None = None,
) -> Path:
    """Build and atomically install the three final public artifacts.

    At most one segmented checkpoint and one per-shard finalized Arrow table
    are resident at a time.  The final Parquet stream itself is written once.
    """

    if os.environ.get("OAR_JOB_ID", "").isdigit() is False:
        raise StreamingFinalizationError("finalization requires an OAR compute job")
    cache = Path(local_cache_dir)
    scratch = Path(scratch_dir)
    output = Path(output_dir)
    if output.exists():
        raise StreamingFinalizationError("final output directory must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    identity = _identity(
        source_commit=source_commit,
        input_dataset_revision=input_dataset_revision,
        pipeline_version=pipeline_version,
        model_name=model_name,
        batch_size=batch_size,
    )
    if expected_shard_keys is None:
        expected_shard_keys = list_remote_shard_keys(
            hub_api=hub_api,
            repo_id=upstream_repo_id,
            revision=input_dataset_revision,
        )
    handles = discover_run(
        hub_api=hub_api,
        repo_id=repo_id,
        run_id=run_id,
        staging_revision=staging_revision,
        local_cache_dir=cache,
        expected_identity=identity,
    )
    ordered = _validate_inventory(handles, expected_shard_keys)

    tmp_dir = Path(tempfile.mkdtemp(prefix=".finalizing-", dir=output.parent))
    backup: Path | None = None
    parquet_path = tmp_dir / "sentences.parquet"
    metadata: Mapping[bytes, bytes] = {
        b"input_dataset_revision": input_dataset_revision.encode("utf-8"),
        b"pipeline_version": pipeline_version.encode("utf-8"),
        b"input_dataset_id": upstream_repo_id.encode("utf-8"),
    }
    writer = pq.ParquetWriter(
        parquet_path,
        OUTPUT_SENTENCE_SCHEMA.with_metadata(metadata),
    )
    reports: list[FinalizationReport] = []
    try:
        for handle in ordered:
            materialized = materialize_checkpoint(
                handle, hub_api=hub_api, local_cache_dir=cache
            )
            if materialized.local_table_path is None:
                raise StreamingFinalizationError("materialized checkpoint has no table")
            segmented = pq.read_table(materialized.local_table_path)
            _verify_shard_namespace(segmented, handle.shard_key)
            finalized = finalize_sentence_dataset(
                segmented,
                input_dataset_revision=input_dataset_revision,
                pipeline_version=pipeline_version,
                input_dataset_id=upstream_repo_id,
            )
            writer.write_table(finalized.table)
            reports.append(finalized.report)
            del segmented, finalized
            _evict_materialized(materialized, cache)
        writer.close()

        report = _aggregate_reports(reports)
        digest = sha256_file(parquet_path)
        statistics = compute_parquet_statistics(
            parquet_path,
            input_dataset_revision=input_dataset_revision,
            pipeline_version=pipeline_version,
            parquet_sha256=digest,
            input_dataset_id=upstream_repo_id,
            scratch_dir=scratch,
        )
        if report.output_sentence_count != statistics.row_count:
            raise StreamingFinalizationError(
                "aggregated finalization report does not match final Parquet rows"
            )
        manifest = build_manifest_data_from_statistics(statistics, report)
        write_manifest(tmp_dir / "manifest.json", manifest)
        (tmp_dir / "README.md").write_text(
            render_dataset_card(statistics), encoding="utf-8"
        )
        backup = install_atomic(tmp_dir, output)
        if backup is not None:
            remove_backup(backup)
        return output
    except Exception:
        writer.close()
        cleanup_on_failure(tmp_dir, backup)
        raise


def main(argv: list[str] | None = None) -> int:
    """CLI used inside a non-frontend OAR compute allocation."""

    parser = argparse.ArgumentParser(prog="scripts.streaming.finalization")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--upstream-repo-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--staging-revision", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--input-revision", required=True)
    parser.add_argument("--pipeline-version", default="0.1.0")
    parser.add_argument("--model-name", default="sat-3l-sm")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--scratch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-shard",
        action="append",
        default=None,
        help=(
            "Restrict finalization to exactly this shard key (repeatable). "
            "When supplied, the upstream is NOT enumerated and the staging "
            "checkpoint set must match the supplied keys exactly. Required "
            "for one-shard canary finalizations (e.g. afghanistan-latest)."
        ),
    )
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi

    result = finalize_streamed_run(
        hub_api=HfApi(),
        repo_id=args.repo_id,
        upstream_repo_id=args.upstream_repo_id,
        run_id=args.run_id,
        staging_revision=args.staging_revision,
        source_commit=args.source_commit,
        input_dataset_revision=args.input_revision,
        pipeline_version=args.pipeline_version,
        model_name=args.model_name,
        batch_size=args.batch_size,
        local_cache_dir=Path(args.cache_dir),
        scratch_dir=Path(args.scratch_dir),
        output_dir=Path(args.output_dir),
        expected_shard_keys=args.expected_shard,
    )
    print(json.dumps({"output_dir": str(result)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "StreamingFinalizationError",
    "finalize_streamed_run",
    "main",
]
