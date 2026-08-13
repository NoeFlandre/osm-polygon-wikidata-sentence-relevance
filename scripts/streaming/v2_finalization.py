"""Resumable V2 finalization over durable per-shard artifacts."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.contracts.schemas import OUTPUT_SENTENCE_SCHEMA
from osm_polygon_sentence_relevance.labeling.v2_input import (
    download_v2_polygon_metadata,
    enrich_v2_table,
)
from osm_polygon_sentence_relevance.labeling.v2_resumable_sampling import (
    FinalizedShard,
    select_v2_shards_resumable,
)
from osm_polygon_sentence_relevance.output.atomic import (
    cleanup_on_failure,
    install_atomic,
    remove_backup,
)
from osm_polygon_sentence_relevance.output.checksum import sha256_file
from osm_polygon_sentence_relevance.sentences.finalization import (
    FinalizationReport,
    finalize_sentence_dataset,
)
from scripts.streaming.finalized_offload import (
    FinalizedArtifactOffloader,
    schema_sha256,
)
from scripts.streaming.offload import OffloadHandle, materialize_checkpoint


def _report_dict(report: FinalizationReport) -> dict[str, int]:
    return {
        "input_sentence_occurrence_count": report.input_sentence_occurrence_count,
        "output_sentence_count": report.output_sentence_count,
        "duplicate_occurrence_count_removed": report.duplicate_occurrence_count_removed,
        "cross_source_duplicate_group_count": report.cross_source_duplicate_group_count,
    }


def _report_from_metadata(metadata: dict[str, Any]) -> FinalizationReport:
    raw = metadata.get("finalization_report")
    if not isinstance(raw, dict):
        raise ValueError("finalized artifact has no finalization report")
    values = {
        key: raw.get(key)
        for key in (
            "input_sentence_occurrence_count",
            "output_sentence_count",
            "duplicate_occurrence_count_removed",
            "cross_source_duplicate_group_count",
        )
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values.values()
    ):
        raise ValueError("finalized artifact report is invalid")
    return FinalizationReport(**values)


def _aggregate_reports(reports: Sequence[FinalizationReport]) -> FinalizationReport:
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


def finalize_v2_resumable(
    *,
    hub_api: Any,
    ordered_handles: Sequence[OffloadHandle],
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
    persistent_dir: Path,
    output_dir: Path,
    sampling_target: int,
    sampling_seed: str,
) -> Path:
    """Finalize and sample V2 while retaining every completed shard."""

    output = Path(output_dir)
    if output.exists():
        raise ValueError("final output directory must be fresh")
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    persistent = Path(persistent_dir)
    persistent.mkdir(parents=True, exist_ok=True)
    metadata: dict[bytes, bytes] = {
        b"input_dataset_revision": input_dataset_revision.encode("utf-8"),
        b"pipeline_version": pipeline_version.encode("utf-8"),
        b"input_dataset_id": upstream_repo_id.encode("utf-8"),
    }
    stream_schema = OUTPUT_SENTENCE_SCHEMA.append(pa.field("area_km2", pa.float64()))
    stream_schema = stream_schema.append(pa.field("area_bucket", pa.string()))
    stream_schema = stream_schema.with_metadata(metadata)
    expected_identity = {
        "source_commit": source_commit,
        "input_dataset_revision": input_dataset_revision,
        "pipeline_version": pipeline_version,
        "model_name": model_name,
        "batch_size": batch_size,
    }
    offloader = FinalizedArtifactOffloader(
        hub_api=hub_api,
        repo_id=repo_id,
        staging_revision=staging_revision,
        run_id=run_id,
        local_cache_dir=local_cache_dir,
        schema=stream_schema,
        expected_identity=expected_identity,
    )
    handles = {handle.shard_key: handle for handle in ordered_handles}
    descriptors = [
        FinalizedShard(
            handle.shard_key,
            None,
            f"{handle.expected_table_sha256}:{source_commit}:{input_dataset_revision}",
        )
        for handle in sorted(ordered_handles, key=lambda item: item.shard_key)
    ]
    reports: dict[str, FinalizationReport] = {}
    tmp_dir = Path(scratch) / ".v2-finalizing"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    active_root = tmp_dir / "artifacts"
    active_root.mkdir(parents=True, exist_ok=True)
    output_parquet = tmp_dir / "sentences.parquet"
    backup: Path | None = None

    @contextmanager
    def materialize(shard: FinalizedShard) -> Iterator[Path]:
        existing = offloader.inspect(shard.shard_key, materialize=True)
        if existing is not None:
            if existing.local_table_path is None:
                raise ValueError("finalized artifact was not materialized")
            reports[shard.shard_key] = _report_from_metadata(dict(existing.metadata))
            path = existing.local_table_path
            try:
                yield path
            finally:
                path.unlink(missing_ok=True)
            return

        segmented_handle = handles[shard.shard_key]
        materialized = materialize_checkpoint(
            segmented_handle,
            hub_api=hub_api,
            local_cache_dir=local_cache_dir,
        )
        if materialized.local_table_path is None:
            raise ValueError("segmented checkpoint was not materialized")
        active = active_root / shard.shard_key
        active.mkdir(parents=True, exist_ok=True)
        try:
            segmented = pq.read_table(materialized.local_table_path)
            finalized = finalize_sentence_dataset(
                segmented,
                input_dataset_revision=input_dataset_revision,
                pipeline_version=pipeline_version,
                input_dataset_id=upstream_repo_id,
            )
            polygon_metadata = download_v2_polygon_metadata(
                dataset_id=upstream_repo_id,
                revision=input_dataset_revision,
                shard_key=shard.shard_key,
                cache_dir=local_cache_dir,
            )
            output_table = enrich_v2_table(
                finalized.table, {shard.shard_key: polygon_metadata}
            )
            output_table = output_table.replace_schema_metadata(stream_schema.metadata)
            table_path = active / "finalized.parquet"
            pq.write_table(output_table, table_path, compression="zstd")
            digest = sha256_file(table_path)
            artifact_metadata = {
                "schema_version": 1,
                "shard_key": shard.shard_key,
                "table_sha256": digest,
                "table_bytes": table_path.stat().st_size,
                "row_count": output_table.num_rows,
                "schema_sha256": schema_sha256(stream_schema),
                **expected_identity,
                "finalization_report": _report_dict(finalized.report),
            }
            (active / "metadata.json").write_text(
                json.dumps(artifact_metadata, sort_keys=True), encoding="utf-8"
            )
            offloader.upload_and_verify(
                shard_key=shard.shard_key,
                active_dir=active,
                metadata=artifact_metadata,
            )
            reports[shard.shard_key] = finalized.report
            yield table_path
        finally:
            materialized.local_table_path.unlink(missing_ok=True)
            shutil.rmtree(active, ignore_errors=True)

    try:
        select_v2_shards_resumable(
            descriptors,
            output_parquet,
            target=sampling_target,
            seed=sampling_seed,
            state_dir=persistent / "sampling",
            materialize_shard=materialize,
        )
        for descriptor in descriptors:
            if descriptor.shard_key in reports:
                continue
            existing = offloader.inspect(descriptor.shard_key, materialize=False)
            metadata: Mapping[str, Any] | None = getattr(existing, "metadata", None)
            if metadata is not None:
                reports[descriptor.shard_key] = _report_from_metadata(
                    {str(key): value for key, value in metadata.items()}
                )
        missing_reports = [
            descriptor.shard_key
            for descriptor in descriptors
            if descriptor.shard_key not in reports
        ]
        if missing_reports:
            raise ValueError(
                f"finalization reports are missing for shards: {missing_reports!r}"
            )
        report = _aggregate_reports(
            [reports[descriptor.shard_key] for descriptor in descriptors]
        )
        digest = sha256_file(output_parquet)
        selected_rows = pq.ParquetFile(output_parquet).metadata.num_rows
        manifest = {
            "manifest_version": 1,
            "purpose": "v2-worldwide-label-input",
            "row_count": selected_rows,
            "sha256": digest,
            "input_dataset_id": upstream_repo_id,
            "input_dataset_revision": input_dataset_revision,
            "source_commit": source_commit,
            "pipeline_version": pipeline_version,
            "sampling": {
                "target": sampling_target,
                "seed": sampling_seed,
                "source_finalized_rows": report.output_sentence_count,
            },
        }
        final_tmp = tmp_dir / "out"
        final_tmp.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_parquet, final_tmp / "sentences.parquet")
        (final_tmp / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (final_tmp / "README.md").write_text(
            "# Worldwide V2 labeling input\n\n"
            "This internal artifact was generated deterministically from the "
            "complete validated split checkpoints.\n\n"
            f"- Selected sentences: {selected_rows:,}\n"
            f"- Source finalized sentences: {report.output_sentence_count:,}\n"
            f"- Sampling seed: `{sampling_seed}`\n"
            f"- Input revision: `{input_dataset_revision}`\n"
            f"- SHA-256: `{digest}`\n",
            encoding="utf-8",
        )
        backup = install_atomic(final_tmp, output)
        if backup is not None:
            remove_backup(backup)
        return output
    except Exception:
        cleanup_on_failure(tmp_dir, backup)
        raise


__all__ = ["finalize_v2_resumable"]
