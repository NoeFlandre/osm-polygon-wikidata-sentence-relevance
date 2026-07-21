"""Manifest construction and deterministic JSON serialization.

The manifest records row counts, deduplication/duplicate statistics, the
resolved input revision, pipeline version, and the Parquet SHA-256 so that
exports are reproducible and verifiable.

All quantitative top-level fields (``row_count``, ``sha256``,
``input_dataset_revision``, ``pipeline_version``, and the three
``counts_by_*`` mappings) are derived from the single ``DatasetStatistics``
instance stored in the manifest's versioned ``statistics`` object. The
validator reconciles them on load and rejects drift between them.
"""

from __future__ import annotations

import json
from pathlib import Path

from osm_polygon_sentence_relevance.output.dataset_card import (
    DatasetStatistics,
    compute_statistics,
    statistics_to_dict,
)
from osm_polygon_sentence_relevance.sentences.finalization import (
    FinalizationReport,
    FinalizedDataset,
)


def build_manifest_data_from_statistics(
    statistics: DatasetStatistics,
    report: FinalizationReport,
) -> dict:
    """Build a manifest from already-computed factual statistics.

    This is the bounded-memory counterpart to :func:`build_manifest_data`.
    Both paths share this single serializer so top-level counts and the
    versioned statistics object cannot drift.
    """

    serialized = statistics_to_dict(statistics)
    return {
        "row_count": serialized["row_count"],
        "input_occurrence_count": report.input_sentence_occurrence_count,
        "duplicates_removed": report.duplicate_occurrence_count_removed,
        "cross_source_duplicate_groups": report.cross_source_duplicate_group_count,
        "counts_by_source": serialized["source_counts"],
        "counts_by_language": serialized["language_counts"],
        "counts_by_region": serialized["region_counts"],
        "input_dataset_revision": serialized["input_dataset_revision"],
        "pipeline_version": serialized["pipeline_version"],
        "input_dataset_id": serialized["input_dataset_id"],
        "sha256": serialized["parquet_sha256"],
        "statistics": serialized,
    }


def build_manifest_data(
    dataset: FinalizedDataset,
    input_dataset_revision: str | None,
    pipeline_version: str | None,
    sha256_hex: str,
    input_dataset_id: str | None = None,
) -> dict:
    """Assemble the full manifest dictionary for *dataset*.

    All quantitative top-level fields are derived from a single
    ``DatasetStatistics`` instance so the manifest cannot disagree with
    itself. ``input_dataset_id`` is threaded explicitly so callers do
    not have to re-derive it from Parquet schema metadata; when absent,
    the metadata-derived value (or ``None`` if no metadata key was
    written) is used.
    """
    statistics = compute_statistics(
        dataset.table,
        input_dataset_revision=input_dataset_revision or "",
        pipeline_version=pipeline_version or "",
        parquet_sha256=sha256_hex,
        input_dataset_id=input_dataset_id,
    )
    return build_manifest_data_from_statistics(statistics, dataset.report)


def write_manifest(path: str | Path, manifest_data: dict) -> None:
    """Write *manifest_data* as deterministic UTF-8 JSON with a trailing newline."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                manifest_data,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        )
