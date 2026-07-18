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
    compute_statistics,
    statistics_to_dict,
)
from osm_polygon_sentence_relevance.sentences.finalization import FinalizedDataset


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
    report = dataset.report
    statistics = statistics_to_dict(
        compute_statistics(
            dataset.table,
            input_dataset_revision=input_dataset_revision or "",
            pipeline_version=pipeline_version or "",
            parquet_sha256=sha256_hex,
            input_dataset_id=input_dataset_id,
        )
    )
    return {
        "row_count": statistics["row_count"],
        "input_occurrence_count": (
            report.input_sentence_occurrence_count if report else 0
        ),
        "duplicates_removed": (
            report.duplicate_occurrence_count_removed if report else 0
        ),
        "cross_source_duplicate_groups": (
            report.cross_source_duplicate_group_count if report else 0
        ),
        "counts_by_source": statistics["source_counts"],
        "counts_by_language": statistics["language_counts"],
        "counts_by_region": statistics["region_counts"],
        "input_dataset_revision": statistics["input_dataset_revision"],
        "pipeline_version": statistics["pipeline_version"],
        "input_dataset_id": statistics["input_dataset_id"],
        "sha256": statistics["parquet_sha256"],
        "statistics": statistics,
    }


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
