"""Manifest construction and deterministic JSON serialization.

The manifest records row counts, deduplication/duplicate statistics, the
resolved input revision, pipeline version, and the Parquet SHA-256 so that
exports are reproducible and verifiable.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from osm_polygon_sentence_relevance.finalization import FinalizedDataset


def compute_counts(dataset: FinalizedDataset) -> dict:
    """Compute per-source / per-language / per-region row counts."""
    if dataset.table.num_rows > 0:
        by_source = dict(Counter(dataset.table.column("source").to_pylist()))
        by_language = dict(Counter(dataset.table.column("language").to_pylist()))
        by_region = dict(Counter(dataset.table.column("region").to_pylist()))
    else:
        by_source = {}
        by_language = {}
        by_region = {}
    return {
        "counts_by_source": by_source,
        "counts_by_language": by_language,
        "counts_by_region": by_region,
    }


def build_manifest_data(
    dataset: FinalizedDataset,
    input_dataset_revision: str | None,
    pipeline_version: str | None,
    sha256_hex: str,
) -> dict:
    """Assemble the full manifest dictionary for *dataset*."""
    counts = compute_counts(dataset)
    report = dataset.report
    return {
        "row_count": dataset.table.num_rows,
        "input_occurrence_count": (
            report.input_sentence_occurrence_count if report else 0
        ),
        "duplicates_removed": (
            report.duplicate_occurrence_count_removed if report else 0
        ),
        "cross_source_duplicate_groups": (
            report.cross_source_duplicate_group_count if report else 0
        ),
        "counts_by_source": counts["counts_by_source"],
        "counts_by_language": counts["counts_by_language"],
        "counts_by_region": counts["counts_by_region"],
        "input_dataset_revision": input_dataset_revision,
        "pipeline_version": pipeline_version,
        "sha256": sha256_hex,
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
