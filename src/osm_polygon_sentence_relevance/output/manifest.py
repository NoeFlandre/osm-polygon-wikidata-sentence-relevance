"""Manifest construction and deterministic JSON serialization.

The manifest records row counts, deduplication/duplicate statistics, the
resolved input revision, pipeline version, and the Parquet SHA-256 so that
exports are reproducible and verifiable.

All quantitative top-level fields (``row_count``, ``sha256``,
``input_dataset_revision``, ``pipeline_version``, and the three
``counts_by_*`` mappings) are derived from the single ``DatasetStatistics``
instance stored in the manifest's versioned ``statistics`` object. The
validator reconciles them on load and rejects drift between them.

Phase 9P adds profile-derived fields (``segmentation_model``,
``segmentation_revision``, ``source_commit``, asset SHA-256s, and the
full profile JSON) so the validator can reject drift between the
export directory and the published assets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from osm_polygon_sentence_relevance.output.dataset_card import (
    DatasetStatistics,
    compute_statistics,
    statistics_to_dict,
)
from osm_polygon_sentence_relevance.output.profile import DatasetProfile
from osm_polygon_sentence_relevance.sentences.finalization import (
    FinalizationReport,
    FinalizedDataset,
)

# Schema version of the versioned manifest statistics object stored
# on disk.  Phase 9P bumps this to 2 because the profile-shaped fields
# (``assets``, ``segmentation_model``, etc.) are now required by the
# validator.  Validators and the rendering paths pin against this.
MANIFEST_VERSION = 2


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
        "manifest_version": MANIFEST_VERSION,
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


def merge_profile_into_manifest(
    manifest_data: dict[str, Any],
    profile: DatasetProfile,
    *,
    assets: list[dict[str, Any]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Overlay profile-derived fields onto an existing manifest dict.

    The merged manifest's quantitative figures remain identical to the
    input (the profile embeds the same row count, language counts,
    etc., as the existing ``statistics`` object), but the
    asset SHA-256s, segmentation model, segmentation revision, source
    commit, example row, lat / lon extents, sentence-length summary,
    and profile-version metadata are attached at the top level for the
    validator to cross-check.

    Parameters
    ----------
    manifest_data
        Output of :func:`build_manifest_data` or
        :func:`build_manifest_data_from_statistics`.
    profile
        The :class:`DatasetProfile` built from the same Parquet.
    assets
        Optional explicit list of asset dicts (e.g. ``[{"name":
        "geographic_coverage.png", "sha256": ..., "bytes": ...}]``).
        Defaults to the profile's ``assets`` mapping.
    generated_at
        Optional ISO-8601 timestamp recorded in the manifest for
        auditing.  When omitted, no ``generated_at`` field is added.
    """
    merged = dict(manifest_data)
    merged["manifest_version"] = MANIFEST_VERSION

    # Asset block: list of {"name", "sha256", "bytes"} in deterministic
    # alphabetical order so the manifest can be byte-equal across
    # reruns without re-ordering the assets dict.
    if assets is None:
        assets = [
            {
                "name": info.name,
                "sha256": info.sha256,
                "bytes": info.bytes_,
            }
            for _, info in sorted(profile.assets.items())
        ]
    merged["assets"] = assets
    merged["segmentation_model"] = profile.segmentation_model
    merged["segmentation_revision"] = profile.segmentation_revision
    merged["source_commit"] = profile.source_commit
    merged["rows_with_polygon_name"] = profile.rows_with_polygon_name
    merged["lat_min"] = profile.lat_min
    merged["lat_max"] = profile.lat_max
    merged["lon_min"] = profile.lon_min
    merged["lon_max"] = profile.lon_max
    merged["sentence_length_min"] = profile.sentence_length_min
    merged["sentence_length_mean"] = profile.sentence_length_mean
    merged["sentence_length_max"] = profile.sentence_length_max
    # Example row in schema-column order for stable comparison.
    from osm_polygon_sentence_relevance.contracts.schemas import (
        OUTPUT_SENTENCE_SCHEMA,
    )

    row: dict[str, Any] = {}
    for col in OUTPUT_SENTENCE_SCHEMA.names:
        row[col] = profile.example_row.fields.get(col)
    merged["example_row"] = row
    if generated_at is not None:
        merged["generated_at"] = generated_at
    return merged


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


__all__ = [
    "MANIFEST_VERSION",
    "build_manifest_data",
    "build_manifest_data_from_statistics",
    "merge_profile_into_manifest",
    "write_manifest",
]

