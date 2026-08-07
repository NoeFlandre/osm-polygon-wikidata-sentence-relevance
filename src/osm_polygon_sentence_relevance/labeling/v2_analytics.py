"""Factual analytics for the V2 binary score table."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import pyarrow as pa

from .analytics import h3_sentence_distribution


@dataclass(frozen=True, slots=True)
class V2Analytics:
    """Metrics derived only from the finalized V2 Parquet table."""

    total_labeled_sentences: int
    unique_polygons: int
    unique_languages: int
    place_counts: dict[str, int]
    place_percentages: dict[str, float]
    area_bucket_counts: dict[str, int]
    h3_cell_count: int
    missing_coordinate_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_v2_analytics(table: pa.Table, *, h3_resolution: int) -> V2Analytics:
    """Compute all public V2 summary values from one table."""

    required = {
        "polygon_id",
        "language",
        "place_relevance",
        "area_bucket",
        "lat",
        "lon",
    }
    if missing := sorted(required.difference(table.column_names)):
        raise ValueError(f"missing required V2 analytics columns: {missing}")
    labels = [str(value) for value in table["place_relevance"].to_pylist()]
    invalid = sorted(set(labels) - {"yes", "no"})
    if invalid:
        raise ValueError(f"invalid V2 place_relevance values: {invalid}")
    total = table.num_rows
    counts = dict(sorted(Counter(labels).items()))
    cells, missing_coordinates = h3_sentence_distribution(
        table, resolution=h3_resolution
    )
    return V2Analytics(
        total_labeled_sentences=total,
        unique_polygons=len(set(table["polygon_id"].to_pylist())),
        unique_languages=len(set(table["language"].to_pylist())),
        place_counts=counts,
        place_percentages={key: value / total for key, value in counts.items()},
        area_bucket_counts=dict(
            sorted(Counter(table["area_bucket"].to_pylist()).items())
        ),
        h3_cell_count=len(cells),
        missing_coordinate_count=missing_coordinates,
    )


__all__ = ["V2Analytics", "build_v2_analytics"]
