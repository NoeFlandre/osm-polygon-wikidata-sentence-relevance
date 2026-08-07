"""Deterministic V2 polygon-area and H3 sampling.

Large polygons are all retained in the candidate pool. Tiny, small, and
medium polygons are ordered proportionally across occupied H3 cells before
their sentences are ranked. A larger target extends the same ordered prefix.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import pyarrow as pa

V2_H3_RESOLUTION = 3
V2_SAMPLING_VERSION = "v2-area-h3-logit"
AREA_BUCKETS: dict[str, tuple[float, float]] = {
    "tiny": (0.0, 0.1),
    "small": (0.1, 1.0),
    "medium": (1.0, 10.0),
    "large": (10.0, math.inf),
}
_SOURCE_BUCKET_RANGES: dict[str, tuple[float, float]] = {
    "<100m2": (0.0, 0.0001),
    "100m2-1k_m2": (0.0001, 0.001),
    "1k_m2-10k_m2": (0.001, 0.01),
    "10k_m2-100k_m2": (0.01, 0.1),
    "0.1-1km2": (0.1, 1.0),
    "1-10km2": (1.0, 10.0),
    "10-100km2": (10.0, 100.0),
    ">100km2": (100.0, math.inf),
}
_REQUIRED = frozenset(
    {
        "sentence_id",
        "polygon_id",
        "lat",
        "lon",
        "area_km2",
        "area_bucket",
        "language",
        "osm_primary_tag",
    }
)


def _rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _cell(lat: object, lon: object) -> str:
    if lat is None or lon is None:
        return "(missing)"
    if not isinstance(lat, (int, float, str)) or not isinstance(lon, (int, float, str)):
        raise ValueError("coordinates must be numeric")
    try:
        lat_value = float(lat)
        lon_value = float(lon)
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinates must be numeric") from exc
    if not math.isfinite(lat_value) or not math.isfinite(lon_value):
        raise ValueError("coordinates must be finite")
    try:
        import h3

        return str(h3.latlng_to_cell(lat_value, lon_value, V2_H3_RESOLUTION))
    except ImportError as exc:  # pragma: no cover
        raise ValueError("H3 support is required for V2 sampling") from exc
    except ValueError as exc:
        raise ValueError("coordinates are outside the valid range") from exc


def canonical_area_bucket(area: object, recorded: object) -> str:
    """Normalize upstream area labels to the four V2 sampling buckets.

    The upstream polygon repository has several unit-specific labels and an
    open-ended ``>100km2`` label. V2 deliberately folds every area of at least
    10 km2 into ``large``. The numeric area must agree with the recorded source
    range, preventing silent sampling skew when metadata is corrupted.
    """

    if not isinstance(recorded, str):
        raise ValueError("area_bucket is invalid")
    source_range = _SOURCE_BUCKET_RANGES.get(recorded)
    if source_range is None:
        source_range = AREA_BUCKETS.get(recorded)
    if source_range is None:
        raise ValueError("area_bucket is invalid")
    if not isinstance(area, (int, float, str)):
        raise ValueError("area_km2 must be numeric")
    try:
        value = float(area)
    except (TypeError, ValueError) as exc:
        raise ValueError("area_km2 must be numeric") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("area_km2 must be finite and non-negative")
    lower, upper = source_range
    if not lower <= value < upper:
        raise ValueError("area_km2 and area_bucket do not agree")
    if value < 0.1:
        return "tiny"
    if value < 1.0:
        return "small"
    if value < 10.0:
        return "medium"
    return "large"


def _bucket(area: object, recorded: object) -> str:
    """Compatibility wrapper used by the sampling implementation and tests."""

    return canonical_area_bucket(area, recorded)


def _ordered_polygons(table: pa.Table, seed: str) -> list[str]:
    rows = table.to_pylist()
    by_polygon: dict[str, dict[str, str]] = {}
    for row in rows:
        polygon_id = row["polygon_id"]
        if not isinstance(polygon_id, str) or not polygon_id:
            raise ValueError("polygon_id must be non-empty")
        bucket = _bucket(row["area_km2"], row["area_bucket"])
        cell = _cell(row["lat"], row["lon"])
        current = by_polygon.get(polygon_id)
        candidate = {"bucket": bucket, "cell": cell}
        if current is not None and current != candidate:
            raise ValueError("polygon metadata is inconsistent")
        by_polygon[polygon_id] = candidate
    large = sorted(
        (polygon for polygon, info in by_polygon.items() if info["bucket"] == "large"),
        key=lambda value: _rank(seed, value),
    )
    by_cell: dict[str, list[str]] = defaultdict(list)
    for polygon, info in by_polygon.items():
        if info["bucket"] != "large":
            by_cell[str(info["cell"])].append(polygon)
    for values in by_cell.values():
        values.sort(key=lambda value: _rank(seed, value))
    tail: list[str] = []
    served = dict.fromkeys(by_cell, 0)
    while by_cell:
        cell = min(
            by_cell,
            key=lambda value: (
                served[value] / max(1, len(by_cell[value]) + served[value]),
                _rank(seed, value),
            ),
        )
        values = by_cell[cell]
        tail.append(values.pop(0))
        served[cell] += 1
        if not values:
            del by_cell[cell]
    return large + tail


def _ordered_rows(
    table: pa.Table, *, polygon_order: dict[str, int], seed: str
) -> list[int]:
    """Return a proportional nested order across H3/language/tag strata."""

    rows_by_stratum: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(table.to_pylist()):
        cell = _cell(row["lat"], row["lon"])
        stratum = (
            cell,
            _normalized(row["language"]),
            _normalized(row["osm_primary_tag"]),
        )
        rows_by_stratum[stratum].append(index)
    for indexes in rows_by_stratum.values():
        indexes.sort(
            key=lambda index: (
                polygon_order[str(table["polygon_id"][index].as_py())],
                _rank(seed, str(table["sentence_id"][index].as_py())),
                index,
            )
        )
    initial_sizes = {key: len(value) for key, value in rows_by_stratum.items()}
    served = dict.fromkeys(rows_by_stratum, 0)
    ordered: list[int] = []
    while rows_by_stratum:
        stratum = min(
            rows_by_stratum,
            key=lambda key: (
                served[key] / initial_sizes[key],
                _rank(seed, "\0".join(key)),
            ),
        )
        ordered.append(rows_by_stratum[stratum].pop(0))
        served[stratum] += 1
        if not rows_by_stratum[stratum]:
            del rows_by_stratum[stratum]
    return ordered


def _normalized(value: object) -> str:
    if value is None:
        return "(missing)"
    text = str(value).strip()
    return text or "(missing)"


def select_v2_rows(table: pa.Table, *, target: int, seed: str) -> pa.Table:
    """Select a deterministic nested sentence prefix from the V2 candidate pool."""

    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("target must be a positive integer")
    missing = _REQUIRED.difference(table.column_names)
    if missing:
        raise ValueError(
            f"V2 sampling input is missing required columns: {sorted(missing)}"
        )
    ids = table["sentence_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise ValueError("sampling input contains duplicate sentence IDs")
    ordered_polygons = _ordered_polygons(table, seed)
    polygon_order = {value: index for index, value in enumerate(ordered_polygons)}
    row_indexes = _ordered_rows(table, polygon_order=polygon_order, seed=seed)
    return table.take(pa.array(row_indexes[:target], type=pa.int64()))


__all__ = [
    "AREA_BUCKETS",
    "V2_H3_RESOLUTION",
    "V2_SAMPLING_VERSION",
    "canonical_area_bucket",
    "select_v2_rows",
]
