"""Deterministic H3, language, and OSM-tag stratified row selection."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass

import pyarrow as pa

DEFAULT_SAMPLE_TARGET = 200_000
DEFAULT_SAMPLE_SEED = "sentence-relevance-v2"
DEFAULT_H3_RESOLUTION = 3
SAMPLING_VERSION = "labeling-v2-h3-language-osm-primary"
MISSING_STRATUM = "(missing)"
_REQUIRED_COLUMNS = frozenset(
    {"sentence_id", "lat", "lon", "language", "osm_primary_tag"}
)


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Immutable parameters that define one row selection."""

    target: int = DEFAULT_SAMPLE_TARGET
    seed: str = DEFAULT_SAMPLE_SEED
    h3_resolution: int = DEFAULT_H3_RESOLUTION
    version: str = SAMPLING_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.target, bool) or not isinstance(self.target, int):
            raise ValueError("sampling target must be a positive integer")
        if self.target < 1:
            raise ValueError("sampling target must be a positive integer")
        if (
            not isinstance(self.seed, str)
            or not self.seed
            or self.seed != self.seed.strip()
        ):
            raise ValueError("sampling seed must be a non-blank string")
        if (
            isinstance(self.h3_resolution, bool)
            or not isinstance(self.h3_resolution, int)
            or not 0 <= self.h3_resolution <= 15
        ):
            raise ValueError("H3 resolution must be an integer in [0, 15]")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("sampling version must be non-blank")


def _normalized(value: object) -> str:
    if value is None:
        return MISSING_STRATUM
    text = str(value).strip()
    return text or MISSING_STRATUM


def _h3_cell(lat: object, lon: object, resolution: int) -> str:
    if lat is None or lon is None:
        return MISSING_STRATUM
    if isinstance(lat, bool) or not isinstance(lat, (int, float, str)):
        raise ValueError("coordinate values must be numeric")
    if isinstance(lon, bool) or not isinstance(lon, (int, float, str)):
        raise ValueError("coordinate values must be numeric")
    try:
        lat_value = float(lat)
        lon_value = float(lon)
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinate values must be numeric") from exc
    if not math.isfinite(lat_value) or not math.isfinite(lon_value):
        raise ValueError("coordinate values must be finite")
    if not -90.0 <= lat_value <= 90.0 or not -180.0 <= lon_value <= 180.0:
        raise ValueError("coordinate values are outside the valid range")
    try:
        import h3

        return str(h3.latlng_to_cell(lat_value, lon_value, resolution))
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ValueError("H3 support is required for stratified sampling") from exc
    except (ValueError, h3.H3ValueError) as exc:
        raise ValueError("could not assign coordinate to an H3 cell") from exc


def _rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _allocation(
    capacities: dict[tuple[str, str, str], int], target: int, seed: str
) -> dict[tuple[str, str, str], int]:
    """Allocate a nested proportional prefix with a weighted fair queue.

    The next row always comes from the stratum with the smallest prospective
    selected fraction. Therefore the first ``200_000`` rows are a prefix of
    the first ``400_000`` rows when the same input and seed are used.
    """

    result = dict.fromkeys(capacities, 0)
    heap: list[tuple[float, str, tuple[str, str, str]]] = [
        (1.0 / capacity, _rank(seed, "\0".join(key)), key)
        for key, capacity in capacities.items()
        if capacity > 0
    ]
    heapq.heapify(heap)
    for _ in range(target):
        ratio, tie_breaker, key = heapq.heappop(heap)
        result[key] += 1
        if result[key] < capacities[key]:
            next_ratio = (result[key] + 1) / capacities[key]
            heapq.heappush(heap, (next_ratio, tie_breaker, key))
    return result


def select_stratified_rows(
    table: pa.Table,
    *,
    target: int = DEFAULT_SAMPLE_TARGET,
    seed: str = DEFAULT_SAMPLE_SEED,
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
) -> pa.Table:
    """Select up to ``target`` rows across H3, language, and primary-tag strata.

    The H3 resolution and all selection parameters are explicit so a future
    larger sample can be reproduced exactly. Rows retain their original order;
    only the selected IDs change. Missing coordinates, language, or tags are
    kept in an explicit ``(missing)`` stratum rather than silently discarded.
    """

    config = SamplingConfig(target=target, seed=seed, h3_resolution=h3_resolution)
    ids = table["sentence_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise ValueError("sampling input contains duplicate sentence IDs")
    if table.num_rows <= config.target:
        # Still validate every coordinate so an invalid row cannot be hidden by
        # the target being larger than the input.
        if {"lat", "lon"}.issubset(table.column_names):
            for lat, lon in zip(
                table["lat"].to_pylist(), table["lon"].to_pylist(), strict=True
            ):
                _h3_cell(lat, lon, config.h3_resolution)
        return table
    missing = _REQUIRED_COLUMNS.difference(table.column_names)
    if missing:
        raise ValueError(
            f"sampling input is missing required columns: {sorted(missing)}"
        )

    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, (lat, lon, language, tag) in enumerate(
        zip(
            table["lat"].to_pylist(),
            table["lon"].to_pylist(),
            table["language"].to_pylist(),
            table["osm_primary_tag"].to_pylist(),
            strict=True,
        )
    ):
        key = (
            _h3_cell(lat, lon, config.h3_resolution),
            _normalized(language),
            _normalized(tag),
        )
        groups[key].append(index)
    allocations = _allocation(
        {key: len(indexes) for key, indexes in groups.items()},
        config.target,
        config.seed,
    )
    selected: list[int] = []
    for key, indexes in groups.items():
        ranked = sorted(
            indexes,
            key=lambda index: (_rank(config.seed, str(ids[index])), index),
        )
        selected.extend(ranked[: allocations.get(key, 0)])
    return table.take(pa.array(sorted(selected), type=pa.int64()))


def select_label_rows(
    table: pa.Table,
    *,
    row_limit: int,
    sampling_target: int | None,
    sampling_seed: str | None,
    h3_resolution: int | None,
) -> pa.Table:
    """Apply the legacy canary path or the default stratified label path."""

    if row_limit:
        from .canary import select_canary_rows

        return select_canary_rows(table, row_limit)
    if sampling_target is None or sampling_target == 0:
        return table
    return select_stratified_rows(
        table,
        target=sampling_target,
        seed=sampling_seed or DEFAULT_SAMPLE_SEED,
        h3_resolution=(
            h3_resolution if h3_resolution is not None else DEFAULT_H3_RESOLUTION
        ),
    )


__all__ = [
    "DEFAULT_H3_RESOLUTION",
    "DEFAULT_SAMPLE_SEED",
    "DEFAULT_SAMPLE_TARGET",
    "MISSING_STRATUM",
    "SAMPLING_VERSION",
    "SamplingConfig",
    "select_label_rows",
    "select_stratified_rows",
]
