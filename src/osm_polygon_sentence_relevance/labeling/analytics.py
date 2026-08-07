"""Deterministic descriptive analytics for the final label table."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

LABELS: tuple[str, ...] = ("yes", "no", "uncertain")
SLICE_DIMENSIONS: tuple[str, ...] = ("language", "source", "osm_primary_tag")
MIN_SLICE_ROWS = 100
H3_MAP_ASSET_NAME = "h3_sentence_distribution.png"
_REQUIRED_COLUMNS = frozenset(
    {
        "polygon_id",
        "language",
        "source",
        "osm_primary_tag",
        "landuse_relevance",
        "polygon_relevance",
        "landuse_reason",
        "polygon_reason",
    }
)


@dataclass(frozen=True, slots=True)
class SliceYield:
    """Yield facts for one sufficiently large categorical slice."""

    dimension: str
    value: str
    sample_size: int
    both_yes_rate: float
    uncertain_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LabelAnalytics:
    """All metrics and plot inputs derived from one finalized Parquet table."""

    total_labeled_sentences: int
    unique_polygons: int
    unique_languages: int
    strong_positive_count: int
    strong_positive_yield: float
    joint_counts: dict[str, int]
    joint_percentages: dict[str, float]
    coverage_funnel: dict[str, int]
    landuse_reason_counts: dict[str, int]
    landuse_reason_percentages: dict[str, float]
    polygon_reason_counts: dict[str, int]
    polygon_reason_percentages: dict[str, float]
    slices: tuple[SliceYield, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, stable representation for the manifest and card."""

        result = asdict(self)
        result["slices"] = [item.to_dict() for item in self.slices]
        return result


def _normalized(value: object) -> str:
    if value is None:
        return "(missing)"
    text = str(value).strip()
    return text or "(missing)"


def _column_values(table: pa.Table, name: str) -> list[object]:
    return table[name].to_pylist()


def _counts(values: Sequence[object]) -> dict[str, int]:
    return dict(sorted(Counter(_normalized(value) for value in values).items()))


def _percentages(counts: dict[str, int], total: int) -> dict[str, float]:
    return {
        key: (value / total if total else 0.0) for key, value in sorted(counts.items())
    }


def _validate_table(table: pa.Table) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(table.column_names))
    if missing:
        raise ValueError(f"missing required analytics columns: {missing}")
    for field in ("landuse_relevance", "polygon_relevance"):
        invalid = sorted(
            {
                _normalized(value)
                for value in _column_values(table, field)
                if _normalized(value) not in LABELS
            }
        )
        if invalid:
            raise ValueError(f"invalid {field}: {invalid}")


def _build_slices(
    table: pa.Table,
    landuse: list[str],
    polygon: list[str],
) -> tuple[SliceYield, ...]:
    result: list[SliceYield] = []
    for dimension in SLICE_DIMENSIONS:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(_column_values(table, dimension)):
            groups[_normalized(value)].append(index)
        for value in sorted(groups):
            indexes = groups[value]
            sample_size = len(indexes)
            if sample_size < MIN_SLICE_ROWS:
                continue
            both_yes = sum(
                landuse[index] == "yes" and polygon[index] == "yes" for index in indexes
            )
            uncertain = sum(
                landuse[index] == "uncertain" or polygon[index] == "uncertain"
                for index in indexes
            )
            result.append(
                SliceYield(
                    dimension=dimension,
                    value=value,
                    sample_size=sample_size,
                    both_yes_rate=both_yes / sample_size,
                    uncertain_rate=uncertain / sample_size,
                )
            )
    return tuple(result)


def build_label_analytics(table: pa.Table) -> LabelAnalytics:
    """Compute every public metric directly from the finalized table."""

    _validate_table(table)
    total = table.num_rows
    landuse = [
        _normalized(value) for value in _column_values(table, "landuse_relevance")
    ]
    polygon = [
        _normalized(value) for value in _column_values(table, "polygon_relevance")
    ]
    joint_counts = {
        f"{land}|{poly}": sum(
            left == land and right == poly
            for left, right in zip(landuse, polygon, strict=True)
        )
        for land in LABELS
        for poly in LABELS
    }
    polygons = {
        _normalized(value)
        for value in _column_values(table, "polygon_id")
        if _normalized(value) != "(missing)"
    }
    languages = {
        _normalized(value)
        for value in _column_values(table, "language")
        if _normalized(value) != "(missing)"
    }
    polygon_values = _column_values(table, "polygon_id")
    polygon_relevant = {
        _normalized(polygon_values[index])
        for index, value in enumerate(polygon)
        if value == "yes" and _normalized(polygon_values[index]) != "(missing)"
    }
    landuse_relevant = {
        _normalized(polygon_values[index])
        for index, value in enumerate(landuse)
        if value == "yes" and _normalized(polygon_values[index]) != "(missing)"
    }
    both_yes_polygons = {
        _normalized(polygon_values[index])
        for index, (land, poly) in enumerate(zip(landuse, polygon, strict=True))
        if land == "yes"
        and poly == "yes"
        and _normalized(polygon_values[index]) != "(missing)"
    }
    strong_positive_count = joint_counts["yes|yes"]
    land_reasons = _counts(_column_values(table, "landuse_reason"))
    polygon_reasons = _counts(_column_values(table, "polygon_reason"))
    return LabelAnalytics(
        total_labeled_sentences=total,
        unique_polygons=len(polygons),
        unique_languages=len(languages),
        strong_positive_count=strong_positive_count,
        strong_positive_yield=strong_positive_count / total if total else 0.0,
        joint_counts=joint_counts,
        joint_percentages=_percentages(joint_counts, total),
        coverage_funnel={
            "all_polygons": len(polygons),
            "polygon_relevant_polygons": len(polygon_relevant),
            "landuse_relevant_polygons": len(landuse_relevant),
            "both_yes_polygons": len(both_yes_polygons),
        },
        landuse_reason_counts=land_reasons,
        landuse_reason_percentages=_percentages(land_reasons, total),
        polygon_reason_counts=polygon_reasons,
        polygon_reason_percentages=_percentages(polygon_reasons, total),
        slices=_build_slices(table, landuse, polygon),
    )


def h3_sentence_distribution(
    table: pa.Table, *, resolution: int
) -> tuple[dict[str, int], int]:
    """Count labeled sentences per H3 cell and report missing coordinates.

    Counts are derived from every finalized row. Rows without coordinates are
    retained in the returned missing count and are not assigned an invented
    geographic cell.
    """

    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, int)
        or not 0 <= resolution <= 15
    ):
        raise ValueError("H3 resolution must be an integer in [0, 15]")
    if "lat" not in table.column_names or "lon" not in table.column_names:
        return {}, table.num_rows
    try:
        import h3
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("install the operator extra to render the H3 map") from exc
    counts: Counter[str] = Counter()
    missing = 0
    for lat, lon in zip(
        table["lat"].to_pylist(), table["lon"].to_pylist(), strict=True
    ):
        if lat is None or lon is None:
            missing += 1
            continue
        try:
            cell = str(h3.latlng_to_cell(float(lat), float(lon), resolution))
        except (TypeError, ValueError, h3.H3ValueError) as exc:
            raise ValueError("finalized coordinates cannot be mapped to H3") from exc
        counts[cell] += 1
    return dict(sorted(counts.items())), missing


def _plotter() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional extra boundary
        raise RuntimeError("install the hub extra to render analytics") from exc
    return plt


def _save_png(fig: Any, path: Path) -> None:
    fig.savefig(path, metadata={"Software": ""}, bbox_inches="tight")
    fig.clf()


def _render_heatmap(analytics: LabelAnalytics, path: Path) -> None:
    plt = _plotter()
    fig, axis = plt.subplots(figsize=(8, 6), dpi=140)
    matrix = [
        [analytics.joint_counts[f"{land}|{polygon}"] for polygon in LABELS]
        for land in LABELS
    ]
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, label="Sentences")
    axis.set_xticks(range(3), LABELS)
    axis.set_yticks(range(3), LABELS)
    axis.set_xlabel("Polygon relevance")
    axis.set_ylabel("Land use / land cover relevance")
    axis.set_title("Joint label counts and share of all sentences")
    for row, land in enumerate(LABELS):
        for column, polygon in enumerate(LABELS):
            key = f"{land}|{polygon}"
            axis.text(
                column,
                row,
                f"{analytics.joint_counts[key]:,}\n"
                f"{analytics.joint_percentages[key] * 100:.1f}%",
                ha="center",
                va="center",
                color="white"
                if matrix[row][column] > max(max(row) for row in matrix) / 2
                else "#17324d",
            )
    fig.tight_layout()
    _save_png(fig, path)
    plt.close(fig)


def _render_funnel(analytics: LabelAnalytics, path: Path) -> None:
    plt = _plotter()
    labels = (
        "All polygons",
        "Polygon-relevant",
        "Land-use-relevant",
        "Both yes",
    )
    values = list(analytics.coverage_funnel.values())
    fig, axis = plt.subplots(figsize=(9, 5), dpi=140)
    bars = axis.bar(labels, values, color=("#1f6f8b", "#278ea5", "#4fb3bf", "#8fd5c5"))
    axis.set_ylabel("Unique polygons")
    axis.set_title("Polygon coverage funnel")
    axis.bar_label(bars, labels=[f"{value:,}" for value in values], padding=3)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_png(fig, path)
    plt.close(fig)


def _render_reasons(analytics: LabelAnalytics, path: Path) -> None:
    plt = _plotter()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=140)
    datasets = (
        ("Land-use / land-cover reasons", analytics.landuse_reason_percentages),
        ("Polygon-relevance reasons", analytics.polygon_reason_percentages),
    )
    for axis, (title, values) in zip(axes, datasets, strict=True):
        names = list(values)
        amounts = [values[name] * 100 for name in names]
        bars = axis.barh(names, amounts, color="#2878b5")
        axis.set_title(title)
        axis.set_xlabel("Share of sentences (%)")
        axis.bar_label(bars, labels=[f"{amount:.1f}%" for amount in amounts], padding=3)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_png(fig, path)
    plt.close(fig)


def render_h3_sentence_distribution(
    table: pa.Table,
    path: Path,
    *,
    resolution: int,
    scope_label: str,
) -> dict[str, Any]:
    """Render a deterministic hexagon map of sentence coordinates.

    The returned summary is written into the manifest by finalization, so the
    map's resolution, cell counts, and unlocated-row count are auditable from
    the same Parquet input.
    """

    counts, missing = h3_sentence_distribution(table, resolution=resolution)
    plt = _plotter()
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon

    fig, axis = plt.subplots(figsize=(14, 7), dpi=140)
    patches: list[Any] = []
    values: list[int] = []
    try:
        import h3
    except ImportError as exc:  # pragma: no cover - guarded above
        raise RuntimeError("install the operator extra to render the H3 map") from exc
    for cell, count in counts.items():
        boundary = h3.cell_to_boundary(cell)
        longitudes = [float(point[1]) for point in boundary]
        latitudes = [float(point[0]) for point in boundary]
        if max(longitudes) - min(longitudes) > 180:
            longitudes = [
                longitude - 360 if longitude > 0 else longitude
                for longitude in longitudes
            ]
        patches.append(
            Polygon(list(zip(longitudes, latitudes, strict=True)), closed=True)
        )
        values.append(count)
    if patches:
        collection = PatchCollection(
            patches,
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=max(values)),
            linewidth=0.15,
            edgecolor="#ffffff",
        )
        collection.set_array(values)
        axis.add_collection(collection)
        fig.colorbar(collection, ax=axis, label="Labeled sentences per H3 cell")
    axis.set_xlim(-180, 180)
    axis.set_ylim(-90, 90)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(
        f"{scope_label} labeled-sentence distribution by H3 cell (r{resolution})"
    )
    axis.grid(color="#d8e1e8", linewidth=0.4, alpha=0.7)
    fig.tight_layout()
    _save_png(fig, path)
    plt.close(fig)
    return {
        "resolution": resolution,
        "cell_count": len(counts),
        "sentence_count": sum(counts.values()),
        "missing_coordinate_count": missing,
        "cells": counts,
    }


def render_analytics_assets(analytics: LabelAnalytics, assets: Path) -> None:
    """Render all requested visual assets into an existing staging directory."""

    assets.mkdir(mode=0o700, parents=True, exist_ok=True)
    _render_heatmap(analytics, assets / "joint_label_heatmap.png")
    _render_funnel(analytics, assets / "polygon_coverage_funnel.png")
    _render_reasons(analytics, assets / "reason_code_distribution.png")
    for path in assets.iterdir():
        os.chmod(path, 0o600)


__all__ = [
    "H3_MAP_ASSET_NAME",
    "LABELS",
    "MIN_SLICE_ROWS",
    "SLICE_DIMENSIONS",
    "LabelAnalytics",
    "SliceYield",
    "build_label_analytics",
    "h3_sentence_distribution",
    "render_h3_sentence_distribution",
    "render_analytics_assets",
]
