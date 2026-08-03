"""Deterministic descriptive analytics for the final label table."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

LABELS: tuple[str, ...] = ("yes", "no", "uncertain")
SLICE_DIMENSIONS: tuple[str, ...] = ("language", "source", "osm_primary_tag")
MIN_SLICE_ROWS = 100
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


def _counts(values: list[object]) -> dict[str, int]:
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


def _render_slice_html(analytics: LabelAnalytics, path: Path) -> None:
    data = json.dumps([item.to_dict() for item in analytics.slices], sort_keys=True)
    dimensions = json.dumps(list(SLICE_DIMENSIONS))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Strong-positive yield by slice</title>
<style>body{{font:15px system-ui,sans-serif;color:#17324d;margin:2rem;max-width:70rem}}\nlabel{{font-weight:600}} select{{margin-left:.5rem;padding:.3rem}} table{{border-collapse:collapse;margin-top:1rem;width:100%}} th,td{{border-bottom:1px solid #d7e1e8;padding:.5rem;text-align:left}} th{{background:#edf4f7}} .note{{color:#526875}}</style>
</head><body><h1>Strong-positive yield by slice</h1>
<p class="note">Only groups with at least {MIN_SLICE_ROWS} sentences are shown. Rates are computed from the final labeled table.</p>
<label for="dimension">Slice dimension</label><select id="dimension"></select>
<table><thead><tr><th>Value</th><th>Both yes</th><th>Uncertain</th><th>Sample size</th></tr></thead><tbody id="rows"></tbody></table>
<script>const SLICES = {data}; const DIMENSIONS = {dimensions}; const SLICE_FIELDS = ["both_yes_rate","uncertain_rate","sample_size"]; const select = document.getElementById('dimension'); const rows = document.getElementById('rows');
DIMENSIONS.forEach((d) => {{ const option=document.createElement('option'); option.value=d; option.textContent=d; select.appendChild(option); }});
function render() {{ rows.replaceChildren(); SLICES.filter((item) => item.dimension === select.value).forEach((item) => {{ const row=document.createElement('tr'); [item.value, (item.both_yes_rate*100).toFixed(2)+'%', (item.uncertain_rate*100).toFixed(2)+'%', item.sample_size.toLocaleString()].forEach((value) => {{ const cell=document.createElement('td'); cell.textContent=value; row.appendChild(cell); }}); rows.appendChild(row); }}); }}
select.addEventListener('change', render); if (DIMENSIONS.length) {{ select.value = DIMENSIONS[0]; render(); }}
</script></body></html>"""
    path.write_text(document)


def render_analytics_assets(analytics: LabelAnalytics, assets: Path) -> None:
    """Render all requested visual assets into an existing staging directory."""

    assets.mkdir(mode=0o700, parents=True, exist_ok=True)
    _render_heatmap(analytics, assets / "joint_label_heatmap.png")
    _render_funnel(analytics, assets / "polygon_coverage_funnel.png")
    _render_reasons(analytics, assets / "reason_code_distribution.png")
    _render_slice_html(analytics, assets / "slice_yield.html")
    for path in assets.iterdir():
        os.chmod(path, 0o600)


__all__ = [
    "LABELS",
    "MIN_SLICE_ROWS",
    "SLICE_DIMENSIONS",
    "LabelAnalytics",
    "SliceYield",
    "build_label_analytics",
    "render_analytics_assets",
]
