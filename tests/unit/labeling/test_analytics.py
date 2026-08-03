from __future__ import annotations

import json

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.labeling.analytics import (
    MIN_SLICE_ROWS,
    build_label_analytics,
    render_analytics_assets,
)


def _table(row_count: int = 6) -> pa.Table:
    rows = [
        {
            "polygon_id": "p1",
            "language": "en",
            "source": "wikipedia",
            "osm_primary_tag": "landuse=farmland",
            "landuse_relevance": "yes",
            "polygon_relevance": "yes",
            "landuse_reason": "explicit_land_use",
            "polygon_reason": "direct_polygon_reference",
        },
        {
            "polygon_id": "p1",
            "language": "en",
            "source": "wikipedia",
            "osm_primary_tag": "landuse=farmland",
            "landuse_relevance": "no",
            "polygon_relevance": "yes",
            "landuse_reason": "no_landuse_or_cover",
            "polygon_reason": "direct_polygon_reference",
        },
        {
            "polygon_id": "p2",
            "language": "fa",
            "source": "wikivoyage",
            "osm_primary_tag": "place=city",
            "landuse_relevance": "yes",
            "polygon_relevance": "no",
            "landuse_reason": "explicit_land_cover",
            "polygon_reason": "nearby_or_broader_area",
        },
        {
            "polygon_id": "p3",
            "language": None,
            "source": "wikipedia",
            "osm_primary_tag": None,
            "landuse_relevance": "uncertain",
            "polygon_relevance": "uncertain",
            "landuse_reason": "uncertain_context",
            "polygon_reason": "uncertain_context",
        },
        {
            "polygon_id": "p4",
            "language": "en",
            "source": "wikipedia",
            "osm_primary_tag": "place=city",
            "landuse_relevance": "no",
            "polygon_relevance": "no",
            "landuse_reason": "no_landuse_or_cover",
            "polygon_reason": "unrelated_fact",
        },
        {
            "polygon_id": "p5",
            "language": "fa",
            "source": "wikivoyage",
            "osm_primary_tag": "place=city",
            "landuse_relevance": "yes",
            "polygon_relevance": "yes",
            "landuse_reason": "explicit_land_use",
            "polygon_reason": "direct_polygon_reference",
        },
    ]
    return pa.Table.from_pylist(rows[:row_count])


def test_build_analytics_computes_kpis_heatmap_funnel_reasons_and_slices() -> None:
    analytics = build_label_analytics(_table())

    assert analytics.total_labeled_sentences == 6
    assert analytics.unique_polygons == 5
    assert analytics.unique_languages == 2
    assert analytics.strong_positive_count == 2
    assert analytics.strong_positive_yield == pytest.approx(2 / 6)
    assert analytics.joint_counts["yes|yes"] == 2
    assert analytics.joint_counts["no|uncertain"] == 0
    assert analytics.joint_percentages["yes|yes"] == pytest.approx(2 / 6)
    assert analytics.coverage_funnel == {
        "all_polygons": 5,
        "polygon_relevant_polygons": 2,
        "landuse_relevant_polygons": 3,
        "both_yes_polygons": 2,
    }
    assert analytics.landuse_reason_percentages["explicit_land_use"] == pytest.approx(
        2 / 6
    )
    assert analytics.polygon_reason_counts["direct_polygon_reference"] == 3
    assert analytics.slices == ()


def test_slice_threshold_and_uncertainty_are_deterministic() -> None:
    table = pa.Table.from_pylist(
        [
            {
                "polygon_id": f"p{i % 2}",
                "language": "en" if i < MIN_SLICE_ROWS else "fr",
                "source": "wikipedia",
                "osm_primary_tag": "landuse=farmland",
                "landuse_relevance": "yes" if i % 2 else "no",
                "polygon_relevance": "yes" if i % 3 else "uncertain",
                "landuse_reason": "x",
                "polygon_reason": "y",
            }
            for i in range(MIN_SLICE_ROWS + 1)
        ]
    )
    analytics = build_label_analytics(table)
    assert [(item.dimension, item.value) for item in analytics.slices] == [
        ("language", "en"),
        ("source", "wikipedia"),
        ("osm_primary_tag", "landuse=farmland"),
    ]
    language = analytics.slices[0]
    assert language.sample_size == MIN_SLICE_ROWS
    assert language.both_yes_rate == pytest.approx(33 / MIN_SLICE_ROWS)
    assert language.uncertain_rate == pytest.approx(34 / MIN_SLICE_ROWS)


def test_analytics_rejects_missing_or_invalid_label_columns() -> None:
    with pytest.raises(ValueError, match="missing required analytics columns"):
        build_label_analytics(pa.table({"polygon_id": ["p1"]}))
    invalid = _table().set_column(
        4,
        "landuse_relevance",
        pa.array(["yes", "bad", "yes", "uncertain", "no", "yes"]),
    )
    with pytest.raises(ValueError, match="invalid landuse_relevance"):
        build_label_analytics(invalid)


def test_render_analytics_assets_writes_pngs_and_selector_html(tmp_path) -> None:
    analytics = build_label_analytics(_table())
    render_analytics_assets(analytics, tmp_path)
    for name in (
        "joint_label_heatmap.png",
        "polygon_coverage_funnel.png",
        "reason_code_distribution.png",
    ):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG")
    html = (tmp_path / "slice_yield.html").read_text()
    assert 'id="dimension"' in html
    assert (
        'const SLICE_FIELDS = ["both_yes_rate","uncertain_rate","sample_size"]' in html
    )
    assert '"sample_size"' in html
    json.loads(html.split("const SLICES = ", 1)[1].split(";", 1)[0])
