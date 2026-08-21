from __future__ import annotations

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.labeling.v2_analytics import build_v2_analytics


def _table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "polygon_id": "p1",
                "language": "en",
                "place_relevance": "yes",
                "area_bucket": "large",
                "lat": 45.0,
                "lon": 2.0,
            },
            {
                "polygon_id": "p2",
                "language": "fr",
                "place_relevance": "no",
                "area_bucket": "small",
                "lat": None,
                "lon": None,
            },
        ]
    )


def test_v2_analytics_is_derived_from_rows() -> None:
    analytics = build_v2_analytics(_table(), h3_resolution=3)
    assert analytics.total_labeled_sentences == 2
    assert analytics.unique_polygons == 2
    assert analytics.unique_languages == 2
    assert analytics.place_counts == {"no": 1, "yes": 1}
    assert analytics.place_percentages == {"no": 0.5, "yes": 0.5}
    assert analytics.missing_coordinate_count == 1
    assert analytics.to_dict()["area_bucket_counts"] == {"large": 1, "small": 1}


def test_v2_analytics_rejects_missing_columns_and_invalid_labels() -> None:
    with pytest.raises(ValueError, match="missing"):
        build_v2_analytics(pa.table({"place_relevance": ["yes"]}), h3_resolution=3)
    invalid = _table().set_column(2, "place_relevance", pa.array(["maybe", "no"]))
    with pytest.raises(ValueError, match="invalid"):
        build_v2_analytics(invalid, h3_resolution=3)
