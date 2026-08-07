from __future__ import annotations

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.labeling import v2_sampling
from osm_polygon_sentence_relevance.labeling.v2_sampling import (
    AREA_BUCKETS,
    V2_H3_RESOLUTION,
    canonical_area_bucket,
    select_v2_rows,
)


def _table() -> pa.Table:
    rows: list[dict[str, object]] = []
    for bucket, area in (
        ("tiny", 0.05),
        ("small", 0.5),
        ("medium", 5.0),
        ("large", 20.0),
    ):
        for index in range(4):
            rows.append(
                {
                    "sentence_id": f"{bucket}-{index}",
                    "polygon_id": f"{bucket}-polygon-{index // 2}",
                    "lat": 45.0 + index // 2,
                    "lon": 2.0 + index // 2,
                    "area_km2": area,
                    "area_bucket": bucket,
                    "language": "en" if index % 2 else "fr",
                    "osm_primary_tag": "landuse=forest"
                    if index % 2
                    else "natural=wood",
                }
            )
    return pa.Table.from_pylist(rows)


def test_v2_uses_exact_area_buckets_and_resolution_three() -> None:
    assert {
        "tiny": (0.0, 0.1),
        "small": (0.1, 1.0),
        "medium": (1.0, 10.0),
        "large": (10.0, float("inf")),
    } == AREA_BUCKETS
    assert V2_H3_RESOLUTION == 3


@pytest.mark.parametrize(
    ("area", "source_bucket", "canonical"),
    [
        (0.00005, "<100m2", "tiny"),
        (0.0005, "100m2-1k_m2", "tiny"),
        (0.005, "1k_m2-10k_m2", "tiny"),
        (0.05, "10k_m2-100k_m2", "tiny"),
        (0.5, "0.1-1km2", "small"),
        (5.0, "1-10km2", "medium"),
        (10.0, "10-100km2", "large"),
        (125.0, ">100km2", "large"),
    ],
)
def test_v2_canonicalizes_all_upstream_area_labels(
    area: float, source_bucket: str, canonical: str
) -> None:
    assert canonical_area_bucket(area, source_bucket) == canonical


def test_v2_keeps_large_polygons_and_samples_other_buckets_deterministically() -> None:
    first = select_v2_rows(_table(), target=6, seed="seed")
    second = select_v2_rows(_table(), target=6, seed="seed")
    assert first["sentence_id"].to_pylist() == second["sentence_id"].to_pylist()
    selected = set(first["sentence_id"].to_pylist())
    assert any(value.startswith("large-") for value in selected)
    assert len(selected) == 6


def test_v2_orders_rows_across_language_and_primary_tag_strata() -> None:
    selected = select_v2_rows(_table(), target=8, seed="seed")
    strata = set(
        zip(
            selected["language"].to_pylist(),
            selected["osm_primary_tag"].to_pylist(),
            strict=True,
        )
    )
    assert strata == {("en", "landuse=forest"), ("fr", "natural=wood")}


def test_v2_larger_target_is_a_prefix_of_smaller_target() -> None:
    smaller = select_v2_rows(_table(), target=5, seed="seed")
    larger = select_v2_rows(_table(), target=7, seed="seed")
    assert larger["sentence_id"].to_pylist()[:5] == smaller["sentence_id"].to_pylist()


def test_v2_rejects_area_bucket_mismatch_and_missing_columns() -> None:
    table = _table().set_column(4, "area_km2", pa.array([0.2] * _table().num_rows))
    with pytest.raises(ValueError, match="area_bucket"):
        select_v2_rows(table, target=2, seed="seed")
    with pytest.raises(ValueError, match="required"):
        select_v2_rows(pa.table({"sentence_id": ["s"]}), target=1, seed="seed")


@pytest.mark.parametrize("target", [0, -1, True, "1"])
def test_v2_rejects_invalid_targets(target: object) -> None:
    with pytest.raises(ValueError, match="target"):
        select_v2_rows(_table(), target=target, seed="seed")  # type: ignore[arg-type]


def test_v2_rejects_duplicate_ids_and_inconsistent_polygon_metadata() -> None:
    duplicate = _table().set_column(
        0, "sentence_id", pa.array(["same"] * _table().num_rows)
    )
    with pytest.raises(ValueError, match="duplicate"):
        select_v2_rows(duplicate, target=1, seed="seed")

    inconsistent = _table().set_column(
        1,
        "polygon_id",
        pa.array(["p"] + ["tiny-polygon"] * (_table().num_rows - 1)),
    )
    with pytest.raises(ValueError, match="inconsistent"):
        select_v2_rows(inconsistent, target=1, seed="seed")


@pytest.mark.parametrize(
    ("area", "bucket", "message"),
    [
        ("not-a-number", "tiny", "numeric"),
        (0.1, "tiny", "agree"),
        (1.0, "unknown", "invalid"),
        (object(), "tiny", "numeric"),
    ],
)
def test_v2_rejects_invalid_area_metadata(
    area: object, bucket: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        v2_sampling._bucket(area, bucket)


@pytest.mark.parametrize(
    "coordinates",
    [
        ("not-a-number", 1.0),
        (float("nan"), 1.0),
        (1.0, object()),
    ],
)
def test_v2_rejects_invalid_coordinates(coordinates: tuple[object, object]) -> None:
    with pytest.raises(ValueError, match="coordinates"):
        v2_sampling._cell(*coordinates)


def test_v2_treats_missing_coordinates_and_text_strata_as_stable_values() -> None:
    assert v2_sampling._cell(None, None) == "(missing)"
    assert v2_sampling._normalized(None) == "(missing)"
    assert v2_sampling._normalized("  ") == "(missing)"
    assert v2_sampling._normalized(" en ") == "en"


def test_v2_wraps_h3_coordinate_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenH3:
        @staticmethod
        def latlng_to_cell(*_args: object) -> str:
            raise ValueError("outside")

    monkeypatch.setitem(__import__("sys").modules, "h3", BrokenH3())
    with pytest.raises(ValueError, match="outside the valid range"):
        v2_sampling._cell(91.0, 1.0)
