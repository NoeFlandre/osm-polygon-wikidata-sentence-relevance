from __future__ import annotations

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.labeling.sampling import (
    DEFAULT_H3_RESOLUTION,
    DEFAULT_SAMPLE_TARGET,
    SAMPLING_VERSION,
    SamplingConfig,
    _h3_cell,
    _normalized,
    select_label_rows,
    select_stratified_rows,
)


def _table() -> pa.Table:
    rows: list[dict[str, object]] = []
    for index in range(48):
        rows.append(
            {
                "sentence_id": f"s-{index:03d}",
                "lat": 34.0 + (index % 6) * 8.0,
                "lon": -120.0 + (index % 8) * 35.0,
                "language": ("en", "fr", "ar")[index % 3],
                "osm_primary_tag": ("landuse", "natural")[index % 2],
            }
        )
    return pa.Table.from_pylist(rows)


def test_stratified_selection_is_deterministic_bounded_and_ordered() -> None:
    table = _table()
    first = select_stratified_rows(table, target=17, seed="test-seed")
    second = select_stratified_rows(table, target=17, seed="test-seed")

    assert first.equals(second)
    assert first.num_rows == 17
    ids = first["sentence_id"].to_pylist()
    assert ids == sorted(ids, key=lambda value: int(value.split("-")[1]))


def test_joint_strata_cover_each_requested_dimension_when_budget_allows() -> None:
    selected = select_stratified_rows(_table(), target=24, seed="coverage")

    assert len(set(selected["language"].to_pylist())) == 3
    assert len(set(selected["osm_primary_tag"].to_pylist())) == 2
    assert len(set(selected["lat"].to_pylist())) >= 4


def test_larger_target_is_a_nested_proportional_continuation() -> None:
    table = _table()
    first = select_stratified_rows(table, target=17, seed="continuation")
    larger = select_stratified_rows(table, target=31, seed="continuation")

    assert set(first["sentence_id"].to_pylist()).issubset(
        set(larger["sentence_id"].to_pylist())
    )


def test_continuation_allocates_each_joint_stratum_proportionally() -> None:
    rows = [
        {
            "sentence_id": f"major-{index:02d}",
            "lat": 0.0,
            "lon": 0.0,
            "language": "en",
            "osm_primary_tag": "landuse",
        }
        for index in range(30)
    ] + [
        {
            "sentence_id": f"minor-{index:02d}",
            "lat": 45.0,
            "lon": 45.0,
            "language": "fr",
            "osm_primary_tag": "natural",
        }
        for index in range(10)
    ]
    selected = select_stratified_rows(
        pa.Table.from_pylist(rows), target=20, seed="proportional"
    )

    selected_ids = selected["sentence_id"].to_pylist()
    major = sum(value.startswith("major-") for value in selected_ids)
    minor = sum(value.startswith("minor-") for value in selected_ids)
    assert (major, minor) == (15, 5)


def test_target_above_input_returns_all_rows() -> None:
    table = _table()
    selected = select_stratified_rows(table, target=DEFAULT_SAMPLE_TARGET)
    assert selected.equals(table)


def test_zero_target_is_the_explicit_full_input_compatibility_mode() -> None:
    table = _table()
    assert select_label_rows(
        table,
        row_limit=0,
        sampling_target=0,
        sampling_seed=None,
        h3_resolution=None,
    ).equals(table)


def test_nonzero_sampling_target_uses_v2_selector() -> None:
    selected = select_label_rows(
        _table(),
        row_limit=0,
        sampling_target=17,
        sampling_seed="seed",
        h3_resolution=DEFAULT_H3_RESOLUTION,
    )
    assert selected.num_rows == 17


def test_canary_selection_precedes_stratified_selection() -> None:
    table = _table()
    table = table.append_column("source", pa.array(["wikipedia"] * table.num_rows))
    table = table.append_column("region", pa.array(["afghanistan"] * table.num_rows))
    selected = select_label_rows(
        table,
        row_limit=5,
        sampling_target=1,
        sampling_seed="seed",
        h3_resolution=3,
    )
    assert selected.num_rows == 5


def test_sampling_config_rejects_invalid_types_and_version() -> None:
    invalid = (
        {"target": True},
        {"target": 1.5},
        {"seed": " "},
        {"seed": None},
        {"h3_resolution": True},
        {"h3_resolution": 1.5},
        {"version": ""},
        {"version": None},
    )
    for values in invalid:
        with pytest.raises(ValueError, match="sampling|H3"):
            SamplingConfig(**values)  # type: ignore[arg-type]


def test_missing_values_are_normalized_to_an_explicit_stratum() -> None:
    assert _normalized(None) == "(missing)"
    assert _normalized(" ") == "(missing)"
    assert _normalized(" en ") == "en"


@pytest.mark.parametrize(
    ("lat", "lon", "message"),
    [
        (True, 0.0, "numeric"),
        (0.0, False, "numeric"),
        (object(), 0.0, "numeric"),
        ("not-a-number", 0.0, "numeric"),
        (float("nan"), 0.0, "finite"),
        (0.0, float("inf"), "finite"),
        (91.0, 0.0, "outside"),
        (0.0, 181.0, "outside"),
    ],
)
def test_h3_cell_rejects_invalid_coordinates(
    lat: object, lon: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _h3_cell(lat, lon, DEFAULT_H3_RESOLUTION)


def test_h3_cell_wraps_h3_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    h3 = pytest.importorskip("h3")
    monkeypatch.setattr(
        h3,
        "latlng_to_cell",
        lambda *_args: (_ for _ in ()).throw(h3.H3ValueError("bad cell")),
    )
    with pytest.raises(ValueError, match="H3 cell"):
        _h3_cell(0.0, 0.0, DEFAULT_H3_RESOLUTION)


def test_duplicate_ids_are_rejected() -> None:
    table = pa.table({"sentence_id": ["same", "same"]})
    with pytest.raises(ValueError, match="duplicate"):
        select_stratified_rows(table, target=1)


def test_missing_columns_are_rejected_when_sampling_is_needed() -> None:
    table = pa.table({"sentence_id": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="required columns"):
        select_stratified_rows(table, target=2)


def test_small_input_without_coordinate_columns_is_accepted() -> None:
    table = pa.table({"sentence_id": ["a", "b"]})
    assert select_stratified_rows(table, target=2).equals(table)


def test_missing_coordinates_are_retained_in_explicit_missing_stratum() -> None:
    table = _table()
    table = table.set_column(
        1, "lat", pa.array([None] + table["lat"].to_pylist()[1:], type=pa.float64())
    )
    selected = select_stratified_rows(table, target=48)
    assert selected.num_rows == table.num_rows
    assert selected["sentence_id"][0].as_py() == "s-000"


@pytest.mark.parametrize(
    ("target", "seed", "resolution"),
    [
        (0, "seed", DEFAULT_H3_RESOLUTION),
        (-1, "seed", DEFAULT_H3_RESOLUTION),
        (1, "", 3),
        (1, "seed", 16),
    ],
)
def test_invalid_sampling_configuration_is_rejected(
    target: int, seed: str, resolution: int
) -> None:
    with pytest.raises(ValueError, match="sampling|H3"):
        select_stratified_rows(
            _table(), target=target, seed=seed, h3_resolution=resolution
        )


def test_invalid_coordinate_is_rejected() -> None:
    table = _table()
    table = table.set_column(
        1, "lat", pa.array([91.0] + table["lat"].to_pylist()[1:], type=pa.float64())
    )
    with pytest.raises(ValueError, match="coordinate"):
        select_stratified_rows(table, target=4)


def test_public_sampling_contract_is_pinned() -> None:
    assert DEFAULT_H3_RESOLUTION == 3
    assert DEFAULT_SAMPLE_TARGET == 200_000
    assert SAMPLING_VERSION == "labeling-v2-h3-language-osm-primary"
