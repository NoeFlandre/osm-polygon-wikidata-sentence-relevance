from __future__ import annotations

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.labeling.canary import (
    select_canary_rows,
    select_v2_canary_rows,
)


def _table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sentence_id": f"s-{index:02d}",
                "source": source,
                "language": language,
                "region": "afghanistan",
            }
            for index, (source, language) in enumerate(
                [
                    ("wikipedia", "en"),
                    ("wikipedia", "fa"),
                    ("wikivoyage", "en"),
                    ("wikipedia", "ps"),
                    ("wikivoyage", "fr"),
                    ("wikipedia", "ar"),
                    ("wikipedia", "uz"),
                    ("wikipedia", "de"),
                ]
            )
        ]
    )


def test_zero_limit_keeps_full_input() -> None:
    table = _table()
    assert select_canary_rows(table, 0).equals(table)


def test_canary_is_deterministic_bounded_and_preserves_input_order() -> None:
    first = select_canary_rows(_table(), 5)
    second = select_canary_rows(_table(), 5)

    assert first.equals(second)
    assert first.num_rows == 5
    indexes = [int(value.split("-")[1]) for value in first["sentence_id"].to_pylist()]
    assert indexes == sorted(indexes)
    assert set(first["source"].to_pylist()) == {"wikipedia", "wikivoyage"}


def test_canary_prioritizes_language_coverage() -> None:
    selected = select_canary_rows(_table(), 6)
    assert len(set(selected["language"].to_pylist())) == 6


@pytest.mark.parametrize("limit", [-1, True, 8, 9])
def test_canary_rejects_invalid_or_non_partial_limit(limit: object) -> None:
    with pytest.raises(ValueError, match="row limit"):
        select_canary_rows(_table(), limit)  # type: ignore[arg-type]


def test_canary_rejects_missing_required_columns() -> None:
    table = _table().drop(["region"])

    with pytest.raises(ValueError, match="missing required columns"):
        select_canary_rows(table, 2)


def test_canary_rejects_non_afghanistan_rows() -> None:
    base = _table()
    table = base.set_column(
        base.column_names.index("region"),
        "region",
        pa.array(["france"] * base.num_rows),
    )

    with pytest.raises(ValueError, match="only Afghanistan"):
        select_canary_rows(table, 2)


def test_v2_canary_accepts_worldwide_rows_and_is_deterministic() -> None:
    table = (
        _table()
        .set_column(
            0,
            "sentence_id",
            pa.array([f"world-{index}" for index in range(8)]),
        )
        .drop(["region"])
    )

    first = select_v2_canary_rows(table, 3)
    second = select_v2_canary_rows(table, 3)

    assert first.equals(second)
    assert first.num_rows == 3
    assert set(first.column_names) == set(table.column_names)
