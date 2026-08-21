"""Tests for schema-validated Parquet loading.

Uses tiny temporary Parquet files.  No network, no external data.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.errors import (
    MissingColumnsError,
)
from osm_polygon_sentence_relevance.schemas import POLYGONS_SCHEMA
from tests.helpers import make_polygon_row, rows_to_table


def _write_polygons_parquet(
    tmp_path: Path, rows: list[dict[str, list]] | None = None
) -> Path:
    """Write a minimal polygons Parquet file and return its path."""
    if rows is None:
        rows = [make_polygon_row()]
    table = rows_to_table(rows, POLYGONS_SCHEMA)
    fpath = tmp_path / "test_polygons.parquet"
    pq.write_table(table, fpath)
    return fpath


class TestValidLoad:
    """A valid Parquet file matching the contract is loaded correctly."""

    def test_load_returns_table(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.loading import load_validated_table

        fpath = _write_polygons_parquet(tmp_path)
        result = load_validated_table("polygons", fpath)
        assert isinstance(result, pa.Table)
        assert result.num_rows == 1

    def test_all_columns_present(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.loading import load_validated_table

        fpath = _write_polygons_parquet(tmp_path)
        result = load_validated_table("polygons", fpath)
        expected_names = {f.name for f in POLYGONS_SCHEMA}
        actual_names = set(result.column_names)
        assert expected_names == actual_names


class TestSchemaMismatch:
    """A Parquet file with a bad schema raises SchemaContractError."""

    def test_missing_column_raises(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.loading import load_validated_table

        # Write a file missing a column
        row = make_polygon_row()
        del row["polygon_id"]
        truncated_schema = pa.schema(
            [f for f in POLYGONS_SCHEMA if f.name != "polygon_id"]
        )
        table = rows_to_table([row], truncated_schema)
        fpath = tmp_path / "bad.parquet"
        pq.write_table(table, fpath)

        with pytest.raises(MissingColumnsError):
            load_validated_table("polygons", fpath)


class TestProjection:
    """Column projection returns only requested columns."""

    def test_projection_returns_subset(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.loading import load_validated_table

        fpath = _write_polygons_parquet(tmp_path)
        result = load_validated_table(
            "polygons", fpath, columns=("polygon_id", "region")
        )
        assert result.column_names == ["polygon_id", "region"]
        assert result.num_rows == 1

    def test_unknown_projection_column_raises(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.loading import load_validated_table

        fpath = _write_polygons_parquet(tmp_path)
        with pytest.raises(ValueError, match="nonexistent_col"):
            load_validated_table(
                "polygons", fpath, columns=("polygon_id", "nonexistent_col")
            )

    def test_unknown_projection_is_rejected_before_table_read(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from osm_polygon_sentence_relevance.ingestion import loading

        monkeypatch.setattr(
            loading.pq,
            "read_schema",
            lambda _path: pa.schema([pa.field("known", pa.string())]),
        )
        monkeypatch.setattr(loading, "validate_table_schema", lambda *_args: None)

        def unexpected_read(*_args, **_kwargs):
            raise AssertionError("table read must not happen for an unknown column")

        monkeypatch.setattr(loading.pq, "read_table", unexpected_read)

        with pytest.raises(ValueError, match="Unknown projection columns") as exc_info:
            loading.load_validated_table(
                "polygons",
                Path("unused.parquet"),
                columns=("unknown",),
            )
        assert str(exc_info.value) == (
            "Unknown projection columns for 'polygons': ['unknown']"
        )


class TestEmptyColumnsProjection:
    """Projection to an empty column set yields a zero-column table."""

    def test_empty_columns_projection(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.loading import load_validated_table

        fpath = tmp_path / "polygons.parquet"
        table = rows_to_table([make_polygon_row(polygon_id="poly-1")], POLYGONS_SCHEMA)
        pq.write_table(table, fpath)

        # Calling load_validated_table with columns=() must return a table
        # with 0 columns and 1 row.
        res = load_validated_table("polygons", fpath, columns=())
        assert res.num_columns == 0
        assert res.num_rows == 1
