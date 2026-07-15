"""Tests for schema-validated Parquet loading.

Uses tiny temporary Parquet files.  No network, no external data.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.errors import (
    IncompatibleTypesError,
    MissingColumnsError,
    SchemaContractError,
)
from osm_polygon_sentence_relevance.schemas import POLYGONS_SCHEMA

from tests.helpers import make_polygon_row, rows_to_table


def _write_polygons_parquet(tmp_path: Path, rows: list[dict[str, list]] | None = None) -> Path:
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
        truncated_schema = pa.schema([f for f in POLYGONS_SCHEMA if f.name != "polygon_id"])
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
        result = load_validated_table("polygons", fpath, columns=("polygon_id", "region"))
        assert result.column_names == ["polygon_id", "region"]
        assert result.num_rows == 1

    def test_unknown_projection_column_raises(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.loading import load_validated_table

        fpath = _write_polygons_parquet(tmp_path)
        with pytest.raises(ValueError, match="nonexistent_col"):
            load_validated_table("polygons", fpath, columns=("polygon_id", "nonexistent_col"))
