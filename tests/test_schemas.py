"""Tests for PyArrow schema contracts and validate_table_schema().

All tests use in-memory PyArrow tables.  No network, no disk data, no
credentials, no downloaded Parquet files.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.errors import (
    IncompatibleTypesError,
    MissingColumnsError,
    UnknownTableError,
)
from osm_polygon_sentence_relevance.schemas import (
    OUTPUT_SENTENCE_SCHEMA,
    SCHEMA_REGISTRY,
    SECTIONS_SCHEMA,
    validate_table_schema,
)


# ===================================================================
# Helpers – minimal Afghanistan-shaped rows
# ===================================================================

def _minimal_row(schema: pa.Schema) -> dict[str, list]:
    """Return a single-row column dict with plausible dummy values."""
    row: dict[str, list] = {}
    for field in schema:
        if field.type == pa.string():
            row[field.name] = ["test"]
        elif field.type == pa.int64():
            row[field.name] = [1]
        elif field.type == pa.float64():
            row[field.name] = [1.0]
        elif field.type == pa.bool_():
            row[field.name] = [True]
        else:
            raise ValueError(f"Unhandled type {field.type} for {field.name}")
    return row


# ===================================================================
# Registry completeness
# ===================================================================

class TestRegistryCompleteness:
    """All six logical input tables must be registered."""

    EXPECTED_TABLES = {
        "polygons",
        "polygon_articles",
        "wikipedia_documents",
        "wikivoyage_documents",
        "wikipedia_sections",
        "wikivoyage_sections",
    }

    def test_all_six_input_tables_registered(self):
        assert set(SCHEMA_REGISTRY.keys()) == self.EXPECTED_TABLES

    def test_section_schemas_identical(self):
        assert SCHEMA_REGISTRY["wikipedia_sections"] is SCHEMA_REGISTRY["wikivoyage_sections"]
        assert SCHEMA_REGISTRY["wikipedia_sections"] == SECTIONS_SCHEMA


# ===================================================================
# Conforming rows
# ===================================================================

class TestConformingRows:
    """Afghanistan-shaped rows must pass validation."""

    @pytest.mark.parametrize("table_name", list(SCHEMA_REGISTRY.keys()))
    def test_minimal_row_conforms(self, table_name: str):
        schema = SCHEMA_REGISTRY[table_name]
        row = _minimal_row(schema)
        table = pa.table(row, schema=schema)
        # Should not raise.
        validate_table_schema(table_name, table.schema)


# ===================================================================
# Missing columns
# ===================================================================

class TestMissingColumns:
    """validate_table_schema must detect missing required columns."""

    def test_missing_single_column(self):
        schema = SCHEMA_REGISTRY["polygons"]
        # Drop the first column.
        truncated = pa.schema(list(schema)[1:])
        with pytest.raises(MissingColumnsError) as exc_info:
            validate_table_schema("polygons", truncated)
        assert schema.field(0).name in exc_info.value.missing

    def test_missing_multiple_columns(self):
        schema = SCHEMA_REGISTRY["polygon_articles"]
        # Keep only the first two columns.
        truncated = pa.schema(list(schema)[:2])
        with pytest.raises(MissingColumnsError) as exc_info:
            validate_table_schema("polygon_articles", truncated)
        assert len(exc_info.value.missing) > 0


# ===================================================================
# Incompatible types
# ===================================================================

class TestIncompatibleTypes:
    """validate_table_schema must detect type mismatches."""

    def test_wrong_type_detected(self):
        schema = SCHEMA_REGISTRY["polygons"]
        # Replace int64 osm_id with string.
        fields = []
        for f in schema:
            if f.name == "osm_id":
                fields.append(pa.field("osm_id", pa.string()))
            else:
                fields.append(f)
        bad_schema = pa.schema(fields)
        with pytest.raises(IncompatibleTypesError) as exc_info:
            validate_table_schema("polygons", bad_schema)
        cols = [m[0] for m in exc_info.value.mismatches]
        assert "osm_id" in cols


# ===================================================================
# Unknown table
# ===================================================================

class TestUnknownTable:
    def test_unknown_table_raises(self):
        with pytest.raises(UnknownTableError):
            validate_table_schema("nonexistent_table", pa.schema([]))


# ===================================================================
# Nullable Wikipedia thumbnail dimensions
# ===================================================================

class TestNullableThumbnails:
    """Null int64 values in thumbnail_width/height must be accepted."""

    def test_null_thumbnail_dimensions_accepted(self):
        schema = SCHEMA_REGISTRY["wikipedia_documents"]
        row = _minimal_row(schema)
        # Set thumbnail dimensions to None.
        row["thumbnail_width"] = [None]
        row["thumbnail_height"] = [None]
        table = pa.table(row, schema=schema)
        # Should not raise.
        validate_table_schema("wikipedia_documents", table.schema)


# ===================================================================
# Empty Wikivoyage article_id
# ===================================================================

class TestEmptyWikivoyageArticleId:
    """Empty-string article_id is legitimate for Wikivoyage."""

    def test_empty_article_id_accepted(self):
        schema = SCHEMA_REGISTRY["wikivoyage_documents"]
        row = _minimal_row(schema)
        row["article_id"] = [""]
        table = pa.table(row, schema=schema)
        # Should not raise (schema validates types only, not content).
        validate_table_schema("wikivoyage_documents", table.schema)

    def test_empty_article_id_in_sections(self):
        schema = SCHEMA_REGISTRY["wikivoyage_sections"]
        row = _minimal_row(schema)
        row["article_id"] = [""]
        table = pa.table(row, schema=schema)
        validate_table_schema("wikivoyage_sections", table.schema)


# ===================================================================
# Output schema structure
# ===================================================================

class TestOutputSchema:
    """The output sentence schema must match the spec exactly."""

    EXPECTED_COLUMNS = [
        "sentence_id",
        "polygon_id",
        "wikidata",
        "document_id",
        "article_id",
        "source",
        "language",
        "site",
        "page_title",
        "section_id",
        "section_index",
        "section_path",
        "sentence_index",
        "sentence_text_raw",
        "sentence_text_normalized",
        "previous_sentence",
        "next_sentence",
        "url",
        "page_id",
        "revision_id",
        "revision_timestamp",
        "document_content_hash",
        "section_content_hash",
        "sentence_content_hash",
        "duplicate_occurrence_count",
        "duplicate_sources",
        "polygon_name",
        "osm_primary_tag",
        "osm_tags",
        "region",
        "lat",
        "lon",
        "input_dataset_revision",
        "pipeline_version",
    ]

    def test_exact_columns_in_order(self):
        actual = [f.name for f in OUTPUT_SENTENCE_SCHEMA]
        assert actual == self.EXPECTED_COLUMNS

    def test_section_path_is_list_string(self):
        field = OUTPUT_SENTENCE_SCHEMA.field("section_path")
        assert field.type == pa.list_(pa.string())

    def test_osm_tags_is_map_string_string(self):
        field = OUTPUT_SENTENCE_SCHEMA.field("osm_tags")
        assert field.type == pa.map_(pa.string(), pa.string())

    def test_duplicate_sources_is_list_string(self):
        field = OUTPUT_SENTENCE_SCHEMA.field("duplicate_sources")
        assert field.type == pa.list_(pa.string())

    def test_paragraph_index_absent(self):
        names = [f.name for f in OUTPUT_SENTENCE_SCHEMA]
        assert "paragraph_index" not in names

    def test_article_id_nullable(self):
        field = OUTPUT_SENTENCE_SCHEMA.field("article_id")
        assert field.nullable is True

    def test_lat_lon_nullable(self):
        assert OUTPUT_SENTENCE_SCHEMA.field("lat").nullable is True
        assert OUTPUT_SENTENCE_SCHEMA.field("lon").nullable is True

    def test_polygon_name_nullable(self):
        assert OUTPUT_SENTENCE_SCHEMA.field("polygon_name").nullable is True

    def test_sentence_id_non_nullable(self):
        assert OUTPUT_SENTENCE_SCHEMA.field("sentence_id").nullable is False


# ===================================================================
# Extra columns (forward-compatibility)
# ===================================================================

class TestExtraColumns:
    """Extra upstream columns should be silently allowed."""

    def test_extra_columns_accepted(self):
        schema = SCHEMA_REGISTRY["polygons"]
        fields = list(schema) + [pa.field("new_upstream_col", pa.string())]
        extended = pa.schema(fields)
        # Should not raise.
        validate_table_schema("polygons", extended)
