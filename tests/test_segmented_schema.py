"""Tests for the Phase 3C intermediate segmented-sentence Arrow schema."""

from __future__ import annotations

import pyarrow as pa

from osm_polygon_sentence_relevance.schemas import SEGMENTED_SENTENCES_SCHEMA


EXPECTED_FIELDS = [
    ("polygon_id", pa.string(), False),
    ("wikidata", pa.string(), False),
    ("document_id", pa.string(), False),
    ("article_id", pa.string(), True),
    ("source", pa.string(), False),
    ("language", pa.string(), False),
    ("site", pa.string(), False),
    ("page_title", pa.string(), False),
    ("url", pa.string(), False),
    ("page_id", pa.int64(), False),
    ("revision_id", pa.int64(), False),
    ("revision_timestamp", pa.string(), False),
    ("document_content_hash", pa.string(), False),
    ("section_id", pa.string(), False),
    ("section_index", pa.int64(), False),
    ("section_path", pa.list_(pa.string()), False),
    ("sentence_index", pa.int64(), False),
    ("sentence_text_raw", pa.string(), False),
    ("sentence_text_normalized", pa.string(), False),
    ("section_content_hash", pa.string(), False),
    ("polygon_name", pa.string(), True),
    ("osm_primary_tag", pa.string(), True),
    ("osm_tags", pa.map_(pa.string(), pa.string()), False),
    ("region", pa.string(), False),
    ("lat", pa.float64(), True),
    ("lon", pa.float64(), True),
]

FORBIDDEN_PHASE4_FIELDS = [
    "sentence_id",
    "sentence_content_hash",
    "previous_sentence",
    "next_sentence",
    "duplicate_occurrence_count",
    "duplicate_sources",
]


class TestSegmentedSchemaStructure:
    def test_column_names_and_order(self):
        names = SEGMENTED_SENTENCES_SCHEMA.names
        assert names == [name for name, _, _ in EXPECTED_FIELDS]

    def test_field_count(self):
        assert len(SEGMENTED_SENTENCES_SCHEMA) == len(EXPECTED_FIELDS)

    def test_exact_types_and_nullability(self):
        fields = SEGMENTED_SENTENCES_SCHEMA
        for name, expected_type, expected_nullable in EXPECTED_FIELDS:
            field = fields.field(name)
            assert field.type == expected_type, f"{name} type"
            assert field.nullable == expected_nullable, f"{name} nullable"

    def test_section_path_is_list_of_string(self):
        field = SEGMENTED_SENTENCES_SCHEMA.field("section_path")
        assert pa.types.is_list(field.type)
        assert field.type.value_type == pa.string()

    def test_osm_tags_is_map_string_string(self):
        field = SEGMENTED_SENTENCES_SCHEMA.field("osm_tags")
        assert pa.types.is_map(field.type)
        assert field.type.key_type == pa.string()
        assert field.type.item_type == pa.string()

    def test_no_phase4_fields(self):
        names = set(SEGMENTED_SENTENCES_SCHEMA.names)
        assert names.isdisjoint(FORBIDDEN_PHASE4_FIELDS)
