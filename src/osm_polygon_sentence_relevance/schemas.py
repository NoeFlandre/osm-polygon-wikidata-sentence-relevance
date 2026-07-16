"""PyArrow schema contracts for all pipeline tables.

Each schema is defined as a module-level constant and registered in
``SCHEMA_REGISTRY`` keyed by the logical table name from
:pymod:`osm_polygon_sentence_relevance.constants`.

The public entry point is :func:`validate_table_schema`.
"""

from __future__ import annotations

import pyarrow as pa

from osm_polygon_sentence_relevance.errors import (
    IncompatibleTypesError,
    MissingColumnsError,
    UnknownTableError,
)

# ===================================================================
# Input schemas — column order matches the upstream Afghanistan shards
# ===================================================================

POLYGONS_SCHEMA = pa.schema(
    [
        pa.field("polygon_id", pa.string()),
        pa.field("region", pa.string()),
        pa.field("source_pbf", pa.string()),
        pa.field("osm_type", pa.string()),
        pa.field("osm_id", pa.int64()),
        pa.field("wikidata", pa.string()),
        pa.field("name", pa.string()),
        pa.field("tags", pa.string()),
        pa.field("tag_keys", pa.string()),
        pa.field("tag_count", pa.int64()),
        pa.field("osm_primary_tag", pa.string()),
        pa.field("centroid", pa.string()),
        pa.field("lat", pa.float64()),
        pa.field("lon", pa.float64()),
        pa.field("bbox", pa.string()),
        pa.field("geometry", pa.string()),
        pa.field("area_m2", pa.float64()),
        pa.field("area_km2", pa.float64()),
        pa.field("area_bucket", pa.string()),
        pa.field("has_name", pa.bool_()),
        pa.field("has_wikidata", pa.bool_()),
        pa.field("has_wikipedia", pa.bool_()),
        pa.field("wikipedia_language_count", pa.int64()),
        pa.field("wikipedia_languages", pa.string()),
        pa.field("wikipedia_article_count", pa.int64()),
        pa.field("has_english_wikipedia", pa.bool_()),
        pa.field("has_french_wikipedia", pa.bool_()),
        pa.field("text_available", pa.bool_()),
        pa.field("best_language", pa.string()),
        pa.field("extraction_version", pa.string()),
        pa.field("extracted_at", pa.string()),
    ]
)

POLYGON_ARTICLES_SCHEMA = pa.schema(
    [
        pa.field("polygon_id", pa.string()),
        pa.field("article_id", pa.string()),
        pa.field("wikidata", pa.string()),
        pa.field("language", pa.string()),
        pa.field("source_pbf", pa.string()),
        pa.field("region", pa.string()),
        pa.field("osm_type", pa.string()),
        pa.field("osm_id", pa.int64()),
        pa.field("page_id", pa.int64()),
        pa.field("revision_id", pa.int64()),
        pa.field("is_best_language", pa.bool_()),
    ]
)

WIKIPEDIA_DOCUMENTS_SCHEMA = pa.schema(
    [
        pa.field("document_id", pa.string()),
        pa.field("article_id", pa.string()),
        pa.field("wikidata", pa.string()),
        pa.field("project", pa.string()),
        pa.field("language", pa.string()),
        pa.field("site", pa.string()),
        pa.field("title", pa.string()),
        pa.field("url", pa.string()),
        pa.field("page_id", pa.int64()),
        pa.field("revision_id", pa.int64()),
        pa.field("revision_timestamp", pa.string()),
        pa.field("retrieved_at", pa.string()),
        pa.field("wikidata_label", pa.string()),
        pa.field("wikidata_description", pa.string()),
        pa.field("wikidata_aliases", pa.string()),
        pa.field("lead_text", pa.string()),
        pa.field("extract", pa.string()),
        pa.field("full_text", pa.string()),
        pa.field("full_text_format", pa.string()),
        pa.field("article_length_chars", pa.int64()),
        pa.field("article_length_words", pa.int64()),
        pa.field("article_length_tokens_estimate", pa.int64()),
        pa.field("thumbnail_url", pa.string()),
        pa.field("thumbnail_width", pa.int64()),  # nullable upstream
        pa.field("thumbnail_height", pa.int64()),  # nullable upstream
        pa.field("categories", pa.string()),
        pa.field("license", pa.string()),
        pa.field("attribution", pa.string()),
        pa.field("source_api", pa.string()),
        pa.field("fetch_status", pa.string()),
        pa.field("fetch_error", pa.string()),
        pa.field("content_hash", pa.string()),
    ]
)

WIKIVOYAGE_DOCUMENTS_SCHEMA = pa.schema(
    [
        pa.field("document_id", pa.string()),
        pa.field("article_id", pa.string()),
        pa.field("wikidata", pa.string()),
        pa.field("project", pa.string()),
        pa.field("language", pa.string()),
        pa.field("site", pa.string()),
        pa.field("title", pa.string()),
        pa.field("url", pa.string()),
        pa.field("page_id", pa.int64()),
        pa.field("revision_id", pa.int64()),
        pa.field("revision_timestamp", pa.string()),
        pa.field("retrieved_at", pa.string()),
        pa.field("full_text", pa.string()),
        pa.field("full_text_format", pa.string()),
        pa.field("article_length_chars", pa.int64()),
        pa.field("article_length_words", pa.int64()),
        pa.field("article_length_tokens_estimate", pa.int64()),
        pa.field("license", pa.string()),
        pa.field("attribution", pa.string()),
        pa.field("source_api", pa.string()),
        pa.field("fetch_status", pa.string()),
        pa.field("fetch_error", pa.string()),
        pa.field("content_hash", pa.string()),
    ]
)

# Wikipedia and Wikivoyage sections share an identical schema.
SECTIONS_SCHEMA = pa.schema(
    [
        pa.field("section_id", pa.string()),
        pa.field("document_id", pa.string()),
        pa.field("article_id", pa.string()),
        pa.field("wikidata", pa.string()),
        pa.field("project", pa.string()),
        pa.field("language", pa.string()),
        pa.field("site", pa.string()),
        pa.field("page_id", pa.int64()),
        pa.field("revision_id", pa.int64()),
        pa.field("section_index", pa.int64()),
        pa.field("heading", pa.string()),
        pa.field("anchor", pa.string()),
        pa.field("level", pa.int64()),
        pa.field("parent_section_id", pa.string()),
        pa.field("section_path", pa.string()),
        pa.field("text", pa.string()),
        pa.field("text_length_chars", pa.int64()),
        pa.field("text_length_words", pa.int64()),
        pa.field("text_length_tokens_estimate", pa.int64()),
        pa.field("content_hash", pa.string()),
        pa.field("license", pa.string()),
        pa.field("attribution", pa.string()),
    ]
)

# ===================================================================
# Output schema — sentence-level target table
# ===================================================================

OUTPUT_SENTENCE_SCHEMA = pa.schema(
    [
        pa.field("sentence_id", pa.string(), nullable=False),
        pa.field("polygon_id", pa.string(), nullable=False),
        pa.field("wikidata", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("article_id", pa.string(), nullable=True),
        pa.field("source", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=False),
        pa.field("site", pa.string(), nullable=False),
        pa.field("page_title", pa.string(), nullable=False),
        pa.field("section_id", pa.string(), nullable=False),
        pa.field("section_index", pa.int64(), nullable=False),
        pa.field("section_path", pa.list_(pa.string()), nullable=False),
        pa.field("sentence_index", pa.int64(), nullable=False),
        pa.field("sentence_text_raw", pa.string(), nullable=False),
        pa.field("sentence_text_normalized", pa.string(), nullable=False),
        pa.field("previous_sentence", pa.string(), nullable=True),
        pa.field("next_sentence", pa.string(), nullable=True),
        pa.field("url", pa.string(), nullable=False),
        pa.field("page_id", pa.int64(), nullable=False),
        pa.field("revision_id", pa.int64(), nullable=False),
        pa.field("revision_timestamp", pa.string(), nullable=False),
        pa.field("document_content_hash", pa.string(), nullable=False),
        pa.field("section_content_hash", pa.string(), nullable=False),
        pa.field("sentence_content_hash", pa.string(), nullable=False),
        pa.field("duplicate_occurrence_count", pa.int64(), nullable=False),
        pa.field("duplicate_sources", pa.list_(pa.string()), nullable=False),
        pa.field("polygon_name", pa.string(), nullable=True),
        pa.field("osm_primary_tag", pa.string(), nullable=True),
        pa.field("osm_tags", pa.map_(pa.string(), pa.string()), nullable=False),
        pa.field("region", pa.string(), nullable=False),
        pa.field("lat", pa.float64(), nullable=True),
        pa.field("lon", pa.float64(), nullable=True),
        pa.field("input_dataset_revision", pa.string(), nullable=False),
        pa.field("pipeline_version", pa.string(), nullable=False),
    ]
)

# ===================================================================
# Intermediate schema — Phase 2 joined sections (not the final output)
# ===================================================================

JOINED_SECTIONS_SCHEMA = pa.schema(
    [
        pa.field("polygon_id", pa.string(), nullable=False),
        pa.field("wikidata", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("article_id", pa.string(), nullable=True),
        pa.field("source", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=False),
        pa.field("site", pa.string(), nullable=False),
        pa.field("page_title", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("page_id", pa.int64(), nullable=False),
        pa.field("revision_id", pa.int64(), nullable=False),
        pa.field("revision_timestamp", pa.string(), nullable=False),
        pa.field("document_content_hash", pa.string(), nullable=False),
        pa.field("section_id", pa.string(), nullable=False),
        pa.field("section_index", pa.int64(), nullable=False),
        pa.field("section_path_raw", pa.string(), nullable=False),
        pa.field("section_text_raw", pa.string(), nullable=False),
        pa.field("section_content_hash", pa.string(), nullable=False),
        pa.field("polygon_name", pa.string(), nullable=True),
        pa.field("osm_primary_tag", pa.string(), nullable=True),
        pa.field("osm_tags_raw", pa.string(), nullable=False),
        pa.field("region", pa.string(), nullable=False),
        pa.field("lat", pa.float64(), nullable=True),
        pa.field("lon", pa.float64(), nullable=True),
    ]
)

# ===================================================================
# Intermediate schema — Phase 3C segmented sentences
# ===================================================================

SEGMENTED_SENTENCES_SCHEMA = pa.schema(
    [
        pa.field("polygon_id", pa.string(), nullable=False),
        pa.field("wikidata", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("article_id", pa.string(), nullable=True),
        pa.field("source", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=False),
        pa.field("site", pa.string(), nullable=False),
        pa.field("page_title", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("page_id", pa.int64(), nullable=False),
        pa.field("revision_id", pa.int64(), nullable=False),
        pa.field("revision_timestamp", pa.string(), nullable=False),
        pa.field("document_content_hash", pa.string(), nullable=False),
        pa.field("section_id", pa.string(), nullable=False),
        pa.field("section_index", pa.int64(), nullable=False),
        pa.field("section_path", pa.list_(pa.string()), nullable=False),
        pa.field("sentence_index", pa.int64(), nullable=False),
        pa.field("sentence_text_raw", pa.string(), nullable=False),
        pa.field("sentence_text_normalized", pa.string(), nullable=False),
        pa.field("section_content_hash", pa.string(), nullable=False),
        pa.field("polygon_name", pa.string(), nullable=True),
        pa.field("osm_primary_tag", pa.string(), nullable=True),
        pa.field("osm_tags", pa.map_(pa.string(), pa.string()), nullable=False),
        pa.field("region", pa.string(), nullable=False),
        pa.field("lat", pa.float64(), nullable=True),
        pa.field("lon", pa.float64(), nullable=True),
    ]
)

# ===================================================================
# Schema registry
# ===================================================================

SCHEMA_REGISTRY: dict[str, pa.Schema] = {
    "polygons": POLYGONS_SCHEMA,
    "polygon_articles": POLYGON_ARTICLES_SCHEMA,
    "wikipedia_documents": WIKIPEDIA_DOCUMENTS_SCHEMA,
    "wikivoyage_documents": WIKIVOYAGE_DOCUMENTS_SCHEMA,
    "wikipedia_sections": SECTIONS_SCHEMA,
    "wikivoyage_sections": SECTIONS_SCHEMA,
}

# ===================================================================
# Public validation API
# ===================================================================


def validate_table_schema(table_name: str, actual_schema: pa.Schema) -> None:
    """Validate *actual_schema* against the registered contract for *table_name*.

    Raises
    ------
    UnknownTableError
        If *table_name* is not in :data:`SCHEMA_REGISTRY`.
    MissingColumnsError
        If any required columns are absent from *actual_schema*.
    IncompatibleTypesError
        If any present columns have a type incompatible with the contract.

    Notes
    -----
    * Extra columns in *actual_schema* are silently allowed so the pipeline
      remains forward-compatible with upstream additions.
    * Column order in *actual_schema* need not match the contract.
    """
    if table_name not in SCHEMA_REGISTRY:
        raise UnknownTableError(table_name)

    expected = SCHEMA_REGISTRY[table_name]
    actual_names = {f.name for f in actual_schema}

    # --- missing columns ---
    expected_names = {f.name for f in expected}
    missing = sorted(expected_names - actual_names)
    if missing:
        raise MissingColumnsError(table_name, missing)

    # --- type mismatches ---
    mismatches: list[tuple[str, str, str]] = []
    for field in expected:
        actual_field = actual_schema.field(field.name)
        if actual_field.type != field.type:
            mismatches.append((field.name, str(field.type), str(actual_field.type)))
    if mismatches:
        raise IncompatibleTypesError(table_name, mismatches)
