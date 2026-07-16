"""Upstream input-table PyArrow schemas.

Column order matches the upstream Afghanistan shards. These schemas are
immutable contracts; do not reorder or change field types.
"""

from __future__ import annotations

import pyarrow as pa

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

__all__ = [
    "POLYGONS_SCHEMA",
    "POLYGON_ARTICLES_SCHEMA",
    "WIKIPEDIA_DOCUMENTS_SCHEMA",
    "WIKIVOYAGE_DOCUMENTS_SCHEMA",
    "SECTIONS_SCHEMA",
]
