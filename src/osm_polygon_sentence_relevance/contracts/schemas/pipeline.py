"""Pipeline (joined / segmented / final) output-table PyArrow schemas.

Immutable contracts used by the join, segmentation, and finalization
stages. Do not reorder or change field types.

``OUTPUT_SENTENCE_SCHEMA.osm_tags`` is encoded as
``list<struct<key:string, value:string>>`` because the Hugging Face
``datasets`` library cannot ingest ``map<string, string>``.  The
intermediate ``SEGMENTED_SENTENCES_SCHEMA.osm_tags`` keeps the
``map`` form for the segmentation boundary; the conversion happens at
finalization time via
:func:`osm_polygon_sentence_relevance.sentences.finalization.convert_osm_tags_to_list_of_struct`.
"""

from __future__ import annotations

import pyarrow as pa

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
        pa.field(
            "osm_tags",
            pa.list_(
                pa.struct(
                    [
                        pa.field("key", pa.string(), nullable=False),
                        pa.field("value", pa.string(), nullable=False),
                    ]
                )
            ),
            nullable=False,
        ),
        pa.field("region", pa.string(), nullable=False),
        pa.field("lat", pa.float64(), nullable=True),
        pa.field("lon", pa.float64(), nullable=True),
        pa.field("input_dataset_revision", pa.string(), nullable=False),
        pa.field("pipeline_version", pa.string(), nullable=False),
    ]
)

# Intermediate schema — joined sections (not the final output).
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

# Intermediate schema — segmented sentences.
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

__all__ = [
    "OUTPUT_SENTENCE_SCHEMA",
    "JOINED_SECTIONS_SCHEMA",
    "SEGMENTED_SENTENCES_SCHEMA",
]
