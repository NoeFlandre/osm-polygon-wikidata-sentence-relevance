"""Compatibility facade: ``osm_polygon_sentence_relevance.schemas``.

The canonical schemas and ``validate_table_schema`` now live in
:mod:`osm_polygon_sentence_relevance.contracts.schemas`. Import from there in
new code; this module re-exports the stable public symbols so existing
imports keep working.
"""

from osm_polygon_sentence_relevance.contracts.schemas import (
    JOINED_SECTIONS_SCHEMA,
    OUTPUT_SENTENCE_SCHEMA,
    POLYGON_ARTICLES_SCHEMA,
    POLYGONS_SCHEMA,
    SCHEMA_REGISTRY,
    SECTIONS_SCHEMA,
    SEGMENTED_SENTENCES_SCHEMA,
    WIKIPEDIA_DOCUMENTS_SCHEMA,
    WIKIVOYAGE_DOCUMENTS_SCHEMA,
    validate_table_schema,
)

__all__ = [
    "POLYGONS_SCHEMA",
    "POLYGON_ARTICLES_SCHEMA",
    "WIKIPEDIA_DOCUMENTS_SCHEMA",
    "WIKIVOYAGE_DOCUMENTS_SCHEMA",
    "SECTIONS_SCHEMA",
    "OUTPUT_SENTENCE_SCHEMA",
    "JOINED_SECTIONS_SCHEMA",
    "SEGMENTED_SENTENCES_SCHEMA",
    "SCHEMA_REGISTRY",
    "validate_table_schema",
]
