"""Canonical schema API for the pipeline.

Re-exports only stable, documented public symbols from the internal
``input``, ``pipeline``, and ``registry`` modules. Import from
``osm_polygon_sentence_relevance.contracts.schemas`` in production code;
do not import the submodules directly.
"""

from osm_polygon_sentence_relevance.contracts.schemas.input import (
    POLYGON_ARTICLES_SCHEMA,
    POLYGONS_SCHEMA,
    SECTIONS_SCHEMA,
    WIKIPEDIA_DOCUMENTS_SCHEMA,
    WIKIVOYAGE_DOCUMENTS_SCHEMA,
)
from osm_polygon_sentence_relevance.contracts.schemas.pipeline import (
    JOINED_SECTIONS_SCHEMA,
    OUTPUT_SENTENCE_SCHEMA,
    SEGMENTED_SENTENCES_SCHEMA,
)
from osm_polygon_sentence_relevance.contracts.schemas.registry import (
    SCHEMA_REGISTRY,
    validate_table_schema,
)

__all__ = [
    # Input-table schemas
    "POLYGONS_SCHEMA",
    "POLYGON_ARTICLES_SCHEMA",
    "WIKIPEDIA_DOCUMENTS_SCHEMA",
    "WIKIVOYAGE_DOCUMENTS_SCHEMA",
    "SECTIONS_SCHEMA",
    # Pipeline-table schemas
    "OUTPUT_SENTENCE_SCHEMA",
    "JOINED_SECTIONS_SCHEMA",
    "SEGMENTED_SENTENCES_SCHEMA",
    # Registry + validation
    "SCHEMA_REGISTRY",
    "validate_table_schema",
]
