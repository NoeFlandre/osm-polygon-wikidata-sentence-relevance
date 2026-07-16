"""Compatibility facade: ``osm_polygon_sentence_relevance.constants``.

The canonical definitions now live in
:mod:`osm_polygon_sentence_relevance.contracts.constants`. Import from there
in new code; this module re-exports the stable public symbols so existing
imports keep working.
"""

from osm_polygon_sentence_relevance.contracts.constants import (
    ALLOWED_INPUT_PATHS,
    ALLOWED_SOURCES,
    DEFAULT_INPUT_REVISION,
    INPUT_DATASET_ID,
    OUTPUT_DATASET_ID,
    PIPELINE_VERSION,
    SCHEMA_NAMES,
)

__all__ = [
    "INPUT_DATASET_ID",
    "OUTPUT_DATASET_ID",
    "DEFAULT_INPUT_REVISION",
    "PIPELINE_VERSION",
    "ALLOWED_SOURCES",
    "SCHEMA_NAMES",
    "ALLOWED_INPUT_PATHS",
]
