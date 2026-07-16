"""Compatibility facade: ``osm_polygon_sentence_relevance.acquisition``.

The implementation now lives in
:mod:`osm_polygon_sentence_relevance.ingestion.acquisition`. Import from there
in new code; this module re-exports the stable public symbols.
"""

from osm_polygon_sentence_relevance.ingestion.acquisition import (
    ALLOW_PATTERNS,
    IGNORE_PATTERNS,
    AcquisitionResult,
    acquire_dataset_snapshot,
)

__all__ = [
    "AcquisitionResult",
    "acquire_dataset_snapshot",
    "ALLOW_PATTERNS",
    "IGNORE_PATTERNS",
]
