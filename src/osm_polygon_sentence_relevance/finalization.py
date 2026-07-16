"""Compatibility facade: ``osm_polygon_sentence_relevance.finalization``.

The implementation now lives in
:mod:`osm_polygon_sentence_relevance.sentences.finalization`. Import from
there in new code; this module re-exports the stable public symbols.
"""

from osm_polygon_sentence_relevance.sentences.finalization import (
    FinalizationReport,
    FinalizedDataset,
    deterministic_sentence_id,
    finalize_sentence_dataset,
    sentence_content_hash,
)

__all__ = [
    "FinalizationReport",
    "FinalizedDataset",
    "deterministic_sentence_id",
    "finalize_sentence_dataset",
    "sentence_content_hash",
]
