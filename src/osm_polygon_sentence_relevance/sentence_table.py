"""Compatibility facade: ``osm_polygon_sentence_relevance.sentence_table``.

The implementation now lives in
:mod:`osm_polygon_sentence_relevance.sentences.table`. Import from there in
new code; this module re-exports the stable public symbols.
"""

from osm_polygon_sentence_relevance.sentences.table import (
    SegmentedTableResult,
    segment_joined_sections,
    validate_joined_sections_table,
)

__all__ = [
    "SegmentedTableResult",
    "segment_joined_sections",
    "validate_joined_sections_table",
]
