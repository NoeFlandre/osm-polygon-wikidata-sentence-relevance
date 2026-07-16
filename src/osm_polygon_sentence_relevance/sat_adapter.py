"""Compatibility facade: ``osm_polygon_sentence_relevance.sat_adapter``.

The implementation now lives in
:mod:`osm_polygon_sentence_relevance.sentences.sat`. Import from there in new
code; this module re-exports the stable public symbol.
"""

from osm_polygon_sentence_relevance.sentences.sat import SaTSentenceSegmenter

__all__ = ["SaTSentenceSegmenter"]
