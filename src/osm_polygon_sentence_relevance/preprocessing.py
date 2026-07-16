"""Compatibility facade: ``osm_polygon_sentence_relevance.preprocessing``.

The implementation now lives in
:mod:`osm_polygon_sentence_relevance.sentences.preprocessing`. Import from
there in new code; this module re-exports the stable public symbols.
"""

from osm_polygon_sentence_relevance.sentences.preprocessing import (
    normalize_sentence,
    parse_osm_tags,
    parse_section_path,
)

__all__ = ["normalize_sentence", "parse_osm_tags", "parse_section_path"]
