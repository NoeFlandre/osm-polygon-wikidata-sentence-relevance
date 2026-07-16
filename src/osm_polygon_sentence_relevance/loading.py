"""Compatibility facade: ``osm_polygon_sentence_relevance.loading``.

The implementation now lives in
:mod:`osm_polygon_sentence_relevance.ingestion.loading`. Import from there
in new code; this module re-exports the stable public symbols.
"""

from osm_polygon_sentence_relevance.ingestion.loading import load_validated_table

__all__ = ["load_validated_table"]
