"""Compatibility facade: ``osm_polygon_sentence_relevance.settings``.

The implementation now lives in
:mod:`osm_polygon_sentence_relevance.application.settings`. Import from
there in new code; this module re-exports the stable public symbol so
existing imports keep working.
"""

from osm_polygon_sentence_relevance.application.settings import PipelineSettings

__all__ = ["PipelineSettings"]
