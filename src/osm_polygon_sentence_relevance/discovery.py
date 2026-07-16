"""Compatibility facade: ``osm_polygon_sentence_relevance.discovery``.

The implementation now lives in
:mod:`osm_polygon_sentence_relevance.ingestion.discovery`. Import from there
in new code; this module re-exports the stable public symbols.
"""

from osm_polygon_sentence_relevance.ingestion.discovery import (
    RegionShardSet,
    discover_shards,
)

__all__ = ["RegionShardSet", "discover_shards"]
