"""Ingestion layer: dataset acquisition, shard discovery, tabular loading."""

from osm_polygon_sentence_relevance.ingestion.acquisition import (
    AcquisitionResult,
    acquire_dataset_snapshot,
)
from osm_polygon_sentence_relevance.ingestion.discovery import (
    RegionShardSet,
    discover_shards,
)
from osm_polygon_sentence_relevance.ingestion.loading import load_validated_table

__all__ = [
    "AcquisitionResult",
    "acquire_dataset_snapshot",
    "RegionShardSet",
    "discover_shards",
    "load_validated_table",
]
