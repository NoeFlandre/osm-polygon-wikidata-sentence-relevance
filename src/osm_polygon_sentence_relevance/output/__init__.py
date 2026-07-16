"""Output layer: deterministic, atomically installed dataset export."""

from osm_polygon_sentence_relevance.output.exporter import (
    ExportResult,
    export_finalized_dataset,
)

__all__ = ["ExportResult", "export_finalized_dataset"]
