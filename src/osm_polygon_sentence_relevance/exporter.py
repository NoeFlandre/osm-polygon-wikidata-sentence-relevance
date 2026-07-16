"""Compatibility facade: ``osm_polygon_sentence_relevance.exporter``.

The implementation now lives in
:mod:`osm_polygon_sentence_relevance.output.exporter`. Import from there in
new code; this module re-exports the stable public symbols.
"""

from osm_polygon_sentence_relevance.output.exporter import (
    ExportResult,
    export_finalized_dataset,
)

__all__ = ["ExportResult", "export_finalized_dataset"]
