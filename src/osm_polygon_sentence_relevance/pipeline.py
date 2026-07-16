"""Compatibility facade: ``osm_polygon_sentence_relevance.pipeline``.

The implementation now lives in
:mod:`osm_polygon_sentence_relevance.application.pipeline`. Import from there
in new code; this module re-exports the stable public symbols.
"""

from osm_polygon_sentence_relevance.application.pipeline import (
    PipelineResult,
    run_pipeline,
)
from osm_polygon_sentence_relevance.joins import build_region_section_occurrences
from osm_polygon_sentence_relevance.sentences.segmentation import SegmentationReport

__all__ = [
    "PipelineResult",
    "run_pipeline",
    "build_region_section_occurrences",
    "SegmentationReport",
]
