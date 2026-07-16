"""Compatibility facade: ``osm_polygon_sentence_relevance.segmentation``.

The implementation now lives in
:mod:`osm_polygon_sentence_relevance.sentences.segmentation`. Import from
there in new code; this module re-exports the stable public symbols.
"""

from osm_polygon_sentence_relevance.sentences.segmentation import (
    PreparedSection,
    PreparedSentence,
    SegmentationReport,
    SentenceSegmenter,
    build_segmentation_report,
    segment_one_section,
    segment_sections_batch,
    split_validated_batch,
)

__all__ = [
    "PreparedSection",
    "PreparedSentence",
    "SegmentationReport",
    "SentenceSegmenter",
    "build_segmentation_report",
    "segment_one_section",
    "segment_sections_batch",
    "split_validated_batch",
]
