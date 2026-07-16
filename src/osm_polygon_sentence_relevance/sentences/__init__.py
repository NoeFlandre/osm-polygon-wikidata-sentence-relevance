"""Sentence layer: normalization, segmentation, tabular segmentation, finalization."""

from osm_polygon_sentence_relevance.sentences.finalization import (
    FinalizationReport,
    FinalizedDataset,
    deterministic_sentence_id,
    finalize_sentence_dataset,
    sentence_content_hash,
)
from osm_polygon_sentence_relevance.sentences.preprocessing import (
    normalize_sentence,
    parse_osm_tags,
    parse_section_path,
)
from osm_polygon_sentence_relevance.sentences.sat import SaTSentenceSegmenter
from osm_polygon_sentence_relevance.sentences.segmentation import (
    SegmentationReport,
    SentenceSegmenter,
    split_validated_batch,
)
from osm_polygon_sentence_relevance.sentences.table import (
    SegmentedTableResult,
    segment_joined_sections,
    validate_joined_sections_table,
)

__all__ = [
    "normalize_sentence",
    "parse_osm_tags",
    "parse_section_path",
    "SaTSentenceSegmenter",
    "SentenceSegmenter",
    "SegmentationReport",
    "split_validated_batch",
    "SegmentedTableResult",
    "segment_joined_sections",
    "validate_joined_sections_table",
    "FinalizationReport",
    "FinalizedDataset",
    "deterministic_sentence_id",
    "finalize_sentence_dataset",
    "sentence_content_hash",
]
