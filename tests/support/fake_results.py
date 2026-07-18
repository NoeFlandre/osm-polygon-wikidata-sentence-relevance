"""Deterministic fake result builders (no real pipeline run)."""

from __future__ import annotations

from pathlib import Path

from osm_polygon_sentence_relevance.exporter import ExportResult
from osm_polygon_sentence_relevance.finalization import FinalizationReport
from osm_polygon_sentence_relevance.pipeline import PipelineResult
from osm_polygon_sentence_relevance.segmentation import SegmentationReport


def make_fake_pipeline_result(
    *,
    parquet_path: Path = Path("/tmp/out/sentences.parquet"),
    manifest_path: Path = Path("/tmp/out/manifest.json"),
    card_path: Path = Path("/tmp/out/README.md"),
    processed_regions_count: int = 2,
    total_joined_section_occurrences: int = 15,
    input_section_occurrence_count: int = 10,
    emitted_segment_count: int = 8,
    retained_sentence_occurrence_count: int = 8,
    dropped_empty_raw_count: int = 1,
    dropped_empty_normalized_count: int = 1,
    wikipedia_sentence_occurrence_count: int = 5,
    wikivoyage_sentence_occurrence_count: int = 3,
    input_sentence_occurrence_count: int = 8,
    output_sentence_count: int = 6,
    duplicate_occurrence_count_removed: int = 2,
    cross_source_duplicate_group_count: int = 1,
) -> PipelineResult:
    """Build a deterministic fake PipelineResult for CLI/orchestration tests."""
    export = ExportResult(
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        card_path=card_path,
        manifest_data={},
    )
    seg_report = SegmentationReport(
        input_section_occurrence_count=input_section_occurrence_count,
        emitted_segment_count=emitted_segment_count,
        retained_sentence_occurrence_count=retained_sentence_occurrence_count,
        dropped_empty_raw_count=dropped_empty_raw_count,
        dropped_empty_normalized_count=dropped_empty_normalized_count,
        wikipedia_sentence_occurrence_count=wikipedia_sentence_occurrence_count,
        wikivoyage_sentence_occurrence_count=wikivoyage_sentence_occurrence_count,
    )
    fin_report = FinalizationReport(
        input_sentence_occurrence_count=input_sentence_occurrence_count,
        output_sentence_count=output_sentence_count,
        duplicate_occurrence_count_removed=duplicate_occurrence_count_removed,
        cross_source_duplicate_group_count=cross_source_duplicate_group_count,
    )
    return PipelineResult(
        export_result=export,
        processed_regions_count=processed_regions_count,
        total_joined_section_occurrences=total_joined_section_occurrences,
        segmentation_report=seg_report,
        finalization_report=fin_report,
    )


def make_segmented_row(
    *,
    polygon_id: str = "poly-1",
    wikidata: str = "Q1",
    document_id: str = "doc-1",
    article_id: str = "art-1",
    source: str = "wikipedia",
    language: str = "en",
    site: str = "en.wikipedia.org",
    page_title: str = "Page Title",
    url: str = "https://example.com",
    page_id: int = 1,
    revision_id: int = 1,
    revision_timestamp: str = "2026-07-15T00:00:00Z",
    document_content_hash: str = "doc-hash-1",
    section_id: str = "sec-1",
    section_index: int = 0,
    section_path: list[str] | None = None,
    sentence_index: int = 0,
    sentence_text_raw: str = "Raw text.",
    sentence_text_normalized: str = "normalized text",
    section_content_hash: str = "sec-hash-1",
    polygon_name: str = "Poly Name",
    osm_primary_tag: str = "primary",
    osm_tags: list[tuple[str, str]] | None = None,
    region: str = "reg-1",
    lat: float = 12.34,
    lon: float = 56.78,
) -> dict:
    """Build a single SEGMENTED_SENTENCES_SCHEMA row dict."""
    if section_path is None:
        section_path = ["Intro"]
    if osm_tags is None:
        osm_tags = [("name", "Poly Name")]
    return {
        "polygon_id": polygon_id,
        "wikidata": wikidata,
        "document_id": document_id,
        "article_id": article_id,
        "source": source,
        "language": language,
        "site": site,
        "page_title": page_title,
        "url": url,
        "page_id": page_id,
        "revision_id": revision_id,
        "revision_timestamp": revision_timestamp,
        "document_content_hash": document_content_hash,
        "section_id": section_id,
        "section_index": section_index,
        "section_path": section_path,
        "sentence_index": sentence_index,
        "sentence_text_raw": sentence_text_raw,
        "sentence_text_normalized": sentence_text_normalized,
        "section_content_hash": section_content_hash,
        "polygon_name": polygon_name,
        "osm_primary_tag": osm_primary_tag,
        "osm_tags": osm_tags,
        "region": region,
        "lat": lat,
        "lon": lon,
    }
