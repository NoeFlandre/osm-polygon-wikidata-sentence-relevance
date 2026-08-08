"""Validation and transformation of joined-section tables.

This module provides :func:`validate_joined_sections_table`, a pure structural
check. :func:`segment_joined_sections` validates a joined table, then expands
each retained sentence into one output row conforming to
``SEGMENTED_SENTENCES_SCHEMA`` through an injected segmenter.

Neither function transforms, reorders, or mutates the caller's table, and no
model adapter or finalization logic is present here.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from osm_polygon_sentence_relevance.contracts.errors import (
    SegmentationError,
)
from osm_polygon_sentence_relevance.contracts.schemas import (
    JOINED_SECTIONS_SCHEMA,
    SEGMENTED_SENTENCES_SCHEMA,
)
from osm_polygon_sentence_relevance.sentences.preprocessing import (
    parse_osm_tags,
    parse_section_path,
)
from osm_polygon_sentence_relevance.sentences.segmentation import (
    SegmentationReport,
    SentenceSegmenter,
    build_segmentation_report,
    segment_sections_batch,
)


def validate_joined_sections_table(table: pa.Table) -> None:
    """Validate *table* against the joined-sections contract.

    Requires every field of :data:`JOINED_SECTIONS_SCHEMA` to be present with
    its exact Arrow type, and forbids nulls in fields marked non-null. Extra
    columns and arbitrary column order are allowed. The table is not mutated
    or reordered.

    Raises
    ------
    SegmentationError
        If any required field is missing, has the wrong type, or holds a null
        in a non-null field. The offending field name(s) are included in the
        message.
    """
    missing: list[str] = []
    type_mismatches: list[str] = []
    null_violations: list[str] = []

    for field in JOINED_SECTIONS_SCHEMA:
        if field.name not in table.column_names:
            missing.append(field.name)
            continue

        actual = table.schema.field(field.name)
        if actual.type != field.type:
            type_mismatches.append(
                f"{field.name} (expected {field.type}, got {actual.type})"
            )
            continue

        if not field.nullable and table.column(field.name).null_count > 0:
            null_violations.append(field.name)

    if missing:
        raise SegmentationError(
            "validate_joined_sections_table: missing required field(s): "
            + ", ".join(missing)
        )
    if type_mismatches:
        raise SegmentationError(
            "validate_joined_sections_table: type mismatch(es): "
            + "; ".join(type_mismatches)
        )
    if null_violations:
        raise SegmentationError(
            "validate_joined_sections_table: null in non-null field(s): "
            + ", ".join(null_violations)
        )

    return None


@dataclass(frozen=True, slots=True)
class SegmentedTableResult:
    """The result of :func:`segment_joined_sections`."""

    table: pa.Table
    report: SegmentationReport


@dataclass(frozen=True, slots=True)
class SegmentedBatch:
    """One deterministic section batch produced by the segmenter."""

    start_index: int
    end_index: int
    table: pa.Table
    report: SegmentationReport


# Deterministic sort key applied before segmentation (ascending on each).
_SORT_KEYS = [
    ("polygon_id", "ascending"),
    ("source", "ascending"),
    ("language", "ascending"),
    ("document_id", "ascending"),
    ("section_index", "ascending"),
    ("section_id", "ascending"),
]


def _check_batch_size(batch_size: int) -> None:
    """Reject invalid batch sizes (non-int, boolean, or <= 0)."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise SegmentationError(
            "segment_joined_sections: batch_size must be a positive integer, "
            f"got {type(batch_size).__name__}"
        )
    if batch_size <= 0:
        raise SegmentationError(
            f"segment_joined_sections: batch_size must be positive, got {batch_size}"
        )


def segment_joined_sections(
    table: pa.Table,
    segmenter: SentenceSegmenter,
    *,
    batch_size: int = 128,
) -> SegmentedTableResult:
    """Validate, sort, and segment a joined-sections table.

    Steps: validate the table; sort deterministically by polygon_id, source,
    language, document_id, section_index, section_id; then run a preflight
    that parses every row's ``section_path_raw`` and ``osm_tags_raw`` and
    validates every ``source`` (raising before any segmenter call); process
    sections in chunks of ``batch_size`` via :func:`segment_sections_batch`;
    expand each retained sentence into one row of
    ``SEGMENTED_SENTENCES_SCHEMA``; and build a :class:`SegmentationReport`.
    The input table is never mutated.

    Raises
    ------
    SegmentationError
        On validation failure, an invalid batch_size, or an invalid source.
    PreprocessingError
        On malformed ``section_path_raw`` or ``osm_tags_raw`` JSON.
    """
    batches = list(iter_segmented_batches(table, segmenter, batch_size=batch_size))
    if not batches:
        return SegmentedTableResult(
            table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
            report=_empty_segmentation_report(),
        )
    return SegmentedTableResult(
        table=pa.concat_tables([batch.table for batch in batches]),
        report=_merge_segmentation_reports(batch.report for batch in batches),
    )


def iter_segmented_batches(
    table: pa.Table,
    segmenter: SentenceSegmenter,
    *,
    batch_size: int = 128,
    start_index: int = 0,
) -> Iterator[SegmentedBatch]:
    """Yield deterministic segmentation batches, optionally resuming at an index.

    All validation and preflight parsing happens before the first segmenter
    call. ``start_index`` is a section-row offset, not an output-row offset,
    and must be aligned with ``batch_size`` unless it equals the table size.
    Earlier batches are skipped without invoking the segmenter, allowing a
    caller to resume after a durable batch checkpoint.
    """
    validate_joined_sections_table(table)
    _check_batch_size(batch_size)
    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise SegmentationError(
            "iter_segmented_batches: start_index must be an integer"
        )
    if start_index < 0 or start_index > table.num_rows:
        raise SegmentationError(
            "iter_segmented_batches: start_index must be within table bounds"
        )
    if start_index != table.num_rows and start_index % batch_size:
        raise SegmentationError(
            "iter_segmented_batches: start_index must align with batch_size"
        )
    if table.num_rows == 0 or start_index == table.num_rows:
        return

    compute: Any = pc
    indices = compute.sort_indices(table, sort_keys=_SORT_KEYS)
    rows = _to_sorted_provenance_rows(table.take(indices))
    for start in range(start_index, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        batch_rows = rows[start:end]
        texts = [row["section_text_raw"] for row in batch_rows]
        languages = [row["language"] for row in batch_rows]
        section_results = segment_sections_batch(texts, languages, segmenter)
        columns: dict[str, list[Any]] = {
            name: [] for name in SEGMENTED_SENTENCES_SCHEMA.names
        }
        sources: list[str] = []
        prepared_sections: list[Any] = []
        for row, prepared in zip(batch_rows, section_results, strict=True):
            source = row["source"]
            for sentence in prepared.sentences:
                _append_sentence_row(columns, row, source, sentence)
            prepared_sections.append(prepared)
            sources.append(source)
        yield SegmentedBatch(
            start_index=start,
            end_index=end,
            table=pa.table(
                {
                    name: pa.array(
                        columns[name],
                        type=SEGMENTED_SENTENCES_SCHEMA.field(name).type,
                    )
                    for name in SEGMENTED_SENTENCES_SCHEMA.names
                },
                schema=SEGMENTED_SENTENCES_SCHEMA,
            ),
            report=build_segmentation_report(prepared_sections, sources),
        )


def _append_sentence_row(
    columns: dict[str, list[Any]], row: dict[str, Any], source: str, sentence: Any
) -> None:
    """Append one prepared sentence while preserving the output schema order."""
    columns["polygon_id"].append(row["polygon_id"])
    columns["wikidata"].append(row["wikidata"])
    columns["document_id"].append(row["document_id"])
    columns["article_id"].append(row["article_id"])
    columns["source"].append(source)
    columns["language"].append(row["language"])
    columns["site"].append(row["site"])
    columns["page_title"].append(row["page_title"])
    columns["url"].append(row["url"])
    columns["page_id"].append(row["page_id"])
    columns["revision_id"].append(row["revision_id"])
    columns["revision_timestamp"].append(row["revision_timestamp"])
    columns["document_content_hash"].append(row["document_content_hash"])
    columns["section_id"].append(row["section_id"])
    columns["section_index"].append(row["section_index"])
    columns["section_path"].append(row["section_path"])
    columns["sentence_index"].append(sentence.sentence_index)
    columns["sentence_text_raw"].append(sentence.sentence_text_raw)
    columns["sentence_text_normalized"].append(sentence.sentence_text_normalized)
    columns["section_content_hash"].append(row["section_content_hash"])
    columns["polygon_name"].append(row["polygon_name"])
    columns["osm_primary_tag"].append(row["osm_primary_tag"])
    columns["osm_tags"].append(row["osm_tags"])
    columns["region"].append(row["region"])
    columns["lat"].append(row["lat"])
    columns["lon"].append(row["lon"])


def _empty_segmentation_report() -> SegmentationReport:
    return SegmentationReport(
        input_section_occurrence_count=0,
        emitted_segment_count=0,
        retained_sentence_occurrence_count=0,
        dropped_empty_raw_count=0,
        dropped_empty_normalized_count=0,
        wikipedia_sentence_occurrence_count=0,
        wikivoyage_sentence_occurrence_count=0,
    )


def _merge_segmentation_reports(
    reports: Iterator[SegmentationReport],
) -> SegmentationReport:
    values = list(reports)
    return SegmentationReport(
        input_section_occurrence_count=sum(
            report.input_section_occurrence_count for report in values
        ),
        emitted_segment_count=sum(report.emitted_segment_count for report in values),
        retained_sentence_occurrence_count=sum(
            report.retained_sentence_occurrence_count for report in values
        ),
        dropped_empty_raw_count=sum(
            report.dropped_empty_raw_count for report in values
        ),
        dropped_empty_normalized_count=sum(
            report.dropped_empty_normalized_count for report in values
        ),
        wikipedia_sentence_occurrence_count=sum(
            report.wikipedia_sentence_occurrence_count for report in values
        ),
        wikivoyage_sentence_occurrence_count=sum(
            report.wikivoyage_sentence_occurrence_count for report in values
        ),
    )


# Fields copied verbatim from each joined row into the output row.
_PROVENANCE_FIELDS = (
    "polygon_id",
    "wikidata",
    "document_id",
    "article_id",
    "language",
    "site",
    "page_title",
    "url",
    "page_id",
    "revision_id",
    "revision_timestamp",
    "document_content_hash",
    "section_id",
    "section_index",
    "section_content_hash",
    "polygon_name",
    "osm_primary_tag",
    "region",
    "lat",
    "lon",
    "section_text_raw",
)


def _to_sorted_provenance_rows(table: pa.Table) -> list[dict]:
    """Convert a table to preflight-validated, parsed Python rows.

    Each returned row dict carries the provenance fields plus ``source`` and
    the already-parsed ``section_path`` and ``osm_tags``. Malformed
    ``section_path_raw``/``osm_tags_raw`` raise :class:`PreprocessingError`;
    an invalid ``source`` raises :class:`SegmentationError`. Runs entirely
    before any segmenter call.
    """
    raw_rows = table.to_pylist()
    rows: list[dict] = []
    for raw in raw_rows:
        source = raw["source"]
        if source not in ("wikipedia", "wikivoyage"):
            raise SegmentationError(
                "segment_joined_sections: source must be 'wikipedia' or "
                f"'wikivoyage', got {source!r}"
            )
        row = {name: raw[name] for name in _PROVENANCE_FIELDS}
        row["source"] = source
        row["section_path"] = parse_section_path(raw["section_path_raw"])
        row["osm_tags"] = parse_osm_tags(raw["osm_tags_raw"])
        rows.append(row)
    return rows
