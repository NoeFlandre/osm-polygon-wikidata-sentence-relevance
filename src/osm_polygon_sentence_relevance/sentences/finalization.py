from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from osm_polygon_sentence_relevance.contracts.constants import ALLOWED_SOURCES
from osm_polygon_sentence_relevance.contracts.errors import FinalizationError
from osm_polygon_sentence_relevance.contracts.schemas import (
    OUTPUT_SENTENCE_SCHEMA,
    SEGMENTED_SENTENCES_SCHEMA,
)


@dataclass(frozen=True, slots=True)
class FinalizationReport:
    input_sentence_occurrence_count: int
    output_sentence_count: int
    duplicate_occurrence_count_removed: int
    cross_source_duplicate_group_count: int


@dataclass(frozen=True, slots=True)
class FinalizedDataset:
    table: pa.Table
    report: FinalizationReport


def sentence_content_hash(normalized_text: str) -> str:
    """Compute the SHA-256 hash of a normalized sentence.

    Parameters
    ----------
    normalized_text : str
        The normalized UTF-8 text of the sentence.

    Returns
    -------
    str
        The lowercase SHA-256 hex string of the normalized text.
    """
    if not isinstance(normalized_text, str):
        raise TypeError("normalized_text must be a string")
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest().lower()


def deterministic_sentence_id(
    polygon_id: str,
    language: str,
    sentence_hash: str,
) -> str:
    """Compute a deterministic ID for a sentence occurrence.

    Parameters
    ----------
    polygon_id : str
        The OSM polygon identifier.
    language : str
        The language code of the sentence.
    sentence_hash : str
        The sentence content hash.

    Returns
    -------
    str
        The lowercase SHA-256 hex string of the compact canonical JSON.
    """
    if not isinstance(polygon_id, str):
        raise TypeError("polygon_id must be a string")
    if not isinstance(language, str):
        raise TypeError("language must be a string")
    if not isinstance(sentence_hash, str):
        raise TypeError("sentence_hash must be a string")

    data = {
        "version": 1,
        "polygon_id": polygon_id,
        "language": language,
        "sentence_content_hash": sentence_hash,
    }
    # Sorted keys, compact separators, UTF-8, and ensure_ascii=False
    json_str = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest().lower()


def _row_to_stable_string(row: dict[str, Any]) -> str:
    """Construct a stable, canonical JSON string representation of the complete row.

    This maps the row fields based on SEGMENTED_SENTENCES_SCHEMA, ensuring a
    type-stable representation that avoids direct comparison of None with other types.
    """
    data = {f.name: row[f.name] for f in SEGMENTED_SENTENCES_SCHEMA}
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def finalize_sentence_dataset(
    table: pa.Table,
    *,
    input_dataset_revision: str,
    pipeline_version: str,
) -> FinalizedDataset:
    """Validate, enrich, deduplicate, hash, and sort a segmented sentence dataset.

    Parameters
    ----------
    table : pa.Table
        The input segmented sentences table conforming to SEGMENTED_SENTENCES_SCHEMA.
    input_dataset_revision : str
        The revision identifier of the input dataset.
    pipeline_version : str
        The pipeline version identifier.

    Returns
    -------
    FinalizedDataset
        The finalized table and the finalization report.
    """
    # 1. Reject blank/whitespace-only configuration
    if not isinstance(input_dataset_revision, str):
        raise FinalizationError("input_dataset_revision must be a string")
    if not isinstance(pipeline_version, str):
        raise FinalizationError("pipeline_version must be a string")
    if not input_dataset_revision.strip():
        raise FinalizationError(
            "input_dataset_revision cannot be blank or whitespace-only"
        )
    if not pipeline_version.strip():
        raise FinalizationError("pipeline_version cannot be blank or whitespace-only")

    # 2. Validate input table schema compatibility with SEGMENTED_SENTENCES_SCHEMA
    if not isinstance(table, pa.Table):
        raise TypeError("Input must be a pyarrow.Table")

    for field in SEGMENTED_SENTENCES_SCHEMA:
        if field.name not in table.column_names:
            raise FinalizationError(f"Missing required field: {field.name}")
        actual_field = table.schema.field(field.name)
        if actual_field.type != field.type:
            raise FinalizationError(
                f"Type mismatch for field '{field.name}': "
                f"expected {field.type}, got {actual_field.type}"
            )
        if not field.nullable and table.column(field.name).null_count > 0:
            raise FinalizationError(
                f"Field '{field.name}' is non-nullable but contains nulls"
            )

    # 3. Handle empty input
    if table.num_rows == 0:
        metadata = {
            b"input_dataset_revision": input_dataset_revision.encode("utf-8"),
            b"pipeline_version": pipeline_version.encode("utf-8"),
        }
        empty_schema = OUTPUT_SENTENCE_SCHEMA.with_metadata(metadata)
        return FinalizedDataset(
            table=empty_schema.empty_table(),
            report=FinalizationReport(
                input_sentence_occurrence_count=0,
                output_sentence_count=0,
                duplicate_occurrence_count_removed=0,
                cross_source_duplicate_group_count=0,
            ),
        )

    # 4. Validate sources using ALLOWED_SOURCES before context or deduplication
    sources = set(table.column("source").to_pylist())
    invalid_sources = sources - ALLOWED_SOURCES
    if invalid_sources:
        raise FinalizationError(f"Invalid source(s): {invalid_sources}")

    # 5. Context before deduplication: group by (polygon_id, source, document_id, section_id)
    rows = table.to_pylist()

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["polygon_id"], r["source"], r["document_id"], r["section_id"])
        groups.setdefault(key, []).append(r)

    # Sort each group by sentence_index and assign previous_sentence and next_sentence
    for group_rows in groups.values():
        group_rows.sort(key=lambda x: (x["sentence_index"], _row_to_stable_string(x)))
        n = len(group_rows)
        for i in range(n):
            group_rows[i]["previous_sentence"] = (
                group_rows[i - 1]["sentence_text_normalized"] if i > 0 else None
            )
            group_rows[i]["next_sentence"] = (
                group_rows[i + 1]["sentence_text_normalized"] if i < n - 1 else None
            )

    # 6. Exact deduplication key: (polygon_id, language, sentence_text_normalized)
    dedup_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        dedup_key = (r["polygon_id"], r["language"], r["sentence_text_normalized"])
        dedup_groups.setdefault(dedup_key, []).append(r)

    output_rows = []
    cross_source_duplicate_group_count = 0

    for _dedup_key, group_rows in dedup_groups.items():
        # Canonical occurrence selection:
        # Wikipedia before Wikivoyage; then document_id; section_index; section_id; sentence_index; stable row representation.
        def canonical_sort_key(
            item: dict[str, Any],
        ) -> tuple[Any, Any, Any, Any, Any, Any]:
            src_val = 0 if item["source"] == "wikipedia" else 1
            stable_val = _row_to_stable_string(item)
            return (
                src_val,
                item["document_id"],
                item["section_index"],
                item["section_id"],
                item["sentence_index"],
                stable_val,
            )

        group_rows.sort(key=canonical_sort_key)
        canonical = group_rows[0]

        # Calculate hashes
        text_norm = canonical["sentence_text_normalized"]
        sent_hash = sentence_content_hash(text_norm)
        sent_id = deterministic_sentence_id(
            polygon_id=canonical["polygon_id"],
            language=canonical["language"],
            sentence_hash=sent_hash,
        )

        # Deduplication fields
        dup_count = len(group_rows)
        dup_sources = sorted({item["source"] for item in group_rows})

        # Track cross-source groups
        sources_set = {item["source"] for item in group_rows}
        if "wikipedia" in sources_set and "wikivoyage" in sources_set:
            cross_source_duplicate_group_count += 1

        # Build output row conforming to OUTPUT_SENTENCE_SCHEMA
        out_row = {
            "sentence_id": sent_id,
            "polygon_id": canonical["polygon_id"],
            "wikidata": canonical["wikidata"],
            "document_id": canonical["document_id"],
            "article_id": canonical["article_id"],
            "source": canonical["source"],
            "language": canonical["language"],
            "site": canonical["site"],
            "page_title": canonical["page_title"],
            "section_id": canonical["section_id"],
            "section_index": canonical["section_index"],
            "section_path": canonical["section_path"],
            "sentence_index": canonical["sentence_index"],
            "sentence_text_raw": canonical["sentence_text_raw"],
            "sentence_text_normalized": canonical["sentence_text_normalized"],
            "previous_sentence": canonical["previous_sentence"],
            "next_sentence": canonical["next_sentence"],
            "url": canonical["url"],
            "page_id": canonical["page_id"],
            "revision_id": canonical["revision_id"],
            "revision_timestamp": canonical["revision_timestamp"],
            "document_content_hash": canonical["document_content_hash"],
            "section_content_hash": canonical["section_content_hash"],
            "sentence_content_hash": sent_hash,
            "duplicate_occurrence_count": dup_count,
            "duplicate_sources": dup_sources,
            "polygon_name": canonical["polygon_name"],
            "osm_primary_tag": canonical["osm_primary_tag"],
            "osm_tags": canonical["osm_tags"],
            "region": canonical["region"],
            "lat": canonical["lat"],
            "lon": canonical["lon"],
            "input_dataset_revision": input_dataset_revision,
            "pipeline_version": pipeline_version,
        }
        output_rows.append(out_row)

    # 7. Sort output rows by polygon_id, language, sentence_id ascending
    output_rows.sort(key=lambda x: (x["polygon_id"], x["language"], x["sentence_id"]))

    # 8. Convert to pyarrow Table with the exact schema
    out_dict = {}
    for field in OUTPUT_SENTENCE_SCHEMA:
        col_data = [r[field.name] for r in output_rows]
        out_dict[field.name] = pa.array(col_data, type=field.type)

    metadata = {
        b"input_dataset_revision": input_dataset_revision.encode("utf-8"),
        b"pipeline_version": pipeline_version.encode("utf-8"),
    }
    out_table = pa.table(
        out_dict, schema=OUTPUT_SENTENCE_SCHEMA.with_metadata(metadata)
    )

    # 9. Report metrics
    input_count = len(rows)
    output_count = len(output_rows)
    removed_count = input_count - output_count

    report = FinalizationReport(
        input_sentence_occurrence_count=input_count,
        output_sentence_count=output_count,
        duplicate_occurrence_count_removed=removed_count,
        cross_source_duplicate_group_count=cross_source_duplicate_group_count,
    )

    return FinalizedDataset(table=out_table, report=report)
