"""Regional join orchestration and report composition.

Composes the Wikipedia and Wikivoyage joins into a single deterministically
sorted table and a statistics report.  Kept separate from the per-source join
algorithms so each concern can be understood independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from osm_polygon_sentence_relevance.discovery import RegionShardSet
from osm_polygon_sentence_relevance.errors import JoinIntegrityError
from osm_polygon_sentence_relevance.joins._wikipedia import join_wikipedia_sections
from osm_polygon_sentence_relevance.joins._wikivoyage import join_wikivoyage_sections


@dataclass(frozen=True)
class JoinReport:
    """Immutable report of join statistics."""

    shard_key: str
    polygon_count: int
    polygon_article_count: int
    wikipedia_document_count: int
    wikipedia_section_count: int
    wikipedia_occurrence_count: int
    wikivoyage_document_count: int
    wikivoyage_section_count: int
    wikivoyage_occurrence_count: int
    total_occurrence_count: int


@dataclass(frozen=True)
class JoinedRegionSections:
    """Result of build_region_section_occurrences."""

    table: pa.Table
    report: JoinReport


# Sort order for deterministic output
_SORT_KEYS = [
    ("polygon_id", "ascending"),
    ("source", "ascending"),
    ("language", "ascending"),
    ("document_id", "ascending"),
    ("section_index", "ascending"),
    ("section_id", "ascending"),
]


def _build_region_section_occurrences_from_tables(
    shards: RegionShardSet,
    polygons: pa.Table,
    polygon_articles: pa.Table,
    wp_documents: pa.Table,
    wp_sections: pa.Table,
    wv_documents: pa.Table | None = None,
    wv_sections: pa.Table | None = None,
) -> JoinedRegionSections:
    """Compose the joined intermediate tables.

    Validates presence consistency of optional Wikivoyage parameters.
    """
    # Table composition boundary consistency check
    if (wv_documents is not None) != (wv_sections is not None):
        raise JoinIntegrityError(
            source="wikivoyage",
            table_name="composition",
            key="wv_documents/wv_sections",
            violation=(
                "Inconsistent Wikivoyage optional inputs: both documents and "
                "sections must be provided together or both omitted"
            ),
            sample=[],
        )

    # Wikipedia join
    wp_result = join_wikipedia_sections(
        polygons, polygon_articles, wp_documents, wp_sections
    )
    wp_occ = wp_result.num_rows

    # Wikivoyage join
    wv_occ = 0
    wv_doc_count = 0
    wv_sec_count = 0
    if wv_documents is not None and wv_sections is not None:
        wv_result = join_wikivoyage_sections(polygons, wv_documents, wv_sections)
        wv_occ = wv_result.num_rows
        wv_doc_count = wv_documents.num_rows
        wv_sec_count = wv_sections.num_rows
        combined = pa.concat_tables([wp_result, wv_result])
    else:
        combined = wp_result

    # Deterministic sort
    sorted_table = combined.sort_by(_SORT_KEYS)

    report = JoinReport(
        shard_key=shards.shard_key,
        polygon_count=polygons.num_rows,
        polygon_article_count=polygon_articles.num_rows,
        wikipedia_document_count=wp_documents.num_rows,
        wikipedia_section_count=wp_sections.num_rows,
        wikipedia_occurrence_count=wp_occ,
        wikivoyage_document_count=wv_doc_count,
        wikivoyage_section_count=wv_sec_count,
        wikivoyage_occurrence_count=wv_occ,
        total_occurrence_count=wp_occ + wv_occ,
    )

    return JoinedRegionSections(table=sorted_table, report=report)
