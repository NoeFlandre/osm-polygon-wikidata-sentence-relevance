"""Deterministic section-to-polygon joins.

Validates key integrity, builds Wikipedia and Wikivoyage section
occurrences, unions them, and sorts deterministically.

No pandas.  All joins are done with PyArrow compute.

This package is the public import surface previously provided by
``osm_polygon_sentence_relevance.joins``.  Implementation is split into
focused submodules; this module re-exports the stable public API.
"""

from __future__ import annotations

from osm_polygon_sentence_relevance.ingestion.discovery import RegionShardSet
from osm_polygon_sentence_relevance.ingestion.loading import load_validated_table
from osm_polygon_sentence_relevance.joins._composition import (
    JoinedRegionSections,
    JoinReport,
    _build_region_section_occurrences_from_tables,
)
from osm_polygon_sentence_relevance.joins._projection import (
    POLYGON_ARTICLES_COLS,
    POLYGONS_COLS,
    WIKIPEDIA_DOCUMENTS_COLS,
    WIKIPEDIA_SECTIONS_COLS,
    WIKIVOYAGE_DOCUMENTS_COLS,
    WIKIVOYAGE_SECTIONS_COLS,
)
from osm_polygon_sentence_relevance.joins._wikipedia import join_wikipedia_sections
from osm_polygon_sentence_relevance.joins._wikivoyage import join_wikivoyage_sections

__all__ = [
    "build_region_section_occurrences",
    "join_wikipedia_sections",
    "join_wikivoyage_sections",
    "JoinReport",
    "JoinedRegionSections",
    "POLYGONS_COLS",
    "POLYGON_ARTICLES_COLS",
    "WIKIPEDIA_DOCUMENTS_COLS",
    "WIKIPEDIA_SECTIONS_COLS",
    "WIKIVOYAGE_DOCUMENTS_COLS",
    "WIKIVOYAGE_SECTIONS_COLS",
    "_build_region_section_occurrences_from_tables",
]


def build_region_section_occurrences(
    shards: RegionShardSet,
) -> JoinedRegionSections:
    """Build the deterministic intermediate joined-section table from files.

    Loads the required and optional tables using projected columns needed
    by joins, applying pre-projection validation.
    """
    polygons = load_validated_table("polygons", shards.polygons, columns=POLYGONS_COLS)
    polygon_articles = load_validated_table(
        "polygon_articles", shards.polygon_articles, columns=POLYGON_ARTICLES_COLS
    )
    wp_documents = load_validated_table(
        "wikipedia_documents",
        shards.wikipedia_documents,
        columns=WIKIPEDIA_DOCUMENTS_COLS,
    )
    wp_sections = load_validated_table(
        "wikipedia_sections", shards.wikipedia_sections, columns=WIKIPEDIA_SECTIONS_COLS
    )

    wv_documents = None
    wv_sections = None
    if shards.wikivoyage_documents is not None:
        wv_documents = load_validated_table(
            "wikivoyage_documents",
            shards.wikivoyage_documents,
            columns=WIKIVOYAGE_DOCUMENTS_COLS,
        )
    if shards.wikivoyage_sections is not None:
        wv_sections = load_validated_table(
            "wikivoyage_sections",
            shards.wikivoyage_sections,
            columns=WIKIVOYAGE_SECTIONS_COLS,
        )

    return _build_region_section_occurrences_from_tables(
        shards=shards,
        polygons=polygons,
        polygon_articles=polygon_articles,
        wp_documents=wp_documents,
        wp_sections=wp_sections,
        wv_documents=wv_documents,
        wv_sections=wv_sections,
    )
