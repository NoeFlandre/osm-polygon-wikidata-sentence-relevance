"""Tests for combined Wikipedia/Wikivoyage union, report generation, and determinism.

Uses tiny in-memory Afghanistan-shaped Tables.
"""

from __future__ import annotations

import random

import pyarrow as pa

from osm_polygon_sentence_relevance.discovery import RegionShardSet
from osm_polygon_sentence_relevance.joins import (
    _build_region_section_occurrences_from_tables,
)
from osm_polygon_sentence_relevance.schemas import (
    JOINED_SECTIONS_SCHEMA,
    POLYGON_ARTICLES_SCHEMA,
    POLYGONS_SCHEMA,
    SECTIONS_SCHEMA,
    WIKIPEDIA_DOCUMENTS_SCHEMA,
    WIKIVOYAGE_DOCUMENTS_SCHEMA,
)
from tests.helpers import (
    make_polygon_article_row,
    make_polygon_row,
    make_section_row,
    make_wikipedia_document_row,
    make_wikivoyage_document_row,
    rows_to_table,
)

# Helper to build mock RegionShardSet
MOCK_SHARDS = RegionShardSet(
    shard_key="afghanistan-latest",
    polygons=None,
    polygon_articles=None,
    wikipedia_documents=None,
    wikipedia_sections=None,
    wikivoyage_documents=None,
    wikivoyage_sections=None,
)


def _shuffle_table(table: pa.Table) -> pa.Table:
    """Deterministically shuffle the rows of a PyArrow table using random indices."""
    num_rows = table.num_rows
    if num_rows <= 1:
        return table
    indices = list(range(num_rows))
    # Use fixed seed for deterministic shuffling
    random.Random(42).shuffle(indices)
    return table.take(pa.array(indices))


class TestCombinedJoins:
    def test_combined_wikipedia_and_wikivoyage(self):
        # We need a polygon linked to Wikipedia and Wikivoyage
        polygons = rows_to_table(
            [
                make_polygon_row(polygon_id="poly-1", wikidata="Q889", name="Poly 1"),
            ],
            POLYGONS_SCHEMA,
        )

        polygon_articles = rows_to_table(
            [
                make_polygon_article_row(polygon_id="poly-1", article_id="art-wp-1"),
            ],
            POLYGON_ARTICLES_SCHEMA,
        )

        wp_docs = rows_to_table(
            [
                make_wikipedia_document_row(
                    document_id="doc-wp-1", article_id="art-wp-1", wikidata="Q889"
                ),
            ],
            WIKIPEDIA_DOCUMENTS_SCHEMA,
        )

        wp_sections = rows_to_table(
            [
                make_section_row(
                    section_id="sec-wp-1",
                    document_id="doc-wp-1",
                    article_id="art-wp-1",
                    section_index=0,
                ),
            ],
            SECTIONS_SCHEMA,
        )

        wv_docs = rows_to_table(
            [
                make_wikivoyage_document_row(document_id="doc-wv-1", wikidata="Q889"),
            ],
            WIKIVOYAGE_DOCUMENTS_SCHEMA,
        )

        wv_sections = rows_to_table(
            [
                make_section_row(
                    section_id="sec-wv-1",
                    document_id="doc-wv-1",
                    article_id="",
                    project="wikivoyage",
                    site="en.wikivoyage.org",
                    section_index=0,
                ),
            ],
            SECTIONS_SCHEMA,
        )

        result = _build_region_section_occurrences_from_tables(
            shards=MOCK_SHARDS,
            polygons=polygons,
            polygon_articles=polygon_articles,
            wp_documents=wp_docs,
            wp_sections=wp_sections,
            wv_documents=wv_docs,
            wv_sections=wv_sections,
        )

        assert result.table.num_rows == 2
        # Wikipedia and Wikivoyage both present
        sources = result.table.column("source").to_pylist()
        assert "wikipedia" in sources
        assert "wikivoyage" in sources

        # Verify schema columns, order, types, and nullability
        assert result.table.schema == JOINED_SECTIONS_SCHEMA

        # Verify report
        assert result.report.shard_key == "afghanistan-latest"
        assert result.report.polygon_count == 1
        assert result.report.polygon_article_count == 1
        assert result.report.wikipedia_document_count == 1
        assert result.report.wikipedia_section_count == 1
        assert result.report.wikipedia_occurrence_count == 1
        assert result.report.wikivoyage_document_count == 1
        assert result.report.wikivoyage_section_count == 1
        assert result.report.wikivoyage_occurrence_count == 1
        assert result.report.total_occurrence_count == 2

    def test_wikipedia_only_behavior(self):
        polygons = rows_to_table(
            [
                make_polygon_row(polygon_id="poly-1", wikidata="Q889", name="Poly 1"),
            ],
            POLYGONS_SCHEMA,
        )

        polygon_articles = rows_to_table(
            [
                make_polygon_article_row(polygon_id="poly-1", article_id="art-wp-1"),
            ],
            POLYGON_ARTICLES_SCHEMA,
        )

        wp_docs = rows_to_table(
            [
                make_wikipedia_document_row(
                    document_id="doc-wp-1", article_id="art-wp-1", wikidata="Q889"
                ),
            ],
            WIKIPEDIA_DOCUMENTS_SCHEMA,
        )

        wp_sections = rows_to_table(
            [
                make_section_row(
                    section_id="sec-wp-1",
                    document_id="doc-wp-1",
                    article_id="art-wp-1",
                    section_index=0,
                ),
            ],
            SECTIONS_SCHEMA,
        )

        result = _build_region_section_occurrences_from_tables(
            shards=MOCK_SHARDS,
            polygons=polygons,
            polygon_articles=polygon_articles,
            wp_documents=wp_docs,
            wp_sections=wp_sections,
            wv_documents=None,
            wv_sections=None,
        )

        assert result.table.num_rows == 1
        assert result.table.column("source").to_pylist() == ["wikipedia"]
        assert result.table.schema == JOINED_SECTIONS_SCHEMA

        # Verify report has 0 counts for Wikivoyage
        assert result.report.wikivoyage_document_count == 0
        assert result.report.wikivoyage_section_count == 0
        assert result.report.wikivoyage_occurrence_count == 0
        assert result.report.total_occurrence_count == 1

    def test_determinism_under_shuffle(self):
        # We need multiple rows to test sorting and determinism
        poly_list = [
            make_polygon_row(polygon_id="poly-2", wikidata="Q2", name="Poly 2"),
            make_polygon_row(polygon_id="poly-1", wikidata="Q1", name="Poly 1"),
        ]
        pa_list = [
            make_polygon_article_row(
                polygon_id="poly-2", article_id="art-2", wikidata="Q2"
            ),
            make_polygon_article_row(
                polygon_id="poly-1", article_id="art-1", wikidata="Q1"
            ),
        ]
        wp_doc_list = [
            make_wikipedia_document_row(
                document_id="doc-wp-2", article_id="art-2", wikidata="Q2", language="en"
            ),
            make_wikipedia_document_row(
                document_id="doc-wp-1", article_id="art-1", wikidata="Q1", language="en"
            ),
        ]
        wp_sec_list = [
            make_section_row(
                section_id="sec-wp-2b",
                document_id="doc-wp-2",
                article_id="art-2",
                wikidata="Q2",
                section_index=1,
            ),
            make_section_row(
                section_id="sec-wp-2a",
                document_id="doc-wp-2",
                article_id="art-2",
                wikidata="Q2",
                section_index=0,
            ),
            make_section_row(
                section_id="sec-wp-1",
                document_id="doc-wp-1",
                article_id="art-1",
                wikidata="Q1",
                section_index=0,
            ),
        ]

        polygons = rows_to_table(poly_list, POLYGONS_SCHEMA)
        polygon_articles = rows_to_table(pa_list, POLYGON_ARTICLES_SCHEMA)
        wp_docs = rows_to_table(wp_doc_list, WIKIPEDIA_DOCUMENTS_SCHEMA)
        wp_sections = rows_to_table(wp_sec_list, SECTIONS_SCHEMA)

        # Build first set of occurrences (sorted)
        res1 = _build_region_section_occurrences_from_tables(
            shards=MOCK_SHARDS,
            polygons=polygons,
            polygon_articles=polygon_articles,
            wp_documents=wp_docs,
            wp_sections=wp_sections,
        )

        # Now shuffle inputs and build again
        polygons_shuf = _shuffle_table(polygons)
        pa_shuf = _shuffle_table(polygon_articles)
        wp_docs_shuf = _shuffle_table(wp_docs)
        wp_sections_shuf = _shuffle_table(wp_sections)

        res2 = _build_region_section_occurrences_from_tables(
            shards=MOCK_SHARDS,
            polygons=polygons_shuf,
            polygon_articles=pa_shuf,
            wp_documents=wp_docs_shuf,
            wp_sections=wp_sections_shuf,
        )

        # Verify tables are identical
        assert res1.table.equals(res2.table)
        assert res1.report == res2.report

        # Verify sort order is deterministic:
        # 1. polygon_id
        # 2. source
        # 3. language
        # 4. document_id
        # 5. section_index
        # 6. section_id
        pids = res1.table.column("polygon_id").to_pylist()
        sec_indices = res1.table.column("section_index").to_pylist()
        sec_ids = res1.table.column("section_id").to_pylist()

        # poly-1 should come first, then poly-2
        assert pids == ["poly-1", "poly-2", "poly-2"]
        # for poly-2, sec-wp-2a (index 0) should come before sec-wp-2b (index 1)
        assert sec_indices == [0, 0, 1]
        assert sec_ids == ["sec-wp-1", "sec-wp-2a", "sec-wp-2b"]
