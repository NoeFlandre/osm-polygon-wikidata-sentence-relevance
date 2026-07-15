"""Tests for Wikipedia join logic.

All data is tiny, synthetic, Afghanistan-shaped, and in-memory.
No network, no disk data, no pandas.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.errors import JoinIntegrityError
from osm_polygon_sentence_relevance.schemas import (
    POLYGON_ARTICLES_SCHEMA,
    POLYGONS_SCHEMA,
    SECTIONS_SCHEMA,
    WIKIPEDIA_DOCUMENTS_SCHEMA,
)

from tests.helpers import (
    make_polygon_row,
    make_polygon_article_row,
    make_section_row,
    make_wikipedia_document_row,
    rows_to_table,
)


def _polygons_table(rows: list[dict[str, list]]) -> pa.Table:
    return rows_to_table(rows, POLYGONS_SCHEMA)


def _pa_table(rows: list[dict[str, list]]) -> pa.Table:
    return rows_to_table(rows, POLYGON_ARTICLES_SCHEMA)


def _wp_docs_table(rows: list[dict[str, list]]) -> pa.Table:
    return rows_to_table(rows, WIKIPEDIA_DOCUMENTS_SCHEMA)


def _wp_sections_table(rows: list[dict[str, list]]) -> pa.Table:
    return rows_to_table(rows, SECTIONS_SCHEMA)


class TestWikipediaJoinExpansion:
    """One article linked to two polygons produces rows for each polygon."""

    def test_two_polygons_one_article(self):
        from osm_polygon_sentence_relevance.joins import join_wikipedia_sections

        polygons = _polygons_table([
            make_polygon_row(polygon_id="poly-1", wikidata="Q889", name="Polygon A"),
            make_polygon_row(polygon_id="poly-2", wikidata="Q889", name="Polygon B"),
        ])
        polygon_articles = _pa_table([
            make_polygon_article_row(polygon_id="poly-1", article_id="art-1"),
            make_polygon_article_row(polygon_id="poly-2", article_id="art-1"),
        ])
        wp_docs = _wp_docs_table([
            make_wikipedia_document_row(document_id="doc-1", article_id="art-1"),
        ])
        wp_sections = _wp_sections_table([
            make_section_row(section_id="sec-1", document_id="doc-1", article_id="art-1", section_index=0),
            make_section_row(section_id="sec-2", document_id="doc-1", article_id="art-1", section_index=1,
                             heading="History", text="History of Afghanistan."),
        ])

        result = join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)

        # 2 polygons × 2 sections = 4 rows
        assert result.num_rows == 4
        # All polygon_ids present
        pids = result.column("polygon_id").to_pylist()
        assert sorted(pids) == ["poly-1", "poly-1", "poly-2", "poly-2"]
        # Source is always "wikipedia"
        sources = result.column("source").to_pylist()
        assert all(s == "wikipedia" for s in sources)

    def test_field_mappings(self):
        from osm_polygon_sentence_relevance.joins import join_wikipedia_sections

        polygons = _polygons_table([
            make_polygon_row(polygon_id="poly-1", name="Poly Name",
                             tags='{"key":"val"}', osm_primary_tag="place=city"),
        ])
        polygon_articles = _pa_table([
            make_polygon_article_row(polygon_id="poly-1", article_id="art-1"),
        ])
        wp_docs = _wp_docs_table([
            make_wikipedia_document_row(document_id="doc-1", article_id="art-1",
                                        title="Afghanistan Page"),
        ])
        wp_sections = _wp_sections_table([
            make_section_row(section_id="sec-1", document_id="doc-1", article_id="art-1",
                             text="Section text here.", section_path='["Intro"]'),
        ])

        result = join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)
        row = {col: result.column(col).to_pylist()[0] for col in result.column_names}

        assert row["page_title"] == "Afghanistan Page"
        assert row["document_content_hash"] == "abc123"
        assert row["section_text_raw"] == "Section text here."
        assert row["section_path_raw"] == '["Intro"]'
        assert row["section_content_hash"] == "hash-sec-1"
        assert row["polygon_name"] == "Poly Name"
        assert row["osm_tags_raw"] == '{"key":"val"}'
        assert row["osm_primary_tag"] == "place=city"


class TestWikipediaOrphanChecks:
    """Orphan rows produce JoinIntegrityError."""

    def test_polygon_article_missing_polygon(self):
        from osm_polygon_sentence_relevance.joins import join_wikipedia_sections

        polygons = _polygons_table([make_polygon_row(polygon_id="poly-1")])
        polygon_articles = _pa_table([
            make_polygon_article_row(polygon_id="poly-MISSING", article_id="art-1"),
        ])
        wp_docs = _wp_docs_table([make_wikipedia_document_row(document_id="doc-1", article_id="art-1")])
        wp_sections = _wp_sections_table([make_section_row(section_id="sec-1", document_id="doc-1")])

        with pytest.raises(JoinIntegrityError, match="polygon_id"):
            join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)

    def test_polygon_article_missing_document(self):
        from osm_polygon_sentence_relevance.joins import join_wikipedia_sections

        polygons = _polygons_table([make_polygon_row(polygon_id="poly-1")])
        polygon_articles = _pa_table([
            make_polygon_article_row(polygon_id="poly-1", article_id="art-MISSING"),
        ])
        wp_docs = _wp_docs_table([make_wikipedia_document_row(document_id="doc-1", article_id="art-1")])
        wp_sections = _wp_sections_table([make_section_row(section_id="sec-1", document_id="doc-1")])

        with pytest.raises(JoinIntegrityError, match="article_id"):
            join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)

    def test_section_missing_document(self):
        from osm_polygon_sentence_relevance.joins import join_wikipedia_sections

        polygons = _polygons_table([make_polygon_row(polygon_id="poly-1")])
        polygon_articles = _pa_table([
            make_polygon_article_row(polygon_id="poly-1", article_id="art-1"),
        ])
        wp_docs = _wp_docs_table([make_wikipedia_document_row(document_id="doc-1", article_id="art-1")])
        wp_sections = _wp_sections_table([
            make_section_row(section_id="sec-1", document_id="doc-ORPHAN"),
        ])

        with pytest.raises(JoinIntegrityError, match="document_id"):
            join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)


class TestWikipediaDuplicateKeys:
    """Duplicate keys produce JoinIntegrityError."""

    def test_duplicate_polygon_id(self):
        from osm_polygon_sentence_relevance.joins import join_wikipedia_sections

        polygons = _polygons_table([
            make_polygon_row(polygon_id="poly-1"),
            make_polygon_row(polygon_id="poly-1"),
        ])
        polygon_articles = _pa_table([
            make_polygon_article_row(polygon_id="poly-1", article_id="art-1"),
        ])
        wp_docs = _wp_docs_table([make_wikipedia_document_row(document_id="doc-1", article_id="art-1")])
        wp_sections = _wp_sections_table([make_section_row(section_id="sec-1", document_id="doc-1")])

        with pytest.raises(JoinIntegrityError, match="polygon_id"):
            join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)

    def test_duplicate_document_id(self):
        from osm_polygon_sentence_relevance.joins import join_wikipedia_sections

        polygons = _polygons_table([make_polygon_row(polygon_id="poly-1")])
        polygon_articles = _pa_table([
            make_polygon_article_row(polygon_id="poly-1", article_id="art-1"),
        ])
        wp_docs = _wp_docs_table([
            make_wikipedia_document_row(document_id="doc-1", article_id="art-1"),
            make_wikipedia_document_row(document_id="doc-1", article_id="art-2"),
        ])
        wp_sections = _wp_sections_table([make_section_row(section_id="sec-1", document_id="doc-1")])

        with pytest.raises(JoinIntegrityError, match="document_id"):
            join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)

    def test_duplicate_polygon_article_pair(self):
        from osm_polygon_sentence_relevance.joins import join_wikipedia_sections

        polygons = _polygons_table([make_polygon_row(polygon_id="poly-1")])
        polygon_articles = _pa_table([
            make_polygon_article_row(polygon_id="poly-1", article_id="art-1"),
            make_polygon_article_row(polygon_id="poly-1", article_id="art-1"),
        ])
        wp_docs = _wp_docs_table([make_wikipedia_document_row(document_id="doc-1", article_id="art-1")])
        wp_sections = _wp_sections_table([make_section_row(section_id="sec-1", document_id="doc-1")])

        with pytest.raises(JoinIntegrityError, match="polygon_id.*article_id"):
            join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)
