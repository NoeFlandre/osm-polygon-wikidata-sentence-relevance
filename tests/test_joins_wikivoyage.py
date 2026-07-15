"""Tests for Wikivoyage join logic.

All data is tiny, synthetic, Afghanistan-shaped, and in-memory.
No network, no disk data, no pandas.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.errors import JoinIntegrityError
from osm_polygon_sentence_relevance.schemas import (
    POLYGONS_SCHEMA,
    SECTIONS_SCHEMA,
    WIKIVOYAGE_DOCUMENTS_SCHEMA,
)

from tests.helpers import (
    make_polygon_row,
    make_section_row,
    make_wikivoyage_document_row,
    rows_to_table,
)


def _polygons(rows: list[dict[str, list]]) -> pa.Table:
    return rows_to_table(rows, POLYGONS_SCHEMA)


def _wv_docs(rows: list[dict[str, list]]) -> pa.Table:
    return rows_to_table(rows, WIKIVOYAGE_DOCUMENTS_SCHEMA)


def _wv_sections(rows: list[dict[str, list]]) -> pa.Table:
    return rows_to_table(rows, SECTIONS_SCHEMA)


class TestWikivoyageJoinExpansion:
    """Two polygons sharing a Wikidata QID both receive all sections."""

    def test_two_polygons_one_wikidata(self):
        from osm_polygon_sentence_relevance.joins import join_wikivoyage_sections

        polygons = _polygons([
            make_polygon_row(polygon_id="poly-1", wikidata="Q889", name="Poly A"),
            make_polygon_row(polygon_id="poly-2", wikidata="Q889", name="Poly B"),
        ])
        wv_docs = _wv_docs([
            make_wikivoyage_document_row(document_id="doc-wv-1", wikidata="Q889"),
        ])
        wv_secs = _wv_sections([
            make_section_row(section_id="sec-wv-1", document_id="doc-wv-1",
                             article_id="", project="wikivoyage",
                             site="en.wikivoyage.org", section_index=0),
            make_section_row(section_id="sec-wv-2", document_id="doc-wv-1",
                             article_id="", project="wikivoyage",
                             site="en.wikivoyage.org", section_index=1,
                             heading="Getting there", text="Fly to Kabul."),
        ])

        result = join_wikivoyage_sections(polygons, wv_docs, wv_secs)

        # 2 polygons × 2 sections = 4 rows
        assert result.num_rows == 4
        pids = sorted(result.column("polygon_id").to_pylist())
        assert pids == ["poly-1", "poly-1", "poly-2", "poly-2"]
        assert all(s == "wikivoyage" for s in result.column("source").to_pylist())


class TestBlankArticleIdConversion:
    """Empty Wikivoyage article_id is converted to null."""

    def test_empty_article_id_becomes_null(self):
        from osm_polygon_sentence_relevance.joins import join_wikivoyage_sections

        polygons = _polygons([make_polygon_row(polygon_id="poly-1", wikidata="Q889")])
        wv_docs = _wv_docs([
            make_wikivoyage_document_row(document_id="doc-wv-1", article_id="", wikidata="Q889"),
        ])
        wv_secs = _wv_sections([
            make_section_row(section_id="sec-wv-1", document_id="doc-wv-1",
                             article_id="", project="wikivoyage",
                             site="en.wikivoyage.org"),
        ])

        result = join_wikivoyage_sections(polygons, wv_docs, wv_secs)
        article_ids = result.column("article_id").to_pylist()
        assert article_ids == [None]

    def test_non_empty_article_id_preserved(self):
        from osm_polygon_sentence_relevance.joins import join_wikivoyage_sections

        polygons = _polygons([make_polygon_row(polygon_id="poly-1", wikidata="Q889")])
        wv_docs = _wv_docs([
            make_wikivoyage_document_row(document_id="doc-wv-1", article_id="art-real", wikidata="Q889"),
        ])
        wv_secs = _wv_sections([
            make_section_row(section_id="sec-wv-1", document_id="doc-wv-1",
                             article_id="art-real", project="wikivoyage",
                             site="en.wikivoyage.org"),
        ])

        result = join_wikivoyage_sections(polygons, wv_docs, wv_secs)
        assert result.column("article_id").to_pylist() == ["art-real"]


class TestWikivoyageOrphans:
    """Unmatched Wikidata and orphan sections produce JoinIntegrityError."""

    def test_unmatched_wikidata(self):
        from osm_polygon_sentence_relevance.joins import join_wikivoyage_sections

        polygons = _polygons([make_polygon_row(polygon_id="poly-1", wikidata="Q111")])
        wv_docs = _wv_docs([
            make_wikivoyage_document_row(document_id="doc-wv-1", wikidata="Q999"),
        ])
        wv_secs = _wv_sections([
            make_section_row(section_id="sec-wv-1", document_id="doc-wv-1",
                             article_id="", project="wikivoyage",
                             site="en.wikivoyage.org"),
        ])

        with pytest.raises(JoinIntegrityError, match="wikidata"):
            join_wikivoyage_sections(polygons, wv_docs, wv_secs)

    def test_orphan_section_missing_document(self):
        from osm_polygon_sentence_relevance.joins import join_wikivoyage_sections

        polygons = _polygons([make_polygon_row(polygon_id="poly-1", wikidata="Q889")])
        wv_docs = _wv_docs([
            make_wikivoyage_document_row(document_id="doc-wv-1", wikidata="Q889"),
        ])
        wv_secs = _wv_sections([
            make_section_row(section_id="sec-wv-1", document_id="doc-ORPHAN",
                             article_id="", project="wikivoyage",
                             site="en.wikivoyage.org"),
        ])

        with pytest.raises(JoinIntegrityError, match="document_id"):
            join_wikivoyage_sections(polygons, wv_docs, wv_secs)
