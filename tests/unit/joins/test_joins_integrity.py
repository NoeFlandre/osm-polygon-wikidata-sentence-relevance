"""Tests for Phase 2 amendments (TDD)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.discovery import RegionShardSet, discover_shards

# We can also import the internal helper if it exists, or dynamically resolve it
from osm_polygon_sentence_relevance.errors import (
    JoinIntegrityError,
    MissingColumnsError,
)
from osm_polygon_sentence_relevance.joins import (
    build_region_section_occurrences,
    join_wikipedia_sections,
    join_wikivoyage_sections,
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
    write_shard_parquet,
)


def _write_complete_shard_for_test(
    root: Path, shard_key: str = "afghanistan-latest"
) -> None:
    write_shard_parquet(
        root,
        shard_key,
        polygons_rows=[make_polygon_row(polygon_id="poly-1", wikidata="Q889")],
        polygon_articles_rows=[
            make_polygon_article_row(polygon_id="poly-1", article_id="art-wp-1")
        ],
        wikipedia_documents_rows=[
            make_wikipedia_document_row(
                document_id="doc-wp-1", article_id="art-wp-1", wikidata="Q889"
            )
        ],
        wikipedia_sections_rows=[
            make_section_row(
                section_id="sec-wp-1", document_id="doc-wp-1", article_id="art-wp-1"
            )
        ],
        wikivoyage_documents_rows=[
            make_wikivoyage_document_row(document_id="doc-wv-1", wikidata="Q889")
        ],
        wikivoyage_sections_rows=[
            {
                **make_section_row(
                    section_id="sec-wv-1",
                    document_id="doc-wv-1",
                    article_id="",
                    project="wikivoyage",
                    site="en.wikivoyage.org",
                ),
                "page_id": [500],
                "revision_id": [600],
            }
        ],
    )


def _write_wikipedia_only_shard_for_test(
    root: Path, shard_key: str = "afghanistan-latest"
) -> None:
    write_shard_parquet(
        root,
        shard_key,
        polygons_rows=[make_polygon_row(polygon_id="poly-1", wikidata="Q889")],
        polygon_articles_rows=[
            make_polygon_article_row(polygon_id="poly-1", article_id="art-wp-1")
        ],
        wikipedia_documents_rows=[
            make_wikipedia_document_row(
                document_id="doc-wp-1", article_id="art-wp-1", wikidata="Q889"
            )
        ],
        wikipedia_sections_rows=[
            make_section_row(
                section_id="sec-wp-1", document_id="doc-wp-1", article_id="art-wp-1"
            )
        ],
    )


class TestIssue1OrchestrationRed:
    """Issue 1 TDD tests."""

    def test_end_to_end_complete_shard(self, tmp_path: Path):
        _write_complete_shard_for_test(tmp_path, "afghanistan-latest")
        shards = discover_shards(tmp_path)
        assert len(shards) == 1

        result = build_region_section_occurrences(shards[0])

        assert result.table is not None
        assert result.report is not None
        assert result.table.schema == JOINED_SECTIONS_SCHEMA

        sources = result.table.column("source").to_pylist()
        assert "wikipedia" in sources
        assert "wikivoyage" in sources

        assert result.report.wikipedia_occurrence_count == 1
        assert result.report.wikivoyage_occurrence_count == 1

    def test_end_to_end_wikipedia_only_shard(self, tmp_path: Path):
        _write_wikipedia_only_shard_for_test(tmp_path, "afghanistan-latest")
        shards = discover_shards(tmp_path)
        assert len(shards) == 1

        result = build_region_section_occurrences(shards[0])

        assert result.table is not None
        assert result.report is not None
        assert result.table.schema == JOINED_SECTIONS_SCHEMA

        sources = result.table.column("source").to_pylist()
        assert "wikipedia" in sources
        assert "wikivoyage" not in sources

        assert result.report.wikipedia_occurrence_count == 1
        assert result.report.wikivoyage_occurrence_count == 0

    def test_end_to_end_schema_defect_rejection(self, tmp_path: Path):
        bad_polygons_schema = pa.schema(
            [f for f in POLYGONS_SCHEMA if f.name != "name"]
        )
        poly_row = make_polygon_row(polygon_id="poly-1", wikidata="Q889")
        del poly_row["name"]

        poly_table = rows_to_table([poly_row], bad_polygons_schema)

        _write_wikipedia_only_shard_for_test(tmp_path, "afghanistan-latest")
        pq.write_table(poly_table, tmp_path / "polygons" / "afghanistan-latest.parquet")

        shards = discover_shards(tmp_path)
        assert len(shards) == 1

        with pytest.raises(MissingColumnsError):
            build_region_section_occurrences(shards[0])


class TestIssue2IntegrityRed:
    """Issue 2 TDD tests."""

    @pytest.mark.parametrize(
        ("table_name", "schema", "make_row_func", "key_name", "bad_value"),
        [
            ("polygons", POLYGONS_SCHEMA, make_polygon_row, "polygon_id", None),
            ("polygons", POLYGONS_SCHEMA, make_polygon_row, "polygon_id", ""),
            (
                "polygon_articles",
                POLYGON_ARTICLES_SCHEMA,
                make_polygon_article_row,
                "polygon_id",
                None,
            ),
            (
                "polygon_articles",
                POLYGON_ARTICLES_SCHEMA,
                make_polygon_article_row,
                "polygon_id",
                "",
            ),
            (
                "polygon_articles",
                POLYGON_ARTICLES_SCHEMA,
                make_polygon_article_row,
                "article_id",
                None,
            ),
            (
                "polygon_articles",
                POLYGON_ARTICLES_SCHEMA,
                make_polygon_article_row,
                "article_id",
                "",
            ),
            (
                "wikipedia_documents",
                WIKIPEDIA_DOCUMENTS_SCHEMA,
                make_wikipedia_document_row,
                "document_id",
                None,
            ),
            (
                "wikipedia_documents",
                WIKIPEDIA_DOCUMENTS_SCHEMA,
                make_wikipedia_document_row,
                "document_id",
                "",
            ),
            (
                "wikipedia_documents",
                WIKIPEDIA_DOCUMENTS_SCHEMA,
                make_wikipedia_document_row,
                "article_id",
                None,
            ),
            (
                "wikipedia_documents",
                WIKIPEDIA_DOCUMENTS_SCHEMA,
                make_wikipedia_document_row,
                "article_id",
                "",
            ),
            (
                "wikipedia_sections",
                SECTIONS_SCHEMA,
                make_section_row,
                "section_id",
                None,
            ),
            ("wikipedia_sections", SECTIONS_SCHEMA, make_section_row, "section_id", ""),
            (
                "wikipedia_sections",
                SECTIONS_SCHEMA,
                make_section_row,
                "document_id",
                None,
            ),
            (
                "wikipedia_sections",
                SECTIONS_SCHEMA,
                make_section_row,
                "document_id",
                "",
            ),
        ],
    )
    def test_wikipedia_join_keys_null_and_empty(
        self, table_name, schema, make_row_func, key_name, bad_value
    ):
        polygons = rows_to_table(
            [make_polygon_row(polygon_id="poly-1")], POLYGONS_SCHEMA
        )
        polygon_articles = rows_to_table(
            [make_polygon_article_row(polygon_id="poly-1", article_id="art-1")],
            POLYGON_ARTICLES_SCHEMA,
        )
        wp_docs = rows_to_table(
            [make_wikipedia_document_row(document_id="doc-1", article_id="art-1")],
            WIKIPEDIA_DOCUMENTS_SCHEMA,
        )
        wp_sections = rows_to_table(
            [
                make_section_row(
                    section_id="sec-1", document_id="doc-1", article_id="art-1"
                )
            ],
            SECTIONS_SCHEMA,
        )

        row = make_row_func()
        if table_name == "polygon_articles":
            row["polygon_id"] = [bad_value if key_name == "polygon_id" else "poly-1"]
            row["article_id"] = [bad_value if key_name == "article_id" else "art-1"]
        elif table_name == "wikipedia_documents":
            row["document_id"] = [bad_value if key_name == "document_id" else "doc-1"]
            row["article_id"] = [bad_value if key_name == "article_id" else "art-1"]
        elif table_name == "wikipedia_sections":
            row["section_id"] = [bad_value if key_name == "section_id" else "sec-1"]
            row["document_id"] = [bad_value if key_name == "document_id" else "doc-1"]
            row["article_id"] = ["art-1"]

        bad_table = rows_to_table([row], schema)

        if table_name == "polygons":
            polygons = bad_table
        elif table_name == "polygon_articles":
            polygon_articles = bad_table
        elif table_name == "wikipedia_documents":
            wp_docs = bad_table
        elif table_name == "wikipedia_sections":
            wp_sections = bad_table

        with pytest.raises(JoinIntegrityError) as exc_info:
            join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)

        assert table_name in str(exc_info.value)
        assert key_name in str(exc_info.value)

    @pytest.mark.parametrize(
        ("table_name", "schema", "make_row_func", "key_name", "bad_value"),
        [
            (
                "wikivoyage_documents",
                WIKIVOYAGE_DOCUMENTS_SCHEMA,
                make_wikivoyage_document_row,
                "document_id",
                None,
            ),
            (
                "wikivoyage_documents",
                WIKIVOYAGE_DOCUMENTS_SCHEMA,
                make_wikivoyage_document_row,
                "document_id",
                "",
            ),
            (
                "wikivoyage_documents",
                WIKIVOYAGE_DOCUMENTS_SCHEMA,
                make_wikivoyage_document_row,
                "wikidata",
                None,
            ),
            (
                "wikivoyage_documents",
                WIKIVOYAGE_DOCUMENTS_SCHEMA,
                make_wikivoyage_document_row,
                "wikidata",
                "",
            ),
            (
                "wikivoyage_sections",
                SECTIONS_SCHEMA,
                make_section_row,
                "section_id",
                None,
            ),
            (
                "wikivoyage_sections",
                SECTIONS_SCHEMA,
                make_section_row,
                "section_id",
                "",
            ),
            (
                "wikivoyage_sections",
                SECTIONS_SCHEMA,
                make_section_row,
                "document_id",
                None,
            ),
            (
                "wikivoyage_sections",
                SECTIONS_SCHEMA,
                make_section_row,
                "document_id",
                "",
            ),
        ],
    )
    def test_wikivoyage_join_keys_null_and_empty(
        self, table_name, schema, make_row_func, key_name, bad_value
    ):
        polygons = rows_to_table(
            [make_polygon_row(polygon_id="poly-1", wikidata="Q889")], POLYGONS_SCHEMA
        )
        wv_docs = rows_to_table(
            [make_wikivoyage_document_row(document_id="doc-1", wikidata="Q889")],
            WIKIVOYAGE_DOCUMENTS_SCHEMA,
        )
        wv_sections = rows_to_table(
            [
                make_section_row(
                    section_id="sec-1",
                    document_id="doc-1",
                    article_id="",
                    project="wikivoyage",
                )
            ],
            SECTIONS_SCHEMA,
        )

        row = make_row_func()
        if table_name == "wikivoyage_sections":
            row["project"] = ["wikivoyage"]
            row["site"] = ["en.wikivoyage.org"]
            row["section_id"] = [bad_value if key_name == "section_id" else "sec-1"]
            row["document_id"] = [bad_value if key_name == "document_id" else "doc-1"]
        elif table_name == "wikivoyage_documents":
            row["document_id"] = [bad_value if key_name == "document_id" else "doc-1"]
            row["wikidata"] = [bad_value if key_name == "wikidata" else "Q889"]

        bad_table = rows_to_table([row], schema)

        if table_name == "polygons":
            polygons = bad_table
        elif table_name == "wikivoyage_documents":
            wv_docs = bad_table
        elif table_name == "wikivoyage_sections":
            wv_sections = bad_table

        with pytest.raises(JoinIntegrityError) as exc_info:
            join_wikivoyage_sections(polygons, wv_docs, wv_sections)

        assert table_name in str(exc_info.value)
        assert key_name in str(exc_info.value)

    @pytest.mark.parametrize("section_index_val", [None, -1, -99])
    def test_section_index_invalid(self, section_index_val):
        polygons = rows_to_table(
            [make_polygon_row(polygon_id="poly-1")], POLYGONS_SCHEMA
        )
        polygon_articles = rows_to_table(
            [make_polygon_article_row(polygon_id="poly-1", article_id="art-1")],
            POLYGON_ARTICLES_SCHEMA,
        )
        wp_docs = rows_to_table(
            [make_wikipedia_document_row(document_id="doc-1", article_id="art-1")],
            WIKIPEDIA_DOCUMENTS_SCHEMA,
        )

        row = make_section_row(document_id="doc-1", article_id="art-1")
        row["section_index"] = [section_index_val]
        wp_sections = rows_to_table([row], SECTIONS_SCHEMA)

        with pytest.raises(JoinIntegrityError) as exc_info:
            join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)

        assert "section_index" in str(exc_info.value)


class TestIssue3WikivoyageArgsRed:
    """Issue 3 TDD tests."""

    def test_documents_without_sections_raises(self):
        from osm_polygon_sentence_relevance.joins import (
            _build_region_section_occurrences_from_tables,
        )

        polygons = rows_to_table(
            [make_polygon_row(polygon_id="poly-1")], POLYGONS_SCHEMA
        )
        polygon_articles = rows_to_table(
            [make_polygon_article_row(polygon_id="poly-1", article_id="art-1")],
            POLYGON_ARTICLES_SCHEMA,
        )
        wp_docs = rows_to_table(
            [make_wikipedia_document_row(document_id="doc-1", article_id="art-1")],
            WIKIPEDIA_DOCUMENTS_SCHEMA,
        )
        wp_sections = rows_to_table(
            [
                make_section_row(
                    section_id="sec-1", document_id="doc-1", article_id="art-1"
                )
            ],
            SECTIONS_SCHEMA,
        )
        wv_docs = rows_to_table(
            [make_wikivoyage_document_row(document_id="doc-wv-1")],
            WIKIVOYAGE_DOCUMENTS_SCHEMA,
        )

        # Calling with wv_documents but without wv_sections should raise ValueError or JoinIntegrityError
        with pytest.raises((ValueError, JoinIntegrityError), match="Wikivoyage"):
            _build_region_section_occurrences_from_tables(
                shards=RegionShardSet("af", None, None, None, None, None, None),
                polygons=polygons,
                polygon_articles=polygon_articles,
                wp_documents=wp_docs,
                wp_sections=wp_sections,
                wv_documents=wv_docs,
                wv_sections=None,
            )

    def test_sections_without_documents_raises(self):
        from osm_polygon_sentence_relevance.joins import (
            _build_region_section_occurrences_from_tables,
        )

        polygons = rows_to_table(
            [make_polygon_row(polygon_id="poly-1")], POLYGONS_SCHEMA
        )
        polygon_articles = rows_to_table(
            [make_polygon_article_row(polygon_id="poly-1", article_id="art-1")],
            POLYGON_ARTICLES_SCHEMA,
        )
        wp_docs = rows_to_table(
            [make_wikipedia_document_row(document_id="doc-1", article_id="art-1")],
            WIKIPEDIA_DOCUMENTS_SCHEMA,
        )
        wp_sections = rows_to_table(
            [
                make_section_row(
                    section_id="sec-1", document_id="doc-1", article_id="art-1"
                )
            ],
            SECTIONS_SCHEMA,
        )
        wv_sections = rows_to_table(
            [
                make_section_row(
                    section_id="sec-wv-1", document_id="doc-wv-1", article_id=""
                )
            ],
            SECTIONS_SCHEMA,
        )

        # Calling with wv_sections but without wv_documents should raise ValueError or JoinIntegrityError
        with pytest.raises((ValueError, JoinIntegrityError), match="Wikivoyage"):
            _build_region_section_occurrences_from_tables(
                shards=RegionShardSet("af", None, None, None, None, None, None),
                polygons=polygons,
                polygon_articles=polygon_articles,
                wp_documents=wp_docs,
                wp_sections=wp_sections,
                wv_documents=None,
                wv_sections=wv_sections,
            )


class TestIssue5IdentityConsistencyRed:
    """Issue 5 TDD tests."""

    def test_polygon_articles_wikidata_mismatch_raises(self):
        polygons = rows_to_table(
            [make_polygon_row(polygon_id="poly-1", wikidata="Q111")], POLYGONS_SCHEMA
        )
        polygon_articles = rows_to_table(
            [
                make_polygon_article_row(
                    polygon_id="poly-1", article_id="art-1", wikidata="Q222"
                )
            ],
            POLYGON_ARTICLES_SCHEMA,
        )
        wp_docs = rows_to_table(
            [
                make_wikipedia_document_row(
                    document_id="doc-1", article_id="art-1", wikidata="Q111"
                )
            ],
            WIKIPEDIA_DOCUMENTS_SCHEMA,
        )
        wp_sections = rows_to_table(
            [
                make_section_row(
                    section_id="sec-1",
                    document_id="doc-1",
                    article_id="art-1",
                    wikidata="Q111",
                )
            ],
            SECTIONS_SCHEMA,
        )

        with pytest.raises(JoinIntegrityError, match="wikidata"):
            join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)

    def test_polygon_articles_document_wikidata_mismatch_raises(self):
        polygons = rows_to_table(
            [make_polygon_row(polygon_id="poly-1", wikidata="Q111")], POLYGONS_SCHEMA
        )
        polygon_articles = rows_to_table(
            [
                make_polygon_article_row(
                    polygon_id="poly-1", article_id="art-1", wikidata="Q111"
                )
            ],
            POLYGON_ARTICLES_SCHEMA,
        )
        wp_docs = rows_to_table(
            [
                make_wikipedia_document_row(
                    document_id="doc-1", article_id="art-1", wikidata="Q333"
                )
            ],
            WIKIPEDIA_DOCUMENTS_SCHEMA,
        )
        wp_sections = rows_to_table(
            [
                make_section_row(
                    section_id="sec-1",
                    document_id="doc-1",
                    article_id="art-1",
                    wikidata="Q111",
                )
            ],
            SECTIONS_SCHEMA,
        )

        with pytest.raises(JoinIntegrityError, match="wikidata"):
            join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)

    def test_polygon_articles_language_mismatch_raises(self):
        polygons = rows_to_table(
            [make_polygon_row(polygon_id="poly-1")], POLYGONS_SCHEMA
        )
        polygon_articles = rows_to_table(
            [
                make_polygon_article_row(
                    polygon_id="poly-1", article_id="art-1", language="fr"
                )
            ],
            POLYGON_ARTICLES_SCHEMA,
        )
        wp_docs = rows_to_table(
            [
                make_wikipedia_document_row(
                    document_id="doc-1", article_id="art-1", language="en"
                )
            ],
            WIKIPEDIA_DOCUMENTS_SCHEMA,
        )
        wp_sections = rows_to_table(
            [
                make_section_row(
                    section_id="sec-1",
                    document_id="doc-1",
                    article_id="art-1",
                    language="en",
                )
            ],
            SECTIONS_SCHEMA,
        )

        with pytest.raises(JoinIntegrityError, match="language"):
            join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)

    def test_section_wikidata_mismatch_raises(self):
        polygons = rows_to_table(
            [make_polygon_row(polygon_id="poly-1")], POLYGONS_SCHEMA
        )
        polygon_articles = rows_to_table(
            [make_polygon_article_row(polygon_id="poly-1", article_id="art-1")],
            POLYGON_ARTICLES_SCHEMA,
        )
        wp_docs = rows_to_table(
            [
                make_wikipedia_document_row(
                    document_id="doc-1", article_id="art-1", wikidata="Q111"
                )
            ],
            WIKIPEDIA_DOCUMENTS_SCHEMA,
        )
        wp_sections = rows_to_table(
            [
                make_section_row(
                    section_id="sec-1",
                    document_id="doc-1",
                    article_id="art-1",
                    wikidata="Q999",
                )
            ],
            SECTIONS_SCHEMA,
        )

        with pytest.raises(JoinIntegrityError, match="wikidata"):
            join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)

    def test_section_article_id_mismatch_raises(self):
        polygons = rows_to_table(
            [make_polygon_row(polygon_id="poly-1")], POLYGONS_SCHEMA
        )
        polygon_articles = rows_to_table(
            [make_polygon_article_row(polygon_id="poly-1", article_id="art-1")],
            POLYGON_ARTICLES_SCHEMA,
        )
        wp_docs = rows_to_table(
            [make_wikipedia_document_row(document_id="doc-1", article_id="art-1")],
            WIKIPEDIA_DOCUMENTS_SCHEMA,
        )
        wp_sections = rows_to_table(
            [
                make_section_row(
                    section_id="sec-1", document_id="doc-1", article_id="art-mismatch"
                )
            ],
            SECTIONS_SCHEMA,
        )

        with pytest.raises(JoinIntegrityError, match="article_id"):
            join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)
