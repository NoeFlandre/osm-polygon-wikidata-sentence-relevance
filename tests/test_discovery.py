"""Tests for shard discovery.

All tests use tmp_path fixtures with tiny synthetic Parquet files.
No network, no downloaded data, no external storage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osm_polygon_sentence_relevance.errors import ShardDiscoveryError
from tests.helpers import (
    make_polygon_article_row,
    make_polygon_row,
    make_section_row,
    make_wikipedia_document_row,
    make_wikivoyage_document_row,
    write_shard_parquet,
)


def _write_complete_shard(root: Path, shard_key: str = "afghanistan-latest") -> None:
    """Write a complete six-file shard."""
    write_shard_parquet(
        root,
        shard_key,
        polygons_rows=[make_polygon_row()],
        polygon_articles_rows=[make_polygon_article_row()],
        wikipedia_documents_rows=[make_wikipedia_document_row()],
        wikipedia_sections_rows=[make_section_row()],
        wikivoyage_documents_rows=[make_wikivoyage_document_row()],
        wikivoyage_sections_rows=[
            make_section_row(
                section_id="sec-wv-1",
                document_id="doc-wv-af-1",
                article_id="",
                project="wikivoyage",
                site="en.wikivoyage.org",
            )
        ],
    )


def _write_wikipedia_only_shard(
    root: Path, shard_key: str = "afghanistan-latest"
) -> None:
    """Write a shard with core files only (no Wikivoyage)."""
    write_shard_parquet(
        root,
        shard_key,
        polygons_rows=[make_polygon_row()],
        polygon_articles_rows=[make_polygon_article_row()],
        wikipedia_documents_rows=[make_wikipedia_document_row()],
        wikipedia_sections_rows=[make_section_row()],
    )


class TestCompleteShardDiscovery:
    """A complete six-file shard is discovered correctly."""

    def test_complete_shard(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.discovery import discover_shards

        _write_complete_shard(tmp_path)
        shards = discover_shards(tmp_path)
        assert len(shards) == 1
        s = shards[0]
        assert s.shard_key == "afghanistan-latest"
        assert s.polygons == tmp_path / "polygons" / "afghanistan-latest.parquet"
        assert (
            s.polygon_articles
            == tmp_path / "polygon_articles" / "afghanistan-latest.parquet"
        )
        assert (
            s.wikipedia_documents
            == tmp_path / "wikipedia" / "documents" / "afghanistan-latest.parquet"
        )
        assert (
            s.wikipedia_sections
            == tmp_path / "wikipedia" / "sections" / "afghanistan-latest.parquet"
        )
        assert (
            s.wikivoyage_documents
            == tmp_path / "wikivoyage" / "documents" / "afghanistan-latest.parquet"
        )
        assert (
            s.wikivoyage_sections
            == tmp_path / "wikivoyage" / "sections" / "afghanistan-latest.parquet"
        )


class TestWikipediaOnlyShard:
    """A shard with core files but no Wikivoyage is processable."""

    def test_wikipedia_only(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.discovery import discover_shards

        _write_wikipedia_only_shard(tmp_path)
        shards = discover_shards(tmp_path)
        assert len(shards) == 1
        s = shards[0]
        assert s.shard_key == "afghanistan-latest"
        assert s.wikivoyage_documents is None
        assert s.wikivoyage_sections is None


class TestMissingCoreFile:
    """Missing a core file raises ShardDiscoveryError."""

    def test_missing_polygon_articles(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.discovery import discover_shards

        # Write only polygons, wp_documents, wp_sections (missing polygon_articles)
        write_shard_parquet(
            tmp_path,
            "afghanistan-latest",
            polygons_rows=[make_polygon_row()],
            wikipedia_documents_rows=[make_wikipedia_document_row()],
            wikipedia_sections_rows=[make_section_row()],
        )
        with pytest.raises(ShardDiscoveryError, match="afghanistan-latest"):
            discover_shards(tmp_path)


class TestHalfPresentWikivoyage:
    """Only one Wikivoyage file present raises ShardDiscoveryError."""

    def test_wikivoyage_docs_without_sections(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.discovery import discover_shards

        write_shard_parquet(
            tmp_path,
            "afghanistan-latest",
            polygons_rows=[make_polygon_row()],
            polygon_articles_rows=[make_polygon_article_row()],
            wikipedia_documents_rows=[make_wikipedia_document_row()],
            wikipedia_sections_rows=[make_section_row()],
            wikivoyage_documents_rows=[make_wikivoyage_document_row()],
            # No wikivoyage_sections
        )
        with pytest.raises(ShardDiscoveryError, match="wikivoyage"):
            discover_shards(tmp_path)

    def test_wikivoyage_sections_without_docs(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.discovery import discover_shards

        write_shard_parquet(
            tmp_path,
            "afghanistan-latest",
            polygons_rows=[make_polygon_row()],
            polygon_articles_rows=[make_polygon_article_row()],
            wikipedia_documents_rows=[make_wikipedia_document_row()],
            wikipedia_sections_rows=[make_section_row()],
            wikivoyage_sections_rows=[
                make_section_row(
                    section_id="sec-wv-1",
                    document_id="doc-wv-af-1",
                    article_id="",
                    project="wikivoyage",
                    site="en.wikivoyage.org",
                )
            ],
            # No wikivoyage_documents
        )
        with pytest.raises(ShardDiscoveryError, match="wikivoyage"):
            discover_shards(tmp_path)


class TestMultipleShards:
    """Multiple shards are discovered and sorted lexicographically."""

    def test_sorted_order(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.discovery import discover_shards

        _write_wikipedia_only_shard(tmp_path, "zambia-latest")
        _write_wikipedia_only_shard(tmp_path, "afghanistan-latest")
        shards = discover_shards(tmp_path)
        assert len(shards) == 2
        assert shards[0].shard_key == "afghanistan-latest"
        assert shards[1].shard_key == "zambia-latest"


class TestNonParquetIgnored:
    """Non-parquet files are ignored."""

    def test_txt_file_ignored(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.discovery import discover_shards

        _write_wikipedia_only_shard(tmp_path)
        # Add a non-parquet file
        (tmp_path / "polygons" / "junk.txt").write_text("not parquet")
        shards = discover_shards(tmp_path)
        assert len(shards) == 1


class TestArticlesNeverScanned:
    """The obsolete articles/ directory is never scanned."""

    def test_articles_trap(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.discovery import discover_shards

        _write_wikipedia_only_shard(tmp_path)
        # Create an articles/ trap with a parquet file
        articles_dir = tmp_path / "articles"
        articles_dir.mkdir()
        (articles_dir / "afghanistan-latest.parquet").write_bytes(b"trap")
        shards = discover_shards(tmp_path)
        # Should still find exactly one shard from the legitimate paths
        assert len(shards) == 1
        assert shards[0].shard_key == "afghanistan-latest"


class TestEmptyRoot:
    """Empty root returns empty tuple."""

    def test_empty(self, tmp_path: Path):
        from osm_polygon_sentence_relevance.discovery import discover_shards

        shards = discover_shards(tmp_path)
        assert shards == ()
