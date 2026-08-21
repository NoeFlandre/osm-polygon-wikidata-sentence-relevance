"""Projection selection contracts for join inputs."""

from __future__ import annotations

from osm_polygon_sentence_relevance.joins._projection import (
    POLYGON_ARTICLES_COLS,
    POLYGON_ARTICLES_DOCUMENT_COLS,
    polygon_articles_columns,
)


def test_polygon_article_projection_distinguishes_document_layout() -> None:
    assert polygon_articles_columns(("polygon_id", "document_id")) == (
        POLYGON_ARTICLES_DOCUMENT_COLS
    )
    assert polygon_articles_columns(("polygon_id", "document_id", "article_id")) == (
        POLYGON_ARTICLES_COLS
    )
    assert polygon_articles_columns(("polygon_id", "article_id")) == (
        POLYGON_ARTICLES_COLS
    )
