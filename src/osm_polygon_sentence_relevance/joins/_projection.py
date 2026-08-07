"""Projection column tuples for join inputs.

These are kept close to the code that consumes them so the set of columns
pulled from each input table stays obviously in sync with the joins.
"""

from __future__ import annotations

from collections.abc import Collection

POLYGONS_COLS = (
    "polygon_id",
    "wikidata",
    "name",
    "tags",
    "osm_primary_tag",
    "region",
    "lat",
    "lon",
)

POLYGON_ARTICLES_COLS = (
    "polygon_id",
    "article_id",
    "wikidata",
    "language",
    "page_id",
    "revision_id",
)

POLYGON_ARTICLES_DOCUMENT_COLS = (
    "polygon_id",
    "document_id",
    "project",
    "wikidata",
    "language",
    "page_id",
    "revision_id",
)


def polygon_articles_columns(column_names: Collection[str]) -> tuple[str, ...]:
    """Choose the projection for the legacy or document-key input layout."""

    names = set(column_names)
    if "document_id" in names and "article_id" not in names:
        return POLYGON_ARTICLES_DOCUMENT_COLS
    return POLYGON_ARTICLES_COLS


WIKIPEDIA_DOCUMENTS_COLS = (
    "document_id",
    "article_id",
    "wikidata",
    "language",
    "site",
    "title",
    "url",
    "page_id",
    "revision_id",
    "revision_timestamp",
    "content_hash",
)

WIKIPEDIA_SECTIONS_COLS = (
    "section_id",
    "document_id",
    "article_id",
    "wikidata",
    "language",
    "site",
    "page_id",
    "revision_id",
    "section_index",
    "section_path",
    "text",
    "content_hash",
)

WIKIVOYAGE_DOCUMENTS_COLS = (
    "document_id",
    "article_id",
    "wikidata",
    "language",
    "site",
    "title",
    "url",
    "page_id",
    "revision_id",
    "revision_timestamp",
    "content_hash",
)

WIKIVOYAGE_SECTIONS_COLS = (
    "section_id",
    "document_id",
    "article_id",
    "wikidata",
    "language",
    "site",
    "page_id",
    "revision_id",
    "section_index",
    "section_path",
    "text",
    "content_hash",
)
