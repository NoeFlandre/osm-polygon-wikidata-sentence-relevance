"""Compatibility facade: ``tests.helpers``.

Shared fixtures now live in :mod:`tests.support`. Import from there in new
tests; this module re-exports the stable helper names so existing test
imports keep working.
"""

from tests.support import (
    get_checksum,
    make_fake_pipeline_result,
    make_polygon_article_row,
    make_polygon_row,
    make_section_row,
    make_segmented_row,
    make_wikipedia_document_row,
    make_wikivoyage_document_row,
    rows_to_table,
    write_shard_parquet,
)

__all__ = [
    "make_polygon_row",
    "make_polygon_article_row",
    "make_wikipedia_document_row",
    "make_wikivoyage_document_row",
    "make_section_row",
    "rows_to_table",
    "write_shard_parquet",
    "make_fake_pipeline_result",
    "get_checksum",
    "make_segmented_row",
]
