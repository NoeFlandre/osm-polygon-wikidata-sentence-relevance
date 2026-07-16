"""Test support factories and builders.

This package centralizes the shared, deterministic fixtures used across
the test suite. The legacy ``tests.helpers`` module re-exports these names
so older import statements keep working; prefer importing from
``tests.support`` in new tests.
"""

from tests.support.arrow_factories import (
    make_polygon_article_row,
    make_polygon_row,
    make_section_row,
    make_wikipedia_document_row,
    make_wikivoyage_document_row,
    rows_to_table,
)
from tests.support.checksums import get_checksum
from tests.support.fake_results import (
    make_fake_pipeline_result,
    make_segmented_row,
)
from tests.support.parquet_layouts import write_shard_parquet

__all__ = [
    "make_polygon_row",
    "make_polygon_article_row",
    "make_wikipedia_document_row",
    "make_wikivoyage_document_row",
    "make_section_row",
    "rows_to_table",
    "write_shard_parquet",
    "get_checksum",
    "make_fake_pipeline_result",
    "make_segmented_row",
]
