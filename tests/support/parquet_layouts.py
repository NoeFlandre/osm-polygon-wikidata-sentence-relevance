"""Temporary on-disk Parquet shard layout helpers for integration tests."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.schemas import (
    POLYGON_ARTICLES_SCHEMA,
    POLYGONS_SCHEMA,
    SECTIONS_SCHEMA,
    WIKIPEDIA_DOCUMENTS_SCHEMA,
    WIKIVOYAGE_DOCUMENTS_SCHEMA,
)
from tests.support.arrow_factories import rows_to_table


def write_shard_parquet(
    root: Path,
    shard_key: str,
    *,
    polygons_rows: list[dict[str, list]] | None = None,
    polygon_articles_rows: list[dict[str, list]] | None = None,
    wikipedia_documents_rows: list[dict[str, list]] | None = None,
    wikipedia_sections_rows: list[dict[str, list]] | None = None,
    wikivoyage_documents_rows: list[dict[str, list]] | None = None,
    wikivoyage_sections_rows: list[dict[str, list]] | None = None,
) -> None:
    """Write synthetic Parquet shard files for testing.

    Only writes files for which rows are provided (not None).
    """
    pairs: list[tuple[str, pa.Schema, list[dict[str, list]] | None]] = [
        ("polygons", POLYGONS_SCHEMA, polygons_rows),
        ("polygon_articles", POLYGON_ARTICLES_SCHEMA, polygon_articles_rows),
        ("wikipedia/documents", WIKIPEDIA_DOCUMENTS_SCHEMA, wikipedia_documents_rows),
        ("wikipedia/sections", SECTIONS_SCHEMA, wikipedia_sections_rows),
        (
            "wikivoyage/documents",
            WIKIVOYAGE_DOCUMENTS_SCHEMA,
            wikivoyage_documents_rows,
        ),
        ("wikivoyage/sections", SECTIONS_SCHEMA, wikivoyage_sections_rows),
    ]
    for subdir, schema, rows in pairs:
        if rows is None:
            continue
        dirpath = root / subdir
        dirpath.mkdir(parents=True, exist_ok=True)
        table = rows_to_table(rows, schema)
        pq.write_table(table, dirpath / f"{shard_key}.parquet")
