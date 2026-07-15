"""Shared test helpers and small Afghanistan-shaped fixture factories.

All data is tiny, synthetic, and in-memory.  No network, no disk data,
no downloaded fixtures, no pandas.
"""

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

# ===================================================================
# Minimal single-row factories for each table
# ===================================================================

def make_polygon_row(
    *,
    polygon_id: str = "poly-af-1",
    wikidata: str = "Q889",
    name: str = "Afghanistan",
    region: str = "afghanistan-latest",
    tags: str = '{"name":"Afghanistan"}',
    osm_primary_tag: str = "boundary=administrative",
    lat: float = 34.5,
    lon: float = 69.1,
) -> dict[str, list]:
    """Return a minimal polygons row dict."""
    return {
        "polygon_id": [polygon_id],
        "region": [region],
        "source_pbf": ["afghanistan-latest.osm.pbf"],
        "osm_type": ["relation"],
        "osm_id": [303427],
        "wikidata": [wikidata],
        "name": [name],
        "tags": [tags],
        "tag_keys": ["name"],
        "tag_count": [1],
        "osm_primary_tag": [osm_primary_tag],
        "centroid": ["POINT(69.1 34.5)"],
        "lat": [lat],
        "lon": [lon],
        "bbox": ["(60.5,29.4,74.9,38.5)"],
        "geometry": ["POLYGON(...)"],
        "area_m2": [652230.0],
        "area_km2": [652.23],
        "area_bucket": ["large"],
        "has_name": [True],
        "has_wikidata": [True],
        "has_wikipedia": [True],
        "wikipedia_language_count": [1],
        "wikipedia_languages": ["en"],
        "wikipedia_article_count": [1],
        "has_english_wikipedia": [True],
        "has_french_wikipedia": [False],
        "text_available": [True],
        "best_language": ["en"],
        "extraction_version": ["1.0"],
        "extracted_at": ["2024-01-01T00:00:00Z"],
    }


def make_polygon_article_row(
    *,
    polygon_id: str = "poly-af-1",
    article_id: str = "art-wp-af-1",
    wikidata: str = "Q889",
    language: str = "en",
) -> dict[str, list]:
    """Return a minimal polygon_articles row dict."""
    return {
        "polygon_id": [polygon_id],
        "article_id": [article_id],
        "wikidata": [wikidata],
        "language": [language],
        "source_pbf": ["afghanistan-latest.osm.pbf"],
        "region": ["afghanistan-latest"],
        "osm_type": ["relation"],
        "osm_id": [303427],
        "page_id": [100],
        "revision_id": [200],
        "is_best_language": [True],
    }


def make_wikipedia_document_row(
    *,
    document_id: str = "doc-wp-af-1",
    article_id: str = "art-wp-af-1",
    wikidata: str = "Q889",
    title: str = "Afghanistan",
    language: str = "en",
    url: str = "https://en.wikipedia.org/wiki/Afghanistan",
) -> dict[str, list]:
    """Return a minimal wikipedia_documents row dict."""
    return {
        "document_id": [document_id],
        "article_id": [article_id],
        "wikidata": [wikidata],
        "project": ["wikipedia"],
        "language": [language],
        "site": ["en.wikipedia.org"],
        "title": [title],
        "url": [url],
        "page_id": [100],
        "revision_id": [200],
        "revision_timestamp": ["2024-01-01T00:00:00Z"],
        "retrieved_at": ["2024-01-02T00:00:00Z"],
        "wikidata_label": ["Afghanistan"],
        "wikidata_description": ["country in South Asia"],
        "wikidata_aliases": ["AF"],
        "lead_text": ["Afghanistan is a country."],
        "extract": ["Afghanistan is a country."],
        "full_text": ["Afghanistan is a country in South Asia."],
        "full_text_format": ["plain"],
        "article_length_chars": [38],
        "article_length_words": [8],
        "article_length_tokens_estimate": [10],
        "thumbnail_url": ["https://example.com/thumb.jpg"],
        "thumbnail_width": [100],
        "thumbnail_height": [75],
        "categories": ["Countries in Asia"],
        "license": ["CC-BY-SA-4.0"],
        "attribution": ["Wikipedia contributors"],
        "source_api": ["rest"],
        "fetch_status": ["ok"],
        "fetch_error": [""],
        "content_hash": ["abc123"],
    }


def make_wikivoyage_document_row(
    *,
    document_id: str = "doc-wv-af-1",
    article_id: str = "",
    wikidata: str = "Q889",
    title: str = "Afghanistan",
    language: str = "en",
    url: str = "https://en.wikivoyage.org/wiki/Afghanistan",
) -> dict[str, list]:
    """Return a minimal wikivoyage_documents row dict."""
    return {
        "document_id": [document_id],
        "article_id": [article_id],
        "wikidata": [wikidata],
        "project": ["wikivoyage"],
        "language": [language],
        "site": ["en.wikivoyage.org"],
        "title": [title],
        "url": [url],
        "page_id": [500],
        "revision_id": [600],
        "revision_timestamp": ["2024-01-03T00:00:00Z"],
        "retrieved_at": ["2024-01-04T00:00:00Z"],
        "full_text": ["Afghanistan is a travel destination."],
        "full_text_format": ["plain"],
        "article_length_chars": [36],
        "article_length_words": [6],
        "article_length_tokens_estimate": [8],
        "license": ["CC-BY-SA-4.0"],
        "attribution": ["Wikivoyage contributors"],
        "source_api": ["rest"],
        "fetch_status": ["ok"],
        "fetch_error": [""],
        "content_hash": ["wv-hash-1"],
    }


def make_section_row(
    *,
    section_id: str = "sec-1",
    document_id: str = "doc-wp-af-1",
    article_id: str = "art-wp-af-1",
    wikidata: str = "Q889",
    project: str = "wikipedia",
    language: str = "en",
    site: str = "en.wikipedia.org",
    section_index: int = 0,
    heading: str = "Introduction",
    text: str = "Afghanistan is a country in South Asia.",
    section_path: str = '["Introduction"]',
    page_id: int | None = None,
    revision_id: int | None = None,
) -> dict[str, list]:
    """Return a minimal sections row dict (usable for both sources)."""
    if page_id is None:
        page_id = 500 if project == "wikivoyage" else 100
    if revision_id is None:
        revision_id = 600 if project == "wikivoyage" else 200
    return {
        "section_id": [section_id],
        "document_id": [document_id],
        "article_id": [article_id],
        "wikidata": [wikidata],
        "project": [project],
        "language": [language],
        "site": [site],
        "page_id": [page_id],
        "revision_id": [revision_id],
        "section_index": [section_index],
        "heading": [heading],
        "anchor": [heading.lower().replace(" ", "_")],
        "level": [2],
        "parent_section_id": [""],
        "section_path": [section_path],
        "text": [text],
        "text_length_chars": [len(text)],
        "text_length_words": [len(text.split())],
        "text_length_tokens_estimate": [len(text.split()) + 2],
        "content_hash": [f"hash-{section_id}"],
        "license": ["CC-BY-SA-4.0"],
        "attribution": [f"{project} contributors"],
    }


# ===================================================================
# Table construction helpers
# ===================================================================

def rows_to_table(rows: list[dict[str, list]], schema: pa.Schema) -> pa.Table:
    """Merge multiple single-row dicts into one PyArrow table."""
    merged: dict[str, list] = {}
    for row in rows:
        for k, v in row.items():
            merged.setdefault(k, []).extend(v)
    return pa.table(merged, schema=schema)


# ===================================================================
# Temporary Parquet shard layout on disk
# ===================================================================

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
        ("wikivoyage/documents", WIKIVOYAGE_DOCUMENTS_SCHEMA, wikivoyage_documents_rows),
        ("wikivoyage/sections", SECTIONS_SCHEMA, wikivoyage_sections_rows),
    ]
    for subdir, schema, rows in pairs:
        if rows is None:
            continue
        dirpath = root / subdir
        dirpath.mkdir(parents=True, exist_ok=True)
        table = rows_to_table(rows, schema)
        pq.write_table(table, dirpath / f"{shard_key}.parquet")
