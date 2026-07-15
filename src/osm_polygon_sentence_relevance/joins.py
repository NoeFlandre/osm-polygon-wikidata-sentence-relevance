"""Deterministic section-to-polygon joins.

Validates key integrity, builds Wikipedia and Wikivoyage section
occurrences, unions them, and sorts deterministically.

No pandas.  All joins are done with PyArrow compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from osm_polygon_sentence_relevance.errors import JoinIntegrityError
from osm_polygon_sentence_relevance.schemas import JOINED_SECTIONS_SCHEMA
from osm_polygon_sentence_relevance.loading import load_validated_table


# ===================================================================
# Projection column tuples close to the code that consumes them
# ===================================================================
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


# ===================================================================
# Integrity checks
# ===================================================================

def _check_non_empty(
    table: pa.Table,
    column: str,
    source: str,
    table_name: str,
) -> None:
    """Raise JoinIntegrityError if *column* has nulls or empty strings."""
    col = table.column(column)
    null_count = col.null_count
    if null_count > 0:
        raise JoinIntegrityError(
            source,
            table_name,
            column,
            "contains nulls",
            ["<null>"] * min(null_count, 5),
        )
    values = col.to_pylist()
    empties = [v for v in values if v == ""]
    if empties:
        raise JoinIntegrityError(
            source,
            table_name,
            column,
            "contains empty strings",
            empties[:5],
        )


def _check_unique(
    table: pa.Table,
    column: str,
    source: str,
    table_name: str,
) -> None:
    """Raise JoinIntegrityError if *column* has duplicate values."""
    values = table.column(column).to_pylist()
    seen: set = set()
    dupes: list[str] = []
    for v in values:
        if v is not None:
            if v in seen and v not in dupes:
                dupes.append(str(v))
            seen.add(v)
    if dupes:
        raise JoinIntegrityError(
            source,
            table_name,
            column,
            "contains duplicates",
            dupes[:5],
        )


def _check_unique_pairs(
    table: pa.Table,
    col_a: str,
    col_b: str,
    source: str,
    table_name: str,
) -> None:
    """Raise JoinIntegrityError if (col_a, col_b) pairs are not unique."""
    a_vals = table.column(col_a).to_pylist()
    b_vals = table.column(col_b).to_pylist()
    seen: set[tuple] = set()
    dupes: list[str] = []
    for a, b in zip(a_vals, b_vals):
        pair = (a, b)
        if pair in seen:
            rep = f"({a!r}, {b!r})"
            if rep not in dupes:
                dupes.append(rep)
        seen.add(pair)
    if dupes:
        raise JoinIntegrityError(
            source,
            table_name,
            f"{col_a}, {col_b}",
            "contains duplicate pairs",
            dupes[:5],
        )


def _check_section_index(
    table: pa.Table,
    column: str,
    source: str,
    table_name: str,
) -> None:
    """Raise JoinIntegrityError if any value in *column* is null or negative."""
    col = table.column(column)
    null_count = col.null_count
    if null_count > 0:
        raise JoinIntegrityError(
            source,
            table_name,
            column,
            "contains nulls",
            ["<null>"] * min(null_count, 5),
        )
    values = col.to_pylist()
    negatives = [str(v) for v in values if v is not None and v < 0]
    if negatives:
        raise JoinIntegrityError(
            source,
            table_name,
            column,
            "contains negative values",
            negatives[:5],
        )


def _check_no_orphans(
    child_table: pa.Table,
    child_key: str,
    parent_table: pa.Table,
    parent_key: str,
    source: str,
    child_table_name: str,
    context: str,
) -> None:
    """Raise JoinIntegrityError if child has keys absent from parent.

    Uses string representation to avoid sorting comparison errors.
    """
    child_vals = {
        str(v)
        for v in child_table.column(child_key).to_pylist()
        if v is not None
    }
    parent_vals = {
        str(v)
        for v in parent_table.column(parent_key).to_pylist()
        if v is not None
    }
    orphans = sorted(child_vals - parent_vals)
    if orphans:
        raise JoinIntegrityError(
            source,
            child_table_name,
            child_key,
            f"orphan keys not found in {context}",
            orphans[:5],
        )


# ===================================================================
# Wikipedia join
# ===================================================================

def join_wikipedia_sections(
    polygons: pa.Table,
    polygon_articles: pa.Table,
    wp_documents: pa.Table,
    wp_sections: pa.Table,
) -> pa.Table:
    """Join Wikipedia sections to polygons via polygon_articles.

    Join direction:
      polygon_articles.article_id → wp_documents.article_id
      wp_documents.document_id → wp_sections.document_id
      polygon_articles.polygon_id → polygons.polygon_id

    Returns an unsorted table conforming to JOINED_SECTIONS_SCHEMA columns.
    """
    # --- Integrity checks ---
    _check_non_empty(polygons, "polygon_id", "wikipedia", "polygons")
    _check_unique(polygons, "polygon_id", "wikipedia", "polygons")

    _check_non_empty(polygon_articles, "polygon_id", "wikipedia", "polygon_articles")
    _check_non_empty(polygon_articles, "article_id", "wikipedia", "polygon_articles")

    _check_non_empty(wp_documents, "document_id", "wikipedia", "wikipedia_documents")
    _check_unique(wp_documents, "document_id", "wikipedia", "wikipedia_documents")
    _check_non_empty(wp_documents, "article_id", "wikipedia", "wikipedia_documents")
    _check_unique(wp_documents, "article_id", "wikipedia", "wikipedia_documents")

    _check_non_empty(wp_sections, "section_id", "wikipedia", "wikipedia_sections")
    _check_unique(wp_sections, "section_id", "wikipedia", "wikipedia_sections")
    _check_non_empty(wp_sections, "document_id", "wikipedia", "wikipedia_sections")
    _check_section_index(wp_sections, "section_index", "wikipedia", "wikipedia_sections")

    _check_unique_pairs(
        polygon_articles,
        "polygon_id",
        "article_id",
        "wikipedia",
        "polygon_articles",
    )

    # --- Orphan checks ---
    _check_no_orphans(
        polygon_articles,
        "polygon_id",
        polygons,
        "polygon_id",
        "wikipedia",
        "polygon_articles",
        "polygons",
    )
    _check_no_orphans(
        polygon_articles,
        "article_id",
        wp_documents,
        "article_id",
        "wikipedia",
        "polygon_articles",
        "wikipedia_documents",
    )
    _check_no_orphans(
        wp_sections,
        "document_id",
        wp_documents,
        "document_id",
        "wikipedia",
        "wikipedia_sections",
        "wikipedia_documents",
    )

    # --- Identity consistency checks ---
    poly_wd = dict(
        zip(
            polygons.column("polygon_id").to_pylist(),
            polygons.column("wikidata").to_pylist(),
        )
    )

    doc_info = {}
    doc_cols = zip(
        wp_documents.column("document_id").to_pylist(),
        wp_documents.column("article_id").to_pylist(),
        wp_documents.column("wikidata").to_pylist(),
        wp_documents.column("language").to_pylist(),
        wp_documents.column("site").to_pylist(),
        wp_documents.column("page_id").to_pylist(),
        wp_documents.column("revision_id").to_pylist(),
    )
    for did, aid, wd, lang, site, pid, rid in doc_cols:
        doc_info[did] = (aid, wd, lang, site, pid, rid)

    doc_by_art = {}
    for did, (aid, wd, lang, site, pid, rid) in doc_info.items():
        doc_by_art[aid] = (did, wd, lang, pid, rid)

    pa_cols = zip(
        polygon_articles.column("polygon_id").to_pylist(),
        polygon_articles.column("article_id").to_pylist(),
        polygon_articles.column("wikidata").to_pylist(),
        polygon_articles.column("language").to_pylist(),
        polygon_articles.column("page_id").to_pylist(),
        polygon_articles.column("revision_id").to_pylist(),
    )
    for p_id, a_id, wd, lang, p_id_val, r_id_val in pa_cols:
        expected_poly_wd = poly_wd.get(p_id)
        if expected_poly_wd != wd:
            raise JoinIntegrityError(
                "wikipedia",
                "polygon_articles",
                "wikidata",
                f"mismatch with linked polygon wikidata: {wd!r} vs {expected_poly_wd!r}",
                [wd],
            )

        doc_data = doc_by_art.get(a_id)
        if doc_data:
            d_id, doc_wd, doc_lang, doc_page_id, doc_rev_id = doc_data
            if doc_wd != wd:
                raise JoinIntegrityError(
                    "wikipedia",
                    "polygon_articles",
                    "wikidata",
                    f"mismatch with linked document wikidata: {wd!r} vs {doc_wd!r}",
                    [wd],
                )
            if doc_lang != lang:
                raise JoinIntegrityError(
                    "wikipedia",
                    "polygon_articles",
                    "language",
                    f"mismatch with linked document language: {lang!r} vs {doc_lang!r}",
                    [lang],
                )
            if doc_page_id != p_id_val:
                raise JoinIntegrityError(
                    "wikipedia",
                    "polygon_articles",
                    "page_id",
                    f"mismatch with linked document page_id: {p_id_val!r} vs {doc_page_id!r}",
                    [str(p_id_val)],
                )
            if doc_rev_id != r_id_val:
                raise JoinIntegrityError(
                    "wikipedia",
                    "polygon_articles",
                    "revision_id",
                    f"mismatch with linked document revision_id: {r_id_val!r} vs {doc_rev_id!r}",
                    [str(r_id_val)],
                )

    sec_cols = zip(
        wp_sections.column("section_id").to_pylist(),
        wp_sections.column("document_id").to_pylist(),
        wp_sections.column("article_id").to_pylist(),
        wp_sections.column("wikidata").to_pylist(),
        wp_sections.column("language").to_pylist(),
        wp_sections.column("site").to_pylist(),
        wp_sections.column("page_id").to_pylist(),
        wp_sections.column("revision_id").to_pylist(),
    )
    for sid, did, aid, wd, lang, site, pid, rid in sec_cols:
        doc_data = doc_info.get(did)
        if doc_data:
            doc_aid, doc_wd, doc_lang, doc_site, doc_pid, doc_rid = doc_data
            if doc_wd != wd:
                raise JoinIntegrityError(
                    "wikipedia",
                    "wikipedia_sections",
                    "wikidata",
                    f"mismatch with linked document wikidata: {wd!r} vs {doc_wd!r}",
                    [wd],
                )
            if doc_lang != lang:
                raise JoinIntegrityError(
                    "wikipedia",
                    "wikipedia_sections",
                    "language",
                    f"mismatch with linked document language: {lang!r} vs {doc_lang!r}",
                    [lang],
                )
            if doc_site != site:
                raise JoinIntegrityError(
                    "wikipedia",
                    "wikipedia_sections",
                    "site",
                    f"mismatch with linked document site: {site!r} vs {doc_site!r}",
                    [site],
                )
            if doc_pid != pid:
                raise JoinIntegrityError(
                    "wikipedia",
                    "wikipedia_sections",
                    "page_id",
                    f"mismatch with linked document page_id: {pid!r} vs {doc_pid!r}",
                    [str(pid)],
                )
            if doc_rid != rid:
                raise JoinIntegrityError(
                    "wikipedia",
                    "wikipedia_sections",
                    "revision_id",
                    f"mismatch with linked document revision_id: {rid!r} vs {doc_rid!r}",
                    [str(rid)],
                )
            if aid != "" and doc_aid != aid:
                raise JoinIntegrityError(
                    "wikipedia",
                    "wikipedia_sections",
                    "article_id",
                    f"mismatch with linked document article_id: {aid!r} vs {doc_aid!r}",
                    [aid],
                )

    # --- Perform joins with cached python lists ---
    doc_article_ids = wp_documents.column("article_id").to_pylist()
    doc_by_article = {aid: i for i, aid in enumerate(doc_article_ids)}

    doc_ids_list = wp_documents.column("document_id").to_pylist()
    doc_id_to_section_indices: dict[str, list[int]] = {}
    for i, did in enumerate(wp_sections.column("document_id").to_pylist()):
        doc_id_to_section_indices.setdefault(did, []).append(i)

    polygons_polygon_ids = polygons.column("polygon_id").to_pylist()
    poly_by_id = {pid: i for i, pid in enumerate(polygons_polygon_ids)}

    # Cache lists for columns to avoid calling table.column(name)[idx] in loops
    pa_polygon_ids = polygon_articles.column("polygon_id").to_pylist()
    pa_article_ids = polygon_articles.column("article_id").to_pylist()

    doc_wikidatas = wp_documents.column("wikidata").to_pylist()
    doc_languages = wp_documents.column("language").to_pylist()
    doc_sites = wp_documents.column("site").to_pylist()
    doc_titles = wp_documents.column("title").to_pylist()
    doc_urls = wp_documents.column("url").to_pylist()
    doc_page_ids = wp_documents.column("page_id").to_pylist()
    doc_revision_ids = wp_documents.column("revision_id").to_pylist()
    doc_timestamps = wp_documents.column("revision_timestamp").to_pylist()
    doc_hashes = wp_documents.column("content_hash").to_pylist()

    sec_ids = wp_sections.column("section_id").to_pylist()
    sec_indices = wp_sections.column("section_index").to_pylist()
    sec_paths = wp_sections.column("section_path").to_pylist()
    sec_texts = wp_sections.column("text").to_pylist()
    sec_hashes = wp_sections.column("content_hash").to_pylist()

    poly_names = polygons.column("name").to_pylist()
    poly_primary_tags = polygons.column("osm_primary_tag").to_pylist()
    poly_tags = polygons.column("tags").to_pylist()
    poly_regions = polygons.column("region").to_pylist()
    poly_lats = polygons.column("lat").to_pylist()
    poly_lons = polygons.column("lon").to_pylist()

    result_arrays: dict[str, list] = {
        f.name: [] for f in JOINED_SECTIONS_SCHEMA
    }

    for pa_idx in range(polygon_articles.num_rows):
        pa_polygon_id = pa_polygon_ids[pa_idx]
        pa_article_id = pa_article_ids[pa_idx]

        doc_idx = doc_by_article[pa_article_id]
        poly_idx = poly_by_id[pa_polygon_id]

        doc_id = doc_ids_list[doc_idx]
        section_indices = doc_id_to_section_indices.get(doc_id, [])

        for sec_idx in section_indices:
            result_arrays["polygon_id"].append(pa_polygon_id)
            result_arrays["wikidata"].append(doc_wikidatas[doc_idx])
            result_arrays["document_id"].append(doc_id)
            result_arrays["article_id"].append(pa_article_id)
            result_arrays["source"].append("wikipedia")
            result_arrays["language"].append(doc_languages[doc_idx])
            result_arrays["site"].append(doc_sites[doc_idx])
            result_arrays["page_title"].append(doc_titles[doc_idx])
            result_arrays["url"].append(doc_urls[doc_idx])
            result_arrays["page_id"].append(doc_page_ids[doc_idx])
            result_arrays["revision_id"].append(doc_revision_ids[doc_idx])
            result_arrays["revision_timestamp"].append(doc_timestamps[doc_idx])
            result_arrays["document_content_hash"].append(doc_hashes[doc_idx])
            result_arrays["section_id"].append(sec_ids[sec_idx])
            result_arrays["section_index"].append(sec_indices[sec_idx])
            result_arrays["section_path_raw"].append(sec_paths[sec_idx])
            result_arrays["section_text_raw"].append(sec_texts[sec_idx])
            result_arrays["section_content_hash"].append(sec_hashes[sec_idx])
            result_arrays["polygon_name"].append(poly_names[poly_idx])
            result_arrays["osm_primary_tag"].append(
                poly_primary_tags[poly_idx]
            )
            result_arrays["osm_tags_raw"].append(poly_tags[poly_idx])
            result_arrays["region"].append(poly_regions[poly_idx])
            result_arrays["lat"].append(poly_lats[poly_idx])
            result_arrays["lon"].append(poly_lons[poly_idx])

    return pa.table(result_arrays, schema=JOINED_SECTIONS_SCHEMA)


# ===================================================================
# Wikivoyage join
# ===================================================================

def join_wikivoyage_sections(
    polygons: pa.Table,
    wv_documents: pa.Table,
    wv_sections: pa.Table,
) -> pa.Table:
    """Join Wikivoyage sections to polygons via shared Wikidata QID.

    Join direction:
      wv_documents.wikidata → polygons.wikidata
      wv_documents.document_id → wv_sections.document_id

    Empty article_id values are converted to null.

    Returns an unsorted table conforming to JOINED_SECTIONS_SCHEMA columns.
    """
    # --- Integrity checks ---
    _check_non_empty(polygons, "polygon_id", "wikivoyage", "polygons")
    _check_unique(polygons, "polygon_id", "wikivoyage", "polygons")

    _check_non_empty(wv_documents, "document_id", "wikivoyage", "wikivoyage_documents")
    _check_unique(wv_documents, "document_id", "wikivoyage", "wikivoyage_documents")
    _check_non_empty(wv_documents, "wikidata", "wikivoyage", "wikivoyage_documents")

    _check_non_empty(wv_sections, "section_id", "wikivoyage", "wikivoyage_sections")
    _check_unique(wv_sections, "section_id", "wikivoyage", "wikivoyage_sections")
    _check_non_empty(wv_sections, "document_id", "wikivoyage", "wikivoyage_sections")
    _check_section_index(wv_sections, "section_index", "wikivoyage", "wikivoyage_sections")

    # --- Orphan checks ---
    _check_no_orphans(
        wv_sections,
        "document_id",
        wv_documents,
        "document_id",
        "wikivoyage",
        "wikivoyage_sections",
        "wikivoyage_documents",
    )

    doc_wikidata = set(wv_documents.column("wikidata").to_pylist())
    poly_wikidata = set(polygons.column("wikidata").to_pylist())
    unmatched = sorted(doc_wikidata - poly_wikidata)
    if unmatched:
        raise JoinIntegrityError(
            "wikivoyage",
            "wikivoyage_documents",
            "wikidata",
            "Wikidata QIDs not found in polygons",
            [str(u) for u in unmatched[:5]],
        )

    # --- Identity consistency checks ---
    doc_info = {}
    doc_cols = zip(
        wv_documents.column("document_id").to_pylist(),
        wv_documents.column("article_id").to_pylist(),
        wv_documents.column("wikidata").to_pylist(),
        wv_documents.column("language").to_pylist(),
        wv_documents.column("site").to_pylist(),
        wv_documents.column("page_id").to_pylist(),
        wv_documents.column("revision_id").to_pylist(),
    )
    for did, aid, wd, lang, site, pid, rid in doc_cols:
        doc_info[did] = (aid, wd, lang, site, pid, rid)

    sec_cols = zip(
        wv_sections.column("section_id").to_pylist(),
        wv_sections.column("document_id").to_pylist(),
        wv_sections.column("article_id").to_pylist(),
        wv_sections.column("wikidata").to_pylist(),
        wv_sections.column("language").to_pylist(),
        wv_sections.column("site").to_pylist(),
        wv_sections.column("page_id").to_pylist(),
        wv_sections.column("revision_id").to_pylist(),
    )
    for sid, did, aid, wd, lang, site, pid, rid in sec_cols:
        doc_data = doc_info.get(did)
        if doc_data:
            doc_aid, doc_wd, doc_lang, doc_site, doc_pid, doc_rid = doc_data
            if doc_wd != wd:
                raise JoinIntegrityError(
                    "wikivoyage",
                    "wikivoyage_sections",
                    "wikidata",
                    f"mismatch with linked document wikidata: {wd!r} vs {doc_wd!r}",
                    [wd],
                )
            if doc_lang != lang:
                raise JoinIntegrityError(
                    "wikivoyage",
                    "wikivoyage_sections",
                    "language",
                    f"mismatch with linked document language: {lang!r} vs {doc_lang!r}",
                    [lang],
                )
            if doc_site != site:
                raise JoinIntegrityError(
                    "wikivoyage",
                    "wikivoyage_sections",
                    "site",
                    f"mismatch with linked document site: {site!r} vs {doc_site!r}",
                    [site],
                )
            if doc_pid != pid:
                raise JoinIntegrityError(
                    "wikivoyage",
                    "wikivoyage_sections",
                    "page_id",
                    f"mismatch with linked document page_id: {pid!r} vs {doc_pid!r}",
                    [str(pid)],
                )
            if doc_rid != rid:
                raise JoinIntegrityError(
                    "wikivoyage",
                    "wikivoyage_sections",
                    "revision_id",
                    f"mismatch with linked document revision_id: {rid!r} vs {doc_rid!r}",
                    [str(rid)],
                )
            if aid != "" and doc_aid != aid:
                raise JoinIntegrityError(
                    "wikivoyage",
                    "wikivoyage_sections",
                    "article_id",
                    f"mismatch with linked document article_id: {aid!r} vs {doc_aid!r}",
                    [aid],
                )

    # --- Build indices ---
    poly_wikidata_list = polygons.column("wikidata").to_pylist()
    wikidata_to_poly_indices: dict[str, list[int]] = {}
    for i, wd in enumerate(poly_wikidata_list):
        wikidata_to_poly_indices.setdefault(wd, []).append(i)

    doc_to_section_indices: dict[str, list[int]] = {}
    for i, did in enumerate(wv_sections.column("document_id").to_pylist()):
        doc_to_section_indices.setdefault(did, []).append(i)

    # Cache lists for columns to avoid calling table.column(name)[idx] in loops
    doc_ids = wv_documents.column("document_id").to_pylist()
    doc_wikidatas = wv_documents.column("wikidata").to_pylist()
    doc_languages = wv_documents.column("language").to_pylist()
    doc_sites = wv_documents.column("site").to_pylist()
    doc_titles = wv_documents.column("title").to_pylist()
    doc_urls = wv_documents.column("url").to_pylist()
    doc_page_ids = wv_documents.column("page_id").to_pylist()
    doc_revision_ids = wv_documents.column("revision_id").to_pylist()
    doc_timestamps = wv_documents.column("revision_timestamp").to_pylist()
    doc_hashes = wv_documents.column("content_hash").to_pylist()
    doc_article_ids_raw = wv_documents.column("article_id").to_pylist()

    sec_ids = wv_sections.column("section_id").to_pylist()
    sec_indices = wv_sections.column("section_index").to_pylist()
    sec_paths = wv_sections.column("section_path").to_pylist()
    sec_texts = wv_sections.column("text").to_pylist()
    sec_hashes = wv_sections.column("content_hash").to_pylist()

    poly_ids = polygons.column("polygon_id").to_pylist()
    poly_names = polygons.column("name").to_pylist()
    poly_primary_tags = polygons.column("osm_primary_tag").to_pylist()
    poly_tags = polygons.column("tags").to_pylist()
    poly_regions = polygons.column("region").to_pylist()
    poly_lats = polygons.column("lat").to_pylist()
    poly_lons = polygons.column("lon").to_pylist()

    result_arrays: dict[str, list] = {
        f.name: [] for f in JOINED_SECTIONS_SCHEMA
    }

    for doc_idx in range(wv_documents.num_rows):
        doc_wikidata_val = doc_wikidatas[doc_idx]
        doc_id = doc_ids[doc_idx]
        article_id_raw = doc_article_ids_raw[doc_idx]
        # Convert empty article_id to None
        article_id = None if article_id_raw == "" else article_id_raw

        poly_indices = wikidata_to_poly_indices.get(doc_wikidata_val, [])
        section_indices = doc_to_section_indices.get(doc_id, [])

        for poly_idx in poly_indices:
            for sec_idx in section_indices:
                result_arrays["polygon_id"].append(poly_ids[poly_idx])
                result_arrays["wikidata"].append(doc_wikidata_val)
                result_arrays["document_id"].append(doc_id)
                result_arrays["article_id"].append(article_id)
                result_arrays["source"].append("wikivoyage")
                result_arrays["language"].append(doc_languages[doc_idx])
                result_arrays["site"].append(doc_sites[doc_idx])
                result_arrays["page_title"].append(doc_titles[doc_idx])
                result_arrays["url"].append(doc_urls[doc_idx])
                result_arrays["page_id"].append(doc_page_ids[doc_idx])
                result_arrays["revision_id"].append(doc_revision_ids[doc_idx])
                result_arrays["revision_timestamp"].append(
                    doc_timestamps[doc_idx]
                )
                result_arrays["document_content_hash"].append(
                    doc_hashes[doc_idx]
                )
                result_arrays["section_id"].append(sec_ids[sec_idx])
                result_arrays["section_index"].append(sec_indices[sec_idx])
                result_arrays["section_path_raw"].append(sec_paths[sec_idx])
                result_arrays["section_text_raw"].append(sec_texts[sec_idx])
                result_arrays["section_content_hash"].append(sec_hashes[sec_idx])
                result_arrays["polygon_name"].append(poly_names[poly_idx])
                result_arrays["osm_primary_tag"].append(
                    poly_primary_tags[poly_idx]
                )
                result_arrays["osm_tags_raw"].append(poly_tags[poly_idx])
                result_arrays["region"].append(poly_regions[poly_idx])
                result_arrays["lat"].append(poly_lats[poly_idx])
                result_arrays["lon"].append(poly_lons[poly_idx])

    return pa.table(result_arrays, schema=JOINED_SECTIONS_SCHEMA)


# ===================================================================
# Join report
# ===================================================================

@dataclass(frozen=True)
class JoinReport:
    """Immutable report of join statistics."""

    shard_key: str
    polygon_count: int
    polygon_article_count: int
    wikipedia_document_count: int
    wikipedia_section_count: int
    wikipedia_occurrence_count: int
    wikivoyage_document_count: int
    wikivoyage_section_count: int
    wikivoyage_occurrence_count: int
    total_occurrence_count: int


@dataclass(frozen=True)
class JoinedRegionSections:
    """Result of build_region_section_occurrences."""

    table: pa.Table
    report: JoinReport


# ===================================================================
# Sort order for deterministic output
# ===================================================================

_SORT_KEYS = [
    ("polygon_id", "ascending"),
    ("source", "ascending"),
    ("language", "ascending"),
    ("document_id", "ascending"),
    ("section_index", "ascending"),
    ("section_id", "ascending"),
]


# ===================================================================
# Internal Composition Helper
# ===================================================================

def _build_region_section_occurrences_from_tables(
    shards: "RegionShardSet",  # noqa: F821
    polygons: pa.Table,
    polygon_articles: pa.Table,
    wp_documents: pa.Table,
    wp_sections: pa.Table,
    wv_documents: pa.Table | None = None,
    wv_sections: pa.Table | None = None,
) -> JoinedRegionSections:
    """Compose the joined intermediate tables.

    Validates presence consistency of optional Wikivoyage parameters.
    """
    # Issue 3: Table composition boundary consistency check
    if (wv_documents is not None) != (wv_sections is not None):
        raise JoinIntegrityError(
            source="wikivoyage",
            table_name="composition",
            key="wv_documents/wv_sections",
            violation="Inconsistent Wikivoyage optional inputs: both documents and sections must be provided together or both omitted",
            sample=[],
        )

    # Wikipedia join
    wp_result = join_wikipedia_sections(
        polygons, polygon_articles, wp_documents, wp_sections
    )
    wp_occ = wp_result.num_rows

    # Wikivoyage join
    wv_occ = 0
    wv_doc_count = 0
    wv_sec_count = 0
    if wv_documents is not None and wv_sections is not None:
        wv_result = join_wikivoyage_sections(polygons, wv_documents, wv_sections)
        wv_occ = wv_result.num_rows
        wv_doc_count = wv_documents.num_rows
        wv_sec_count = wv_sections.num_rows
        combined = pa.concat_tables([wp_result, wv_result])
    else:
        combined = wp_result

    # Deterministic sort
    sorted_table = combined.sort_by(_SORT_KEYS)

    report = JoinReport(
        shard_key=shards.shard_key,
        polygon_count=polygons.num_rows,
        polygon_article_count=polygon_articles.num_rows,
        wikipedia_document_count=wp_documents.num_rows,
        wikipedia_section_count=wp_sections.num_rows,
        wikipedia_occurrence_count=wp_occ,
        wikivoyage_document_count=wv_doc_count,
        wikivoyage_section_count=wv_sec_count,
        wikivoyage_occurrence_count=wv_occ,
        total_occurrence_count=wp_occ + wv_occ,
    )

    return JoinedRegionSections(table=sorted_table, report=report)


# ===================================================================
# Public Orchestration API
# ===================================================================

def build_region_section_occurrences(
    shards: "RegionShardSet",  # noqa: F821
) -> JoinedRegionSections:
    """Build the deterministic intermediate joined-section table from files.

    Loads the required and optional tables using projected columns needed
    by joins, applying pre-projection validation.
    """
    polygons = load_validated_table(
        "polygons", shards.polygons, columns=POLYGONS_COLS
    )
    polygon_articles = load_validated_table(
        "polygon_articles", shards.polygon_articles, columns=POLYGON_ARTICLES_COLS
    )
    wp_documents = load_validated_table(
        "wikipedia_documents", shards.wikipedia_documents, columns=WIKIPEDIA_DOCUMENTS_COLS
    )
    wp_sections = load_validated_table(
        "wikipedia_sections", shards.wikipedia_sections, columns=WIKIPEDIA_SECTIONS_COLS
    )

    wv_documents = None
    wv_sections = None
    if shards.wikivoyage_documents is not None:
        wv_documents = load_validated_table(
            "wikivoyage_documents", shards.wikivoyage_documents, columns=WIKIVOYAGE_DOCUMENTS_COLS
        )
    if shards.wikivoyage_sections is not None:
        wv_sections = load_validated_table(
            "wikivoyage_sections", shards.wikivoyage_sections, columns=WIKIVOYAGE_SECTIONS_COLS
        )

    return _build_region_section_occurrences_from_tables(
        shards=shards,
        polygons=polygons,
        polygon_articles=polygon_articles,
        wp_documents=wp_documents,
        wp_sections=wp_sections,
        wv_documents=wv_documents,
        wv_sections=wv_sections,
    )
