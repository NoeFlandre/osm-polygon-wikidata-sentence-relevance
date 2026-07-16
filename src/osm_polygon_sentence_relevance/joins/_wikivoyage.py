"""Wikivoyage-specific section-to-polygon join.

Join direction:
  wv_documents.wikidata → polygons.wikidata
  wv_documents.document_id → wv_sections.document_id

Empty article_id values are converted to null.  The Wikivoyage join keys on
Wikidata QID rather than article_id, so its algorithm is kept separate from
the Wikipedia join rather than forced into a shared abstraction.
"""

from __future__ import annotations

import pyarrow as pa

from osm_polygon_sentence_relevance.contracts.errors import JoinIntegrityError
from osm_polygon_sentence_relevance.contracts.schemas import JOINED_SECTIONS_SCHEMA
from osm_polygon_sentence_relevance.joins._integrity import (
    _check_no_orphans,
    _check_non_empty,
    _check_section_index,
    _check_unique,
)


def join_wikivoyage_sections(
    polygons: pa.Table,
    wv_documents: pa.Table,
    wv_sections: pa.Table,
) -> pa.Table:
    """Join Wikivoyage sections to polygons via shared Wikidata QID.

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
    _check_section_index(
        wv_sections, "section_index", "wikivoyage", "wikivoyage_sections"
    )

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
        strict=True,
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
        strict=True,
    )
    for _sid, did, aid, wd, lang, _site, pid, rid in sec_cols:
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

    result_arrays: dict[str, list] = {f.name: [] for f in JOINED_SECTIONS_SCHEMA}

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
                result_arrays["revision_timestamp"].append(doc_timestamps[doc_idx])
                result_arrays["document_content_hash"].append(doc_hashes[doc_idx])
                result_arrays["section_id"].append(sec_ids[sec_idx])
                result_arrays["section_index"].append(sec_indices[sec_idx])
                result_arrays["section_path_raw"].append(sec_paths[sec_idx])
                result_arrays["section_text_raw"].append(sec_texts[sec_idx])
                result_arrays["section_content_hash"].append(sec_hashes[sec_idx])
                result_arrays["polygon_name"].append(poly_names[poly_idx])
                result_arrays["osm_primary_tag"].append(poly_primary_tags[poly_idx])
                result_arrays["osm_tags_raw"].append(poly_tags[poly_idx])
                result_arrays["region"].append(poly_regions[poly_idx])
                result_arrays["lat"].append(poly_lats[poly_idx])
                result_arrays["lon"].append(poly_lons[poly_idx])

    return pa.table(result_arrays, schema=JOINED_SECTIONS_SCHEMA)
