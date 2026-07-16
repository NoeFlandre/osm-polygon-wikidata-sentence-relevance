"""Wikipedia-specific section-to-polygon join.

Join direction:
  polygon_articles.article_id → wp_documents.article_id
  wp_documents.document_id → wp_sections.document_id
  polygon_articles.polygon_id → polygons.polygon_id

The algorithm is intentionally Wikipedia-specific; it is not forced into a
generic abstraction shared with Wikivoyage.
"""

from __future__ import annotations

import pyarrow as pa

from osm_polygon_sentence_relevance.errors import JoinIntegrityError
from osm_polygon_sentence_relevance.joins._integrity import (
    _check_no_orphans,
    _check_non_empty,
    _check_section_index,
    _check_unique,
    _check_unique_pairs,
)
from osm_polygon_sentence_relevance.schemas import JOINED_SECTIONS_SCHEMA


def join_wikipedia_sections(
    polygons: pa.Table,
    polygon_articles: pa.Table,
    wp_documents: pa.Table,
    wp_sections: pa.Table,
) -> pa.Table:
    """Join Wikipedia sections to polygons via polygon_articles.

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
    _check_section_index(
        wp_sections, "section_index", "wikipedia", "wikipedia_sections"
    )

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
            strict=True,
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
        strict=True,
    )
    for did, aid, wd, lang, site, pid, rid in doc_cols:
        doc_info[did] = (aid, wd, lang, site, pid, rid)

    doc_by_art = {}
    for did, (aid, wd, lang, _site, pid, rid) in doc_info.items():
        doc_by_art[aid] = (did, wd, lang, pid, rid)

    pa_cols = zip(
        polygon_articles.column("polygon_id").to_pylist(),
        polygon_articles.column("article_id").to_pylist(),
        polygon_articles.column("wikidata").to_pylist(),
        polygon_articles.column("language").to_pylist(),
        polygon_articles.column("page_id").to_pylist(),
        polygon_articles.column("revision_id").to_pylist(),
        strict=True,
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

        poly_doc_data = doc_by_art.get(a_id)
        if poly_doc_data:
            d_id, doc_wd, doc_lang, doc_page_id, doc_rev_id = poly_doc_data
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
        strict=True,
    )
    for _sid, did, aid, wd, lang, site, pid, rid in sec_cols:
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

    result_arrays: dict[str, list] = {f.name: [] for f in JOINED_SECTIONS_SCHEMA}

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
            result_arrays["osm_primary_tag"].append(poly_primary_tags[poly_idx])
            result_arrays["osm_tags_raw"].append(poly_tags[poly_idx])
            result_arrays["region"].append(poly_regions[poly_idx])
            result_arrays["lat"].append(poly_lats[poly_idx])
            result_arrays["lon"].append(poly_lons[poly_idx])

    return pa.table(result_arrays, schema=JOINED_SECTIONS_SCHEMA)
