"""Regression test for the Wikivoyage multisite join defect.

A shard may contain Wikivoyage documents from multiple sites/languages.
The join must compare each section's own ``site`` against the ``site`` of
the document it links to, not a stale value carried over from another
document.  Adversarial row ordering exposes the stale-site comparison.
"""

from __future__ import annotations

import pyarrow as pa

from osm_polygon_sentence_relevance.joins import join_wikivoyage_sections
from osm_polygon_sentence_relevance.schemas import (
    POLYGONS_SCHEMA,
    SECTIONS_SCHEMA,
    WIKIVOYAGE_DOCUMENTS_SCHEMA,
)
from tests.helpers import (
    make_polygon_row,
    make_section_row,
    make_wikivoyage_document_row,
    rows_to_table,
)


def _polygons(rows: list[dict[str, list]]) -> pa.Table:
    return rows_to_table(rows, POLYGONS_SCHEMA)


def _wv_docs(rows: list[dict[str, list]]) -> pa.Table:
    return rows_to_table(rows, WIKIVOYAGE_DOCUMENTS_SCHEMA)


def _wv_sections(rows: list[dict[str, list]]) -> pa.Table:
    return rows_to_table(rows, SECTIONS_SCHEMA)


class TestWikivoyageMultisiteJoin:
    """A shard with multiple Wikivoyage sites must not report false mismatch."""

    def test_multisite_shard_joins_all_sections(self):
        polygons = _polygons(
            [
                make_polygon_row(polygon_id="poly-en", wikidata="Q889"),
                make_polygon_row(polygon_id="poly-zh", wikidata="Q290"),
            ]
        )

        # Distinct documents: English (Q889) and Chinese (Q290).
        en_doc = make_wikivoyage_document_row(
            document_id="doc-wv-en",
            wikidata="Q889",
            language="en",
            title="Afghanistan",
            url="https://en.wikivoyage.org/wiki/Afghanistan",
        )
        zh_doc = make_wikivoyage_document_row(
            document_id="doc-wv-zh",
            wikidata="Q290",
            language="zh",
            title="阿富汗",
            url="https://zh.wikivoyage.org/wiki/阿富汗",
        )
        # Override the default site to Chinese Wikivoyage.
        zh_doc["site"] = ["zh.wikivoyage.org"]

        wv_docs = _wv_docs([en_doc, zh_doc])

        # Adversarial ordering: Chinese section first, so a stale-site value
        # carried from the Chinese document comparison would poison the
        # English section check (and vice versa).
        wv_secs = _wv_sections(
            [
                make_section_row(
                    section_id="sec-wv-zh",
                    document_id="doc-wv-zh",
                    article_id="",
                    project="wikivoyage",
                    wikidata="Q290",
                    language="zh",
                    site="zh.wikivoyage.org",
                    section_index=0,
                    text="去喀布尔。",
                ),
                make_section_row(
                    section_id="sec-wv-en",
                    document_id="doc-wv-en",
                    article_id="",
                    project="wikivoyage",
                    wikidata="Q889",
                    language="en",
                    site="en.wikivoyage.org",
                    section_index=0,
                    text="Go to Kabul.",
                ),
            ]
        )

        result = join_wikivoyage_sections(polygons, wv_docs, wv_secs)

        assert result.num_rows == 2

        by_doc = {}
        for i in range(result.num_rows):
            did = result.column("document_id")[i].as_py()
            by_doc[did] = {
                "site": result.column("site")[i].as_py(),
                "language": result.column("language")[i].as_py(),
                "section_id": result.column("section_id")[i].as_py(),
            }

        assert by_doc["doc-wv-en"] == {
            "site": "en.wikivoyage.org",
            "language": "en",
            "section_id": "sec-wv-en",
        }
        assert by_doc["doc-wv-zh"] == {
            "site": "zh.wikivoyage.org",
            "language": "zh",
            "section_id": "sec-wv-zh",
        }
