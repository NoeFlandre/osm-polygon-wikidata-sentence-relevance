"""Direct contracts for the Wikipedia input validation boundary."""

from __future__ import annotations

import pyarrow as pa

from osm_polygon_sentence_relevance.joins import _wikipedia


def test_validate_wikipedia_inputs_passes_exact_integrity_context(monkeypatch):
    """Every integrity check keeps its source and table context."""

    polygons = pa.table({"polygon_id": ["poly-1"]})
    polygon_articles = pa.table(
        {"polygon_id": ["poly-1"], "article_id": ["article-1"]}
    )
    wp_documents = pa.table(
        {"document_id": ["document-1"], "article_id": ["article-1"]}
    )
    wp_sections = pa.table(
        {
            "section_id": ["section-1"],
            "document_id": ["document-1"],
            "section_index": [0],
        }
    )
    labels = {id(table): label for table, label in (
        (polygons, "polygons"),
        (polygon_articles, "polygon_articles"),
        (wp_documents, "wp_documents"),
        (wp_sections, "wp_sections"),
    )}
    calls: list[tuple[str, tuple[object, ...]]] = []

    def record(name: str):
        def check(*args):
            calls.append(
                (
                    name,
                    tuple(labels.get(id(value), value) for value in args),
                )
            )

        return check

    for name in (
        "_check_non_empty",
        "_check_unique",
        "_check_section_index",
        "_check_unique_pairs",
        "_check_no_orphans",
    ):
        monkeypatch.setattr(_wikipedia, name, record(name))

    article_key = _wikipedia._validate_wikipedia_inputs(
        polygons, polygon_articles, wp_documents, wp_sections
    )

    assert article_key == "article_id"
    assert calls == [
        ("_check_non_empty", ("polygons", "polygon_id", "wikipedia", "polygons")),
        ("_check_unique", ("polygons", "polygon_id", "wikipedia", "polygons")),
        (
            "_check_non_empty",
            ("polygon_articles", "polygon_id", "wikipedia", "polygon_articles"),
        ),
        (
            "_check_non_empty",
            ("polygon_articles", "article_id", "wikipedia", "polygon_articles"),
        ),
        (
            "_check_non_empty",
            ("wp_documents", "document_id", "wikipedia", "wikipedia_documents"),
        ),
        (
            "_check_unique",
            ("wp_documents", "document_id", "wikipedia", "wikipedia_documents"),
        ),
        (
            "_check_non_empty",
            ("wp_documents", "article_id", "wikipedia", "wikipedia_documents"),
        ),
        (
            "_check_unique",
            ("wp_documents", "article_id", "wikipedia", "wikipedia_documents"),
        ),
        (
            "_check_non_empty",
            ("wp_sections", "section_id", "wikipedia", "wikipedia_sections"),
        ),
        (
            "_check_unique",
            ("wp_sections", "section_id", "wikipedia", "wikipedia_sections"),
        ),
        (
            "_check_non_empty",
            ("wp_sections", "document_id", "wikipedia", "wikipedia_sections"),
        ),
        (
            "_check_section_index",
            ("wp_sections", "section_index", "wikipedia", "wikipedia_sections"),
        ),
        (
            "_check_unique_pairs",
            (
                "polygon_articles",
                "polygon_id",
                "article_id",
                "wikipedia",
                "polygon_articles",
            ),
        ),
        (
            "_check_no_orphans",
            (
                "polygon_articles",
                "polygon_id",
                "polygons",
                "polygon_id",
                "wikipedia",
                "polygon_articles",
                "polygons",
            ),
        ),
        (
            "_check_no_orphans",
            (
                "polygon_articles",
                "article_id",
                "wp_documents",
                "article_id",
                "wikipedia",
                "polygon_articles",
                "wikipedia_documents",
            ),
        ),
        (
            "_check_no_orphans",
            (
                "wp_sections",
                "document_id",
                "wp_documents",
                "document_id",
                "wikipedia",
                "wikipedia_sections",
                "wikipedia_documents",
            ),
        ),
    ]
