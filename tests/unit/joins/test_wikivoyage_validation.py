"""Direct contracts for the Wikivoyage input validation boundary."""

from __future__ import annotations

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.joins import _wikivoyage


def test_validate_wikivoyage_inputs_passes_exact_integrity_context(monkeypatch):
    """Every Wikivoyage integrity check keeps its source and table context."""

    polygons = pa.table({"polygon_id": ["poly-1"], "wikidata": ["Q1"]})
    wv_documents = pa.table(
        {"document_id": ["document-1"], "wikidata": ["Q1"]}
    )
    wv_sections = pa.table(
        {"section_id": ["section-1"], "document_id": ["document-1"]}
    )
    labels = {
        id(table): label
        for table, label in (
            (polygons, "polygons"),
            (wv_documents, "wv_documents"),
            (wv_sections, "wv_sections"),
        )
    }
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
        "_check_no_orphans",
    ):
        monkeypatch.setattr(_wikivoyage, name, record(name))

    _wikivoyage._validate_wikivoyage_inputs(
        polygons, wv_documents, wv_sections
    )

    assert calls == [
        ("_check_non_empty", ("polygons", "polygon_id", "wikivoyage", "polygons")),
        ("_check_unique", ("polygons", "polygon_id", "wikivoyage", "polygons")),
        (
            "_check_non_empty",
            ("wv_documents", "document_id", "wikivoyage", "wikivoyage_documents"),
        ),
        (
            "_check_unique",
            ("wv_documents", "document_id", "wikivoyage", "wikivoyage_documents"),
        ),
        (
            "_check_non_empty",
            ("wv_documents", "wikidata", "wikivoyage", "wikivoyage_documents"),
        ),
        (
            "_check_non_empty",
            ("wv_sections", "section_id", "wikivoyage", "wikivoyage_sections"),
        ),
        (
            "_check_unique",
            ("wv_sections", "section_id", "wikivoyage", "wikivoyage_sections"),
        ),
        (
            "_check_non_empty",
            ("wv_sections", "document_id", "wikivoyage", "wikivoyage_sections"),
        ),
        (
            "_check_section_index",
            ("wv_sections", "section_index", "wikivoyage", "wikivoyage_sections"),
        ),
        (
            "_check_no_orphans",
            (
                "wv_sections",
                "document_id",
                "wv_documents",
                "document_id",
                "wikivoyage",
                "wikivoyage_sections",
                "wikivoyage_documents",
            ),
        ),
    ]


def test_validate_wikivoyage_inputs_reports_exact_unmatched_qids(monkeypatch):
    """Unmatched document QIDs retain the public integrity error contract."""

    for name in (
        "_check_non_empty",
        "_check_unique",
        "_check_section_index",
        "_check_no_orphans",
    ):
        monkeypatch.setattr(_wikivoyage, name, lambda *args: None)

    polygons = pa.table({"polygon_id": ["poly-1"], "wikidata": ["Q1"]})
    wv_documents = pa.table(
        {
            "document_id": [
                "document-1",
                "document-2",
                "document-3",
                "document-4",
                "document-5",
                "document-6",
                "document-7",
            ],
            "wikidata": ["Q7", "Q6", "Q5", "Q4", "Q3", "Q2", "Q8"],
        }
    )
    wv_sections = pa.table({"section_id": ["section-1"]})

    with pytest.raises(_wikivoyage.JoinIntegrityError) as raised:
        _wikivoyage._validate_wikivoyage_inputs(
            polygons, wv_documents, wv_sections
        )

    error = raised.value
    assert error.source == "wikivoyage"
    assert error.table_name == "wikivoyage_documents"
    assert error.key == "wikidata"
    assert error.violation == "Wikidata QIDs not found in polygons"
    assert error.sample == ["Q2", "Q3", "Q4", "Q5", "Q6"]
