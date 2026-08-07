from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.labeling.v2_contracts import V2LogitRecord
from osm_polygon_sentence_relevance.labeling.v2_manual_eval import (
    write_v2_manual_eval,
)


def test_manual_eval_is_deterministic_and_reviewable(tmp_path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "sentence_id": "b",
                "sentence_text_raw": "The lake is shallow.",
                "previous_sentence": None,
                "next_sentence": "It is surrounded by reeds.",
                "page_title": "Example lake",
                "section_path": ["Geography", "Landscape"],
            },
            {
                "sentence_id": "a",
                "sentence_text_raw": "The village was founded in 1900.",
                "previous_sentence": "The lake is shallow.",
                "next_sentence": None,
                "page_title": "Example village",
                "section_path": ["History"],
            },
        ]
    )
    records = [
        V2LogitRecord("a", "no", -2.0, -0.1),
        V2LogitRecord("b", "yes", -0.1, -2.0),
    ]
    path = tmp_path / "manual_eval.jsonl"
    write_v2_manual_eval(table, records, path, limit=2)
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert [row["sentence_id"] for row in rows] == ["a", "b"]
    assert rows[0]["human_label"] == ""
    assert rows[0]["notes"] == ""
    assert rows[0]["section_title"] == "History"
    assert set(rows[0]) == {
        "sentence_id",
        "page_title",
        "section_title",
        "previous_sentence",
        "sentence_text",
        "next_sentence",
        "model_label",
        "yes_logprob",
        "no_logprob",
        "logit_margin",
        "two_class_probability",
        "human_label",
        "notes",
    }


def test_manual_eval_rejects_missing_or_duplicate_scores(tmp_path: Path) -> None:
    table = pa.table({"sentence_id": ["a"]})
    path = tmp_path / "eval.jsonl"
    with pytest.raises(ValueError, match="missing columns"):
        write_v2_manual_eval(table, [], path, limit=1)


def test_manual_eval_rejects_invalid_limit_and_duplicate_scores(tmp_path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "sentence_id": "a",
                "sentence_text_raw": "A valley.",
                "previous_sentence": None,
                "next_sentence": None,
                "page_title": "Valley",
                "section_path": "Geography",
            }
        ]
    )
    record = V2LogitRecord("a", "yes", -0.1, -1.1)
    with pytest.raises(ValueError, match="limit"):
        write_v2_manual_eval(table, [record], tmp_path / "eval.jsonl", limit=0)
    with pytest.raises(ValueError, match="duplicate"):
        write_v2_manual_eval(table, [record, record], tmp_path / "eval.jsonl", limit=1)


def test_manual_eval_rejects_missing_score_and_invalid_section(tmp_path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "sentence_id": "a",
                "sentence_text_raw": "A valley.",
                "previous_sentence": None,
                "next_sentence": None,
                "page_title": "Valley",
                "section_path": 3,
            }
        ]
    )
    with pytest.raises(ValueError, match="section_path"):
        write_v2_manual_eval(
            table, [V2LogitRecord("a", "yes", -0.1, -1.1)], tmp_path / "eval.jsonl"
        )
    table = table.set_column(5, "section_path", pa.array([["Geography"]]))
    with pytest.raises(ValueError, match="missing a score"):
        write_v2_manual_eval(table, [], tmp_path / "eval.jsonl")


def test_manual_eval_removes_temporary_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "sentence_id": "a",
                "sentence_text_raw": "A valley.",
                "previous_sentence": None,
                "next_sentence": None,
                "page_title": "Valley",
                "section_path": ["Geography"],
            }
        ]
    )
    import osm_polygon_sentence_relevance.labeling.v2_manual_eval as module

    monkeypatch.setattr(
        module.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace")),
    )
    with pytest.raises(OSError, match="replace"):
        write_v2_manual_eval(
            table,
            [V2LogitRecord("a", "yes", -0.1, -1.1)],
            tmp_path / "eval.jsonl",
        )
    assert list(tmp_path.glob(".eval.jsonl.*")) == []
