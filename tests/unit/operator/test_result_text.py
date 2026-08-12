"""Contracts for normalizing subprocess and SSH result text."""

from __future__ import annotations

import pytest

from osm_polygon_sentence_relevance.operator.result_text import result_text


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            type("Result", (), {"text": "preferred", "stdout": "fallback"})(),
            "preferred",
        ),
        (type("Result", (), {"text": "", "stdout": "fallback"})(), ""),
        (type("Result", (), {"stdout": "fallback"})(), "fallback"),
    ],
)
def test_result_text_prefers_text_attribute_without_truthiness_fallback(
    result: object,
    expected: str,
) -> None:
    assert result_text(result) == expected


def test_result_text_returns_empty_string_when_no_output_attribute_exists() -> None:
    assert result_text(object()) == ""


def test_result_text_can_preserve_legacy_empty_text_fallback() -> None:
    result = type("Result", (), {"text": "", "stdout": "fallback"})()

    assert result_text(result, fallback_on_empty=True) == "fallback"
