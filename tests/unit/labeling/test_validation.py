from __future__ import annotations

import json

import pytest

from osm_polygon_sentence_relevance.labeling.validation import (
    LabelValidationError,
    parse_label_response,
)


def _response(**overrides: object) -> str:
    value: dict[str, object] = {
        "landuse_relevance": "yes",
        "polygon_relevance": "yes",
        "landuse_reason": "explicit_land_use",
        "polygon_reason": "direct_polygon_reference",
        "evidence": "irrigated farming",
    }
    value.update(overrides)
    return json.dumps(value)


def test_parse_valid_response() -> None:
    label = parse_label_response(
        _response(), target_sentence="The valley supports irrigated farming."
    )
    assert label.landuse_relevance.value == "yes"
    assert label.polygon_relevance.value == "yes"
    assert label.evidence == "irrigated farming"


@pytest.mark.parametrize("raw", ["", "not json", "[]", "null"])
def test_rejects_non_object_json(raw: str) -> None:
    with pytest.raises(LabelValidationError):
        parse_label_response(raw, target_sentence="text")


def test_rejects_missing_and_extra_fields() -> None:
    with pytest.raises(LabelValidationError, match="fields"):
        parse_label_response(_response(evidence_marker="x"), target_sentence="text")
    value = json.loads(_response())
    value.pop("evidence")
    with pytest.raises(LabelValidationError, match="fields"):
        parse_label_response(json.dumps(value), target_sentence="text")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("landuse_relevance", "maybe"),
        ("polygon_relevance", True),
        ("landuse_reason", "invented"),
        ("polygon_reason", "invented"),
    ],
)
def test_rejects_invalid_enums(field: str, value: object) -> None:
    with pytest.raises(LabelValidationError, match=field):
        parse_label_response(_response(**{field: value}), target_sentence="text")


def test_evidence_must_be_exact_target_substring() -> None:
    with pytest.raises(LabelValidationError, match="exact substring"):
        parse_label_response(
            _response(evidence="neighbor fact"), target_sentence="target"
        )


def test_empty_evidence_is_valid() -> None:
    assert (
        parse_label_response(_response(evidence=""), target_sentence="target").evidence
        == ""
    )


def test_evidence_length_is_bounded() -> None:
    text = "x" * 241
    with pytest.raises(LabelValidationError, match="240"):
        parse_label_response(_response(evidence=text), target_sentence=text)


@pytest.mark.parametrize(
    "overrides",
    [
        {"landuse_relevance": "no", "landuse_reason": "explicit_land_use"},
        {"landuse_relevance": "uncertain", "landuse_reason": "no_landuse_or_cover"},
        {"polygon_relevance": "no", "polygon_reason": "direct_polygon_reference"},
        {"polygon_relevance": "uncertain", "polygon_reason": "unrelated_fact"},
    ],
)
def test_reason_must_match_question_and_decision(overrides: dict[str, str]) -> None:
    with pytest.raises(LabelValidationError, match="inconsistent"):
        parse_label_response(
            _response(**overrides), target_sentence="irrigated farming"
        )
