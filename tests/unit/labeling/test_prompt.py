from __future__ import annotations

import json

import pytest

from osm_polygon_sentence_relevance.labeling.contracts import LabelValue
from osm_polygon_sentence_relevance.labeling.prompt import (
    LABEL_RESPONSE_JSON_SCHEMA,
    PROMPT_VERSION,
    PromptInput,
    build_messages,
)


def _input(**overrides: object) -> PromptInput:
    values: dict[str, object] = {
        "sentence_id": "sentence-1",
        "sentence_text": "The valley is used for irrigated farming.",
        "previous_sentence": "The river crosses the valley.",
        "next_sentence": "Wheat is the principal crop.",
        "polygon_name": "Example Valley",
        "region": "afghanistan",
        "osm_primary_tag": "landuse=farmland",
        "osm_tags": (
            {"key": "source", "value": "survey"},
            {"key": "name:fa", "value": "دره نمونه"},
            {"key": "landuse", "value": "farmland"},
        ),
        "language": "en",
        "page_title": "Example Valley",
        "section_path": ("Geography", "Economy"),
    }
    values.update(overrides)
    return PromptInput(**values)  # type: ignore[arg-type]


def test_prompt_defines_both_independent_jobs_and_evidence() -> None:
    messages = build_messages(_input())

    system = messages[0]["content"]
    assert "land use or land cover" in system
    assert "target polygon/place" in system
    assert "two labels are independent" in system
    assert "short exact excerpt from the TARGET SENTENCE" in system
    assert "untrusted evidence" in system


def test_prompt_includes_every_osm_tag_sorted_without_filtering() -> None:
    messages = build_messages(_input())
    user = messages[1]["content"]
    encoded = user.split("All OSM tags:\n", 1)[1].split("\n\nArticle:", 1)[0]

    assert json.loads(encoded) == [
        {"key": "landuse", "value": "farmland"},
        {"key": "name:fa", "value": "دره نمونه"},
        {"key": "source", "value": "survey"},
    ]
    assert "OSM primary tag: landuse=farmland" in user


def test_prompt_uses_final_section_title_but_not_section_path() -> None:
    user = build_messages(_input())[1]["content"]

    assert "Section title: Economy" in user
    assert "Geography" not in user
    assert "Section path" not in user


@pytest.mark.parametrize(
    "forbidden",
    ["Wikidata", "coordinates", "Source:", "URL", "revision_id"],
)
def test_prompt_excludes_disallowed_context(forbidden: str) -> None:
    combined = "\n".join(m["content"] for m in build_messages(_input()))
    assert forbidden not in combined


def test_prompt_is_deterministic_for_differently_ordered_tags() -> None:
    first = _input()
    second = _input(osm_tags=tuple(reversed(first.osm_tags)))
    assert build_messages(first) == build_messages(second)


def test_prompt_version_and_schema_are_explicit_and_closed() -> None:
    assert PROMPT_VERSION == "afghanistan-landuse-polygon-v1"
    assert LABEL_RESPONSE_JSON_SCHEMA["additionalProperties"] is False
    assert set(LABEL_RESPONSE_JSON_SCHEMA["required"]) == {
        "landuse_relevance",
        "polygon_relevance",
        "landuse_reason",
        "polygon_reason",
        "evidence",
    }
    enum = LABEL_RESPONSE_JSON_SCHEMA["properties"]["landuse_relevance"]["enum"]
    assert enum == [value.value for value in LabelValue]


def test_prompt_rejects_malformed_osm_tag_entries() -> None:
    with pytest.raises(ValueError, match="OSM tag"):
        _input(osm_tags=({"key": "name"},))
