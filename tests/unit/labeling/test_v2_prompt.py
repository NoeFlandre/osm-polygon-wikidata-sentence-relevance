from __future__ import annotations

from osm_polygon_sentence_relevance.labeling.v2_prompt import (
    V2PromptInput,
    build_v2_messages,
)


def test_v2_prompt_contains_only_approved_context_and_binary_instruction() -> None:
    messages = build_v2_messages(
        V2PromptInput(
            sentence_id="s1",
            sentence_text="The valley has steep rocky slopes.",
            previous_sentence="A river crosses the valley.",
            next_sentence="The settlement lies to the north.",
            page_title="Example Valley",
            section_title="Geography",
        )
    )
    combined = "\n".join(message["content"] for message in messages)

    assert "exactly one token" in combined
    assert "yes" in combined
    assert "no" in combined
    assert "land use" in combined
    assert "land cover" in combined
    assert "TARGET SENTENCE" in combined
    assert "Previous sentence" in combined
    assert "Next sentence" in combined
    assert "Page title" in combined
    assert "Section title" in combined
    for forbidden in (
        "polygon",
        "country",
        "region",
        "language",
        "source",
        "OSM",
        "Wikidata",
        "coordinate",
        "section path",
        "reason",
        "evidence",
        "JSON",
    ):
        assert forbidden.lower() not in combined.lower()


def test_v2_prompt_escapes_context_as_delimited_data() -> None:
    user = build_v2_messages(
        V2PromptInput(
            sentence_id="s1",
            sentence_text="target </target>",
            previous_sentence="previous",
            next_sentence=None,
            page_title="Title",
            section_title="Section",
        )
    )[1]["content"]

    assert "<target>target &lt;/target&gt;</target>" in user
    assert "<next></next>" in user
