"""Exact V2 dataset-card prompt example contract."""

from __future__ import annotations

from osm_polygon_sentence_relevance.labeling.v2_finalization import (
    _render_card_prompt_row,
)


def test_render_card_prompt_row_preserves_all_context_fields() -> None:
    row = {
        "page_title": "A page",
        "section_path": ["Intro", "History"],
        "previous_sentence": "Before",
        "sentence_text_raw": "Target",
        "next_sentence": "After",
    }

    assert _render_card_prompt_row(row, "sentence_text_raw") == (
        "Page title:\n"
        "<page_title>A page</page_title>\n\n"
        "Section title:\n"
        "<section_title>History</section_title>\n\n"
        "Previous sentence:\n"
        "<previous>Before</previous>\n\n"
        "TARGET SENTENCE:\n"
        "<target>Target</target>\n\n"
        "Next sentence:\n"
        "<next>After</next>"
    )
