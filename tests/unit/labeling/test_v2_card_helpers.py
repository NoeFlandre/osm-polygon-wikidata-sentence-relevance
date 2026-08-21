"""Small pure-helper contracts used by the V2 dataset card."""

from __future__ import annotations

import pyarrow as pa

import osm_polygon_sentence_relevance.labeling.v2_finalization as v2_finalization


def test_card_target_column_supports_raw_and_legacy_sentence_layouts() -> None:
    required = {
        "page_title": ["Page"],
        "section_path": [["History"]],
        "previous_sentence": ["Before"],
        "next_sentence": ["After"],
    }

    assert v2_finalization._card_target_column(
        pa.table({**required, "sentence_text_raw": ["Raw"]})
    ) == "sentence_text_raw"
    assert v2_finalization._card_target_column(
        pa.table({**required, "sentence_text": ["Legacy"]})
    ) == "sentence_text"
    assert v2_finalization._card_target_column(pa.table({"sentence_text_raw": ["Raw"]})) is None


def test_card_data_escapes_markup_without_escaping_quotes(
    monkeypatch,
) -> None:
    seen: list[tuple[str, object]] = []

    def record(value: str, *, quote: object) -> str:
        seen.append((value, quote))
        return "encoded"

    monkeypatch.setattr(v2_finalization, "escape", record)

    assert v2_finalization._card_data(None) == "encoded"
    assert seen == [("", False)]


def test_card_section_title_uses_last_path_element_or_none() -> None:
    assert v2_finalization._card_section_title({"section_path": ["Intro", "History"]}) == "History"
    assert v2_finalization._card_section_title({"section_path": []}) == "none"
    assert v2_finalization._card_section_title({}) == "none"


def test_card_prompt_example_uses_first_row_and_escapes_fences() -> None:
    table = pa.table(
        {
            "page_title": ["First", "Second"],
            "section_path": [["History"], ["Other"]],
            "previous_sentence": ["Before", "Wrong"],
            "sentence_text_raw": ["Target ```", "Wrong target"],
            "next_sentence": ["After", "Wrong"],
        }
    )

    assert v2_finalization._card_prompt_example(table) == (
        "Page title:\n"
        "<page_title>First</page_title>\n\n"
        "Section title:\n"
        "<section_title>History</section_title>\n\n"
        "Previous sentence:\n"
        "<previous>Before</previous>\n\n"
        "TARGET SENTENCE:\n"
        "<target>Target ``\\`</target>\n\n"
        "Next sentence:\n"
        "<next>After</next>"
    )
    assert v2_finalization._card_prompt_example(pa.table({"sentence_text_raw": ["x"]})) == ""


def test_card_prompt_example_reads_exactly_one_row() -> None:
    class RecordingTable:
        column_names = [
            "page_title",
            "section_path",
            "previous_sentence",
            "sentence_text_raw",
            "next_sentence",
        ]

        def __init__(self) -> None:
            self.slices: list[tuple[int, int]] = []

        def slice(self, offset: int, length: int):
            self.slices.append((offset, length))
            return self

        @staticmethod
        def to_pylist() -> list[dict[str, object]]:
            return [
                {
                    "page_title": "Page",
                    "section_path": ["History"],
                    "previous_sentence": "Before",
                    "sentence_text_raw": "Target",
                    "next_sentence": "After",
                }
            ]

    table = RecordingTable()

    v2_finalization._card_prompt_example(table)

    assert table.slices == [(0, 1)]
