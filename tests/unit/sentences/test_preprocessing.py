"""Tests for deterministic preprocessing utilities (Phase 3A).

Covers section-path parsing, OSM-tag parsing, and sentence normalization.
"""

from __future__ import annotations

import pytest

from osm_polygon_sentence_relevance.errors import PreprocessingError
from osm_polygon_sentence_relevance.preprocessing import (
    normalize_sentence,
    parse_osm_tags,
    parse_section_path,
)


# ===================================================================
# parse_section_path
# ===================================================================
class TestParseSectionPath:
    def test_empty_array(self):
        assert parse_section_path("[]") == []

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('["Introduction"]', ["Introduction"]),
            ('["a", "b", "c"]', ["a", "b", "c"]),
            ('["  spaced  ", "x"]', ["  spaced  ", "x"]),
        ],
    )
    def test_valid_arrays(self, raw, expected):
        assert parse_section_path(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "not json",
            "null",
            "{}",
            "123",
            '"a string"',
            "true",
            "[1, 2, 3]",
            '["a", 2]',
            '["a", null]',
            '{"k": "v"}',
            "",
            "[,]",  # type: ignore
        ],
    )
    def test_invalid_values_raise(self, raw):
        with pytest.raises(PreprocessingError) as exc:
            parse_section_path(raw)
        assert "section_path" in str(exc.value)


# ===================================================================
# parse_osm_tags
# ===================================================================
class TestParseOsmTags:
    def test_empty_object(self):
        assert parse_osm_tags("{}") == {}

    def test_valid_object(self):
        raw = '{"name": "Afghanistan", "highway": "residential"}'
        assert parse_osm_tags(raw) == {
            "highway": "residential",
            "name": "Afghanistan",
        }

    def test_sorted_keys(self):
        raw = '{"z": "1", "a": "2", "m": "3"}'
        assert list(parse_osm_tags(raw).keys()) == ["a", "m", "z"]

    @pytest.mark.parametrize(
        "raw",
        [
            "not json",
            "null",
            "[]",
            "123",
            '"a string"',
            "true",
            '{"k": 1}',
            '{"k": null}',
            '{"k": "v", "n": 5}',
            "{} extra",
            "",
            "{,}",
        ],
    )
    def test_invalid_values_raise(self, raw):
        with pytest.raises(PreprocessingError) as exc:
            parse_osm_tags(raw)
        assert "osm_tags" in str(exc.value)


# ===================================================================
# normalize_sentence
# ===================================================================
class TestNormalizeSentence:
    def test_reject_non_string(self):
        for value in [None, 123, ["x"], {"a": "b"}]:
            with pytest.raises(PreprocessingError) as exc:
                normalize_sentence(value)  # type: ignore[arg-type]
            msg = str(exc.value)
            assert "normalize_sentence" in msg
            assert "str" in msg

    def test_nfc_normalization(self):
        # Precomposed vs decomposed forms must match after NFC.
        composed = "\u00e9"  # é
        decomposed = "\u0065\u0301"  # e + combining acute
        import unicodedata

        normalized = normalize_sentence(decomposed)
        assert normalized == composed
        assert normalized == unicodedata.normalize("NFC", decomposed)

    def test_whitespace_and_nbsp(self):
        assert normalize_sentence("a\u00a0\u00a0b") == "a b"
        assert normalize_sentence("a   \t  b") == "a b"

    def test_control_characters_replaced_with_spaces(self):
        # U+0007 bell and U+0001 become spaces, then collapse.
        text = "ab\u0007cd\u0001ef"
        assert normalize_sentence(text) == "ab cd ef"

    def test_removable_zero_width_characters(self):
        for ch in ("\u200b", "\u2060", "\ufeff"):
            assert normalize_sentence(f"a{ch}b") == "ab"

    def test_preserved_zwnj_zwj(self):
        # ZWNJ (U+200C) and ZWJ (U+200D) must be preserved.
        assert normalize_sentence("a\u200cb") == "a\u200cb"
        assert normalize_sentence("a\u200db") == "a\u200db"

    def test_english_edit_marker_removed(self):
        text = "[ edit | edit source ] The bridge was built in 1900."
        assert normalize_sentence(text) == "The bridge was built in 1900."

    def test_non_english_edit_marker_removed(self):
        # Wikimedia marker using French-style label.
        text = "[ modifier | modifier le code ] La ville est ancienne."
        assert normalize_sentence(text) == "La ville est ancienne."

    def test_repeated_leading_markers_removed(self):
        text = "[ edit | edit ] [ edit | history ] The river flows north."
        assert normalize_sentence(text) == "The river flows north."

    def test_bracketed_content_later_remains(self):
        # Marker must only be removed when at the leading position.
        text = "The lake [ see also | reference ] is deep."
        assert normalize_sentence(text) == (
            "The lake [ see also | reference ] is deep."
        )


# ===================================================================
# Amend 3A: marker removal requires a pipe in the first bracket
# ===================================================================
class TestEditMarkerRequiresPipe:
    def test_simple_marker_with_pipe(self):
        assert normalize_sentence("[ edit | source ] Text.") == "Text."

    def test_non_english_marker_with_pipe(self):
        assert normalize_sentence("[ سمول | سرچينه سمول ] متن.") == "متن."

    def test_repeated_valid_markers(self):
        text = "[ edit | source ] [ edit | history ] The river flows north."
        assert normalize_sentence(text) == "The river flows north."

    def test_first_marker_without_pipe_preserves_text(self):
        assert (
            normalize_sentence("[citation] Important text.")
            == "[citation] Important text."
        )

    def test_first_marker_without_pipe_blocks_later_marker(self):
        assert (
            normalize_sentence("[no pipe] [ edit | source ] Text.")
            == "[no pipe] [ edit | source ] Text."
        )

    def test_bracketed_text_later_remains(self):
        text = "The lake [ see also | reference ] is deep."
        assert normalize_sentence(text) == (
            "The lake [ see also | reference ] is deep."
        )

    def test_closing_bracket_beyond_120_chars_not_removed(self):
        long_content = "x" * 125
        text = f"[{long_content}] Text."
        assert normalize_sentence(text) == text

    def test_pipe_after_closing_bracket_does_not_qualify(self):
        text = "[citation] | trailing Text."
        assert normalize_sentence(text) == "[citation] | trailing Text."

    def test_empty_marker_without_pipe_preserved(self):
        text = "[] Text."
        assert normalize_sentence(text) == "[] Text."

    def test_marker_closing_at_character_120_is_removed(self):
        text = "[" + ("x" * 117) + "|" + "] Text."
        assert text.index("]") + 1 == 120
        assert normalize_sentence(text) == "Text."

    def test_marker_closing_at_character_121_is_preserved(self):
        text = "[" + ("x" * 118) + "|" + "] Text."
        assert text.index("]") + 1 == 121
        assert normalize_sentence(text) == text

    def test_punctuation_case_diacritics_remain(self):
        text = "Café: Bézier's 100% ( naïve )."
        assert normalize_sentence(text) == "Café: Bézier's 100% ( naïve )."

    def test_empty_normalized_result(self):
        # Whitespace / zero-width only input collapses to empty string.
        assert normalize_sentence("\u200b \u00a0 \u200b") == ""
        assert normalize_sentence("   ") == ""

    def test_trim_and_collapse(self):
        assert normalize_sentence("  Test.  ") == "Test."
