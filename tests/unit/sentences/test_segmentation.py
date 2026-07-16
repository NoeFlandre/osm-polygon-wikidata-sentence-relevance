"""Tests for the Phase 3B sentence-segmenter interface and batch validation."""

from __future__ import annotations

import pytest

from osm_polygon_sentence_relevance.errors import SegmentationError
from osm_polygon_sentence_relevance.segmentation import (
    PreparedSection,
    PreparedSentence,
    SegmentationReport,
    SentenceSegmenter,
    build_segmentation_report,
    segment_one_section,
    segment_sections_batch,
    split_validated_batch,
)


class FakeSegmenter:
    """A tiny segmenter used to assert call behaviour in tests.

    Stores the exact result (including ``None`` or a generator) so tests can
    exercise invalid container shapes without substitution.
    """

    def __init__(self, result=None, side_effect=None):
        self.result = result
        self.side_effect = side_effect
        self.calls = 0

    def split_batch(self, texts, languages):
        self.calls += 1
        if self.side_effect is not None:
            raise self.side_effect
        return self.result


def make_segmenter(mapping):
    """Map input texts to outputs and ignore languages."""

    class _Seg:
        def split_batch(self, texts, languages):
            return [mapping[text] for text in texts]

    return _Seg()


class TestSplitValidatedBatch:
    def test_valid_batch_and_exact_order(self):
        seg = make_segmenter(
            {
                "A. B.": ["A.", "B."],
                "C.": ["C."],
            }
        )
        out = split_validated_batch(seg, ["A. B.", "C."], ["en", "en"])
        assert out == (("A.", "B."), ("C.",))

    def test_immutable_tuple_result(self):
        seg = make_segmenter({"A. B.": ["A.", "B."]})
        out = split_validated_batch(seg, ["A. B."], ["en"])
        assert isinstance(out, tuple)
        assert all(isinstance(group, tuple) for group in out)
        # Outer nested tuple must be immutable (rebinding raises TypeError).
        with pytest.raises(TypeError):
            out[0] = ("x",)  # type: ignore[index]
        # Each inner group is itself an immutable tuple.
        assert tuple(out[0]) is out[0]

    def test_mismatched_input_lengths(self):
        seg = FakeSegmenter(result=[[]])
        with pytest.raises(SegmentationError) as exc:
            split_validated_batch(seg, ["a", "b"], ["en"])
        assert "length" in str(exc.value).lower()

    def test_non_string_text(self):
        seg = FakeSegmenter(result=[[]])
        with pytest.raises(SegmentationError) as exc:
            split_validated_batch(seg, ["a", 1], ["en", "en"])  # type: ignore[list-item]
        assert "text" in str(exc.value).lower()

    def test_non_string_language(self):
        seg = FakeSegmenter(result=[[]])
        with pytest.raises(SegmentationError) as exc:
            split_validated_batch(seg, ["a", "b"], ["en", 2])  # type: ignore[list-item]
        assert "language" in str(exc.value).lower()

    def test_empty_input_does_not_call_fake(self):
        seg = FakeSegmenter(result=[])
        out = split_validated_batch(seg, [], [])
        assert out == ()
        assert seg.calls == 0

    def test_segmenter_called_exactly_once(self):
        seg = FakeSegmenter(result=[["x"], ["y"], ["z"]])
        split_validated_batch(seg, ["a", "b", "c"], ["en"] * 3)
        assert seg.calls == 1

    def test_wrong_outer_result_length(self):
        # Segmenter returns fewer groups than inputs.
        seg = FakeSegmenter(result=[["x"]])
        with pytest.raises(SegmentationError) as exc:
            split_validated_batch(seg, ["a", "b"], ["en", "en"])
        assert "length" in str(exc.value).lower()

    def test_outer_result_is_a_string(self):
        seg = FakeSegmenter(result="not a list of groups")  # type: ignore[list-item]
        with pytest.raises(SegmentationError) as exc:
            split_validated_batch(seg, ["a"], ["en"])
        assert "string" in str(exc.value).lower()

    def test_inner_group_is_a_string(self):
        seg = FakeSegmenter(result=["not a sequence"])  # type: ignore[list-item]
        with pytest.raises(SegmentationError) as exc:
            split_validated_batch(seg, ["a"], ["en"])
        assert "string" in str(exc.value).lower()

    def test_non_string_emitted_segment(self):
        seg = FakeSegmenter(result=[["x", 5]])  # type: ignore[list-item]
        with pytest.raises(SegmentationError) as exc:
            split_validated_batch(seg, ["a"], ["en"])
        assert "string" in str(exc.value).lower()

    def test_empty_emitted_string_preserved(self):
        seg = make_segmenter({"a": ["", "b"]})
        out = split_validated_batch(seg, ["a"], ["en"])
        assert out == (("", "b"),)

    def test_adapter_exception_wrapped_and_retained(self):
        cause = RuntimeError("boom")
        seg = FakeSegmenter(side_effect=cause)
        with pytest.raises(SegmentationError) as exc:
            split_validated_batch(seg, ["a"], ["en"])
        assert exc.value.__cause__ is cause


def test_protocol_is_runtime_checkable():
    assert isinstance(FakeSegmenter(), SentenceSegmenter)


class TestInvalidSegmenterOutput:
    @pytest.mark.parametrize(
        ("result", "scope"),
        [
            (None, "outer"),  # type: ignore[arg-type]
            (123, "outer"),  # type: ignore[arg-type]
            ([None], "group"),  # type: ignore[list-item]
            ([123], "group"),  # type: ignore[list-item]
            ((s for s in ["x"]), "outer"),  # type: ignore[arg-type]
            ([(s for s in ["x"])], "group"),  # type: ignore[list-item]
        ],
    )
    def test_invalid_output_raises_segmentation_error(self, result, scope):
        seg = FakeSegmenter(result=result)
        with pytest.raises(SegmentationError) as exc:
            split_validated_batch(seg, ["a"], ["en"])
        msg = str(exc.value).lower()
        if scope == "outer":
            assert "outer" in msg
        else:
            assert "group" in msg


class TestSegmentOneSection:
    def test_multiple_segments_preserve_order(self):
        seg = make_segmenter({"text": ["First.", "Second.", "Third."]})
        result = segment_one_section("text", "en", seg)
        texts = [s.sentence_text_raw for s in result.sentences]
        assert texts == ["First.", "Second.", "Third."]

    def test_raw_surrounding_whitespace_trimmed(self):
        seg = make_segmenter({"text": ["  Padded.  "]})
        result = segment_one_section("text", "en", seg)
        assert result.sentences[0].sentence_text_raw == "Padded."

    def test_normalized_text_uses_normalize_sentence(self):
        seg = make_segmenter({"text": ["Café."]})
        result = segment_one_section("text", "en", seg)
        assert result.sentences[0].sentence_text_normalized == "Café."

    def test_whitespace_only_raw_segment_dropped_and_counted(self):
        seg = make_segmenter({"text": ["  ", "Kept."]})
        result = segment_one_section("text", "en", seg)
        assert [s.sentence_text_raw for s in result.sentences] == ["Kept."]
        assert result.dropped_empty_raw_count == 1
        assert result.dropped_empty_normalized_count == 0

    def test_marker_only_normalized_segment_dropped_and_counted(self):
        # A zero-width-only raw segment is non-empty before normalization
        # but normalizes (after zero-width removal) to an empty string.
        seg = make_segmenter({"text": ["\u200b", "Kept."]})
        result = segment_one_section("text", "en", seg)
        assert [s.sentence_text_raw for s in result.sentences] == ["Kept."]
        assert result.dropped_empty_normalized_count == 1

    def test_indices_assigned_after_filtering(self):
        seg = make_segmenter({"text": ["  ", "One.", "Two."]})
        result = segment_one_section("text", "en", seg)
        assert [s.sentence_index for s in result.sentences] == [0, 1]

    def test_duplicate_sentences_preserved(self):
        seg = make_segmenter({"text": ["Same.", "Same."]})
        result = segment_one_section("text", "en", seg)
        assert len(result.sentences) == 2

    def test_one_character_sentence_preserved(self):
        seg = make_segmenter({"text": ["A."]})
        result = segment_one_section("text", "en", seg)
        assert result.sentences[0].sentence_text_raw == "A."

    def test_empty_segmenter_result(self):
        seg = make_segmenter({"text": []})
        result = segment_one_section("text", "en", seg)
        assert result.sentences == ()
        assert result.emitted_segment_count == 0

    def test_language_passed_unchanged(self):
        captured = {}

        class _Seg:
            def split_batch(self, texts, languages):
                captured["languages"] = languages
                return [["x"]]

        segment_one_section("text", "fr", _Seg())
        assert captured["languages"] == ["fr"]

    def test_dataclasses_immutable(self):
        seg = make_segmenter({"text": ["One."]})
        result = segment_one_section("text", "en", seg)
        with pytest.raises(AttributeError):
            result.sentences = ()  # type: ignore[misc]
        with pytest.raises(AttributeError):
            result.sentences[0].sentence_index = 9  # type: ignore[misc]

    def test_validation_errors_propagate_as_segmentation_error(self):
        seg = make_segmenter({"text": [5]})  # type: ignore[list-item]
        with pytest.raises(SegmentationError):
            segment_one_section("text", "en", seg)


class TestSegmentSectionsBatch:
    def test_two_sections_processed_in_one_call(self):
        mapping = {
            "a": ["One.", "Two."],
            "b": ["Three."],
        }

        class _Counting:
            def __init__(self):
                self.calls = 0

            def split_batch(self, texts, languages):
                self.calls += 1
                return [mapping[text] for text in texts]

        seg = _Counting()
        results = segment_sections_batch(["a", "b"], ["en", "en"], seg)
        assert len(results) == 2
        assert seg.calls == 1

    def test_section_order_preserved(self):
        mapping = {
            "first": ["Alpha."],
            "second": ["Beta."],
        }
        seg = make_segmenter(mapping)
        results = segment_sections_batch(["first", "second"], ["en", "en"], seg)
        assert results[0].sentences[0].sentence_text_raw == "Alpha."
        assert results[1].sentences[0].sentence_text_raw == "Beta."

    def test_sentence_indices_reset_per_section(self):
        mapping = {
            "a": ["One.", "Two."],
            "b": ["Three."],
        }
        seg = make_segmenter(mapping)
        results = segment_sections_batch(["a", "b"], ["en", "en"], seg)
        assert [s.sentence_index for s in results[0].sentences] == [0, 1]
        assert [s.sentence_index for s in results[1].sentences] == [0]

    def test_drop_counters_independent_per_section(self):
        mapping = {
            "a": ["  ", "One."],
            "b": ["\u200b", "Two."],
        }
        seg = make_segmenter(mapping)
        results = segment_sections_batch(["a", "b"], ["en", "en"], seg)
        assert results[0].dropped_empty_raw_count == 1
        assert results[1].dropped_empty_normalized_count == 1

    def test_empty_input_does_not_call_segmenter(self):
        seg = FakeSegmenter(result=[])
        results = segment_sections_batch([], [], seg)
        assert results == ()
        assert seg.calls == 0

    def test_mismatched_lengths_raise_segmentation_error(self):
        seg = make_segmenter({})
        with pytest.raises(SegmentationError):
            segment_sections_batch(["a", "b"], ["en"], seg)

    def test_duplicate_sentences_remain(self):
        mapping = {"a": ["Same.", "Same."], "b": ["Other."]}
        seg = make_segmenter(mapping)
        results = segment_sections_batch(["a", "b"], ["en", "en"], seg)
        assert len(results[0].sentences) == 2
        assert len(results[1].sentences) == 1

    def test_segment_one_section_unchanged_after_refactor(self):
        seg = make_segmenter({"text": ["First.", "Second."]})
        single = segment_one_section("text", "en", seg)
        batch = segment_sections_batch(["text"], ["en"], seg)[0]
        assert single == batch


def _section(sentences, dropped_raw=0, dropped_norm=0):
    return PreparedSection(
        sentences=tuple(sentences),
        emitted_segment_count=len(sentences) + dropped_raw + dropped_norm,
        dropped_empty_raw_count=dropped_raw,
        dropped_empty_normalized_count=dropped_norm,
    )


class TestBuildSegmentationReport:
    def test_mixed_source_aggregation(self):
        sections = [
            _section([PreparedSentence(0, "A.", "A.")]),
            _section([PreparedSentence(0, "B.", "B.")]),
        ]
        report = build_segmentation_report(sections, ["wikipedia", "wikivoyage"])
        assert report.wikipedia_sentence_occurrence_count == 1
        assert report.wikivoyage_sentence_occurrence_count == 1
        assert report.retained_sentence_occurrence_count == 2

    def test_multiple_sections_one_source(self):
        sections = [
            _section([PreparedSentence(0, "A.", "A.")]),
            _section(
                [
                    PreparedSentence(0, "B.", "B."),
                    PreparedSentence(1, "C.", "C."),
                ]
            ),
        ]
        report = build_segmentation_report(sections, ["wikipedia", "wikipedia"])
        assert report.wikipedia_sentence_occurrence_count == 3
        assert report.wikivoyage_sentence_occurrence_count == 0

    def test_dropped_counters(self):
        sections = [
            _section([PreparedSentence(0, "A.", "A.")], dropped_raw=2),
            _section([], dropped_norm=1),
        ]
        report = build_segmentation_report(sections, ["wikipedia", "wikivoyage"])
        assert report.dropped_empty_raw_count == 2
        assert report.dropped_empty_normalized_count == 1
        assert report.emitted_segment_count == 4

    def test_zero_sentence_section(self):
        sections = [_section([])]
        report = build_segmentation_report(sections, ["wikivoyage"])
        assert report.retained_sentence_occurrence_count == 0
        assert report.wikivoyage_sentence_occurrence_count == 0

    def test_empty_inputs(self):
        report = build_segmentation_report([], [])
        assert report == SegmentationReport(
            input_section_occurrence_count=0,
            emitted_segment_count=0,
            retained_sentence_occurrence_count=0,
            dropped_empty_raw_count=0,
            dropped_empty_normalized_count=0,
            wikipedia_sentence_occurrence_count=0,
            wikivoyage_sentence_occurrence_count=0,
        )

    def test_mismatched_lengths(self):
        sections = [_section([PreparedSentence(0, "A.", "A.")])]
        with pytest.raises(SegmentationError):
            build_segmentation_report(sections, ["wikipedia", "wikivoyage"])

    def test_invalid_source(self):
        sections = [_section([PreparedSentence(0, "A.", "A.")])]
        with pytest.raises(SegmentationError):
            build_segmentation_report(sections, ["web"])

    def test_immutable(self):
        sections = [_section([PreparedSentence(0, "A.", "A.")])]
        report = build_segmentation_report(sections, ["wikipedia"])
        with pytest.raises(AttributeError):
            report.emitted_segment_count = 9  # type: ignore[misc]
