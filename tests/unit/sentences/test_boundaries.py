"""Regression tests for conservative residual sentence-boundary repair."""

from __future__ import annotations

import pytest

from osm_polygon_sentence_relevance.sentences.boundaries import (
    find_high_confidence_boundaries,
    refine_sentence_boundaries,
)
from osm_polygon_sentence_relevance.sentences.segmentation import segment_one_section

ARABIC_REGRESSION = (
    "مطار نيلي ( الإنجليزية : Nili Airport ) ( إيكاو : OANL ) يتم استخدامه "
    "لاغراض العسكرية والعامة. المطار يقع في نيلي، مقاطعة دايكندي ، أفغانستان . "
    "أعيد بناء المطار في السنوات الأخيرة بدعم من القوة الدولية للمساعدة الأمنية "
    "(إيساف)، وتستخدم بشكل أساسي من قبل القوة الدولية للمساعدة الأمنية والقوات "
    "الجوية الأفغانية لأغراض عسكرية أو إيصال مساعدات الطوارئ إلى الناس في المنطقة."
)


class _OneSegment:
    def split_batch(self, texts, languages):
        return [[text] for text in texts]


def test_exact_arabic_regression_is_three_sentences() -> None:
    assert refine_sentence_boundaries((ARABIC_REGRESSION,), "ar") == (
        "مطار نيلي ( الإنجليزية : Nili Airport ) ( إيكاو : OANL ) يتم استخدامه "
        "لاغراض العسكرية والعامة.",
        "المطار يقع في نيلي، مقاطعة دايكندي ، أفغانستان .",
        "أعيد بناء المطار في السنوات الأخيرة بدعم من القوة الدولية للمساعدة "
        "الأمنية (إيساف)، وتستخدم بشكل أساسي من قبل القوة الدولية للمساعدة "
        "الأمنية والقوات الجوية الأفغانية لأغراض عسكرية أو إيصال مساعدات الطوارئ "
        "إلى الناس في المنطقة.",
    )


def test_pipeline_repairs_model_false_negative_before_indexing() -> None:
    result = segment_one_section(ARABIC_REGRESSION, "ar", _OneSegment())
    assert [sentence.sentence_index for sentence in result.sentences] == [0, 1, 2]
    assert len(result.sentences) == 3
    assert result.emitted_segment_count == 3


def test_arabic_abbreviation_and_decimal_are_not_split() -> None:
    text = "قابل د. محمد في الساعة 10.5 صباحًا ثم غادر المطار."
    assert refine_sentence_boundaries((text,), "ar") == (text,)


def test_period_repair_is_not_applied_to_non_arabic_language() -> None:
    text = "Dr. Smith met the delegation. They left together."
    assert refine_sentence_boundaries((text,), "en") == (text,)


def test_universal_terminal_marks_are_repaired() -> None:
    assert refine_sentence_boundaries(("هل وصل الوفد؟ نعم، وصل الوفد.",), "ar") == (
        "هل وصل الوفد؟",
        "نعم، وصل الوفد.",
    )
    assert refine_sentence_boundaries(("第一句。第二句。",), "zh") == (
        "第一句。",
        "第二句。",
    )
    assert refine_sentence_boundaries(("पहला वाक्य। दूसरा वाक्य।",), "hi") == (
        "पहला वाक्य।",
        "दूसरा वाक्य।",
    )


def test_existing_model_segments_and_order_are_preserved() -> None:
    assert refine_sentence_boundaries(("First.", "Second.", "Third."), "en") == (
        "First.",
        "Second.",
        "Third.",
    )


def test_audit_finds_only_residual_high_confidence_boundaries() -> None:
    assert len(find_high_confidence_boundaries(ARABIC_REGRESSION, "ar")) == 2
    assert find_high_confidence_boundaries("د. محمد وصل.", "ar") == ()
    assert find_high_confidence_boundaries("One sentence.", "en") == ()


@pytest.mark.parametrize(
    ("text", "language"),
    [(None, "ar"), ("text", None), (1, "ar"), ("text", 1)],
)
def test_audit_rejects_non_string_inputs(text: object, language: object) -> None:
    with pytest.raises(TypeError, match="text and language must be strings"):
        find_high_confidence_boundaries(text, language)  # type: ignore[arg-type]


def test_closing_quote_is_kept_with_preceding_sentence() -> None:
    assert refine_sentence_boundaries(('هل وصل؟" نعم، وصل.',), "ar") == (
        'هل وصل؟"',
        "نعم، وصل.",
    )


def test_ambiguous_boundary_candidates_are_not_repaired() -> None:
    assert find_high_confidence_boundaries("هل؟", "ar") == ()
    assert find_high_confidence_boundaries("هل؟ 123", "ar") == ()
    assert (
        find_high_confidence_boundaries(
            "هذه جملة عربية طويلة بما يكفي. This continuation is Latin.",
            "ar",
        )
        == ()
    )
    assert find_high_confidence_boundaries("أ. هذه تتمة عربية طويلة.", "ar") == ()
    assert find_high_confidence_boundaries("A? B", "en") == ()


@pytest.mark.parametrize(
    "text",
    [
        "Reference https://www.youtube.com/watch?v=5F8SREfehZ4",
        "Reference http://example.org/path?q=value followed by prose",
        "Reference www.example.org/search?q=value followed by prose",
    ],
)
def test_question_mark_inside_url_is_not_a_sentence_boundary(text: str) -> None:
    assert find_high_confidence_boundaries(text, "en") == ()
    assert refine_sentence_boundaries((text,), "en") == (text,)


def test_empty_model_segment_is_preserved_for_shared_drop_accounting() -> None:
    assert refine_sentence_boundaries(("",), "en") == ("",)
