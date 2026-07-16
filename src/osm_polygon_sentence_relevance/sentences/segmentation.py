"""Sentence-segmenter interface and validated batch boundary (Phase 3B).

This module defines the segmentation contract used by later orchestration.
It does NOT provide a concrete model adapter (e.g. wtpsplit), nor does it
perform normalization, PyArrow transforms, deduplication, or any writing.

The only public entry point is :func:`split_validated_batch`, which wraps an
arbitrary :class:`SentenceSegmenter` and enforces the structural contract on
inputs and on the returned segments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from osm_polygon_sentence_relevance.contracts.errors import SegmentationError
from osm_polygon_sentence_relevance.sentences.preprocessing import normalize_sentence


@runtime_checkable
class SentenceSegmenter(Protocol):
    """Protocol implemented by concrete sentence segmenters."""

    def split_batch(
        self,
        texts: Sequence[str],
        languages: Sequence[str],
    ) -> Sequence[Sequence[str]]:
        """Segment each text into a sequence of sentence strings."""
        ...


@dataclass(frozen=True, slots=True)
class PreparedSentence:
    """A single validated sentence extracted from one section."""

    sentence_index: int
    sentence_text_raw: str
    sentence_text_normalized: str


@dataclass(frozen=True, slots=True)
class PreparedSection:
    """The per-section result of sentence preparation."""

    sentences: tuple[PreparedSentence, ...]
    emitted_segment_count: int
    dropped_empty_raw_count: int
    dropped_empty_normalized_count: int


@dataclass(frozen=True, slots=True)
class SegmentationReport:
    """Aggregated statistics across prepared sections."""

    input_section_occurrence_count: int
    emitted_segment_count: int
    retained_sentence_occurrence_count: int
    dropped_empty_raw_count: int
    dropped_empty_normalized_count: int
    wikipedia_sentence_occurrence_count: int
    wikivoyage_sentence_occurrence_count: int


def segment_one_section(
    text: str,
    language: str,
    segmenter: SentenceSegmenter,
) -> PreparedSection:
    """Segment a single section and prepare immutable sentence records.

    Calls :func:`split_validated_batch` with one text/language pair, then
    prepares the returned group via :func:`_prepare_section`.
    """
    (group,) = split_validated_batch(segmenter, [text], [language])
    return _prepare_section(group)


def segment_sections_batch(
    texts: Sequence[str],
    languages: Sequence[str],
    segmenter: SentenceSegmenter,
) -> tuple[PreparedSection, ...]:
    """Segment and prepare multiple sections in one segmenter call.

    Calls :func:`split_validated_batch` exactly once with all texts and
    languages, then prepares each returned group independently via
    :func:`_prepare_section`, preserving section order. Empty input returns
    ``()`` without invoking the segmenter. Validation errors surface as
    :class:`SegmentationError`.
    """
    groups = split_validated_batch(segmenter, texts, languages)
    return tuple(_prepare_section(group) for group in groups)


def _prepare_section(segments: Sequence[str]) -> PreparedSection:
    """Apply the shared preparation rules to one segmenter group.

    Trims each raw segment's surrounding whitespace, drops and counts
    segments that are empty after trimming (raw) or after normalization
    (normalized), keeps all other segments (including duplicates and short
    text), and assigns ``sentence_index`` after filtering, starting at zero.
    """
    prepared: list[PreparedSentence] = []
    dropped_empty_raw = 0
    dropped_empty_normalized = 0

    for raw in segments:
        stripped = raw.strip()
        if stripped == "":
            dropped_empty_raw += 1
            continue
        normalized = normalize_sentence(stripped)
        if normalized == "":
            dropped_empty_normalized += 1
            continue
        prepared.append(
            PreparedSentence(
                sentence_index=len(prepared),
                sentence_text_raw=stripped,
                sentence_text_normalized=normalized,
            )
        )

    return PreparedSection(
        sentences=tuple(prepared),
        emitted_segment_count=len(segments),
        dropped_empty_raw_count=dropped_empty_raw,
        dropped_empty_normalized_count=dropped_empty_normalized,
    )


def build_segmentation_report(
    sections: Sequence[PreparedSection],
    sources: Sequence[str],
) -> SegmentationReport:
    """Aggregate statistics across prepared sections.

    ``sections`` and ``sources`` must have equal lengths, and every source
    must be exactly ``"wikipedia"`` or ``"wikivoyage"``. Per-source counts
    reflect retained sentences (after drops), not sections. Empty inputs
    yield an all-zero report. Inputs are not mutated.
    """
    if len(sections) != len(sources):
        raise SegmentationError(
            "build_segmentation_report: sections and sources must have equal "
            f"lengths (got {len(sections)} sections, {len(sources)} sources)"
        )

    input_section_occurrence_count = 0
    emitted_segment_count = 0
    retained_sentence_occurrence_count = 0
    dropped_empty_raw_count = 0
    dropped_empty_normalized_count = 0
    wikipedia_sentence_occurrence_count = 0
    wikivoyage_sentence_occurrence_count = 0

    for section, source in zip(sections, sources, strict=True):
        if source not in ("wikipedia", "wikivoyage"):
            raise SegmentationError(
                "build_segmentation_report: source must be 'wikipedia' or "
                f"'wikivoyage', got {source!r}"
            )
        input_section_occurrence_count += 1
        emitted_segment_count += section.emitted_segment_count
        dropped_empty_raw_count += section.dropped_empty_raw_count
        dropped_empty_normalized_count += section.dropped_empty_normalized_count
        retained = len(section.sentences)
        retained_sentence_occurrence_count += retained
        if source == "wikipedia":
            wikipedia_sentence_occurrence_count += retained
        else:
            wikivoyage_sentence_occurrence_count += retained

    return SegmentationReport(
        input_section_occurrence_count=input_section_occurrence_count,
        emitted_segment_count=emitted_segment_count,
        retained_sentence_occurrence_count=retained_sentence_occurrence_count,
        dropped_empty_raw_count=dropped_empty_raw_count,
        dropped_empty_normalized_count=dropped_empty_normalized_count,
        wikipedia_sentence_occurrence_count=wikipedia_sentence_occurrence_count,
        wikivoyage_sentence_occurrence_count=wikivoyage_sentence_occurrence_count,
    )


def split_validated_batch(
    segmenter: SentenceSegmenter,
    texts: Sequence[str],
    languages: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    """Validate inputs, call the segmenter once, and validate its output.

    Returns an immutable nested tuple of sentence groups, one per input text,
    preserving input and segment order. Empty inputs return ``()`` without
    invoking the segmenter. Structural contract violations and exceptions
    raised by the segmenter are surfaced as :class:`SegmentationError`.
    """
    if len(texts) != len(languages):
        raise SegmentationError(
            "split_validated_batch: texts and languages must have equal lengths "
            f"(got {len(texts)} texts, {len(languages)} languages)"
        )

    for text in texts:
        if not isinstance(text, str):
            raise SegmentationError(
                "split_validated_batch: every text must be a string, "
                f"got {type(text).__name__}"
            )

    for language in languages:
        if not isinstance(language, str):
            raise SegmentationError(
                "split_validated_batch: every language must be a string, "
                f"got {type(language).__name__}"
            )

    if not texts:
        return ()

    try:
        result = segmenter.split_batch(texts, languages)
    except Exception as exc:  # intentionally broad: wrap any adapter failure
        raise SegmentationError(
            "split_validated_batch: segmenter raised an error"
        ) from exc

    _assert_valid_sequence(
        result,
        "outer",
        "sequence of sentence groups",
        len(texts),
    )
    result_seq = result  # type is now Sequence[Sequence[str]]

    validated: list[tuple[str, ...]] = []
    for group in result_seq:
        _assert_valid_sequence(
            group,
            "sentence group",
            "sequence of segments",
        )
        segments = tuple(group)
        for segment in segments:
            if not isinstance(segment, str):
                raise SegmentationError(
                    "split_validated_batch: every emitted segment must be a "
                    f"string, got {type(segment).__name__}"
                )
        validated.append(segments)

    return tuple(validated)


def _assert_valid_sequence(
    value: object,
    scope: str,
    expected: str,
    expected_length: int | None = None,
) -> None:
    """Validate ``value`` as a non-string ``Sequence``.

    Raises :class:`SegmentationError` when the value is a string, is not a
    ``Sequence`` (e.g. a generator or scalar), or, when ``expected_length``
    is given, has a mismatched length. Strings get a specific message because
    they are technically sequences but never valid here.
    """
    if isinstance(value, str):
        raise SegmentationError(
            f"split_validated_batch: {scope} was a string instead of a {expected}"
        )
    if not isinstance(value, Sequence):
        raise SegmentationError(
            f"split_validated_batch: {scope} must be a {expected}, "
            f"got {type(value).__name__}"
        )
    if expected_length is not None and len(value) != expected_length:
        raise SegmentationError(
            "split_validated_batch: length mismatch, segmenter returned "
            f"{len(value)} groups for {expected_length} texts"
        )
