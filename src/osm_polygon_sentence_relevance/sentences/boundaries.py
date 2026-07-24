"""Conservative repair of high-confidence sentence-boundary false negatives.

SaT remains the authoritative segmenter.  This module only separates obvious
boundaries that remain inside one model segment; it is deliberately too
conservative to serve as a standalone rule-based sentence tokenizer.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

_ARABIC_SCRIPT_LANGUAGES = frozenset({"ar", "fa", "ps", "ur"})
_UNIVERSAL_TERMINALS = frozenset({"?", "!", "؟", "。", "！", "？", "।", "॥"})
_CLOSERS = frozenset({'"', "'", "»", "”", "’", ")", "]", "}"})


def _primary_language(language: str) -> str:
    return language.casefold().split("-", 1)[0].split("_", 1)[0]


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_word_character(character: str) -> bool:
    return unicodedata.category(character)[0] in {"L", "M"}


def _is_arabic_letter(character: str) -> bool:
    codepoint = ord(character)
    in_arabic_block = (
        0x0600 <= codepoint <= 0x06FF
        or 0x0750 <= codepoint <= 0x077F
        or 0x08A0 <= codepoint <= 0x08FF
        or 0xFB50 <= codepoint <= 0xFDFF
        or 0xFE70 <= codepoint <= 0xFEFF
    )
    return in_arabic_block and _is_letter(character)


def _preceding_word(text: str, terminal_index: int) -> str:
    cursor = terminal_index - 1
    while cursor >= 0 and text[cursor].isspace():
        cursor -= 1
    end = cursor + 1
    while cursor >= 0 and _is_word_character(text[cursor]):
        cursor -= 1
    return text[cursor + 1 : end]


def _is_url_query_mark(text: str, terminal_index: int) -> bool:
    """Return whether ``text[terminal_index]`` is a URL query delimiter."""

    token_start = terminal_index - 1
    while token_start >= 0 and not text[token_start].isspace():
        token_start -= 1
    prefix = text[token_start + 1 : terminal_index].casefold()
    unwrapped = prefix.lstrip("([{<\"'")
    return "://" in prefix or unwrapped.startswith("www.")


def find_high_confidence_boundaries(text: str, language: str) -> tuple[int, ...]:
    """Return exclusive offsets for unambiguous residual boundaries.

    Offsets include terminal punctuation and any immediately following closing
    quote/bracket, but not inter-sentence whitespace.
    """

    if not isinstance(text, str) or not isinstance(language, str):
        raise TypeError("text and language must be strings")

    arabic_language = _primary_language(language) in _ARABIC_SCRIPT_LANGUAGES
    boundaries: list[int] = []
    clause_start = 0

    for index, character in enumerate(text):
        is_universal = character in _UNIVERSAL_TERMINALS
        is_period = character == "."
        if not is_universal and not is_period:
            continue
        if character == "?" and _is_url_query_mark(text, index):
            continue

        boundary_end = index + 1
        while boundary_end < len(text) and text[boundary_end] in _CLOSERS:
            boundary_end += 1
        next_index = boundary_end
        while next_index < len(text) and text[next_index].isspace():
            next_index += 1
        if next_index >= len(text) or not _is_letter(text[next_index]):
            continue

        left = text[clause_start:boundary_end].strip()
        right = text[next_index:].strip()
        if is_period:
            if arabic_language and not _is_arabic_letter(text[next_index]):
                continue
            # A lowercase continuation is much more likely to follow an
            # abbreviation than to start a new sentence.  Uncased scripts
            # report neither lower nor upper and remain eligible.
            if text[next_index].islower():
                continue
            # Do not split initials/short titles such as ``د. محمد`` or
            # ``Dr. Smith``.
            if len(_preceding_word(text, index)) <= 2:
                continue
            if len(left) < 12 or len(right) < 12:
                continue
        elif len(left) < 2 or len(right) < 2:
            continue

        boundaries.append(boundary_end)
        clause_start = next_index

    return tuple(boundaries)


def refine_sentence_boundaries(
    segments: Sequence[str],
    language: str,
) -> tuple[str, ...]:
    """Split high-confidence residual boundaries while preserving order."""

    refined: list[str] = []
    for segment in segments:
        offsets = find_high_confidence_boundaries(segment, language)
        start = 0
        for end in offsets:
            candidate = segment[start:end].strip()
            if candidate:
                refined.append(candidate)
            start = end
            while start < len(segment) and segment[start].isspace():
                start += 1
        tail = segment[start:].strip()
        if tail:
            refined.append(tail)
        elif not offsets:
            # Preserve empty model outputs for the shared preparation layer,
            # which owns empty-segment accounting.
            refined.append(segment)
    return tuple(refined)


__all__ = ["find_high_confidence_boundaries", "refine_sentence_boundaries"]
