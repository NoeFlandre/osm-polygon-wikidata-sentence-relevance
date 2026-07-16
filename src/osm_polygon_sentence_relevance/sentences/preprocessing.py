"""Deterministic preprocessing utilities (Phase 3A).

These helpers parse and normalize raw string values that arrive from the
input dataset before any later-phase work (segmentation, modelling, etc.).
Behaviour is intentionally strict and deterministic:

* JSON parsing is validated to an exact shape.
* Tag keys are returned in sorted order.
* Sentence normalization follows a fixed, order-dependent pipeline that
  preserves case, punctuation, accents, and joiner characters.
"""

from __future__ import annotations

import json
import re
import unicodedata

from osm_polygon_sentence_relevance.contracts.errors import PreprocessingError

# Zero-width characters that are silently removed.
_ZERO_WIDTH_REMOVE = ("\u200b", "\u2060", "\ufeff")

# A leading MediaWiki-style edit marker "[ text | text ]" is only removed
# when its closing bracket occurs within this many leading characters.
_EDIT_MARKER_MAX = 120

# Horizontal tab included in the Cc set is already matched by \s; we operate
# directly on unicode whitespace below.


def parse_section_path(value: str) -> list[str]:
    """Parse a JSON array of strings.

    Accepts only a JSON array whose elements are all strings. Rejects
    malformed JSON, null, objects, scalars, and non-string elements.

    Raises:
        PreprocessingError: if the value is not a JSON array of strings.
    """
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError) as exc:
        raise PreprocessingError(
            f"section_path: expected a JSON array of strings, got invalid JSON: {exc}"
        ) from exc

    if not isinstance(parsed, list):
        raise PreprocessingError(
            f"section_path: expected a JSON array of strings, got {type(parsed).__name__}"
        )

    for element in parsed:
        if not isinstance(element, str):
            raise PreprocessingError(
                "section_path: expected a JSON array of strings, "
                f"got non-string element of type {type(element).__name__}"
            )

    return list(parsed)


def parse_osm_tags(value: str) -> dict[str, str]:
    """Parse a JSON object with string keys and string values.

    Returns the mapping with keys sorted for determinism. Rejects malformed
    JSON, null, arrays, scalars, and any non-string key or value.

    Raises:
        PreprocessingError: if the value is not a JSON object of strings.
    """
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError) as exc:
        raise PreprocessingError(
            f"osm_tags: expected a JSON object of strings, got invalid JSON: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise PreprocessingError(
            f"osm_tags: expected a JSON object of strings, got {type(parsed).__name__}"
        )

    for key, val in parsed.items():
        if not isinstance(key, str) or not isinstance(val, str):
            raise PreprocessingError(
                "osm_tags: expected a JSON object with string keys and string "
                f"values, got {type(key).__name__}: {type(val).__name__}"
            )

    return {key: parsed[key] for key in sorted(parsed)}


def normalize_sentence(text: str) -> str:
    """Normalize a sentence deterministically.

    Steps, in order:
        1. reject non-string input (PreprocessingError);
        2. Unicode NFC normalization;
        3. remove U+200B, U+2060, U+FEFF;
        4. preserve U+200C and U+200D (no-op);
        5. replace Unicode ``Cc`` control characters with spaces;
        6. collapse Unicode whitespace to one ASCII space;
        7. trim;
        8. remove consecutive leading MediaWiki edit markers "[ text | text ]"
           whose closing bracket occurs within 120 characters and whose
           bracketed content contains a pipe ``|``;
        9. collapse and trim whitespace again.

    Case, punctuation, accents, and joiner characters are preserved.
    """
    if not isinstance(text, str):
        raise PreprocessingError(
            f"normalize_sentence: expected str, got {type(text).__name__}"
        )

    # 2. NFC normalization.
    text = unicodedata.normalize("NFC", text)

    # 3. Remove removable zero-width characters.
    for ch in _ZERO_WIDTH_REMOVE:
        text = text.replace(ch, "")

    # 5. Replace Cc control characters (excluding those already handled)
    #    with spaces. ZWNJ/ZWJ (Cf) are preserved, as are joiners.
    chars = []
    for ch in text:
        if unicodedata.category(ch) == "Cc":
            chars.append(" ")
        else:
            chars.append(ch)
    text = "".join(chars)

    # 6 + 7. Collapse Unicode whitespace to a single ASCII space and trim.
    text = _collapse_whitespace(text)

    # 8. Remove consecutive leading edit markers. A marker only qualifies
    #    when it starts at the current position, its closing bracket occurs
    #    within 120 characters, and its content contains a pipe "|". If the
    #    first leading bracket is not a valid marker, stop entirely.
    while True:
        text = text.lstrip()
        consumed = _strip_one_marker(text)
        if consumed is None:
            break
        text = text[consumed:]

    # 9. Collapse and trim again.
    text = _collapse_whitespace(text)

    return text


def _strip_one_marker(text: str) -> int | None:
    """Return the number of leading chars forming a valid edit marker.

    A valid marker starts with ``[`` (the slice is already left-trimmed),
    closes with ``]`` within ``_EDIT_MARKER_MAX`` characters, and contains a
    ``|`` in its bracketed content. Returns ``None`` otherwise.
    """
    if not text.startswith("["):
        return None
    end = text.find("]")
    if end == -1 or end >= _EDIT_MARKER_MAX:
        return None
    if "|" not in text[: end + 1]:
        return None
    return end + 1


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of Unicode whitespace to one ASCII space and trim."""
    return re.sub(r"\s+", " ", text).strip()
