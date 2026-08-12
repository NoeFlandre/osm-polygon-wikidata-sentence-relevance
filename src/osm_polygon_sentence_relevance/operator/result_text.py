"""Shared text normalization for local and remote command results."""

from __future__ import annotations


def result_text(result: object, *, fallback_on_empty: bool = False) -> str:
    """Return the result's textual output, preferring ``text`` when present.

    ``subprocess.CompletedProcess`` exposes ``stdout`` while the SSH adapter
    and test doubles may expose ``text``. The explicit ``is not None`` check
    preserves an intentionally empty ``text`` value instead of silently
    replacing it with ``stdout``. Set ``fallback_on_empty`` only for legacy
    compatibility seams whose historical contract treated an empty ``text``
    field as absent.
    """

    text = getattr(result, "text", None)
    if text is not None and (not fallback_on_empty or text):
        return str(text)
    return str(getattr(result, "stdout", ""))


__all__ = ["result_text"]
