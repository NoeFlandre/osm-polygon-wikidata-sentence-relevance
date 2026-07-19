"""Render an exception chain for safe stderr reporting.

Used by the CLI to surface ``__cause__`` / ``__context__`` chains
without printing Python tracebacks (which embed local file paths,
line numbers, and frame state) and without recursing forever if
the chain contains a cycle.

Public contract:

    format_exception_chain(exc, *, max_depth=8, max_message_length=512)
        -> str

The returned string is intended for stable, machine-parseable
output: it is ``\\n``-separated, uses ``Error:`` and ``Caused by:``
prefixes, and never embeds traceback frames, file paths, or local
variable bindings.
"""

from __future__ import annotations

from typing import Final

#: Default depth cap. 8 is enough for typical CUDA / allocator /
#: segmenter / wtpsplit / pipeline nested failures, while preventing
#: pathological chains from producing runaway output.
DEFAULT_MAX_DEPTH: Final[int] = 8

#: Default per-message length cap. Truncated with an ellipsis if a
#: single exception message exceeds this length, so a 5 MiB tensor
#: printed inside a CUDA error does not blow up the build log.
DEFAULT_MAX_MESSAGE_LENGTH: Final[int] = 512


def _truncate(message: str, max_length: int) -> str:
    """Return *message* clipped to *max_length* characters.

    A trailing ellipsis is appended when truncation occurred. The
    function is intentionally narrow: it does not strip newlines,
    redact paths, or alter content other than clipping by char count.
    """
    if max_length <= 0:
        return ""
    if len(message) <= max_length:
        return message
    # Reserve 1 char for the ellipsis.
    return message[: max_length - 1] + "\u2026"


def format_exception_chain(
    exc: BaseException,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
) -> str:
    """Render an exception chain as a stable multi-line string.

    The first line is ``Error: <Type>: <message>``. Each subsequent
    line is ``Caused by: <Type>: <message>`` for the chain rooted at
    ``exc.__cause__`` (explicit ``raise X from Y``) or, when no
    explicit cause is set, ``exc.__context__`` (implicit, set when an
    exception is raised during another exception's ``except`` block).

    The renderer is:

    * **Cycle-safe**: visited exception objects are tracked by
      identity; cycles are stopped at the first repeat and rendered
      as ``Caused by: <Type>: <circular>``.
    * **Bounded**: at most *max_depth* lines are emitted. If the
      chain is longer, the final line is ``Caused by: <truncated>``.
    * **Stable**: ordering is depth-first along the *cause* edge,
      falling back to the *context* edge when *cause* is ``None``.
    * **Safe**: no traceback frames, file paths, line numbers, or
      local variable bindings are emitted. Only the exception type
      name and a length-clipped message are written.

    Parameters
    ----------
    exc
        The root exception to render.
    max_depth
        Maximum number of chain entries to emit (including the root).
    max_message_length
        Maximum length of any single message; longer messages are
        clipped with an ellipsis.

    Returns
    -------
    str
        The rendered chain. Always contains at least one line.
    """
    if max_depth <= 0:
        return "Error: <truncated>"

    lines: list[str] = []
    visited: set[int] = set()
    current: BaseException | None = exc

    # We always reserve one slot for a possible truncation marker
    # so the total line count is bounded by max_depth: at most
    # max_depth - 1 real entries plus one ``Caused by: <truncated>``
    # when the chain is longer than the cap. When the chain fits,
    # no truncation marker is emitted.
    reserve_truncation = True

    for step in range(max_depth):
        # If we are on the final iteration and a truncation marker
        # is still reserved, claim that slot now: peek the next link
        # in the chain. If a non-visited link exists, emit a
        # truncation marker as the last line and stop.
        if reserve_truncation and step == max_depth - 1 and current is not None:
            nxt_peek: BaseException | None = current.__cause__
            if nxt_peek is None and not current.__suppress_context__:
                nxt_peek = current.__context__
            if nxt_peek is not None and id(nxt_peek) not in visited:
                lines.append("Caused by: <truncated>")
                current = None
                break
            # No further cause; render normally and do not truncate.
            reserve_truncation = False

        if current is None:
            break
        ident = id(current)
        if ident in visited:
            lines.append("Caused by: <circular>")
            break
        visited.add(ident)

        type_name = type(current).__name__
        message = _truncate(str(current), max_message_length)
        if not lines:
            lines.append(f"Error: {type_name}: {message}")
        else:
            lines.append(f"Caused by: {type_name}: {message}")

        # Prefer explicit __cause__ (raise X from Y); fall back to
        # __context__ (implicit, raised-inside-except) only when the
        # cause was suppressed (__suppress_context__ is False by
        # default).
        nxt: BaseException | None = current.__cause__
        if nxt is None and not current.__suppress_context__:
            nxt = current.__context__
        current = nxt

    return "\n".join(lines)


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_MESSAGE_LENGTH",
    "format_exception_chain",
]
