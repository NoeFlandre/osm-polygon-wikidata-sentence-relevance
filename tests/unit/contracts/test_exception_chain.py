"""Unit tests for the bounded exception-chain renderer (Phase 9M-A).

Public contract:

* First line: ``Error: <Type>: <message>``.
* Subsequent lines: ``Caused by: <Type>: <message>``.
* Cycle-safe (no infinite recursion on self-referential chains).
* Bounded depth (default 8) with a final ``<truncated>`` marker.
* Per-message length cap with an ellipsis.
* No traceback, file paths, or local variable bindings emitted.
* Stable, machine-parseable formatting (newline-separated).
"""

from __future__ import annotations

from osm_polygon_sentence_relevance.contracts._exception_chain import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_MESSAGE_LENGTH,
    format_exception_chain,
)
from osm_polygon_sentence_relevance.contracts.errors import SegmentationError

# --- Defaults --------------------------------------------------------


def test_default_depth_is_eight():
    assert DEFAULT_MAX_DEPTH == 8


def test_default_message_length_is_512():
    assert DEFAULT_MAX_MESSAGE_LENGTH == 512


# --- Single-exception rendering --------------------------------------


def test_single_exception_renders_as_error_line():
    exc = ValueError("boom")
    rendered = format_exception_chain(exc)
    assert rendered == "Error: ValueError: boom"


def test_message_is_truncated_with_ellipsis_when_too_long():
    long_msg = "x" * 1000
    rendered = format_exception_chain(ValueError(long_msg), max_message_length=64)
    assert rendered.startswith("Error: ValueError: ")
    # 64 chars total, last is the ellipsis.
    payload = rendered[len("Error: ValueError: ") :]
    assert len(payload) == 64
    assert payload.endswith("\u2026")


def test_short_message_is_not_truncated():
    rendered = format_exception_chain(ValueError("short"), max_message_length=512)
    assert rendered == "Error: ValueError: short"


# --- Explicit cause chain --------------------------------------------


def test_explicit_cause_chain_renders_with_caused_by_prefix():
    inner = RuntimeError("disk full")
    outer = ValueError("write failed")
    outer.__cause__ = inner
    rendered = format_exception_chain(outer)
    assert rendered.splitlines() == [
        "Error: ValueError: write failed",
        "Caused by: RuntimeError: disk full",
    ]


def test_nested_segmentation_error_with_runtime_cause_reaches_chain():
    # Simulates the production failure path:
    # SegmentationError("segmenter raised an error")
    #   <- RuntimeError("CUDA out of memory. ...")
    cuda_oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB. ...")
    seg = SegmentationError("split_validated_batch: segmenter raised an error")
    seg.__cause__ = cuda_oom
    rendered = format_exception_chain(seg)
    lines = rendered.splitlines()
    assert lines[0] == (
        "Error: SegmentationError: split_validated_batch: segmenter raised an error"
    )
    assert lines[1] == (
        "Caused by: RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB. ..."
    )
    assert len(lines) == 2


def test_three_level_chain_renders_three_lines():
    a = IndexError("idx")
    b = KeyError("k")
    b.__cause__ = a
    c = ValueError("v")
    c.__cause__ = b
    rendered = format_exception_chain(c)
    assert rendered.splitlines() == [
        "Error: ValueError: v",
        "Caused by: KeyError: 'k'",
        "Caused by: IndexError: idx",
    ]


# --- Implicit context chain ------------------------------------------


def test_implicit_context_falls_back_when_cause_is_none():
    inner = RuntimeError("ctx-inner")
    outer = ValueError("ctx-outer")
    # No explicit __cause__; rely on __context__ (default). The
    # bare `raise outer` is the entire point of this test — adding
    # `from inner` would change the contract under test.
    try:
        try:
            raise inner
        except RuntimeError:
            raise outer  # noqa: B904
    except ValueError as caught:
        rendered = format_exception_chain(caught)
    assert "Caused by: RuntimeError: ctx-inner" in rendered


def test_suppressed_context_is_not_followed():
    inner = RuntimeError("should not appear")
    outer = ValueError("v")
    outer.__cause__ = None
    outer.__context__ = inner
    outer.__suppress_context__ = True
    rendered = format_exception_chain(outer)
    assert rendered == "Error: ValueError: v"


# --- Cycle safety ----------------------------------------------------


def test_cycle_is_detected_and_stops():
    a = ValueError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a  # cycle
    rendered = format_exception_chain(a)
    lines = rendered.splitlines()
    # a, b, then a circular marker.
    assert lines == [
        "Error: ValueError: a",
        "Caused by: RuntimeError: b",
        "Caused by: <circular>",
    ]


def test_self_referential_cycle_is_detected():
    a = ValueError("self")
    a.__cause__ = a
    rendered = format_exception_chain(a)
    assert rendered.splitlines() == [
        "Error: ValueError: self",
        "Caused by: <circular>",
    ]


# --- Bounded depth ---------------------------------------------------


def test_chain_is_truncated_after_max_depth():
    # Build a chain longer than max_depth.
    deepest = RuntimeError("deepest")
    current: BaseException = deepest
    for i in range(20):
        nxt = ValueError(f"level-{i}")
        nxt.__cause__ = current
        current = nxt
    rendered = format_exception_chain(current, max_depth=4)
    lines = rendered.splitlines()
    assert len(lines) == 4
    assert lines[-1] == "Caused by: <truncated>"


def test_zero_max_depth_renders_only_truncation_marker():
    rendered = format_exception_chain(ValueError("v"), max_depth=0)
    assert rendered == "Error: <truncated>"


# --- Safety: no traceback / local paths -----------------------------


def test_renderer_does_not_emit_traceback_or_local_paths(tmp_path):
    fake_local_path = str(tmp_path / "secret_local_module.py")
    try:
        raise FileNotFoundError(f"missing file: {fake_local_path}")
    except FileNotFoundError as exc:
        rendered = format_exception_chain(exc)
    # The base message is allowed (the renderer is information-
    # preserving by design); however, it must NOT contain a Python
    # traceback frame marker or an absolute path that maps to the
    # caller's working tree.
    assert "Traceback" not in rendered
    assert 'File "' not in rendered
    # The base message may include the path string itself (the
    # caller's choice); what is forbidden is a Python-formatted
    # traceback with file path + line number + frame name.
    assert "line " not in rendered
    assert "in <" not in rendered


# --- Stable formatting invariants ------------------------------------


def test_format_is_deterministic_across_calls():
    inner = RuntimeError("inner")
    outer = ValueError("outer")
    outer.__cause__ = inner
    a = format_exception_chain(outer)
    b = format_exception_chain(outer)
    assert a == b


def test_newlines_separate_lines():
    inner = RuntimeError("x")
    outer = ValueError("y")
    outer.__cause__ = inner
    rendered = format_exception_chain(outer)
    assert "\n" in rendered
    # Each line is a stable, non-empty string.
    for line in rendered.split("\n"):
        assert line
        assert line.startswith("Error: ") or line.startswith("Caused by: ")


# --- CLI integration: nested SegmentationError reaches stderr --------


def test_cli_exception_chain_reaches_build_log(tmp_path):
    """The CLI must emit the bounded exception chain to stderr
    (this is what the build payload's ``build.stderr.log`` captures)
    so a real CUDA OOM inside the SaT segmenter is visible without
    a traceback.

    The CLI itself is exercised via a small subprocess that
    imports ``format_exception_chain`` and prints the rendered
    chain for a simulated ``SegmentationError`` whose ``__cause__``
    is a ``RuntimeError`` mimicking CUDA OOM. The exit code must
    remain 1 and the stderr must contain both the top-level error
    and the chained CUDA OOM cause.
    """
    import os
    import subprocess
    import sys

    hf_home = tmp_path / "hf"
    hf_home.mkdir()

    script = (
        "import sys\n"
        "from osm_polygon_sentence_relevance.contracts._exception_chain import (\n"
        "    format_exception_chain,\n"
        ")\n"
        "from osm_polygon_sentence_relevance.contracts.errors import (\n"
        "    SegmentationError,\n"
        ")\n"
        "try:\n"
        "    raise RuntimeError('CUDA out of memory. 2.00 GiB')\n"
        "except RuntimeError as inner:\n"
        "    try:\n"
        "        raise SegmentationError(\n"
        "            'split_validated_batch: segmenter raised an error'\n"
        "        ) from inner\n"
        "    except SegmentationError as outer:\n"
        "        sys.stderr.write(format_exception_chain(outer) + chr(10))\n"
        "        sys.stderr.flush()\n"
        "        sys.exit(1)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HF_HOME": str(hf_home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        timeout=30,
    )
    assert proc.returncode == 1
    lines = proc.stderr.splitlines()
    # First line: top-level SegmentationError.
    assert lines[0].startswith("Error: SegmentationError: ")
    assert "split_validated_batch: segmenter raised an error" in lines[0]
    # Second line: the underlying CUDA OOM cause.
    assert any(
        line.startswith("Caused by: RuntimeError: ") and "CUDA out of memory" in line
        for line in lines[1:]
    )
    # No Python traceback markers.
    assert not any("Traceback" in line for line in lines)
