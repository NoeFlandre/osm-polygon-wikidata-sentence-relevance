"""Terminal presentation contracts for the autonomous operator."""

from __future__ import annotations

import json
from io import StringIO

import pytest

from osm_polygon_sentence_relevance.operator.console import OperatorConsole


def test_noninteractive_output_is_plain_and_stable() -> None:
    stdout = StringIO()
    stderr = StringIO()
    console = OperatorConsole(stdout=stdout, stderr=stderr, interactive=False)

    console.milestone("Preparing [remote] checkout")
    console.error("failed")
    console.job_lines(42, "[red]literal[/red]\nsecond")

    assert stdout.getvalue() == (
        "[operator] Preparing [remote] checkout\n"
        "[job 42] [red]literal[/red]\n"
        "[job 42] second\n"
    )
    assert stderr.getvalue() == "Error: failed\n"
    assert "\x1b[" not in stdout.getvalue() + stderr.getvalue()


def test_no_color_forces_plain_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    stream = StringIO()
    console = OperatorConsole(stdout=stream, stderr=StringIO(), interactive=True)

    console.milestone("ready")

    assert stream.getvalue() == "[operator] ready\n"
    assert "\x1b[" not in stream.getvalue()


def test_interactive_milestone_and_error_use_rich_styles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    stdout = StringIO()
    stderr = StringIO()
    console = OperatorConsole(stdout=stdout, stderr=stderr, interactive=True)

    console.milestone("ready")
    console.error("failed")

    assert "\x1b[" in stdout.getvalue()
    assert "[operator]" in stdout.getvalue()
    assert "\x1b[" in stderr.getvalue()
    assert "Error:" in stderr.getvalue()


def test_remote_log_markup_is_always_literal() -> None:
    stdout = StringIO()
    console = OperatorConsole(stdout=stdout, stderr=StringIO(), interactive=True)

    console.job_lines(7, "[bold red]remote[/bold red]")

    assert stdout.getvalue() == "[job 7] [bold red]remote[/bold red]\n"


def test_json_is_stable_plain_machine_output() -> None:
    stdout = StringIO()
    console = OperatorConsole(stdout=stdout, stderr=StringIO(), interactive=True)

    console.json({"z": 1, "a": {"nested": True}})

    assert (
        stdout.getvalue()
        == json.dumps(
            {"z": 1, "a": {"nested": True}},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert "\x1b[" not in stdout.getvalue()


def test_plain_can_target_stdout_or_stderr() -> None:
    stdout = StringIO()
    stderr = StringIO()
    console = OperatorConsole(stdout=stdout, stderr=stderr, interactive=False)

    console.plain("normal")
    console.plain("problem", error=True)

    assert stdout.getvalue() == "normal\n"
    assert stderr.getvalue() == "problem\n"


def test_noninteractive_progress_is_silent() -> None:
    stdout = StringIO()
    console = OperatorConsole(stdout=stdout, stderr=StringIO(), interactive=False)

    with console.progress(description="Probing", total=2) as progress:
        progress.advance()
        progress.advance()
    console.plain("done")

    assert stdout.getvalue() == "done\n"


def test_interactive_progress_reaches_total_and_closes_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    stdout = StringIO()
    console = OperatorConsole(stdout=stdout, stderr=StringIO(), interactive=True)

    with console.progress(description="Probing", total=2) as progress:
        progress.advance()
        progress.advance()
    console.plain("done")

    output = stdout.getvalue()
    assert "Probing" in output
    assert "2/2" in output
    assert output.endswith("done\n")


def test_default_streams_are_resolved_when_output_is_emitted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    console = OperatorConsole(interactive=False)

    console.milestone("captured")
    console.error("captured failure")

    captured = capsys.readouterr()
    assert captured.out == "[operator] captured\n"
    assert captured.err == "Error: captured failure\n"
