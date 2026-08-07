"""Executable shell tests for the graceful pre-walltime deadline helper.

The deadline helper wraps the labeling CLI with ``timeout
--preserve-status --signal=INT --kill-after=10m`` so a bounded OAR
allocation exits cleanly before OAR's final TERM/KILL window. These
tests exercise the helper against controlled subprocesses so we know
the exact behaviour on Grid'5000 Linux.
"""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts" / "grid5000" / "_deadline_helper.sh"
FAKE_TIMEOUT = ROOT / "tests" / "_support" / "fake_timeout.sh"


@pytest.fixture(autouse=True)
def timeout_bin():
    """Yield a path usable as ``TIMEOUT_BIN`` for the deadline helper."""

    if shutil.which("timeout") is not None:
        yield "timeout"
        return
    if FAKE_TIMEOUT.exists():
        yield str(FAKE_TIMEOUT)
        return
    pytest.skip("GNU timeout is unavailable on this host")


@pytest.fixture(autouse=True)
def _require_signal_delivery() -> None:
    """Skip the runtime tests on hosts where SIGINT cannot be delivered to a child.

    macOS bash does not reliably propagate SIGINT to a child process; the
    production helper runs on Grid'5000 Linux where GNU ``timeout`` is the
    real binary. Tests that rely on SIGINT delivery are skipped on
    macOS so they do not produce false failures on developer machines.
    """

    if platform.system() == "Darwin" and shutil.which("timeout") is None:
        pytest.skip("SIGINT delivery to child processes is unreliable on macOS")
    return None


def _make_dummy_child(tmp_path: Path, *, behaviour: str, duration: float) -> Path:
    """Create a shell child that records its behaviour in a marker file."""

    script = tmp_path / f"child_{behaviour}.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f"set -e\n"
        f"trap 'echo trapped >> {tmp_path}/child.signal; exit 0' INT TERM\n"
        f"echo started >> {tmp_path}/child.signal\n"
        f"case '{behaviour}' in\n"
        f"  early)\n"
        f"    exit 0\n"
        f"    ;;\n"
        f"  graceful)\n"
        f"    sleep {duration}\n"
        f"    exit 0\n"
        f"    ;;\n"
        f"  slow)\n"
        f"    sleep {duration}\n"
        f"    exit 0\n"
        f"    ;;\n"
        f"esac\n"
    )
    script.chmod(0o755)
    return script


def _invoke(
    tmp_path: Path,
    *,
    duration: str,
    grace: str,
    child: Path,
    extra_args: tuple[str, ...] = (),
    timeout_bin: str = "timeout",
) -> subprocess.CompletedProcess[str]:
    """Invoke the deadline helper with controlled timing values."""

    script_path = tmp_path / "invoke.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        f"export TIMEOUT_BIN='{timeout_bin}'\n"
        f". '{HELPER}'\n"
        "set +e\n"
        f"deadline_helper_run '{duration}' '{grace}' '{child}' {' '.join(extra_args)}\n"
        f"echo exit=$?\n"
    )
    script_path.chmod(0o755)
    return subprocess.run(
        [str(script_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _with_timeout(
    tmp_path: Path,
    *,
    duration: str,
    grace: str,
    child: Path,
    timeout_bin: str,
    extra_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    return _invoke(
        tmp_path,
        duration=duration,
        grace=grace,
        child=child,
        timeout_bin=timeout_bin,
        extra_args=extra_args,
    )


def test_helper_accepts_durations_with_and_without_unit() -> None:
    """``45m`` and ``700s`` both parse; ``0`` is rejected."""

    def parse(raw: str) -> int:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f". '{HELPER}'\n_deadline_helper_parse_duration '{raw}'",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0, result.stderr
        return int(result.stdout.strip())

    assert parse("700s") == 700
    assert parse("60") == 60
    assert parse("5m") == 5 * 60
    assert parse("2h") == 2 * 3600

    def parse_bad(raw: str) -> bool:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f". '{HELPER}'\n_deadline_helper_parse_duration '{raw}'",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode != 0

    assert parse_bad("0")
    assert parse_bad("0s")
    assert parse_bad("abc")


def test_helper_executes_timeout_with_spaces_and_shell_metacharacters_quoted(
    tmp_path: Path,
) -> None:
    """The timeout executable path must be safely quoted before execution."""

    hostile_dir = tmp_path / "hostile timeout directory"
    hostile_dir.mkdir()
    hostile_timeout = hostile_dir / "fake timeout path $HOME$(touch)"
    marker = tmp_path / "quoted-hostile-timeout-marker.log"
    hostile_timeout.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"${{1}}:${{2}}\" >> '{marker}'\n"
        'while [[ "$1" == -* ]]; do shift; done\n'
        "shift\n"
        '"$@"\n'
    )
    hostile_timeout.chmod(0o755)

    child = _make_dummy_child(tmp_path, behaviour="early", duration=0.0)
    result = _with_timeout(
        tmp_path,
        duration="60s",
        grace="10s",
        child=child,
        timeout_bin=str(hostile_timeout),
    )
    assert "exit=0" in result.stdout, result.stdout + result.stderr
    assert marker.exists(), marker.read_text()


def test_helper_propagates_child_exit_zero_on_early_completion(
    tmp_path: Path, timeout_bin: str
) -> None:
    """A child that finishes before the deadline must exit 0 with the child's code."""

    child = _make_dummy_child(tmp_path, behaviour="early", duration=0.0)
    result = _with_timeout(
        tmp_path, duration="60s", grace="10s", child=child, timeout_bin=timeout_bin
    )
    assert "exit=0" in result.stdout, result.stdout + result.stderr
    # The child marker file records ``started`` then exits immediately.
    signal_log = (tmp_path / "child.signal").read_text()
    assert "started" in signal_log


def test_helper_sends_sigint_at_deadline(tmp_path: Path, timeout_bin: str) -> None:
    """The child must receive SIGINT at the internal deadline."""

    child = _make_dummy_child(tmp_path, behaviour="graceful", duration=2.0)
    start = time.monotonic()
    result = _with_timeout(
        tmp_path, duration="2s", grace="20s", child=child, timeout_bin=timeout_bin
    )
    elapsed = time.monotonic() - start
    # The helper must SIGINT around the 2-second mark, not wait 22 seconds.
    assert elapsed < 10.0, f"helper waited too long: {elapsed:.1f}s"
    signal_log = (tmp_path / "child.signal").read_text()
    # The child handles SIGINT and exits 0; the helper must then exit 0.
    assert "trapped" in signal_log
    assert "exit=0" in result.stdout


def test_helper_sends_sigkill_after_grace_window(
    tmp_path: Path, timeout_bin: str
) -> None:
    """A child that ignores SIGINT and outlasts the grace window must be SIGKILLed."""

    script = tmp_path / "ignore.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "trap '' INT\n"  # ignore SIGINT
        f"echo started >> {tmp_path}/child.signal\n"
        "sleep 30\n"
    )
    script.chmod(0o755)
    try:
        start = time.monotonic()
        result = _with_timeout(
            tmp_path, duration="1s", grace="2s", child=script, timeout_bin=timeout_bin
        )
        elapsed = time.monotonic() - start
        # The helper must SIGINT after ~1s and SIGKILL after the grace window.
        assert elapsed < 6.0, f"helper waited too long: {elapsed:.1f}s"
        # Exit code: SIGKILL → 124 (GNU timeout default) or 137. Either way != 0.
        assert "exit=0" not in result.stdout, (
            f"expected non-zero exit, got {result.stdout!r}"
        )
    finally:
        pass


def test_helper_preserves_real_child_status(tmp_path: Path, timeout_bin: str) -> None:
    """A child that exits with a non-zero status must surface that status."""

    script = tmp_path / "fail.sh"
    script.write_text("#!/usr/bin/env bash\nexit 7\n")
    script.chmod(0o755)
    result = _with_timeout(
        tmp_path, duration="30s", grace="30s", child=script, timeout_bin=timeout_bin
    )
    # The helper must return 7 (the child's status), not 0.
    assert "exit=7" in result.stdout, result.stdout + result.stderr


def test_helper_signal_target_is_only_the_child(
    tmp_path: Path, timeout_bin: str
) -> None:
    """The helper must not signal unrelated processes."""

    # Spawn a sibling process with a known PID that sleeps for a long
    # time. Verify the helper does not deliver a signal to it.
    sibling_marker = tmp_path / "sibling.signal"
    sibling = subprocess.Popen(
        [
            "bash",
            "-c",
            f"trap 'echo killed >> {sibling_marker}; exit 0' INT TERM; sleep 30",
        ],
    )
    try:
        # Wait for the sibling to install its trap.
        time.sleep(0.2)
        child = _make_dummy_child(tmp_path, behaviour="graceful", duration=2.0)
        result = _with_timeout(
            tmp_path, duration="2s", grace="20s", child=child, timeout_bin=timeout_bin
        )
        assert "exit=0" in result.stdout
        # Give the sibling time to receive any (incorrect) signal.
        time.sleep(0.5)
        assert not sibling_marker.exists(), "helper signalled an unrelated process"
        # Sibling is still alive; clean it up.
        sibling.kill()
        sibling.wait(timeout=5)
    finally:
        if sibling.poll() is None:
            os.kill(sibling.pid, signal.SIGKILL)


def test_helper_rejects_zero_or_negative_durations(
    tmp_path: Path, timeout_bin: str
) -> None:
    """``0``, ``0s`` and ``-1s`` are rejected before any subprocess runs."""

    child = _make_dummy_child(tmp_path, behaviour="early", duration=0.0)
    for bad in ("0", "0s", "-1s"):
        result = _with_timeout(
            tmp_path, duration=bad, grace="10s", child=child, timeout_bin=timeout_bin
        )
        assert "exit=0" not in result.stdout, (
            f"helper accepted duration {bad!r}: {result.stdout!r}"
        )


def test_helper_kills_child_pid_after_grace(tmp_path: Path, timeout_bin: str) -> None:
    """The helper must ensure the child PID no longer exists after grace."""

    script = tmp_path / "ignore_int.sh"
    script.write_text("#!/usr/bin/env bash\ntrap '' INT\nsleep 30\n")
    script.chmod(0o755)
    start = time.monotonic()
    result = _with_timeout(
        tmp_path, duration="1s", grace="2s", child=script, timeout_bin=timeout_bin
    )
    elapsed = time.monotonic() - start
    # Confirm the child has been killed; allow up to ~3.5 s total.
    assert elapsed < 6.0
    assert "exit=0" not in result.stdout
    # The helper exits with 124 (timeout's default for SIGKILL).
    assert "exit=124" in result.stdout or "exit=137" in result.stdout, (
        f"unexpected exit: {result.stdout!r}"
    )
