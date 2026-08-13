"""Detached, resumable supervision for long Grid'5000 workflows.

The normal CLI remains foreground and streams progress to the terminal.  This
module provides the explicit ``--detach`` mode: a small local supervisor runs
the normal command, reads the durable run state after each allocation, and
reattaches with ``resume`` whenever validated checkpoints remain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Final, Literal

from osm_polygon_sentence_relevance.operator.config import DATA_ROOT

RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{20}$")
_SUPERVISOR_MODULE: Final[str] = "osm_polygon_sentence_relevance.operator.supervisor"
_CLI_ENTRYPOINT: Final[str] = (
    "from osm_polygon_sentence_relevance.operator.cli import main; "
    "raise SystemExit(main())"
)
_RESUME_OPTIONS: Final[frozenset[str]] = frozenset(
    {"--site", "--gpu-memory-mb", "--poll-seconds", "--sampling-target"}
)
_RESUMABLE_PHASES: Final[frozenset[str]] = frozenset(
    {
        "remote_prepared",
        "submitted",
        "queued",
        "running",
        "checkpointed",
        "validated",
        "finalizing",
        "publishing",
        "verifying",
    }
)
_PERMANENT_PREFLIGHT_ERROR: Final[str] = "current source checkout must be clean"

RunProcess = Callable[..., subprocess.CompletedProcess[str]]
PopenFactory = Callable[..., object]


@dataclass(frozen=True, slots=True)
class SupervisorLaunch:
    """Evidence returned after a detached supervisor was started."""

    session_name: str
    log_path: Path
    command: tuple[str, ...]
    backend: Literal["tmux", "process"]


def _valid_run_id(run_id: str | None) -> bool:
    return run_id is not None and RUN_ID_PATTERN.fullmatch(run_id) is not None


def session_name_for(arguments: Sequence[str], run_id: str | None = None) -> str:
    """Build a deterministic, shell-safe session name."""

    basis = (
        run_id if run_id is not None and _valid_run_id(run_id) else "\0".join(arguments)
    )
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
    return f"osm-grid5000-{digest}"


def build_resume_arguments(arguments: Sequence[str], run_id: str) -> tuple[str, ...]:
    """Convert a foreground run invocation into a safe resume invocation."""

    if not _valid_run_id(run_id):
        raise ValueError("run_id must be exactly 20 lowercase hexadecimal characters")
    preserved: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == "--detach":
            index += 1
            continue
        if token in _RESUME_OPTIONS:
            if index + 1 >= len(arguments):
                raise ValueError(f"missing value for {token}")
            preserved.extend((token, arguments[index + 1]))
            index += 2
            continue
        index += 1
    return ("resume", run_id, *preserved)


def _log_path(data_root: Path, session_name: str, run_id: str | None) -> Path:
    if run_id is not None and _valid_run_id(run_id):
        return data_root / "runs" / run_id / "supervisor.log"
    return data_root / "supervisors" / f"{session_name}.log"


def _prepare_log(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    if path.is_symlink():
        raise RuntimeError("supervisor log must not be a symlink")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)


def _child_command(arguments: Sequence[str], python_executable: str) -> tuple[str, ...]:
    return (python_executable, "-c", _CLI_ENTRYPOINT, *arguments)


def _child_resume_command(command: Sequence[str], run_id: str) -> tuple[str, ...]:
    """Build a resume child command without dropping its executable prefix."""

    try:
        entrypoint_index = command.index(_CLI_ENTRYPOINT)
    except ValueError:
        return build_resume_arguments(command, run_id)
    prefix = tuple(command[: entrypoint_index + 1])
    cli_arguments = command[entrypoint_index + 1 :]
    return (*prefix, *build_resume_arguments(cli_arguments, run_id))


def _supervisor_command(
    arguments: Sequence[str],
    *,
    data_root: Path,
    log_path: Path,
    run_id: str | None,
    python_executable: str,
) -> tuple[str, ...]:
    child = _child_command(arguments, python_executable)
    prefix = (
        python_executable,
        "-m",
        _SUPERVISOR_MODULE,
        "--data-root",
        str(data_root),
        "--log-path",
        str(log_path),
    )
    if run_id is not None and _valid_run_id(run_id):
        prefix += ("--run-id", run_id)
    return (*prefix, "--", *child)


def start_detached_supervisor(
    arguments: Sequence[str],
    *,
    data_root: Path = DATA_ROOT,
    run_id: str | None = None,
    tmux_path: str | None = None,
    python_executable: str = sys.executable,
    runner: RunProcess = subprocess.run,
    popen_factory: PopenFactory = subprocess.Popen,
) -> SupervisorLaunch:
    """Start one detached supervisor, refusing duplicate sessions."""

    if not arguments or arguments[0] not in {"run", "resume"}:
        raise ValueError("detached supervisor requires a run or resume command")
    if any(not isinstance(argument, str) or not argument for argument in arguments):
        raise ValueError("detached command arguments must be non-empty strings")

    session_name = session_name_for(arguments, run_id)
    log_path = _log_path(data_root, session_name, run_id)
    _prepare_log(log_path)
    command = _supervisor_command(
        arguments,
        data_root=data_root,
        log_path=log_path,
        run_id=run_id,
        python_executable=python_executable,
    )
    resolved_tmux = tmux_path if tmux_path is not None else shutil.which("tmux")
    if resolved_tmux:
        probe = runner(
            [resolved_tmux, "has-session", "-t", session_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if probe.returncode == 0:
            raise RuntimeError(
                f"detached supervisor session already active: {session_name}"
            )
        shell_command = f"{shlex.join(command)} >> {shlex.quote(str(log_path))} 2>&1"
        runner(
            [resolved_tmux, "new-session", "-d", "-s", session_name, shell_command],
            check=True,
            text=True,
        )
        return SupervisorLaunch(session_name, log_path, command, "tmux")

    with log_path.open("ab") as log_handle:
        popen_factory(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    return SupervisorLaunch(session_name, log_path, command, "process")


def _read_phase(data_root: Path, run_id: str | None) -> str | None:
    if run_id is None or not _valid_run_id(run_id):
        return None
    state_path = data_root / "runs" / run_id / "state.json"
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    phase = payload.get("phase") if isinstance(payload, dict) else None
    return phase if isinstance(phase, str) else None


def _find_run_id(log_path: Path) -> str | None:
    try:
        text = log_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    matches = re.findall(r"Durable run ID: ([0-9a-f]{20})", text)
    return matches[-1] if matches else None


def _stage(arguments: Sequence[str]) -> str | None:
    for index, token in enumerate(arguments[:-1]):
        if token == "--stage":
            return arguments[index + 1]
    return None


def _log_contains_since(path: Path, offset: int, needle: str) -> bool:
    try:
        with path.open("rb") as stream:
            stream.seek(offset)
            return needle in stream.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return False


def supervise(
    arguments: Sequence[str],
    *,
    data_root: Path,
    log_path: Path,
    run_id: str | None = None,
    max_attempts: int | None = None,
    retry_seconds: float = 10.0,
    run_process: RunProcess = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run foreground allocations and resume validated partial runs."""

    if max_attempts is not None and max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if retry_seconds < 0:
        raise ValueError("retry_seconds must be non-negative")
    log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    current = tuple(arguments)
    current_run_id = run_id
    stop_after_split = _stage(current) == "split"
    for attempt in count(1):
        log_offset = log_path.stat().st_size if log_path.exists() else 0
        with log_path.open("a", encoding="utf-8") as log_handle:
            print(
                f"[supervisor] allocation attempt {attempt}",
                file=log_handle,
                flush=True,
            )
            result = run_process(
                current,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if _log_contains_since(log_path, log_offset, _PERMANENT_PREFLIGHT_ERROR):
            return result.returncode if result.returncode else 1
        current_run_id = current_run_id or _find_run_id(log_path)
        phase = _read_phase(data_root, current_run_id)
        if phase == "complete" or (stop_after_split and phase == "checkpointed"):
            return result.returncode
        if phase == "failed" or phase not in _RESUMABLE_PHASES:
            return result.returncode if result.returncode else 1
        if current_run_id is None:
            return result.returncode if result.returncode else 1
        if max_attempts is not None and attempt == max_attempts:
            return 1
        current = _child_resume_command(current, current_run_id)
        sleep(retry_seconds)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the internal supervisor protocol."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--retry-seconds", type=float, default=10.0)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("child", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    child = tuple(parsed.child[1:] if parsed.child[:1] == ["--"] else parsed.child)
    if not child:
        parser.error("a child command is required")
    return supervise(
        child,
        data_root=parsed.data_root,
        log_path=parsed.log_path,
        run_id=parsed.run_id,
        max_attempts=parsed.max_attempts,
        retry_seconds=parsed.retry_seconds,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SupervisorLaunch",
    "build_resume_arguments",
    "main",
    "session_name_for",
    "start_detached_supervisor",
    "supervise",
]
