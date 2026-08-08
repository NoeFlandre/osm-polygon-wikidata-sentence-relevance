"""Tests for the detached operator supervisor."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from osm_polygon_sentence_relevance.operator.supervisor import (
    _CLI_ENTRYPOINT,
    SupervisorLaunch,
    _find_run_id,
    _read_phase,
    _stage,
    build_resume_arguments,
    main,
    session_name_for,
    start_detached_supervisor,
    supervise,
)


def test_session_name_is_stable_and_safe() -> None:
    arguments = ("run", "--scope", "all", "--stage", "all")

    first = session_name_for(arguments)
    second = session_name_for(arguments)

    assert first == second
    assert first.startswith("osm-grid5000-")
    assert first.replace("-", "").isalnum()
    assert len(first) <= 64


def test_session_name_uses_run_id_without_exposing_arguments() -> None:
    assert session_name_for(("resume", "a" * 20), "b" * 20) != session_name_for(
        ("resume", "a" * 20), "c" * 20
    )


def test_build_resume_arguments_rejects_invalid_or_incomplete_options() -> None:
    with pytest.raises(ValueError, match="20 lowercase"):
        build_resume_arguments(("run",), "not-a-run")
    with pytest.raises(ValueError, match="missing value"):
        build_resume_arguments(("run", "--poll-seconds"), "a" * 20)


def test_start_detached_supervisor_uses_tmux_and_persists_log(
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, ...]] = []

    def runner(
        command: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(command))
        if list(command)[1:3] == ["has-session", "-t"]:
            return subprocess.CompletedProcess(command, 1)
        return subprocess.CompletedProcess(command, 0)

    launch = start_detached_supervisor(
        ("run", "--scope", "all", "--stage", "all"),
        data_root=tmp_path,
        tmux_path="/usr/bin/tmux",
        runner=runner,
    )

    assert isinstance(launch, SupervisorLaunch)
    assert launch.backend == "tmux"
    assert launch.log_path.is_file()
    assert launch.log_path.stat().st_mode & 0o777 == 0o600
    assert len(calls) == 2
    new_session = calls[1]
    assert new_session[:4] == ("/usr/bin/tmux", "new-session", "-d", "-s")
    assert ">>" in str(new_session[-1])
    assert str(launch.log_path) in str(new_session[-1])


def test_start_detached_supervisor_validates_command_and_log_symlink(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires a run"):
        start_detached_supervisor(("status", "a" * 20), data_root=tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        start_detached_supervisor(("run", ""), data_root=tmp_path)

    log_dir = tmp_path / "runs" / ("a" * 20)
    log_dir.mkdir(parents=True)
    log_path = log_dir / "supervisor.log"
    log_path.symlink_to(tmp_path / "other.log")
    with pytest.raises(RuntimeError, match="must not be a symlink"):
        start_detached_supervisor(
            ("resume", "a" * 20),
            data_root=tmp_path,
            run_id="a" * 20,
            tmux_path="",
        )


def test_start_detached_supervisor_rejects_existing_tmux_session(
    tmp_path: Path,
) -> None:
    def runner(
        command: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(RuntimeError, match="already active"):
        start_detached_supervisor(
            ("resume", "a" * 20),
            data_root=tmp_path,
            tmux_path="/usr/bin/tmux",
            runner=runner,
        )


def test_start_detached_supervisor_falls_back_to_session_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class FakeProcess:
        pid = 1234

    def popen(command: object, **kwargs: object) -> FakeProcess:
        seen["command"] = command
        seen["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "osm_polygon_sentence_relevance.operator.supervisor.shutil.which",
        lambda _: None,
    )
    launch = start_detached_supervisor(
        ("resume", "a" * 20),
        data_root=tmp_path,
        popen_factory=popen,
    )

    assert launch.backend == "process"
    assert seen["command"]
    kwargs = cast(dict[str, object], seen["kwargs"])
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True
    assert kwargs["stdout"] is not None


def test_start_detached_supervisor_includes_run_specific_identity(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def runner(
        command: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 1)

    launch = start_detached_supervisor(
        ("resume", "a" * 20),
        data_root=tmp_path,
        run_id="a" * 20,
        tmux_path="/usr/bin/tmux",
        runner=runner,
    )

    assert "--run-id" in launch.command
    assert calls[0][0:4] == ("/usr/bin/tmux", "has-session", "-t", launch.session_name)


def test_build_resume_arguments_preserves_safe_runtime_options() -> None:
    arguments = (
        "run",
        "--scope",
        "all",
        "--stage",
        "all",
        "--site",
        "grenoble",
        "--gpu-memory-mb",
        "40000",
        "--poll-seconds",
        "30",
        "--detach",
    )

    assert build_resume_arguments(arguments, "a" * 20) == (
        "resume",
        "a" * 20,
        "--site",
        "grenoble",
        "--gpu-memory-mb",
        "40000",
        "--poll-seconds",
        "30",
    )


def test_supervise_retries_until_durable_run_is_complete(tmp_path: Path) -> None:
    state_path = tmp_path / "runs" / ("a" * 20) / "state.json"
    state_path.parent.mkdir(parents=True)
    attempts: list[tuple[str, ...]] = []

    def run_process(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        attempts.append(command)
        phase = "remote_prepared" if len(attempts) == 1 else "complete"
        state_path.write_text(json.dumps({"phase": phase}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 1 if len(attempts) == 1 else 0)

    result = supervise(
        ("run", "--scope", "all", "--stage", "all"),
        data_root=tmp_path,
        log_path=tmp_path / "supervisor.log",
        run_id="a" * 20,
        run_process=run_process,
        sleep=lambda _: None,
    )

    assert result == 0
    assert attempts == [
        ("run", "--scope", "all", "--stage", "all"),
        ("resume", "a" * 20),
    ]


def test_supervise_keeps_executable_when_rebuilding_child_resume_command(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "runs" / ("a" * 20) / "state.json"
    state_path.parent.mkdir(parents=True)
    prefix = ("python", "-c", _CLI_ENTRYPOINT)
    attempts: list[tuple[str, ...]] = []

    def run_process(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        attempts.append(command)
        phase = "remote_prepared" if len(attempts) == 1 else "complete"
        state_path.write_text(json.dumps({"phase": phase}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 1 if len(attempts) == 1 else 0)

    assert (
        supervise(
            (*prefix, "run", "--scope", "all", "--stage", "all"),
            data_root=tmp_path,
            log_path=tmp_path / "supervisor.log",
            run_id="a" * 20,
            run_process=run_process,
            sleep=lambda _: None,
        )
        == 0
    )
    assert attempts[1] == (*prefix, "resume", "a" * 20)


def test_supervise_stops_after_split_checkpoint(tmp_path: Path) -> None:
    state_path = tmp_path / "runs" / ("a" * 20) / "state.json"
    state_path.parent.mkdir(parents=True)
    attempts = 0

    def run_process(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        state_path.write_text(json.dumps({"phase": "checkpointed"}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    assert (
        supervise(
            ("run", "--stage", "split"),
            data_root=tmp_path,
            log_path=tmp_path / "supervisor.log",
            run_id="a" * 20,
            run_process=run_process,
            sleep=lambda _: pytest.fail("split run must not sleep"),
        )
        == 0
    )
    assert attempts == 1


def test_supervise_does_not_retry_failed_or_unknown_state(tmp_path: Path) -> None:
    attempts = 0

    def run_process(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        return subprocess.CompletedProcess(command, 1)

    assert (
        supervise(
            ("resume", "a" * 20),
            data_root=tmp_path,
            log_path=tmp_path / "supervisor.log",
            run_id="a" * 20,
            run_process=run_process,
            sleep=lambda _: pytest.fail("unknown state must not sleep"),
        )
        == 1
    )
    assert attempts == 1


def test_supervise_rejects_invalid_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        supervise(
            ("run",), data_root=tmp_path, log_path=tmp_path / "log", max_attempts=0
        )
    with pytest.raises(ValueError, match="retry_seconds"):
        supervise(
            ("run",), data_root=tmp_path, log_path=tmp_path / "log", retry_seconds=-1
        )


def test_supervise_honors_max_attempts(tmp_path: Path) -> None:
    state_path = tmp_path / "runs" / ("a" * 20) / "state.json"
    state_path.parent.mkdir(parents=True)

    def run_process(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        state_path.write_text(
            json.dumps({"phase": "remote_prepared"}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 1)

    assert (
        supervise(
            ("run",),
            data_root=tmp_path,
            log_path=tmp_path / "log",
            run_id="a" * 20,
            max_attempts=1,
            run_process=run_process,
            sleep=lambda _: pytest.fail("max attempts must stop immediately"),
        )
        == 1
    )


def test_supervisor_helpers_handle_missing_and_malformed_evidence(
    tmp_path: Path,
) -> None:
    assert _read_phase(tmp_path, None) is None
    assert _read_phase(tmp_path, "a" * 20) is None
    log_path = tmp_path / "missing.log"
    assert _find_run_id(log_path) is None
    assert _stage(("run", "--scope", "all")) is None

    state_path = tmp_path / "runs" / ("a" * 20) / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("[]", encoding="utf-8")
    assert _read_phase(tmp_path, "a" * 20) is None
    log_path.write_text("Durable run ID: not-valid\nDurable run ID: " + "a" * 20)
    assert _find_run_id(log_path) == "a" * 20


def test_supervisor_main_requires_and_forwards_child_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "osm_polygon_sentence_relevance.operator.supervisor.supervise",
        lambda arguments, **kwargs: seen.append((arguments, kwargs)) or 7,
    )
    assert (
        main(
            [
                "--data-root",
                str(tmp_path),
                "--log-path",
                str(tmp_path / "log"),
                "--",
                "python",
                "-c",
                "pass",
            ]
        )
        == 7
    )
    assert seen[0][0] == ("python", "-c", "pass")


def test_supervisor_main_requires_a_child_command(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--data-root",
                str(tmp_path),
                "--log-path",
                str(tmp_path / "log"),
            ]
        )
