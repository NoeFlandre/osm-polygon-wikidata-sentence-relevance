"""Unit tests for the CUDA llama-server build and recovery lifecycle."""

from __future__ import annotations

from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest

from osm_polygon_sentence_relevance.operator import llama_server
from osm_polygon_sentence_relevance.operator.oar import JobState, JobStatus
from osm_polygon_sentence_relevance.operator.state import RunPhase
from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout


class _FakeStore:
    def __init__(
        self,
        phase: RunPhase = RunPhase.REMOTE_PREPARED,
        facts: dict[str, Any] | None = None,
    ) -> None:
        self.value = SimpleNamespace(phase=phase, facts=facts or {})
        self.transitions: list[dict[str, Any]] = []

    def load(self) -> SimpleNamespace:
        return self.value

    def transition(
        self,
        *,
        expected: RunPhase,
        target: RunPhase,
        facts: dict[str, object],
    ) -> SimpleNamespace:
        assert self.value.phase is expected
        self.transitions.append(
            {"expected": expected, "target": target, "facts": dict(facts)}
        )
        self.value = SimpleNamespace(phase=target, facts={**self.value.facts, **facts})
        return self.value


class _FakeSsh:
    def __init__(self, output: str = "yes") -> None:
        self.output = output
        self.commands: list[str] = []

    def run(self, command: str) -> SimpleNamespace:
        self.commands.append(command)
        if "llama-server" in command and "printf yes" in command:
            return SimpleNamespace(stdout=self.output)
        return SimpleNamespace(stdout="")


class _FakeOar:
    def __init__(
        self, statuses: list[JobStatus] | None = None, next_job_id: int = 99
    ) -> None:
        self.statuses = list(statuses) if statuses else []
        self.next_job_id = next_job_id
        self.submitted_requests: list[Any] = []

    def submit(self, request: Any) -> int:
        self.submitted_requests.append(request)
        return self.next_job_id

    def status(self, job_id: int) -> JobStatus:
        if self.statuses:
            return self.statuses.pop(0)
        return JobStatus(job_id, JobState.TERMINATED, exit_code=0)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("yes", True),
        ("yes\n", False),
        (" yes", False),
        ("yes ", False),
        ("no", False),
        ("", False),
    ],
)
def test_llama_server_ready_requires_exact_yes_match(
    output: str, expected: bool
) -> None:
    ssh = _FakeSsh(output=output)
    layout = RemoteLayout(PurePosixPath("/r"))
    assert llama_server.llama_server_ready(ssh, layout) is expected  # type: ignore[arg-type]


def test_llama_server_ready_with_text_attribute() -> None:
    class ResultWithText:
        text = "yes"

    class SshWithText:
        def run(self, _cmd: str) -> ResultWithText:
            return ResultWithText()

    layout = RemoteLayout(PurePosixPath("/r"))
    assert llama_server.llama_server_ready(SshWithText(), layout) is True  # type: ignore[arg-type]


def test_ensure_llama_server_delegates_to_monitor_job_with_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _FakeStore(phase=RunPhase.REMOTE_PREPARED)
    ssh = _FakeSsh(output="yes")
    oar = _FakeOar(next_job_id=99)
    layout = RemoteLayout(PurePosixPath("/r"))

    calls: list[dict[str, object]] = []

    def mock_monitor(
        ssh_arg: object,
        oar_arg: object,
        layout_arg: object,
        job_id_arg: int,
        log_name_arg: str,
        poll_seconds_arg: float,
        *,
        sleeper: object | None = None,
    ) -> None:
        calls.append(
            {
                "ssh": ssh_arg,
                "oar": oar_arg,
                "layout": layout_arg,
                "job_id": job_id_arg,
                "log_name": log_name_arg,
                "poll_seconds": poll_seconds_arg,
                "sleeper": sleeper,
            }
        )

    monkeypatch.setattr(llama_server, "monitor_job_with_log", mock_monitor)

    def dummy_sleeper(_sec: float) -> None:
        pass

    job_id = llama_server.ensure_llama_server(
        ssh,  # type: ignore[arg-type]
        oar,  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        layout,
        5.0,
        sleeper=dummy_sleeper,
    )

    assert job_id == 99
    assert len(calls) == 1
    assert calls[0]["ssh"] is ssh
    assert calls[0]["oar"] is oar
    assert calls[0]["layout"] is layout
    assert calls[0]["job_id"] == 99
    assert calls[0]["log_name"] == "build.stdout.log"
    assert calls[0]["poll_seconds"] == 5.0
    assert calls[0]["sleeper"] is dummy_sleeper


@pytest.mark.parametrize(
    "state_name", [JobState.QUEUED, JobState.RUNNING, JobState.FINISHING]
)
def test_ensure_llama_server_reattaches_to_durable_job(
    state_name: JobState,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _FakeStore(facts={"llama_build_job_id": 42})
    ssh = _FakeSsh(output="yes")
    oar = _FakeOar(statuses=[JobStatus(42, state_name)])
    layout = RemoteLayout(PurePosixPath("/r"))

    monkeypatch.setattr(llama_server, "monitor_job_with_log", lambda *_a, **_kw: None)

    job_id = llama_server.ensure_llama_server(
        ssh,  # type: ignore[arg-type]
        oar,  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        layout,
        1.0,
        sleeper=lambda _: None,
    )

    assert job_id == 42
    assert len(oar.submitted_requests) == 0
    assert capsys.readouterr().out == "Reattaching to CUDA llama-server build job 42\n"


def test_ensure_llama_server_reuses_completed_job_with_binary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _FakeStore(facts={"llama_build_job_id": 42})
    ssh = _FakeSsh(output="yes")
    oar = _FakeOar(statuses=[JobStatus(42, JobState.TERMINATED, exit_code=0)])
    layout = RemoteLayout(PurePosixPath("/r"))

    job_id = llama_server.ensure_llama_server(
        ssh,  # type: ignore[arg-type]
        oar,  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        layout,
        1.0,
        sleeper=lambda _: None,
    )

    assert job_id == 42
    assert len(oar.submitted_requests) == 0
    assert (
        capsys.readouterr().out == "Reusing completed CUDA llama-server build job 42\n"
    )


@pytest.mark.parametrize(
    ("status", "initial_output"),
    [
        (JobStatus(41, JobState.ERROR, exit_code=1), "yes"),
        (JobStatus(41, JobState.TERMINATED, exit_code=1), "yes"),
        (JobStatus(41, JobState.TERMINATED, exit_code=0), "no"),
    ],
)
def test_ensure_llama_server_replaces_failed_or_incomplete_job(
    status: JobStatus,
    initial_output: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _FakeStore(facts={"llama_build_job_id": 41})
    ssh = _FakeSsh(output=initial_output)
    oar = _FakeOar(statuses=[status], next_job_id=42)
    layout = RemoteLayout(PurePosixPath("/r"))

    def mock_monitor(*_a: Any, **_kw: Any) -> None:
        ssh.output = "yes"

    monkeypatch.setattr(llama_server, "monitor_job_with_log", mock_monitor)

    job_id = llama_server.ensure_llama_server(
        ssh,  # type: ignore[arg-type]
        oar,  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        layout,
        1.0,
        sleeper=lambda _: None,
    )

    assert job_id == 42
    assert len(oar.submitted_requests) == 1
    assert state.value.facts["llama_build_job_id"] == 42
    assert capsys.readouterr().out == (
        "[operator] CUDA llama-server binary is absent; submitting its build\n"
        "Submitted CUDA llama-server build job 42\n"
    )


def test_ensure_llama_server_persists_job_id_before_monitoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _FakeStore(phase=RunPhase.REMOTE_PREPARED)
    ssh = _FakeSsh(output="yes")
    oar = _FakeOar(next_job_id=88)
    layout = RemoteLayout(PurePosixPath("/r"))

    persisted_during_monitor: int | None = None

    def mock_monitor(*_a: Any, **_kw: Any) -> None:
        nonlocal persisted_during_monitor
        persisted_during_monitor = state.value.facts.get("llama_build_job_id")  # type: ignore[assignment]

    monkeypatch.setattr(llama_server, "monitor_job_with_log", mock_monitor)

    job_id = llama_server.ensure_llama_server(
        ssh,  # type: ignore[arg-type]
        oar,  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        layout,
        1.0,
        sleeper=lambda _: None,
    )

    assert job_id == 88
    assert persisted_during_monitor == 88


def test_ensure_llama_server_incorrect_phase_prevents_submission_and_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid durable phase must raise before calling oar.submit() or mutating state."""

    state = _FakeStore(phase=RunPhase.CREATED)
    ssh = _FakeSsh(output="no")
    oar = _FakeOar(next_job_id=77)
    layout = RemoteLayout(PurePosixPath("/r"))

    monitor_calls: list[Any] = []
    monkeypatch.setattr(
        llama_server,
        "monitor_job_with_log",
        lambda *_a, **_kw: monitor_calls.append(_a),
    )

    with pytest.raises(
        RuntimeError, match="CUDA build submission has invalid durable phase"
    ):
        llama_server.ensure_llama_server(
            ssh,  # type: ignore[arg-type]
            oar,  # type: ignore[arg-type]
            state,  # type: ignore[arg-type]
            layout,
            1.0,
            sleeper=lambda _: None,
        )

    assert len(oar.submitted_requests) == 0
    assert len(state.transitions) == 0
    assert len(monitor_calls) == 0
    assert "llama_build_job_id" not in state.value.facts


def test_ensure_llama_server_keyboard_interrupt_leaves_durable_job_id_recoverable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _FakeStore(phase=RunPhase.REMOTE_PREPARED)
    ssh = _FakeSsh(output="yes")
    oar = _FakeOar(next_job_id=42)
    layout = RemoteLayout(PurePosixPath("/r"))

    def interrupt_monitor(*_a: Any, **_kw: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(llama_server, "monitor_job_with_log", interrupt_monitor)

    with pytest.raises(KeyboardInterrupt):
        llama_server.ensure_llama_server(
            ssh,  # type: ignore[arg-type]
            oar,  # type: ignore[arg-type]
            state,  # type: ignore[arg-type]
            layout,
            30.0,
            sleeper=lambda _: None,
        )

    assert state.value.facts["llama_build_job_id"] == 42


def test_ensure_llama_server_missing_binary_after_build_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _FakeStore(phase=RunPhase.REMOTE_PREPARED)
    ssh = _FakeSsh(output="no")
    oar = _FakeOar(next_job_id=99)
    layout = RemoteLayout(PurePosixPath("/r"))

    monkeypatch.setattr(llama_server, "monitor_job_with_log", lambda *_a, **_kw: None)

    with pytest.raises(
        RuntimeError, match="CUDA llama-server build did not produce a binary"
    ):
        llama_server.ensure_llama_server(
            ssh,  # type: ignore[arg-type]
            oar,  # type: ignore[arg-type]
            state,  # type: ignore[arg-type]
            layout,
            1.0,
            sleeper=lambda _: None,
        )


def test_ensure_llama_server_sleeper_forwarded_without_real_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: custom sleeper is passed through to monitor_job_with_log."""
    state = _FakeStore(phase=RunPhase.REMOTE_PREPARED)
    ssh = _FakeSsh(output="yes")
    oar = _FakeOar(next_job_id=99)
    layout = RemoteLayout(PurePosixPath("/r"))

    received_sleeper: Any = None

    def mock_monitor(*_a: Any, sleeper: Any = None, **_kw: Any) -> None:
        nonlocal received_sleeper
        received_sleeper = sleeper

    monkeypatch.setattr(llama_server, "monitor_job_with_log", mock_monitor)

    def custom_sleeper(_sec: float) -> None:
        pass

    llama_server.ensure_llama_server(
        ssh,  # type: ignore[arg-type]
        oar,  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        layout,
        1.0,
        sleeper=custom_sleeper,
    )

    assert received_sleeper is custom_sleeper
