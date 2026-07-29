"""Unit tests for stateless auxiliary-job monitoring."""

from __future__ import annotations

from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator.job_monitor import (
    _default_emit_line,
    monitor_job_with_log,
    report_job_status,
)
from osm_polygon_sentence_relevance.operator.oar import (
    JobState,
    JobStatus,
    format_job_status,
)
from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout


class _FakeOar:
    def __init__(self, statuses: list[JobStatus]) -> None:
        self._statuses = list(statuses)
        self.calls: list[int] = []

    def status(self, job_id: int) -> JobStatus:
        self.calls.append(job_id)
        if len(self._statuses) > 1:
            return self._statuses.pop(0)
        return self._statuses[0]


class _FakeSsh:
    def __init__(self, chunks: list[SimpleNamespace] | None = None) -> None:
        self._chunks = list(chunks) if chunks is not None else []
        self.read_calls: list[tuple[str, int]] = []

    def read_since(self, path: str, offset: int) -> SimpleNamespace:
        self.read_calls.append((path, offset))
        if self._chunks:
            return self._chunks.pop(0)
        return SimpleNamespace(reset=False, text="", next_offset=offset)


def test_default_emit_line_prints_with_flush(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_default_emit_line writes to stdout."""

    _default_emit_line("hello world")
    captured = capsys.readouterr()
    assert captured.out == "hello world\n"


def test_report_job_status_emits_first_observed_status() -> None:
    """The first status observation is emitted to the line handler."""

    emitted: list[str] = []
    status = JobStatus(
        job_id=42,
        state=JobState.RUNNING,
        node="graphene-1",
        scheduled_start="10:00:00",
        walltime_seconds=3600,
        exit_code=None,
    )
    key = report_job_status(status, None, emitted.append)
    assert len(emitted) == 1
    assert emitted[0] == f"[job 42] {format_job_status(status)}"
    assert key == (JobState.RUNNING, "graphene-1", "10:00:00", 3600, None)


def test_report_job_status_suppresses_identical_status() -> None:
    """Identical consecutive status observations do not produce output."""

    emitted: list[str] = []
    status = JobStatus(
        job_id=42,
        state=JobState.RUNNING,
        node="graphene-1",
        scheduled_start="10:00:00",
        walltime_seconds=3600,
        exit_code=None,
    )
    key1 = report_job_status(status, None, emitted.append)
    key2 = report_job_status(status, key1, emitted.append)
    assert len(emitted) == 1
    assert key1 == key2


@pytest.mark.parametrize(
    ("attr", "new_val"),
    [
        ("node", "graphene-2"),
        ("scheduled_start", "11:00:00"),
        ("walltime_seconds", 7200),
        ("exit_code", 0),
    ],
)
def test_report_job_status_emits_on_field_transitions(
    attr: str, new_val: object
) -> None:
    """Transitions in node, scheduled start, walltime, or exit code trigger emissions."""

    emitted: list[str] = []
    base_kwargs: dict[str, object] = {
        "job_id": 100,
        "state": JobState.RUNNING,
        "node": "node-1",
        "scheduled_start": "00:00:00",
        "walltime_seconds": 3600,
        "exit_code": None,
    }
    status1 = JobStatus(**base_kwargs)  # type: ignore[arg-type]
    key1 = report_job_status(status1, None, emitted.append)

    updated_kwargs = dict(base_kwargs)
    updated_kwargs[attr] = new_val
    status2 = JobStatus(**updated_kwargs)  # type: ignore[arg-type]
    key2 = report_job_status(status2, key1, emitted.append)

    assert len(emitted) == 2
    assert key1 != key2


def test_monitor_job_with_log_polls_advances_offset_and_streams_multiline_output() -> (
    None
):
    """Log-enabled monitoring polls, advances byte offsets, handles reset, and streams lines."""

    statuses = [
        JobStatus(job_id=1, state=JobState.QUEUED, walltime_seconds=3600),
        JobStatus(
            job_id=1,
            state=JobState.RUNNING,
            node="n1",
            scheduled_start="10:00",
            walltime_seconds=3600,
        ),
        JobStatus(
            job_id=1,
            state=JobState.TERMINATED,
            node="n1",
            scheduled_start="10:00",
            walltime_seconds=3600,
            exit_code=0,
        ),
    ]
    chunks = [
        SimpleNamespace(reset=False, text="", next_offset=0),
        SimpleNamespace(reset=False, text="line1\nline2\n", next_offset=12),
        SimpleNamespace(reset=False, text="line3\n", next_offset=18),
    ]
    oar = _FakeOar(statuses)
    ssh = _FakeSsh(chunks)
    layout = RemoteLayout(root=PurePosixPath("/home/u/work"))
    emitted: list[str] = []
    slept: list[float] = []

    monitor_job_with_log(
        ssh,  # type: ignore[arg-type]
        oar,  # type: ignore[arg-type]
        layout,
        1,
        "build.log",
        5.0,
        emit_line=emitted.append,
        sleeper=slept.append,
    )

    assert slept == [5.0, 5.0]
    assert ssh.read_calls == [
        ("/home/u/work/logs/1/build.log", 0),
        ("/home/u/work/logs/1/build.log", 0),
        ("/home/u/work/logs/1/build.log", 12),
    ]
    assert "[job 1] line1" in emitted
    assert "[job 1] line2" in emitted
    assert "[job 1] line3" in emitted


def test_monitor_job_with_log_resets_offset_on_truncation() -> None:
    """Chunk reset flag forces offset back to zero."""

    statuses = [
        JobStatus(job_id=1, state=JobState.RUNNING, node="n1", walltime_seconds=3600),
        JobStatus(
            job_id=1,
            state=JobState.TERMINATED,
            node="n1",
            walltime_seconds=3600,
            exit_code=0,
        ),
    ]
    chunks = [
        SimpleNamespace(reset=True, text="", next_offset=0),
        SimpleNamespace(reset=False, text="fresh\n", next_offset=6),
    ]
    oar = _FakeOar(statuses)
    ssh = _FakeSsh(chunks)
    layout = RemoteLayout(root=PurePosixPath("/home/u/work"))
    emitted: list[str] = []
    slept: list[float] = []

    monitor_job_with_log(
        ssh,  # type: ignore[arg-type]
        oar,  # type: ignore[arg-type]
        layout,
        1,
        "build.log",
        1.0,
        emit_line=emitted.append,
        sleeper=slept.append,
    )

    assert ssh.read_calls == [
        ("/home/u/work/logs/1/build.log", 0),
        ("/home/u/work/logs/1/build.log", 0),
    ]
    assert "[job 1] fresh" in emitted


def test_monitor_job_with_log_returns_on_terminated_even_if_nonzero_exit() -> None:
    """Log-enabled monitor returns on TERMINATED state regardless of scheduler exit code."""

    statuses = [
        JobStatus(
            job_id=1,
            state=JobState.TERMINATED,
            node="n1",
            walltime_seconds=3600,
            exit_code=130,
        ),
    ]
    oar = _FakeOar(statuses)
    ssh = _FakeSsh()
    layout = RemoteLayout(root=PurePosixPath("/home/u/work"))

    # Should not raise exception because caller validates exit file
    monitor_job_with_log(
        ssh,  # type: ignore[arg-type]
        oar,  # type: ignore[arg-type]
        layout,
        1,
        "build.log",
        1.0,
        emit_line=lambda _: None,
        sleeper=lambda _: None,
    )


@pytest.mark.parametrize("state", [JobState.ERROR, JobState.MISSING])
def test_monitor_job_with_log_raises_on_scheduler_failure(state: JobState) -> None:
    """ERROR or MISSING scheduler state raises RuntimeError('remote allocation failed')."""

    statuses = [JobStatus(job_id=1, state=state)]
    oar = _FakeOar(statuses)
    ssh = _FakeSsh()
    layout = RemoteLayout(root=PurePosixPath("/home/u/work"))

    with pytest.raises(RuntimeError, match="remote allocation failed"):
        monitor_job_with_log(
            ssh,  # type: ignore[arg-type]
            oar,  # type: ignore[arg-type]
            layout,
            1,
            "build.log",
            1.0,
            emit_line=lambda _: None,
            sleeper=lambda _: None,
        )
