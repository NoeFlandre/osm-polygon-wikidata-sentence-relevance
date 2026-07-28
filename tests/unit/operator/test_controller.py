"""Behaviour tests for the resumable allocation controller."""

from __future__ import annotations

from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator.config import OperatorConfig, Stage
from osm_polygon_sentence_relevance.operator.controller import (
    Controller,
    ControllerError,
)
from osm_polygon_sentence_relevance.operator.oar import JobState, JobStatus
from osm_polygon_sentence_relevance.operator.ssh import LogChunk
from osm_polygon_sentence_relevance.operator.state import RunPhase
from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout


def _config(stage: str = "split") -> OperatorConfig:
    return OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage=stage,
        source_commit="a" * 40,
        input_revision="b" * 40,
    )


class FakeState:
    def __init__(self, phase: RunPhase = RunPhase.CREATED) -> None:
        self.value = SimpleNamespace(phase=phase, facts={})

    def load_or_create(self, _identity: object) -> SimpleNamespace:
        return self.value

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
        merged = {**self.value.facts, **facts}
        self.value = SimpleNamespace(phase=target, facts=merged)
        return self.value


class FakeStager:
    def __init__(self) -> None:
        self.calls = 0

    def prepare(self, _config: object, layout: RemoteLayout) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(layout=layout, reused=True)


class FakeOar:
    def __init__(self, statuses: list[JobStatus] | None = None) -> None:
        self.requests: list[object] = []
        self.statuses = statuses or []

    def submit(self, request: object) -> int:
        self.requests.append(request)
        return 42

    def status(self, _job_id: int) -> JobStatus:
        return self.statuses.pop(0)


class FakeSsh:
    def __init__(self, chunks: list[LogChunk] | None = None) -> None:
        self.chunks = chunks or []

    def read_since(self, _path: str, _offset: int) -> LogChunk:
        return self.chunks.pop(0) if self.chunks else LogChunk("", _offset, False)


def _controller(
    *,
    stage: str = "split",
    phase: RunPhase = RunPhase.CREATED,
    statuses: list[JobStatus] | None = None,
    chunks: list[LogChunk] | None = None,
) -> tuple[Controller, FakeState, FakeOar, FakeStager, list[object]]:
    state = FakeState(phase)
    oar = FakeOar(statuses)
    stager = FakeStager()
    emitted: list[object] = []
    controller = Controller(
        config=_config(stage),
        state=state,  # type: ignore[arg-type]
        ssh=FakeSsh(chunks),  # type: ignore[arg-type]
        oar=oar,  # type: ignore[arg-type]
        stager=stager,  # type: ignore[arg-type]
        layout=RemoteLayout(PurePosixPath("/remote/run")),
        emit=emitted.append,
        sleeper=lambda _seconds: None,
        poll_seconds=0,
    )
    return controller, state, oar, stager, emitted


def test_prepare_advances_every_durable_phase_and_is_idempotent() -> None:
    controller, state, _oar, stager, _emitted = _controller()
    controller.prepare(site="nancy")
    assert state.value.phase is RunPhase.REMOTE_PREPARED
    assert state.value.facts["site"] == "nancy"
    assert state.value.facts["checkout_reused"] is True
    assert stager.calls == 1
    controller.prepare(site="nancy")
    assert stager.calls == 1


def test_submit_split_and_reuse_existing_job() -> None:
    controller, state, oar, _stager, _emitted = _controller(
        phase=RunPhase.REMOTE_PREPARED
    )
    assert controller.submit(component=Stage.SPLIT) == 42
    assert state.value.phase is RunPhase.SUBMITTED
    assert state.value.facts["active_stage"] == "split"
    assert len(oar.requests) == 1
    assert controller.submit(component=Stage.SPLIT) == 42
    assert len(oar.requests) == 1


def test_submit_label_requires_assets_and_all_requires_component() -> None:
    controller, _state, _oar, _stager, _emitted = _controller(
        stage="label", phase=RunPhase.REMOTE_PREPARED
    )
    with pytest.raises(ControllerError, match="assets"):
        controller.submit(component=Stage.LABEL)
    controller, _state, _oar, _stager, _emitted = _controller(
        stage="all", phase=RunPhase.REMOTE_PREPARED
    )
    with pytest.raises(ControllerError, match="explicit component"):
        controller.submit()


def test_submit_label_serializes_assets() -> None:
    controller, state, oar, _stager, _emitted = _controller(
        stage="label", phase=RunPhase.REMOTE_PREPARED
    )
    assert (
        controller.submit(
            component=Stage.LABEL,
            input_parquet=PurePosixPath("/remote/input.parquet"),
            model_file=PurePosixPath("/remote/model.gguf"),
            tokenizer_dir=PurePosixPath("/remote/tokenizer"),
        )
        == 42
    )
    assert state.value.phase is RunPhase.SUBMITTED
    assert "/remote/model.gguf" in oar.requests[0].command  # type: ignore[union-attr]


def test_submit_rejects_unrelated_phase() -> None:
    controller, _state, _oar, _stager, _emitted = _controller(phase=RunPhase.COMPLETE)
    with pytest.raises(ControllerError, match="not ready"):
        controller.submit(component=Stage.SPLIT)


def test_monitor_streams_logs_and_records_offset() -> None:
    statuses = [
        JobStatus(42, JobState.RUNNING, node="gpu-1"),
        JobStatus(42, JobState.TERMINATED, exit_code=0),
    ]
    chunks = [LogChunk("progress\n", 9, False), LogChunk("", 9, False)]
    controller, state, _oar, _stager, emitted = _controller(
        phase=RunPhase.SUBMITTED, statuses=statuses, chunks=chunks
    )
    state.value.facts["job_id"] = 42
    state.value.facts["log_offset"] = "invalid"
    assert controller.monitor(42, log_name="stdout.log") is JobState.TERMINATED
    assert state.value.phase is RunPhase.RUNNING
    assert state.value.facts["node"] == "gpu-1"
    assert state.value.facts["log_offset"] == 9
    assert emitted[0].text == "progress\n"  # type: ignore[union-attr]


def test_monitor_handles_truncation_and_error_terminal() -> None:
    controller, state, _oar, _stager, emitted = _controller(
        phase=RunPhase.QUEUED,
        statuses=[JobStatus(42, JobState.ERROR)],
        chunks=[LogChunk("", 0, True)],
    )
    state.value.facts["log_offset"] = 99
    assert controller.monitor(42, log_name="stderr.log") is JobState.ERROR
    assert emitted == []
