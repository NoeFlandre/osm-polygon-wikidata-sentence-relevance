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
from osm_polygon_sentence_relevance.operator.label_lanes import label_lane_plan
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
        self.cleanup_calls = 0

    def prepare(self, _config: object, layout: RemoteLayout) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(layout=layout, reused=True)

    def clean_generated_python_caches(self, _layout: RemoteLayout) -> None:
        self.cleanup_calls += 1


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
    controller, state, oar, stager, _emitted = _controller(
        phase=RunPhase.REMOTE_PREPARED
    )
    assert controller.submit(component=Stage.SPLIT) == 42
    assert state.value.phase is RunPhase.SUBMITTED
    assert state.value.facts["active_stage"] == "split"
    assert len(oar.requests) == 1
    assert stager.cleanup_calls == 1
    assert controller.submit(component=Stage.SPLIT) == 42
    assert len(oar.requests) == 1


def test_submit_split_serializes_and_persists_resume_bundle() -> None:
    controller, state, oar, _stager, _emitted = _controller(
        phase=RunPhase.REMOTE_PREPARED
    )
    bundle = PurePosixPath("/remote/run/split-resume/" + "c" * 20)

    assert controller.submit(component=Stage.SPLIT, split_resume_bundle=bundle) == 42

    assert oar.requests[0].command[-1] == str(bundle)  # type: ignore[union-attr]
    assert state.value.facts["split_resume_bundle"] == str(bundle)


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


def test_submit_worldwide_label_persists_exact_lane() -> None:
    config = OperatorConfig.build(
        scope="all",
        stage="all",
        source_commit="a" * 40,
        input_revision="b" * 40,
        row_limit=128,
        sampling_target=200_000,
    )
    state = FakeState(RunPhase.REMOTE_PREPARED)
    oar = FakeOar()
    layout = RemoteLayout(PurePosixPath("/remote/run"))
    controller = Controller(
        config=config,
        state=state,  # type: ignore[arg-type]
        ssh=FakeSsh(),  # type: ignore[arg-type]
        oar=oar,  # type: ignore[arg-type]
        stager=FakeStager(),  # type: ignore[arg-type]
        layout=layout,
    )
    plan = label_lane_plan(config, layout.root, {})

    assert (
        controller.submit(
            component=Stage.LABEL,
            input_parquet=PurePosixPath("/remote/input.parquet"),
            model_file=PurePosixPath("/remote/model.gguf"),
            tokenizer_dir=PurePosixPath("/remote/tokenizer"),
            label_plan=plan,
        )
        == 42
    )

    assert state.value.facts["label_lane"] == "smoke"
    command = oar.requests[0].command  # type: ignore[union-attr]
    assert "/remote/run/label-smoke-work" in command
    assert command[-2] == "smoke"


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
    assert [item.text for item in emitted] == [
        "running on gpu-1",
        "progress\n",
        "terminated (exit 0)",
    ]  # type: ignore[union-attr]


def test_monitor_handles_truncation_and_error_terminal() -> None:
    controller, state, _oar, _stager, emitted = _controller(
        phase=RunPhase.QUEUED,
        statuses=[JobStatus(42, JobState.ERROR)],
        chunks=[LogChunk("", 0, True)],
    )
    state.value.facts["log_offset"] = 99
    assert controller.monitor(42, log_name="stderr.log") is JobState.ERROR
    assert [item.text for item in emitted] == ["error"]


def test_monitor_reports_queued_schedule_before_payload_logs_exist() -> None:
    controller, state, _oar, _stager, emitted = _controller(
        phase=RunPhase.SUBMITTED,
        statuses=[
            JobStatus(
                42,
                JobState.QUEUED,
                scheduled_start="2026-07-29 19:00:00",
                walltime_seconds=3300,
            ),
            JobStatus(
                42,
                JobState.QUEUED,
                scheduled_start="2026-07-29 19:00:00",
                walltime_seconds=3300,
            ),
            JobStatus(42, JobState.TERMINATED, exit_code=0),
        ],
    )
    state.value.facts["job_id"] = 42

    assert controller.monitor(42, log_name="stdout.log") is JobState.TERMINATED
    assert [item.text for item in emitted] == [
        "queued; scheduled start 2026-07-29 19:00:00 Europe/Paris; walltime 00:55:00",
        "terminated (exit 0)",
    ]


def test_monitor_explains_when_scheduler_has_no_start_prediction() -> None:
    controller, state, _oar, _stager, emitted = _controller(
        phase=RunPhase.SUBMITTED,
        statuses=[
            JobStatus(42, JobState.QUEUED, walltime_seconds=3300),
            JobStatus(42, JobState.TERMINATED, exit_code=0),
        ],
    )
    state.value.facts["job_id"] = 42

    assert controller.monitor(42, log_name="stdout.log") is JobState.TERMINATED
    assert emitted[0].text == (
        "queued; scheduler has no start-time prediction; walltime 00:55:00"
    )


def test_monitor_running_reports_assigned_node_and_walltime() -> None:
    controller, state, _oar, _stager, emitted = _controller(
        phase=RunPhase.QUEUED,
        statuses=[
            JobStatus(42, JobState.RUNNING, node="chifflet-6", walltime_seconds=3600),
            JobStatus(42, JobState.TERMINATED, exit_code=0),
        ],
    )
    state.value.facts["job_id"] = 42

    assert controller.monitor(42, log_name="stdout.log") is JobState.TERMINATED
    assert emitted[0].text == "running on chifflet-6; walltime 01:00:00"


def test_monitor_does_not_repeat_identical_scheduler_states() -> None:
    queued = JobStatus(
        42,
        JobState.QUEUED,
        scheduled_start="2026-07-29 19:00:00",
        walltime_seconds=3300,
    )
    controller, state, _oar, _stager, emitted = _controller(
        phase=RunPhase.SUBMITTED,
        statuses=[
            queued,
            queued,
            queued,
            JobStatus(42, JobState.TERMINATED, exit_code=0),
        ],
    )
    state.value.facts["job_id"] = 42

    assert controller.monitor(42, log_name="stdout.log") is JobState.TERMINATED
    assert [item.text for item in emitted] == [
        "queued; scheduled start 2026-07-29 19:00:00 Europe/Paris; walltime 00:55:00",
        "terminated (exit 0)",
    ]
