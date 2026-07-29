"""Resumable controller for one Grid'5000 allocation at a time."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath

from osm_polygon_sentence_relevance.operator.config import OperatorConfig, Stage
from osm_polygon_sentence_relevance.operator.oar import (
    JobState,
    JobStatus,
    OarClient,
    SubmissionRequest,
    format_job_status,
)
from osm_polygon_sentence_relevance.operator.ssh import LogChunk, SshClient
from osm_polygon_sentence_relevance.operator.staging import Stager
from osm_polygon_sentence_relevance.operator.state import RunPhase, StateStore
from osm_polygon_sentence_relevance.operator.workflows import (
    RemoteLayout,
    label_submission,
    split_submission,
)


@dataclass(frozen=True, slots=True)
class LiveProgress:
    """One terminal-safe live log update."""

    job_id: int
    stream: str
    text: str
    offset: int


class ControllerError(RuntimeError):
    """The autonomous run cannot safely continue."""


class Controller:
    """Prepare, submit, and monitor the requested production stage."""

    def __init__(
        self,
        *,
        config: OperatorConfig,
        state: StateStore,
        ssh: SshClient,
        oar: OarClient,
        stager: Stager,
        layout: RemoteLayout,
        emit: Callable[[LiveProgress], None] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        poll_seconds: float = 30.0,
    ) -> None:
        self.config = config
        self.state = state
        self.ssh = ssh
        self.oar = oar
        self.stager = stager
        self.layout = layout
        self.emit = emit or (lambda _progress: None)
        self.sleeper = sleeper
        self.poll_seconds = poll_seconds

    def prepare(self, *, site: str) -> None:
        """Persist deterministic preparation phases and stage the checkout."""

        current = self.state.load_or_create(self.config.run_identity)
        if current.phase is RunPhase.CREATED:
            current = self.state.transition(
                expected=RunPhase.CREATED,
                target=RunPhase.INPUTS_RESOLVED,
                facts={"input_revision": self.config.input_dataset_revision or ""},
            )
        if current.phase is RunPhase.INPUTS_RESOLVED:
            current = self.state.transition(
                expected=RunPhase.INPUTS_RESOLVED,
                target=RunPhase.SITE_SELECTED,
                facts={"site": site},
            )
        if current.phase is RunPhase.SITE_SELECTED:
            current = self.state.transition(
                expected=RunPhase.SITE_SELECTED,
                target=RunPhase.STORAGE_READY,
                facts={"remote_root": str(self.layout.root)},
            )
        if current.phase is RunPhase.STORAGE_READY:
            result = self.stager.prepare(self.config, self.layout)
            self.state.transition(
                expected=RunPhase.STORAGE_READY,
                target=RunPhase.REMOTE_PREPARED,
                facts={"checkout_reused": result.reused},
            )

    def submit(
        self,
        *,
        component: Stage | None = None,
        input_parquet: PurePosixPath | None = None,
        model_file: PurePosixPath | None = None,
        tokenizer_dir: PurePosixPath | None = None,
    ) -> int:
        """Submit exactly once from REMOTE_PREPARED."""

        state = self.state.load()
        if state.phase is not RunPhase.REMOTE_PREPARED:
            existing = state.facts.get("job_id")
            if (
                state.phase
                in {
                    RunPhase.SUBMITTED,
                    RunPhase.QUEUED,
                    RunPhase.RUNNING,
                    RunPhase.CHECKPOINTED,
                }
                and type(existing) is int
            ):
                return existing
            raise ControllerError("run is not ready for submission")

        request: SubmissionRequest
        selected = component or self.config.stage
        if selected is Stage.SPLIT:
            request = split_submission(self.config, self.layout)
        elif selected is Stage.LABEL:
            if input_parquet is None or model_file is None or tokenizer_dir is None:
                raise ControllerError("label assets are required")
            request = label_submission(
                self.config,
                self.layout,
                input_parquet=input_parquet,
                model_file=model_file,
                tokenizer_dir=tokenizer_dir,
            )
        else:
            raise ControllerError("stage=all requires an explicit component")

        job_id = self.oar.submit(request)
        self.state.transition(
            expected=RunPhase.REMOTE_PREPARED,
            target=RunPhase.SUBMITTED,
            facts={"job_id": job_id, "log_offset": 0, "active_stage": selected.value},
        )
        return job_id

    def monitor(self, job_id: int, *, log_name: str) -> JobState:
        """Stream a growing job log until OAR reaches a terminal state."""

        state = self.state.load()
        if state.phase is RunPhase.SUBMITTED:
            self.state.transition(
                expected=RunPhase.SUBMITTED,
                target=RunPhase.QUEUED,
                facts={"job_id": job_id},
            )
        offset_raw = self.state.load().facts.get("log_offset", 0)
        offset = offset_raw if type(offset_raw) is int else 0
        remote_log = str(self.layout.logs / str(job_id) / log_name)
        previous_status: (
            tuple[JobState, str | None, str | None, int | None, int | None] | None
        ) = None

        while True:
            status = self.oar.status(job_id)
            status_key = _status_key(status)
            if status_key != previous_status:
                self.emit(
                    LiveProgress(
                        job_id,
                        "scheduler",
                        format_job_status(status),
                        offset,
                    )
                )
                previous_status = status_key
            current = self.state.load()
            if status.state is JobState.RUNNING and current.phase is RunPhase.QUEUED:
                self.state.transition(
                    expected=RunPhase.QUEUED,
                    target=RunPhase.RUNNING,
                    facts={"node": status.node or "", "job_id": job_id},
                )
            chunk: LogChunk = self.ssh.read_since(remote_log, offset)
            if chunk.reset:
                offset = 0
            elif chunk.text:
                offset = chunk.next_offset
                self.emit(LiveProgress(job_id, log_name, chunk.text, offset))
                current = self.state.load()
                if current.phase in {RunPhase.QUEUED, RunPhase.RUNNING}:
                    self.state.transition(
                        expected=current.phase,
                        target=current.phase,
                        facts={"log_offset": offset},
                    )
            if status.state in {
                JobState.TERMINATED,
                JobState.ERROR,
                JobState.MISSING,
            }:
                return status.state
            self.sleeper(self.poll_seconds)


def _status_key(
    status: JobStatus,
) -> tuple[JobState, str | None, str | None, int | None, int | None]:
    return (
        status.state,
        status.node,
        status.scheduled_start,
        status.exit_code,
        status.walltime_seconds,
    )


__all__ = ["Controller", "ControllerError", "LiveProgress"]
