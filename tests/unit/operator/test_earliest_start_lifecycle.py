"""Lifecycle tests for preserving a queued fallback during optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pytest

from osm_polygon_sentence_relevance.operator.earliest_start import (
    ReplacementCandidate,
    ReplacementOutcome,
    attempt_immediate_replacement,
    policy_type_for,
)
from osm_polygon_sentence_relevance.operator.oar import GRID5000_TZ, JobState, JobStatus
from osm_polygon_sentence_relevance.operator.sites import SiteProbe


def _candidate(name: str) -> ReplacementCandidate:
    return ReplacementCandidate(
        SiteProbe(
            name,
            name,
            True,
            80_000,
            (8, 0),
            100,
            0,
            idle_compatible=True,
            label_runtime_ready=True,
        )
    )


@dataclass
class _Harness:
    statuses: dict[int, list[JobStatus]] = field(default_factory=dict)
    prepared: list[str] = field(default_factory=list)
    submitted: list[str] = field(default_factory=list)
    cancelled: list[tuple[str, int]] = field(default_factory=list)
    persisted: list[tuple[str, int, float]] = field(default_factory=list)
    adopted: list[tuple[str, int]] = field(default_factory=list)
    cleared: list[int] = field(default_factory=list)
    emitted: list[str] = field(default_factory=list)
    clock: float = 0.0
    next_job: int = 100
    fail_prepare: frozenset[str] = frozenset()

    def prepare(self, candidate: ReplacementCandidate) -> None:
        self.prepared.append(candidate.site.name)
        if candidate.site.name in self.fail_prepare:
            raise RuntimeError("not ready")

    def submit(self, candidate: ReplacementCandidate) -> int:
        self.submitted.append(candidate.site.name)
        self.next_job += 1
        return self.next_job

    def status(self, _site: str, job_id: int) -> JobStatus:
        if job_id == 42 and job_id not in self.statuses:
            return JobStatus(42, JobState.QUEUED)
        queue = self.statuses[job_id]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def cancel(self, site: str, job_id: int) -> None:
        self.cancelled.append((site, job_id))

    def persist(self, site: str, job_id: int, deadline: float) -> None:
        self.persisted.append((site, job_id, deadline))

    def adopt(self, site: str, job_id: int) -> None:
        self.adopted.append((site, job_id))

    def clear(self, job_id: int) -> None:
        self.cleared.append(job_id)

    def sleep(self, seconds: float) -> None:
        self.clock += seconds


def _run(
    harness: _Harness,
    candidates: tuple[ReplacementCandidate, ...],
    *,
    timeout: float = 60,
) -> ReplacementOutcome:
    return attempt_immediate_replacement(
        fallback_site="sophia",
        fallback_job_id=42,
        candidates=candidates,
        prepare=harness.prepare,
        submit=harness.submit,
        status=harness.status,
        cancel=harness.cancel,
        persist_trial=harness.persist,
        adopt_trial=harness.adopt,
        clear_trial=harness.clear,
        emit=harness.emitted.append,
        monotonic=lambda: harness.clock,
        sleep=harness.sleep,
        wall_clock=lambda: datetime(2026, 7, 30, 14, 0, tzinfo=GRID5000_TZ),
        trial_seconds=timeout,
        poll_seconds=30,
    )


def test_running_replacement_is_adopted_before_fallback_is_cancelled() -> None:
    harness = _Harness(
        statuses={
            101: [
                JobStatus(101, JobState.QUEUED),
                JobStatus(101, JobState.RUNNING),
            ]
        }
    )
    outcome = _run(harness, (_candidate("nancy"),))
    assert outcome == ReplacementOutcome("nancy", 101, replaced=True)
    assert harness.persisted == [("nancy", 101, 60)]
    assert harness.adopted == [("nancy", 101)]
    assert harness.cancelled == [("sophia", 42)]
    assert harness.cleared == []


def test_running_fallback_skips_expensive_trial_preparation() -> None:
    harness = _Harness(statuses={42: [JobStatus(42, JobState.RUNNING)]})

    outcome = _run(harness, (_candidate("nancy"),))

    assert outcome == ReplacementOutcome("sophia", 42, replaced=False)
    assert harness.prepared == []
    assert harness.submitted == []
    assert harness.cancelled == []


def test_trial_timeout_cancels_only_trial_and_retains_fallback() -> None:
    harness = _Harness(statuses={101: [JobStatus(101, JobState.QUEUED)]})
    outcome = _run(harness, (_candidate("nancy"),), timeout=30)
    assert outcome == ReplacementOutcome("sophia", 42, replaced=False)
    assert harness.cancelled == [("nancy", 101)]
    assert harness.adopted == []
    assert harness.clock == 30
    assert harness.cleared == [101]


def test_late_scheduler_forecast_cancels_without_waiting_and_tries_next() -> None:
    harness = _Harness(
        statuses={
            101: [
                JobStatus(
                    101,
                    JobState.QUEUED,
                    scheduled_start="2026-07-30 19:00:00",
                )
            ],
            102: [JobStatus(102, JobState.RUNNING)],
        }
    )

    outcome = _run(harness, (_candidate("grenoble"), _candidate("nancy")))

    assert outcome == ReplacementOutcome("nancy", 102, replaced=True)
    assert harness.submitted == ["grenoble", "nancy"]
    assert harness.cancelled == [("grenoble", 101), ("sophia", 42)]
    assert harness.cleared == [101]
    assert harness.clock == 0
    assert any(
        "forecast 2026-07-30 19:00:00 exceeds immediate-start window" in line
        for line in harness.emitted
    )


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 7, 30, 10, 0, tzinfo=GRID5000_TZ), "day"),
        (datetime(2026, 7, 30, 18, 50, tzinfo=GRID5000_TZ), "night"),
        (datetime(2026, 7, 30, 8, 50, tzinfo=GRID5000_TZ), "day"),
        (datetime(2026, 7, 30, 8, 30, tzinfo=GRID5000_TZ), "night"),
        (datetime(2026, 8, 1, 12, 0, tzinfo=GRID5000_TZ), "night"),
    ],
)
def test_policy_type_chooses_window_that_can_fit_micro_allocation(
    now: datetime, expected: str
) -> None:
    assert policy_type_for(now, walltime_seconds=1_200) == expected


def test_policy_type_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        policy_type_for(datetime(2026, 7, 30, 10, 0), walltime_seconds=1_200)
    with pytest.raises(ValueError, match="positive"):
        policy_type_for(
            datetime(2026, 7, 30, 10, 0, tzinfo=GRID5000_TZ),
            walltime_seconds=0,
        )


def test_failed_candidate_preparation_moves_to_next_candidate() -> None:
    harness = _Harness(
        fail_prepare=frozenset({"nancy"}),
        statuses={101: [JobStatus(101, JobState.RUNNING)]},
    )
    outcome = _run(harness, (_candidate("nancy"), _candidate("rennes")))
    assert outcome.site == "rennes"
    assert harness.submitted == ["rennes"]
    assert any("nancy" in line and "not ready" in line for line in harness.emitted)


def test_terminal_trial_is_cleared_then_next_candidate_is_tried() -> None:
    harness = _Harness(
        statuses={
            101: [JobStatus(101, JobState.ERROR)],
            102: [JobStatus(102, JobState.RUNNING)],
        }
    )
    outcome = _run(harness, (_candidate("nancy"), _candidate("rennes")))
    assert outcome == ReplacementOutcome("rennes", 102, replaced=True)
    assert harness.cleared == [101]
    assert harness.cancelled == [("sophia", 42)]


def test_ctrl_c_after_trial_persistence_preserves_both_job_records() -> None:
    harness = _Harness(statuses={101: [JobStatus(101, JobState.QUEUED)]})

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    harness.sleep = interrupt  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        _run(harness, (_candidate("nancy"),))
    assert harness.persisted == [("nancy", 101, 60)]
    assert harness.cancelled == []
    assert harness.adopted == []


def test_fallback_start_cancels_trial_before_it_can_run() -> None:
    harness = _Harness(
        statuses={
            42: [
                JobStatus(42, JobState.QUEUED),
                JobStatus(42, JobState.RUNNING),
            ],
            101: [JobStatus(101, JobState.QUEUED)],
        }
    )
    outcome = _run(harness, (_candidate("nancy"),))
    assert outcome == ReplacementOutcome("sophia", 42, replaced=False)
    assert harness.cancelled == [("nancy", 101)]
    assert harness.adopted == []
    assert harness.clock == 0
