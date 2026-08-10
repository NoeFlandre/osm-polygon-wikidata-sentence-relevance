"""Lifecycle tests for preserving a queued fallback during optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pytest

from osm_polygon_sentence_relevance.operator.earliest_start import (
    QUEUED_RESCAN_SECONDS,
    ReplacementCandidate,
    ReplacementOutcome,
    attempt_immediate_replacement,
    forecast_exceeds_immediate_window,
    policy_type_for,
    race_queued_replacements,
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
        (datetime(2026, 7, 30, 20, 0, tzinfo=GRID5000_TZ), "night"),
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


def test_forecast_window_rejects_naive_clock() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        forecast_exceeds_immediate_window(
            JobStatus(42, JobState.QUEUED),
            now=datetime(2026, 7, 30, 10, 0),
        )


def test_replacement_rejects_invalid_job_and_durations() -> None:
    harness = _Harness()
    common = {
        "fallback_site": "sophia",
        "candidates": (),
        "prepare": harness.prepare,
        "submit": harness.submit,
        "status": harness.status,
        "cancel": harness.cancel,
        "persist_trial": harness.persist,
        "adopt_trial": harness.adopt,
        "clear_trial": harness.clear,
        "emit": harness.emitted.append,
        "monotonic": lambda: harness.clock,
        "sleep": harness.sleep,
        "wall_clock": lambda: datetime(2026, 7, 30, 14, 0, tzinfo=GRID5000_TZ),
    }
    with pytest.raises(ValueError, match="fallback_job_id must be positive"):
        attempt_immediate_replacement(fallback_job_id=0, **common)
    with pytest.raises(ValueError, match="trial and poll durations must be positive"):
        attempt_immediate_replacement(
            fallback_job_id=42,
            trial_seconds=0,
            **common,
        )


def test_existing_trial_is_reattached_without_resubmission() -> None:
    harness = _Harness(statuses={101: [JobStatus(101, JobState.RUNNING)]})
    outcome = attempt_immediate_replacement(
        fallback_site="sophia",
        fallback_job_id=42,
        candidates=(),
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
        existing_trial=(_candidate("nancy"), 101, 60),
    )
    assert outcome == ReplacementOutcome("nancy", 101, replaced=True)
    assert harness.submitted == []
    assert harness.emitted == [
        "Reattaching to immediate-start trial job 101 on nancy",
        "Trial job 101 is running on nancy; cancelled fallback job 42",
    ]


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


def test_distant_fallback_is_rescanned_until_a_replacement_starts() -> None:
    attempts = iter(
        (
            ReplacementOutcome("sophia", 42, replaced=False),
            ReplacementOutcome("nancy", 101, replaced=True),
        )
    )
    observed: list[tuple[str, int]] = []
    sleeps: list[float] = []

    def status(site: str, job_id: int) -> JobStatus:
        observed.append((site, job_id))
        if job_id == 42:
            return JobStatus(
                42,
                JobState.QUEUED,
                scheduled_start="2026-07-30 19:00:00",
            )
        return JobStatus(101, JobState.RUNNING)

    outcome = race_queued_replacements(
        fallback_site="sophia",
        fallback_job_id=42,
        attempt=lambda _site, _job_id: next(attempts),
        status=status,
        emit=lambda _message: None,
        sleep=sleeps.append,
        wall_clock=lambda: datetime(2026, 7, 30, 14, 0, tzinfo=GRID5000_TZ),
    )

    assert outcome == ReplacementOutcome("nancy", 101, replaced=True)
    assert observed == [("sophia", 42)]
    assert sleeps == [QUEUED_RESCAN_SECONDS]


def test_unpredicted_queue_is_rescanned_but_near_forecast_is_not() -> None:
    attempts = 0
    sleeps: list[float] = []

    def attempt(site: str, job_id: int) -> ReplacementOutcome:
        nonlocal attempts
        attempts += 1
        return ReplacementOutcome(site, job_id, replaced=False)

    statuses = iter(
        (
            JobStatus(42, JobState.QUEUED, scheduled_start=None),
            JobStatus(
                42,
                JobState.QUEUED,
                scheduled_start="2026-07-30 14:08:00",
            ),
        )
    )
    outcome = race_queued_replacements(
        fallback_site="sophia",
        fallback_job_id=42,
        attempt=attempt,
        status=lambda _site, _job_id: next(statuses),
        emit=lambda _message: None,
        sleep=sleeps.append,
        wall_clock=lambda: datetime(2026, 7, 30, 14, 0, tzinfo=GRID5000_TZ),
    )

    assert outcome == ReplacementOutcome("sophia", 42, replaced=False)
    assert attempts == 2
    assert sleeps == [QUEUED_RESCAN_SECONDS]


def test_rolling_race_reports_next_scan_and_validates_interval() -> None:
    emitted: list[str] = []
    statuses = iter(
        (
            JobStatus(42, JobState.QUEUED, scheduled_start=None),
            JobStatus(42, JobState.RUNNING),
        )
    )
    outcome = race_queued_replacements(
        fallback_site="sophia",
        fallback_job_id=42,
        attempt=lambda site, job_id: ReplacementOutcome(site, job_id, False),
        status=lambda _site, _job_id: next(statuses),
        emit=emitted.append,
        sleep=lambda _seconds: None,
        wall_clock=lambda: datetime(2026, 7, 30, 14, 0, tzinfo=GRID5000_TZ),
    )
    assert outcome == ReplacementOutcome("sophia", 42, replaced=False)
    assert emitted == [
        "Job 42 on sophia remains queued; checking every site again in 300 seconds"
    ]

    with pytest.raises(ValueError, match="rescan interval must be positive"):
        race_queued_replacements(
            fallback_site="sophia",
            fallback_job_id=42,
            attempt=lambda site, job_id: ReplacementOutcome(site, job_id, False),
            status=lambda _site, _job_id: JobStatus(42, JobState.RUNNING),
            emit=lambda _message: None,
            sleep=lambda _seconds: None,
            wall_clock=lambda: datetime(2026, 7, 30, 14, 0, tzinfo=GRID5000_TZ),
            rescan_seconds=0,
        )
    with pytest.raises(ValueError, match="fallback_job_id must be positive"):
        race_queued_replacements(
            fallback_site="sophia",
            fallback_job_id=0,
            attempt=lambda site, job_id: ReplacementOutcome(site, job_id, False),
            status=lambda _site, _job_id: JobStatus(42, JobState.RUNNING),
            emit=lambda _message: None,
            sleep=lambda _seconds: None,
            wall_clock=lambda: datetime(2026, 7, 30, 14, 0, tzinfo=GRID5000_TZ),
        )
