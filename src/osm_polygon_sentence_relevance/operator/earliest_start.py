"""Pure planning for replacing a distant queued Grid'5000 allocation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from osm_polygon_sentence_relevance.operator.oar import (
    GRID5000_TZ,
    JobState,
    JobStatus,
)
from osm_polygon_sentence_relevance.operator.sites import (
    SiteProbe,
    SiteRequirements,
    evaluate_site,
)

IMMEDIATE_START_LIMIT = timedelta(minutes=10)
UNPREDICTED_TRIAL_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class ReplacementCandidate:
    """One compatible site with a factually idle suitable GPU."""

    site: SiteProbe


@dataclass(frozen=True, slots=True)
class ReplacementOutcome:
    """The sole allocation the caller should monitor after optimization."""

    site: str
    job_id: int
    replaced: bool


def _forecast_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return parsed.replace(tzinfo=GRID5000_TZ)


def should_seek_replacement(
    status: JobStatus,
    *,
    now: datetime,
    immediate_start_limit: timedelta = IMMEDIATE_START_LIMIT,
) -> bool:
    """Return whether a queued forecast is far enough away to optimize."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if immediate_start_limit <= timedelta(0):
        raise ValueError("immediate_start_limit must be positive")
    if status.state is not JobState.QUEUED:
        return False
    forecast = _forecast_datetime(status.scheduled_start)
    if forecast is None:
        return False
    local_now = now.astimezone(GRID5000_TZ)
    # A forecast in the past means the scheduler has rolled the estimate
    # without starting the job. Treat that as stale and try another site.
    if forecast <= local_now:
        return True
    return forecast > local_now + immediate_start_limit


def policy_type_for(now: datetime, *, walltime_seconds: int) -> str:
    """Return the earliest policy window that can contain the whole job."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if walltime_seconds <= 0:
        raise ValueError("walltime_seconds must be positive")
    local = now.astimezone(GRID5000_TZ)
    if local.weekday() < 5:
        duration = timedelta(seconds=walltime_seconds)
        day_start = local.replace(hour=9, minute=0, second=0, microsecond=0)
        day_end = local.replace(hour=19, minute=0, second=0, microsecond=0)
        if local < day_start:
            return "night" if local + duration <= day_start else "day"
        if local < day_end:
            return "day" if local + duration <= day_end else "night"
    return "night"


def forecast_exceeds_immediate_window(
    status: JobStatus,
    *,
    now: datetime,
    immediate_start_limit: timedelta = IMMEDIATE_START_LIMIT,
) -> bool:
    """Return whether OAR predicts this queued trial too late for the exception."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if status.state is not JobState.QUEUED:
        return False
    forecast = _forecast_datetime(status.scheduled_start)
    if forecast is None:
        return False
    return forecast > now.astimezone(GRID5000_TZ) + immediate_start_limit


def rank_replacement_candidates(
    probes: tuple[SiteProbe, ...] | list[SiteProbe],
    *,
    requirements: SiteRequirements,
    excluded_sites: frozenset[str] = frozenset(),
    require_label_runtime: bool = True,
) -> tuple[ReplacementCandidate, ...]:
    """Rank compatible sites with a currently idle suitable GPU.

    Label continuations require a staged llama runtime. Split continuations
    only need the CUDA sentence-splitting environment, so callers can disable
    that additional filter for the split component.
    """

    candidates = [
        ReplacementCandidate(probe)
        for probe in probes
        if probe.name not in excluded_sites
        and probe.idle_compatible
        and (not require_label_runtime or probe.label_runtime_ready)
        and evaluate_site(probe, requirements).compatible
    ]
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                not candidate.site.has_managed_run,
                candidate.site.name,
            ),
        )
    )


def attempt_immediate_replacement(
    *,
    fallback_site: str,
    fallback_job_id: int,
    candidates: tuple[ReplacementCandidate, ...],
    prepare: Callable[[ReplacementCandidate], None],
    submit: Callable[[ReplacementCandidate], int],
    status: Callable[[str, int], JobStatus],
    cancel: Callable[[str, int], None],
    persist_trial: Callable[[str, int, float], None],
    adopt_trial: Callable[[str, int], None],
    clear_trial: Callable[[int], None],
    emit: Callable[[str], None],
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    wall_clock: Callable[[], datetime],
    existing_trial: tuple[ReplacementCandidate, int, float] | None = None,
    trial_seconds: float = 600.0,
    poll_seconds: float = 30.0,
) -> ReplacementOutcome:
    """Try candidates sequentially while retaining one queued fallback."""

    if fallback_job_id <= 0:
        raise ValueError("fallback_job_id must be positive")
    if trial_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("trial and poll durations must be positive")
    pending = list(candidates)
    initial = existing_trial
    while initial is not None or pending:
        if initial is not None:
            candidate, job_id, deadline = initial
            initial = None
            submitted_now = False
        else:
            candidate = pending.pop(0)
            submitted_now = True
        site = candidate.site.name
        if submitted_now:
            try:
                prepare(candidate)
                job_id = submit(candidate)
            except Exception as exc:
                emit(f"Candidate {site} rejected before submission: {exc}")
                continue
            deadline = monotonic() + trial_seconds
            persist_trial(site, job_id, deadline)
            emit(
                f"Trial job {job_id} submitted on {site}; "
                f"must start within {int(trial_seconds)} seconds"
            )
        else:
            emit(f"Reattaching to immediate-start trial job {job_id} on {site}")
        while True:
            fallback_observed = status(fallback_site, fallback_job_id)
            if fallback_observed.state is JobState.RUNNING:
                cancel(site, job_id)
                clear_trial(job_id)
                emit(
                    f"Fallback job {fallback_job_id} started first; "
                    f"cancelled trial job {job_id}"
                )
                return ReplacementOutcome(
                    fallback_site,
                    fallback_job_id,
                    replaced=False,
                )
            observed = status(site, job_id)
            if observed.state is JobState.RUNNING:
                adopt_trial(site, job_id)
                cancel(fallback_site, fallback_job_id)
                emit(
                    f"Trial job {job_id} is running on {site}; "
                    f"cancelled fallback job {fallback_job_id}"
                )
                return ReplacementOutcome(site, job_id, replaced=True)
            if forecast_exceeds_immediate_window(
                observed,
                now=wall_clock(),
            ):
                cancel(site, job_id)
                clear_trial(job_id)
                emit(
                    f"Trial job {job_id} forecast {observed.scheduled_start} "
                    "exceeds immediate-start window; cancelled"
                )
                break
            if observed.state in {
                JobState.TERMINATED,
                JobState.ERROR,
                JobState.MISSING,
            }:
                clear_trial(job_id)
                emit(f"Trial job {job_id} did not start; fallback retained")
                break
            remaining = deadline - monotonic()
            if remaining <= 0:
                cancel(site, job_id)
                clear_trial(job_id)
                emit(
                    f"Trial job {job_id} missed the immediate-start deadline; "
                    "fallback retained"
                )
                break
            sleep(min(poll_seconds, remaining))
    return ReplacementOutcome(fallback_site, fallback_job_id, replaced=False)


__all__ = [
    "IMMEDIATE_START_LIMIT",
    "UNPREDICTED_TRIAL_SECONDS",
    "ReplacementCandidate",
    "ReplacementOutcome",
    "attempt_immediate_replacement",
    "forecast_exceeds_immediate_window",
    "policy_type_for",
    "rank_replacement_candidates",
    "should_seek_replacement",
]
