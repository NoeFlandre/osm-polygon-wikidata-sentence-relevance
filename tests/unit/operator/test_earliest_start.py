"""Pure contracts for safe earliest-start replacement planning."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from osm_polygon_sentence_relevance.operator.earliest_start import (
    IMMEDIATE_START_LIMIT,
    ReplacementCandidate,
    rank_replacement_candidates,
    should_seek_replacement,
)
from osm_polygon_sentence_relevance.operator.oar import (
    GRID5000_TZ,
    JobState,
    JobStatus,
)
from osm_polygon_sentence_relevance.operator.sites import (
    SiteProbe,
    SiteRequirements,
)


def _status(
    state: JobState = JobState.QUEUED,
    *,
    start: str | None = "2026-07-29 19:00:00",
) -> JobStatus:
    return JobStatus(42, state, scheduled_start=start, walltime_seconds=3300)


def _probe(
    name: str,
    *,
    idle: bool = True,
    managed: bool = False,
    memory: int = 80_000,
    free: int = 100 * 1024**3,
    runtime_ready: bool = True,
) -> SiteProbe:
    return SiteProbe(
        name,
        name,
        True,
        memory,
        (8, 0),
        free,
        0,
        idle_compatible=idle,
        has_managed_run=managed,
        label_runtime_ready=runtime_ready,
    )


def test_distant_queued_forecast_seeks_replacement() -> None:
    now = datetime(2026, 7, 29, 14, 0, tzinfo=GRID5000_TZ)
    assert should_seek_replacement(_status(), now=now)


def test_forecast_inside_ten_minutes_keeps_fallback() -> None:
    now = datetime(2026, 7, 29, 18, 51, tzinfo=GRID5000_TZ)
    assert not should_seek_replacement(_status(), now=now)
    assert timedelta(minutes=10) == IMMEDIATE_START_LIMIT


def test_nonqueued_or_unknown_forecast_never_seeks_replacement() -> None:
    now = datetime(2026, 7, 29, 14, 0, tzinfo=GRID5000_TZ)
    for state in (JobState.RUNNING, JobState.FINISHING, JobState.TERMINATED):
        assert not should_seek_replacement(_status(state), now=now)
    assert not should_seek_replacement(_status(start=None), now=now)


def test_invalid_forecast_never_seeks_replacement_but_stale_forecast_does() -> None:
    now = datetime(2026, 7, 29, 14, 0, tzinfo=GRID5000_TZ)
    assert not should_seek_replacement(
        _status(start="not-a-timestamp"),
        now=now,
    )
    assert should_seek_replacement(
        _status(start="2026-07-29 13:00:00"),
        now=now,
    )


def test_naive_now_is_rejected() -> None:
    now = datetime(2026, 7, 29, 14, 0)
    with pytest.raises(ValueError, match="now must be timezone-aware"):
        should_seek_replacement(_status(), now=now)


def test_candidates_require_factual_idle_compatible_capacity() -> None:
    requirements = SiteRequirements()
    ranked = rank_replacement_candidates(
        (
            _probe("idle"),
            _probe("busy", idle=False),
            _probe("small", memory=20_000),
            _probe("full", free=1),
            _probe("not-ready", runtime_ready=False),
        ),
        requirements=requirements,
    )
    assert [candidate.site.name for candidate in ranked] == ["idle"]


def test_split_candidates_do_not_require_label_runtime() -> None:
    ranked = rank_replacement_candidates(
        (_probe("split-ready", runtime_ready=False),),
        requirements=SiteRequirements(),
        require_label_runtime=False,
    )
    assert [candidate.site.name for candidate in ranked] == ["split-ready"]


def test_candidates_prefer_prepared_run_then_name() -> None:
    ranked = rank_replacement_candidates(
        (
            _probe("zeta"),
            _probe("sophia", managed=True),
            _probe("alpha"),
        ),
        requirements=SiteRequirements(),
    )
    assert ranked == (
        ReplacementCandidate(_probe("sophia", managed=True)),
        ReplacementCandidate(_probe("alpha")),
        ReplacementCandidate(_probe("zeta")),
    )


def test_current_site_can_be_excluded() -> None:
    ranked = rank_replacement_candidates(
        (_probe("nancy"), _probe("sophia")),
        requirements=SiteRequirements(),
        excluded_sites=frozenset({"sophia"}),
    )
    assert [candidate.site.name for candidate in ranked] == ["nancy"]
