"""Contract tests for durable operator recovery policy helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator import recovery
from osm_polygon_sentence_relevance.operator.state import RunPhase


def _state(phase: RunPhase, **facts: object) -> SimpleNamespace:
    return SimpleNamespace(phase=phase, facts=facts)


def test_reattach_decision_returns_a_valid_live_job() -> None:
    state = _state(RunPhase.RUNNING, site="grenoble", job_id=42)

    assert recovery.reattach_decision(state) == ("grenoble", 42)


@pytest.mark.parametrize(
    ("phase", "facts"),
    [
        (RunPhase.RUNNING, {"site": "grenoble", "job_id": 0}),
        (RunPhase.RUNNING, {"site": "grenoble", "job_id": True}),
        (RunPhase.RUNNING, {"site": "", "job_id": 42}),
        (RunPhase.CREATED, {"site": "grenoble", "job_id": 42}),
    ],
)
def test_reattach_decision_rejects_invalid_or_unsubmitted_state(
    phase: RunPhase,
    facts: dict[str, object],
) -> None:
    assert recovery.reattach_decision(_state(phase, **facts)) is None


def test_reattach_decision_allows_a_fresh_failed_job_recovery() -> None:
    state = _state(
        RunPhase.FAILED,
        site="grenoble",
        failed_job_id=44,
        recovered_from_job_id=43,
    )

    assert recovery.reattach_decision(state) == ("grenoble", 44)


def test_reattach_decision_rejects_recovering_the_same_failed_job_twice() -> None:
    state = _state(
        RunPhase.FAILED,
        site="grenoble",
        failed_job_id=44,
        recovered_from_job_id=44,
    )

    assert recovery.reattach_decision(state) is None


@pytest.mark.parametrize(
    "facts",
    [
        {"site": "grenoble", "failed_job_id": 0},
        {"site": "", "failed_job_id": 44},
    ],
)
def test_reattach_decision_rejects_invalid_failed_state(
    facts: dict[str, object],
) -> None:
    assert recovery.reattach_decision(_state(RunPhase.FAILED, **facts)) is None


def test_transition_terminal_checks_the_current_phase_before_writing() -> None:
    class Store:
        def __init__(self) -> None:
            self.state = _state(RunPhase.RUNNING)
            self.calls: list[dict[str, object]] = []

        def load(self) -> SimpleNamespace:
            return self.state

        def transition(self, **kwargs: object) -> None:
            self.calls.append(kwargs)

    store = Store()
    recovery.transition_terminal(
        store,  # type: ignore[arg-type]
        expected=(RunPhase.RUNNING,),
        target=RunPhase.FAILED,
        facts={"failed_job_id": 42},
    )

    assert store.calls == [
        {
            "expected": RunPhase.RUNNING,
            "target": RunPhase.FAILED,
            "facts": {"failed_job_id": 42},
        }
    ]


def test_transition_terminal_refuses_an_unexpected_phase() -> None:
    class Store:
        def load(self) -> SimpleNamespace:
            return _state(RunPhase.COMPLETE)

        def transition(self, **_kwargs: object) -> None:
            raise AssertionError("transition must not be called")

    with pytest.raises(RuntimeError, match="unexpected durable phase"):
        recovery.transition_terminal(  # type: ignore[arg-type]
            Store(),
            expected=(RunPhase.RUNNING,),
            target=RunPhase.COMPLETE,
            facts={},
        )


@pytest.mark.parametrize(
    "facts", [{}, {"recovery_attempt": True}, {"recovery_attempt": "2"}]
)
def test_next_recovery_attempt_starts_at_one_for_invalid_facts(
    facts: dict[str, object],
) -> None:
    assert recovery.next_recovery_attempt(facts) == 1


def test_next_recovery_attempt_increments_an_existing_integer() -> None:
    assert recovery.next_recovery_attempt({"recovery_attempt": 3}) == 4
