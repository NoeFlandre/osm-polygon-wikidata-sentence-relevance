"""Durable state decisions used while recovering Grid'5000 allocations."""

from __future__ import annotations

from collections.abc import Mapping

from osm_polygon_sentence_relevance.operator.state import RunPhase, RunState, StateStore


def transition_terminal(
    state: StateStore,
    *,
    expected: tuple[RunPhase, ...],
    target: RunPhase,
    facts: Mapping[str, object],
) -> None:
    """Validate the durable phase before applying a terminal transition."""

    current = state.load()
    if current.phase not in expected:
        raise RuntimeError("operator reached an unexpected durable phase")
    state.transition(expected=current.phase, target=target, facts=facts)


def reattach_decision(state: RunState) -> tuple[str, int] | None:
    """Return a recorded allocation candidate when recovery is safe.

    The caller still performs read-only OAR inspection and terminal evidence
    classification. A previously recovered failed allocation is excluded so
    repeated resume commands cannot submit the same recovery twice.
    """

    if state.phase in {
        RunPhase.SUBMITTED,
        RunPhase.QUEUED,
        RunPhase.RUNNING,
    }:
        job_id = state.facts.get("job_id")
        site = state.facts.get("site")
        if type(job_id) is not int or job_id <= 0:
            return None
        if not isinstance(site, str) or not site:
            return None
        return site, job_id
    if state.phase is RunPhase.FAILED:
        failed_job_id = state.facts.get("failed_job_id")
        site = state.facts.get("site")
        if type(failed_job_id) is not int or failed_job_id <= 0:
            return None
        if not isinstance(site, str) or not site:
            return None
        recovered_from = state.facts.get("recovered_from_job_id")
        if (
            isinstance(recovered_from, int)
            and not isinstance(recovered_from, bool)
            and recovered_from == failed_job_id
        ):
            return None
        return site, failed_job_id
    return None


def next_recovery_attempt(facts: Mapping[str, object]) -> int:
    """Return the next monotonic recovery-attempt number."""

    existing = facts.get("recovery_attempt")
    if isinstance(existing, bool) or not isinstance(existing, int):
        return 1
    return existing + 1


__all__ = ["next_recovery_attempt", "reattach_decision", "transition_terminal"]
