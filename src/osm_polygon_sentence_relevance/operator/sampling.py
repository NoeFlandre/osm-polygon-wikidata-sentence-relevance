"""Sampling policy and durable target continuation for operator runs."""

from __future__ import annotations

from types import SimpleNamespace

from osm_polygon_sentence_relevance.operator.config import (
    DEFAULT_SAMPLING_TARGET,
    OperatorConfig,
    Scope,
    Stage,
)
from osm_polygon_sentence_relevance.operator.state import RunPhase, StateStore


def sync_sampling_target(store: StateStore, config: OperatorConfig) -> None:
    """Persist a V2 target and reject attempts to shrink an existing run."""

    target = config.requirements.sampling_target
    if target is None or config.run_identity.sampling_version is None:
        return
    current = store.load()
    recorded = current.facts.get("sampling_target")
    expanded = False
    if recorded is not None:
        if isinstance(recorded, bool) or not isinstance(recorded, int) or recorded < 1:
            raise RuntimeError("persisted sampling target is invalid")
        if target < recorded:
            raise RuntimeError(
                "sampling target cannot decrease after checkpoints exist; "
                f"use at least {recorded}"
            )
        expanded = target > recorded
    if recorded != target:
        if expanded and current.phase is RunPhase.COMPLETE:
            store.transition(
                expected=RunPhase.COMPLETE,
                target=RunPhase.REMOTE_PREPARED,
                facts={
                    "sampling_target": target,
                    "continued_from_sampling_target": recorded,
                },
            )
            return
        store.transition(
            expected=current.phase,
            target=current.phase,
            facts={"sampling_target": target},
        )


def sampling_target_for_run(args: SimpleNamespace) -> int | None:
    """Resolve the release-safe sampling default for a production command.

    The historical regional command remains an unsampled V1 workflow. The
    worldwide ``all`` label command opts into the V2 stratified selector by
    default. An explicit positive target on a regional run is rejected rather
    than silently publishing a worldwide lane with regional data.
    """

    requested = getattr(args, "sampling_target", None)
    if args.stage == Stage.LABEL.value:
        if args.scope == Scope.ALL.value:
            target = DEFAULT_SAMPLING_TARGET if requested is None else requested
            if target is None or target <= 0:
                raise ValueError(
                    "worldwide labeling requires a positive sampling target"
                )
            return target
        if requested not in (None, 0):
            raise ValueError("stratified sampling requires --scope all")
    return requested


__all__ = ["sampling_target_for_run", "sync_sampling_target"]
