"""Durable execution lanes for worldwide V2 smoke and production labels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import PurePosixPath

from osm_polygon_sentence_relevance.labeling.v2_contracts import (
    V2_LOGIT_PROMPT_VERSION,
)
from osm_polygon_sentence_relevance.operator.config import OperatorConfig, Scope


class LabelLane(StrEnum):
    """One isolated V2 labeling checkpoint and output namespace."""

    SMOKE = "smoke"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class LabelLanePlan:
    """Runtime identity and paths for one lane inside a parent operator run."""

    lane: LabelLane
    config: OperatorConfig
    parent_run_id: str
    work_dir: PurePosixPath
    output_dir: PurePosixPath
    checkpoint_namespace: str
    publishes: bool


def _lane(config: OperatorConfig, facts: Mapping[str, object]) -> LabelLane:
    recorded = facts.get("label_lane")
    if recorded is None:
        return (
            LabelLane.SMOKE
            if config.requirements.row_limit > 0
            else LabelLane.PRODUCTION
        )
    if not isinstance(recorded, str):
        raise ValueError("persisted label lane is invalid")
    try:
        return LabelLane(recorded)
    except ValueError as exc:
        raise ValueError("persisted label lane is invalid") from exc


def label_lane_plan(
    config: OperatorConfig,
    root: PurePosixPath,
    facts: Mapping[str, object],
) -> LabelLanePlan:
    """Return the isolated V2 lane selected by durable run facts.

    The parent run identity remains bound to the already-computed split
    checkpoints. Only the nested labeling identity changes: the smoke keeps
    the requested positive ``row_limit`` while production always uses the
    complete deterministic ``sampling_target`` with ``row_limit=0``.
    """

    if (
        config.scope is not Scope.ALL
        or config.prompt_version != V2_LOGIT_PROMPT_VERSION
    ):
        raise ValueError("label lanes are only available for worldwide V2 runs")
    lane = _lane(config, facts)
    row_limit = config.requirements.row_limit if lane is LabelLane.SMOKE else 0
    lane_config = replace(
        config,
        requirements=replace(config.requirements, row_limit=row_limit),
    )
    is_smoke = lane is LabelLane.SMOKE
    work_name = "label-smoke-work" if is_smoke else "label-work"
    output_name = "label-smoke-output" if is_smoke else "label-output"
    return LabelLanePlan(
        lane=lane,
        config=lane_config,
        parent_run_id=config.run_id,
        work_dir=root / work_name,
        output_dir=root / output_name,
        checkpoint_namespace=f"checkpoints/{config.run_id}/{lane.value}",
        publishes=not is_smoke,
    )


__all__ = ["LabelLane", "LabelLanePlan", "label_lane_plan"]
