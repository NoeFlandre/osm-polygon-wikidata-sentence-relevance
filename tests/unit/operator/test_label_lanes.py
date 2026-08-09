"""Contracts for autonomous V2 smoke and production label lanes."""

from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from osm_polygon_sentence_relevance.operator.config import OperatorConfig
from osm_polygon_sentence_relevance.operator.label_lanes import (
    LabelLane,
    label_lane_plan,
)


def _config(*, row_limit: int = 128) -> OperatorConfig:
    return OperatorConfig.build(
        scope="all",
        stage="all",
        source_commit="a" * 40,
        input_revision="b" * 40,
        row_limit=row_limit,
        sampling_target=200_000,
    )


def test_v2_canary_plan_preserves_parent_run_and_uses_isolated_paths() -> None:
    config = _config(row_limit=128)

    plan = label_lane_plan(config, PurePosixPath("/runs") / config.run_id, {})

    assert plan.lane is LabelLane.SMOKE
    assert plan.config.requirements.row_limit == 128
    assert plan.config.requirements.sampling_target == 200_000
    assert plan.parent_run_id == config.run_id
    assert config.requirements.row_limit == 128
    assert plan.work_dir == PurePosixPath("/runs") / config.run_id / "label-smoke-work"
    assert plan.output_dir == (
        PurePosixPath("/runs") / config.run_id / "label-smoke-output"
    )
    assert plan.checkpoint_namespace == f"checkpoints/{config.run_id}/smoke"
    assert plan.publishes is False


def test_v2_production_plan_uses_full_target_without_changing_parent_identity() -> None:
    config = _config(row_limit=128)

    plan = label_lane_plan(
        config,
        PurePosixPath("/runs") / config.run_id,
        {"label_lane": "production", "smoke_completed": True},
    )

    assert plan.lane is LabelLane.PRODUCTION
    assert plan.config.requirements.row_limit == 0
    assert plan.config.requirements.sampling_target == 200_000
    assert plan.parent_run_id == config.run_id
    assert plan.config.run_id != config.run_id
    assert config.requirements.row_limit == 128
    assert plan.work_dir == PurePosixPath("/runs") / config.run_id / "label-work"
    assert plan.output_dir == PurePosixPath("/runs") / config.run_id / "label-output"
    assert plan.checkpoint_namespace == f"checkpoints/{config.run_id}/production"
    assert plan.publishes is True


def test_v2_without_canary_starts_directly_in_production() -> None:
    config = _config(row_limit=0)

    plan = label_lane_plan(config, PurePosixPath("/r"), {})

    assert plan.lane is LabelLane.PRODUCTION
    assert plan.config.requirements.row_limit == 0
    assert plan.publishes is True


@pytest.mark.parametrize("value", ["", "SMOKE", "other", 1, True])
def test_v2_lane_rejects_invalid_persisted_values(value: object) -> None:
    with pytest.raises(ValueError, match="label lane"):
        label_lane_plan(_config(), PurePosixPath("/r"), {"label_lane": value})


def test_label_lane_plan_refuses_non_worldwide_workflows() -> None:
    config = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="a" * 40,
        input_revision="b" * 40,
        row_limit=128,
        sampling_target=None,
    )

    with pytest.raises(ValueError, match="worldwide V2"):
        label_lane_plan(config, PurePosixPath("/r"), {})
