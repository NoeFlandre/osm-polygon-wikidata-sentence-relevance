"""Sampling policy and continuation contracts for the operator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator.config import OperatorConfig
from osm_polygon_sentence_relevance.operator.sampling import (
    sampling_target_for_run,
    sync_sampling_target,
)
from osm_polygon_sentence_relevance.operator.state import RunPhase, StateStore


def _args(
    *,
    scope: str = "region",
    stage: str = "label",
    sampling_target: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        scope=scope,
        stage=stage,
        sampling_target=sampling_target,
    )


def _config(*, target: int | None) -> OperatorConfig:
    return OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="a" * 40,
        input_revision="b" * 40,
        sampling_target=target,
    )


def test_worldwide_labeling_defaults_to_the_configured_target() -> None:
    assert sampling_target_for_run(_args(scope="all")) == 200_000


def test_worldwide_labeling_accepts_a_positive_target() -> None:
    assert (
        sampling_target_for_run(_args(scope="all", sampling_target=400_000)) == 400_000
    )


@pytest.mark.parametrize("target", [0, -1])
def test_worldwide_labeling_rejects_non_positive_targets(target: int) -> None:
    with pytest.raises(ValueError, match="positive sampling target"):
        sampling_target_for_run(_args(scope="all", sampling_target=target))


def test_regional_labeling_rejects_positive_sampling() -> None:
    with pytest.raises(ValueError, match="scope all"):
        sampling_target_for_run(_args(sampling_target=200_000))


def test_non_label_commands_preserve_the_requested_target() -> None:
    assert sampling_target_for_run(_args(stage="split", sampling_target=None)) is None
    assert sampling_target_for_run(_args(stage="split", sampling_target=0)) == 0


def test_sync_sampling_target_records_and_expands_target(tmp_path: Path) -> None:
    config = _config(target=200_000)
    store = StateStore(tmp_path)
    store.load_or_create(config.run_identity)

    sync_sampling_target(store, config)
    assert store.load().facts["sampling_target"] == 200_000

    expanded = _config(target=400_000)
    sync_sampling_target(store, expanded)
    assert store.load().facts["sampling_target"] == 400_000


def test_sync_sampling_target_rejects_a_shrink(tmp_path: Path) -> None:
    config = _config(target=200_000)
    store = StateStore(tmp_path)
    store.load_or_create(config.run_identity)
    sync_sampling_target(store, config)

    with pytest.raises(RuntimeError, match="cannot decrease"):
        sync_sampling_target(store, _config(target=100_000))


def test_sync_sampling_target_reopens_completed_run_on_expansion(
    tmp_path: Path,
) -> None:
    config = _config(target=200_000)
    store = StateStore(tmp_path)
    store.load_or_create(config.run_identity)
    sync_sampling_target(store, config)
    store.transition(expected=RunPhase.CREATED, target=RunPhase.COMPLETE)

    sync_sampling_target(store, _config(target=400_000))

    state = store.load()
    assert state.phase is RunPhase.REMOTE_PREPARED
    assert state.facts["continued_from_sampling_target"] == 200_000


def test_sync_sampling_target_rejects_invalid_persisted_target(
    tmp_path: Path,
) -> None:
    config = _config(target=200_000)
    store = StateStore(tmp_path)
    store.load_or_create(config.run_identity)
    store.transition(
        expected=RunPhase.CREATED,
        target=RunPhase.CREATED,
        facts={"sampling_target": 0},
    )

    with pytest.raises(RuntimeError, match="persisted sampling target"):
        sync_sampling_target(store, config)
