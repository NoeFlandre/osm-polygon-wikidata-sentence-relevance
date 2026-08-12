"""Contracts for split finalization orchestration."""

from __future__ import annotations

from pathlib import PurePosixPath
from types import SimpleNamespace

from osm_polygon_sentence_relevance.operator import split_finalization
from osm_polygon_sentence_relevance.operator.config import OperatorConfig, Stage
from osm_polygon_sentence_relevance.operator.state import RunPhase
from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout


def test_finalize_split_checkpointed_delegates_the_existing_lifecycle(
    monkeypatch,
) -> None:
    """The extracted entry point keeps the existing finalizer seam intact."""

    config = OperatorConfig.build(
        scope="all",
        stage="split",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )

    class FakeStore:
        value = SimpleNamespace(phase=RunPhase.CHECKPOINTED, facts={})

        def transition(self, **kwargs: object) -> None:
            self.value.phase = kwargs["target"]
            self.value.facts.update(kwargs.get("facts", {}))

    class FakeOar:
        def submit(self, _request: object) -> int:
            return 1

    store = FakeStore()
    ssh = object()
    layout = RemoteLayout(PurePosixPath("/run"))
    oar = FakeOar()
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        split_finalization,
        "split_finalization_submission",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        split_finalization,
        "monitor_job_with_log",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        split_finalization,
        "assert_remote_exit_zero",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(split_finalization, "publish_split", lambda *args: "c" * 40)
    monkeypatch.setattr(
        split_finalization,
        "mark_remote_status",
        lambda *args: calls.append((args, {})),
    )

    result = split_finalization.finalize_split_checkpointed(
        store=store,
        config=config,
        ssh=ssh,
        layout=layout,
        oar=oar,
        poll_seconds=0.0,
    )

    assert result == 1
    assert len(calls) == 3


def test_finalize_split_checkpointed_reopens_all_stage_for_labeling(
    monkeypatch,
) -> None:
    """The all-stage path preserves the split output before labeling."""

    config = OperatorConfig.build(
        scope="all",
        stage="all",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )

    class FakeStore:
        value = SimpleNamespace(phase=RunPhase.CHECKPOINTED, facts={})

        def transition(self, **kwargs: object) -> None:
            self.value.phase = kwargs["target"]
            self.value.facts.update(kwargs.get("facts", {}))

    class FakeOar:
        def submit(self, _request: object) -> int:
            return 1

    monkeypatch.setattr(
        split_finalization, "split_finalization_submission", lambda *_: object()
    )
    monkeypatch.setattr(
        split_finalization, "monitor_job_with_log", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        split_finalization, "assert_remote_exit_zero", lambda *_a, **_k: None
    )

    store = FakeStore()
    split_finalization.finalize_split_checkpointed(
        store=store,
        config=config,
        ssh=object(),
        layout=RemoteLayout(PurePosixPath("/run")),
        oar=FakeOar(),
        poll_seconds=0.0,
    )

    assert store.value.phase is RunPhase.REMOTE_PREPARED
    assert store.value.facts["active_stage"] == Stage.LABEL.value
