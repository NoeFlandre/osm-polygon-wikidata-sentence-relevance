"""Contract tests for durable allocation classification transitions."""

from __future__ import annotations

from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator import classification, cli
from osm_polygon_sentence_relevance.operator.config import OperatorConfig
from osm_polygon_sentence_relevance.operator.label_lanes import label_lane_plan
from osm_polygon_sentence_relevance.operator.oar import ExitClass
from osm_polygon_sentence_relevance.operator.state import RunPhase
from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout


def test_classification_module_exposes_apply_classification() -> None:
    assert callable(classification.apply_classification)


def test_cli_adapter_delegates_to_classification_module(monkeypatch) -> None:
    """Keep the public CLI seam thin while preserving dependency injection."""

    sentinel_services = object()
    captured: dict[str, object] = {}

    def fake_apply(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli, "_classification_services", lambda: sentinel_services)
    monkeypatch.setattr(classification, "apply_classification", fake_apply)

    store = object()
    config = object()
    ssh = object()
    layout = object()
    cli._apply_classification(
        store=store,  # type: ignore[arg-type]
        config=config,  # type: ignore[arg-type]
        ssh=ssh,  # type: ignore[arg-type]
        layout=layout,  # type: ignore[arg-type]
        job_id=42,
        active_stage="label",
        classification=ExitClass.CONTINUE,
        resume_artifact_path="/tmp/checkpoints",
        failure_reason_token="walltime",
        label_plan=None,
    )

    assert captured == {
        "store": store,
        "config": config,
        "ssh": ssh,
        "layout": layout,
        "job_id": 42,
        "active_stage": "label",
        "classification": ExitClass.CONTINUE,
        "resume_artifact_path": "/tmp/checkpoints",
        "failure_reason_token": "walltime",
        "label_plan": None,
        "services": sentinel_services,
    }


def _config(*, row_limit: int = 0) -> OperatorConfig:
    return OperatorConfig.build(
        scope="all",
        stage="label",
        source_commit="a" * 40,
        input_revision="b" * 40,
        row_limit=row_limit,
        sampling_target=200_000,
    )


def _store(
    phase: RunPhase, facts: dict[str, object]
) -> tuple[object, list[dict[str, object]]]:
    state = SimpleNamespace(phase=phase, facts=facts)
    transitions: list[dict[str, object]] = []

    def transition(**kwargs: object) -> None:
        transitions.append(kwargs)

    return SimpleNamespace(load=lambda: state, transition=transition), transitions


def _services(
    calls: list[tuple[str, object]],
) -> classification.ClassificationServices:
    def transition_terminal(_store: object, **kwargs: object) -> None:
        calls.append(("transition_terminal", kwargs))

    def next_attempt(_facts: object) -> int:
        calls.append(("next_recovery_attempt", None))
        return 3

    return classification.ClassificationServices(
        next_recovery_attempt=next_attempt,
        transition_terminal=transition_terminal,
        preserve_label=lambda *_args: PurePosixPath("/preserved"),
        preserve_manual_eval=lambda *_args, **_kwargs: PurePosixPath("/manual.jsonl"),
        label_publication_commit=lambda *_args: "c" * 40,
        publish_label=lambda *_args, **_kwargs: "d" * 40,
        mark_remote_status=lambda *_args: calls.append(("mark_remote_status", None)),
    )


def test_failed_smoke_completion_records_recovery_attempt() -> None:
    config = _config(row_limit=8)
    store, transitions = _store(
        RunPhase.FAILED,
        {"active_stage": "label", "label_lane": "smoke"},
    )
    calls: list[tuple[str, object]] = []
    plan = label_lane_plan(config, PurePosixPath("/run"), {"label_lane": "smoke"})

    classification.apply_classification(
        store=store,  # type: ignore[arg-type]
        config=config,
        ssh=object(),  # type: ignore[arg-type]
        layout=RemoteLayout(PurePosixPath("/run")),
        job_id=7,
        active_stage="label",
        classification=ExitClass.COMPLETE,
        label_plan=plan,
        services=_services(calls),
    )

    _name, transition = calls[1]
    assert isinstance(transition, dict)
    facts = transition["facts"]
    assert isinstance(facts, dict)
    assert facts["recovery_attempt"] == 3
    assert transitions[0]["target"] is RunPhase.REMOTE_PREPARED


def test_completed_label_without_publication_records_no_hub_commit() -> None:
    config = _config(row_limit=8)
    store, transitions = _store(RunPhase.RUNNING, {"active_stage": "label"})
    calls: list[tuple[str, object]] = []

    classification.apply_classification(
        store=store,  # type: ignore[arg-type]
        config=config,
        ssh=object(),  # type: ignore[arg-type]
        layout=RemoteLayout(PurePosixPath("/run")),
        job_id=8,
        active_stage="label",
        classification=ExitClass.COMPLETE,
        services=_services(calls),
    )

    _name, first_transition = calls[0]
    assert isinstance(first_transition, dict)
    facts = first_transition["facts"]
    assert isinstance(facts, dict)
    assert "hub_commit" not in facts
    assert transitions[-1]["target"] is RunPhase.COMPLETE


def test_publication_fallback_propagates_unexpected_error() -> None:
    config = _config()
    store, _transitions = _store(RunPhase.RUNNING, {"active_stage": "label"})
    services = _services([])

    def fail_commit(*_args: object) -> str:
        raise RuntimeError("unexpected publication error")

    services = classification.ClassificationServices(
        next_recovery_attempt=services.next_recovery_attempt,
        transition_terminal=services.transition_terminal,
        preserve_label=services.preserve_label,
        preserve_manual_eval=services.preserve_manual_eval,
        label_publication_commit=fail_commit,
        publish_label=services.publish_label,
        mark_remote_status=services.mark_remote_status,
    )
    with pytest.raises(RuntimeError, match="unexpected publication error"):
        classification.apply_classification(
            store=store,  # type: ignore[arg-type]
            config=config,
            ssh=object(),  # type: ignore[arg-type]
            layout=RemoteLayout(PurePosixPath("/run")),
            job_id=9,
            active_stage="label",
            classification=ExitClass.COMPLETE,
            services=services,
        )


@pytest.mark.parametrize("phase", [RunPhase.RUNNING, RunPhase.FAILED])
def test_completed_split_transitions_to_checkpointed(phase: RunPhase) -> None:
    config = _config()
    store, transitions = _store(phase, {"active_stage": "split"})
    calls: list[tuple[str, object]] = []

    classification.apply_classification(
        store=store,  # type: ignore[arg-type]
        config=config,
        ssh=object(),  # type: ignore[arg-type]
        layout=RemoteLayout(PurePosixPath("/run")),
        job_id=10,
        active_stage="split",
        classification=ExitClass.COMPLETE,
        services=_services(calls),
    )

    _name, transition = calls[-1]
    assert isinstance(transition, dict)
    assert transition["target"] is RunPhase.CHECKPOINTED
    if phase is RunPhase.FAILED:
        facts = transition["facts"]
        assert isinstance(facts, dict)
        assert facts["recovery_attempt"] == 3


def test_unknown_exit_class_is_rejected() -> None:
    config = _config()
    store, _transitions = _store(RunPhase.RUNNING, {"active_stage": "split"})
    with pytest.raises(RuntimeError, match="unhandled exit class"):
        classification.apply_classification(
            store=store,  # type: ignore[arg-type]
            config=config,
            ssh=object(),  # type: ignore[arg-type]
            layout=RemoteLayout(PurePosixPath("/run")),
            job_id=11,
            active_stage="split",
            classification=ExitClass.CANCELLED,
            services=_services([]),
        )
