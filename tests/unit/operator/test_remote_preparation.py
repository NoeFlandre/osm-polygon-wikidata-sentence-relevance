"""Contract tests for remote continuation preparation."""

from __future__ import annotations

from pathlib import PurePosixPath
from types import SimpleNamespace

from osm_polygon_sentence_relevance.operator import cli, remote_preparation
from osm_polygon_sentence_relevance.operator.config import OperatorConfig
from osm_polygon_sentence_relevance.operator.state import RunPhase


def _config() -> OperatorConfig:
    return OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )


def test_prepare_destination_exposes_module_contract() -> None:
    assert callable(remote_preparation.prepare_destination)


def test_prepare_destination_same_site_refreshes_and_preserves_phase() -> None:
    config = _config()
    state = SimpleNamespace(
        phase=RunPhase.REMOTE_PREPARED,
        facts={"active_stage": "split"},
    )
    transitions: list[dict[str, object]] = []
    calls: list[str] = []

    class Stager:
        def __init__(self, _ssh: object) -> None:
            pass

        def prepare(self, _config: object, _layout: object) -> None:
            calls.append("prepare")

    services = remote_preparation.RemotePreparationServices(
        ssh_factory=lambda **_kwargs: object(),
        remote_home=lambda _ssh: PurePosixPath("/home/u"),
        usage_policy_preflight=lambda *_args: calls.append("policy"),
        ensure_home_headroom=lambda *_args, **_kwargs: calls.append("quota"),
        stager_type=Stager,
        stage_hf_token=lambda *_args: calls.append("token"),
        oar_type=object,
        ensure_llama_server=lambda *_args: 0,
        label_staging_headroom_bytes=100,
        submission_headroom_bytes=200,
    )

    def transition(**kwargs: object) -> None:
        transitions.append(kwargs)

    store = SimpleNamespace(load=lambda: state, transition=transition)
    remote_preparation.prepare_destination(
        store=store,  # type: ignore[arg-type]
        config=config,
        site="sophia",
        relay_root=None,
        poll_seconds=0,
        services=services,
    )

    assert calls == ["policy", "quota", "prepare", "token"]
    assert transitions == []


def test_prepare_destination_reuses_submission_headroom_after_asset_staging() -> None:
    config = _config()
    state = SimpleNamespace(
        phase=RunPhase.REMOTE_PREPARED,
        facts={"active_stage": "label", "llama_build_job_id": 42},
    )
    headroom: list[int] = []

    class Stager:
        def __init__(self, _ssh: object) -> None:
            pass

        def prepare(self, _config: object, _layout: object) -> None:
            pass

    services = remote_preparation.RemotePreparationServices(
        ssh_factory=lambda **_kwargs: object(),
        remote_home=lambda _ssh: PurePosixPath("/home/u"),
        usage_policy_preflight=lambda *_args: None,
        ensure_home_headroom=lambda *_args, **kwargs: headroom.append(
            kwargs["minimum_headroom_bytes"]
        ),
        stager_type=Stager,
        stage_hf_token=lambda *_args: None,
        oar_type=object,
        ensure_llama_server=lambda *_args: 0,
        label_staging_headroom_bytes=100,
        submission_headroom_bytes=200,
    )
    store = SimpleNamespace(load=lambda: state, transition=lambda **_kwargs: None)

    remote_preparation.prepare_destination(
        store=store,  # type: ignore[arg-type]
        config=config,
        site="grenoble",
        relay_root=None,
        poll_seconds=0,
        services=services,
    )

    assert headroom == [200]


def test_cli_adapter_delegates_to_remote_preparation(monkeypatch) -> None:
    sentinel_services = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "_remote_preparation_services", lambda: sentinel_services)

    def fake_prepare(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(remote_preparation, "prepare_destination", fake_prepare)
    store = object()
    config = object()
    relay_root = PurePosixPath("/relay")

    cli._prepare_destination_for_resume(
        store=store,  # type: ignore[arg-type]
        config=config,  # type: ignore[arg-type]
        site="nancy",
        relay_root=relay_root,
        poll_seconds=2.5,
    )

    assert captured == {
        "store": store,
        "config": config,
        "site": "nancy",
        "relay_root": relay_root,
        "poll_seconds": 2.5,
        "services": sentinel_services,
    }
