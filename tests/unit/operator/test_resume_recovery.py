"""Recovery tests for the public ``resume RUN_ID`` control path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osm_polygon_sentence_relevance.operator import cli
from osm_polygon_sentence_relevance.operator.config import OperatorConfig
from osm_polygon_sentence_relevance.operator.oar import ExitClass
from osm_polygon_sentence_relevance.operator.sites import SiteProbe, SiteRequirements
from osm_polygon_sentence_relevance.operator.sites_availability import AvailabilityProbe
from osm_polygon_sentence_relevance.operator.ssh import SshError
from osm_polygon_sentence_relevance.operator.state import RunPhase, StateStore


def _config() -> OperatorConfig:
    return OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )


def _store_at(
    root: Path,
    phase: RunPhase,
    *,
    facts: dict[str, object],
) -> tuple[OperatorConfig, StateStore]:
    config = _config()
    store = StateStore(root)
    store.load_or_create(config.run_identity)
    chain = [
        RunPhase.INPUTS_RESOLVED,
        RunPhase.SITE_SELECTED,
        RunPhase.STORAGE_READY,
        RunPhase.REMOTE_PREPARED,
        RunPhase.SUBMITTED,
        RunPhase.QUEUED,
        RunPhase.RUNNING,
    ]
    current = RunPhase.CREATED
    for target in chain:
        if current is phase:
            break
        store.transition(expected=current, target=target, facts=facts)
        current = target
        if current is phase:
            break
    return config, store


def test_resume_rejects_malformed_and_missing_run_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    args = cli.build_parser().parse_args(["resume", "a" * 20])
    with pytest.raises(RuntimeError, match="does not exist"):
        cli._resume_run("a" * 20, args)
    with pytest.raises(RuntimeError, match="twenty lowercase"):
        cli._resume_run("INVALID", args)


def test_resume_live_job_reattaches_without_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _store = _store_at(
        tmp_path,
        RunPhase.QUEUED,
        facts={"site": "sophia", "job_id": 123, "active_stage": "label"},
    )
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    seen: list[tuple[str, int, str | None]] = []
    optimized: list[tuple[str, int]] = []

    def optimize(
        _args: Any,
        _store: Any,
        _config: Any,
        site: str,
        job_id: int,
    ) -> tuple[str, int]:
        optimized.append((site, job_id))
        return ("nancy", 456)

    def classify(
        args: Any,
        store: Any,
        config: Any,
        site: str,
        job_id: int,
        *,
        destination_site: str | None,
    ) -> ExitClass:
        del args, store, config
        seen.append((site, job_id, destination_site))
        return ExitClass.COMPLETE

    monkeypatch.setattr(cli, "_classify_or_continue", classify)
    monkeypatch.setattr(cli, "_optimize_queued_start", optimize)
    args = cli.build_parser().parse_args(["resume", config.run_id])
    assert cli._resume_run(config.run_id, args) == 0
    assert optimized == [("sophia", 123)]
    assert seen == [("nancy", 456, "nancy")]


def test_resume_prepared_continuation_validates_assets_and_submits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _store_at(
        tmp_path,
        RunPhase.REMOTE_PREPARED,
        facts={"site": "grenoble", "resume_relay_root": str(tmp_path / "relay")},
    )
    relay_root = tmp_path / "relay"
    relay_root.mkdir()
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    calls: list[str] = []

    class FakeStager:
        def prepare(self, _config: Any, _layout: Any) -> Any:
            calls.append("checkout")
            return SimpleNamespace(reused=True)

        def prepare_label_assets(
            self, _config: Any, layout: Any, *, download_input: bool
        ) -> Any:
            assert download_input is True
            calls.append("assets")
            return SimpleNamespace(
                input_parquet=layout.root / "input/sentences.parquet",
                model_file=layout.root / "model/model.gguf",
                tokenizer_dir=layout.root / "tokenizer",
                llama_server_ready=True,
            )

    class FakeController:
        def submit(self, **_kwargs: Any) -> int:
            store.transition(
                expected=RunPhase.REMOTE_PREPARED,
                target=RunPhase.SUBMITTED,
                facts={"job_id": 456, "active_stage": "label"},
            )
            calls.append("submit")
            return 456

    layout = SimpleNamespace(root=Path("/home/u/run"))
    monkeypatch.setattr(
        cli,
        "_attach_to_site",
        lambda *_a, **_kw: (
            object(),
            layout,
            object(),
            FakeController(),
        ),
    )
    monkeypatch.setattr(
        cli, "_usage_policy_preflight", lambda *_a: calls.append("policy")
    )
    monkeypatch.setattr(
        cli, "_storage_preflight", lambda *_a, **_kw: calls.append("quota")
    )
    monkeypatch.setattr(cli, "Stager", lambda _ssh: FakeStager())
    monkeypatch.setattr(
        cli,
        "_ensure_relay_at_destination",
        lambda **_kw: calls.append("relay"),
    )
    seen: list[int] = []
    monkeypatch.setattr(
        cli,
        "_classify_or_continue",
        lambda **kwargs: (seen.append(int(kwargs["job_id"])) or ExitClass.COMPLETE),
    )
    monkeypatch.setattr(
        cli,
        "_optimize_queued_start",
        lambda _args, _store, _config, site, job_id: (site, job_id),
    )
    args = cli.build_parser().parse_args(["resume", config.run_id])
    assert cli._resume_run(config.run_id, args) == 0
    assert calls == ["policy", "quota", "checkout", "assets", "relay", "submit"]
    assert seen == [456]


def test_resume_refuses_prepared_state_without_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _store = _store_at(tmp_path, RunPhase.REMOTE_PREPARED, facts={})
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    args = cli.build_parser().parse_args(["resume", config.run_id])
    with pytest.raises(RuntimeError, match="no recorded site"):
        cli._resume_run(config.run_id, args)


def test_resume_refuses_tampered_persisted_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _store = _store_at(tmp_path, RunPhase.CREATED, facts={})
    state_path = tmp_path / "runs" / config.run_id / "state.json"
    payload = json.loads(state_path.read_text())
    payload["run_identity"]["source_commit"] = "c" * 40
    state_path.write_text(json.dumps(payload))
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    args = cli.build_parser().parse_args(["resume", config.run_id])
    with pytest.raises(RuntimeError, match="does not reproduce"):
        cli._resume_run(config.run_id, args)


def test_small_cli_validation_and_fallback_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert cli._aggregate_peak_gpu(AvailabilityProbe(())) == (0, None)
    with pytest.raises(ValueError, match="non-negative"):
        cli._required_staging_headroom("label", -1)
    with pytest.raises(ValueError, match="twenty"):
        cli._probe_target("nancy", "bad")

    class BrokenSsh:
        def run(self, _command: str) -> Any:
            raise SshError("offline", category="transport", returncode=255, attempts=1)

    assert cli._queue_depth(BrokenSsh()) == 0  # type: ignore[arg-type]
    requirements = SiteRequirements(persistent_free_bytes=10)
    assert cli._storage_cleanup_can_help(
        [SiteProbe("x", "x", True, 80_000, (8, 0), 1, 0)], requirements
    )


def test_prepare_cross_site_runs_build_when_binary_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _store_at(
        tmp_path,
        RunPhase.QUEUED,
        facts={"site": "sophia", "job_id": 1},
    )
    calls: list[str] = []
    fake_ssh = object()

    class FakeStager:
        def __init__(self, ssh: object) -> None:
            assert ssh is fake_ssh

        def prepare(self, _config: Any, _layout: Any) -> None:
            calls.append("prepare")

        def prepare_label_assets(self, *_a: Any, **_kw: Any) -> Any:
            calls.append("assets")
            return SimpleNamespace(llama_server_ready=False)

    class FakeOar:
        def __init__(self, _ssh: Any, *, preflight: Any) -> None:
            preflight()
            calls.append("oar")

    monkeypatch.setattr(cli, "SshClient", lambda **_kw: fake_ssh)
    monkeypatch.setattr(cli, "_remote_home", lambda _ssh: Path("/home/u"))
    monkeypatch.setattr(
        cli, "_usage_policy_preflight", lambda *_a: calls.append("policy")
    )
    monkeypatch.setattr(
        cli, "_storage_preflight", lambda *_a, **_kw: calls.append("quota")
    )
    monkeypatch.setattr(cli, "Stager", FakeStager)
    monkeypatch.setattr(cli, "OarClient", FakeOar)
    monkeypatch.setattr(cli, "_ensure_llama_server", lambda *_a: calls.append("llama"))
    relay_root = tmp_path / "relay"
    relay_root.mkdir()
    cli._prepare_destination_for_resume(
        store=store,
        config=config,
        site="grenoble",
        relay_root=relay_root,
        poll_seconds=0,
    )
    assert store.load().phase is RunPhase.REMOTE_PREPARED
    assert calls == [
        "policy",
        "quota",
        "prepare",
        "assets",
        "policy",
        "quota",
        "oar",
        "llama",
    ]


def test_ensure_relay_refuses_disappeared_generation(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="disappeared"):
        cli._ensure_relay_at_destination(
            store=object(),  # type: ignore[arg-type]
            config=_config(),
            site="nancy",
            layout=SimpleNamespace(),
            relay_root=tmp_path / "missing",
        )


def test_resume_idle_run_reports_nothing_to_reattach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _store = _store_at(tmp_path, RunPhase.CREATED, facts={})
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    args = cli.build_parser().parse_args(["resume", config.run_id])
    assert cli._resume_run(config.run_id, args) == 0
    assert "nothing to reattach" in capsys.readouterr().out
