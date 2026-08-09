"""Recovery tests for the public ``resume RUN_ID`` control path."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest

from osm_polygon_sentence_relevance.operator import cli
from osm_polygon_sentence_relevance.operator.config import OperatorConfig
from osm_polygon_sentence_relevance.operator.oar import ExitClass
from osm_polygon_sentence_relevance.operator.staging import LabelAssets
from osm_polygon_sentence_relevance.operator.state import RunPhase, StateStore


def _resume_args(run_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        command="resume",
        run_id=run_id,
        site=list(cli.DEFAULT_SITES),
        gpu_memory_mb=40_000,
        poll_seconds=30.0,
    )


def _config(*, stage: str = "label") -> OperatorConfig:
    return OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage=stage,
        source_commit="a" * 40,
        input_revision="b" * 40,
    )


def _store_at(
    root: Path,
    phase: RunPhase,
    *,
    facts: dict[str, object],
    stage: str = "label",
) -> tuple[OperatorConfig, StateStore]:
    config = _config(stage=stage)
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
    args = _resume_args("a" * 20)
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
    args = _resume_args(config.run_id)
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
            return LabelAssets(
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
        cli, "ensure_home_headroom", lambda *_a, **_kw: calls.append("quota")
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
    args = _resume_args(config.run_id)
    assert cli._resume_run(config.run_id, args) == 0
    assert calls == ["policy", "quota", "checkout", "assets", "relay", "submit"]
    assert seen == [456]


def test_resume_prepared_stage_all_reuses_finalized_split_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = _store_at(
        tmp_path,
        RunPhase.REMOTE_PREPARED,
        facts={
            "site": "grenoble",
            "split_output_job_id": 789,
        },
        stage="all",
    )
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_git_head", lambda: config.source_commit)
    calls: list[object] = []

    class FakeStager:
        def prepare(self, _config: Any, _layout: Any) -> Any:
            calls.append("checkout")
            return SimpleNamespace(reused=True)

        def prepare_label_assets(
            self, _config: Any, layout: Any, *, download_input: bool
        ) -> Any:
            calls.append(("assets", download_input))
            return LabelAssets(
                input_parquet=layout.root / "input/sentences.parquet",
                model_file=layout.root / "model/model.gguf",
                tokenizer_dir=layout.root / "tokenizer",
                llama_server_ready=True,
            )

    class FakeSsh:
        def run(self, command: str) -> SimpleNamespace:
            calls.append(command)
            return SimpleNamespace(stdout="")

    class FakeController:
        def submit(self, **kwargs: Any) -> int:
            calls.append(kwargs["input_parquet"])
            store.transition(
                expected=RunPhase.REMOTE_PREPARED,
                target=RunPhase.SUBMITTED,
                facts={"job_id": 456, "active_stage": "label"},
            )
            return 456

    layout = SimpleNamespace(
        root=Path("/home/u/run"),
        logs=Path("/home/u/run/logs"),
    )
    monkeypatch.setattr(
        cli,
        "_attach_to_site",
        lambda *_a, **_kw: (FakeSsh(), layout, object(), FakeController()),
    )
    monkeypatch.setattr(cli, "_usage_policy_preflight", lambda *_a: None)
    monkeypatch.setattr(cli, "ensure_home_headroom", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "Stager", lambda _ssh: FakeStager())
    monkeypatch.setattr(
        cli, "_classify_or_continue", lambda **_kwargs: ExitClass.COMPLETE
    )
    monkeypatch.setattr(
        cli,
        "_optimize_queued_start",
        lambda _args, _store, _config, site, job_id: (site, job_id),
    )

    args = _resume_args(config.run_id)
    assert cli._resume_run(config.run_id, args) == 0
    assert ("assets", False) in calls
    assert Path("/home/u/run/logs/789/output/sentences.parquet") in calls


def test_resume_prepared_v2_production_submits_full_lane_without_resplitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = OperatorConfig.build(
        scope="all",
        stage="all",
        source_commit="a" * 40,
        input_revision="b" * 40,
        row_limit=128,
        sampling_target=200_000,
    )
    store = StateStore(tmp_path)
    store.load_or_create(config.run_identity)
    facts = {
        "site": "sophia",
        "split_output_job_id": 789,
        "active_stage": "label",
        "label_lane": "production",
        "smoke_completed": True,
    }
    phase = RunPhase.CREATED
    for target in (
        RunPhase.INPUTS_RESOLVED,
        RunPhase.SITE_SELECTED,
        RunPhase.STORAGE_READY,
        RunPhase.REMOTE_PREPARED,
    ):
        store.transition(expected=phase, target=target, facts=facts)
        phase = target
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_git_head", lambda: config.source_commit)
    submitted: list[dict[str, object]] = []

    class FakeStager:
        def prepare(self, _config: object, _layout: object) -> object:
            return SimpleNamespace(reused=True)

        def prepare_v2_input(
            self, _config: object, layout: object, source: PurePosixPath
        ) -> PurePosixPath:
            assert source == PurePosixPath(
                "/home/u/run/logs/789/output/sentences.parquet"
            )
            return layout.v2_input  # type: ignore[no-any-return,union-attr]

        def prepare_label_assets(
            self, _config: object, layout: object, *, download_input: bool
        ) -> LabelAssets:
            assert download_input is False
            return LabelAssets(
                input_parquet=layout.v2_input,  # type: ignore[union-attr]
                model_file=layout.root / "model/model.gguf",  # type: ignore[union-attr]
                tokenizer_dir=layout.root / "tokenizer",  # type: ignore[union-attr]
                llama_server_ready=True,
            )

    class FakeController:
        def submit(self, **kwargs: object) -> int:
            submitted.append(kwargs)
            store.transition(
                expected=RunPhase.REMOTE_PREPARED,
                target=RunPhase.SUBMITTED,
                facts={
                    "job_id": 456,
                    "active_stage": "label",
                    "label_lane": "production",
                },
            )
            return 456

    layout = cli.RemoteLayout(PurePosixPath("/home/u/run"))
    monkeypatch.setattr(
        cli,
        "_attach_to_site",
        lambda *_args, **_kwargs: (object(), layout, object(), FakeController()),
    )
    monkeypatch.setattr(cli, "_usage_policy_preflight", lambda *_args: None)
    monkeypatch.setattr(cli, "ensure_home_headroom", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "Stager", lambda _ssh: FakeStager())
    monkeypatch.setattr(
        cli, "_classify_or_continue", lambda **_kwargs: ExitClass.COMPLETE
    )
    monkeypatch.setattr(
        cli,
        "_optimize_queued_start",
        lambda _args, _store, _config, site, job_id: (site, job_id),
    )

    assert cli._resume_run(config.run_id, _resume_args(config.run_id)) == 0

    assert len(submitted) == 1
    plan = submitted[0]["label_plan"]
    assert plan.lane.value == "production"  # type: ignore[union-attr]
    assert plan.config.requirements.row_limit == 0  # type: ignore[union-attr]
    assert plan.work_dir == PurePosixPath("/home/u/run/label-work")  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("lane", "expected_work", "expected_row_limit"),
    [
        ("smoke", "label-smoke-work", 128),
        ("production", "label-work", 0),
    ],
)
def test_cross_site_relay_uses_current_v2_lane_identity_and_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane: str,
    expected_work: str,
    expected_row_limit: int,
) -> None:
    config = OperatorConfig.build(
        scope="all",
        stage="all",
        source_commit="a" * 40,
        input_revision="b" * 40,
        row_limit=128,
        sampling_target=200_000,
    )

    class FakeStore:
        value = SimpleNamespace(
            phase=RunPhase.REMOTE_PREPARED,
            facts={"active_stage": "label", "label_lane": lane},
        )

        def load(self) -> SimpleNamespace:
            return self.value

        def transition(self, **kwargs: object) -> None:
            assert kwargs["expected"] is self.value.phase
            assert kwargs["target"] is self.value.phase

    captured: dict[str, object] = {}
    inventory = SimpleNamespace(root=tmp_path / "relay")
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(cli, "SshClient", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "_remote_home", lambda _ssh: PurePosixPath("/home/u"))

    def retrieve(**kwargs: object) -> object:
        captured["source"] = kwargs
        return inventory

    def stage(**kwargs: object) -> None:
        captured["destination"] = kwargs

    monkeypatch.setattr(cli.relay, "retrieve_to_seagate", retrieve)
    monkeypatch.setattr(cli.relay, "stage_to_destination", stage)

    assert cli._relay_for_continuation(
        store=FakeStore(),  # type: ignore[arg-type]
        config=config,
        source_site="sophia",
        destination_site="grenoble",
    ) == str(inventory.root)

    source = captured["source"]
    destination = captured["destination"]
    assert isinstance(source, dict)
    assert isinstance(destination, dict)
    expected_root = f"/home/u/osm-polygon-operator/{config.run_id}/{expected_work}"
    assert source["source_checkpoint_root"] == expected_root
    assert destination["destination_checkpoint_root"] == expected_root
    identity = source["expected_run_identity"]
    assert isinstance(identity, dict)
    assert identity["row_limit"] == expected_row_limit


def test_resume_refuses_prepared_state_without_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _store = _store_at(tmp_path, RunPhase.REMOTE_PREPARED, facts={})
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    args = _resume_args(config.run_id)
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
    args = _resume_args(config.run_id)
    with pytest.raises(RuntimeError, match="does not reproduce"):
        cli._resume_run(config.run_id, args)


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

    llama_calls: list[dict[str, Any]] = []

    def mock_ensure_llama_server(
        ssh_arg: Any,
        oar_arg: Any,
        store_arg: Any,
        layout_arg: Any,
        poll_seconds_arg: float,
    ) -> int:
        calls.append("llama")
        llama_calls.append(
            {
                "ssh": ssh_arg,
                "oar": oar_arg,
                "store": store_arg,
                "layout": layout_arg,
                "poll_seconds": poll_seconds_arg,
            }
        )
        return 42

    monkeypatch.setattr(cli, "SshClient", lambda **_kw: fake_ssh)
    monkeypatch.setattr(cli, "_remote_home", lambda _ssh: Path("/home/u"))
    monkeypatch.setattr(
        cli, "_usage_policy_preflight", lambda *_a: calls.append("policy")
    )
    monkeypatch.setattr(
        cli, "ensure_home_headroom", lambda *_a, **_kw: calls.append("quota")
    )
    monkeypatch.setattr(cli, "Stager", FakeStager)
    monkeypatch.setattr(cli, "OarClient", FakeOar)
    monkeypatch.setattr(cli, "ensure_llama_server", mock_ensure_llama_server)
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
    assert len(llama_calls) == 1
    assert llama_calls[0]["ssh"] is fake_ssh
    assert isinstance(llama_calls[0]["oar"], FakeOar)
    assert llama_calls[0]["store"] is store
    assert llama_calls[0]["layout"].root == Path(
        f"/home/u/osm-polygon-operator/{config.run_id}"
    )
    assert llama_calls[0]["poll_seconds"] == 0


def test_prepare_same_site_refreshes_execution_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same-site continuation must install the current execution commit."""
    config, store = _store_at(
        tmp_path,
        RunPhase.REMOTE_PREPARED,
        facts={"site": "sophia", "active_stage": "split"},
        stage="split",
    )
    calls: list[str] = []
    fake_ssh = object()

    class FakeStager:
        def __init__(self, ssh: object) -> None:
            assert ssh is fake_ssh

        def prepare(self, _config: Any, _layout: Any) -> None:
            calls.append("prepare")

    monkeypatch.setattr(cli, "SshClient", lambda **_kw: fake_ssh)
    monkeypatch.setattr(cli, "_remote_home", lambda _ssh: Path("/home/u"))
    monkeypatch.setattr(
        cli, "_usage_policy_preflight", lambda *_a: calls.append("policy")
    )
    monkeypatch.setattr(
        cli, "ensure_home_headroom", lambda *_a, **_kw: calls.append("quota")
    )
    monkeypatch.setattr(cli, "Stager", FakeStager)
    monkeypatch.setattr(cli, "_stage_hf_token", lambda *_a: calls.append("token"))

    cli._prepare_destination_for_resume(
        store=store,
        config=config,
        site="sophia",
        relay_root=None,
        poll_seconds=0,
    )

    assert calls == ["policy", "quota", "prepare", "token"]


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
    args = _resume_args(config.run_id)
    assert cli._resume_run(config.run_id, args) == 0
    assert "nothing to reattach" in capsys.readouterr().out


# ------------------------------------------------------------------
# Narrow recovery from a previously FAILED allocation
# ------------------------------------------------------------------


def _failed_state(
    *,
    failed_job_id: int = 2961476,
    site: str = "grenoble",
    recovered_from_job_id: int | None = None,
) -> SimpleNamespace:
    """Build the minimal facts payload of an FAILED state for _reattach_decision tests."""

    facts: dict[str, object] = {
        "site": site,
        "job_id": failed_job_id,
        "failed_job_id": failed_job_id,
        "active_stage": "label",
    }
    if recovered_from_job_id is not None:
        facts["recovered_from_job_id"] = recovered_from_job_id
    return SimpleNamespace(phase=RunPhase.FAILED, facts=facts)


class _RecordingSsh:
    """Tiny SSH stub that records the commands issued by ``mark_remote_status``."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, command: str) -> SimpleNamespace:
        self.commands.append(command)
        return SimpleNamespace(stdout="", text="")


def test_reattach_decision_returns_failed_candidate_with_valid_facts() -> None:
    """A FAILED state with a recorded ``failed_job_id`` and ``site`` is a candidate."""

    state = _failed_state()
    assert cli._reattach_decision(state) == ("grenoble", 2961476)


def test_reattach_decision_refuses_failed_state_without_site() -> None:
    state = SimpleNamespace(
        phase=RunPhase.FAILED,
        facts={"failed_job_id": 2961476, "active_stage": "label"},
    )
    assert cli._reattach_decision(state) is None


def test_reattach_decision_refuses_failed_state_without_failed_job_id() -> None:
    state = SimpleNamespace(
        phase=RunPhase.FAILED,
        facts={"site": "grenoble", "active_stage": "label"},
    )
    assert cli._reattach_decision(state) is None


def test_reattach_decision_refuses_already_recovered_failed_job() -> None:
    """A previously-recovered FAILED allocation must not be recovered twice."""

    state = _failed_state(recovered_from_job_id=2961476)
    assert cli._reattach_decision(state) is None


def test_reattach_decision_still_recognises_failed_state_with_different_recovery() -> (
    None
):
    """If a prior recovery exists for a different job, the current failed_job_id is fresh."""

    state = _failed_state(
        failed_job_id=2962000,
        recovered_from_job_id=2961476,
    )
    assert cli._reattach_decision(state) == ("grenoble", 2962000)


def test_apply_classification_failed_from_failed_keeps_state_idempotent() -> None:
    """A re-inspection that still classifies FAILED must stay in FAILED but advance the sequence."""

    state = SimpleNamespace(
        phase=RunPhase.FAILED,
        facts={
            "site": "grenoble",
            "job_id": 2961476,
            "failed_job_id": 2961476,
            "active_stage": "label",
        },
    )
    transitions: list[tuple[Any, Any, dict[str, object]]] = []

    def transition(*, expected: Any, target: Any, facts: dict[str, object]) -> Any:
        transitions.append((expected, target, dict(facts)))
        return SimpleNamespace(phase=target, facts=facts)

    def load() -> Any:
        return state

    store = SimpleNamespace(load=load, transition=transition)
    layout = cli.RemoteLayout(PurePosixPath("/r"))
    ssh = _RecordingSsh()
    with pytest.raises(RuntimeError, match="failed deterministically"):
        cli._apply_classification(
            store=store,  # type: ignore[arg-type]
            config=_config(),
            ssh=ssh,  # type: ignore[arg-type]
            layout=layout,
            job_id=2961476,
            active_stage="label",
            classification=ExitClass.FAILED,
        )
    assert transitions == [
        (
            RunPhase.FAILED,
            RunPhase.FAILED,
            {"failed_job_id": 2961476, "failure_reason": "deterministic-failure"},
        )
    ]


def test_apply_classification_continue_from_failed_records_recovery_facts() -> None:
    """A re-inspected FAILED allocation whose evidence is CONTINUE must record recovery facts."""

    state = SimpleNamespace(
        phase=RunPhase.FAILED,
        facts={
            "site": "grenoble",
            "job_id": 2961476,
            "failed_job_id": 2961476,
            "active_stage": "label",
        },
    )
    transitions: list[tuple[Any, Any, dict[str, object]]] = []

    def transition(*, expected: Any, target: Any, facts: dict[str, object]) -> Any:
        transitions.append((expected, target, dict(facts)))
        return SimpleNamespace(phase=target, facts=facts)

    def load() -> Any:
        return state

    store = SimpleNamespace(load=load, transition=transition)
    cli._apply_classification(
        store=store,  # type: ignore[arg-type]
        config=_config(),
        ssh=object(),  # type: ignore[arg-type]
        layout=SimpleNamespace(),
        job_id=2961476,
        active_stage="label",
        classification=ExitClass.CONTINUE,
    )
    assert len(transitions) == 1
    expected, target, facts = transitions[0]
    assert expected is RunPhase.FAILED
    assert target is RunPhase.REMOTE_PREPARED
    assert facts["recovered_from_job_id"] == 2961476
    assert facts["recovery_attempt"] == 1
    assert "walltime-killed" in facts["recovery_reason"]
    assert facts["continued_after_job"] == 2961476


def test_apply_classification_continue_from_failed_increments_recovery_attempt() -> (
    None
):
    """Subsequent recoveries must increment the monotonic recovery_attempt counter."""

    state = SimpleNamespace(
        phase=RunPhase.FAILED,
        facts={
            "site": "grenoble",
            "job_id": 2962000,
            "failed_job_id": 2962000,
            "recovered_from_job_id": 2961476,
            "recovery_attempt": 1,
            "recovery_reason": "previous walltime-kill",
            "active_stage": "label",
        },
    )
    transitions: list[tuple[Any, Any, dict[str, object]]] = []

    def transition(*, expected: Any, target: Any, facts: dict[str, object]) -> Any:
        transitions.append((expected, target, dict(facts)))
        return SimpleNamespace(phase=target, facts=facts)

    def load() -> Any:
        return state

    store = SimpleNamespace(load=load, transition=transition)
    cli._apply_classification(
        store=store,  # type: ignore[arg-type]
        config=_config(),
        ssh=object(),  # type: ignore[arg-type]
        layout=SimpleNamespace(),
        job_id=2962000,
        active_stage="label",
        classification=ExitClass.CONTINUE,
    )
    _, _, facts = transitions[0]
    assert facts["recovery_attempt"] == 2
    assert facts["recovered_from_job_id"] == 2962000


def test_apply_classification_complete_from_failed_records_recovery_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A previously FAILED allocation that re-inspects as COMPLETE advances to COMPLETE with recovery."""

    state = SimpleNamespace(
        phase=RunPhase.FAILED,
        facts={
            "site": "grenoble",
            "job_id": 2961476,
            "failed_job_id": 2961476,
            "active_stage": "label",
        },
    )
    transitions: list[tuple[Any, Any, dict[str, object]]] = []

    def transition(*, expected: Any, target: Any, facts: dict[str, object]) -> Any:
        transitions.append((expected, target, dict(facts)))
        return SimpleNamespace(phase=target, facts=facts)

    def load() -> Any:
        return state

    store = SimpleNamespace(load=load, transition=transition)
    layout = cli.RemoteLayout(PurePosixPath("/r"))
    ssh = _RecordingSsh()
    monkeypatch.setattr(cli, "label_publication_commit", lambda *a, **kw: "a" * 40)
    cli._apply_classification(
        store=store,  # type: ignore[arg-type]
        config=_config(),
        ssh=ssh,  # type: ignore[arg-type]
        layout=layout,
        job_id=2961476,
        active_stage="label",
        classification=ExitClass.COMPLETE,
    )
    # FAILED -> VALIDATED with recovery, then VALIDATED -> VERIFYING, VERIFYING -> COMPLETE
    assert transitions[0][1] is RunPhase.VALIDATED
    assert transitions[0][2]["recovered_from_job_id"] == 2961476
    assert transitions[0][2]["recovery_attempt"] == 1
    assert transitions[1][1] is RunPhase.VERIFYING
    assert transitions[2][1] is RunPhase.COMPLETE


def test_apply_classification_publishes_completed_output_when_log_has_no_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SimpleNamespace(
        phase=RunPhase.RUNNING,
        facts={"active_stage": "label"},
    )
    transitions: list[tuple[Any, Any, dict[str, object]]] = []

    def transition(*, expected: Any, target: Any, facts: dict[str, object]) -> Any:
        transitions.append((expected, target, dict(facts)))
        return SimpleNamespace(phase=target, facts=facts)

    store = SimpleNamespace(load=lambda: state, transition=transition)
    monkeypatch.setattr(
        cli,
        "label_publication_commit",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            RuntimeError("label publication did not report an immutable Hub commit")
        ),
    )
    monkeypatch.setattr(cli, "publish_label", lambda *_a, **_kw: "d" * 40)

    cli._apply_classification(
        store=store,  # type: ignore[arg-type]
        config=_config(),
        ssh=_RecordingSsh(),  # type: ignore[arg-type]
        layout=cli.RemoteLayout(PurePosixPath("/r")),
        job_id=2963288,
        active_stage="label",
        classification=ExitClass.COMPLETE,
    )

    assert transitions[0][2]["hub_commit"] == "d" * 40
    assert transitions[-1][2]["published"] is True
