"""Public terminal contracts for the autonomous operator."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from osm_polygon_sentence_relevance.operator import cli
from osm_polygon_sentence_relevance.operator.config import OperatorConfig
from osm_polygon_sentence_relevance.operator.controller import LiveProgress
from osm_polygon_sentence_relevance.operator.label_lanes import (
    LabelLane,
    label_lane_plan,
)
from osm_polygon_sentence_relevance.operator.oar import ExitClass, JobState, JobStatus
from osm_polygon_sentence_relevance.operator.sites import SiteProbe
from osm_polygon_sentence_relevance.operator.staging import LabelAssets
from osm_polygon_sentence_relevance.operator.state import RunPhase, StateStore
from osm_polygon_sentence_relevance.operator.supervisor import SupervisorLaunch
from osm_polygon_sentence_relevance.operator.workflows import (
    RemoteLayout,
    label_submission,
    llama_build_submission,
)

runner = CliRunner()


def _run_args(
    *,
    stage: str = "label",
    sites: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=128,
        command="run",
        gpu_memory_mb=40_000,
        input_revision="b" * 40,
        llama_parallel=8,
        llama_per_slot_context=8192,
        poll_seconds=0.0,
        region="afghanistan-latest",
        remote_free_bytes=8 * 1024**3,
        request_concurrency=None,
        row_limit=0,
        sampling_h3_resolution=3,
        sampling_seed="sentence-relevance-v2",
        sampling_target=200000,
        scope="region",
        site=list(cli.DEFAULT_SITES) if sites is None else sites,
        stage=stage,
    )


def test_typer_help_exposes_exact_command_set() -> None:
    result = runner.invoke(cli.app, ["--help"], color=False)

    assert result.exit_code == 0
    for command in ("run", "resume", "status", "cleanup"):
        assert command in result.stdout


def test_v2_sampling_target_expands_in_place_and_never_decreases(
    tmp_path: Path,
) -> None:
    kwargs = {
        "scope": "region",
        "region": "afghanistan-latest",
        "stage": "label",
        "source_commit": "a" * 40,
        "input_revision": "b" * 40,
    }
    base = OperatorConfig.build(**kwargs, sampling_target=200_000)
    store = StateStore(tmp_path)
    store.load_or_create(base.run_identity)
    cli._sync_sampling_target(store, base)
    assert store.load().facts["sampling_target"] == 200_000

    expanded = OperatorConfig.build(**kwargs, sampling_target=400_000)
    store.load_or_create(expanded.run_identity)
    cli._sync_sampling_target(store, expanded)
    assert store.load().facts["sampling_target"] == 400_000

    reduced = OperatorConfig.build(**kwargs, sampling_target=100_000)
    store.load_or_create(reduced.run_identity)
    with pytest.raises(RuntimeError, match="cannot decrease"):
        cli._sync_sampling_target(store, reduced)


def test_v2_expansion_reopens_a_completed_run_for_continuation(
    tmp_path: Path,
) -> None:
    kwargs = {
        "scope": "region",
        "region": "afghanistan-latest",
        "stage": "label",
        "source_commit": "a" * 40,
        "input_revision": "b" * 40,
    }
    initial = OperatorConfig.build(**kwargs, sampling_target=200_000)
    store = StateStore(tmp_path)
    store.load_or_create(initial.run_identity)
    cli._sync_sampling_target(store, initial)
    store.transition(expected=RunPhase.CREATED, target=RunPhase.COMPLETE)

    expanded = OperatorConfig.build(**kwargs, sampling_target=400_000)
    cli._sync_sampling_target(store, expanded)

    state = store.load()
    assert state.phase is RunPhase.REMOTE_PREPARED
    assert state.facts["sampling_target"] == 400_000


def test_typer_run_delegates_current_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[SimpleNamespace] = []
    monkeypatch.setattr(cli, "_run", lambda args: seen.append(args) or 0)

    result = runner.invoke(
        cli.app,
        [
            "run",
            "--scope",
            "region",
            "--region",
            "afghanistan-latest",
            "--stage",
            "label",
        ],
        color=False,
    )

    assert result.exit_code == 0
    assert len(seen) == 1
    args = seen[0]
    assert vars(args) == {
        "batch_size": 128,
        "command": "run",
        "gpu_memory_mb": 40_000,
        "input_revision": None,
        "llama_parallel": 8,
        "llama_per_slot_context": 8192,
        "poll_seconds": 30.0,
        "region": "afghanistan-latest",
        "remote_free_bytes": 8 * 1024**3,
        "request_concurrency": None,
        "row_limit": 0,
        "sampling_h3_resolution": 3,
        "sampling_seed": "sentence-relevance-v2",
        "sampling_target": None,
        "scope": "region",
        "site": list(cli.DEFAULT_SITES),
        "stage": "label",
        "optimize_continuations": True,
    }


def test_typer_run_detach_starts_one_supervisor_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        cli,
        "start_detached_supervisor",
        lambda arguments, **kwargs: seen.append((arguments, kwargs))
        or SupervisorLaunch("session", Path("/data/supervisor.log"), (), "tmux"),
    )
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda *_: pytest.fail("detached mode must not dispatch in the parent"),
    )

    result = runner.invoke(
        cli.app,
        [
            "run",
            "--scope",
            "all",
            "--stage",
            "all",
            "--detach",
        ],
        color=False,
    )

    assert result.exit_code == 0
    assert len(seen) == 1
    arguments, kwargs = seen[0]
    assert arguments[0] == "run"
    assert "--detach" not in arguments
    assert arguments[1:5] == ("--scope", "all", "--stage", "all")
    assert kwargs["data_root"] == cli.DATA_ROOT
    assert "Detached supervisor started" in result.stdout


def test_typer_resume_detach_uses_run_specific_supervisor_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        cli,
        "start_detached_supervisor",
        lambda arguments, **kwargs: seen.append((arguments, kwargs))
        or SupervisorLaunch("session", Path("/data/supervisor.log"), (), "process"),
    )
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda *_: pytest.fail("detached mode must not dispatch in the parent"),
    )
    run_id = "a" * 20

    result = runner.invoke(
        cli.app,
        ["resume", run_id, "--detach", "--poll-seconds", "5"],
        color=False,
    )

    assert result.exit_code == 0
    arguments, kwargs = seen[0]
    assert arguments[:2] == ("resume", run_id)
    assert ("--poll-seconds", "5.0") in zip(arguments, arguments[1:], strict=False)
    assert "--detach" not in arguments
    assert kwargs["run_id"] == run_id
    assert "Detached supervisor started" in result.stdout


def test_sampling_target_defaults_to_v2_only_for_all_label_runs() -> None:
    regional = _run_args()
    regional.sampling_target = None
    assert cli._sampling_target_for_run(regional) is None

    worldwide = _run_args()
    worldwide.scope = "all"
    worldwide.region = None
    worldwide.sampling_target = None
    assert cli._sampling_target_for_run(worldwide) == 200_000


def test_positive_sampling_target_is_rejected_for_regional_runs() -> None:
    with pytest.raises(ValueError, match="scope all"):
        cli._sampling_target_for_run(_run_args())


def test_typer_resume_exposes_sampling_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[SimpleNamespace] = []
    monkeypatch.setattr(cli, "_resume_handler", lambda args: seen.append(args) or 0)

    result = runner.invoke(
        cli.app,
        [
            "resume",
            "a" * 20,
            "--sampling-target",
            "400000",
        ],
        color=False,
    )

    assert result.exit_code == 0
    assert len(seen) == 1
    assert seen[0].sampling_target == 400_000


def test_resume_expands_target_without_changing_run_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="a" * 40,
        input_revision="b" * 40,
        sampling_target=200_000,
    )
    store = StateStore(tmp_path)
    store.load_or_create(config.run_identity)
    cli._sync_sampling_target(store, config)
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    args = SimpleNamespace(
        command="resume",
        run_id=config.run_id,
        site=list(cli.DEFAULT_SITES),
        gpu_memory_mb=40_000,
        poll_seconds=0.0,
        sampling_target=400_000,
    )

    assert cli._resume_run(config.run_id, args) == 0

    assert store.load().facts["sampling_target"] == 400_000
    assert (
        config.run_id
        == OperatorConfig.build(
            scope="region",
            region="afghanistan-latest",
            stage="label",
            source_commit="a" * 40,
            input_revision="b" * 40,
            sampling_target=400_000,
        ).run_id
    )


def test_typer_explicit_sites_extend_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[SimpleNamespace] = []
    monkeypatch.setattr(cli, "_resume_handler", lambda args: seen.append(args) or 0)

    result = runner.invoke(
        cli.app,
        [
            "resume",
            "a" * 20,
            "--site",
            "nancy",
            "--site",
            "nantes",
        ],
        color=False,
    )

    assert result.exit_code == 0
    assert seen[0].site == [*cli.DEFAULT_SITES, "nancy", "nantes"]


def test_help_exposes_run_status_and_public_stage_choices(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--help"]) == 0
    text = capsys.readouterr().out
    assert "run" in text
    assert "status" in text


def test_sigint_only_stops_local_monitoring(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(_args: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_run", interrupted)
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "run",
                "--scope",
                "all",
                "--stage",
                "split",
                "--input-revision",
                "a" * 40,
            ]
        )
    assert exc.value.code == 130
    assert "remote job and checkpoints were preserved" in capsys.readouterr().err


def test_status_prints_durable_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "runs" / ("a" * 20)
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps({"phase": "running"}), encoding="utf-8"
    )
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    assert cli.main(["status", "a" * 20]) == 0
    assert '"phase": "running"' in capsys.readouterr().out


def test_live_progress_is_rendered(capsys: pytest.CaptureFixture[str]) -> None:
    cli._emit(LiveProgress(42, "build.stdout.log", "one\ntwo\n", 8))
    assert capsys.readouterr().out.splitlines() == ["[job 42] one", "[job 42] two"]


def test_git_head_requires_immutable_clean_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter(
        [
            SimpleNamespace(stdout="short\n"),
            SimpleNamespace(stdout="a" * 40 + "\n"),
            SimpleNamespace(stdout="dirty\n"),
        ]
    )
    monkeypatch.setattr(cli.subprocess, "run", lambda *_a, **_k: next(values))
    with pytest.raises(RuntimeError, match="immutable"):
        cli._git_head()
    with pytest.raises(RuntimeError, match="clean"):
        cli._git_head()


def test_resolve_input_revision_explicit_and_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert cli._resolve_input_revision("a" * 40, "split") == "a" * 40

    class Api:
        def dataset_info(self, dataset_id: str, *, revision: str) -> SimpleNamespace:
            assert dataset_id == cli.OUTPUT_DATASET_ID
            assert revision == "main"
            return SimpleNamespace(sha="b" * 40)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    assert cli._resolve_input_revision(None, "label") == "b" * 40
    sys.modules.pop("huggingface_hub", None)


def test_remote_home_validation() -> None:
    class FakeSsh:
        def __init__(self, outputs: list[str]) -> None:
            self.outputs = outputs

        def run(self, _command: str) -> SimpleNamespace:
            return SimpleNamespace(stdout=self.outputs.pop(0))

    assert cli._remote_home(FakeSsh(["/home/user\n"])) == PurePosixPath("/home/user")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="invalid"):
        cli._remote_home(FakeSsh(["relative\n"]))  # type: ignore[arg-type]


def test_usage_policy_preflight_runs_live_checker() -> None:
    class PolicySsh:
        command = ""

        def run(self, command: str) -> SimpleNamespace:
            self.__class__.command = command
            return SimpleNamespace(stdout="")

    cli._usage_policy_preflight(PolicySsh(), "nancy")  # type: ignore[arg-type]
    assert "usagepolicycheck -l --sites nancy" in PolicySsh.command
    assert "usagepolicycheck -t" in PolicySsh.command


@pytest.mark.parametrize("site", ["", "Nancy", "nancy;true", "nancy site"])
def test_usage_policy_preflight_rejects_unsafe_site(site: str) -> None:
    with pytest.raises(ValueError, match="site name"):
        cli._usage_policy_preflight(SimpleNamespace(), site)  # type: ignore[arg-type]


class _FakeStore:
    instances: list[_FakeStore] = []

    def __init__(self, _root: Path) -> None:
        self.value = SimpleNamespace(phase=RunPhase.CREATED, facts={})
        self.__class__.instances.append(self)

    def load_or_create(self, _identity: object) -> SimpleNamespace:
        return self.value

    def load(self) -> SimpleNamespace:
        return self.value

    def transition(
        self,
        *,
        expected: RunPhase,
        target: RunPhase,
        facts: dict[str, object],
    ) -> SimpleNamespace:
        assert self.value.phase is expected
        self.value = SimpleNamespace(phase=target, facts={**self.value.facts, **facts})
        return self.value


class _FakeSsh:
    def __init__(self, **_kwargs: object) -> None:
        self.target = str(_kwargs.get("target", ""))
        self.commands: list[str] = []

    def run(self, command: str) -> SimpleNamespace:
        self.commands.append(command)
        if "quota_output=$(timeout 15s quota" in command:
            return SimpleNamespace(stdout=" 1000 25000000 100000000\n")
        if 'printf "%s\\n" "$HOME"' in command:
            return SimpleNamespace(stdout="/home/user\n")
        if "build.exit_code" in command or "finalize.exit_code" in command:
            return SimpleNamespace(stdout="0\n")
        if "labeling.exit_code" in command:
            return SimpleNamespace(stdout="0\n")
        if "labeling.stdout.log" in command:
            return SimpleNamespace(
                stdout='progress\n{"commit_id":"' + "c" * 40 + '"}\n'
            )
        if "manifest.json" in command:
            return SimpleNamespace(stdout="yes")
        if "llama-server" in command and "printf yes" in command:
            return SimpleNamespace(stdout="yes")
        if "publish_export_directory" in command:
            return SimpleNamespace(stdout="abcdef123456\n")
        return SimpleNamespace(stdout="")

    def read_since(self, _path: str, offset: int) -> SimpleNamespace:
        return SimpleNamespace(text="", next_offset=offset, reset=False)


class _FakeOar:
    def __init__(self, _ssh: object, **_kwargs: object) -> None:
        self.next_job = 90

    def submit(self, _request: object) -> int:
        self.next_job += 1
        return self.next_job

    def status(self, job_id: int) -> JobStatus:
        return JobStatus(job_id, JobState.TERMINATED, exit_code=0)


class _FakeStager:
    def __init__(self, _ssh: object) -> None:
        pass

    def prepare_label_assets(
        self, _config: object, layout: cli.RemoteLayout, *, download_input: bool
    ) -> LabelAssets:
        return LabelAssets(
            layout.root / "input/sentences.parquet",
            layout.root / "model/model.gguf",
            layout.root / "tokenizer",
            True,
        )

    def prepare_v2_input(
        self,
        _config: object,
        layout: cli.RemoteLayout,
        _source: object,
    ) -> PurePosixPath:
        return layout.v2_input


class _FakeController:
    def __init__(self, **kwargs: object) -> None:
        self.state = kwargs["state"]

    def prepare(self, *, site: str) -> None:
        store = self.state
        store.value = SimpleNamespace(
            phase=RunPhase.REMOTE_PREPARED, facts={"site": site}
        )

    def submit(self, **_kwargs: object) -> int:
        self.state.value = SimpleNamespace(
            phase=RunPhase.RUNNING, facts={**self.state.value.facts, "job_id": 80}
        )
        return 80

    def monitor(self, _job_id: int, *, log_name: str) -> JobState:
        assert log_name
        return JobState.TERMINATED


def _install_run_fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    _FakeStore.instances.clear()
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_git_head", lambda: "a" * 40)
    monkeypatch.setattr(cli, "_resolve_input_revision", lambda *_a: "b" * 40)
    monkeypatch.setattr(
        cli,
        "probe_site",
        lambda target, _run_id, _requirements=None: SiteProbe(
            target, target, True, 80_000, (8, 0), 100 * 1024**3, 0
        ),
    )
    monkeypatch.setattr(cli, "SshClient", _FakeSsh)
    monkeypatch.setattr(cli, "StateStore", _FakeStore)
    monkeypatch.setattr(cli, "OarClient", _FakeOar)
    monkeypatch.setattr(cli, "Stager", _FakeStager)
    monkeypatch.setattr(cli, "Controller", _FakeController)
    monkeypatch.setattr(
        cli,
        "preserve_manual_eval",
        lambda *_args, lane: tmp_path / f"manual-eval-{lane}.jsonl",
    )


def test_run_split_finalizes_publishes_and_marks_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_run_fakes(monkeypatch, tmp_path)
    args = _run_args(stage="split", sites=["nancy"])
    assert cli._run(args) == 0
    state = _FakeStore.instances[-1].value
    assert state.phase is RunPhase.COMPLETE
    assert state.facts["published"] is True
    assert state.facts["hub_commit"] == "abcdef123456"


def test_finalize_split_checkpointed_publishes_and_marks_complete(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A resumed split run must finalize and publish after checkpoints complete."""

    config = OperatorConfig.build(
        scope="all",
        stage="split",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )
    store = _FakeStore(Path("/state"))
    store.value = SimpleNamespace(phase=RunPhase.CHECKPOINTED, facts={})
    ssh = _FakeSsh(target="grenoble")
    oar = _FakeOar(ssh)
    layout = cli.RemoteLayout(PurePosixPath("/run"))
    events: list[str] = []
    monkeypatch.setattr(
        cli.split_finalization,
        "split_finalization_submission",
        lambda *_: object(),
    )
    monkeypatch.setattr(
        cli.split_finalization,
        "monitor_job_with_log",
        lambda *_args, **_kwargs: events.append("monitored"),
    )
    monkeypatch.setattr(
        cli.split_finalization,
        "assert_remote_exit_zero",
        lambda *_args, **_kwargs: events.append("validated"),
    )
    monkeypatch.setattr(
        cli.split_finalization, "publish_split", lambda *_args: "c" * 40
    )
    monkeypatch.setattr(
        cli.split_finalization,
        "mark_remote_status",
        lambda *_args: events.append("marked"),
    )

    cli.split_finalization.finalize_split_checkpointed(
        store=store,
        config=config,
        ssh=ssh,
        layout=layout,
        oar=oar,
        poll_seconds=0.0,
    )

    assert store.value.phase is RunPhase.COMPLETE
    assert store.value.facts["hub_commit"] == "c" * 40
    assert events == ["monitored", "validated", "marked"]
    assert "Sentence splitting complete" in capsys.readouterr().out


def test_finalize_split_checkpointed_stage_all_hands_off_to_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumed all-stage run must label after split finalization."""

    config = OperatorConfig.build(
        scope="all",
        stage="all",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )
    store = _FakeStore(Path("/state"))
    store.value = SimpleNamespace(phase=RunPhase.CHECKPOINTED, facts={})
    ssh = _FakeSsh(target="sophia")
    oar = _FakeOar(ssh)
    layout = cli.RemoteLayout(PurePosixPath("/run"))
    monkeypatch.setattr(
        cli.split_finalization,
        "split_finalization_submission",
        lambda *_: object(),
    )
    monkeypatch.setattr(
        cli.split_finalization,
        "monitor_job_with_log",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        cli.split_finalization,
        "assert_remote_exit_zero",
        lambda *_args, **_kwargs: None,
    )

    final_job = cli.split_finalization.finalize_split_checkpointed(
        store=store,
        config=config,
        ssh=ssh,
        layout=layout,
        oar=oar,
        poll_seconds=0.0,
    )

    assert final_job == 91
    assert store.value.phase is RunPhase.REMOTE_PREPARED
    assert store.value.facts["split_output_job_id"] == 91
    assert store.value.facts["active_stage"] == "label"
    assert store.value.facts["label_lane"] == "production"


def test_completed_v2_smoke_is_preserved_then_reopens_production(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = OperatorConfig.build(
        scope="all",
        stage="all",
        source_commit="a" * 40,
        input_revision="b" * 40,
        row_limit=128,
        sampling_target=200_000,
    )
    store = _FakeStore(tmp_path)
    store.value = SimpleNamespace(
        phase=RunPhase.RUNNING,
        facts={"active_stage": "label", "label_lane": "smoke"},
    )
    layout = cli.RemoteLayout(PurePosixPath("/run"))
    plan = label_lane_plan(config, layout.root, store.value.facts)
    preserved = tmp_path / "runs" / config.run_id / "label-smoke"
    manual_eval = tmp_path / "runs" / config.run_id / "manual-eval-smoke.jsonl"
    manual_eval_calls: list[tuple[PurePosixPath, str]] = []
    monkeypatch.setattr(cli, "preserve_label", lambda *_args: preserved)

    def preserve_manual_eval(
        _ssh: object,
        _layout: object,
        work_dir: PurePosixPath,
        *,
        lane: str,
    ) -> Path:
        manual_eval_calls.append((work_dir, lane))
        return manual_eval

    monkeypatch.setattr(cli, "preserve_manual_eval", preserve_manual_eval)
    monkeypatch.setattr(
        cli,
        "label_publication_commit",
        lambda *_args: pytest.fail("smoke must never be published"),
    )
    monkeypatch.setattr(
        cli,
        "publish_label",
        lambda *_args: pytest.fail("smoke must never be published"),
    )
    monkeypatch.setattr(
        cli,
        "mark_remote_status",
        lambda *_args: pytest.fail("parent run remains active after smoke"),
    )

    cli._apply_classification(
        store=store,  # type: ignore[arg-type]
        config=config,
        ssh=_FakeSsh(target="sophia"),  # type: ignore[arg-type]
        layout=layout,
        job_id=92,
        active_stage="label",
        classification=ExitClass.COMPLETE,
        label_plan=plan,
    )

    assert store.value.phase is RunPhase.REMOTE_PREPARED
    assert store.value.facts["smoke_completed"] is True
    assert store.value.facts["smoke_job_id"] == 92
    assert store.value.facts["smoke_artifact_path"] == str(preserved)
    assert store.value.facts["smoke_manual_eval_path"] == str(manual_eval)
    assert manual_eval_calls == [(plan.work_dir, "smoke")]
    assert store.value.facts["label_lane"] == LabelLane.PRODUCTION.value
    assert store.value.facts["active_stage"] == "label"
    assert "published" not in store.value.facts


def test_completed_v2_production_preserves_manual_eval_before_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = OperatorConfig.build(
        scope="all",
        stage="all",
        source_commit="a" * 40,
        input_revision="b" * 40,
        row_limit=128,
        sampling_target=200_000,
    )
    store = _FakeStore(tmp_path)
    store.value = SimpleNamespace(
        phase=RunPhase.RUNNING,
        facts={"active_stage": "label", "label_lane": "production"},
    )
    layout = cli.RemoteLayout(PurePosixPath("/run"))
    plan = label_lane_plan(config, layout.root, store.value.facts)
    manual_eval = tmp_path / "manual-eval-production.jsonl"
    calls: list[tuple[PurePosixPath, str]] = []

    def preserve_manual_eval(
        _ssh: object,
        _layout: object,
        work_dir: PurePosixPath,
        *,
        lane: str,
    ) -> Path:
        calls.append((work_dir, lane))
        return manual_eval

    monkeypatch.setattr(cli, "preserve_manual_eval", preserve_manual_eval)
    monkeypatch.setattr(cli, "label_publication_commit", lambda *_args: "c" * 40)
    monkeypatch.setattr(cli, "mark_remote_status", lambda *_args: None)

    cli._apply_classification(
        store=store,  # type: ignore[arg-type]
        config=config,
        ssh=_FakeSsh(target="sophia"),  # type: ignore[arg-type]
        layout=layout,
        job_id=94,
        active_stage="label",
        classification=ExitClass.COMPLETE,
        label_plan=plan,
    )

    assert store.value.phase is RunPhase.COMPLETE
    assert store.value.facts["manual_eval_path"] == str(manual_eval)
    assert store.value.facts["hub_commit"] == "c" * 40
    assert calls == [(plan.work_dir, "production")]


def test_v2_production_resume_inspects_isolated_full_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = OperatorConfig.build(
        scope="all",
        stage="all",
        source_commit="a" * 40,
        input_revision="b" * 40,
        row_limit=128,
        sampling_target=200_000,
    )
    store = _FakeStore(Path("/state"))
    store.value = SimpleNamespace(
        phase=RunPhase.QUEUED,
        facts={
            "active_stage": "label",
            "label_lane": "production",
            "site": "sophia",
            "job_id": 93,
        },
    )
    ssh = _FakeSsh(target="sophia")
    layout = cli.RemoteLayout(PurePosixPath("/run"))
    oar = _FakeOar(ssh)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "_attach_to_site",
        lambda *_args, **_kwargs: (
            ssh,
            layout,
            oar,
            _FakeController(state=store),
        ),
    )

    def inspect(_ssh: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(cli.recorded_job, "inspect_remote_resume", inspect)
    monkeypatch.setattr(
        cli.recorded_job,
        "classify_terminal",
        lambda *_args: ExitClass.CONTINUE,
    )

    result = cli._classify_or_continue(
        _run_args(stage="all"),
        store,  # type: ignore[arg-type]
        config,
        "sophia",
        93,
    )

    assert result is ExitClass.CONTINUE
    assert captured["label_work_root"] == "/run/label-work"
    assert captured["label_output_root"] == "/run/label-output"
    identity = captured["expected_identity"]
    assert isinstance(identity, dict)
    assert identity["row_limit"] == 0


def test_fresh_v2_all_runs_smoke_then_full_without_replaying_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_run_fakes(monkeypatch, tmp_path)
    submitted_lanes: list[str] = []
    split_submissions = 0

    class LaneController(_FakeController):
        def submit(self, **kwargs: object) -> int:
            nonlocal split_submissions
            component = kwargs["component"]
            facts = dict(self.state.value.facts)
            facts.update({"job_id": 80, "active_stage": component.value})
            if component is cli.Stage.SPLIT:
                split_submissions += 1
            else:
                plan = kwargs["label_plan"]
                submitted_lanes.append(plan.lane.value)  # type: ignore[union-attr]
                facts["label_lane"] = plan.lane.value  # type: ignore[union-attr]
            self.state.value = SimpleNamespace(phase=RunPhase.RUNNING, facts=facts)
            return 80

    monkeypatch.setattr(cli, "Controller", LaneController)
    monkeypatch.setattr(cli, "_race_queued_start", lambda *_args: ("sophia", 80))
    monkeypatch.setattr(cli, "preserve_label", lambda *_args: tmp_path / "smoke")
    monkeypatch.setattr(
        cli,
        "preserve_manual_eval",
        lambda *_args, lane: tmp_path / f"manual-eval-{lane}.jsonl",
    )
    monkeypatch.setattr(cli, "mark_remote_status", lambda *_args: None)
    args = _run_args(stage="all", sites=["sophia"])
    args.scope = "all"
    args.region = None
    args.row_limit = 128

    assert cli._run(args) == 0

    state = _FakeStore.instances[-1].value
    assert split_submissions == 1
    assert submitted_lanes == ["smoke", "production"]
    assert state.phase is RunPhase.COMPLETE
    assert state.facts["smoke_completed"] is True
    assert state.facts["published"] is True


def test_fresh_split_submission_optimizes_before_monitoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh split run must trial earlier sites before it starts waiting."""

    _install_run_fakes(monkeypatch, tmp_path)
    calls: list[tuple[str, int]] = []

    def fake_optimize(
        _args: SimpleNamespace,
        _store: object,
        _config: OperatorConfig,
        fallback_site: str,
        fallback_job_id: int,
    ) -> tuple[str, int]:
        calls.append((fallback_site, fallback_job_id))
        return fallback_site, fallback_job_id

    monkeypatch.setattr(cli, "_race_queued_start", fake_optimize)

    assert cli._run(_run_args(stage="split", sites=["nancy"])) == 0
    assert calls == [("nancy", 80)]


def test_fresh_split_rebinds_monitoring_after_site_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trial winner must become the controller used for the same allocation."""

    _install_run_fakes(monkeypatch, tmp_path)
    attached: list[str] = []
    original_attach = cli._attach_to_site

    def recording_attach(
        store: object,
        config: OperatorConfig,
        site: str,
        *,
        poll_seconds: float,
        preflight: object = None,
    ) -> tuple[object, object, object, object]:
        attached.append(site)
        return original_attach(
            store, config, site, poll_seconds=poll_seconds, preflight=preflight
        )

    monkeypatch.setattr(cli, "_attach_to_site", recording_attach)

    def replace_with_sophia(
        _args: SimpleNamespace,
        store: _FakeStore,
        _config: OperatorConfig,
        _fallback_site: str,
        _fallback_job_id: int,
    ) -> tuple[str, int]:
        current = store.load()
        store.transition(
            expected=current.phase,
            target=current.phase,
            facts={"site": "sophia", "job_id": 81},
        )
        return "sophia", 81

    monkeypatch.setattr(cli, "_race_queued_start", replace_with_sophia)

    assert cli._run(_run_args(stage="split", sites=["nancy", "sophia"])) == 0
    assert attached == ["sophia"]


def test_fresh_label_rebind_restages_candidate_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regional label trial must restage assets on its adopted site."""

    _install_run_fakes(monkeypatch, tmp_path)
    attached: list[str] = []
    original_attach = cli._attach_to_site

    def recording_attach(
        store: object,
        config: OperatorConfig,
        site: str,
        *,
        poll_seconds: float,
        preflight: object = None,
    ) -> tuple[object, object, object, object]:
        attached.append(site)
        return original_attach(
            store, config, site, poll_seconds=poll_seconds, preflight=preflight
        )

    monkeypatch.setattr(cli, "_attach_to_site", recording_attach)

    def replace_with_sophia(
        _args: SimpleNamespace,
        store: _FakeStore,
        _config: OperatorConfig,
        _fallback_site: str,
        _fallback_job_id: int,
    ) -> tuple[str, int]:
        current = store.load()
        store.transition(
            expected=current.phase,
            target=RunPhase.RUNNING,
            facts={"site": "sophia", "job_id": 81},
        )
        return "sophia", 81

    monkeypatch.setattr(cli, "_race_queued_start", replace_with_sophia)

    args = _run_args(stage="label", sites=["nancy", "sophia"])
    args.sampling_target = None
    assert cli._run(args) == 0
    assert attached == ["sophia"]


def test_run_reclaims_managed_storage_then_reprobes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_run_fakes(monkeypatch, tmp_path)
    probes = iter(
        [
            SiteProbe("nancy", "nancy", True, 80_000, (8, 0), 1, 0),
            SiteProbe("nancy", "nancy", True, 80_000, (8, 0), 100 * 1024**3, 0),
        ]
    )
    cleaned: list[bool] = []
    monkeypatch.setattr(
        cli, "probe_site", lambda _target, _run_id, _requirements=None: next(probes)
    )
    monkeypatch.setattr(
        cli,
        "cleanup_managed_runs",
        lambda _ssh, *, execute: cleaned.append(execute) or ("/managed/old",),
    )
    args = _run_args(stage="split", sites=["nancy"])
    assert cli._run(args) == 0
    assert cleaned == [True]


def test_run_reclaims_terminal_storage_on_every_reachable_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_run_fakes(monkeypatch, tmp_path)
    probes = iter(
        [
            SiteProbe("nancy", "nancy", True, 80_000, (8, 0), 100 * 1024**3, 0),
            SiteProbe("sophia", "sophia", True, 80_000, (8, 0), 100 * 1024**3, 0),
            SiteProbe("bordeaux", "bordeaux", False, 0, None, 0, 0),
            SiteProbe("nancy", "nancy", True, 80_000, (8, 0), 100 * 1024**3, 0),
            SiteProbe("sophia", "sophia", True, 80_000, (8, 0), 100 * 1024**3, 0),
            SiteProbe("bordeaux", "bordeaux", False, 0, None, 0, 0),
        ]
    )
    cleaned: list[str] = []
    monkeypatch.setattr(
        cli,
        "probe_site",
        lambda target, _run_id, _requirements=None: next(probes),
    )
    monkeypatch.setattr(
        cli,
        "cleanup_managed_runs",
        lambda ssh, *, execute: cleaned.append(ssh.target) or (),
    )

    assert (
        cli._run(_run_args(stage="split", sites=["nancy", "sophia", "bordeaux"])) == 0
    )
    assert cleaned == ["nancy", "sophia"]


def test_terminal_storage_cleanup_skips_unreachable_sites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        cli,
        "cleanup_managed_runs",
        lambda *_args, **_kwargs: calls.append(True) or (),
    )
    cli._reclaim_terminal_managed_storage(
        [SiteProbe("bordeaux", "bordeaux", False, 0, None, 0, 0)]
    )
    assert calls == []


def test_terminal_storage_cleanup_reports_removed_roots(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "cleanup_managed_runs",
        lambda *_args, **_kwargs: ("/home/run-a", "/home/run-b"),
    )
    cli._reclaim_terminal_managed_storage(
        [SiteProbe("nancy", "nancy", True, 80_000, (8, 0), 1, 0)]
    )
    assert "Site nancy: removed 2 terminal managed run(s)" in capsys.readouterr().out


def test_run_rejects_missing_external_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path / "missing")
    with pytest.raises(RuntimeError, match="external data root"):
        cli._run(SimpleNamespace())


def test_run_reclaims_low_storage_before_selecting_another_compatible_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_run_fakes(monkeypatch, tmp_path)
    probes = iter(
        [
            SiteProbe("nancy", "nancy", True, 80_000, (8, 0), 1, 0),
            SiteProbe("sophia", "sophia", True, 80_000, (8, 0), 100 * 1024**3, 0),
            SiteProbe("nancy", "nancy", True, 80_000, (8, 0), 100 * 1024**3, 0),
            SiteProbe("sophia", "sophia", True, 80_000, (8, 0), 100 * 1024**3, 0),
        ]
    )
    cleaned: list[bool] = []
    monkeypatch.setattr(
        cli,
        "probe_site",
        lambda _target, _run_id, _requirements=None: next(probes),
    )
    monkeypatch.setattr(
        cli,
        "cleanup_managed_runs",
        lambda _ssh, *, execute: cleaned.append(execute) or (),
    )

    assert cli._run(_run_args(stage="split", sites=["nancy", "sophia"])) == 0
    assert cleaned == [True, True]


def test_run_storage_cleanup_skips_unreachable_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_run_fakes(monkeypatch, tmp_path)
    probes = iter(
        [
            SiteProbe("nancy", "nancy", True, 80_000, (8, 0), 1, 0),
            SiteProbe("bordeaux", "bordeaux", False, 0, None, 0, 0),
            SiteProbe("nancy", "nancy", True, 80_000, (8, 0), 100 * 1024**3, 0),
            SiteProbe("bordeaux", "bordeaux", False, 0, None, 0, 0),
        ]
    )
    cleaned: list[bool] = []
    monkeypatch.setattr(
        cli, "probe_site", lambda _target, _run_id, _requirements=None: next(probes)
    )
    monkeypatch.setattr(
        cli,
        "cleanup_managed_runs",
        lambda _ssh, *, execute: cleaned.append(execute) or (),
    )

    assert cli._run(_run_args(stage="split", sites=["nancy", "bordeaux"])) == 0
    assert cleaned == [True]


def test_run_retries_cleanup_when_storage_remains_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_run_fakes(monkeypatch, tmp_path)
    probes = iter(
        [
            SiteProbe("nancy", "nancy", True, 80_000, (8, 0), 1, 0),
            SiteProbe("bordeaux", "bordeaux", False, 0, None, 0, 0),
            SiteProbe("nancy", "nancy", True, 80_000, (8, 0), 1, 0),
            SiteProbe("bordeaux", "bordeaux", False, 0, None, 0, 0),
            SiteProbe("nancy", "nancy", True, 80_000, (8, 0), 100 * 1024**3, 0),
            SiteProbe("bordeaux", "bordeaux", False, 0, None, 0, 0),
        ]
    )
    cleaned: list[bool] = []
    monkeypatch.setattr(
        cli, "probe_site", lambda _target, _run_id, _requirements=None: next(probes)
    )
    monkeypatch.setattr(
        cli,
        "cleanup_managed_runs",
        lambda _ssh, *, execute: cleaned.append(execute) or (),
    )

    assert cli._run(_run_args(stage="split", sites=["nancy", "bordeaux"])) == 0
    assert cleaned == [True, True]


def test_run_label_reuses_checkpoints_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_run_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli,
        "probe_site",
        lambda target, _run_id, _requirements=None: SiteProbe(
            target,
            target,
            True,
            80_000,
            (8, 0),
            6 * 1024**3,
            0,
            has_managed_run=True,
        ),
    )
    headroom: list[int] = []
    monkeypatch.setattr(
        cli,
        "ensure_home_headroom",
        lambda _ssh, *, protected_root, minimum_headroom_bytes: headroom.append(
            minimum_headroom_bytes
        ),
    )
    args = _run_args(stage="all", sites=["nancy"])
    args.scope = "all"
    args.region = None
    assert cli._run(args) == 0
    state = _FakeStore.instances[-1].value
    assert state.phase is RunPhase.COMPLETE
    assert state.facts["published"] is True
    assert state.facts["hub_commit"] == "c" * 40
    output = capsys.readouterr().out
    assert "Resolving immutable input revision" in output
    assert "Probing Grid'5000 site: nancy" in output
    assert "Preparing remote checkout" in output
    assert "Staging immutable labeling assets" in output
    assert "Labeling complete" in output
    assert headroom
    assert set(headroom) == {512 * 1024**2}


def test_run_all_prepares_remote_phase_before_building_missing_llama_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage-all run must make VALIDATED resumable before the CUDA build."""

    _install_run_fakes(monkeypatch, tmp_path)

    class MissingLlamaStager(_FakeStager):
        def prepare_label_assets(
            self,
            _config: object,
            layout: cli.RemoteLayout,
            *,
            download_input: bool,
        ) -> LabelAssets:
            assets = super().prepare_label_assets(
                _config, layout, download_input=download_input
            )
            return LabelAssets(
                assets.input_parquet,
                assets.model_file,
                assets.tokenizer_dir,
                False,
            )

    monkeypatch.setattr(cli, "Stager", MissingLlamaStager)

    args = _run_args(stage="all", sites=["nancy"])
    args.scope = "all"
    args.region = None

    assert cli._run(args) == 0
    state = _FakeStore.instances[-1].value
    assert state.phase is RunPhase.COMPLETE
    assert state.facts["llama_build_job_id"] == 92


def test_run_rejects_v2_label_stage_before_remote_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_run_fakes(monkeypatch, tmp_path)
    args = _run_args(stage="label", sites=["nancy"])
    args.scope = "all"
    args.region = None

    with pytest.raises(RuntimeError, match="requires --stage all"):
        cli._run(args)


def test_run_reraises_when_no_site_can_satisfy_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_run_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli,
        "probe_site",
        lambda target, _run_id, _requirements=None: SiteProbe(
            target, target, False, 0, (0, 0), 0, 0
        ),
    )
    monkeypatch.setattr(cli, "cleanup_can_restore_compatibility", lambda *_a: False)

    with pytest.raises(cli.NoCompatibleSiteError):
        cli._run(_run_args(stage="split", sites=["nancy"]))


def test_run_retries_cleanup_before_reporting_persistent_storage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_run_fakes(monkeypatch, tmp_path)
    low = SiteProbe("nancy", "nancy", True, 80_000, (8, 0), 1, 0)
    monkeypatch.setattr(cli, "probe_site", lambda *_args, **_kwargs: low)
    cleaned: list[str] = []
    monkeypatch.setattr(
        cli,
        "cleanup_managed_runs",
        lambda ssh, *, execute: cleaned.append(ssh.target) or (),
    )

    with pytest.raises(cli.NoCompatibleSiteError):
        cli._run(_run_args(stage="split", sites=["nancy"]))
    assert cleaned == ["nancy", "nancy", "nancy"]


def test_failed_allocation_marks_managed_remote_root_eligible_for_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A classified failure must not leave its managed marker permanently active."""

    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        cli,
        "_transition_terminal",
        lambda *args, **kwargs: calls.append(("transition", kwargs["target"])),
    )
    monkeypatch.setattr(
        cli,
        "mark_remote_status",
        lambda ssh, layout, status: calls.append(("marker", ssh, layout, status)),
    )
    ssh = object()
    layout = cli.RemoteLayout(PurePosixPath("/r"))
    state = SimpleNamespace(
        phase=RunPhase.QUEUED,
        facts={"site": "sophia", "job_id": 2895249, "active_stage": "label"},
    )
    store = SimpleNamespace(load=lambda: state)

    with pytest.raises(RuntimeError, match="failed deterministically"):
        cli._apply_classification(
            store=store,  # type: ignore[arg-type]
            config=OperatorConfig.build(
                scope="region",
                region="afghanistan-latest",
                stage="label",
                source_commit="a" * 40,
                input_revision="b" * 40,
            ),
            ssh=ssh,  # type: ignore[arg-type]
            layout=layout,
            job_id=2895249,
            active_stage="label",
            classification=ExitClass.FAILED,
        )

    assert calls == [
        ("transition", RunPhase.FAILED),
        ("marker", ssh, layout, "failed"),
    ]


def test_terminal_transition_rejects_unexpected_phase(tmp_path: Path) -> None:
    state = _FakeStore(tmp_path)
    with pytest.raises(RuntimeError, match="unexpected durable"):
        cli._transition_terminal(
            state,  # type: ignore[arg-type]
            expected=(RunPhase.RUNNING,),
            target=RunPhase.COMPLETE,
            facts={},
        )


def test_status_missing_and_cleanup_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="does not exist"):
        cli._status(SimpleNamespace(run_id="a" * 20))
    monkeypatch.setattr(cli, "cleanup_managed_runs", lambda _ssh, *, execute: ())
    monkeypatch.setattr(cli, "SshClient", _FakeSsh)
    assert cli._cleanup(SimpleNamespace(site="nancy", execute=False)) == 0
    assert "No pipeline-managed" in capsys.readouterr().out
    monkeypatch.setattr(
        cli, "cleanup_managed_runs", lambda _ssh, *, execute: ("/home/user/run",)
    )
    assert cli._cleanup(SimpleNamespace(site="nancy", execute=True)) == 0
    assert "removed: /home/user/run" in capsys.readouterr().out


def test_main_reports_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "_status",
        lambda _args: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert cli.main(["status", "a" * 20]) == 1
    assert "Error: boom" in capsys.readouterr().err


def _live_state(phase: RunPhase, **facts: object) -> SimpleNamespace:
    return SimpleNamespace(phase=phase, facts={"site": "sophia", **facts})


@pytest.mark.parametrize(
    ("phase", "facts", "expected"),
    [
        (RunPhase.QUEUED, {"job_id": 2895249}, ("sophia", 2895249)),
        (RunPhase.RUNNING, {"job_id": 1}, ("sophia", 1)),
        (RunPhase.SUBMITTED, {"job_id": 7}, ("sophia", 7)),
        (RunPhase.COMPLETE, {"job_id": 7}, None),
        (RunPhase.REMOTE_PREPARED, {"job_id": 7}, None),
        (RunPhase.QUEUED, {}, None),
        (RunPhase.QUEUED, {"job_id": 0}, None),
        (RunPhase.QUEUED, {"job_id": "x"}, None),
        (RunPhase.QUEUED, {"job_id": 7, "site": ""}, None),
    ],
)
def test_reattach_decision_inspects_recorded_site_and_job(
    phase: RunPhase, facts: dict[str, object], expected: tuple[str, int] | None
) -> None:
    assert cli._reattach_decision(_live_state(phase, **facts)) == expected


def test_rerun_reattaches_to_stored_live_job_without_submitting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live = _live_state(RunPhase.QUEUED, job_id=2895249, active_stage="label")

    class FakeStore:
        def __init__(self, _root: object) -> None:
            self.state = live

        def load_or_create(self, _identity: object) -> SimpleNamespace:
            return self.state

        def load(self) -> SimpleNamespace:
            return self.state

        def transition(
            self, *, expected: RunPhase, target: RunPhase, facts: dict[str, object]
        ) -> SimpleNamespace:
            assert self.state.phase is expected
            merged = {**self.state.facts, **facts}
            self.state = SimpleNamespace(phase=target, facts=merged)
            return self.state

    store = FakeStore(tmp_path)
    monkeypatch.setattr(cli, "StateStore", lambda _root: store)
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_git_head", lambda: "a" * 40)
    monkeypatch.setattr(cli, "_resolve_input_revision", lambda *_a, **_k: "b" * 40)

    submitted = {"oar": 0, "controller": 0}
    monitored = {"calls": 0}

    class ReattachSsh:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, command: str) -> SimpleNamespace:
            if "$HOME" in command:
                return SimpleNamespace(stdout="/home/u")
            if "labeling.exit_code" in command:
                return SimpleNamespace(stdout="0")
            if "manifest.json" in command:
                return SimpleNamespace(stdout="yes")
            if "labeling.stdout.log" in command:
                return SimpleNamespace(stdout='{"commit_id":"' + "d" * 40 + '"}')
            return SimpleNamespace(stdout="")

    class ReattachOar:
        def __init__(self, _ssh: object, **_kwargs: object) -> None:
            self.statuses = [
                JobStatus(2895249, JobState.QUEUED),
                JobStatus(2895249, JobState.TERMINATED, exit_code=0),
            ]

        def status(self, job_id: int) -> JobStatus:
            if not self.statuses:
                return JobStatus(job_id, JobState.TERMINATED, exit_code=0)
            return self.statuses.pop(0)

        def submit(self, _request: object) -> int:
            submitted["oar"] += 1
            return 999

    class ReattachController:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def monitor(self, _job_id: int, *, log_name: str) -> JobState:
            monitored["calls"] += 1
            return JobState.TERMINATED

        def submit(self, **_kwargs: object) -> int:
            submitted["controller"] += 1
            return 999

    monkeypatch.setattr(cli, "SshClient", ReattachSsh)
    monkeypatch.setattr(cli, "OarClient", ReattachOar)
    monkeypatch.setattr(cli, "Stager", lambda _ssh: None)
    monkeypatch.setattr(cli, "Controller", ReattachController)
    monkeypatch.setattr(cli, "_remote_home", lambda _ssh: PurePosixPath("/home/u"))

    from osm_polygon_sentence_relevance.operator import recorded_job

    _inspection = recorded_job.ResumeInspection(
        exit_code=0,
        manifest_present=True,
        progress=recorded_job.ProgressFacts(
            completed=13952, total=54462, identity_matches=True
        ),
        checkpoint_pairs=3,
        checkpoint_parquet_shas_match=True,
        identity_matches=True,
    )
    monkeypatch.setattr(
        cli.recorded_job,
        "inspect_remote_resume",
        lambda *args, **kwargs: _inspection,
    )
    monkeypatch.setattr(
        cli.recorded_job,
        "classify_terminal",
        lambda status, insp: ExitClass.COMPLETE,
    )

    assert (
        cli.main(
            [
                "run",
                "--scope",
                "region",
                "--region",
                "afghanistan-latest",
                "--stage",
                "label",
                "--input-revision",
                "b" * 40,
            ]
        )
        == 0
    )
    assert monitored["calls"] == 1
    assert submitted == {"oar": 0, "controller": 0}


def test_operator_never_runs_inference_on_the_mac() -> None:
    # The Mac operator entry point must never import or invoke a local CUDA /
    # CPU / MPS inference engine, and must dispatch every GPU workload to a
    # remote OAR allocation that references the assigned compute node.
    operator_dir = Path(cli.__file__).parent
    source = "\n".join(path.read_text() for path in operator_dir.glob("*.py"))
    for module in ("torch", "llama_cpp", "vllm", "transformers", "mlx"):
        assert re.search(rf"(^|\W)import {module}\b", source, re.M) is None
        assert re.search(rf"(^|\W)from {module}\b", source, re.M) is None
    for token in ("os.system", "Popen(", '"mps"', "'mps'", "Metal", "METAL_"):
        assert token not in source

    layout = RemoteLayout(PurePosixPath("/r"))
    config = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )
    label = label_submission(
        config,
        layout,
        input_parquet=PurePosixPath("/r/in.parquet"),
        model_file=PurePosixPath("/r/m.gguf"),
        tokenizer_dir=PurePosixPath("/r/tok"),
    )
    build = llama_build_submission(layout)
    # Both inference-bearing payloads are remote OAR submissions.
    assert label.command[0].endswith("submit_afghanistan_labeling.sh")
    assert build.command[0].endswith("_submit_gpu_job.sh")
    assert "${OAR_JOB_ID" in build.command[-1]


def _patch_reattach_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    classification,
    inspection=None,
    status_state: JobState = JobState.QUEUED,
    monitor_returns: JobState = JobState.TERMINATED,
) -> tuple[SimpleNamespace, dict[str, int]]:
    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.oar import ExitClass

    config = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )
    store = SimpleNamespace(
        phase=RunPhase.QUEUED,
        facts={"job_id": 2895249, "site": "sophia", "active_stage": "label"},
        load_returns=SimpleNamespace(
            phase=RunPhase.QUEUED,
            facts={"job_id": 2895249, "site": "sophia", "active_stage": "label"},
        ),
        transitions=[],
    )

    def load() -> SimpleNamespace:
        return store.load_returns

    def transition(
        *, expected: RunPhase, target: RunPhase, facts: dict[str, object]
    ) -> SimpleNamespace:
        assert store.load_returns.phase is expected
        merged = {**store.load_returns.facts, **facts}
        store.load_returns = SimpleNamespace(phase=target, facts=merged)
        store.transitions.append((expected, target, dict(facts)))
        return store.load_returns

    store.load = load  # type: ignore[method-assign]
    store.transition = transition  # type: ignore[method-assign]

    counters = {"monitored": 0, "submitted": 0}

    class BranchSsh:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, command: str) -> SimpleNamespace:
            if "$HOME" in command:
                return SimpleNamespace(stdout="/home/u")
            if "labeling.stdout.log" in command:
                return SimpleNamespace(stdout='{"commit_id":"' + "d" * 40 + '"}')
            return SimpleNamespace(stdout="")

    class BranchOar:
        def __init__(self, _ssh: object, **_kwargs: object) -> None:
            self._first = True

        def status(self, job_id: int) -> JobStatus:
            if self._first:
                self._first = False
                return JobStatus(job_id, status_state)
            return JobStatus(
                job_id,
                monitor_returns
                if monitor_returns
                in {JobState.TERMINATED, JobState.ERROR, JobState.MISSING}
                else JobState.TERMINATED,
            )

        def submit(self, _request: object) -> int:
            counters["submitted"] += 1
            return 999

    class BranchController:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def monitor(self, _job_id: int, *, log_name: str) -> JobState:
            counters["monitored"] += 1
            return monitor_returns

        def submit(self, **_kwargs: object) -> int:
            counters["submitted"] += 1
            return 999

    monkeypatch.setattr(cli, "SshClient", BranchSsh)
    monkeypatch.setattr(cli, "OarClient", BranchOar)
    monkeypatch.setattr(cli, "Stager", lambda _ssh: None)
    monkeypatch.setattr(cli, "Controller", BranchController)
    monkeypatch.setattr(cli, "_remote_home", lambda _ssh: PurePosixPath("/home/u"))

    if inspection is None:
        inspection = recorded_job.ResumeInspection(
            exit_code=0,
            manifest_present=(classification is ExitClass.COMPLETE),
            progress=recorded_job.ProgressFacts(
                completed=13952,
                total=54462,
                identity_matches=True,
            ),
            checkpoint_pairs=3 if classification is ExitClass.CONTINUE else 0,
            checkpoint_parquet_shas_match=(classification is ExitClass.CONTINUE),
            identity_matches=True,
        )
    monkeypatch.setattr(
        cli.recorded_job,
        "inspect_remote_resume",
        lambda *args, **kwargs: inspection,
    )
    monkeypatch.setattr(
        cli.recorded_job,
        "classify_terminal",
        lambda status, insp: classification,
    )
    return config, store, counters


def _make_args() -> SimpleNamespace:
    return _run_args()


def test_reattach_graceful_deadline_exit_zero_with_valid_checkpoints_is_resumable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, store, counters = _patch_reattach_env(
        monkeypatch, classification=ExitClass.CONTINUE
    )
    cli._classify_or_continue(_make_args(), store, config, "sophia", 2895249)
    assert counters == {"monitored": 1, "submitted": 0}
    final = store.load_returns
    assert final.phase is RunPhase.REMOTE_PREPARED
    assert final.facts["continued_after_job"] == 2895249


def test_reattach_complete_split_runs_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumed split run must not stop at CHECKPOINTED."""

    config = OperatorConfig.build(
        scope="all",
        stage="split",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )
    store = _FakeStore(Path("/state"))
    store.value = SimpleNamespace(
        phase=RunPhase.QUEUED,
        facts={"active_stage": "split", "site": "sophia", "job_id": 42},
    )
    ssh = _FakeSsh(target="sophia")
    oar = _FakeOar(ssh)
    monkeypatch.setattr(
        cli,
        "_attach_to_site",
        lambda *_args, **_kwargs: (
            ssh,
            cli.RemoteLayout(PurePosixPath("/run")),
            oar,
            _FakeController(state=store),
        ),
    )
    monkeypatch.setattr(
        cli,
        "inspect_split_resume",
        lambda **_kwargs: SimpleNamespace(
            exit_code=0,
            checkpoint_count=1,
            total_shards=1,
            identity_matches=True,
        ),
    )
    monkeypatch.setattr(
        cli,
        "classify_split_terminal",
        lambda *_args, **_kwargs: ExitClass.COMPLETE,
    )
    monkeypatch.setattr(
        cli.split_finalization,
        "monitor_job_with_log",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        cli.split_finalization,
        "assert_remote_exit_zero",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(cli.split_finalization, "publish_split", lambda *_a: "c" * 40)
    monkeypatch.setattr(cli.split_finalization, "mark_remote_status", lambda *_a: None)

    result = cli._classify_or_continue(
        _run_args(stage="split"),
        store,
        config,
        "sophia",
        42,
        destination_site="sophia",
    )

    assert result is ExitClass.COMPLETE
    assert store.value.phase is RunPhase.COMPLETE
    assert store.value.facts["hub_commit"] == "c" * 40


def test_reattach_missing_job_raises_without_resubmitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store, counters = _patch_reattach_env(
        monkeypatch,
        status_state=JobState.MISSING,
        monitor_returns=JobState.MISSING,
        classification=ExitClass.FAILED,
    )
    with pytest.raises(RuntimeError, match="missing from OAR"):
        cli._classify_or_continue(_make_args(), store, config, "sophia", 2895249)
    assert counters == {"monitored": 0, "submitted": 0}


def test_reattach_deterministic_failure_does_not_resubmit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store, counters = _patch_reattach_env(
        monkeypatch, classification=ExitClass.FAILED
    )
    with pytest.raises(RuntimeError, match="failed deterministically"):
        cli._classify_or_continue(_make_args(), store, config, "sophia", 2895249)
    assert counters == {"monitored": 1, "submitted": 0}


def test_reattach_exit_zero_without_manifest_and_no_valid_checkpoints_fails_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, store, counters = _patch_reattach_env(
        monkeypatch, classification=ExitClass.FAILED
    )
    with pytest.raises(RuntimeError, match="failed deterministically"):
        cli._classify_or_continue(_make_args(), store, config, "sophia", 2895249)
    assert counters == {"monitored": 1, "submitted": 0}


def test_keyboard_interrupt_prints_invocation_specific_run_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ctrl+C echoes the run ID active for *this* invocation only.

    Two run directories exist in ``DATA_ROOT/runs``; the newer one is
    unrelated. The KeyboardInterrupt handler must echo the active
    invocation's run ID, never scanning by mtime.
    """

    prior = cli._ACTIVE_RUN_ID
    cli._ACTIVE_RUN_ID = None
    monkeypatch.setattr(cli, "_resume_command", lambda rid: f"resume {rid}")

    # Patch _resume_run so we exercise the KeyboardInterrupt handler
    # without invoking the rest of the flow.
    def fake_resume(run_id, args):
        cli._ACTIVE_RUN_ID = run_id
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "_resume_run", fake_resume)

    try:
        with pytest.raises(SystemExit) as exc:
            cli.main(["resume", "6578fb2269130a41d243"])
        assert exc.value.code == 130
        err = capsys.readouterr().err
        assert "6578fb2269130a41d243" in err
    finally:
        cli._ACTIVE_RUN_ID = prior


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (JobState.TERMINATED, True),
        (JobState.ERROR, True),
        (JobState.MISSING, False),
        (JobState.RUNNING, False),
    ],
)
def test_error_is_a_terminal_allocation_state(state: JobState, expected: bool) -> None:
    """Walltime transitions to OAR Error still enter checkpoint classification."""

    assert cli._is_terminal_allocation(state) is expected
