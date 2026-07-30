"""Public terminal contracts for the autonomous operator."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator import cli
from osm_polygon_sentence_relevance.operator.config import OperatorConfig
from osm_polygon_sentence_relevance.operator.controller import LiveProgress
from osm_polygon_sentence_relevance.operator.oar import ExitClass, JobState, JobStatus
from osm_polygon_sentence_relevance.operator.sites import SiteProbe
from osm_polygon_sentence_relevance.operator.staging import LabelAssets
from osm_polygon_sentence_relevance.operator.state import RunPhase
from osm_polygon_sentence_relevance.operator.workflows import (
    RemoteLayout,
    label_submission,
    llama_build_submission,
)


def test_help_exposes_run_status_and_public_stage_choices(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    assert "run" in text
    assert "status" in text
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run",
            "--scope",
            "region",
            "--region",
            "afghanistan-latest",
            "--stage",
            "all",
        ]
    )
    assert args.stage == "all"
    assert set(args.site) == {
        "bordeaux",
        "grenoble",
        "lille",
        "louvain",
        "luxembourg",
        "lyon",
        "nancy",
        "nantes",
        "rennes",
        "sophia",
        "strasbourg",
        "toulouse",
    }


def test_sigint_only_stops_local_monitoring(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(_args: object) -> int:
        raise KeyboardInterrupt

    parser = cli.build_parser()
    monkeypatch.setattr(cli, "build_parser", lambda: parser)
    monkeypatch.setattr(cli, "_run", interrupted)
    # Existing parser stores its original handler, so patch parsed arguments.
    args = parser.parse_args(
        ["run", "--scope", "all", "--stage", "split", "--input-revision", "a" * 40]
    )
    args.handler = interrupted
    monkeypatch.setattr(parser, "parse_args", lambda _argv: args)
    with pytest.raises(SystemExit) as exc:
        cli.main([])
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
        self.commands: list[str] = []

    def run(self, command: str) -> SimpleNamespace:
        self.commands.append(command)
        if "quota_output=$(quota" in command:
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


def test_run_split_finalizes_publishes_and_marks_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_run_fakes(monkeypatch, tmp_path)
    args = cli.build_parser().parse_args(
        [
            "run",
            "--scope",
            "region",
            "--region",
            "afghanistan-latest",
            "--stage",
            "split",
            "--site",
            "nancy",
            "--poll-seconds",
            "0",
        ]
    )
    assert cli._run(args) == 0
    state = _FakeStore.instances[-1].value
    assert state.phase is RunPhase.COMPLETE
    assert state.facts["published"] is True
    assert state.facts["hub_commit"] == "abcdef123456"
    assert "Sentence splitting complete" in capsys.readouterr().out


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
    args = cli.build_parser().parse_args(
        [
            "run",
            "--scope",
            "region",
            "--region",
            "afghanistan-latest",
            "--stage",
            "split",
            "--site",
            "nancy",
            "--poll-seconds",
            "0",
        ]
    )
    args.site = ["nancy"]
    assert cli._run(args) == 0
    assert cleaned == [True]


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
    args = cli.build_parser().parse_args(
        [
            "run",
            "--scope",
            "region",
            "--region",
            "afghanistan-latest",
            "--stage",
            "label",
            "--site",
            "nancy",
            "--poll-seconds",
            "0",
        ]
    )
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

    with pytest.raises(RuntimeError, match="failed deterministically"):
        cli._apply_classification(
            store=object(),  # type: ignore[arg-type]
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
    parser = cli.build_parser()
    args = parser.parse_args(["status", "a" * 20])
    args.handler = lambda _args: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(parser, "parse_args", lambda _argv: args)
    monkeypatch.setattr(cli, "build_parser", lambda: parser)
    assert cli.main([]) == 1
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


def _make_args() -> argparse.Namespace:
    parser = cli.build_parser()
    return parser.parse_args(
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
