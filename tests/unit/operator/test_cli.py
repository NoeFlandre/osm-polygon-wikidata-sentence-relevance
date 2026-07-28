"""Public terminal contracts for the autonomous operator."""

from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator import cli
from osm_polygon_sentence_relevance.operator.controller import LiveProgress
from osm_polygon_sentence_relevance.operator.oar import JobState, JobStatus
from osm_polygon_sentence_relevance.operator.sites import SiteProbe
from osm_polygon_sentence_relevance.operator.staging import LabelAssets
from osm_polygon_sentence_relevance.operator.state import RunPhase


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
    assert cli.main([]) == 130
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


def test_probe_failure_is_incompatible(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenSsh:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _command: str) -> object:
            raise ValueError("offline")

    monkeypatch.setattr(cli, "SshClient", BrokenSsh)
    probe = cli._probe_target("nancy")
    assert not probe.reachable


def test_live_progress_is_rendered(capsys: pytest.CaptureFixture[str]) -> None:
    cli._emit(LiveProgress(42, "build.stdout.log", "one\ntwo\n", 8))
    assert capsys.readouterr().out.splitlines() == ["[job 42] one", "[job 42] two"]


def test_site_probe_parses_frontend_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        stdout = "1000 80000 3\n"

    class FakeSsh:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, _command: str) -> Result:
            return Result()

    monkeypatch.setattr(cli, "SshClient", FakeSsh)
    assert cli._probe_target("nancy") == SiteProbe(
        "nancy", "nancy", True, 80_000, (7, 0), 1_024_000, 180
    )


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


def test_remote_home_and_exit_code_validation() -> None:
    class FakeSsh:
        def __init__(self, outputs: list[str]) -> None:
            self.outputs = outputs

        def run(self, _command: str) -> SimpleNamespace:
            return SimpleNamespace(stdout=self.outputs.pop(0))

    assert cli._remote_home(FakeSsh(["/home/user\n"])) == PurePosixPath("/home/user")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="invalid"):
        cli._remote_home(FakeSsh(["relative\n"]))  # type: ignore[arg-type]
    layout = cli.RemoteLayout(PurePosixPath("/r"))
    assert cli._remote_exit_code(FakeSsh(["0\n"]), layout, 1, "exit") == 0  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="invalid"):
        cli._remote_exit_code(FakeSsh(["bad\n"]), layout, 1, "exit")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="non-zero"):
        cli._assert_remote_exit_zero(FakeSsh(["2\n"]), layout, 1, "exit")  # type: ignore[arg-type]


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
    def __init__(self, _ssh: object) -> None:
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
        "_probe_target",
        lambda target: SiteProbe(
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


def test_run_label_reuses_checkpoints_and_completes(
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
    assert "Labeling complete" in capsys.readouterr().out


def test_simple_monitors_and_public_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ssh = _FakeSsh()
    layout = cli.RemoteLayout(PurePosixPath("/r"))
    oar = _FakeOar(ssh)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    cli._monitor_simple(ssh, oar, layout, 1, "log", 0)  # type: ignore[arg-type]
    cli._monitor_without_log(oar, 1, 0)  # type: ignore[arg-type]
    assert (
        cli._publish_split(  # type: ignore[arg-type]
            ssh, layout, PurePosixPath("/out"), "owner/data"
        )
        == "abcdef123456"
    )
    cli._mark_remote_status(ssh, layout, "failed")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="managed status"):
        cli._mark_remote_status(ssh, layout, "unknown")  # type: ignore[arg-type]


def test_terminal_transition_rejects_unexpected_phase(tmp_path: Path) -> None:
    state = _FakeStore(tmp_path)
    with pytest.raises(RuntimeError, match="unexpected durable"):
        cli._transition_terminal(
            state,  # type: ignore[arg-type]
            expected=(RunPhase.RUNNING,),
            target=RunPhase.COMPLETE,
            facts={},
        )


def test_label_publication_commit_requires_immutable_json() -> None:
    layout = cli.RemoteLayout(PurePosixPath("/r"))
    ssh = _FakeSsh()
    assert cli._label_publication_commit(ssh, layout, 1) == "c" * 40  # type: ignore[arg-type]

    class InvalidSsh:
        def run(self, _command: str) -> SimpleNamespace:
            return SimpleNamespace(stdout='noise\n{"commit_id":"short"}\n')

    with pytest.raises(RuntimeError, match="immutable Hub commit"):
        cli._label_publication_commit(InvalidSsh(), layout, 1)  # type: ignore[arg-type]


def test_monitor_helpers_reject_remote_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ErrorOar:
        def status(self, job_id: int) -> JobStatus:
            return JobStatus(job_id, JobState.ERROR, exit_code=1)

    ssh = _FakeSsh()
    layout = cli.RemoteLayout(PurePosixPath("/r"))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="allocation failed"):
        cli._monitor_simple(ssh, ErrorOar(), layout, 1, "log", 0)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="build allocation"):
        cli._monitor_without_log(ErrorOar(), 1, 0)  # type: ignore[arg-type]


def test_status_missing_and_cleanup_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "DATA_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="does not exist"):
        cli._status(SimpleNamespace(run_id="a" * 20))
    monkeypatch.setattr(cli, "_cleanup_remote", lambda _ssh, execute: ())
    monkeypatch.setattr(cli, "SshClient", _FakeSsh)
    assert cli._cleanup(SimpleNamespace(site="nancy", execute=False)) == 0
    assert "No pipeline-managed" in capsys.readouterr().out
    monkeypatch.setattr(
        cli, "_cleanup_remote", lambda _ssh, execute: ("/home/user/run",)
    )
    assert cli._cleanup(SimpleNamespace(site="nancy", execute=True)) == 0
    assert "removed: /home/user/run" in capsys.readouterr().out


def test_remote_cleanup_builds_guarded_preview_and_delete_scripts() -> None:
    class CleanupSsh:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def run(self, command: str) -> SimpleNamespace:
            self.commands.append(command)
            return SimpleNamespace(stdout="/home/user/run-a\n/home/user/run-b\n")

    ssh = CleanupSsh()
    assert cli._cleanup_remote(ssh, execute=False) == (  # type: ignore[arg-type]
        "/home/user/run-a",
        "/home/user/run-b",
    )
    assert "rm -rf" in ssh.commands[0]
    assert "[ preview = delete ]" in ssh.commands[0]
    cli._cleanup_remote(ssh, execute=True)  # type: ignore[arg-type]
    assert "[ delete = delete ]" in ssh.commands[1]


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
