"""End-to-end RED tests for ``resume RUN_ID`` lifecycle.

These tests drive the full continuation path:

  allocation 1 terminal/resumable
    → cross-site relay (or same-site reuse)
    → allocation 2 submitted exactly once
    → allocation 2 complete (no third allocation)

The tests also verify same-site continuation avoids any relay transfer and
that the historical ``resume RUN_ID`` command never submits a duplicate of
the prior allocation.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
from osm_polygon_sentence_relevance.labeling.contracts import (
    LabelRecord,
    LabelValue,
    RunIdentity,
)
from osm_polygon_sentence_relevance.operator import cli, relay
from osm_polygon_sentence_relevance.operator import (
    recorded_job as recorded,
)
from osm_polygon_sentence_relevance.operator.config import OperatorConfig
from osm_polygon_sentence_relevance.operator.oar import ExitClass, JobState, JobStatus
from osm_polygon_sentence_relevance.operator.state import RunPhase

# ------------------------------------------------------------------
# Stateful fake infrastructure for the full lifecycle
# ------------------------------------------------------------------


def _identity() -> RunIdentity:
    return RunIdentity(
        input_sha256="a" * 64,
        input_dataset_revision="b" * 40,
        model_repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        model_revision="5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace",
        model_file="Qwen3.6-27B-Q4_K_M.gguf",
        model_file_sha256="a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f",
        prompt_version="afghanistan-landuse-polygon-v2",
        source_commit="d" * 40,
        engine="llama.cpp",
        engine_version="b1234",
        batch_size=128,
        row_limit=0,
        llama_parallel=16,
        llama_per_slot_context=4096,
        llama_total_context=65536,
        request_concurrency=16,
    )


def _build_remote_checkpoint_set(path: Path) -> None:
    identity = _identity()
    store = CheckpointStore(path, identity)
    records = [
        LabelRecord(
            sentence_id=f"s{i:08d}",
            landuse_relevance=LabelValue.YES,
            polygon_relevance=LabelValue.YES,
            landuse_reason="x",
            polygon_reason="y",
            evidence="z",
        )
        for i in range(4)
    ]
    store.write_batch(0, records[:2])
    store.write_batch(1, records[2:])
    store.write_progress(completed=4, total=20, elapsed_seconds=10.0)
    store.write_timing({"started_at": 1.0, "finished_at": 2.0})


@dataclass
class _FakeOar:
    """OAR client that returns scripted statuses keyed by job id."""

    scripted: dict[int, list[JobStatus]] = field(default_factory=dict)
    submitted: list[int] = field(default_factory=list)
    next_job_id: int = 7000

    def status(self, job_id: int) -> JobStatus:
        queue = self.scripted.get(job_id, [])
        if queue:
            return queue.pop(0)
        return JobStatus(job_id, JobState.TERMINATED)

    def submit(self, _request: Any) -> int:
        self.next_job_id += 1
        self.submitted.append(self.next_job_id)
        return self.next_job_id


@dataclass
class _FakeSsh:
    """SSH client that resolves remote paths against a per-site root."""

    site_root: Path
    progress_path: Path | None = None
    exit_text: str = "0\n"
    manifest_present: bool = False
    overrides: dict[str, str] = field(default_factory=dict)

    def run(self, command: str) -> Any:
        from osm_polygon_sentence_relevance.operator.ssh import LogChunk

        stripped = command.strip()
        if "$HOME" in stripped:
            return LogChunk(text="/home/u\n", next_offset=0, eof=True)
        if stripped.startswith("test -d") and "find" in stripped:
            tokens = stripped.split()
            # The find path is the token after "find"; the trailing
            # ``-mindepth/-maxdepth 1`` arguments are not the path.
            find_idx = tokens.index("find")
            path = tokens[find_idx + 1]
            local = self._map(path)
            if local.is_dir():
                listing = sorted(
                    f"{('d' if c.is_dir() else 'f')}\t{c.name}\n"
                    for c in local.iterdir()
                )
                return LogChunk(text="".join(listing), next_offset=0, eof=True)
            return LogChunk(text="", next_offset=0, eof=True)
        if "test -f" in stripped and "manifest.json" in stripped:
            if self.manifest_present:
                return LogChunk(text="yes", next_offset=0, eof=True)
            return LogChunk(text="no", next_offset=0, eof=True)
        if "test -f" in stripped and "cat" in stripped:
            local = self._map(stripped.split("cat ")[-1])
            if local.is_file():
                return LogChunk(text=local.read_text(), next_offset=0, eof=True)
            return LogChunk(text="", next_offset=0, eof=True)
        if "sha256sum" in stripped:
            tokens = stripped.split()
            remote_path = tokens[-1]
            actual = self._sha_for(remote_path)
            return LogChunk(text=f"{actual}  {remote_path}\n", next_offset=0, eof=True)
        if stripped.startswith("find ") and "checkpoints" in stripped:
            path = stripped.split()[1]
            local = self._map(path)
            lines = []
            if local.is_dir():
                for child in sorted(local.iterdir()):
                    kind = (
                        "f"
                        if child.is_file() and not child.is_symlink()
                        else "l"
                        if child.is_symlink()
                        else "d"
                        if child.is_dir()
                        else "o"
                    )
                    lines.append(f"{kind}\t{child.name}\n")
            return LogChunk(text="".join(lines), next_offset=0, eof=True)
        return LogChunk(text="", next_offset=0, eof=True)

    def read_since(self, path: str, offset: int) -> Any:
        from osm_polygon_sentence_relevance.operator.ssh import LogChunk

        local = self._map(path)
        if local.is_file():
            return LogChunk(text=local.read_text(), next_offset=0, eof=True)
        if path in self.overrides:
            return LogChunk(text=self.overrides[path], next_offset=0, eof=True)
        raise recorded.ResumeError(f"missing {path}")

    def _sha_for(self, remote_path: str) -> str:
        import hashlib

        local = self._map(remote_path)
        if local.is_file():
            return hashlib.sha256(local.read_bytes()).hexdigest()
        return "0" * 64

    def _map(self, remote_path: str) -> Path:
        rel = remote_path.lstrip("/")
        return self.site_root / rel


@dataclass
class _FakeController:
    monitored: list[int] = field(default_factory=list)
    submitted: list[int] = field(default_factory=list)
    next_job_id: int = 7000
    # Optional reference to a shared ``_FakeOar`` so the operator's
    # use of ``controller.submit`` is observable via ``oar.submitted``.
    oar: _FakeOar | None = None
    store: _FakeStore | None = None

    def monitor(self, job_id: int, *, log_name: str) -> JobState:
        self.monitored.append(job_id)
        if self.store is not None and self.store.load().phase is RunPhase.SUBMITTED:
            self.store.transition(
                expected=RunPhase.SUBMITTED,
                target=RunPhase.QUEUED,
                facts={"job_id": job_id},
            )
        return JobState.TERMINATED

    def submit(
        self,
        *,
        component: Any = None,
        input_parquet: Any = None,
        model_file: Any = None,
        tokenizer_dir: Any = None,
    ) -> int:
        self.next_job_id += 1
        self.submitted.append(self.next_job_id)
        if self.oar is not None:
            self.oar.submitted.append(self.next_job_id)
        if self.store is not None:
            self.store.transition(
                expected=RunPhase.REMOTE_PREPARED,
                target=RunPhase.SUBMITTED,
                facts={
                    "job_id": self.next_job_id,
                    "log_offset": 0,
                    "active_stage": "label",
                },
            )
        return self.next_job_id


@dataclass
class _FakeStore:
    """State store with load/transition/facts tracking."""

    state: Any
    transitions: list[tuple[Any, Any, dict]] = field(default_factory=list)

    def load(self) -> Any:
        return self.state

    def transition(self, *, expected: Any, target: Any, facts: dict[str, Any]) -> Any:
        if self.state.phase is not expected:
            raise RuntimeError(f"unexpected phase {self.state.phase} != {expected}")
        merged = {**self.state.facts, **facts}
        self.state = SimpleNamespace(phase=target, facts=merged)
        self.transitions.append((expected, target, dict(facts)))
        return self.state


# ------------------------------------------------------------------
# Test: terminal → cross-site relay → new submission → complete
# ------------------------------------------------------------------


def test_resume_reroutes_cross_site_and_submits_exactly_one_new_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seagate = tmp_path / "seagate"
    seagate.mkdir()
    (seagate / "runs").mkdir()
    monkeypatch.setattr(cli, "DATA_ROOT", seagate)

    # Remote sophia filesystem: terminal/resumable job 2895249
    sophia_root = tmp_path / "frontend-sophia"
    sophia_root.mkdir()
    sophia_remote = (
        sophia_root
        / "home"
        / "u"
        / "osm-polygon-operator"
        / "6578fb2269130a41d243"
        / "label-work"
    )
    sophia_remote.mkdir(parents=True)
    _build_remote_checkpoint_set(sophia_remote)
    # Add manifest at sophia label-output? No — resumable.
    sophia_ssh = _FakeSsh(
        site_root=sophia_root,
        progress_path=sophia_remote / "progress.json",
        exit_text="0\n",
        manifest_present=False,
    )

    # Remote grenoble filesystem: will receive the relay, then a final allocation
    grenoble_root = tmp_path / "frontend-grenoble"
    grenoble_root.mkdir()
    grenoble_remote = (
        grenoble_root / "home" / "u" / "osm-polygon-operator" / "6578fb2269130a41d243"
    )
    grenoble_remote.mkdir(parents=True)
    # The relay will be staged to grenoble_root/home/u/.../label-work by
    # the orchestrator. Prepare a final manifest under label-output so the
    # first new allocation after staging classifies as COMPLETE.
    (grenoble_remote / "label-output").mkdir(exist_ok=True)
    (grenoble_remote / "label-output" / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "identity": _identity().to_dict()})
    )
    logs_dir = grenoble_remote / "logs" / "7001"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "labeling.stdout.log").write_text(json.dumps({"commit_id": "a" * 40}))
    (logs_dir / "labeling.exit_code").write_text("0")
    grenoble_ssh = _FakeSsh(
        site_root=grenoble_root,
        manifest_present=True,
    )

    # State: job 2895249 is terminal/queued on sophia
    live = SimpleNamespace(
        phase=RunPhase.QUEUED,
        facts={"job_id": 2895249, "site": "sophia", "active_stage": "label"},
    )
    store = _FakeStore(state=live)
    oar = _FakeOar(
        scripted={
            2895249: [JobStatus(2895249, JobState.TERMINATED)],
            # After the relay, a new job 7001 is submitted on grenoble and
            # ends in TERMINATED, which classify_terminal will mark COMPLETE
            # because the (mocked) inspection shows a manifest.
            7001: [JobStatus(7001, JobState.TERMINATED)],
        }
    )

    ssh_by_site = {"sophia": sophia_ssh, "grenoble": grenoble_ssh}
    controllers_by_site: dict[str, _FakeController] = {}

    def _attach_to_site(
        _store: Any, config: OperatorConfig, site: str, *, poll_seconds: float
    ) -> tuple[Any, Any, _FakeOar, Any]:
        from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout

        ssh = ssh_by_site[site]
        from osm_polygon_sentence_relevance.operator.cli import _remote_home

        layout = RemoteLayout(
            _remote_home(ssh) / "osm-polygon-operator" / config.run_id
        )
        if site not in controllers_by_site:
            controllers_by_site[site] = _FakeController(oar=oar, store=store)
        return ssh, layout, oar, controllers_by_site[site]

    monkeypatch.setattr(cli, "_attach_to_site", _attach_to_site)

    # Subprocess stub for scp/ssh on the relay path.
    import subprocess as _subprocess

    def fake_run(
        argv: Sequence[str],
        *,
        check: bool = False,
        shell: bool = False,
        timeout: float | None = None,
        capture_output: bool = False,
        text: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        cmd = list(argv)
        if cmd[0] == "scp":
            src, dst = cmd[-2], cmd[-1]
            if src.startswith("sophia:"):
                _, remote = src.split(":", 1)
                local = dst
            elif dst.startswith("grenoble:"):
                local = src
                _, remote = dst.split(":", 1)
            elif src.startswith("grenoble:"):
                _, remote = src.split(":", 1)
                local = dst
            else:
                local = src
                _, remote = dst.split(":", 1)
            src_local = sophia_root / remote.lstrip("/")
            dst_local = grenoble_root / remote.lstrip("/")
            if src.startswith("sophia") or src.startswith("grenoble"):
                src_local = (
                    sophia_root if src.startswith("sophia") else grenoble_root
                ) / remote.lstrip("/")
                dst_local = Path(local)
                dst_local.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                dst_local.write_bytes(src_local.read_bytes())
                os.chmod(dst_local, 0o600)
                return _subprocess.CompletedProcess(cmd, 0, "", "")
            if dst.startswith("sophia") or dst.startswith("grenoble"):
                target = (
                    sophia_root if dst.startswith("sophia") else grenoble_root
                ) / remote.lstrip("/")
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                target.write_bytes(Path(local).read_bytes())
                os.chmod(target, 0o600)
                return _subprocess.CompletedProcess(cmd, 0, "", "")
            # Both endpoints are local (download from sophia to local Seagate).
            if src.startswith("sophia"):
                src_local = sophia_root / remote.lstrip("/")
                Path(local).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                Path(local).write_bytes(src_local.read_bytes())
                os.chmod(Path(local), 0o600)
            else:
                # src = local, dst = sophia or grenoble
                local_path = Path(local)
                if dst.startswith("grenoble"):
                    target = grenoble_root / remote.lstrip("/")
                else:
                    target = sophia_root / remote.lstrip("/")
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                target.write_bytes(local_path.read_bytes())
                os.chmod(target, 0o600)
            return _subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == "ssh":
            inner = " ".join(cmd[2:])
            if inner.startswith("find "):
                path = inner.split()[1]
                site_root = (
                    sophia_root if cmd[1].startswith("sophia") else grenoble_root
                )
                local = site_root / path.lstrip("/")
                lines = []
                if local.is_dir():
                    for child in sorted(local.iterdir()):
                        kind = (
                            "f"
                            if child.is_file() and not child.is_symlink()
                            else "l"
                            if child.is_symlink()
                            else "d"
                            if child.is_dir()
                            else "o"
                        )
                        lines.append(f"{kind}\t{child.name}\n")
                return _subprocess.CompletedProcess(cmd, 0, "".join(lines), "")
            if "install -d -m 0700" in inner:
                site_root = (
                    sophia_root if cmd[1].startswith("sophia") else grenoble_root
                )
                remote_path = inner.split()[-1]
                target = site_root / remote_path.lstrip("/")
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
                return _subprocess.CompletedProcess(cmd, 0, "", "")
            if inner.startswith("if [ -e ") and "; mv -- " in inner:
                site_root = (
                    sophia_root if cmd[1].startswith("sophia") else grenoble_root
                )
                mv_idx = inner.find("; mv -- ")
                tail = inner[mv_idx + len("; mv -- ") :]
                src_str, dst_str = tail.split(maxsplit=1)
                src = site_root / src_str.lstrip("/")
                dst = site_root / dst_str.lstrip("/")
                if dst.exists():
                    if dst.is_dir():
                        import shutil as _sh

                        _sh.rmtree(dst)
                    else:
                        dst.unlink()
                os.replace(src, dst)
                return _subprocess.CompletedProcess(cmd, 0, "", "")
            return _subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr(relay.subprocess, "run", fake_run)

    # Patch SshClient construction to avoid real SSH.
    from osm_polygon_sentence_relevance.operator import ssh as _ssh_module

    class _NoSsh:
        def run(self, _cmd):
            from osm_polygon_sentence_relevance.operator.ssh import LogChunk

            return LogChunk(text="", next_offset=0, eof=True)

        def read_since(self, _p, _o):
            from osm_polygon_sentence_relevance.operator.ssh import LogChunk

            return LogChunk(text="", next_offset=0, eof=True)

    def _fake_ssh_ctor(*_a, **_kw):
        return _NoSsh()

    monkeypatch.setattr(_ssh_module, "SshClient", _fake_ssh_ctor)
    monkeypatch.setattr(cli, "SshClient", _fake_ssh_ctor)
    monkeypatch.setattr(cli, "_remote_home", lambda _s: Path("/home/u"))
    monkeypatch.setattr(cli, "_usage_policy_preflight", lambda *_a: None)
    monkeypatch.setattr(cli, "_storage_preflight", lambda *_a, **_kw: None)

    class _Stager:
        def __init__(self, _ssh: Any) -> None:
            pass

        def prepare(self, _config: Any, layout: Any) -> Any:
            return SimpleNamespace(layout=layout, reused=False)

        def prepare_label_assets(
            self, _config: Any, layout: Any, *, download_input: bool
        ) -> Any:
            assert download_input is True
            return SimpleNamespace(
                input_parquet=layout.root / "input/sentences.parquet",
                model_file=layout.root / "model/model.gguf",
                tokenizer_dir=layout.root / "tokenizer",
                llama_server_ready=True,
            )

    monkeypatch.setattr(cli, "Stager", _Stager)

    # Stub availability probe so we deterministically pick grenoble.
    def fake_probe(target: str, run_id: str | None = None) -> Any:
        from osm_polygon_sentence_relevance.operator.sites import SiteProbe

        name = target.split("@")[-1].split(".")[0]
        return SiteProbe(
            name=name,
            target=target,
            reachable=True,
            gpu_memory_mb=80_000,
            cuda_capability=(8, 0),
            persistent_free_bytes=10 * 1024**3,
            queued_jobs=0,
            idle_compatible=(name == "grenoble"),
            has_managed_run=False,
        )

    monkeypatch.setattr(cli, "probe_site", fake_probe)

    config = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="d" * 40,
        input_revision="b" * 40,
    )

    # Patch recorded_job inspect/classify to use the actual on-disk fixtures
    # via the real ssh fake.
    def fake_inspect(
        ssh, *, label_work_root, label_output_root, expected_identity, exit_file
    ) -> Any:
        from osm_polygon_sentence_relevance.operator.recorded_job import (
            ProgressFacts,
            ResumeInspection,
        )

        local_work = ssh._map(label_work_root)
        progress = json.loads((local_work / "progress.json").read_text())
        manifests = (local_work.parent / "label-output" / "manifest.json").exists()
        batch_count = 0
        ckpts = local_work / "checkpoints"
        if ckpts.is_dir():
            seen = set()
            for p in ckpts.iterdir():
                m = __import__("re").fullmatch(r"batch-(\d{6})\.parquet", p.name)
                if m:
                    seen.add(int(m.group(1)))
            batch_count = len(seen)
        return ResumeInspection(
            exit_code=0,
            manifest_present=manifests,
            progress=ProgressFacts(
                completed=progress["completed"],
                total=progress["remaining"] + progress["completed"],
                identity_matches=True,
            ),
            checkpoint_pairs=batch_count,
            checkpoint_parquet_shas_match=batch_count > 0,
            identity_matches=True,
            checkpoint_indexes=tuple(range(batch_count)),
        )

    monkeypatch.setattr(cli.recorded_job, "inspect_remote_resume", fake_inspect)
    # Use real classification logic.
    # (cli.recorded_job is the same module as ``recorded``.)

    # Drive the resumable classification.
    initial = cli._classify_or_continue(
        cli.build_parser().parse_args(["resume", "7e8f1a748497e3dbcc56"]),
        store=store,
        config=config,
        site="sophia",
        job_id=2895249,
        destination_site="grenoble",
    )
    assert initial is ExitClass.COMPLETE
    # Exactly one new allocation was submitted on the destination site.
    assert oar.submitted == [7001]
    # The prior recorded job was never resubmitted.
    assert 2895249 not in oar.submitted


def test_resume_same_site_avoids_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seagate = tmp_path / "seagate"
    seagate.mkdir()
    sophia_root = tmp_path / "frontend-sophia"
    sophia_root.mkdir()
    remote = (
        sophia_root
        / "home"
        / "u"
        / "osm-polygon-operator"
        / "6578fb2269130a41d243"
        / "label-work"
    )
    remote.mkdir(parents=True)
    _build_remote_checkpoint_set(remote)
    (remote.parent / "label-output").mkdir(exist_ok=True)
    (remote.parent / "label-output" / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "identity": _identity().to_dict()})
    )
    logs_dir = remote.parent / "logs" / "7001"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "labeling.stdout.log").write_text(json.dumps({"commit_id": "a" * 40}))
    (logs_dir / "labeling.exit_code").write_text("0")
    ssh = _FakeSsh(site_root=sophia_root, exit_text="0\n", manifest_present=False)

    live = SimpleNamespace(
        phase=RunPhase.QUEUED,
        facts={"job_id": 2895249, "site": "sophia", "active_stage": "label"},
    )
    store = _FakeStore(state=live)
    oar = _FakeOar(
        scripted={
            2895249: [JobStatus(2895249, JobState.TERMINATED)],
            7001: [JobStatus(7001, JobState.TERMINATED)],
        }
    )

    def _attach(site: str) -> Any:
        from osm_polygon_sentence_relevance.operator.cli import _remote_home
        from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout

        layout = RemoteLayout(
            _remote_home(ssh) / "osm-polygon-operator" / "6578fb2269130a41d243"
        )
        return ssh, layout, oar, _FakeController(oar=oar, store=store)

    monkeypatch.setattr(
        cli,
        "_attach_to_site",
        lambda *args, **kwargs: _attach(args[2]),
    )

    # Track all subprocess.run calls; same-site must not invoke scp.
    scp_calls: list[list[str]] = []
    ssh_calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        cmd = list(argv)
        if cmd[0] == "scp":
            scp_calls.append(cmd)
        elif cmd[0] == "ssh":
            ssh_calls.append(cmd)
        import subprocess as _sp

        return _sp.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(relay.subprocess, "run", fake_run)

    inspections = 0

    def fake_inspect(ssh, **kwargs):
        nonlocal inspections
        inspections += 1
        from osm_polygon_sentence_relevance.operator.recorded_job import (
            ProgressFacts,
            ResumeInspection,
        )

        local_work = ssh._map(kwargs["label_work_root"])
        progress = json.loads((local_work / "progress.json").read_text())
        ckpts = local_work / "checkpoints"
        batch_count = (
            sum(1 for _ in ckpts.glob("batch-*.parquet")) if ckpts.is_dir() else 0
        )
        return ResumeInspection(
            exit_code=0,
            manifest_present=inspections > 1,
            progress=ProgressFacts(
                completed=progress["completed"],
                total=progress["remaining"] + progress["completed"],
                identity_matches=True,
            ),
            checkpoint_pairs=batch_count,
            checkpoint_parquet_shas_match=batch_count > 0,
            identity_matches=True,
            checkpoint_indexes=tuple(range(batch_count)),
        )

    monkeypatch.setattr(cli.recorded_job, "inspect_remote_resume", fake_inspect)

    config = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="d" * 40,
        input_revision="b" * 40,
    )

    classification = cli._classify_or_continue(
        cli.build_parser().parse_args(["resume", "7e8f1a748497e3dbcc56"]),
        store=store,
        config=config,
        site="sophia",
        job_id=2895249,
        destination_site="sophia",  # SAME SITE
    )
    assert classification is ExitClass.COMPLETE
    assert oar.submitted == [7001]
    # Same-site continuation must not invoke scp at all.
    assert scp_calls == []
    # And it must still submit exactly one new allocation on sophia.
    assert oar.submitted == [7001]


def test_missing_job_with_complete_durable_evidence_classifies_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A MISSING OAR job with full durable evidence still classifies COMPLETE."""

    seagate = tmp_path / "seagate"
    seagate.mkdir()
    sophia_root = tmp_path / "frontend-sophia"
    sophia_root.mkdir()
    remote = (
        sophia_root
        / "home"
        / "u"
        / "osm-polygon-operator"
        / "6578fb2269130a41d243"
        / "label-work"
    )
    remote.mkdir(parents=True)
    _build_remote_checkpoint_set(remote)
    (remote.parent / "label-output").mkdir(exist_ok=True)
    (remote.parent / "label-output" / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "identity": _identity().to_dict()})
    )
    logs_dir = remote.parent / "logs" / "2895249"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "labeling.stdout.log").write_text(json.dumps({"commit_id": "f" * 40}))
    ssh = _FakeSsh(site_root=sophia_root, exit_text="0\n", manifest_present=True)

    live = SimpleNamespace(
        phase=RunPhase.QUEUED,
        facts={"job_id": 2895249, "site": "sophia", "active_stage": "label"},
    )
    store = _FakeStore(state=live)
    oar = _FakeOar(scripted={2895249: [JobStatus(2895249, JobState.MISSING)]})

    from osm_polygon_sentence_relevance.operator.cli import _remote_home
    from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout

    layout = RemoteLayout(
        _remote_home(ssh) / "osm-polygon-operator" / "6578fb2269130a41d243"
    )
    monkeypatch.setattr(
        cli,
        "_attach_to_site",
        lambda *a, **k: (ssh, layout, oar, _FakeController()),
    )

    def fake_inspect(ssh, **kwargs):
        from osm_polygon_sentence_relevance.operator.recorded_job import (
            ProgressFacts,
            ResumeInspection,
        )

        local_work = ssh._map(kwargs["label_work_root"])
        progress = json.loads((local_work / "progress.json").read_text())
        return ResumeInspection(
            exit_code=0,
            manifest_present=True,
            progress=ProgressFacts(
                completed=progress["completed"],
                total=progress["remaining"] + progress["completed"],
                identity_matches=True,
            ),
            checkpoint_pairs=2,
            checkpoint_parquet_shas_match=True,
            identity_matches=True,
            checkpoint_indexes=(0, 1),
        )

    monkeypatch.setattr(cli.recorded_job, "inspect_remote_resume", fake_inspect)

    config = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="d" * 40,
        input_revision="b" * 40,
    )

    classification = cli._classify_or_continue(
        cli.build_parser().parse_args(["resume", "7e8f1a748497e3dbcc56"]),
        store=store,
        config=config,
        site="sophia",
        job_id=2895249,
    )
    assert classification is ExitClass.COMPLETE
    # No new submission: a complete run does not resubmit.
    assert oar.submitted == []
