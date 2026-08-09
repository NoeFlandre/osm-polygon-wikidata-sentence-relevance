"""Integration tests for the CLI's queued-job replacement adapter."""

from __future__ import annotations

import shlex
import time
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any

from osm_polygon_sentence_relevance.operator import cli
from osm_polygon_sentence_relevance.operator.config import OperatorConfig
from osm_polygon_sentence_relevance.operator.oar import JobState, JobStatus
from osm_polygon_sentence_relevance.operator.sites import SiteProbe
from osm_polygon_sentence_relevance.operator.staging import LabelAssets
from osm_polygon_sentence_relevance.operator.state import RunPhase, StateStore


def _config() -> OperatorConfig:
    return OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="a" * 40,
        input_revision="b" * 40,
        llama_parallel=8,
        llama_per_slot_context=8192,
    )


def _queued_store(
    tmp_path: Any,
    *,
    config: OperatorConfig | None = None,
    active_stage: str = "label",
) -> tuple[OperatorConfig, StateStore]:
    config = config or _config()
    store = StateStore(tmp_path)
    store.load_or_create(config.run_identity)
    phase = RunPhase.CREATED
    for target in (
        RunPhase.INPUTS_RESOLVED,
        RunPhase.SITE_SELECTED,
        RunPhase.STORAGE_READY,
        RunPhase.REMOTE_PREPARED,
        RunPhase.SUBMITTED,
        RunPhase.QUEUED,
    ):
        store.transition(
            expected=phase,
            target=target,
            facts={"site": "sophia", "job_id": 42, "active_stage": active_stage},
        )
        phase = target
    return config, store


def _split_config() -> OperatorConfig:
    return OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="split",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )


class _Ssh:
    def __init__(self, *, target: str, **_kwargs: object) -> None:
        self.target = target

    def run(self, _command: str) -> SimpleNamespace:
        return SimpleNamespace(stdout="/home/u\n")


class _Stager:
    def __init__(self, _ssh: _Ssh) -> None:
        pass

    def prepare(self, _config: Any, layout: Any) -> SimpleNamespace:
        return SimpleNamespace(layout=layout, reused=True)

    def prepare_label_assets(
        self,
        _config: Any,
        layout: Any,
        *,
        download_input: bool,
    ) -> LabelAssets:
        assert download_input
        return LabelAssets(
            PurePosixPath(layout.root) / "input/sentences.parquet",
            PurePosixPath(layout.root) / "model/model.gguf",
            PurePosixPath(layout.root) / "tokenizer",
            True,
        )


def _ready_probe(site: str) -> SiteProbe:
    return SiteProbe(
        site,
        site,
        True,
        80_000,
        (8, 0),
        100 * 1024**3,
        0,
        idle_compatible=True,
        has_managed_run=True,
        label_runtime_ready=True,
    )


def test_cli_adopts_running_trial_then_cancels_fallback(
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    config, store = _queued_store(tmp_path)
    cancelled: list[tuple[str, int]] = []

    submitted_requests: list[Any] = []

    class _Oar:
        def __init__(self, ssh: _Ssh, *, preflight: Any = None) -> None:
            self.site = ssh.target
            self.preflight = preflight

        def status(self, job_id: int) -> JobStatus:
            if job_id == 42:
                return JobStatus(
                    42,
                    JobState.QUEUED,
                    scheduled_start="2099-07-29 19:00:00",
                    walltime_seconds=3300,
                )
            return JobStatus(job_id, JobState.RUNNING, walltime_seconds=3300)

        def submit(self, request: Any) -> int:
            if self.preflight is not None:
                self.preflight()
            submitted_requests.append(request)
            return 101

        def cancel(self, job_id: int) -> None:
            cancelled.append((self.site, job_id))

    monkeypatch.setattr(cli, "SshClient", _Ssh)
    monkeypatch.setattr(cli, "OarClient", _Oar)
    monkeypatch.setattr(cli, "Stager", _Stager)
    monkeypatch.setattr(cli, "_remote_home", lambda _ssh: PurePosixPath("/home/u"))
    monkeypatch.setattr(cli, "_usage_policy_preflight", lambda *_a: None)
    monkeypatch.setattr(cli, "ensure_home_headroom", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        cli,
        "probe_site",
        lambda site, *_a, **_kw: _ready_probe(site),
    )
    args = SimpleNamespace(site=["nancy"], gpu_memory_mb=40_000)
    assert cli._optimize_queued_start(args, store, config, "sophia", 42) == (
        "nancy",
        101,
    )
    durable = store.load()
    assert durable.facts["site"] == "nancy"
    assert durable.facts["job_id"] == 101
    assert durable.facts["replacement_status"] == "adopted"
    assert cancelled == [("sophia", 42)]
    assert len(submitted_requests) == 1
    assert submitted_requests[0].command[1:3] == ("40000", "00:20:00")
    assert submitted_requests[0].command[3] in {"day", "night"}


def test_v2_replacement_preserves_the_durable_production_lane(
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    config = OperatorConfig.build(
        scope="all",
        stage="all",
        source_commit="a" * 40,
        input_revision="b" * 40,
        row_limit=128,
        sampling_target=200_000,
    )
    _, store = _queued_store(tmp_path, config=config, active_stage="label")
    current = store.load()
    store.transition(
        expected=current.phase,
        target=current.phase,
        facts={"label_lane": "production", "smoke_completed": True},
    )
    submitted: list[Any] = []

    class _Oar:
        def __init__(self, ssh: _Ssh, *, preflight: Any = None) -> None:
            self.site = ssh.target
            self.preflight = preflight

        def status(self, job_id: int) -> JobStatus:
            if job_id == 42:
                return JobStatus(
                    42,
                    JobState.QUEUED,
                    scheduled_start="2099-07-29 19:00:00",
                    walltime_seconds=3300,
                )
            return JobStatus(job_id, JobState.RUNNING, walltime_seconds=1200)

        def submit(self, request: Any) -> int:
            submitted.append(request)
            return 101

        def cancel(self, _job_id: int) -> None:
            return None

    monkeypatch.setattr(cli, "SshClient", _Ssh)
    monkeypatch.setattr(cli, "OarClient", _Oar)
    monkeypatch.setattr(cli, "Stager", _Stager)
    monkeypatch.setattr(cli, "_remote_home", lambda _ssh: PurePosixPath("/home/u"))
    monkeypatch.setattr(cli, "_usage_policy_preflight", lambda *_a: None)
    monkeypatch.setattr(cli, "ensure_home_headroom", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "probe_site", lambda site, *_a, **_kw: _ready_probe(site))

    assert cli._optimize_queued_start(
        SimpleNamespace(site=["nancy"], gpu_memory_mb=40_000),
        store,
        config,
        "sophia",
        42,
    ) == ("nancy", 101)

    assert len(submitted) == 1
    tokens = shlex.split(submitted[0].command[4])
    assert tokens[-2] == "production"
    assert tokens[-9] == "0"
    assert "/home/u/osm-polygon-operator/" + config.run_id + "/label-work" in tokens


def test_cli_retains_fallback_when_no_runtime_ready_candidate(
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    config, store = _queued_store(tmp_path)

    class _Oar:
        def __init__(self, _ssh: _Ssh, *, preflight: Any = None) -> None:
            del preflight

        def status(self, job_id: int) -> JobStatus:
            return JobStatus(
                job_id,
                JobState.QUEUED,
                scheduled_start="2099-07-29 19:00:00",
                walltime_seconds=3300,
            )

    monkeypatch.setattr(cli, "SshClient", _Ssh)
    monkeypatch.setattr(cli, "OarClient", _Oar)
    monkeypatch.setattr(cli, "_remote_home", lambda _ssh: PurePosixPath("/home/u"))
    monkeypatch.setattr(
        cli,
        "probe_site",
        lambda site, *_a, **_kw: SiteProbe(
            site,
            site,
            True,
            80_000,
            (8, 0),
            100 * 1024**3,
            0,
            idle_compatible=True,
            label_runtime_ready=False,
        ),
    )
    args = SimpleNamespace(site=["nancy"], gpu_memory_mb=40_000)
    assert cli._optimize_queued_start(args, store, config, "sophia", 42) == (
        "sophia",
        42,
    )


def test_cli_trials_candidate_when_scheduler_has_no_forecast(
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    config, store = _queued_store(tmp_path)
    submitted: list[Any] = []

    class _Oar:
        def __init__(self, ssh: _Ssh, *, preflight: Any = None) -> None:
            self.site = ssh.target
            self.preflight = preflight

        def status(self, job_id: int) -> JobStatus:
            if job_id == 42:
                return JobStatus(42, JobState.QUEUED, scheduled_start=None)
            return JobStatus(job_id, JobState.RUNNING)

        def submit(self, request: Any) -> int:
            submitted.append(request)
            return 101

        def cancel(self, _job_id: int) -> None:
            return None

    monkeypatch.setattr(cli, "SshClient", _Ssh)
    monkeypatch.setattr(cli, "OarClient", _Oar)
    monkeypatch.setattr(cli, "Stager", _Stager)
    monkeypatch.setattr(cli, "_remote_home", lambda _ssh: PurePosixPath("/home/u"))
    monkeypatch.setattr(cli, "_usage_policy_preflight", lambda *_a: None)
    monkeypatch.setattr(cli, "ensure_home_headroom", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "probe_site", lambda site, *_a, **_kw: _ready_probe(site))
    args = SimpleNamespace(site=["nancy"], gpu_memory_mb=40_000)

    assert cli._optimize_queued_start(args, store, config, "sophia", 42) == (
        "nancy",
        101,
    )
    assert len(submitted) == 1


def test_cli_split_replacement_uses_split_submission_without_llama_runtime(
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    config, store = _queued_store(
        tmp_path,
        config=_split_config(),
        active_stage="split",
    )
    submitted: list[Any] = []
    prepared: list[str] = []

    class _Oar:
        def __init__(self, ssh: _Ssh, *, preflight: Any = None) -> None:
            self.site = ssh.target
            self.preflight = preflight

        def status(self, job_id: int) -> JobStatus:
            if job_id == 42:
                return JobStatus(
                    42,
                    JobState.QUEUED,
                    scheduled_start="2099-07-29 19:00:00",
                )
            return JobStatus(job_id, JobState.RUNNING)

        def submit(self, request: Any) -> int:
            if self.preflight is not None:
                self.preflight()
            submitted.append(request)
            return 101

        def cancel(self, _job_id: int) -> None:
            return None

    class _SplitStager:
        def __init__(self, _ssh: _Ssh) -> None:
            pass

        def prepare(self, _config: Any, _layout: Any) -> SimpleNamespace:
            prepared.append("split")
            return SimpleNamespace(reused=True)

        def prepare_label_assets(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("split replacement must not stage label assets")

    monkeypatch.setattr(cli, "SshClient", _Ssh)
    monkeypatch.setattr(cli, "OarClient", _Oar)
    monkeypatch.setattr(cli, "Stager", _SplitStager)
    monkeypatch.setattr(cli, "_remote_home", lambda _ssh: PurePosixPath("/home/u"))
    monkeypatch.setattr(cli, "_usage_policy_preflight", lambda *_a: None)
    monkeypatch.setattr(cli, "ensure_home_headroom", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "split_submission", lambda *_args: "split-request")
    monkeypatch.setattr(
        cli,
        "probe_site",
        lambda site, *_a, **_kw: SiteProbe(
            site,
            site,
            True,
            80_000,
            (8, 0),
            100 * 1024**3,
            0,
            idle_compatible=True,
            label_runtime_ready=False,
        ),
    )
    args = SimpleNamespace(site=["nancy"], gpu_memory_mb=40_000)
    assert cli._optimize_queued_start(args, store, config, "sophia", 42) == (
        "nancy",
        101,
    )
    assert prepared == ["split"]
    assert submitted == ["split-request"]


def test_cli_recovers_persisted_running_trial(
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    config, store = _queued_store(tmp_path)
    current = store.load()
    store.transition(
        expected=current.phase,
        target=current.phase,
        facts={
            "replacement_status": "trial",
            "replacement_site": "nancy",
            "replacement_job_id": 101,
            "replacement_deadline_at": time.time() + 300,
        },
    )
    cancelled: list[tuple[str, int]] = []

    class _Oar:
        def __init__(self, ssh: _Ssh, *, preflight: Any = None) -> None:
            self.site = ssh.target
            del preflight

        def status(self, job_id: int) -> JobStatus:
            if job_id == 101:
                return JobStatus(101, JobState.RUNNING)
            return JobStatus(
                42,
                JobState.QUEUED,
                scheduled_start="2099-07-29 19:00:00",
            )

        def cancel(self, job_id: int) -> None:
            cancelled.append((self.site, job_id))

    monkeypatch.setattr(cli, "SshClient", _Ssh)
    monkeypatch.setattr(cli, "OarClient", _Oar)
    monkeypatch.setattr(cli, "_remote_home", lambda _ssh: PurePosixPath("/home/u"))
    monkeypatch.setattr(
        cli,
        "probe_site",
        lambda site, *_a, **_kw: _ready_probe(site),
    )
    args = SimpleNamespace(site=["nancy"], gpu_memory_mb=40_000)
    assert cli._optimize_queued_start(args, store, config, "sophia", 42) == (
        "nancy",
        101,
    )
    assert cancelled == [("sophia", 42)]


def test_cli_finishes_fallback_cancellation_after_adoption(
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    config, store = _queued_store(tmp_path)
    current = store.load()
    store.transition(
        expected=current.phase,
        target=current.phase,
        facts={
            "site": "nancy",
            "job_id": 101,
            "replacement_status": "adopted",
            "fallback_site": "sophia",
            "fallback_job_id": 42,
            "fallback_cancelled": False,
        },
    )
    cancelled: list[tuple[str, int]] = []

    class _Oar:
        def __init__(self, ssh: _Ssh, *, preflight: Any = None) -> None:
            self.site = ssh.target
            del preflight

        def status(self, job_id: int) -> JobStatus:
            state = JobState.RUNNING if job_id == 101 else JobState.QUEUED
            return JobStatus(job_id, state)

        def cancel(self, job_id: int) -> None:
            cancelled.append((self.site, job_id))

    monkeypatch.setattr(cli, "SshClient", _Ssh)
    monkeypatch.setattr(cli, "OarClient", _Oar)
    monkeypatch.setattr(cli, "_remote_home", lambda _ssh: PurePosixPath("/home/u"))
    args = SimpleNamespace(site=["nancy"], gpu_memory_mb=40_000)
    assert cli._optimize_queued_start(args, store, config, "nancy", 101) == (
        "nancy",
        101,
    )
    assert cancelled == [("sophia", 42)]
    assert store.load().facts["fallback_cancelled"] is True


def test_cli_reoptimizes_adopted_queue_without_start_prediction(
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    config, store = _queued_store(tmp_path)
    current = store.load()
    store.transition(
        expected=current.phase,
        target=current.phase,
        facts={
            "replacement_status": "adopted",
            "fallback_site": "sophia",
            "fallback_job_id": 42,
            "fallback_cancelled": True,
        },
    )
    cancelled: list[tuple[str, int]] = []
    submitted: list[Any] = []

    class _Oar:
        def __init__(self, ssh: _Ssh, *, preflight: Any = None) -> None:
            self.site = ssh.target
            self.preflight = preflight

        def status(self, job_id: int) -> JobStatus:
            if job_id == 42:
                return JobStatus(42, JobState.QUEUED, scheduled_start=None)
            return JobStatus(job_id, JobState.RUNNING)

        def submit(self, request: Any) -> int:
            submitted.append(request)
            return 101

        def cancel(self, job_id: int) -> None:
            cancelled.append((self.site, job_id))

    monkeypatch.setattr(cli, "SshClient", _Ssh)
    monkeypatch.setattr(cli, "OarClient", _Oar)
    monkeypatch.setattr(cli, "Stager", _Stager)
    monkeypatch.setattr(cli, "_remote_home", lambda _ssh: PurePosixPath("/home/u"))
    monkeypatch.setattr(cli, "_usage_policy_preflight", lambda *_a: None)
    monkeypatch.setattr(cli, "ensure_home_headroom", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "probe_site", lambda site, *_a, **_kw: _ready_probe(site))
    args = SimpleNamespace(site=["nancy"], gpu_memory_mb=40_000)

    assert cli._optimize_queued_start(args, store, config, "sophia", 42) == (
        "nancy",
        101,
    )
    assert len(submitted) == 1
    assert cancelled == [("sophia", 42)]
