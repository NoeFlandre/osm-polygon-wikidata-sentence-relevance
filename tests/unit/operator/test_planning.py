"""Focused contracts for site, storage, OAR, token, and workflow planning."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator.config import OperatorConfig
from osm_polygon_sentence_relevance.operator.oar import (
    CheckpointFacts,
    ExitClass,
    JobState,
    JobStatus,
    OarClient,
    OarError,
    SubmissionRequest,
    classify_exit,
    parse_job_id,
)
from osm_polygon_sentence_relevance.operator.sites import (
    NoCompatibleSiteError,
    SiteProbe,
    SiteRequirements,
    evaluate_site,
    select_site,
)
from osm_polygon_sentence_relevance.operator.storage import (
    ManagedEntry,
    ManagedStatus,
    StorageSafetyError,
    execute_cleanup,
    plan_cleanup,
)
from osm_polygon_sentence_relevance.operator.token_budget import (
    TokenBudgetError,
    plan_runtime,
)
from osm_polygon_sentence_relevance.operator.workflows import (
    RemoteLayout,
    label_submission,
    llama_build_submission,
    split_finalization_submission,
    split_submission,
)


def _probe(name: str, *, delay: int = 10, memory: int = 80_000) -> SiteProbe:
    return SiteProbe(name, name, True, memory, (8, 0), 100 * 1024**3, delay)


def _config(*, stage: str = "all") -> OperatorConfig:
    return OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage=stage,
        source_commit="a" * 40,
        input_revision="b" * 40,
        llama_parallel=8,
        llama_per_slot_context=8192,
    )


def test_site_selection_uses_delay_then_name() -> None:
    selection = select_site([_probe("rennes"), _probe("nancy"), _probe("nantes")])
    assert selection.selected.name == "nancy"


def test_site_selection_prefers_compatible_existing_managed_run() -> None:
    fresh = _probe("grenoble", delay=0)
    resumed = SiteProbe(
        "sophia",
        "sophia",
        True,
        80_000,
        (8, 0),
        100 * 1024**3,
        100,
        has_managed_run=True,
    )
    assert select_site([fresh, resumed]).selected.name == "sophia"


def test_managed_run_requires_only_incremental_resume_headroom() -> None:
    resumed = SiteProbe(
        "sophia",
        "sophia",
        True,
        80_000,
        (8, 0),
        6 * 1024**3,
        100,
        has_managed_run=True,
    )
    requirements = SiteRequirements(persistent_free_bytes=22 * 1024**3)
    assert evaluate_site(resumed, requirements).compatible
    too_full = SiteProbe(
        "nancy",
        "nancy",
        True,
        80_000,
        (8, 0),
        511 * 1024**2,
        0,
        has_managed_run=True,
    )
    assert evaluate_site(too_full, requirements).reasons == (
        "insufficient_persistent_storage",
    )


def test_site_selection_rejects_incompatible() -> None:
    with pytest.raises(NoCompatibleSiteError):
        select_site([_probe("nancy", memory=10_000)], SiteRequirements())


def test_site_decision_reports_each_hard_constraint() -> None:
    probe = SiteProbe("x", "x", False, 1, None, 1, -1)
    decision = evaluate_site(probe, SiteRequirements())
    assert decision.reasons == (
        "unreachable",
        "insufficient_gpu_memory",
        "insufficient_cuda_capability",
        "insufficient_persistent_storage",
        "invalid_queue_estimate",
    )
    with pytest.raises(NoCompatibleSiteError, match="no Grid"):
        select_site([])


def test_cleanup_only_removes_inventory_owned_terminal_entries(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    root.mkdir()
    old = root / "old"
    old.mkdir()
    active = root / "active"
    active.mkdir()
    plan = plan_cleanup(
        root,
        [
            ManagedEntry(old, ManagedStatus.COMPLETE, 10, 1),
            ManagedEntry(active, ManagedStatus.ACTIVE, 100, 0),
        ],
        5,
    )
    assert [entry.path for entry in plan.candidates] == [old]
    assert execute_cleanup(plan) == 10
    assert not old.exists()
    assert active.exists()


def test_cleanup_filters_unsafe_and_unneeded_entries(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    root.mkdir()
    safe = root / "safe"
    safe.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    missing = root / "missing"
    link = root / "link"
    link.symlink_to(safe)
    protected = root / ".ssh"
    protected.mkdir()
    entries = [
        ManagedEntry(safe, ManagedStatus.COMPLETE, -10, 3, pipeline_owned=False),
        ManagedEntry(safe, ManagedStatus.ACTIVE, 10, 3),
        ManagedEntry(foreign, ManagedStatus.FAILED, 10, 3),
        ManagedEntry(missing, ManagedStatus.FAILED, 10, 3),
        ManagedEntry(link, ManagedStatus.FAILED, 10, 3),
        ManagedEntry(protected, ManagedStatus.FAILED, 10, 3),
    ]
    assert plan_cleanup(root, entries, 0).candidates == ()
    assert plan_cleanup(root, entries, 100).candidates == ()
    with pytest.raises(ValueError, match="non-negative"):
        plan_cleanup(root, entries, -1)
    with pytest.raises(StorageSafetyError, match="real directory"):
        plan_cleanup(link, entries, 1)


def test_cleanup_revalidates_symlink_and_containment(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    root.mkdir()
    candidate = root / "candidate"
    candidate.mkdir()
    plan = plan_cleanup(
        root, [ManagedEntry(candidate, ManagedStatus.COMPLETE, 10, 1)], 10
    )
    candidate.rmdir()
    candidate.symlink_to(tmp_path)
    with pytest.raises(StorageSafetyError, match="symlink"):
        execute_cleanup(plan)


def test_oar_parsing_and_expected_walltime_continuation() -> None:
    assert parse_job_id("OAR_JOB_ID=6807004\n") == 6_807_004
    status = JobStatus(6_807_004, JobState.TERMINATED, 512, "EXPECTED_WALLTIME")
    facts = CheckpointFacts(13_952, 54_462, True, interrupted=True)
    assert classify_exit(status, facts) is ExitClass.CONTINUE


def test_context_overflow_is_not_continued() -> None:
    status = JobStatus(6_808_797, JobState.ERROR, 512, "request exceeds context size")
    facts = CheckpointFacts(13_952, 54_462, True)
    assert classify_exit(status, facts) is ExitClass.FAILED


def test_token_plan_never_puts_7265_tokens_in_4096_slot() -> None:
    plan = plan_runtime(
        max_prompt_tokens=7265,
        response_tokens=512,
        gpu_memory_mb=40_000,
        max_total_context=65_536,
    )
    assert plan.per_slot_context >= 8192
    assert plan.parallel * plan.per_slot_context == plan.total_context


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_prompt_tokens": 0},
        {"response_tokens": True},
        {"gpu_memory_mb": 0},
        {"max_total_context": -1},
        {"model_memory_mb": 0},
        {"context_memory_bytes_per_token": 0},
    ],
)
def test_token_plan_rejects_non_positive_inputs(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "max_prompt_tokens": 100,
        "response_tokens": 100,
        "gpu_memory_mb": 40_000,
        "max_total_context": 65_536,
    }
    values.update(kwargs)
    with pytest.raises(TokenBudgetError, match="positive integer"):
        plan_runtime(**values)  # type: ignore[arg-type]


def test_token_plan_rejects_oversize_and_insufficient_gpu() -> None:
    with pytest.raises(TokenBudgetError, match="supported slot"):
        plan_runtime(
            max_prompt_tokens=40_000,
            response_tokens=1,
            gpu_memory_mb=40_000,
            max_total_context=65_536,
        )
    with pytest.raises(TokenBudgetError, match="model does not fit"):
        plan_runtime(
            max_prompt_tokens=100,
            response_tokens=100,
            gpu_memory_mb=10_000,
            model_memory_mb=20_000,
            max_total_context=65_536,
        )
    with pytest.raises(TokenBudgetError, match="no supported"):
        plan_runtime(
            max_prompt_tokens=7000,
            response_tokens=500,
            gpu_memory_mb=40_000,
            max_total_context=4096,
        )


def test_region_split_serializes_exact_shard() -> None:
    request = split_submission(
        _config(stage="split"), RemoteLayout(PurePosixPath("/r"))
    )
    assert request.command[-1] == "afghanistan-latest"
    assert request.command[-2] == "0"


def test_label_serializes_context_and_concurrency() -> None:
    request = label_submission(
        _config(stage="label"),
        RemoteLayout(PurePosixPath("/r")),
        input_parquet=PurePosixPath("/r/input.parquet"),
        model_file=PurePosixPath("/r/model.gguf"),
        tokenizer_dir=PurePosixPath("/r/tokenizer"),
    )
    assert request.command[-2:] == ("8192", "8")


@pytest.mark.parametrize("output", ["", "OAR_JOB_ID=1\nOAR_JOB_ID=2", "0"])
def test_job_id_parser_rejects_ambiguous_or_invalid_output(output: str) -> None:
    with pytest.raises(OarError):
        parse_job_id(output)


def test_submission_requires_argv_and_quotes_values() -> None:
    with pytest.raises(ValueError, match="empty"):
        SubmissionRequest(()).shell_command()
    assert SubmissionRequest(("oarsub", "value with space")).shell_command() == (
        "oarsub 'value with space'"
    )


class _FakeSsh:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = outputs
        self.commands: list[str] = []

    def run(self, command: str) -> SimpleNamespace:
        self.commands.append(command)
        value = self.outputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        return SimpleNamespace(stdout=value)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Waiting", JobState.QUEUED),
        ("Hold", JobState.QUEUED),
        ("Launching", JobState.QUEUED),
        ("Running", JobState.RUNNING),
        ("Finishing", JobState.FINISHING),
        ("Terminated", JobState.TERMINATED),
        ("Error", JobState.ERROR),
    ],
)
def test_oar_client_parses_all_supported_states(raw: str, expected: JobState) -> None:
    payload = (
        '{"42":{"state":"'
        + raw
        + '","exit_code":"0","message":"ok","assigned_network_address":"n1",'
        '"scheduled_start":"2026-07-28 19:00:00"}}'
    )
    client = OarClient(_FakeSsh([payload]))  # type: ignore[arg-type]
    status = client.status(42)
    assert status.state is expected
    assert status.exit_code == 0
    assert status.node == "n1"
    assert status.scheduled_start == "2026-07-28 19:00:00"


def test_oar_client_submit_cancel_and_status_errors() -> None:
    ssh = _FakeSsh(["OAR_JOB_ID=42\n", ""])
    client = OarClient(ssh)  # type: ignore[arg-type]
    assert client.submit(SubmissionRequest(("oarsub", "payload"))) == 42
    client.cancel(42)
    assert ssh.commands == ["oarsub payload", "oardel 42"]
    with pytest.raises(ValueError, match="positive"):
        client.status(0)
    with pytest.raises(ValueError, match="positive"):
        client.cancel(0)
    with pytest.raises(OarError, match="invalid OAR"):
        OarClient(_FakeSsh(["not-json"])).status(42)  # type: ignore[arg-type]
    with pytest.raises(OarError, match="unsupported"):
        OarClient(_FakeSsh(['{"state":"Surprise"}'])).status(42)  # type: ignore[arg-type]


def test_oar_client_runs_policy_preflight_before_every_submission() -> None:
    ssh = _FakeSsh(["OAR_JOB_ID=41\n", "OAR_JOB_ID=42\n"])
    calls: list[str] = []
    client = OarClient(ssh, preflight=lambda: calls.append("checked"))  # type: ignore[arg-type]
    request = SubmissionRequest(("oarsub", "payload"))

    assert client.submit(request) == 41
    assert client.submit(request) == 42
    assert calls == ["checked", "checked"]


def test_oar_client_does_not_submit_when_policy_preflight_fails() -> None:
    ssh = _FakeSsh(["OAR_JOB_ID=42\n"])

    def reject() -> None:
        raise RuntimeError("policy refused")

    client = OarClient(ssh, preflight=reject)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="policy refused"):
        client.submit(SubmissionRequest(("oarsub", "payload")))
    assert ssh.commands == []


def test_exit_classification_complete_cancelled_and_failed() -> None:
    complete = CheckpointFacts(10, 10, True)
    assert (
        classify_exit(JobStatus(1, JobState.TERMINATED), complete) is ExitClass.COMPLETE
    )
    incomplete = CheckpointFacts(0, 10, False)
    assert (
        classify_exit(
            JobStatus(1, JobState.TERMINATED, message="job deleted"), incomplete
        )
        is ExitClass.CANCELLED
    )
    assert classify_exit(JobStatus(1, JobState.ERROR), incomplete) is ExitClass.FAILED


def test_workflow_layout_and_finalization_commands() -> None:
    layout = RemoteLayout(PurePosixPath("/r"))
    assert layout.repo == PurePosixPath("/r/repo")
    assert layout.hf_home == PurePosixPath("/r/hf_home")
    assert layout.logs == PurePosixPath("/r/logs")
    assert layout.split_work == PurePosixPath("/r/split-work")
    assert layout.label_work == PurePosixPath("/r/label-work")
    assert layout.label_output == PurePosixPath("/r/label-output")
    final = split_finalization_submission(_config(stage="split"), layout)
    assert "submit_streaming_finalization.sh" in final.command[0]
    assert final.command[-2:] == ("02:00:00", "cpu")
    build = llama_build_submission(layout)
    assert build.command[0].endswith("_submit_gpu_job.sh")
    assert build.command[1:4] == ("40000", "01:00:00", "night")
    assert "GGML_CUDA=ON" in build.command[-1]
    assert 'job_log=/r/logs/"${OAR_JOB_ID:?}"' in build.command[-1]
    assert '"$job_log/build.stdout.log"' in build.command[-1]


def test_workflows_require_immutable_revision() -> None:
    config = _config(stage="split")
    object.__setattr__(config, "input_dataset_revision", None)
    layout = RemoteLayout(PurePosixPath("/r"))
    with pytest.raises(ValueError, match="revision"):
        split_submission(config, layout)
    with pytest.raises(ValueError, match="revision"):
        split_finalization_submission(config, layout)
    with pytest.raises(ValueError, match="revision"):
        label_submission(
            config,
            layout,
            input_parquet=PurePosixPath("/i"),
            model_file=PurePosixPath("/m"),
            tokenizer_dir=PurePosixPath("/t"),
        )
