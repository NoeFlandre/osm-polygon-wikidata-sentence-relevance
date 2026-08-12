"""Focused contracts for site, storage, OAR, token, and workflow planning."""

from __future__ import annotations

from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator.config import OperatorConfig
from osm_polygon_sentence_relevance.operator.label_lanes import (
    LabelLane,
    label_lane_plan,
)
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


def _probe(name: str, *, queued: int = 10, memory: int = 80_000) -> SiteProbe:
    return SiteProbe(name, name, True, memory, (8, 0), 100 * 1024**3, queued)


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


def test_site_selection_ties_break_by_name_when_all_queued() -> None:
    selection = select_site([_probe("rennes"), _probe("nancy"), _probe("nantes")])
    assert selection.selected.name == "nancy"


def test_site_selection_managed_run_is_only_a_runnable_tiebreaker() -> None:
    # Both sites are factually available; the site already holding the managed
    # run wins only as a deterministic tiebreaker (avoids a needless transfer).
    fresh = SiteProbe("grenoble", "grenoble", True, 80_000, (8, 0), 100 * 1024**3, 0)
    resumed = SiteProbe(
        "sophia",
        "sophia",
        True,
        80_000,
        (8, 0),
        100 * 1024**3,
        0,
        has_managed_run=True,
    )
    assert select_site([fresh, resumed]).selected.name == "sophia"


def test_site_selection_factual_idle_capacity_wins_deterministically() -> None:
    # The factual signal is "an idle compatible GPU resource is currently free
    # on this frontend", derived from oarnodes -- not queue depth.
    busy = SiteProbe(
        "sophia",
        "sophia",
        True,
        80_000,
        (8, 0),
        100 * 1024**3,
        0,
        idle_compatible=False,
    )
    idle = SiteProbe(
        "rennes",
        "rennes",
        True,
        80_000,
        (8, 0),
        100 * 1024**3,
        10,
        idle_compatible=True,
    )
    assert select_site([busy, idle]).selected.name == "rennes"


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
        "invalid_queue_count",
    )
    with pytest.raises(NoCompatibleSiteError, match="no Grid"):
        select_site([])


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
    assert request.command[-3] == "a" * 40
    assert request.command[-4] == "afghanistan-latest"
    assert request.command[-5] == "0"


def test_split_trial_serializes_short_scheduler_walltime() -> None:
    request = split_submission(
        _config(stage="split"),
        RemoteLayout(PurePosixPath("/r")),
        walltime_seconds=900,
    )
    assert request.command[-2:] == ("900", "")


def test_split_serializes_validated_resume_bundle() -> None:
    request = split_submission(
        _config(stage="split"),
        RemoteLayout(PurePosixPath("/r")),
        resume_bundle=PurePosixPath("/r/split-resume/" + "c" * 20),
    )

    assert request.command[-2:] == ("1800", "/r/split-resume/" + "c" * 20)


@pytest.mark.parametrize("walltime_seconds", [899, 3_601])
def test_split_rejects_out_of_contract_walltime(walltime_seconds: int) -> None:
    with pytest.raises(ValueError, match="between 15 and 60 minutes"):
        split_submission(
            _config(stage="split"),
            RemoteLayout(PurePosixPath("/r")),
            walltime_seconds=walltime_seconds,
        )


def test_label_serializes_context_and_concurrency() -> None:
    request = label_submission(
        _config(stage="label"),
        RemoteLayout(PurePosixPath("/r")),
        input_parquet=PurePosixPath("/r/input.parquet"),
        model_file=PurePosixPath("/r/model.gguf"),
        tokenizer_dir=PurePosixPath("/r/tokenizer"),
    )
    assert request.command[-7:] == (
        "8",
        "8192",
        "8",
        "0",
        "sentence-relevance-v2",
        "3",
        "a" * 40,
    )


def test_label_micro_allocation_uses_existing_helper_and_wrapper_contract() -> None:
    request = label_submission(
        _config(stage="label"),
        RemoteLayout(PurePosixPath("/r")),
        input_parquet=PurePosixPath("/r/input.parquet"),
        model_file=PurePosixPath("/r/model.gguf"),
        tokenizer_dir=PurePosixPath("/r/tokenizer"),
        walltime_seconds=1200,
        policy_type="day",
    )

    assert request.command[:4] == (
        "/r/repo/scripts/grid5000/_submit_gpu_job.sh",
        "40000",
        "00:20:00",
        "day",
    )
    payload = request.command[4]
    assert payload.startswith(
        "exec env LABEL_DEADLINE_DURATION=600s LABEL_DEADLINE_GRACE=300s "
    )
    assert "/r/repo/scripts/grid5000/run_afghanistan_labeling_job.sh" in payload
    assert payload.count("run_afghanistan_labeling_job.sh") == 1


def test_worldwide_label_uses_dedicated_v2_launcher() -> None:
    config = OperatorConfig.build(
        scope="all",
        stage="label",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )
    request = label_submission(
        config,
        RemoteLayout(PurePosixPath("/r")),
        input_parquet=PurePosixPath("/r/input.parquet"),
        model_file=PurePosixPath("/r/model.gguf"),
        tokenizer_dir=PurePosixPath("/r/tokenizer"),
    )
    assert request.command[0].endswith("submit_worldwide_labeling.sh")


def test_worldwide_label_submission_isolates_smoke_and_production_contracts() -> None:
    config = OperatorConfig.build(
        scope="all",
        stage="all",
        source_commit="a" * 40,
        input_revision="b" * 40,
        row_limit=128,
        sampling_target=200_000,
    )
    layout = RemoteLayout(PurePosixPath("/r"))
    smoke = label_lane_plan(config, layout.root, {})
    production = label_lane_plan(
        config,
        layout.root,
        {"label_lane": LabelLane.PRODUCTION.value},
    )

    smoke_command = label_submission(
        config,
        layout,
        input_parquet=PurePosixPath("/r/input.parquet"),
        model_file=PurePosixPath("/r/model.gguf"),
        tokenizer_dir=PurePosixPath("/r/tokenizer"),
        label_plan=smoke,
    ).command
    production_command = label_submission(
        config,
        layout,
        input_parquet=PurePosixPath("/r/input.parquet"),
        model_file=PurePosixPath("/r/model.gguf"),
        tokenizer_dir=PurePosixPath("/r/tokenizer"),
        label_plan=production,
    ).command

    assert smoke_command[5:7] == (
        "/r/label-smoke-work",
        "/r/label-smoke-output",
    )
    assert smoke_command[14] == "128"
    assert smoke_command[-2] == "smoke"
    assert production_command[5:7] == ("/r/label-work", "/r/label-output")
    assert production_command[14] == "0"
    assert production_command[-2] == "production"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"walltime_seconds": 899, "policy_type": "day"}, "between 15 and 60"),
        ({"walltime_seconds": 1200, "policy_type": "besteffort"}, "day or night"),
        (
            {"walltime_seconds": 1200, "policy_type": "day", "gpu_memory_mb": 0},
            "GPU memory must be positive",
        ),
    ],
)
def test_label_micro_allocation_rejects_unsafe_scheduler_parameters(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        label_submission(
            _config(stage="label"),
            RemoteLayout(PurePosixPath("/r")),
            input_parquet=PurePosixPath("/r/input.parquet"),
            model_file=PurePosixPath("/r/model.gguf"),
            tokenizer_dir=PurePosixPath("/r/tokenizer"),
            **overrides,
        )


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
        '"scheduled_start":"2026-07-28 19:00:00","walltime":3300}}'
    )
    client = OarClient(_FakeSsh([payload]))  # type: ignore[arg-type]
    status = client.status(42)
    assert status.state is expected
    assert status.exit_code == 0
    assert status.node == "n1"
    assert status.scheduled_start == "2026-07-28 19:00:00"
    assert status.walltime_seconds == 3300


def test_oar_client_formats_epoch_schedule_in_grid5000_local_time() -> None:
    payload = '{"42":{"state":"Waiting","scheduled_start":1785344400,"walltime":3300}}'
    client = OarClient(_FakeSsh([payload]))  # type: ignore[arg-type]

    status = client.status(42)

    assert status.scheduled_start == "2026-07-29 19:00:00"
    assert status.walltime_seconds == 3300


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


def test_oar_client_status_maps_returncode_six_to_missing() -> None:
    from osm_polygon_sentence_relevance.operator.ssh import SshRemoteError

    class FailingSsh:
        def run(self, command: str) -> object:
            raise SshRemoteError(
                "unknown job",
                category="remote",
                returncode=6,
                attempts=1,
            )

    status = OarClient(FailingSsh()).status(42)  # type: ignore[arg-type]
    assert status.state is JobState.MISSING


def test_oar_client_status_reraises_non_six_ssh_errors() -> None:
    from osm_polygon_sentence_relevance.operator.ssh import SshRemoteError

    class FailingSsh:
        def run(self, command: str) -> object:
            raise SshRemoteError(
                "network down",
                category="remote",
                returncode=255,
                attempts=1,
            )

    with pytest.raises(SshRemoteError):
        OarClient(FailingSsh()).status(42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1785344400, "2026-07-29 19:00:00"),
        ("2026-07-29 19:00:00", "2026-07-29 19:00:00"),
        (0, None),
        (-1, None),
        (True, None),
        ("not a timestamp", None),
        (None, None),
    ],
)
def test_parse_scheduled_start_handles_all_supported_shapes(
    raw: object, expected: str | None
) -> None:
    from osm_polygon_sentence_relevance.operator.oar import _parse_scheduled_start

    assert _parse_scheduled_start(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (3300, 3300),
        ("3300", 3300),
        ("01:00:00", 3600),
        ("10:05:30", 36_330),
        (0, None),
        (-5, None),
        (True, None),
        ("not time", None),
        (None, None),
    ],
)
def test_parse_walltime_handles_all_supported_shapes(
    raw: object, expected: int | None
) -> None:
    from osm_polygon_sentence_relevance.operator.oar import _parse_walltime

    assert _parse_walltime(raw) == expected


def test_format_walltime_rejects_negative_seconds() -> None:
    from osm_polygon_sentence_relevance.operator.oar import format_walltime

    assert format_walltime(1) == "00:00:01"
    assert format_walltime(86_400) == "24:00:00"
    with pytest.raises(ValueError, match="strictly positive"):
        format_walltime(0)
    with pytest.raises(ValueError, match="strictly positive"):
        format_walltime(-1)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (JobState.QUEUED, True),
        (JobState.RUNNING, True),
        (JobState.FINISHING, True),
        (JobState.TERMINATED, False),
        (JobState.ERROR, False),
        (JobState.MISSING, False),
    ],
)
def test_is_live_state_matches_only_non_terminal_states(
    state: JobState, expected: bool
) -> None:
    from osm_polygon_sentence_relevance.operator.oar import is_live_state

    assert is_live_state(state) is expected


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
    assert layout.split_work == PurePosixPath("/r/work")
    assert layout.split_resume == PurePosixPath("/r/split-resume")
    assert layout.label_work == PurePosixPath("/r/label-work")
    assert layout.label_output == PurePosixPath("/r/label-output")
    assert layout.finalization_state == PurePosixPath("/r/finalization-state")
    final = split_finalization_submission(_config(stage="split"), layout)
    assert "submit_streaming_finalization.sh" in final.command[0]
    assert final.command[-3:] == ("01:00:00", "cpu", "/r/finalization-state")
    build = llama_build_submission(layout)
    assert build.command[0].endswith("_submit_gpu_job.sh")
    assert build.command[1:4] == ("40000", "01:00:00", "night")
    assert "GGML_CUDA=ON" in build.command[-1]
    assert "555881ebc8b0fc0402b30e09258a32a7bfd13c52" in build.command[-1]
    assert "fetch --no-tags origin 555881e;" not in build.command[-1]
    assert 'job_log=/r/logs/"${OAR_JOB_ID:?}"' in build.command[-1]
    assert '"$job_log/build.stdout.log"' in build.command[-1]


def test_worldwide_finalization_receives_sampling_contract() -> None:
    config = OperatorConfig.build(
        scope="all",
        region=None,
        stage="all",
        source_commit="a" * 40,
        input_revision="b" * 40,
        llama_parallel=8,
        llama_per_slot_context=8192,
        sampling_target=200_000,
        sampling_seed="worldwide-seed",
    )

    command = split_finalization_submission(
        config, RemoteLayout(PurePosixPath("/r"))
    ).command

    assert command[-6:] == (
        "all",
        "200000",
        "worldwide-seed",
        "01:00:00",
        "cpu",
        "/r/finalization-state",
    )


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


def test_resumable_walltime_interruption_continues_from_valid_checkpoint() -> None:
    # Only a resumable walltime interruption (valid checkpoint, interrupted) may
    # trigger the next allocation automatically.
    status = JobStatus(7, JobState.TERMINATED, 0, "EXPECTED_WALLTIME")
    facts = CheckpointFacts(13_952, 54_462, True, interrupted=True)
    assert classify_exit(status, facts) is ExitClass.CONTINUE


@pytest.mark.parametrize(
    "message",
    ["request exceeds context size", "schema mismatch", "config error"],
)
def test_deterministic_failure_never_auto_resubmits(message: str) -> None:
    status = JobStatus(8, JobState.ERROR, 512, message)
    facts = CheckpointFacts(13_952, 54_462, True)
    assert classify_exit(status, facts) is ExitClass.FAILED
    assert classify_exit(status, facts) is not ExitClass.CONTINUE
