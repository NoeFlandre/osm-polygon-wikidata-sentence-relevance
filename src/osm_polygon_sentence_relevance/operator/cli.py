"""Public Mac-side command for autonomous Grid'5000 dataset production."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Annotated, Any, Final

import click
import typer

from osm_polygon_sentence_relevance.labeling.v2_contracts import (
    V2_LOGIT_PROMPT_VERSION,
)
from osm_polygon_sentence_relevance.operator import (
    classification as classification_transitions,
)
from osm_polygon_sentence_relevance.operator import (
    continuation,
    recorded_job,
    relay,
    remote_preparation,
    split_finalization,
    split_relay,
)
from osm_polygon_sentence_relevance.operator import preflight as _preflight
from osm_polygon_sentence_relevance.operator import resume as _resume
from osm_polygon_sentence_relevance.operator.config import (
    DATA_ROOT,
    DEFAULT_SAMPLING_H3_RESOLUTION,
    DEFAULT_SAMPLING_SEED,
    OperatorConfig,
    Scope,
    Stage,
)
from osm_polygon_sentence_relevance.operator.console import OperatorConsole
from osm_polygon_sentence_relevance.operator.controller import (
    Controller,
    LiveProgress,
)
from osm_polygon_sentence_relevance.operator.earliest_start import (
    IMMEDIATE_TRIAL_WALLTIME_SECONDS,
    QUEUED_RESCAN_SECONDS,
    UNPREDICTED_TRIAL_SECONDS,
    ReplacementCandidate,
    ReplacementOutcome,
    attempt_immediate_replacement,
    policy_type_for,
    race_queued_replacements,
    rank_replacement_candidates,
    should_seek_replacement,
)
from osm_polygon_sentence_relevance.operator.label_lanes import (
    LabelLanePlan,
    label_lane_plan,
)
from osm_polygon_sentence_relevance.operator.llama_server import ensure_llama_server
from osm_polygon_sentence_relevance.operator.oar import (
    GRID5000_TZ,
    ExitClass,
    JobState,
    JobStatus,
    OarClient,
    is_live_state,
)
from osm_polygon_sentence_relevance.operator.preflight import (
    git_head as _git_head,
)
from osm_polygon_sentence_relevance.operator.preflight import (
    remote_home as _remote_home,
)
from osm_polygon_sentence_relevance.operator.preflight import (
    resolve_input_revision as _resolve_input_revision,
)
from osm_polygon_sentence_relevance.operator.preflight import (
    usage_policy_preflight as _usage_policy_preflight,
)
from osm_polygon_sentence_relevance.operator.recovery import (
    next_recovery_attempt as _next_recovery_attempt,
)
from osm_polygon_sentence_relevance.operator.recovery import (
    reattach_decision as _reattach_decision,
)
from osm_polygon_sentence_relevance.operator.recovery import (
    transition_terminal as _transition_terminal,
)
from osm_polygon_sentence_relevance.operator.remote_completion import (
    assert_remote_exit_zero,
    label_publication_commit,
    mark_remote_status,
    preserve_label,
    preserve_manual_eval,
    publish_label,
    publish_split,
    remote_exit_code,
)
from osm_polygon_sentence_relevance.operator.result_text import result_text
from osm_polygon_sentence_relevance.operator.sampling import (
    sampling_target_for_run as _sampling_target_for_run,
)
from osm_polygon_sentence_relevance.operator.sampling import (
    sync_sampling_target as _sync_sampling_target,
)
from osm_polygon_sentence_relevance.operator.site_discovery import (
    DEFAULT_SITES,
    probe_site,
)
from osm_polygon_sentence_relevance.operator.sites import (
    NoCompatibleSiteError,
    SiteProbe,
    SiteRequirements,
    select_site,
)
from osm_polygon_sentence_relevance.operator.split_resume import (
    classify_split_terminal,
    inspect_split_resume,
    split_failure_reason,
)
from osm_polygon_sentence_relevance.operator.ssh import SshClient
from osm_polygon_sentence_relevance.operator.staging import Stager
from osm_polygon_sentence_relevance.operator.state import RunPhase, StateStore
from osm_polygon_sentence_relevance.operator.storage import (
    LABEL_STAGING_HEADROOM_BYTES,
    cleanup_can_restore_compatibility,
    cleanup_managed_runs,
    ensure_home_headroom,
    required_staging_headroom,
)
from osm_polygon_sentence_relevance.operator.supervisor import (
    SupervisorLaunch,
    start_detached_supervisor,
)
from osm_polygon_sentence_relevance.operator.workflows import (
    PREFERRED_LABEL_WALLTIME_SECONDS,
    RemoteLayout,
    label_submission,
    split_submission,
)

_SUBMISSION_HEADROOM_BYTES: Final[int] = 512 * 1024**2
_CONSOLE = OperatorConsole()

app = typer.Typer(
    name="osm-polygon-grid5000",
    help="Run and resume sentence processing on Grid'5000.",
    add_completion=False,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)


class _LocalMonitoringInterrupted(RuntimeError):
    """Typer adapter sentinel preserving the public exit-130 contract."""


def _dispatch(handler: Any, args: SimpleNamespace) -> int:
    try:
        return int(handler(args))
    except KeyboardInterrupt as exc:
        raise _LocalMonitoringInterrupted from exc


def _milestone(message: str) -> None:
    """Print one concise operator milestone immediately."""

    _CONSOLE.milestone(message)


def _stage_hf_token(stager: object, layout: RemoteLayout) -> None:
    """Stage the private Hub credential when the real stager is in use."""

    method = getattr(stager, "stage_hf_token", None)
    if callable(method):
        method(layout)


#: Per-invocation active run ID. Set right after a validated
#: ``store.load_or_create`` (or its persisted equivalent) and cleared in
#: the enclosing ``finally``. Never scanned across unrelated runs.
_ACTIVE_RUN_ID: str | None = None


# Keep the subprocess module visible through the CLI for existing test seams;
# both modules intentionally reference the same module object.
subprocess = _preflight.subprocess
# Preserve the historical module-level constants for callers and tests.
INPUT_DATASET_ID = _preflight.INPUT_DATASET_ID
OUTPUT_DATASET_ID = _preflight.OUTPUT_DATASET_ID


def _emit(progress: LiveProgress) -> None:
    _CONSOLE.job_lines(progress.job_id, progress.text)


def _resume_command(run_id: str) -> str:
    """The exact command the operator prints after a local interrupt."""

    return f"uv run osm-polygon-grid5000 resume {run_id}"


def _announce_detached(launch: SupervisorLaunch) -> None:
    """Report the durable handle returned by detached mode."""

    _milestone(
        f"Detached supervisor started ({launch.backend}); session="
        f"{launch.session_name}; log={launch.log_path}"
    )


def _detached_run_arguments(args: SimpleNamespace) -> tuple[str, ...]:
    """Build a complete foreground ``run`` command for the child supervisor."""

    values: list[str] = [
        "run",
        "--scope",
        str(args.scope),
        "--stage",
        str(args.stage),
        "--batch-size",
        str(args.batch_size),
        "--row-limit",
        str(args.row_limit),
        "--sampling-seed",
        str(args.sampling_seed),
        "--sampling-h3-resolution",
        str(args.sampling_h3_resolution),
        "--llama-parallel",
        str(args.llama_parallel),
        "--llama-per-slot-context",
        str(args.llama_per_slot_context),
        "--gpu-memory-mb",
        str(args.gpu_memory_mb),
        "--remote-free-bytes",
        str(args.remote_free_bytes),
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    if args.region is not None:
        values.extend(("--region", str(args.region)))
    if args.input_revision is not None:
        values.extend(("--input-revision", str(args.input_revision)))
    if args.sampling_target is not None:
        values.extend(("--sampling-target", str(args.sampling_target)))
    if args.request_concurrency is not None:
        values.extend(("--request-concurrency", str(args.request_concurrency)))
    for target in args.site:
        values.extend(("--site", str(target)))
    return tuple(values)


def _detached_resume_arguments(
    run_id: str,
    *,
    site: list[str],
    gpu_memory_mb: int,
    poll_seconds: float,
    sampling_target: int | None,
) -> tuple[str, ...]:
    """Build a complete foreground ``resume`` command for a child supervisor."""

    values: list[str] = [
        "resume",
        run_id,
        "--gpu-memory-mb",
        str(gpu_memory_mb),
        "--poll-seconds",
        str(poll_seconds),
    ]
    if sampling_target is not None:
        values.extend(("--sampling-target", str(sampling_target)))
    for target in site:
        values.extend(("--site", str(target)))
    return tuple(values)


def _checkpoint_root(
    layout: RemoteLayout, label_plan: LabelLanePlan | None = None
) -> str:
    """The remote root consumed by :class:`CheckpointStore`.

    The real production layout written by :class:`CheckpointStore` is::

        ${label_work}/progress.json
        ${label_work}/timing.json   (optional)
        ${label_work}/checkpoints/batch-NNNNNN.parquet
        ${label_work}/checkpoints/batch-NNNNNN.json

    There is no ``${label_work}/checkpoints/<run_id>`` directory.
    """

    return str(label_plan.work_dir if label_plan is not None else layout.label_work)


def _split_state_identity(config: OperatorConfig) -> dict[str, str | int]:
    """Return the exact identity persisted by the streaming split driver."""

    revision = config.input_dataset_revision
    if revision is None:
        raise RuntimeError("immutable input revision is required for split relay")
    return {
        "repo_id": config.output_dataset_id,
        "resolved_revision": revision,
        "source_commit": config.source_commit,
        "run_id": config.run_id,
        "staging_revision": f"checkpoints/{config.run_id}",
        "pipeline_version": config.pipeline_version,
        "model_name": config.split_model,
        "batch_size": config.requirements.batch_size,
    }


def _recorded_split_resume_bundle(store: StateStore) -> PurePosixPath | None:
    value = store.load().facts.get("split_resume_bundle")
    return PurePosixPath(value) if isinstance(value, str) and value else None


def _clear_consumed_split_resume_bundle(store: StateStore) -> None:
    """Forget a one-shot bundle after its allocation reached a terminal state."""

    if _recorded_split_resume_bundle(store) is None:
        return
    current = store.load()
    store.transition(
        expected=current.phase,
        target=current.phase,
        facts={"split_resume_bundle": ""},
    )


def _stage_split_snapshot(
    *,
    config: OperatorConfig,
    source_site: str,
    destination_site: str,
) -> tuple[Path, str]:
    """Relay one immutable split acceleration snapshot between frontends."""

    source_ssh = SshClient(target=source_site, command_timeout=600)
    source_layout = RemoteLayout(
        _remote_home(source_ssh) / "osm-polygon-operator" / config.run_id
    )
    inventory = split_relay.retrieve_to_seagate(
        source=relay.RemoteTransfer(ssh_target=source_site),
        source_work_root=str(source_layout.split_work),
        destination_root=DATA_ROOT / "runs",
        run_id=config.run_id,
        expected_identity=_split_state_identity(config),
    )
    destination_ssh = SshClient(target=destination_site, command_timeout=600)
    destination_layout = RemoteLayout(
        _remote_home(destination_ssh) / "osm-polygon-operator" / config.run_id
    )
    remote_bundle = split_relay.stage_to_destination(
        inventory=inventory,
        destination=relay.RemoteTransfer(ssh_target=destination_site),
        destination_resume_root=str(destination_layout.split_resume),
        expected_identity=_split_state_identity(config),
    )
    return inventory.root, remote_bundle


def _restage_split_snapshot(
    *,
    config: OperatorConfig,
    local_root: Path,
    destination_site: str,
) -> str:
    """Stage an already validated Seagate snapshot to another trial site."""

    from scripts.streaming.resume_bundle import validate_resume_bundle

    inventory = validate_resume_bundle(local_root, _split_state_identity(config))
    destination_ssh = SshClient(target=destination_site, command_timeout=600)
    destination_layout = RemoteLayout(
        _remote_home(destination_ssh) / "osm-polygon-operator" / config.run_id
    )
    return split_relay.stage_to_destination(
        inventory=inventory,
        destination=relay.RemoteTransfer(ssh_target=destination_site),
        destination_resume_root=str(destination_layout.split_resume),
        expected_identity=_split_state_identity(config),
    )


def _current_label_plan(
    store: StateStore,
    config: OperatorConfig,
    layout: RemoteLayout,
) -> LabelLanePlan | None:
    """Return the durable V2 lane, leaving legacy labeling unchanged."""

    if (
        config.scope is not Scope.ALL
        or config.prompt_version != V2_LOGIT_PROMPT_VERSION
    ):
        return None
    return label_lane_plan(config, layout.root, store.load().facts)


def _attach_to_site(
    store: StateStore,
    config: OperatorConfig,
    site: str,
    *,
    poll_seconds: float,
    preflight: Callable[[], None] | None = None,
) -> tuple[SshClient, RemoteLayout, OarClient, Controller]:
    """Open one SSH connection to a recorded site and build the controller."""

    ssh = SshClient(target=site, command_timeout=1800)
    layout = RemoteLayout(_remote_home(ssh) / "osm-polygon-operator" / config.run_id)
    oar = OarClient(ssh, preflight=preflight)
    stager = Stager(ssh)
    controller = Controller(
        config=config,
        state=store,
        ssh=ssh,
        oar=oar,
        stager=stager,
        layout=layout,
        emit=_emit,
        poll_seconds=poll_seconds,
    )
    return ssh, layout, oar, controller


def _monitor_until_terminal(
    controller: Controller, job_id: int, *, log_name: str
) -> JobState:
    """Stream the live log until OAR reports a terminal state, returning it."""

    return controller.monitor(job_id, log_name=log_name)


def _is_terminal_allocation(state: JobState) -> bool:
    """Return whether OAR ended an allocation and checkpoint classification may run."""

    return state in {JobState.TERMINATED, JobState.ERROR}


def _prepare_destination_for_resume(
    *,
    store: StateStore,
    config: OperatorConfig,
    site: str,
    relay_root: Path | None,
    poll_seconds: float,
) -> None:
    """Adapt CLI seams to the isolated continuation preparation policy."""

    return remote_preparation.prepare_destination(
        store=store,
        config=config,
        site=site,
        relay_root=relay_root,
        poll_seconds=poll_seconds,
        services=_remote_preparation_services(),
    )


def _remote_preparation_services() -> remote_preparation.RemotePreparationServices:
    """Bind CLI-owned side effects to continuation preparation."""

    return remote_preparation.RemotePreparationServices(
        ssh_factory=SshClient,
        remote_home=_remote_home,
        usage_policy_preflight=_usage_policy_preflight,
        ensure_home_headroom=ensure_home_headroom,
        stager_type=Stager,
        stage_hf_token=_stage_hf_token,
        oar_type=OarClient,
        ensure_llama_server=ensure_llama_server,
        label_staging_headroom_bytes=LABEL_STAGING_HEADROOM_BYTES,
        submission_headroom_bytes=_SUBMISSION_HEADROOM_BYTES,
    )


def _ensure_relay_at_destination(
    *,
    store: StateStore,
    config: OperatorConfig,
    site: str,
    layout: RemoteLayout,
    relay_root: Path,
) -> None:
    """Verify the validated relay is present at the destination site.

    The Seagate-side ``relay_root`` is the canonical validated inventory.
    For continuation we trust the staged destination (already done by
    :func:`stage_to_destination`). This helper exists so the orchestrator
    can re-check the relay directory exists before submitting.
    """

    if not relay_root.is_dir():
        raise RuntimeError(
            f"validated relay disappeared before submission: {relay_root}"
        )


def _classification_services() -> classification_transitions.ClassificationServices:
    """Bind CLI-owned side effects to the classification state machine."""

    return classification_transitions.ClassificationServices(
        next_recovery_attempt=_next_recovery_attempt,
        transition_terminal=_transition_terminal,
        preserve_label=preserve_label,
        preserve_manual_eval=preserve_manual_eval,
        label_publication_commit=label_publication_commit,
        publish_label=publish_label,
        mark_remote_status=mark_remote_status,
    )


def _apply_classification(
    *,
    store: StateStore,
    config: OperatorConfig,
    ssh: SshClient,
    layout: RemoteLayout,
    job_id: int,
    active_stage: str,
    classification: ExitClass,
    resume_artifact_path: str | None = None,
    failure_reason_token: str | None = None,
    label_plan: LabelLanePlan | None = None,
) -> None:
    """Adapt CLI calls to the isolated classification state machine."""

    return classification_transitions.apply_classification(
        store=store,
        config=config,
        ssh=ssh,
        layout=layout,
        job_id=job_id,
        active_stage=active_stage,
        classification=classification,
        resume_artifact_path=resume_artifact_path,
        failure_reason_token=failure_reason_token,
        label_plan=label_plan,
        services=_classification_services(),
    )


def _relay_for_continuation(
    *,
    store: StateStore,
    config: OperatorConfig,
    source_site: str,
    destination_site: str,
) -> str:
    """Retrieve, validate, and stage the resume set for a new site."""

    current = store.load()
    is_label = current.facts.get("active_stage") == Stage.LABEL.value
    if not is_label:
        local_root, remote_bundle = _stage_split_snapshot(
            config=config,
            source_site=source_site,
            destination_site=destination_site,
        )
        store.transition(
            expected=current.phase,
            target=current.phase,
            facts={
                "relay_destination_site": destination_site,
                "split_resume_bundle": remote_bundle,
            },
        )
        return str(local_root)
    source_ssh = SshClient(target=source_site, command_timeout=600)
    source_layout = RemoteLayout(
        _remote_home(source_ssh) / "osm-polygon-operator" / config.run_id
    )
    source_label_plan = (
        _current_label_plan(store, config, source_layout) if is_label else None
    )
    source_root = _checkpoint_root(source_layout, source_label_plan)
    expected_identity = (
        source_label_plan.config.run_identity.checkpoint_dict()
        if source_label_plan is not None
        else config.run_identity.checkpoint_dict()
    )
    inventory = relay.retrieve_to_seagate(
        source=relay.RemoteTransfer(ssh_target=source_site),
        source_checkpoint_root=source_root,
        destination_root=DATA_ROOT / "runs",
        run_id=config.run_id,
        expected_run_identity=expected_identity,
    )
    destination_ssh = SshClient(target=destination_site, command_timeout=600)
    destination_layout = RemoteLayout(
        _remote_home(destination_ssh) / "osm-polygon-operator" / config.run_id
    )
    destination_label_plan = (
        _current_label_plan(store, config, destination_layout) if is_label else None
    )
    destination_root = _checkpoint_root(destination_layout, destination_label_plan)
    relay.stage_to_destination(
        inventory=inventory,
        destination=relay.RemoteTransfer(ssh_target=destination_site),
        destination_checkpoint_root=destination_root,
    )
    store.transition(
        expected=current.phase,
        target=current.phase,
        facts={"relay_destination_site": destination_site},
    )
    return str(inventory.root)


def _continuation_services() -> continuation.ContinuationServices:
    """Bind CLI-owned side effects to the continuation state machine."""

    return continuation.ContinuationServices(
        attach_to_site=_attach_to_site,
        current_label_plan=_current_label_plan,
        monitor_until_terminal=_monitor_until_terminal,
        is_terminal_allocation=_is_terminal_allocation,
        inspect_split_resume=inspect_split_resume,
        classify_split_terminal=classify_split_terminal,
        split_failure_reason=split_failure_reason,
        recorded_job=recorded_job,
        relay_for_continuation=_relay_for_continuation,
        apply_classification=_apply_classification,
        split_finalization=split_finalization,
        resume_run=_resume_run,
        race_queued_start=_race_queued_start,
        milestone=_milestone,
        prepare_destination_for_resume=_prepare_destination_for_resume,
        ensure_relay_at_destination=_ensure_relay_at_destination,
        recorded_split_resume_bundle=_recorded_split_resume_bundle,
        clear_consumed_split_resume_bundle=_clear_consumed_split_resume_bundle,
        stage_hf_token=_stage_hf_token,
        stager_type=Stager,
        mark_remote_status=mark_remote_status,
    )


def _classify_or_continue(
    args: SimpleNamespace,
    store: StateStore,
    config: OperatorConfig,
    site: str,
    job_id: int,
    *,
    destination_site: str | None = None,
) -> ExitClass:
    """Delegate allocation recovery to the isolated continuation module."""

    return continuation.classify_or_continue(
        args,
        store,
        config,
        site,
        job_id,
        destination_site=destination_site,
        services=_continuation_services(),
    )


def _optimize_queued_start(
    args: SimpleNamespace,
    store: StateStore,
    config: OperatorConfig,
    fallback_site: str,
    fallback_job_id: int,
) -> tuple[str, int]:
    """Replace a distant queued job only after a trial is actually running."""

    clients: dict[str, tuple[SshClient, RemoteLayout, OarClient]] = {}
    assets: dict[str, Any] = {}
    split_bundles: dict[str, PurePosixPath] = {}

    def client(site: str) -> tuple[SshClient, RemoteLayout, OarClient]:
        cached = clients.get(site)
        if cached is not None:
            return cached
        ssh = SshClient(target=site, command_timeout=1800)
        layout = RemoteLayout(
            _remote_home(ssh) / "osm-polygon-operator" / config.run_id
        )

        def preflight() -> None:
            _usage_policy_preflight(ssh, site)
            ensure_home_headroom(
                ssh,
                protected_root=layout.root,
                minimum_headroom_bytes=_SUBMISSION_HEADROOM_BYTES,
            )

        cached = (ssh, layout, OarClient(ssh, preflight=preflight))
        clients[site] = cached
        return cached

    fallback_status = client(fallback_site)[2].status(fallback_job_id)
    durable = store.load()
    active_stage = durable.facts.get("active_stage", config.stage.value)
    requires_label_runtime = active_stage == Stage.LABEL.value
    replacement_status = durable.facts.get("replacement_status")
    adopted_queue = (
        replacement_status == "adopted" and fallback_status.state is JobState.QUEUED
    )
    if replacement_status == "adopted":
        old_site = durable.facts.get("fallback_site")
        old_job = durable.facts.get("fallback_job_id")
        if (
            isinstance(old_site, str)
            and type(old_job) is int
            and old_job > 0
            and (old_site, old_job) != (fallback_site, fallback_job_id)
            and durable.facts.get("fallback_cancelled") is not True
        ):
            old_status = client(old_site)[2].status(old_job)
            if is_live_state(old_status.state):
                client(old_site)[2].cancel(old_job)
            current = store.load()
            store.transition(
                expected=current.phase,
                target=current.phase,
                facts={"fallback_cancelled": True},
            )
        if not adopted_queue:
            return fallback_site, fallback_job_id
        current = store.load()
        store.transition(
            expected=current.phase,
            target=current.phase,
            facts={
                "fallback_site": fallback_site,
                "fallback_job_id": fallback_job_id,
                "fallback_cancelled": False,
                "replacement_status": "inactive",
            },
        )
        replacement_status = "inactive"

    now = datetime.now(tz=GRID5000_TZ)
    existing_trial: tuple[ReplacementCandidate, int, float] | None = None
    if replacement_status == "trial":
        trial_site = durable.facts.get("replacement_site")
        trial_job = durable.facts.get("replacement_job_id")
        deadline_at = durable.facts.get("replacement_deadline_at")
        if (
            isinstance(trial_site, str)
            and type(trial_job) is int
            and trial_job > 0
            and isinstance(deadline_at, (int, float))
        ):
            trial_probe = probe_site(
                trial_site,
                config.run_id,
                SiteRequirements(
                    gpu_memory_mb=getattr(args, "gpu_memory_mb", 40_000),
                    persistent_free_bytes=_SUBMISSION_HEADROOM_BYTES,
                ),
            )
            remaining = max(0.0, float(deadline_at) - time.time())
            existing_trial = (
                ReplacementCandidate(trial_probe),
                trial_job,
                time.monotonic() + remaining,
            )

    seek_unpredicted = (
        fallback_status.state is JobState.QUEUED
        and fallback_status.scheduled_start is None
    )
    if (
        existing_trial is None
        and not seek_unpredicted
        and not should_seek_replacement(
            fallback_status,
            now=now,
        )
    ):
        return fallback_site, fallback_job_id

    requirements = SiteRequirements(
        gpu_memory_mb=getattr(args, "gpu_memory_mb", 40_000),
        persistent_free_bytes=_SUBMISSION_HEADROOM_BYTES,
    )
    targets = tuple(dict.fromkeys(getattr(args, "site", DEFAULT_SITES)))
    probes: list[SiteProbe] = []
    _milestone(
        "Queued start is distant; checking every site for an immediate "
        "policy-compliant GPU"
    )
    for target in targets:
        probe = probe_site(target, config.run_id, requirements)
        probes.append(probe)
        if not probe.reachable:
            _milestone(f"Immediate candidate {target}: unavailable")
        elif not probe.idle_compatible:
            _milestone(f"Immediate candidate {target}: no idle compatible GPU")
        elif requires_label_runtime and not probe.label_runtime_ready:
            _milestone(f"Immediate candidate {target}: labeling runtime not staged")
        else:
            _milestone(f"Immediate candidate {target}: ready")
    excluded_sites: set[str] = set()
    if existing_trial is not None:
        excluded_sites.add(existing_trial[0].site.name)
    if seek_unpredicted:
        excluded_sites.add(fallback_site)
    candidates = rank_replacement_candidates(
        probes,
        requirements=requirements,
        excluded_sites=frozenset(excluded_sites),
        require_label_runtime=requires_label_runtime,
    )
    if existing_trial is None and not candidates:
        _milestone(
            "No demonstrably immediate replacement exists; retaining scheduled "
            f"job {fallback_job_id}"
        )
        return fallback_site, fallback_job_id

    def prepare(candidate: ReplacementCandidate) -> None:
        site = candidate.site.name
        ssh, layout, _oar = client(site)
        _usage_policy_preflight(ssh, site)
        ensure_home_headroom(
            ssh,
            protected_root=layout.root,
            minimum_headroom_bytes=_SUBMISSION_HEADROOM_BYTES,
        )
        stager = Stager(ssh)
        stager.prepare(config, layout)
        _stage_hf_token(stager, layout)
        if requires_label_runtime:
            label_assets = stager.prepare_label_assets(
                config, layout, download_input=True
            )
            if not label_assets.llama_server_ready:
                raise RuntimeError("CUDA llama-server is not staged")
            assets[site] = label_assets
        elif site != fallback_site:
            source_ssh, source_layout, _source_oar = client(fallback_site)
            has_split_state = result_text(
                source_ssh.run(
                    f"if test -f {source_layout.split_work}/state.json; "
                    "then printf yes; else printf no; fi"
                )
            )
            if has_split_state == "yes":
                _local_root, remote_bundle = _stage_split_snapshot(
                    config=config,
                    source_site=fallback_site,
                    destination_site=site,
                )
                split_bundles[site] = PurePosixPath(remote_bundle)
            else:
                local_root_value = store.load().facts.get("resume_relay_root")
                if isinstance(local_root_value, str) and local_root_value:
                    split_bundles[site] = PurePosixPath(
                        _restage_split_snapshot(
                            config=config,
                            local_root=Path(local_root_value),
                            destination_site=site,
                        )
                    )

    def submit(candidate: ReplacementCandidate) -> int:
        site = candidate.site.name
        current_fallback = client(fallback_site)[2].status(fallback_job_id)
        if current_fallback.state is not JobState.QUEUED:
            raise RuntimeError("fallback is no longer queued")
        _ssh, layout, oar = client(site)
        stager = Stager(_ssh)
        if hasattr(stager, "clean_generated_python_caches"):
            stager.clean_generated_python_caches(layout)
        if not requires_label_runtime:
            return oar.submit(
                split_submission(
                    config,
                    layout,
                    walltime_seconds=IMMEDIATE_TRIAL_WALLTIME_SECONDS,
                    resume_bundle=split_bundles.get(site),
                )
            )
        label_assets = assets[site]
        label_plan = _current_label_plan(store, config, layout)
        return oar.submit(
            label_submission(
                config,
                layout,
                input_parquet=label_assets.input_parquet,
                model_file=label_assets.model_file,
                tokenizer_dir=label_assets.tokenizer_dir,
                walltime_seconds=IMMEDIATE_TRIAL_WALLTIME_SECONDS,
                policy_type=policy_type_for(
                    datetime.now(tz=GRID5000_TZ),
                    walltime_seconds=IMMEDIATE_TRIAL_WALLTIME_SECONDS,
                ),
                gpu_memory_mb=getattr(args, "gpu_memory_mb", 40_000),
                label_plan=label_plan,
            )
        )

    def persist_trial(site: str, job_id: int, deadline: float) -> None:
        remaining = max(0.0, deadline - time.monotonic())
        current = store.load()
        store.transition(
            expected=current.phase,
            target=current.phase,
            facts={
                "fallback_site": fallback_site,
                "fallback_job_id": fallback_job_id,
                "fallback_cancelled": False,
                "replacement_site": site,
                "replacement_job_id": job_id,
                "replacement_deadline_at": time.time() + remaining,
                "replacement_status": "trial",
                **(
                    {"split_resume_bundle": str(split_bundles[site])}
                    if site in split_bundles
                    else {}
                ),
            },
        )

    def adopt_trial(site: str, job_id: int) -> None:
        current = store.load()
        store.transition(
            expected=current.phase,
            target=current.phase,
            facts={
                "site": site,
                "job_id": job_id,
                "replacement_status": "adopted",
                "fallback_cancelled": False,
                **(
                    {"split_resume_bundle": str(split_bundles[site])}
                    if site in split_bundles
                    else {}
                ),
            },
        )

    def clear_trial(job_id: int) -> None:
        current = store.load()
        store.transition(
            expected=current.phase,
            target=current.phase,
            facts={
                "replacement_job_id": job_id,
                "replacement_status": "inactive",
            },
        )

    outcome = attempt_immediate_replacement(
        fallback_site=fallback_site,
        fallback_job_id=fallback_job_id,
        candidates=candidates,
        prepare=prepare,
        submit=submit,
        status=lambda site, job_id: client(site)[2].status(job_id),
        cancel=lambda site, job_id: client(site)[2].cancel(job_id),
        persist_trial=persist_trial,
        adopt_trial=adopt_trial,
        clear_trial=clear_trial,
        emit=_milestone,
        monotonic=time.monotonic,
        sleep=time.sleep,
        wall_clock=lambda: datetime.now(tz=GRID5000_TZ),
        existing_trial=existing_trial,
        trial_seconds=UNPREDICTED_TRIAL_SECONDS,
    )
    return outcome.site, outcome.job_id


def _race_queued_start(
    args: SimpleNamespace,
    store: StateStore,
    config: OperatorConfig,
    fallback_site: str,
    fallback_job_id: int,
) -> tuple[str, int]:
    """Re-probe every site while a distant fallback remains queued."""

    status_clients: dict[str, OarClient] = {}

    def status(site: str, job_id: int) -> JobStatus:
        client = status_clients.get(site)
        if client is None:
            client = OarClient(SshClient(target=site, command_timeout=1800))
            status_clients[site] = client
        return client.status(job_id)

    def attempt(site: str, job_id: int) -> ReplacementOutcome:
        optimized_site, optimized_job_id = _optimize_queued_start(
            args,
            store,
            config,
            site,
            job_id,
        )
        return ReplacementOutcome(
            optimized_site,
            optimized_job_id,
            replaced=(optimized_site, optimized_job_id) != (site, job_id),
        )

    outcome = race_queued_replacements(
        fallback_site=fallback_site,
        fallback_job_id=fallback_job_id,
        attempt=attempt,
        status=status,
        emit=_milestone,
        sleep=time.sleep,
        wall_clock=lambda: datetime.now(tz=GRID5000_TZ),
        rescan_seconds=QUEUED_RESCAN_SECONDS,
    )
    return outcome.site, outcome.job_id


def _resume_services() -> _resume.ResumeServices:
    """Bind CLI-owned side effects to the isolated resume workflow."""

    return _resume.ResumeServices(
        data_root=DATA_ROOT,
        state_store_type=StateStore,
        config_type=OperatorConfig,
        git_head=_git_head,
        sync_sampling_target=_sync_sampling_target,
        reattach_decision=_reattach_decision,
        attach_to_site=_attach_to_site,
        usage_policy_preflight=_usage_policy_preflight,
        ensure_home_headroom=ensure_home_headroom,
        stager_type=Stager,
        stage_hf_token=_stage_hf_token,
        recorded_split_resume_bundle=_recorded_split_resume_bundle,
        current_label_plan=_current_label_plan,
        ensure_llama_server=ensure_llama_server,
        ensure_relay_at_destination=_ensure_relay_at_destination,
        race_queued_start=_race_queued_start,
        classify_or_continue=_classify_or_continue,
        resume_command=_resume_command,
        milestone=_milestone,
        monitor_until_terminal=_monitor_until_terminal,
        publish_split=publish_split,
        mark_remote_status=mark_remote_status,
        split_finalization=split_finalization,
        assert_remote_exit_zero=assert_remote_exit_zero,
        submission_headroom_bytes=_SUBMISSION_HEADROOM_BYTES,
    )


def _resume_run(run_id: str, args: SimpleNamespace) -> int:
    """Resume a durable run through the isolated orchestration module."""

    global _ACTIVE_RUN_ID
    _ACTIVE_RUN_ID = run_id
    return _resume.resume_run(run_id, args, _resume_services())


def _reclaim_terminal_managed_storage(probes: list[SiteProbe]) -> None:
    """Remove only terminal pipeline roots on every reachable frontend."""

    reachable = [probe for probe in probes if probe.reachable]
    if not reachable:
        return
    _milestone("Reclaiming terminal pipeline-managed storage on reachable sites")
    for probe in reachable:
        removed = cleanup_managed_runs(
            SshClient(target=probe.target),
            execute=True,
        )
        if removed:
            _milestone(
                f"Site {probe.name}: removed {len(removed)} terminal managed run(s)"
            )


def _run(args: SimpleNamespace) -> int:
    if not DATA_ROOT.exists():
        raise RuntimeError(f"external data root is unavailable: {DATA_ROOT}")
    _milestone("Validating the local source checkout")
    source_commit = _git_head()
    _milestone(f"Source commit: {source_commit[:12]}")
    _milestone("Resolving immutable input revision")
    input_revision = _resolve_input_revision(args.input_revision, args.stage)
    _milestone(f"Input revision: {input_revision[:12]}")
    sampling_target = _sampling_target_for_run(args)
    config = OperatorConfig.build(
        scope=args.scope,
        region=args.region,
        stage=args.stage,
        source_commit=source_commit,
        input_revision=input_revision,
        batch_size=args.batch_size,
        row_limit=args.row_limit,
        llama_parallel=args.llama_parallel,
        llama_per_slot_context=args.llama_per_slot_context,
        request_concurrency=args.request_concurrency,
        sampling_target=sampling_target,
        sampling_seed=getattr(args, "sampling_seed", DEFAULT_SAMPLING_SEED),
        sampling_h3_resolution=getattr(
            args, "sampling_h3_resolution", DEFAULT_SAMPLING_H3_RESOLUTION
        ),
    )
    if config.prompt_version == V2_LOGIT_PROMPT_VERSION and config.stage is Stage.LABEL:
        raise RuntimeError("worldwide V2 labeling requires --stage all")
    store = StateStore(DATA_ROOT)
    store.load_or_create(config.run_identity)
    _sync_sampling_target(store, config)
    global _ACTIVE_RUN_ID
    _ACTIVE_RUN_ID = config.run_id
    _milestone(f"Durable run ID: {config.run_id}")

    candidate = _reattach_decision(store.load())
    if candidate is not None:
        site, job_id = candidate
        _milestone(
            f"Reattaching to live Grid'5000 job {job_id} on {site} "
            "(no new submission, no site probing)"
        )
        _classify_or_continue(
            args=args,
            store=store,
            config=config,
            site=site,
            job_id=job_id,
            destination_site=site,
        )
        return 0

    requirements = SiteRequirements(
        gpu_memory_mb=args.gpu_memory_mb,
        persistent_free_bytes=required_staging_headroom(
            args.stage,
            args.remote_free_bytes,
        ),
    )
    targets = tuple(dict.fromkeys(args.site))

    def probe_targets(description: str) -> list[SiteProbe]:
        probes: list[SiteProbe] = []
        with _CONSOLE.progress(description=description, total=len(targets)) as progress:
            for target in targets:
                _milestone(f"Probing Grid'5000 site: {target}")
                probe = probe_site(target, config.run_id, requirements)
                probes.append(probe)
                if probe.reachable:
                    _milestone(
                        f"Site {probe.name}: reachable, GPU {probe.gpu_memory_mb} MiB, "
                        f"persistent free {probe.persistent_free_bytes // 1024**3} GiB"
                    )
                else:
                    _milestone(f"Site {target}: unavailable")
                progress.advance()
        return probes

    probes = probe_targets("Probing Grid'5000 sites")
    _reclaim_terminal_managed_storage(probes)
    if any(probe.reachable for probe in probes):
        probes = probe_targets("Re-probing Grid'5000 sites after cleanup")

    if cleanup_can_restore_compatibility(probes, requirements):
        _milestone(
            "Reclaiming only terminal pipeline-managed storage before site selection"
        )
        _reclaim_terminal_managed_storage(probes)
        probes = probe_targets("Re-probing Grid'5000 sites after cleanup")
    try:
        selection = select_site(probes, requirements)
    except NoCompatibleSiteError:
        if not cleanup_can_restore_compatibility(probes, requirements):
            raise
        _milestone(
            "No compatible site; reclaiming only completed or failed managed runs"
        )
        _reclaim_terminal_managed_storage(probes)
        probes = probe_targets("Re-probing Grid'5000 sites")
        selection = select_site(probes, requirements)
    target = selection.selected.target
    active_site = selection.selected.name
    _milestone(f"Selected Grid'5000 site: {selection.selected.name}")
    ssh = SshClient(target=target, command_timeout=1800)
    home = _remote_home(ssh)
    layout = RemoteLayout(home / "osm-polygon-operator" / config.run_id)
    _milestone("Checking live Grid'5000 usage-policy constraints")
    _usage_policy_preflight(ssh, selection.selected.name)
    _milestone("Usage-policy preflight passed; submissions are window-bound")
    _milestone("Enforcing Grid'5000 home soft-quota headroom")
    initial_headroom = (
        requirements.resume_persistent_free_bytes
        if selection.selected.has_managed_run
        else requirements.persistent_free_bytes
    )
    ensure_home_headroom(
        ssh,
        protected_root=layout.root,
        minimum_headroom_bytes=initial_headroom,
    )
    _milestone("Storage preflight passed")

    def submission_preflight() -> None:
        _usage_policy_preflight(ssh, active_site)
        ensure_home_headroom(
            ssh,
            protected_root=layout.root,
            minimum_headroom_bytes=_SUBMISSION_HEADROOM_BYTES,
        )

    oar = OarClient(
        ssh,
        preflight=submission_preflight,
    )
    stager = Stager(ssh)
    controller = Controller(
        config=config,
        state=store,
        ssh=ssh,
        oar=oar,
        stager=stager,
        layout=layout,
        emit=_emit,
        poll_seconds=args.poll_seconds,
    )
    _milestone("Preparing remote checkout and locked environment")
    controller.prepare(site=selection.selected.name)
    _milestone("Remote checkout and environment are ready")
    _stage_hf_token(stager, layout)
    _milestone("Hugging Face credential staged privately for remote checkpoint writes")

    durable = store.load()
    split_done = "split_output_job_id" in durable.facts
    if config.stage in {Stage.SPLIT, Stage.ALL} and not split_done:
        for allocation in range(1, 101):
            job_id = controller.submit(component=Stage.SPLIT)
            optimized_site, optimized_job_id = _race_queued_start(
                args,
                store,
                config,
                active_site,
                job_id,
            )
            if optimized_site != active_site:
                mark_remote_status(ssh, layout, "failed")
                active_site = optimized_site
                ssh, layout, oar, controller = _attach_to_site(
                    store,
                    config,
                    active_site,
                    poll_seconds=args.poll_seconds,
                    preflight=submission_preflight,
                )
                stager = Stager(ssh)
                _stage_hf_token(stager, layout)
            job_id = optimized_job_id
            print(
                f"Submitted sentence splitting job {job_id} (allocation {allocation})",
                flush=True,
            )
            outcome = controller.monitor(job_id, log_name="build.stdout.log")
            if outcome is not JobState.TERMINATED:
                raise RuntimeError("sentence splitting allocation failed")
            _clear_consumed_split_resume_bundle(store)
            split_exit_code = remote_exit_code(ssh, layout, job_id, "build.exit_code")
            if split_exit_code == 0:
                break
            if split_exit_code != 130:
                raise RuntimeError("sentence splitting payload failed")
            current = store.load()
            if current.phase not in {RunPhase.RUNNING, RunPhase.QUEUED}:
                raise RuntimeError("split continuation has invalid durable state")
            store.transition(
                expected=current.phase,
                target=RunPhase.REMOTE_PREPARED,
                facts={"continued_after_job": job_id},
            )
            print("Valid remote checkpoints preserved; continuing.", flush=True)
        else:
            raise RuntimeError("sentence splitting exceeded allocation safety bound")
        _transition_terminal(
            store,
            expected=(RunPhase.RUNNING, RunPhase.QUEUED),
            target=RunPhase.CHECKPOINTED,
            facts={"split_job_id": job_id},
        )
        split_finalization.finalize_split_checkpointed(
            store=store,
            config=config,
            ssh=ssh,
            layout=layout,
            oar=oar,
            poll_seconds=args.poll_seconds,
        )
        if config.stage is Stage.SPLIT:
            return 0

    if config.stage in {Stage.LABEL, Stage.ALL}:
        # For `all`, the finalizer's persisted output is authoritative.
        if config.stage is Stage.ALL:
            final_job_raw = store.load().facts["split_output_job_id"]
            assert type(final_job_raw) is int
            input_parquet = (
                layout.logs / str(final_job_raw) / "output/sentences.parquet"
            )
        else:
            input_parquet = layout.root / "input/sentences.parquet"
        if config.prompt_version == V2_LOGIT_PROMPT_VERSION:
            if config.stage is not Stage.ALL:
                raise RuntimeError("worldwide V2 labeling requires --stage all")
            _milestone("Enriching split output with pinned polygon areas")
            input_parquet = stager.prepare_v2_input(config, layout, input_parquet)
        _milestone("Staging immutable labeling assets")
        assets = stager.prepare_label_assets(
            config,
            layout,
            download_input=config.stage is Stage.LABEL,
        )
        _milestone("Input, model, and tokenizer assets are ready")
        if config.stage is Stage.ALL and store.load().phase is RunPhase.VALIDATED:
            store.transition(
                expected=RunPhase.VALIDATED,
                target=RunPhase.REMOTE_PREPARED,
                facts={"label_assets_ready": True},
            )
        if not assets.llama_server_ready:
            ensure_llama_server(ssh, oar, store, layout, args.poll_seconds)
        if config.stage is Stage.ALL:
            assets = replace(
                assets,
                input_parquet=input_parquet,
                llama_server_ready=True,
            )
        while True:
            label_plan = _current_label_plan(store, config, layout)
            for allocation in range(1, 101):
                job_id = controller.submit(
                    component=Stage.LABEL,
                    input_parquet=assets.input_parquet,
                    model_file=assets.model_file,
                    tokenizer_dir=assets.tokenizer_dir,
                    walltime_seconds=PREFERRED_LABEL_WALLTIME_SECONDS,
                    policy_type=policy_type_for(
                        datetime.now(tz=GRID5000_TZ),
                        walltime_seconds=PREFERRED_LABEL_WALLTIME_SECONDS,
                    ),
                    gpu_memory_mb=getattr(args, "gpu_memory_mb", 40_000),
                    label_plan=label_plan,
                )
                if config.stage is Stage.LABEL:
                    optimized_site, optimized_job_id = _race_queued_start(
                        args,
                        store,
                        config,
                        active_site,
                        job_id,
                    )
                    if optimized_site != active_site:
                        mark_remote_status(ssh, layout, "failed")
                        active_site = optimized_site
                        ssh, layout, oar, controller = _attach_to_site(
                            store,
                            config,
                            active_site,
                            poll_seconds=args.poll_seconds,
                            preflight=submission_preflight,
                        )
                        stager = Stager(ssh)
                        _stage_hf_token(stager, layout)
                        assets = stager.prepare_label_assets(
                            config,
                            layout,
                            download_input=True,
                        )
                    job_id = optimized_job_id
                print(
                    f"Submitted labeling job {job_id} (allocation {allocation})",
                    flush=True,
                )
                outcome = controller.monitor(job_id, log_name="labeling.stdout.log")
                if outcome is not JobState.TERMINATED:
                    raise RuntimeError("labeling allocation failed")
                assert_remote_exit_zero(ssh, layout, job_id, "labeling.exit_code")
                output_dir = (
                    label_plan.output_dir
                    if label_plan is not None
                    else layout.label_output
                )
                complete = (
                    result_text(
                        ssh.run(
                            "if test -f "
                            f"{output_dir!s}/manifest.json; "
                            "then printf yes; else printf no; fi"
                        )
                    )
                    == "yes"
                )
                if complete:
                    break
                current = store.load()
                if current.phase not in {RunPhase.RUNNING, RunPhase.QUEUED}:
                    raise RuntimeError("label continuation has invalid durable state")
                store.transition(
                    expected=current.phase,
                    target=RunPhase.REMOTE_PREPARED,
                    facts={"continued_after_job": job_id},
                )
                print("Validated label checkpoints preserved; continuing.", flush=True)
            else:
                raise RuntimeError("labeling exceeded allocation safety bound")
            _apply_classification(
                store=store,
                config=config,
                ssh=ssh,
                layout=layout,
                job_id=job_id,
                active_stage=Stage.LABEL.value,
                classification=ExitClass.COMPLETE,
                label_plan=label_plan,
            )
            if store.load().phase is RunPhase.COMPLETE:
                break
    return 0


def _status(args: SimpleNamespace) -> int:
    path = DATA_ROOT / "runs" / args.run_id / "state.json"
    if not path.is_file():
        raise RuntimeError("run state does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("run state is not a JSON object")
    _CONSOLE.json(payload)
    return 0


def _resume_handler(args: SimpleNamespace) -> int:
    return _resume_run(args.run_id, args)


def _cleanup(args: SimpleNamespace) -> int:
    removed = cleanup_managed_runs(
        SshClient(target=args.site),
        execute=args.execute,
    )
    label = "removed" if args.execute else "eligible"
    for path in removed:
        _CONSOLE.plain(f"{label}: {path}")
    if not removed:
        _CONSOLE.plain("No pipeline-managed completed or failed runs are eligible.")
    return 0


@app.command("run")
def run_command(
    scope: Annotated[Scope, typer.Option("--scope")],
    stage: Annotated[Stage, typer.Option("--stage")],
    region: Annotated[str | None, typer.Option("--region")] = None,
    input_revision: Annotated[str | None, typer.Option("--input-revision")] = None,
    site: Annotated[list[str] | None, typer.Option("--site")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 128,
    row_limit: Annotated[int, typer.Option("--row-limit")] = 0,
    sampling_target: Annotated[int | None, typer.Option("--sampling-target")] = None,
    sampling_seed: Annotated[
        str, typer.Option("--sampling-seed")
    ] = DEFAULT_SAMPLING_SEED,
    sampling_h3_resolution: Annotated[
        int, typer.Option("--sampling-h3-resolution")
    ] = DEFAULT_SAMPLING_H3_RESOLUTION,
    llama_parallel: Annotated[int, typer.Option("--llama-parallel")] = 8,
    llama_per_slot_context: Annotated[
        int, typer.Option("--llama-per-slot-context")
    ] = 8192,
    request_concurrency: Annotated[
        int | None, typer.Option("--request-concurrency")
    ] = None,
    gpu_memory_mb: Annotated[int, typer.Option("--gpu-memory-mb")] = 40_000,
    remote_free_bytes: Annotated[int, typer.Option("--remote-free-bytes")] = 8
    * 1024**3,
    poll_seconds: Annotated[float, typer.Option("--poll-seconds")] = 30.0,
    detach: Annotated[
        bool, typer.Option("--detach", help="Run under a detached local supervisor.")
    ] = False,
) -> int:
    """Run or resume a production workflow."""

    args = SimpleNamespace(
        command="run",
        scope=scope.value,
        stage=stage.value,
        region=region,
        input_revision=input_revision,
        site=[*DEFAULT_SITES, *(site or [])],
        batch_size=batch_size,
        row_limit=row_limit,
        sampling_target=sampling_target,
        sampling_seed=sampling_seed,
        sampling_h3_resolution=sampling_h3_resolution,
        llama_parallel=llama_parallel,
        llama_per_slot_context=llama_per_slot_context,
        request_concurrency=request_concurrency,
        gpu_memory_mb=gpu_memory_mb,
        remote_free_bytes=remote_free_bytes,
        poll_seconds=poll_seconds,
        optimize_continuations=True,
    )
    if detach:
        launch = start_detached_supervisor(
            _detached_run_arguments(args),
            data_root=DATA_ROOT,
        )
        _announce_detached(launch)
        return 0
    return _dispatch(_run, args)


@app.command("status")
def status_command(run_id: str) -> int:
    """Show durable local run state."""

    return _dispatch(
        _status,
        SimpleNamespace(command="status", run_id=run_id),
    )


@app.command("resume")
def resume_command(
    run_id: str,
    site: Annotated[list[str] | None, typer.Option("--site")] = None,
    gpu_memory_mb: Annotated[int, typer.Option("--gpu-memory-mb")] = 40_000,
    poll_seconds: Annotated[float, typer.Option("--poll-seconds")] = 30.0,
    sampling_target: Annotated[int | None, typer.Option("--sampling-target")] = None,
    detach: Annotated[
        bool, typer.Option("--detach", help="Run under a detached local supervisor.")
    ] = False,
) -> int:
    """Resume a run, optionally extending its V2 sampling target."""

    if detach:
        launch = start_detached_supervisor(
            _detached_resume_arguments(
                run_id,
                site=[*DEFAULT_SITES, *(site or [])],
                gpu_memory_mb=gpu_memory_mb,
                poll_seconds=poll_seconds,
                sampling_target=sampling_target,
            ),
            data_root=DATA_ROOT,
            run_id=run_id,
        )
        _announce_detached(launch)
        return 0
    return _dispatch(
        _resume_handler,
        SimpleNamespace(
            command="resume",
            run_id=run_id,
            site=[*DEFAULT_SITES, *(site or [])],
            gpu_memory_mb=gpu_memory_mb,
            poll_seconds=poll_seconds,
            sampling_target=sampling_target,
            optimize_continuations=True,
        ),
    )


@app.command("cleanup")
def cleanup_command(
    site: Annotated[str, typer.Option("--site")],
    execute: Annotated[bool, typer.Option("--execute")] = False,
) -> int:
    """Preview or remove completed pipeline-managed remote runs."""

    return _dispatch(
        _cleanup,
        SimpleNamespace(
            command="cleanup",
            site=site,
            execute=execute,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Installed entry point."""

    global _ACTIVE_RUN_ID
    prior_active = _ACTIVE_RUN_ID
    _ACTIVE_RUN_ID = None
    command = typer.main.get_command(app)
    try:
        result = command.main(
            args=argv,
            prog_name="osm-polygon-grid5000",
            standalone_mode=False,
        )
        return int(result) if result is not None else 0
    except click.UsageError as exc:
        exc.show(file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    except click.exceptions.Exit as exc:
        raise SystemExit(exc.exit_code) from exc
    except (KeyboardInterrupt, click.Abort, _LocalMonitoringInterrupted):
        run_id = _ACTIVE_RUN_ID if _ACTIVE_RUN_ID else None
        if run_id is None:
            run_id = prior_active if prior_active else None
        if run_id is None:
            _CONSOLE.plain(
                "Local monitoring stopped; the remote job and checkpoints were "
                "preserved.",
                error=True,
            )
        else:
            _CONSOLE.plain(
                "Local monitoring stopped; the remote job and checkpoints were "
                "preserved.",
                error=True,
            )
            _CONSOLE.plain(
                f"Resume with: {_resume_command(run_id)}",
                error=True,
            )
        sys.exit(130)
    except Exception as exc:
        _CONSOLE.error(str(exc))
        return 1
    finally:
        _ACTIVE_RUN_ID = prior_active


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["app", "main"]
