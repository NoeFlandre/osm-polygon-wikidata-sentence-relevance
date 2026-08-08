"""Public Mac-side command for autonomous Grid'5000 dataset production."""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Final

import click
import typer

from osm_polygon_sentence_relevance.labeling.v2_contracts import (
    V2_LOGIT_PROMPT_VERSION,
)
from osm_polygon_sentence_relevance.operator import preflight as _preflight
from osm_polygon_sentence_relevance.operator import recorded_job, relay
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
    UNPREDICTED_TRIAL_SECONDS,
    ReplacementCandidate,
    attempt_immediate_replacement,
    policy_type_for,
    rank_replacement_candidates,
    should_seek_replacement,
)
from osm_polygon_sentence_relevance.operator.job_monitor import monitor_job_with_log
from osm_polygon_sentence_relevance.operator.llama_server import ensure_llama_server
from osm_polygon_sentence_relevance.operator.oar import (
    GRID5000_TZ,
    ExitClass,
    JobState,
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
    publish_label,
    publish_split,
    remote_exit_code,
)
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
from osm_polygon_sentence_relevance.operator.workflows import (
    MICRO_LABEL_WALLTIME_SECONDS,
    RemoteLayout,
    label_submission,
    split_finalization_submission,
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


def _result_text(result: Any) -> str:
    """Return ``result.text`` if available, else ``result.stdout``."""

    text_attr = getattr(result, "text", None)
    if text_attr is not None:
        return text_attr
    return getattr(result, "stdout", "")


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


def _checkpoint_root(layout: RemoteLayout) -> str:
    """The remote root consumed by :class:`CheckpointStore`.

    The real production layout written by :class:`CheckpointStore` is::

        ${label_work}/progress.json
        ${label_work}/timing.json   (optional)
        ${label_work}/checkpoints/batch-NNNNNN.parquet
        ${label_work}/checkpoints/batch-NNNNNN.json

    There is no ``${label_work}/checkpoints/<run_id>`` directory.
    """

    return str(layout.label_work)


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
    """Prepare a continuation site before allowing a new submission.

    Same-site continuation reuses the already validated checkout and assets.
    Cross-site continuation performs the normal policy, quota, checkout and
    immutable-asset preflights before the validated relay is installed.
    """

    current = store.load()
    if current.phase is not RunPhase.REMOTE_PREPARED:
        store.transition(
            expected=current.phase,
            target=RunPhase.REMOTE_PREPARED,
            facts={"site": site, "job_id": current.facts.get("job_id")},
        )
    if relay_root is not None:
        ssh = SshClient(target=site, command_timeout=1800)
        home = _remote_home(ssh)
        layout = RemoteLayout(home / "osm-polygon-operator" / config.run_id)
        _usage_policy_preflight(ssh, site)
        ensure_home_headroom(
            ssh,
            protected_root=layout.root,
            minimum_headroom_bytes=LABEL_STAGING_HEADROOM_BYTES,
        )
        stager = Stager(ssh)
        stager.prepare(config, layout)
        _stage_hf_token(stager, layout)
        assets = stager.prepare_label_assets(config, layout, download_input=True)
        if not assets.llama_server_ready:

            def submission_preflight() -> None:
                _usage_policy_preflight(ssh, site)
                ensure_home_headroom(
                    ssh,
                    protected_root=layout.root,
                    minimum_headroom_bytes=_SUBMISSION_HEADROOM_BYTES,
                )

            oar = OarClient(
                ssh,
                preflight=submission_preflight,
            )
            ensure_llama_server(ssh, oar, store, layout, poll_seconds)
        store.transition(
            expected=RunPhase.REMOTE_PREPARED,
            target=RunPhase.REMOTE_PREPARED,
            facts={"resume_relay_root": str(relay_root)},
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
) -> None:
    """Drive durable state transitions for a classified terminal allocation.

    Re-entering ``FAILED`` is allowed for idempotent reclassification: if the
    same allocation is re-inspected and the new evidence still proves a
    deterministic failure, the state stays ``FAILED`` but ``failed_job_id``
    and the durable sequence advance.

    Re-entering ``FAILED`` and then transitioning to ``REMOTE_PREPARED`` is
    only permitted when the freshly inspected evidence classifies as
    ``CONTINUE`` and the previous ``FAILED`` was a misclassification of a
    walltime-killed allocation. Recovery facts are appended so the run
    identity is preserved and the recovery is auditable.
    """

    is_label = active_stage == Stage.LABEL.value
    current = store.load()
    is_recovery_from_failed = current.phase is RunPhase.FAILED
    if classification is ExitClass.FAILED:
        reason_token = (
            failure_reason_token if failure_reason_token else "deterministic-failure"
        )
        _transition_terminal(
            store,
            expected=(RunPhase.RUNNING, RunPhase.QUEUED, RunPhase.FAILED),
            target=RunPhase.FAILED,
            facts={"failed_job_id": job_id, "failure_reason": reason_token},
        )
        mark_remote_status(ssh, layout, "failed")
        raise RuntimeError(
            f"recorded allocation {job_id} failed deterministically "
            f"[reason={reason_token}]; not resubmitting automatically"
        )
    if classification is ExitClass.COMPLETE:
        if is_label:
            hub_commit: str | None = None
            if config.requirements.row_limit == 0:
                try:
                    hub_commit = label_publication_commit(ssh, layout, job_id)
                except RuntimeError as exc:
                    if str(exc) != (
                        "label publication did not report an immutable Hub commit"
                    ):
                        raise
                    hub_commit = publish_label(
                        ssh, layout, layout.label_output, config.output_dataset_id
                    )
            facts: dict[str, object] = {"label_job_id": job_id}
            if hub_commit is not None:
                facts["hub_commit"] = hub_commit
            if is_recovery_from_failed:
                facts["recovered_from_job_id"] = job_id
                facts["recovery_reason"] = (
                    "previously-failed allocation re-inspected as complete"
                )
                facts["recovery_attempt"] = _next_recovery_attempt(current.facts)
            _transition_terminal(
                store,
                expected=(RunPhase.RUNNING, RunPhase.QUEUED, RunPhase.FAILED),
                target=RunPhase.VALIDATED,
                facts=facts,
            )
            store.transition(
                expected=RunPhase.VALIDATED,
                target=RunPhase.VERIFYING,
                facts={"dataset_id": config.output_dataset_id},
            )
            store.transition(
                expected=RunPhase.VERIFYING,
                target=RunPhase.COMPLETE,
                facts={"published": config.requirements.row_limit == 0},
            )
            print(f"Labeling complete: run {config.run_id}", flush=True)
            mark_remote_status(ssh, layout, "complete")
            return
        if is_recovery_from_failed:
            _transition_terminal(
                store,
                expected=(RunPhase.FAILED,),
                target=RunPhase.CHECKPOINTED,
                facts={
                    "split_job_id": job_id,
                    "recovered_from_job_id": job_id,
                    "recovery_reason": (
                        "previously-failed split allocation re-inspected as complete"
                    ),
                    "recovery_attempt": _next_recovery_attempt(current.facts),
                },
            )
        else:
            _transition_terminal(
                store,
                expected=(RunPhase.RUNNING, RunPhase.QUEUED),
                target=RunPhase.CHECKPOINTED,
                facts={"split_job_id": job_id},
            )
        print(
            f"Sentence splitting checkpointed: run {config.run_id}; rerun to finalize.",
            flush=True,
        )
        return
    if classification is ExitClass.CONTINUE:
        facts = {"continued_after_job": job_id}
        if resume_artifact_path is not None:
            facts["resume_relay_path"] = resume_artifact_path
        if is_recovery_from_failed:
            facts["recovered_from_job_id"] = job_id
            facts["recovery_reason"] = (
                "walltime-killed allocation re-inspected with validated partial checkpoints"
            )
            facts["recovery_attempt"] = _next_recovery_attempt(current.facts)
        _transition_terminal(
            store,
            expected=(RunPhase.RUNNING, RunPhase.QUEUED, RunPhase.FAILED),
            target=RunPhase.REMOTE_PREPARED,
            facts=facts,
        )
        print(
            "Validated checkpoints preserved; rerun to continue on the next "
            "allocation (possibly another Grid'5000 site).",
            flush=True,
        )
        return
    raise RuntimeError(f"unhandled exit class: {classification}")


def _finalize_split_checkpointed(
    *,
    store: StateStore,
    config: OperatorConfig,
    ssh: SshClient,
    layout: RemoteLayout,
    oar: OarClient,
    poll_seconds: float,
) -> int:
    """Finalize a complete split checkpoint set and continue the workflow.

    This is shared by a fresh run and a resumed run.  For ``stage=split`` it
    publishes the validated split release.  For ``stage=all`` it preserves the
    finalized split output and reopens the durable state for label submission.
    """

    final_job = oar.submit(split_finalization_submission(config, layout))
    store.transition(
        expected=RunPhase.CHECKPOINTED,
        target=RunPhase.FINALIZING,
        facts={"finalization_job_id": final_job},
    )
    print(f"Submitted finalization job {final_job}", flush=True)
    monitor_job_with_log(
        ssh,
        oar,
        layout,
        final_job,
        "finalize.stdout.log",
        poll_seconds,
        sleeper=time.sleep,
    )
    assert_remote_exit_zero(ssh, layout, final_job, "finalize.exit_code")
    store.transition(
        expected=RunPhase.FINALIZING,
        target=RunPhase.VALIDATED,
        facts={"split_output_job_id": final_job},
    )
    if config.stage is Stage.SPLIT:
        output_dir = layout.logs / str(final_job) / "output"
        hub_commit = publish_split(
            ssh,
            layout,
            output_dir,
            config.output_dataset_id,
        )
        store.transition(
            expected=RunPhase.VALIDATED,
            target=RunPhase.COMPLETE,
            facts={"published": True, "hub_commit": hub_commit},
        )
        print(f"Sentence splitting complete: run {config.run_id}", flush=True)
        mark_remote_status(ssh, layout, "complete")
    else:
        store.transition(
            expected=RunPhase.VALIDATED,
            target=RunPhase.REMOTE_PREPARED,
            facts={"active_stage": Stage.SPLIT.value},
        )
    return final_job


def _classify_or_continue(
    args: SimpleNamespace,
    store: StateStore,
    config: OperatorConfig,
    site: str,
    job_id: int,
    *,
    destination_site: str | None = None,
) -> ExitClass:
    """Reattach to the recorded site and classify the terminal allocation.

    When the allocation is resumable, the orchestrator enters a continuation
    loop:

    1. retrieve and validate the checkpoint generation,
    2. probe all compatible sites,
    3. select the factual best site deterministically,
    4. reuse the same site without relay when appropriate,
    5. otherwise prepare exact historical source checkout/assets and stage
       the validated relay to the destination site,
    6. run usage-policy and quota preflights,
    7. submit exactly one new short allocation,
    8. atomically persist destination site/job ID before monitoring,
    9. monitor it until terminal,
    10. classify the new terminal state and repeat until complete.

    No recursion. The allocation safety bound is enforced by the caller.
    """

    ssh, layout, oar, controller = _attach_to_site(
        store, config, site, poll_seconds=args.poll_seconds
    )
    active = str(store.load().facts.get("active_stage", config.stage.value))
    is_label = active == Stage.LABEL.value
    log_name = "labeling.stdout.log" if is_label else "build.stdout.log"

    status = oar.status(job_id)
    if is_live_state(status.state):
        terminal = _monitor_until_terminal(controller, job_id, log_name=log_name)
        status = oar.status(job_id)
        if not _is_terminal_allocation(terminal):
            raise RuntimeError(
                f"recorded allocation {job_id} ended in {terminal.value}"
            )
    # MISSING jobs are first inspected; if no durable evidence exists at
    # all we treat it as an OAR bookkeeping loss and refuse to resubmit.
    if status.state is JobState.MISSING:
        log_dir = layout.logs / str(job_id)
        listing = ssh.run(
            f"test -d {log_dir!s} && find {log_dir!s} -mindepth 1 -maxdepth 1"
        )
        has_any = bool(_result_text(listing).strip())
        if not has_any:
            raise RuntimeError(
                f"recorded allocation {job_id} is missing from OAR with no "
                "durable evidence; refusing to resubmit"
            )
    # MISSING jobs with at least some durable evidence are classified from
    # those artifacts alone (exit file, manifest, checkpoints, progress).

    if active == Stage.SPLIT.value:
        split_inspection = inspect_split_resume(
            ssh=ssh,
            repo_id=config.output_dataset_id,
            input_repo_id=config.input_dataset_id,
            input_revision=config.input_dataset_revision or "",
            run_id=config.run_id,
            source_commit=config.source_commit,
            pipeline_version=config.pipeline_version,
            model_name=config.split_model,
            batch_size=config.requirements.batch_size,
            staging_revision=f"checkpoints/{config.run_id}",
            exit_file=str(layout.logs / str(job_id) / "build.exit_code"),
            cache_dir=config.data_root
            / "runs"
            / config.run_id
            / "split-checkpoint-cache",
        )
        classification = classify_split_terminal(
            status,
            split_inspection,
            exit_code=split_inspection.exit_code,
        )
        inspection: recorded_job.ResumeInspection | None = None
        failure_reason_token = (
            split_failure_reason(
                split_inspection,
                exit_code=split_inspection.exit_code,
            )
            if classification is ExitClass.FAILED
            else None
        )
    else:
        inspection = recorded_job.inspect_remote_resume(
            ssh,
            label_work_root=str(layout.label_work),
            label_output_root=str(layout.label_output),
            expected_identity=config.run_identity.checkpoint_dict(),
            exit_file=str(layout.logs / str(job_id) / "labeling.exit_code"),
        )
        classification = recorded_job.classify_terminal(status, inspection)
        failure_reason_token = (
            recorded_job.failure_reason(status, inspection)
            if classification is ExitClass.FAILED
            else None
        )

    relay_artifact_path: str | None = None
    if (
        classification is ExitClass.CONTINUE
        and destination_site is not None
        and destination_site != site
    ):
        relay_artifact_path = _relay_for_continuation(
            store=store,
            config=config,
            source_site=site,
            destination_site=destination_site,
        )

    _apply_classification(
        store=store,
        config=config,
        ssh=ssh,
        layout=layout,
        job_id=job_id,
        active_stage=active,
        classification=classification,
        resume_artifact_path=relay_artifact_path,
        failure_reason_token=failure_reason_token,
    )

    if classification is ExitClass.COMPLETE and not is_label:
        _finalize_split_checkpointed(
            store=store,
            config=config,
            ssh=ssh,
            layout=layout,
            oar=oar,
            poll_seconds=args.poll_seconds,
        )
        if config.stage is Stage.ALL:
            # The split output is now durable and the state is reopened at
            # REMOTE_PREPARED. Re-enter the persisted continuation path so
            # label assets and the next bounded GPU allocation are staged
            # without requiring a second local command.
            _resume_run(config.run_id, args)
            return (
                ExitClass.COMPLETE
                if store.load().phase is RunPhase.COMPLETE
                else ExitClass.CONTINUE
            )
        return ExitClass.COMPLETE

    if classification is not ExitClass.CONTINUE:
        return classification

    # Without an explicit ``destination_site`` the caller is asking only
    # for the classification of the recorded allocation. The continuation
    # loop (relay + new submission) is driven by the CLI driver, not by
    # this helper, so a ``None`` destination exits here with the recorded
    # job's classification.
    if destination_site is None:
        return classification

    # Continuation loop: select a site, optionally relay, submit exactly
    # one new short allocation, monitor it, and reclassify. The new
    # job ID is persisted atomically before monitoring starts. The
    # allocation safety bound is enforced by this loop's iteration cap.
    target_site = destination_site
    _milestone(f"Continuation selecting site: {target_site} (recorded site was {site})")
    relay_root: Path | None = None
    if target_site != site:
        # Same-site reuse: do not relay; the source already has the
        # validated checkpoints. Cross-site: relay once, then continue.
        if relay_artifact_path is None:
            relay_root = Path(
                _relay_for_continuation(
                    store=store,
                    config=config,
                    source_site=site,
                    destination_site=target_site,
                )
            )
        else:
            relay_root = Path(relay_artifact_path)

    _prepare_destination_for_resume(
        store=store,
        config=config,
        site=target_site,
        relay_root=relay_root,
        poll_seconds=args.poll_seconds,
    )

    for _iteration in range(1, 101):
        # Attach to the destination site and submit exactly one allocation.
        ssh_d, layout_d, oar_d, controller_d = _attach_to_site(
            store, config, target_site, poll_seconds=args.poll_seconds
        )
        # Refresh label assets so the new allocation sees the relay on disk.
        if relay_root is not None:
            _ensure_relay_at_destination(
                store=store,
                config=config,
                site=target_site,
                layout=layout_d,
                relay_root=relay_root,
            )
        if is_label:
            new_job_id = controller_d.submit(
                component=Stage.LABEL,
                input_parquet=layout_d.root / "input/sentences.parquet",
                model_file=layout_d.root / "model" / config.label_model_file,
                tokenizer_dir=layout_d.root / "tokenizer",
                walltime_seconds=MICRO_LABEL_WALLTIME_SECONDS,
                policy_type=policy_type_for(
                    datetime.now(tz=GRID5000_TZ),
                    walltime_seconds=MICRO_LABEL_WALLTIME_SECONDS,
                ),
                gpu_memory_mb=getattr(args, "gpu_memory_mb", 40_000),
            )
            continuation_log_name = "labeling.stdout.log"
        else:
            new_job_id = controller_d.submit(component=Stage.SPLIT)
            continuation_log_name = "build.stdout.log"
        if getattr(args, "optimize_continuations", False):
            optimized_site, optimized_job_id = _optimize_queued_start(
                args,
                store,
                config,
                target_site,
                new_job_id,
            )
        else:
            optimized_site, optimized_job_id = target_site, new_job_id
        if optimized_site != target_site:
            mark_remote_status(ssh_d, layout_d, "failed")
            target_site = optimized_site
            ssh_d, layout_d, oar_d, controller_d = _attach_to_site(
                store,
                config,
                target_site,
                poll_seconds=args.poll_seconds,
            )
            _stage_hf_token(Stager(ssh_d), layout_d)
        new_job_id = optimized_job_id
        # Controller.submit atomically persists SUBMITTED and the job ID
        # before returning. Do not duplicate that state transition here.
        current = store.load()
        if current.phase is not RunPhase.SUBMITTED:
            raise RuntimeError("continuation submit was not durably recorded")
        store.transition(
            expected=RunPhase.SUBMITTED,
            target=RunPhase.SUBMITTED,
            facts={"site": target_site, "destination_site": target_site},
        )
        print(
            f"Submitted continuation job {new_job_id} (allocation {_iteration})",
            flush=True,
        )
        terminal = controller_d.monitor(new_job_id, log_name=continuation_log_name)
        if not _is_terminal_allocation(terminal):
            raise RuntimeError("continuation allocation failed")
        status_d = oar_d.status(new_job_id)
        if is_label:
            inspection_d = recorded_job.inspect_remote_resume(
                ssh_d,
                label_work_root=str(layout_d.label_work),
                label_output_root=str(layout_d.label_output),
                expected_identity=config.run_identity.checkpoint_dict(),
                exit_file=str(layout_d.logs / str(new_job_id) / "labeling.exit_code"),
            )
            classification_d = recorded_job.classify_terminal(status_d, inspection_d)
            failure_reason_token_d = (
                recorded_job.failure_reason(status_d, inspection_d)
                if classification_d is ExitClass.FAILED
                else None
            )
        else:
            split_inspection_d = inspect_split_resume(
                ssh=ssh_d,
                repo_id=config.output_dataset_id,
                input_repo_id=config.input_dataset_id,
                input_revision=config.input_dataset_revision or "",
                run_id=config.run_id,
                source_commit=config.source_commit,
                pipeline_version=config.pipeline_version,
                model_name=config.split_model,
                batch_size=config.requirements.batch_size,
                staging_revision=f"checkpoints/{config.run_id}",
                exit_file=str(layout_d.logs / str(new_job_id) / "build.exit_code"),
                cache_dir=config.data_root
                / "runs"
                / config.run_id
                / "split-checkpoint-cache",
            )
            classification_d = classify_split_terminal(
                status_d,
                split_inspection_d,
                exit_code=split_inspection_d.exit_code,
            )
            failure_reason_token_d = (
                split_failure_reason(
                    split_inspection_d,
                    exit_code=split_inspection_d.exit_code,
                )
                if classification_d is ExitClass.FAILED
                else None
            )
        _apply_classification(
            store=store,
            config=config,
            ssh=ssh_d,
            layout=layout_d,
            job_id=new_job_id,
            active_stage=active,
            classification=classification_d,
            failure_reason_token=failure_reason_token_d,
        )
        if classification_d is ExitClass.COMPLETE:
            return ExitClass.COMPLETE
        if classification_d is ExitClass.CONTINUE:
            # The state is REMOTE_PREPARED again; submit the next bounded
            # allocation without requiring another local invocation.
            relay_root = None
            continue
        # FAILED: stop.
        raise RuntimeError(
            f"continuation allocation {new_job_id} failed deterministically "
            f"[reason={failure_reason_token_d or 'deterministic-failure'}]; "
            "not resubmitting automatically"
        )
    raise RuntimeError("continuation exceeded allocation safety bound")


def _relay_for_continuation(
    *,
    store: StateStore,
    config: OperatorConfig,
    source_site: str,
    destination_site: str,
) -> str:
    """Retrieve, validate, and stage the resume set for a new site."""

    source_ssh = SshClient(target=source_site, command_timeout=600)
    source_layout = RemoteLayout(
        _remote_home(source_ssh) / "osm-polygon-operator" / config.run_id
    )
    source_root = _checkpoint_root(source_layout)
    inventory = relay.retrieve_to_seagate(
        source=relay.RemoteTransfer(ssh_target=source_site),
        source_checkpoint_root=source_root,
        destination_root=DATA_ROOT / "runs",
        run_id=config.run_id,
        expected_run_identity=config.run_identity.checkpoint_dict(),
    )
    destination_ssh = SshClient(target=destination_site, command_timeout=600)
    destination_layout = RemoteLayout(
        _remote_home(destination_ssh) / "osm-polygon-operator" / config.run_id
    )
    destination_root = _checkpoint_root(destination_layout)
    relay.stage_to_destination(
        inventory=inventory,
        destination=relay.RemoteTransfer(ssh_target=destination_site),
        destination_checkpoint_root=destination_root,
    )
    store.transition(
        expected=store.load().phase,
        target=store.load().phase,
        facts={"relay_destination_site": destination_site},
    )
    return str(inventory.root)


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
    if replacement_status == "adopted":
        old_site = durable.facts.get("fallback_site")
        old_job = durable.facts.get("fallback_job_id")
        if (
            isinstance(old_site, str)
            and type(old_job) is int
            and old_job > 0
            and durable.facts.get("fallback_cancelled") is not True
        ):
            old_status = client(old_site)[2].status(old_job)
            if old_status.state is JobState.QUEUED:
                client(old_site)[2].cancel(old_job)
            current = store.load()
            store.transition(
                expected=current.phase,
                target=current.phase,
                facts={"fallback_cancelled": True},
            )
        return fallback_site, fallback_job_id

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
    excluded_trial_sites = (
        frozenset({existing_trial[0].site.name})
        if existing_trial is not None
        else frozenset()
    )
    candidates = rank_replacement_candidates(
        probes,
        requirements=requirements,
        excluded_sites=excluded_trial_sites,
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
            return oar.submit(split_submission(config, layout))
        label_assets = assets[site]
        return oar.submit(
            label_submission(
                config,
                layout,
                input_parquet=label_assets.input_parquet,
                model_file=label_assets.model_file,
                tokenizer_dir=label_assets.tokenizer_dir,
                walltime_seconds=MICRO_LABEL_WALLTIME_SECONDS,
                policy_type=policy_type_for(
                    datetime.now(tz=GRID5000_TZ),
                    walltime_seconds=MICRO_LABEL_WALLTIME_SECONDS,
                ),
                gpu_memory_mb=getattr(args, "gpu_memory_mb", 40_000),
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


def _resume_run(run_id: str, args: SimpleNamespace) -> int:
    """Resume or classify a historical run by its durable run ID.

    Loads the persisted ``state.json`` for ``run_id`` and reconstructs the
    immutable operator configuration from ``run_identity``. Does not require
    the current local Git HEAD to match the run's recorded source commit.
    Never creates a new run ID. Never submits if the recorded job is live.
    """

    if not re.fullmatch(r"[0-9a-f]{20}", run_id):
        raise RuntimeError("run ID must be twenty lowercase hexadecimal characters")
    state_path = DATA_ROOT / "runs" / run_id / "state.json"
    if not state_path.is_file():
        raise RuntimeError(f"run state does not exist: {state_path}")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    persisted_identity = payload["run_identity"]
    if not isinstance(persisted_identity, dict):
        raise RuntimeError("persisted run identity is not an object")
    persisted_facts = payload.get("facts", {})
    if not isinstance(persisted_facts, dict):
        raise RuntimeError("persisted run facts are not an object")
    recorded_target = persisted_facts.get("sampling_target")
    requested_target = getattr(args, "sampling_target", None)
    if requested_target is not None:
        persisted_identity = {
            **persisted_identity,
            "sampling_target": requested_target,
        }
    elif (
        "sampling_version" in persisted_identity
        and "sampling_target" not in persisted_identity
        and isinstance(recorded_target, int)
        and not isinstance(recorded_target, bool)
        and recorded_target > 0
    ):
        persisted_identity = {
            **persisted_identity,
            "sampling_target": recorded_target,
        }
    config = OperatorConfig.from_persisted(persisted_identity)
    if config.run_id != run_id:
        raise RuntimeError(
            "persisted run identity does not reproduce the requested run ID"
        )
    store = StateStore(DATA_ROOT)
    store.load_or_create(config.run_identity)
    _sync_sampling_target(store, config)
    global _ACTIVE_RUN_ID
    _ACTIVE_RUN_ID = run_id
    durable = store.load()
    candidate = _reattach_decision(durable)
    if candidate is None and durable.phase is RunPhase.REMOTE_PREPARED:
        site_value = durable.facts.get("site")
        if not isinstance(site_value, str) or not site_value:
            raise RuntimeError("prepared continuation has no recorded site")
        _milestone(f"Resuming prepared continuation on site {site_value}")
        ssh, layout, oar, controller = _attach_to_site(
            store, config, site_value, poll_seconds=args.poll_seconds
        )
        _usage_policy_preflight(ssh, site_value)
        ensure_home_headroom(
            ssh,
            protected_root=layout.root,
            minimum_headroom_bytes=_SUBMISSION_HEADROOM_BYTES,
        )
        stager = Stager(ssh)
        stager.prepare(config, layout)
        _stage_hf_token(stager, layout)
        if config.prompt_version == V2_LOGIT_PROMPT_VERSION:
            split_job_raw = durable.facts.get("split_output_job_id")
            if type(split_job_raw) is not int:
                raise RuntimeError("V2 continuation has no split output")
            split_output = layout.logs / str(split_job_raw) / "output/sentences.parquet"
            v2_input = stager.prepare_v2_input(config, layout, split_output)
            assets = stager.prepare_label_assets(config, layout, download_input=False)
            assets = replace(assets, input_parquet=v2_input)
        else:
            assets = stager.prepare_label_assets(
                config,
                layout,
                download_input=config.stage is Stage.LABEL,
            )
            if config.stage is Stage.ALL:
                split_job_raw = durable.facts.get("split_output_job_id")
                if type(split_job_raw) is not int or split_job_raw <= 0:
                    raise RuntimeError("stage-all continuation has no split output")
                input_parquet = (
                    layout.logs / str(split_job_raw) / "output/sentences.parquet"
                )
                ssh.run(f"test -s {input_parquet}")
                assets = replace(assets, input_parquet=input_parquet)
        if not assets.llama_server_ready:
            ensure_llama_server(ssh, oar, store, layout, args.poll_seconds)
        relay_root_value = durable.facts.get("resume_relay_root")
        if isinstance(relay_root_value, str):
            _ensure_relay_at_destination(
                store=store,
                config=config,
                site=site_value,
                layout=layout,
                relay_root=Path(relay_root_value),
            )
        job_id = controller.submit(
            component=Stage.LABEL,
            input_parquet=assets.input_parquet,
            model_file=assets.model_file,
            tokenizer_dir=assets.tokenizer_dir,
            walltime_seconds=MICRO_LABEL_WALLTIME_SECONDS,
            policy_type=policy_type_for(
                datetime.now(tz=GRID5000_TZ),
                walltime_seconds=MICRO_LABEL_WALLTIME_SECONDS,
            ),
            gpu_memory_mb=getattr(args, "gpu_memory_mb", 40_000),
        )
        print(f"Submitted continuation job {job_id}", flush=True)
        candidate = (site_value, job_id)
    if candidate is None:
        print(
            f"[operator] run {run_id} has no recorded live allocation; nothing "
            "to reattach",
            flush=True,
        )
        return 0
    site, job_id = candidate
    site, job_id = _optimize_queued_start(
        args,
        store,
        config,
        site,
        job_id,
    )
    _milestone(f"Resuming run {run_id} on site {site}, job {job_id}")
    classification = _classify_or_continue(
        args=args,
        store=store,
        config=config,
        site=site,
        job_id=job_id,
        destination_site=site,
    )
    if classification is ExitClass.CONTINUE:
        _milestone(
            "Validated checkpoints preserved; submit the next allocation via "
            f"{_resume_command(run_id)} or `run`."
        )
    return 0


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
            optimized_site, optimized_job_id = _optimize_queued_start(
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
        _finalize_split_checkpointed(
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
        for allocation in range(1, 101):
            job_id = controller.submit(
                component=Stage.LABEL,
                input_parquet=assets.input_parquet,
                model_file=assets.model_file,
                tokenizer_dir=assets.tokenizer_dir,
                walltime_seconds=MICRO_LABEL_WALLTIME_SECONDS,
                policy_type=policy_type_for(
                    datetime.now(tz=GRID5000_TZ),
                    walltime_seconds=MICRO_LABEL_WALLTIME_SECONDS,
                ),
                gpu_memory_mb=getattr(args, "gpu_memory_mb", 40_000),
            )
            if config.stage is Stage.LABEL:
                optimized_site, optimized_job_id = _optimize_queued_start(
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
            complete = (
                _result_text(
                    ssh.run(
                        "if test -f "
                        f"{layout.label_output!s}/manifest.json; "
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
        _transition_terminal(
            store,
            expected=(RunPhase.RUNNING, RunPhase.QUEUED),
            target=RunPhase.VALIDATED,
            facts={"label_job_id": job_id},
        )
        label_hub_commit: str | None = None
        if config.requirements.row_limit == 0:
            label_hub_commit = label_publication_commit(ssh, layout, job_id)
        store.transition(
            expected=RunPhase.VALIDATED,
            target=RunPhase.VERIFYING,
            facts={
                "dataset_id": config.output_dataset_id,
                **(
                    {"hub_commit": label_hub_commit}
                    if label_hub_commit is not None
                    else {}
                ),
            },
        )
        store.transition(
            expected=RunPhase.VERIFYING,
            target=RunPhase.COMPLETE,
            facts={"published": config.requirements.row_limit == 0},
        )
        print(f"Labeling complete: run {config.run_id}", flush=True)
        mark_remote_status(ssh, layout, "complete")
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
) -> int:
    """Resume a run, optionally extending its V2 sampling target."""

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
