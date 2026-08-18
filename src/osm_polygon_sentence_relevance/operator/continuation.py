"""Continuation state machine for Grid'5000 allocations.

The CLI adapter supplies side-effecting seams. This module owns terminal
classification, checkpoint relay, and the bounded continuation loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

from osm_polygon_sentence_relevance.operator.config import OperatorConfig, Stage
from osm_polygon_sentence_relevance.operator.label_lanes import LabelLanePlan
from osm_polygon_sentence_relevance.operator.label_submission import (
    submit_preferred_label,
)
from osm_polygon_sentence_relevance.operator.oar import (
    ExitClass,
    JobState,
    is_live_state,
)
from osm_polygon_sentence_relevance.operator.recorded_job import ResumeInspection
from osm_polygon_sentence_relevance.operator.result_text import result_text
from osm_polygon_sentence_relevance.operator.state import RunPhase, StateStore
from osm_polygon_sentence_relevance.operator.workflows import (
    RemoteLayout,
)


@dataclass(frozen=True, slots=True)
class ContinuationServices:
    """Replaceable CLI seams used by the continuation workflow."""

    attach_to_site: Callable[..., tuple[Any, RemoteLayout, Any, Any]]
    current_label_plan: Callable[..., LabelLanePlan | None]
    monitor_until_terminal: Callable[..., JobState]
    is_terminal_allocation: Callable[[JobState], bool]
    inspect_split_resume: Callable[..., Any]
    classify_split_terminal: Callable[..., ExitClass]
    split_failure_reason: Callable[..., str]
    recorded_job: Any
    relay_for_continuation: Callable[..., str]
    apply_classification: Callable[..., None]
    split_finalization: Any
    resume_run: Callable[..., int]
    race_queued_start: Callable[..., tuple[str, int]]
    milestone: Callable[[str], None]
    prepare_destination_for_resume: Callable[..., None]
    ensure_relay_at_destination: Callable[..., None]
    recorded_split_resume_bundle: Callable[[Any], PurePosixPath | None]
    clear_consumed_split_resume_bundle: Callable[[Any], None]
    stage_hf_token: Callable[..., None]
    stager_type: Any
    mark_remote_status: Callable[..., None]


def classify_or_continue(
    args: SimpleNamespace,
    store: StateStore,
    config: OperatorConfig,
    site: str,
    job_id: int,
    *,
    destination_site: str | None = None,
    services: ContinuationServices,
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

    ssh, layout, oar, controller = services.attach_to_site(
        store, config, site, poll_seconds=args.poll_seconds
    )
    active = str(store.load().facts.get("active_stage", config.stage.value))
    is_label = active == Stage.LABEL.value
    current_label_plan = (
        services.current_label_plan(store, config, layout) if is_label else None
    )
    log_name = "labeling.stdout.log" if is_label else "build.stdout.log"

    status = oar.status(job_id)
    if is_live_state(status.state):
        terminal = services.monitor_until_terminal(
            controller, job_id, log_name=log_name
        )
        status = oar.status(job_id)
        if not services.is_terminal_allocation(terminal):
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
        has_any = bool(result_text(listing).strip())
        if not has_any:
            raise RuntimeError(
                f"recorded allocation {job_id} is missing from OAR with no "
                "durable evidence; refusing to resubmit"
            )
    # MISSING jobs with at least some durable evidence are classified from
    # those artifacts alone (exit file, manifest, checkpoints, progress).

    if active == Stage.SPLIT.value:
        split_inspection = services.inspect_split_resume(
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
        classification = services.classify_split_terminal(
            status,
            split_inspection,
            exit_code=split_inspection.exit_code,
        )
        inspection: ResumeInspection | None = None
        failure_reason_token = (
            services.split_failure_reason(
                split_inspection,
                exit_code=split_inspection.exit_code,
            )
            if classification is ExitClass.FAILED
            else None
        )
    else:
        label_work_root = (
            current_label_plan.work_dir
            if current_label_plan is not None
            else layout.label_work
        )
        label_output_root = (
            current_label_plan.output_dir
            if current_label_plan is not None
            else layout.label_output
        )
        expected_identity = (
            current_label_plan.config.run_identity.checkpoint_dict()
            if current_label_plan is not None
            else config.run_identity.checkpoint_dict()
        )
        inspection = services.recorded_job.inspect_remote_resume(
            ssh,
            label_work_root=str(label_work_root),
            label_output_root=str(label_output_root),
            expected_identity=expected_identity,
            exit_file=str(layout.logs / str(job_id) / "labeling.exit_code"),
        )
        classification = services.recorded_job.classify_terminal(status, inspection)
        failure_reason_token = (
            services.recorded_job.failure_reason(status, inspection)
            if classification is ExitClass.FAILED
            else None
        )

    relay_artifact_path: str | None = None
    if (
        classification is ExitClass.CONTINUE
        and destination_site is not None
        and destination_site != site
    ):
        relay_artifact_path = services.relay_for_continuation(
            store=store,
            config=config,
            source_site=site,
            destination_site=destination_site,
        )

    services.apply_classification(
        store=store,
        config=config,
        ssh=ssh,
        layout=layout,
        job_id=job_id,
        active_stage=active,
        classification=classification,
        resume_artifact_path=relay_artifact_path,
        failure_reason_token=failure_reason_token,
        label_plan=current_label_plan,
    )

    if classification is ExitClass.COMPLETE and not is_label:
        services.split_finalization.finalize_split_checkpointed(
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
            services.resume_run(config.run_id, args)
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
    services.milestone(
        f"Continuation selecting site: {target_site} (recorded site was {site})"
    )
    relay_root: Path | None = None
    if target_site != site:
        # Same-site reuse: do not relay; the source already has the
        # validated checkpoints. Cross-site: relay once, then continue.
        if relay_artifact_path is None:
            relay_root = Path(
                services.relay_for_continuation(
                    store=store,
                    config=config,
                    source_site=site,
                    destination_site=target_site,
                )
            )
        else:
            relay_root = Path(relay_artifact_path)

    services.prepare_destination_for_resume(
        store=store,
        config=config,
        site=target_site,
        relay_root=relay_root,
        poll_seconds=args.poll_seconds,
    )

    for _iteration in range(1, 101):
        # Attach to the destination site and submit exactly one allocation.
        ssh_d, layout_d, oar_d, controller_d = services.attach_to_site(
            store, config, target_site, poll_seconds=args.poll_seconds
        )
        # Refresh label assets so the new allocation sees the relay on disk.
        if relay_root is not None:
            services.ensure_relay_at_destination(
                store=store,
                config=config,
                site=target_site,
                layout=layout_d,
                relay_root=relay_root,
            )
        if is_label:
            iteration_label_plan = services.current_label_plan(store, config, layout_d)
            new_job_id = submit_preferred_label(
                controller_d,
                input_parquet=layout_d.root / "input/sentences.parquet",
                model_file=layout_d.root / "model" / config.label_model_file,
                tokenizer_dir=layout_d.root / "tokenizer",
                gpu_memory_mb=getattr(args, "gpu_memory_mb", 40_000),
                label_plan=iteration_label_plan,
            )
            continuation_log_name = "labeling.stdout.log"
        else:
            iteration_label_plan = None
            new_job_id = controller_d.submit(
                component=Stage.SPLIT,
                split_resume_bundle=services.recorded_split_resume_bundle(store),
            )
            continuation_log_name = "build.stdout.log"
        if getattr(args, "optimize_continuations", False):
            optimized_site, optimized_job_id = services.race_queued_start(
                args,
                store,
                config,
                target_site,
                new_job_id,
            )
        else:
            optimized_site, optimized_job_id = target_site, new_job_id
        if optimized_site != target_site:
            services.mark_remote_status(ssh_d, layout_d, "failed")
            target_site = optimized_site
            ssh_d, layout_d, oar_d, controller_d = services.attach_to_site(
                store,
                config,
                target_site,
                poll_seconds=args.poll_seconds,
            )
            services.stage_hf_token(services.stager_type(ssh_d), layout_d)
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
        if not services.is_terminal_allocation(terminal):
            raise RuntimeError("continuation allocation failed")
        if not is_label:
            services.clear_consumed_split_resume_bundle(store)
        status_d = oar_d.status(new_job_id)
        if is_label:
            label_work_root_d = (
                iteration_label_plan.work_dir
                if iteration_label_plan is not None
                else layout_d.label_work
            )
            label_output_root_d = (
                iteration_label_plan.output_dir
                if iteration_label_plan is not None
                else layout_d.label_output
            )
            expected_identity_d = (
                iteration_label_plan.config.run_identity.checkpoint_dict()
                if iteration_label_plan is not None
                else config.run_identity.checkpoint_dict()
            )
            inspection_d = services.recorded_job.inspect_remote_resume(
                ssh_d,
                label_work_root=str(label_work_root_d),
                label_output_root=str(label_output_root_d),
                expected_identity=expected_identity_d,
                exit_file=str(layout_d.logs / str(new_job_id) / "labeling.exit_code"),
            )
            classification_d = services.recorded_job.classify_terminal(
                status_d, inspection_d
            )
            failure_reason_token_d = (
                services.recorded_job.failure_reason(status_d, inspection_d)
                if classification_d is ExitClass.FAILED
                else None
            )
        else:
            split_inspection_d = services.inspect_split_resume(
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
            classification_d = services.classify_split_terminal(
                status_d,
                split_inspection_d,
                exit_code=split_inspection_d.exit_code,
            )
            failure_reason_token_d = (
                services.split_failure_reason(
                    split_inspection_d,
                    exit_code=split_inspection_d.exit_code,
                )
                if classification_d is ExitClass.FAILED
                else None
            )
        services.apply_classification(
            store=store,
            config=config,
            ssh=ssh_d,
            layout=layout_d,
            job_id=new_job_id,
            active_stage=active,
            classification=classification_d,
            failure_reason_token=failure_reason_token_d,
            label_plan=iteration_label_plan,
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


__all__ = ["ContinuationServices", "classify_or_continue"]
