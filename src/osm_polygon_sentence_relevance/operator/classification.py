"""Durable state transitions for classified Grid'5000 allocations.

The CLI adapter supplies I/O and publication seams. This module keeps the
phase-transition and recovery-fact policy independent from Typer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from osm_polygon_sentence_relevance.operator.config import OperatorConfig, Stage
from osm_polygon_sentence_relevance.operator.label_lanes import LabelLane, LabelLanePlan
from osm_polygon_sentence_relevance.operator.oar import ExitClass
from osm_polygon_sentence_relevance.operator.ssh import SshClient
from osm_polygon_sentence_relevance.operator.state import RunPhase, StateStore
from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout


@dataclass(frozen=True, slots=True)
class ClassificationServices:
    """Replaceable side-effect seams used by classification transitions."""

    next_recovery_attempt: Callable[..., int]
    transition_terminal: Callable[..., None]
    preserve_label: Callable[..., Any]
    preserve_manual_eval: Callable[..., Any]
    label_publication_commit: Callable[..., str]
    publish_label: Callable[..., str]
    mark_remote_status: Callable[..., None]


def apply_classification(
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
    services: ClassificationServices,
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
        services.transition_terminal(
            store,
            expected=(RunPhase.RUNNING, RunPhase.QUEUED, RunPhase.FAILED),
            target=RunPhase.FAILED,
            facts={"failed_job_id": job_id, "failure_reason": reason_token},
        )
        services.mark_remote_status(ssh, layout, "failed")
        raise RuntimeError(
            f"recorded allocation {job_id} failed deterministically "
            f"[reason={reason_token}]; not resubmitting automatically"
        )
    if classification is ExitClass.COMPLETE:
        if is_label:
            if label_plan is not None and label_plan.lane is LabelLane.SMOKE:
                smoke_path = services.preserve_label(
                    ssh,
                    layout,
                    label_plan.output_dir,
                )
                manual_eval_path = services.preserve_manual_eval(
                    ssh,
                    layout,
                    label_plan.work_dir,
                    lane=LabelLane.SMOKE.value,
                )
                facts: dict[str, object] = {
                    "smoke_job_id": job_id,
                    "smoke_completed": True,
                    "smoke_artifact_path": str(smoke_path),
                    "smoke_manual_eval_path": str(manual_eval_path),
                }
                if is_recovery_from_failed:
                    facts["recovered_from_job_id"] = job_id
                    facts["recovery_reason"] = (
                        "previously-failed smoke allocation re-inspected as complete"
                    )
                    facts["recovery_attempt"] = services.next_recovery_attempt(
                        current.facts
                    )
                services.transition_terminal(
                    store,
                    expected=(RunPhase.RUNNING, RunPhase.QUEUED, RunPhase.FAILED),
                    target=RunPhase.VALIDATED,
                    facts=facts,
                )
                store.transition(
                    expected=RunPhase.VALIDATED,
                    target=RunPhase.REMOTE_PREPARED,
                    facts={
                        "active_stage": Stage.LABEL.value,
                        "label_lane": LabelLane.PRODUCTION.value,
                    },
                )
                print(
                    "V2 smoke complete and preserved; production labeling is ready.",
                    flush=True,
                )
                return
            publishes = (
                label_plan.publishes
                if label_plan is not None
                else config.requirements.row_limit == 0
            )
            output_dir = (
                label_plan.output_dir if label_plan is not None else layout.label_output
            )
            manual_eval_path = (
                services.preserve_manual_eval(
                    ssh,
                    layout,
                    label_plan.work_dir,
                    lane=label_plan.lane.value,
                )
                if label_plan is not None
                else None
            )
            hub_commit: str | None = None
            if publishes:
                try:
                    hub_commit = services.label_publication_commit(ssh, layout, job_id)
                except RuntimeError as exc:
                    if str(exc) != (
                        "label publication did not report an immutable Hub commit"
                    ):
                        raise
                    hub_commit = services.publish_label(
                        ssh,
                        layout,
                        output_dir,
                        config.output_dataset_id,
                        v2=label_plan is not None,
                    )
            facts: dict[str, object] = {"label_job_id": job_id}
            if manual_eval_path is not None:
                facts["manual_eval_path"] = str(manual_eval_path)
            if hub_commit is not None:
                facts["hub_commit"] = hub_commit
            if is_recovery_from_failed:
                facts["recovered_from_job_id"] = job_id
                facts["recovery_reason"] = (
                    "previously-failed allocation re-inspected as complete"
                )
                facts["recovery_attempt"] = services.next_recovery_attempt(
                    current.facts
                )
            services.transition_terminal(
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
                facts={"published": publishes},
            )
            print(f"Labeling complete: run {config.run_id}", flush=True)
            services.mark_remote_status(ssh, layout, "complete")
            return
        if is_recovery_from_failed:
            services.transition_terminal(
                store,
                expected=(RunPhase.FAILED,),
                target=RunPhase.CHECKPOINTED,
                facts={
                    "split_job_id": job_id,
                    "recovered_from_job_id": job_id,
                    "recovery_reason": (
                        "previously-failed split allocation re-inspected as complete"
                    ),
                    "recovery_attempt": services.next_recovery_attempt(current.facts),
                },
            )
        else:
            services.transition_terminal(
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
            facts["recovery_attempt"] = services.next_recovery_attempt(current.facts)
        services.transition_terminal(
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


__all__ = ["ClassificationServices", "apply_classification"]
