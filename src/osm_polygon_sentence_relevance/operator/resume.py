"""Durable resume orchestration for Grid'5000 runs.

The Typer adapter owns command-line concerns.  This module owns the
run-identity reconstruction and continuation decision tree so it can be
tested without importing the CLI application.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

from osm_polygon_sentence_relevance.labeling.v2_contracts import (
    V2_LOGIT_PROMPT_VERSION,
)
from osm_polygon_sentence_relevance.operator.config import Stage
from osm_polygon_sentence_relevance.operator.label_lanes import LabelLanePlan
from osm_polygon_sentence_relevance.operator.label_submission import (
    submit_preferred_label,
)
from osm_polygon_sentence_relevance.operator.oar import (
    ExitClass,
    JobState,
    is_live_state,
)
from osm_polygon_sentence_relevance.operator.state import RunPhase
from osm_polygon_sentence_relevance.operator.storage import (
    LABEL_STAGING_HEADROOM_BYTES,
)
from osm_polygon_sentence_relevance.operator.workflows import (
    RemoteLayout,
)


@dataclass(frozen=True, slots=True)
class ResumeServices:
    """CLI seams required by the durable resume workflow.

    Keeping the side-effecting seams explicit prevents this module from
    importing the Typer adapter and makes each dependency replaceable in unit
    tests.  The concrete service types are intentionally left to the CLI
    boundary because several tests provide small fakes.
    """

    data_root: Path
    state_store_type: Any
    config_type: Any
    git_head: Callable[[], str]
    sync_sampling_target: Callable[[Any, Any], None]
    reattach_decision: Callable[[Any], tuple[str, int] | None]
    attach_to_site: Callable[..., tuple[Any, RemoteLayout, Any, Any]]
    usage_policy_preflight: Callable[..., None]
    ensure_home_headroom: Callable[..., None]
    stager_type: Any
    stage_hf_token: Callable[..., None]
    recorded_split_resume_bundle: Callable[[Any], PurePosixPath | None]
    current_label_plan: Callable[..., LabelLanePlan | None]
    ensure_llama_server: Callable[..., int]
    ensure_relay_at_destination: Callable[..., None]
    race_queued_start: Callable[..., tuple[str, int]]
    classify_or_continue: Callable[..., ExitClass]
    resume_command: Callable[[str], str]
    milestone: Callable[[str], None]
    monitor_until_terminal: Callable[..., JobState]
    publish_split: Callable[..., str]
    mark_remote_status: Callable[..., None]
    split_finalization: Any
    assert_remote_exit_zero: Callable[..., None]
    submission_headroom_bytes: int


def resume_split_finalization(
    *,
    run_id: str,
    args: SimpleNamespace,
    store: Any,
    config: Any,
    services: ResumeServices,
) -> int:
    """Recover split finalization without rerunning completed split shards."""

    durable = store.load()
    site = durable.facts.get("site")
    if not isinstance(site, str) or not site:
        raise RuntimeError("split finalization recovery has no recorded site")
    ssh, layout, oar, controller = services.attach_to_site(
        store,
        config,
        site,
        poll_seconds=args.poll_seconds,
    )

    if durable.phase is RunPhase.FINALIZING:
        final_job = durable.facts.get("finalization_job_id")
        if type(final_job) is not int or final_job <= 0:
            raise RuntimeError("split finalization has no recorded job")
        status = oar.status(final_job)
        if is_live_state(status.state):
            services.monitor_until_terminal(
                controller,
                final_job,
                log_name="finalize.stdout.log",
            )
            status = oar.status(final_job)
        if status.state not in {
            JobState.TERMINATED,
            JobState.ERROR,
            JobState.MISSING,
        }:
            raise RuntimeError(
                f"split finalization job {final_job} ended in {status.state.value}"
            )
        try:
            services.assert_remote_exit_zero(
                ssh,
                layout,
                final_job,
                "finalize.exit_code",
            )
        except RuntimeError:
            store.transition(
                expected=RunPhase.FINALIZING,
                target=RunPhase.CHECKPOINTED,
                facts={
                    "recovered_finalization_job_id": final_job,
                    "finalization_recovery_reason": "terminal allocation failed",
                },
            )
        else:
            store.transition(
                expected=RunPhase.FINALIZING,
                target=RunPhase.VALIDATED,
                facts={"split_output_job_id": final_job},
            )
            if config.stage is Stage.SPLIT:
                output_dir = layout.logs / str(final_job) / "output"
                hub_commit = services.publish_split(
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
                services.mark_remote_status(ssh, layout, "complete")
                return 0
            store.transition(
                expected=RunPhase.VALIDATED,
                target=RunPhase.REMOTE_PREPARED,
                facts={"active_stage": Stage.LABEL.value},
            )
            return resume_run(run_id, args, services)

    if store.load().phase is not RunPhase.CHECKPOINTED:
        raise RuntimeError("split finalization recovery reached an invalid phase")
    # A new finalization submission is the first point at which policy,
    # quota, checkout, and token staging are required.  Keeping these checks
    # out of the reattach path means a live finalizer is never blocked by a
    # slow quota service or an unrelated preflight failure.
    services.usage_policy_preflight(ssh, site)
    services.ensure_home_headroom(
        ssh,
        protected_root=layout.root,
        minimum_headroom_bytes=services.submission_headroom_bytes,
    )
    stager = services.stager_type(ssh)
    stager.prepare(config, layout)
    services.stage_hf_token(stager, layout)
    services.milestone(
        "Re-submitting failed split finalization from complete checkpoints"
    )
    services.split_finalization.finalize_split_checkpointed(
        store=store,
        config=config,
        ssh=ssh,
        layout=layout,
        oar=oar,
        poll_seconds=args.poll_seconds,
    )
    if config.stage is Stage.ALL:
        return resume_run(run_id, args, services)
    return 0


def resume_run(run_id: str, args: SimpleNamespace, services: ResumeServices) -> int:
    """Resume or classify a historical run by its durable run ID.

    The persisted run identity remains authoritative.  A resumed split/all
    workflow may use the current execution commit while preserving the source
    commit that identifies already-produced checkpoints.
    """

    if not re.fullmatch(r"[0-9a-f]{20}", run_id):
        raise RuntimeError("run ID must be twenty lowercase hexadecimal characters")
    state_path = services.data_root / "runs" / run_id / "state.json"
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
    config = services.config_type.from_persisted(persisted_identity)
    if config.run_id != run_id:
        raise RuntimeError(
            "persisted run identity does not reproduce the requested run ID"
        )
    if config.stage in {Stage.SPLIT, Stage.ALL}:
        current_execution_commit = services.git_head()
        if current_execution_commit != config.source_commit:
            config = replace(config, execution_commit=current_execution_commit)
    store = services.state_store_type(services.data_root)
    store.load_or_create(config.run_identity)
    services.sync_sampling_target(store, config)
    durable = store.load()
    candidate = services.reattach_decision(durable)
    if durable.phase in {RunPhase.CHECKPOINTED, RunPhase.FINALIZING}:
        return resume_split_finalization(
            run_id=run_id,
            args=args,
            store=store,
            config=config,
            services=services,
        )
    if candidate is None and durable.phase is RunPhase.REMOTE_PREPARED:
        site_value = durable.facts.get("site")
        if not isinstance(site_value, str) or not site_value:
            raise RuntimeError("prepared continuation has no recorded site")
        services.milestone(f"Resuming prepared continuation on site {site_value}")
        ssh, layout, oar, controller = services.attach_to_site(
            store, config, site_value, poll_seconds=args.poll_seconds
        )
        services.usage_policy_preflight(ssh, site_value)
        staged_label_assets = (
            durable.facts.get("label_assets_ready") is True
            or type(durable.facts.get("llama_build_job_id")) is int
        )
        services.ensure_home_headroom(
            ssh,
            protected_root=layout.root,
            minimum_headroom_bytes=(
                services.submission_headroom_bytes
                if staged_label_assets
                else LABEL_STAGING_HEADROOM_BYTES
            ),
        )
        stager = services.stager_type(ssh)
        stager.prepare(config, layout)
        services.stage_hf_token(stager, layout)
        active_stage = durable.facts.get("active_stage")
        if active_stage == Stage.SPLIT.value:
            job_id = controller.submit(
                component=Stage.SPLIT,
                split_resume_bundle=services.recorded_split_resume_bundle(store),
            )
            print(f"Submitted split continuation job {job_id}", flush=True)
            candidate = (site_value, job_id)
        else:
            current_label_plan = services.current_label_plan(store, config, layout)
            if config.prompt_version == V2_LOGIT_PROMPT_VERSION:
                split_job_raw = durable.facts.get("split_output_job_id")
                if type(split_job_raw) is not int:
                    raise RuntimeError("V2 continuation has no split output")
                split_output = (
                    layout.logs / str(split_job_raw) / "output/sentences.parquet"
                )
                v2_input = stager.prepare_v2_input(config, layout, split_output)
                assets = stager.prepare_label_assets(
                    config, layout, download_input=False
                )
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
            current = store.load()
            store.transition(
                expected=current.phase,
                target=current.phase,
                facts={"label_assets_ready": True},
            )
            if not assets.llama_server_ready:
                services.ensure_llama_server(ssh, oar, store, layout, args.poll_seconds)
            relay_root_value = durable.facts.get("resume_relay_root")
            if isinstance(relay_root_value, str):
                services.ensure_relay_at_destination(
                    store=store,
                    config=config,
                    site=site_value,
                    layout=layout,
                    relay_root=Path(relay_root_value),
                )
            job_id = submit_preferred_label(
                controller,
                input_parquet=assets.input_parquet,
                model_file=assets.model_file,
                tokenizer_dir=assets.tokenizer_dir,
                gpu_memory_mb=getattr(args, "gpu_memory_mb", 40_000),
                label_plan=current_label_plan,
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
    site, job_id = services.race_queued_start(
        args,
        store,
        config,
        site,
        job_id,
    )
    services.milestone(f"Resuming run {run_id} on site {site}, job {job_id}")
    classification = services.classify_or_continue(
        args=args,
        store=store,
        config=config,
        site=site,
        job_id=job_id,
        destination_site=site,
    )
    if classification is ExitClass.CONTINUE:
        services.milestone(
            "Validated checkpoints preserved; submit the next allocation via "
            f"{services.resume_command(run_id)} or `run`."
        )
    return 0


__all__ = ["ResumeServices", "resume_run", "resume_split_finalization"]
