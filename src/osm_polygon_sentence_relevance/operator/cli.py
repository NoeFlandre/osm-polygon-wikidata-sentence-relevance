"""Public Mac-side command for autonomous Grid'5000 dataset production."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

from osm_polygon_sentence_relevance.operator import recorded_job, relay
from osm_polygon_sentence_relevance.operator.config import (
    DATA_ROOT,
    INPUT_DATASET_ID,
    OUTPUT_DATASET_ID,
    OperatorConfig,
    Scope,
    Stage,
)
from osm_polygon_sentence_relevance.operator.controller import (
    Controller,
    LiveProgress,
)
from osm_polygon_sentence_relevance.operator.earliest_start import (
    ReplacementCandidate,
    attempt_immediate_replacement,
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
from osm_polygon_sentence_relevance.operator.remote_completion import (
    assert_remote_exit_zero,
    label_publication_commit,
    mark_remote_status,
    publish_split,
    remote_exit_code,
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
from osm_polygon_sentence_relevance.operator.ssh import SshClient
from osm_polygon_sentence_relevance.operator.staging import Stager
from osm_polygon_sentence_relevance.operator.state import RunPhase, RunState, StateStore
from osm_polygon_sentence_relevance.operator.storage import (
    LABEL_STAGING_HEADROOM_BYTES,
    cleanup_can_restore_compatibility,
    cleanup_managed_runs,
    ensure_home_headroom,
    required_staging_headroom,
)
from osm_polygon_sentence_relevance.operator.workflows import (
    RemoteLayout,
    label_submission,
    split_finalization_submission,
)

_SUBMISSION_HEADROOM_BYTES: Final[int] = 512 * 1024**2


def _milestone(message: str) -> None:
    """Print one concise operator milestone immediately."""

    print(f"[operator] {message}", flush=True)


#: Per-invocation active run ID. Set right after a validated
#: ``store.load_or_create`` (or its persisted equivalent) and cleared in
#: the enclosing ``finally``. Never scanned across unrelated runs.
_ACTIVE_RUN_ID: str | None = None


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    value = _result_text(result).strip()
    if len(value) != 40:
        raise RuntimeError("current source checkout has no immutable commit")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    if dirty.stdout:
        raise RuntimeError("current source checkout must be clean")
    return value


def _resolve_input_revision(explicit: str | None, stage: str) -> str:
    if explicit:
        return explicit
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "the hub extra is required to resolve the input revision"
        ) from exc
    dataset_id = OUTPUT_DATASET_ID if stage == Stage.LABEL.value else INPUT_DATASET_ID
    sha = HfApi().dataset_info(dataset_id, revision="main").sha
    if not sha:
        raise RuntimeError("input dataset main did not resolve to a commit")
    return sha


def _remote_home(ssh: SshClient) -> PurePosixPath:
    result = ssh.run('printf "%s\\n" "$HOME"')
    value = _result_text(result).strip()
    if not value.startswith("/") or "\n" in value or ".." in value.split("/"):
        raise RuntimeError("remote home path is invalid")
    return PurePosixPath(value)


def _result_text(result: Any) -> str:
    """Return ``result.text`` if available, else ``result.stdout``."""

    text_attr = getattr(result, "text", None)
    if text_attr is not None:
        return text_attr
    return getattr(result, "stdout", "")


def _usage_policy_preflight(ssh: SshClient, site: str) -> None:
    """Fail closed unless Grid'5000's live policy checks succeed."""

    if re.fullmatch(r"[a-z][a-z0-9-]*", site) is None:
        raise ValueError("Grid'5000 site name is invalid")
    quoted_site = shlex.quote(site)
    ssh.run(
        "command -v usagepolicycheck >/dev/null && "
        f"usagepolicycheck -l --sites {quoted_site} >/dev/null && "
        "usagepolicycheck -t >/dev/null"
    )


def _emit(progress: LiveProgress) -> None:
    for line in progress.text.splitlines():
        print(f"[job {progress.job_id}] {line}", flush=True)


def _transition_terminal(
    state: StateStore,
    *,
    expected: tuple[RunPhase, ...],
    target: RunPhase,
    facts: dict[str, object],
) -> None:
    current = state.load()
    if current.phase not in expected:
        raise RuntimeError("operator reached an unexpected durable phase")
    state.transition(expected=current.phase, target=target, facts=facts)


def _reattach_decision(
    state: RunState,
) -> tuple[str, int] | None:
    """Return the stored (site, job_id) if a recorded allocation is tracked.

    Conservative: it only returns a candidate. The reattach path then queries
    OAR read-only and dispatches to live monitoring or terminal classification
    without probing other sites or submitting a competing job.
    """

    if state.phase not in {
        RunPhase.SUBMITTED,
        RunPhase.QUEUED,
        RunPhase.RUNNING,
    }:
        return None
    job_id = state.facts.get("job_id")
    site = state.facts.get("site")
    if type(job_id) is not int or job_id <= 0:
        return None
    if not isinstance(site, str) or not site:
        return None
    return site, job_id


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
) -> tuple[SshClient, RemoteLayout, OarClient, Controller]:
    """Open one SSH connection to a recorded site and build the controller."""

    ssh = SshClient(target=site, command_timeout=1800)
    layout = RemoteLayout(_remote_home(ssh) / "osm-polygon-operator" / config.run_id)
    oar = OarClient(ssh)
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
) -> None:
    """Drive durable state transitions for a classified terminal allocation."""

    is_label = active_stage == Stage.LABEL.value
    if classification is ExitClass.FAILED:
        _transition_terminal(
            store,
            expected=(RunPhase.RUNNING, RunPhase.QUEUED),
            target=RunPhase.FAILED,
            facts={"failed_job_id": job_id},
        )
        mark_remote_status(ssh, layout, "failed")
        raise RuntimeError(
            f"recorded allocation {job_id} failed deterministically; not "
            "resubmitting automatically"
        )
    if classification is ExitClass.COMPLETE:
        if is_label:
            hub_commit: str | None = None
            if config.requirements.row_limit == 0:
                hub_commit = label_publication_commit(ssh, layout, job_id)
            facts: dict[str, object] = {"label_job_id": job_id}
            if hub_commit is not None:
                facts["hub_commit"] = hub_commit
            _transition_terminal(
                store,
                expected=(RunPhase.RUNNING, RunPhase.QUEUED),
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
        _transition_terminal(
            store,
            expected=(RunPhase.RUNNING, RunPhase.QUEUED),
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


def _classify_or_continue(
    args: argparse.Namespace,
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
        if terminal is not JobState.TERMINATED:
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

    inspection = recorded_job.inspect_remote_resume(
        ssh,
        label_work_root=str(layout.label_work),
        label_output_root=str(layout.label_output),
        expected_identity=config.run_identity.to_dict(),
        exit_file=str(layout.logs / str(job_id) / "labeling.exit_code"),
    )
    classification = recorded_job.classify_terminal(status, inspection)

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
    )

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
        new_job_id = controller_d.submit(
            component=Stage.LABEL,
            input_parquet=layout_d.root / "input/sentences.parquet",
            model_file=layout_d.root / "model" / config.label_model_file,
            tokenizer_dir=layout_d.root / "tokenizer",
        )
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
        terminal = controller_d.monitor(new_job_id, log_name="labeling.stdout.log")
        if terminal is not JobState.TERMINATED:
            raise RuntimeError("continuation allocation failed")
        status_d = oar_d.status(new_job_id)
        inspection_d = recorded_job.inspect_remote_resume(
            ssh_d,
            label_work_root=str(layout_d.label_work),
            label_output_root=str(layout_d.label_output),
            expected_identity=config.run_identity.to_dict(),
            exit_file=str(layout_d.logs / str(new_job_id) / "labeling.exit_code"),
        )
        classification_d = recorded_job.classify_terminal(status_d, inspection_d)
        _apply_classification(
            store=store,
            config=config,
            ssh=ssh_d,
            layout=layout_d,
            job_id=new_job_id,
            active_stage=active,
            classification=classification_d,
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
            f"continuation allocation {new_job_id} failed deterministically; "
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
        expected_run_identity=config.run_identity.to_dict(),
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
    args: argparse.Namespace,
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

    if existing_trial is None and not should_seek_replacement(
        fallback_status,
        now=now,
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
        elif not probe.label_runtime_ready:
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
        label_assets = stager.prepare_label_assets(config, layout, download_input=True)
        if not label_assets.llama_server_ready:
            raise RuntimeError("CUDA llama-server is not staged")
        assets[site] = label_assets

    def submit(candidate: ReplacementCandidate) -> int:
        site = candidate.site.name
        current_fallback = client(fallback_site)[2].status(fallback_job_id)
        if current_fallback.state is not JobState.QUEUED:
            raise RuntimeError("fallback is no longer queued")
        _ssh, layout, oar = client(site)
        label_assets = assets[site]
        return oar.submit(
            label_submission(
                config,
                layout,
                input_parquet=label_assets.input_parquet,
                model_file=label_assets.model_file,
                tokenizer_dir=label_assets.tokenizer_dir,
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
        existing_trial=existing_trial,
    )
    return outcome.site, outcome.job_id


def _resume_run(run_id: str, args: argparse.Namespace) -> int:
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
    config = OperatorConfig.from_persisted(payload["run_identity"])
    if config.run_id != run_id:
        raise RuntimeError(
            "persisted run identity does not reproduce the requested run ID"
        )
    store = StateStore(DATA_ROOT)
    store.load_or_create(config.run_identity)
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
        assets = stager.prepare_label_assets(config, layout, download_input=True)
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


def _run(args: argparse.Namespace) -> int:
    if not DATA_ROOT.exists():
        raise RuntimeError(f"external data root is unavailable: {DATA_ROOT}")
    _milestone("Validating the local source checkout")
    source_commit = _git_head()
    _milestone(f"Source commit: {source_commit[:12]}")
    _milestone("Resolving immutable input revision")
    input_revision = _resolve_input_revision(args.input_revision, args.stage)
    _milestone(f"Input revision: {input_revision[:12]}")
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
    )
    store = StateStore(DATA_ROOT)
    store.load_or_create(config.run_identity)
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
    probes: list[SiteProbe] = []
    for target in dict.fromkeys(args.site):
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
    try:
        selection = select_site(probes, requirements)
    except NoCompatibleSiteError:
        if not cleanup_can_restore_compatibility(probes, requirements):
            raise
        _milestone(
            "No compatible site; reclaiming only completed or failed managed runs"
        )
        for probe in probes:
            if probe.reachable:
                cleanup_managed_runs(SshClient(target=probe.target), execute=True)
        probes = []
        for target in dict.fromkeys(args.site):
            _milestone(f"Re-probing Grid'5000 site: {target}")
            probes.append(probe_site(target, config.run_id, requirements))
        selection = select_site(probes, requirements)
    target = selection.selected.target
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
        _usage_policy_preflight(ssh, selection.selected.name)
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

    durable = store.load()
    split_done = "split_output_job_id" in durable.facts
    if config.stage in {Stage.SPLIT, Stage.ALL} and not split_done:
        for allocation in range(1, 101):
            job_id = controller.submit(component=Stage.SPLIT)
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
            args.poll_seconds,
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
                ssh, layout, output_dir, config.output_dataset_id
            )
            store.transition(
                expected=RunPhase.VALIDATED,
                target=RunPhase.COMPLETE,
                facts={"published": True, "hub_commit": hub_commit},
            )
            print(f"Sentence splitting complete: run {config.run_id}", flush=True)
            mark_remote_status(ssh, layout, "complete")
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
        _milestone("Staging immutable labeling assets")
        assets = stager.prepare_label_assets(
            config,
            layout,
            download_input=config.stage is Stage.LABEL,
        )
        _milestone("Input, model, and tokenizer assets are ready")
        if not assets.llama_server_ready:
            ensure_llama_server(ssh, oar, store, layout, args.poll_seconds)
        if config.stage is Stage.ALL:
            assets = type(assets)(
                input_parquet,
                assets.model_file,
                assets.tokenizer_dir,
                True,
            )
            if store.load().phase is RunPhase.VALIDATED:
                store.transition(
                    expected=RunPhase.VALIDATED,
                    target=RunPhase.REMOTE_PREPARED,
                    facts={"label_assets_ready": True},
                )
        for allocation in range(1, 101):
            job_id = controller.submit(
                component=Stage.LABEL,
                input_parquet=assets.input_parquet,
                model_file=assets.model_file,
                tokenizer_dir=assets.tokenizer_dir,
            )
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


def _status(args: argparse.Namespace) -> int:
    path = DATA_ROOT / "runs" / args.run_id / "state.json"
    if not path.is_file():
        raise RuntimeError("run state does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _resume_handler(args: argparse.Namespace) -> int:
    return _resume_run(args.run_id, args)


def _cleanup(args: argparse.Namespace) -> int:
    removed = cleanup_managed_runs(
        SshClient(target=args.site),
        execute=args.execute,
    )
    label = "removed" if args.execute else "eligible"
    for path in removed:
        print(f"{label}: {path}")
    if not removed:
        print("No pipeline-managed completed or failed runs are eligible.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the stable public CLI parser."""

    parser = argparse.ArgumentParser(
        prog="osm-polygon-grid5000",
        description="Run and resume sentence processing on Grid'5000.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run or resume a production workflow")
    run.add_argument("--scope", choices=[scope.value for scope in Scope], required=True)
    run.add_argument("--region")
    run.add_argument("--stage", choices=[stage.value for stage in Stage], required=True)
    run.add_argument("--input-revision")
    run.add_argument("--site", action="append", default=list(DEFAULT_SITES))
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--row-limit", type=int, default=0)
    run.add_argument("--llama-parallel", type=int, default=8)
    run.add_argument("--llama-per-slot-context", type=int, default=8192)
    run.add_argument("--request-concurrency", type=int)
    run.add_argument("--gpu-memory-mb", type=int, default=40_000)
    run.add_argument("--remote-free-bytes", type=int, default=8 * 1024**3)
    run.add_argument("--poll-seconds", type=float, default=30.0)
    run.set_defaults(handler=_run)

    status = sub.add_parser("status", help="show durable local run state")
    status.add_argument("run_id")
    status.set_defaults(handler=_status)

    resume = sub.add_parser(
        "resume",
        help="resume or classify a historical run by its durable run ID",
    )
    resume.add_argument("run_id")
    resume.add_argument("--site", action="append", default=list(DEFAULT_SITES))
    resume.add_argument("--gpu-memory-mb", type=int, default=40_000)
    resume.add_argument("--poll-seconds", type=float, default=30.0)
    resume.set_defaults(handler=_resume_handler)

    cleanup = sub.add_parser(
        "cleanup",
        help="preview or remove completed pipeline-managed remote runs",
    )
    cleanup.add_argument("--site", required=True)
    cleanup.add_argument("--execute", action="store_true")
    cleanup.set_defaults(handler=_cleanup)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Installed entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    global _ACTIVE_RUN_ID
    prior_active = _ACTIVE_RUN_ID
    _ACTIVE_RUN_ID = None
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        run_id = _ACTIVE_RUN_ID if _ACTIVE_RUN_ID else None
        if run_id is None:
            run_id = prior_active if prior_active else None
        if run_id is None:
            print(
                "Local monitoring stopped; the remote job and checkpoints were "
                "preserved.",
                file=sys.stderr,
            )
        else:
            print(
                "Local monitoring stopped; the remote job and checkpoints were "
                "preserved.",
                file=sys.stderr,
            )
            print(
                f"Resume with: {_resume_command(run_id)}",
                file=sys.stderr,
            )
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        _ACTIVE_RUN_ID = prior_active


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
