"""Public Mac-side command for autonomous Grid'5000 dataset production."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import PurePosixPath
from typing import Final

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
from osm_polygon_sentence_relevance.operator.oar import JobState, OarClient
from osm_polygon_sentence_relevance.operator.sites import (
    NoCompatibleSiteError,
    SiteProbe,
    SiteRequirements,
    select_site,
)
from osm_polygon_sentence_relevance.operator.ssh import SshClient, SshError
from osm_polygon_sentence_relevance.operator.staging import Stager
from osm_polygon_sentence_relevance.operator.state import RunPhase, StateStore
from osm_polygon_sentence_relevance.operator.workflows import (
    RemoteLayout,
    llama_build_submission,
    split_finalization_submission,
)

DEFAULT_TARGETS: Final[tuple[str, ...]] = ("nancy", "nantes", "rennes")


def _milestone(message: str) -> None:
    """Print one concise operator milestone immediately."""

    print(f"[operator] {message}", flush=True)


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    value = result.stdout.strip()
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


def _probe_target(target: str) -> SiteProbe:
    ssh = SshClient(target=target, attempts=1, command_timeout=30)
    # ``oarnodes -J`` is the authoritative OAR resource inventory. Restrict
    # the maximum to currently Alive GPU resources; ``oarstat -p`` describes
    # jobs and does not expose the node properties needed here.
    command = r"""
set -euo pipefail
command -v oarsub >/dev/null
command -v oarnodes >/dev/null
command -v jq >/dev/null
free_kb=$(df -Pk "$HOME" | awk 'NR==2 {print $4}')
inventory=$(oarnodes -J | jq -r '
  (([.[] | select(.gpu_count > 0 and .state == "Alive") | .gpu_mem]
    | max // 0 | tostring)
   + " " +
   ([.[] | select(.gpu_count > 0 and .state == "Alive")
           | .gpu_compute_capability_major]
    | max // 0 | tostring))
')
read -r gpu_mb gpu_major <<<"$inventory"
waiting=$(oarstat -u 2>/dev/null | awk '$5 ~ /Waiting|Hold/ {n++} END {print n+0}')
printf '%s %s %s %s\n' "$free_kb" "$gpu_mb" "$gpu_major" "$waiting"
""".strip()
    try:
        result = ssh.run(command)
        free_kb_raw, gpu_raw, gpu_major_raw, waiting_raw = result.stdout.splitlines()[
            -1
        ].split()
        gpu_memory = int(gpu_raw)
        gpu_major = int(gpu_major_raw)
        return SiteProbe(
            name=target.split("@")[-1].split(".")[0],
            target=target,
            reachable=True,
            gpu_memory_mb=gpu_memory,
            cuda_capability=(gpu_major, 0) if gpu_memory > 0 else None,
            persistent_free_bytes=int(free_kb_raw) * 1024,
            expected_start_seconds=int(waiting_raw) * 60,
        )
    except (SshError, ValueError, IndexError):
        return SiteProbe(
            target,
            target,
            False,
            0,
            None,
            0,
            0,
        )


def _storage_cleanup_can_help(
    probes: list[SiteProbe],
    requirements: SiteRequirements,
) -> bool:
    """Return whether storage is the only failed hard constraint anywhere."""

    return any(
        probe.reachable
        and probe.gpu_memory_mb >= requirements.gpu_memory_mb
        and probe.cuda_capability is not None
        and probe.cuda_capability >= requirements.cuda_capability
        and probe.expected_start_seconds >= 0
        and probe.persistent_free_bytes < requirements.persistent_free_bytes
        for probe in probes
    )


def _remote_home(ssh: SshClient) -> PurePosixPath:
    value = ssh.run('printf "%s\\n" "$HOME"').stdout.strip()
    if not value.startswith("/") or "\n" in value or ".." in value.split("/"):
        raise RuntimeError("remote home path is invalid")
    return PurePosixPath(value)


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
    _milestone(f"Durable run ID: {config.run_id}")

    requirements = SiteRequirements(
        gpu_memory_mb=args.gpu_memory_mb,
        persistent_free_bytes=args.remote_free_bytes,
    )
    probes: list[SiteProbe] = []
    for target in dict.fromkeys(args.site):
        _milestone(f"Probing Grid'5000 site: {target}")
        probe = _probe_target(target)
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
        if not _storage_cleanup_can_help(probes, requirements):
            raise
        _milestone(
            "No compatible site; reclaiming only completed or failed managed runs"
        )
        for probe in probes:
            if probe.reachable:
                _cleanup_remote(SshClient(target=probe.target), execute=True)
        probes = []
        for target in dict.fromkeys(args.site):
            _milestone(f"Re-probing Grid'5000 site: {target}")
            probes.append(_probe_target(target))
        selection = select_site(probes, requirements)
    target = selection.selected.target
    _milestone(f"Selected Grid'5000 site: {selection.selected.name}")
    ssh = SshClient(target=target, command_timeout=1800)
    home = _remote_home(ssh)
    layout = RemoteLayout(home / "osm-polygon-operator" / config.run_id)
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
            split_exit_code = _remote_exit_code(ssh, layout, job_id, "build.exit_code")
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
        _monitor_simple(
            ssh,
            oar,
            layout,
            final_job,
            "finalize.stdout.log",
            args.poll_seconds,
        )
        _assert_remote_exit_zero(ssh, layout, final_job, "finalize.exit_code")
        store.transition(
            expected=RunPhase.FINALIZING,
            target=RunPhase.VALIDATED,
            facts={"split_output_job_id": final_job},
        )
        if config.stage is Stage.SPLIT:
            output_dir = layout.logs / str(final_job) / "output"
            hub_commit = _publish_split(
                ssh, layout, output_dir, config.output_dataset_id
            )
            store.transition(
                expected=RunPhase.VALIDATED,
                target=RunPhase.COMPLETE,
                facts={"published": True, "hub_commit": hub_commit},
            )
            print(f"Sentence splitting complete: run {config.run_id}", flush=True)
            _mark_remote_status(ssh, layout, "complete")
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
            _milestone("CUDA llama-server binary is absent; submitting its build")
            build_job = oar.submit(llama_build_submission(layout))
            print(f"Submitted CUDA llama-server build job {build_job}", flush=True)
            _monitor_without_log(oar, build_job, args.poll_seconds)
            ready = ssh.run(
                "if test -x "
                f"{layout.root!s}/llama-server-bin/llama-server; "
                "then printf yes; else printf no; fi"
            ).stdout
            if ready != "yes":
                raise RuntimeError("CUDA llama-server build did not produce a binary")
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
            _assert_remote_exit_zero(ssh, layout, job_id, "labeling.exit_code")
            complete = (
                ssh.run(
                    "if test -f "
                    f"{layout.label_output!s}/manifest.json; "
                    "then printf yes; else printf no; fi"
                ).stdout
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
            label_hub_commit = _label_publication_commit(ssh, layout, job_id)
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
        _mark_remote_status(ssh, layout, "complete")
    return 0


def _monitor_simple(
    ssh: SshClient,
    oar: OarClient,
    layout: RemoteLayout,
    job_id: int,
    log_name: str,
    poll_seconds: float,
) -> None:
    offset = 0
    while True:
        status = oar.status(job_id)
        chunk = ssh.read_since(str(layout.logs / str(job_id) / log_name), offset)
        if chunk.reset:
            offset = 0
        elif chunk.text:
            offset = chunk.next_offset
            _emit(LiveProgress(job_id, log_name, chunk.text, offset))
        if status.state is JobState.TERMINATED:
            return
        if status.state in {JobState.ERROR, JobState.MISSING}:
            raise RuntimeError("remote allocation failed")
        time.sleep(poll_seconds)


def _monitor_without_log(
    oar: OarClient,
    job_id: int,
    poll_seconds: float,
) -> None:
    while True:
        status = oar.status(job_id)
        if status.state is JobState.TERMINATED:
            if status.exit_code not in {None, 0}:
                raise RuntimeError("remote build allocation failed")
            return
        if status.state in {JobState.ERROR, JobState.MISSING}:
            raise RuntimeError("remote build allocation failed")
        time.sleep(poll_seconds)


def _remote_exit_code(
    ssh: SshClient,
    layout: RemoteLayout,
    job_id: int,
    filename: str,
) -> int:
    path = layout.logs / str(job_id) / filename
    result = ssh.run(f"test -f {path!s} && cat {path!s}")
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("remote payload exit status is invalid") from exc


def _assert_remote_exit_zero(
    ssh: SshClient,
    layout: RemoteLayout,
    job_id: int,
    filename: str,
) -> None:
    if _remote_exit_code(ssh, layout, job_id, filename) != 0:
        raise RuntimeError("remote payload returned non-zero status")


def _publish_split(
    ssh: SshClient,
    layout: RemoteLayout,
    output_dir: PurePosixPath,
    dataset_id: str,
) -> str:
    code = (
        "from osm_polygon_sentence_relevance.publishing import "
        "publish_export_directory; "
        f"r=publish_export_directory({str(output_dir)!r},{dataset_id!r},"
        "target_revision='main'); print(r.commit_id)"
    )
    command = " ".join(
        shlex.quote(value)
        for value in (str(layout.repo / ".venv/bin/python"), "-c", code)
    )
    result = ssh.run(command)
    commit_id = result.stdout.strip()
    if len(commit_id) < 7:
        raise RuntimeError("Hugging Face publication did not return a commit")
    return commit_id


def _label_publication_commit(
    ssh: SshClient,
    layout: RemoteLayout,
    job_id: int,
) -> str:
    """Read the verified label publisher's immutable Hub commit from its log."""

    path = layout.logs / str(job_id) / "labeling.stdout.log"
    text = ssh.run(f"test -f {path!s} && cat {path!s}").stdout
    for line in reversed(text.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        commit_id = payload.get("commit_id") if isinstance(payload, dict) else None
        if (
            isinstance(commit_id, str)
            and len(commit_id) == 40
            and all(character in "0123456789abcdef" for character in commit_id)
        ):
            return commit_id
    raise RuntimeError("label publication did not report an immutable Hub commit")


def _mark_remote_status(ssh: SshClient, layout: RemoteLayout, status: str) -> None:
    if status not in {"active", "complete", "failed"}:
        raise ValueError("invalid managed status")
    marker = layout.root / ".operator-managed.json"
    ssh.run(
        "printf '%s\\n' "
        + shlex.quote(
            json.dumps(
                {"schema_version": 1, "status": status},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        + f" > {shlex.quote(str(marker))} && chmod 0600 {shlex.quote(str(marker))}"
    )


def _cleanup_remote(ssh: SshClient, *, execute: bool) -> tuple[str, ...]:
    action = "delete" if execute else "preview"
    script = f"""
set -euo pipefail
root="$HOME/osm-polygon-operator"
[ -d "$root" ] || exit 0
find "$root" -mindepth 1 -maxdepth 1 -type d -print0 |
while IFS= read -r -d '' candidate; do
  [ ! -L "$candidate" ] || continue
  marker="$candidate/.operator-managed.json"
  [ -f "$marker" ] && [ ! -L "$marker" ] || continue
  status=$(sed -n 's/.*"status":"\\([^"]*\\)".*/\\1/p' "$marker")
  case "$status" in complete|failed) ;; *) continue ;; esac
  printf '%s\\n' "$candidate"
  if [ {shlex.quote(action)} = delete ]; then rm -rf -- "$candidate"; fi
done
""".strip()
    result = ssh.run(script)
    return tuple(line for line in result.stdout.splitlines() if line)


def _status(args: argparse.Namespace) -> int:
    path = DATA_ROOT / "runs" / args.run_id / "state.json"
    if not path.is_file():
        raise RuntimeError("run state does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cleanup(args: argparse.Namespace) -> int:
    removed = _cleanup_remote(
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
    run.add_argument("--site", action="append", default=list(DEFAULT_TARGETS))
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
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print(
            "Local monitoring stopped; the remote job and checkpoints were preserved.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
