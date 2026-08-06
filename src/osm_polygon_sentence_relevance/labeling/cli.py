"""Command line entry point for labeling, finalization, and publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .canary import select_canary_rows
from .checkpoint import CheckpointStore
from .checkpoint_mirror import CheckpointMirror
from .contracts import RunIdentity
from .engine import LabelEngine
from .finalization import finalize_labeled_dataset
from .prompt import PROMPT_VERSION, build_messages
from .publication import publish_labeled_dataset
from .releases import ReleaseLane, release_lane, trackio_space_id
from .repair import BoundedRepair
from .runner import LabelingRunner, StopController
from .runtime import (
    MIN_PER_SLOT_CONTEXT,
    SUPPORTED_LLAMA_PARALLEL,
    RuntimePlan,
    build_runtime_plan,
    resolve_engine_factory,
)
from .sampling import (
    DEFAULT_H3_RESOLUTION,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_TARGET,
    SAMPLING_VERSION,
    select_label_rows,
)
from .tracking import log_static_labeling_run
from .tracking_progress import TrackioBatchLogger
from .validation import parse_label_response

MODEL_REPO_ID = "unsloth/Qwen3.6-27B-MTP-GGUF"
MODEL_FILE = "Qwen3.6-27B-Q4_K_M.gguf"
DEFAULT_LLAMA_PARALLEL = 16


def _add_server_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llama-parallel",
        type=int,
        default=DEFAULT_LLAMA_PARALLEL,
        help=(
            "Number of parallel llama-server slots. Must be one of "
            f"{', '.join(str(value) for value in SUPPORTED_LLAMA_PARALLEL)}."
        ),
    )
    parser.add_argument(
        "--llama-per-slot-context",
        type=int,
        default=MIN_PER_SLOT_CONTEXT,
        help=(
            "Per-slot context size. Must be at least "
            f"{MIN_PER_SLOT_CONTEXT}; total context is parallel * per-slot."
        ),
    )
    parser.add_argument(
        "--llama-total-context",
        type=int,
        default=None,
        help=(
            "Optional total context. When omitted, derived as "
            "parallel * per-slot-context."
        ),
    )
    parser.add_argument(
        "--request-concurrency",
        type=int,
        default=None,
        help=(
            "Client concurrency. Defaults to the parallel slot count and is "
            "capped to it."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label polygon sentences")
    sub = parser.add_subparsers(dest="command", required=True)
    label = sub.add_parser("label", help="Run or resume LLM labeling")
    label.add_argument("--input-parquet", required=True)
    label.add_argument("--work-dir", required=True)
    label.add_argument("--input-dataset-revision", required=True)
    label.add_argument("--model-revision", required=True)
    label.add_argument("--model-file-sha256", required=True)
    label.add_argument("--source-commit", required=True)
    label.add_argument("--engine", required=True, choices=("llama.cpp",))
    label.add_argument("--engine-version", required=True)
    label.add_argument(
        "--endpoint", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    label.add_argument("--batch-size", type=int, default=128)
    label.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_LLAMA_PARALLEL,
        help=(
            "Client concurrency. Defaults to the parallel slot count; "
            "the server identity will pin it."
        ),
    )
    _add_server_config_args(label)
    label.add_argument(
        "--row-limit",
        type=int,
        default=0,
        help="Deterministic canary size; zero uses the stratified sample",
    )
    label.add_argument("--sampling-target", type=int, default=DEFAULT_SAMPLE_TARGET)
    label.add_argument("--sampling-seed", default=DEFAULT_SAMPLE_SEED)
    label.add_argument("--h3-resolution", type=int, default=DEFAULT_H3_RESOLUTION)
    label.add_argument(
        "--checkpoint-dataset-id",
        default=None,
        help="Optional dataset repository for asynchronous checkpoint staging",
    )
    label.add_argument(
        "--checkpoint-namespace",
        "--checkpoint-branch",
        dest="checkpoint_branch",
        default=None,
        help=(
            "Run-specific checkpoint namespace on the dataset main tree; "
            "the legacy --checkpoint-branch spelling is accepted"
        ),
    )
    label.add_argument(
        "--checkpoint-drain-seconds",
        type=float,
        default=30.0,
        help="Maximum final wait for queued checkpoint uploads",
    )
    label.add_argument(
        "--release-lane",
        choices=tuple(lane.value for lane in ReleaseLane),
        default=None,
        help="Public release lane used for batch tracking",
    )
    label.add_argument("--trackio-project", default=None)
    label.add_argument("--trackio-run-name", default=None)
    label.add_argument("--trackio-space-id", default=None)

    probe = sub.add_parser("probe", help="Validate one live inference engine")
    probe.add_argument("--input-parquet", required=True)
    probe.add_argument("--engine", required=True, choices=("llama.cpp",))
    probe.add_argument(
        "--endpoint", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    probe.add_argument("--sample-size", type=int, default=4)
    _add_server_config_args(probe)

    finalize = sub.add_parser("finalize", help="Build validated labeled artifacts")
    finalize.add_argument("--input-parquet", required=True)
    finalize.add_argument("--work-dir", required=True)
    finalize.add_argument("--output-dir", required=True)
    finalize.add_argument("--dataset-id", required=True)
    for name in (
        "input-dataset-revision",
        "model-revision",
        "model-file-sha256",
        "source-commit",
        "engine",
        "engine-version",
    ):
        finalize.add_argument(f"--{name}", required=True)
    finalize.add_argument("--batch-size", type=int, required=True)
    finalize.add_argument("--row-limit", type=int, default=0)
    finalize.add_argument("--sampling-target", type=int, default=DEFAULT_SAMPLE_TARGET)
    finalize.add_argument("--sampling-seed", default=DEFAULT_SAMPLE_SEED)
    finalize.add_argument("--h3-resolution", type=int, default=DEFAULT_H3_RESOLUTION)
    finalize.add_argument(
        "--release-lane",
        choices=tuple(lane.value for lane in ReleaseLane),
        default=None,
        help="Public release lane recorded in the final manifest",
    )
    _add_server_config_args(finalize)

    publish = sub.add_parser("publish", help="Validate and publish final artifacts")
    publish.add_argument("--output-dir", required=True)
    publish.add_argument("--dataset-id", required=True)
    track = sub.add_parser(
        "track", help="Log one static Trackio run from final labeled artifacts"
    )
    track.add_argument("--output-dir", required=True)
    track.add_argument("--project", required=True)
    track.add_argument("--run-name", default=None)
    track.add_argument(
        "--space-id",
        default=None,
        help=(
            "Hugging Face Space ID for the Trackio static run; inferred from "
            "the release lane when omitted"
        ),
    )
    return parser


def _hex(value: str, length: int, field: str) -> str:
    if len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a {length}-character lowercase hex value")
    return value


def _resolve_runtime_plan(args: argparse.Namespace) -> RuntimePlan:
    plan = build_runtime_plan(
        parallel=args.llama_parallel,
        per_slot_context=args.llama_per_slot_context,
    )
    if args.llama_total_context is not None and (
        args.llama_total_context != plan.total_context
    ):
        raise ValueError(
            "llama total context must equal parallel times per-slot context"
        )
    if args.request_concurrency is not None and (
        args.request_concurrency < 1 or args.request_concurrency > plan.parallel
    ):
        raise ValueError(
            "request concurrency must be between 1 and the parallel slot count"
        )
    return plan


def _identity(
    args: argparse.Namespace, input_path: Path, plan: RuntimePlan
) -> RunIdentity:
    if args.row_limit < 0:
        raise ValueError("row limit must be non-negative")
    input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    request_concurrency = (
        args.request_concurrency
        if args.request_concurrency is not None
        else plan.parallel
    )
    release_lane = getattr(args, "release_lane", None)
    sampling_enabled = release_lane == ReleaseLane.V2_WORLDWIDE.value or (
        args.sampling_target > 0
    )
    return RunIdentity(
        input_sha256=input_sha,
        input_dataset_revision=_hex(args.input_dataset_revision, 40, "input revision"),
        model_repo_id=MODEL_REPO_ID,
        model_revision=_hex(args.model_revision, 40, "model revision"),
        model_file=MODEL_FILE,
        model_file_sha256=_hex(args.model_file_sha256, 64, "model file SHA-256"),
        prompt_version=PROMPT_VERSION,
        source_commit=_hex(args.source_commit, 40, "source commit"),
        engine=args.engine,
        engine_version=args.engine_version,
        batch_size=args.batch_size,
        row_limit=args.row_limit,
        llama_parallel=plan.parallel,
        llama_per_slot_context=plan.per_slot_context,
        llama_total_context=plan.total_context,
        request_concurrency=request_concurrency,
        sampling_target=args.sampling_target if sampling_enabled else None,
        sampling_seed=args.sampling_seed if sampling_enabled else None,
        h3_resolution=args.h3_resolution if sampling_enabled else None,
        sampling_version=SAMPLING_VERSION if sampling_enabled else None,
        release_lane=release_lane,
    )


def _default_engine_factory(args: argparse.Namespace) -> LabelEngine:
    plan = _resolve_runtime_plan(args)
    factory = resolve_engine_factory(plan)
    return factory(endpoint=args.endpoint, model=MODEL_REPO_ID)


_PROMPT_COLUMNS = {
    "sentence_id",
    "sentence_text_raw",
    "previous_sentence",
    "next_sentence",
    "polygon_name",
    "region",
    "osm_primary_tag",
    "osm_tags",
    "language",
    "page_title",
    "section_path",
}


def _load_input(path: Path) -> pa.Table:
    table = pq.read_table(path)
    if missing := _PROMPT_COLUMNS.difference(table.column_names):
        raise ValueError(
            f"input is missing required labeling columns: {sorted(missing)}"
        )
    return table


def _checkpoint_mirror(
    args: argparse.Namespace, store: CheckpointStore
) -> CheckpointMirror | None:
    dataset_id = args.checkpoint_dataset_id
    branch = args.checkpoint_branch
    if (dataset_id is None) != (branch is None):
        raise ValueError(
            "checkpoint dataset ID and checkpoint namespace must be supplied together"
        )
    if args.checkpoint_drain_seconds < 0:
        raise ValueError("checkpoint drain seconds must be non-negative")
    if dataset_id is None:
        return None
    return CheckpointMirror(
        store=store,
        dataset_id=dataset_id,
        branch=branch,
    )


def _batch_tracker(
    args: argparse.Namespace, store: CheckpointStore
) -> TrackioBatchLogger | None:
    """Build the optional asynchronous batch logger for a labeled run."""

    if args.trackio_project is None and args.trackio_space_id is None:
        return None
    if not args.trackio_project or not args.trackio_run_name:
        raise ValueError(
            "Trackio project and run name are required when batch tracking is enabled"
        )
    lane = (
        ReleaseLane(args.release_lane)
        if args.release_lane is not None
        else (
            ReleaseLane.V2_WORLDWIDE
            if store.identity.sampling_version is not None
            else ReleaseLane.V1_AFGHANISTAN
        )
    )
    if (
        lane is ReleaseLane.V1_AFGHANISTAN
        and store.identity.sampling_version is not None
    ):
        raise ValueError("V1 batch tracking cannot use stratified sampling")
    if lane is ReleaseLane.V2_WORLDWIDE and store.identity.sampling_version is None:
        raise ValueError("worldwide batch tracking requires stratified sampling")
    return TrackioBatchLogger(
        work_dir=store.root,
        project=args.trackio_project,
        run_name=args.trackio_run_name,
        lane=lane,
        space_id=args.trackio_space_id,
    )


def _trackio_space_for_output(output_dir: Path) -> str:
    """Infer a lane-specific Trackio Space, with the historical V1 fallback."""

    manifest_path = Path(output_dir) / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        identity = manifest.get("run_identity", {})
        if isinstance(identity, dict):
            return trackio_space_id(release_lane(identity))
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return trackio_space_id(ReleaseLane.V1_AFGHANISTAN)


def _probe(args: argparse.Namespace, engine: LabelEngine) -> int:
    table = _load_input(Path(args.input_parquet))
    if args.sample_size < 1 or args.sample_size > table.num_rows:
        raise ValueError("sample size must be within the input row count")
    selected = (
        table
        if args.sample_size == table.num_rows
        else select_canary_rows(table, args.sample_size)
    )
    prompt_inputs = [LabelingRunner._prompt(row) for row in selected.to_pylist()]
    responses = engine.generate(
        [build_messages(prompt_input) for prompt_input in prompt_inputs]
    )
    if len(responses) != len(prompt_inputs):
        raise ValueError("engine response count does not match probe size")
    for prompt_input, response in zip(prompt_inputs, responses, strict=True):
        parse_label_response(response, target_sentence=prompt_input.sentence_text)
    print(
        json.dumps(
            {
                "engine": args.engine,
                "validated_responses": len(responses),
            },
            sort_keys=True,
        )
    )
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    engine_factory: Callable[
        [argparse.Namespace], LabelEngine
    ] = _default_engine_factory,
    publish_fn: Callable[..., Any] = publish_labeled_dataset,
    track_fn: Callable[..., Any] = log_static_labeling_run,
) -> int:
    """Run one explicit labeling operation."""

    try:
        args = _parser().parse_args(argv)
        if args.command == "publish":
            result = publish_fn(Path(args.output_dir), args.dataset_id)
            print(
                json.dumps(
                    {"commit_id": result.commit_id, "commit_url": result.commit_url},
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "track":
            result = track_fn(
                Path(args.output_dir),
                project=args.project,
                run_name=args.run_name,
                space_id=args.space_id or _trackio_space_for_output(args.output_dir),
            )
            print(
                json.dumps(
                    {
                        "project": result.project,
                        "run_name": result.run_name,
                        "rows": result.row_count,
                        "kpis": result.kpis,
                        "space_id": result.space_id,
                    },
                    sort_keys=True,
                )
            )
            return 0
        plan = _resolve_runtime_plan(args)
        if args.command == "probe":
            return _probe(args, engine_factory(args))
        input_path = Path(args.input_parquet)
        identity = _identity(args, input_path, plan)
        store = CheckpointStore(Path(args.work_dir), identity)
        if args.command == "finalize":
            result = finalize_labeled_dataset(
                input_path=input_path,
                store=store,
                output_dir=Path(args.output_dir),
                dataset_repo_id=args.dataset_id,
            )
            print(
                json.dumps(
                    {"rows": result.row_count, "sha256": result.parquet_sha256},
                    sort_keys=True,
                )
            )
            return 0
        table = select_label_rows(
            _load_input(input_path),
            row_limit=args.row_limit,
            sampling_target=args.sampling_target,
            sampling_seed=args.sampling_seed,
            h3_resolution=args.h3_resolution,
        )
        stop = StopController()
        stop.install()
        repair_log_path = Path(args.work_dir) / "repair.log"
        mirror = _checkpoint_mirror(args, store)
        tracker = _batch_tracker(args, store)
        try:
            if mirror is not None:
                mirror.start()
            if tracker is not None:
                tracker.start()
            result = LabelingRunner(
                engine=engine_factory(args),
                store=store,
                batch_size=args.batch_size,
                stop_requested=stop,
                repair=BoundedRepair(max_attempts=3),
                repair_log_path=repair_log_path,
                checkpoint_mirror=(mirror.enqueue if mirror is not None else None),
                batch_tracker=(tracker.enqueue if tracker is not None else None),
            ).run(table)
        finally:
            if mirror is not None:
                mirror.close(wait=True, timeout=args.checkpoint_drain_seconds)
            if tracker is not None:
                tracker.close(wait=True, timeout=args.checkpoint_drain_seconds)
        print(
            json.dumps(
                {
                    "completed": result.completed,
                    "total": result.total,
                    "interrupted": result.interrupted,
                    "elapsed_seconds": result.elapsed_seconds,
                    "input_sha256": identity.input_sha256,
                    "repair_stats": result.repair_stats.to_dict(),
                },
                sort_keys=True,
            )
        )
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
