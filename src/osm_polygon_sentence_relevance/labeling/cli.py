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
from .contracts import RunIdentity
from .engine import LabelEngine, OpenAICompatibleEngine
from .finalization import finalize_labeled_dataset
from .prompt import PROMPT_VERSION, build_messages
from .publication import publish_labeled_dataset
from .runner import LabelingRunner, StopController
from .validation import parse_label_response

MODEL_REPO_ID = "unsloth/Qwen3.6-27B-MTP-GGUF"
MODEL_FILE = "Qwen3.6-27B-Q4_K_M.gguf"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label Afghanistan polygon sentences")
    sub = parser.add_subparsers(dest="command", required=True)
    label = sub.add_parser("label", help="Run or resume LLM labeling")
    label.add_argument("--input-parquet", required=True)
    label.add_argument("--work-dir", required=True)
    label.add_argument("--input-dataset-revision", required=True)
    label.add_argument("--model-revision", required=True)
    label.add_argument("--model-file-sha256", required=True)
    label.add_argument("--source-commit", required=True)
    label.add_argument("--engine", required=True, choices=("vllm", "llama.cpp"))
    label.add_argument("--engine-version", required=True)
    label.add_argument(
        "--endpoint", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    label.add_argument("--batch-size", type=int, default=128)
    label.add_argument("--concurrency", type=int, default=32)
    label.add_argument(
        "--row-limit",
        type=int,
        default=0,
        help="Deterministic canary size; zero labels the complete input",
    )

    probe = sub.add_parser("probe", help="Validate one live inference engine")
    probe.add_argument("--input-parquet", required=True)
    probe.add_argument("--engine", required=True, choices=("vllm", "llama.cpp"))
    probe.add_argument(
        "--endpoint", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    probe.add_argument("--concurrency", type=int, default=4)
    probe.add_argument("--sample-size", type=int, default=4)

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

    publish = sub.add_parser("publish", help="Validate and publish final artifacts")
    publish.add_argument("--output-dir", required=True)
    publish.add_argument("--dataset-id", required=True)
    return parser


def _hex(value: str, length: int, field: str) -> str:
    if len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a {length}-character lowercase hex value")
    return value


def _identity(args: argparse.Namespace, input_path: Path) -> RunIdentity:
    if args.row_limit < 0:
        raise ValueError("row limit must be non-negative")
    input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
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
    )


def _default_engine(args: argparse.Namespace) -> LabelEngine:
    return OpenAICompatibleEngine(
        endpoint=args.endpoint,
        model=MODEL_REPO_ID,
        concurrency=args.concurrency,
    )


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


def _load_afghanistan(path: Path) -> pa.Table:
    table = pq.read_table(path)
    if missing := _PROMPT_COLUMNS.difference(table.column_names):
        raise ValueError(
            f"input is missing required labeling columns: {sorted(missing)}"
        )
    if set(table["region"].to_pylist()) != {"afghanistan"}:
        raise ValueError("labeling accepts only the Afghanistan proof-of-concept")
    return table


def _probe(args: argparse.Namespace, engine: LabelEngine) -> int:
    table = _load_afghanistan(Path(args.input_parquet))
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
    engine_factory: Callable[[argparse.Namespace], LabelEngine] = _default_engine,
    publish_fn: Callable[..., Any] = publish_labeled_dataset,
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
        if args.command == "probe":
            return _probe(args, engine_factory(args))
        input_path = Path(args.input_parquet)
        identity = _identity(args, input_path)
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
        table = select_canary_rows(
            _load_afghanistan(input_path),
            args.row_limit,
        )
        stop = StopController()
        stop.install()
        result = LabelingRunner(
            engine=engine_factory(args),
            store=store,
            batch_size=args.batch_size,
            stop_requested=stop,
        ).run(table)
        print(
            json.dumps(
                {
                    "completed": result.completed,
                    "total": result.total,
                    "interrupted": result.interrupted,
                    "elapsed_seconds": result.elapsed_seconds,
                    "input_sha256": identity.input_sha256,
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
