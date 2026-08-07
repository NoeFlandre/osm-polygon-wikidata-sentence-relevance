"""Resumable batch runner for V2 one-token score inference."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from .v2_checkpoint import V2CheckpointStore
from .v2_engine import V2Engine
from .v2_prompt import V2PromptInput, build_v2_messages


@dataclass(frozen=True, slots=True)
class V2RunResult:
    """Factual result of a complete or interrupted V2 run."""

    completed: int
    total: int
    interrupted: bool
    elapsed_seconds: float
    inference_seconds: float


def _optional_text(row: dict[str, object], name: str) -> str | None:
    value = row.get(name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"V2 input field {name} must be text or null")
    return value


def _prompt(row: dict[str, object]) -> V2PromptInput:
    section = row.get("section_path")
    if isinstance(section, list):
        section_title = str(section[-1]) if section else "none"
    elif isinstance(section, str):
        section_title = section
    else:
        raise ValueError("V2 input section_path must be a list or string")
    return V2PromptInput(
        sentence_id=str(row["sentence_id"]),
        sentence_text=str(row.get("sentence_text_raw", row.get("sentence_text", ""))),
        previous_sentence=_optional_text(row, "previous_sentence"),
        next_sentence=_optional_text(row, "next_sentence"),
        page_title=str(row.get("page_title", "")),
        section_title=section_title,
    )


def _prior_timing(root: Path) -> tuple[float, float]:
    """Read cumulative timing from a previous interrupted attempt."""

    path = root / "timing.json"
    if not path.is_file():
        return 0.0, 0.0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        elapsed = float(payload["total_wall_seconds"])
        inference = float(payload["inference_seconds"])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise ValueError("V2 timing artifact is invalid") from exc
    if elapsed < 0 or inference < 0:
        raise ValueError("V2 timing values must be non-negative")
    return elapsed, inference


class V2LogitRunner:
    """Label unseen rows, writing one validated checkpoint per batch."""

    def __init__(
        self,
        *,
        engine: V2Engine,
        store: V2CheckpointStore,
        batch_size: int,
        stop_requested: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        batch_callback: Callable[[dict[str, object]], None] | None = None,
        checkpoint_mirror: Callable[[int], None] | None = None,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if batch_size != store.identity.batch_size:
            raise ValueError("batch_size must match V2 checkpoint identity")
        self.engine = engine
        self.store = store
        self.batch_size = batch_size
        self.stop_requested = stop_requested or (lambda: False)
        self.clock = clock
        self.batch_callback = batch_callback
        self.checkpoint_mirror = checkpoint_mirror

    def run(self, table: pa.Table) -> V2RunResult:
        """Run until all rows are checkpointed or a boundary stop is requested."""

        required = {"sentence_id", "page_title", "section_path"}
        if missing := required.difference(table.column_names):
            raise ValueError(f"V2 input is missing required columns: {sorted(missing)}")
        prior_elapsed, prior_inference = _prior_timing(self.store.root)
        started = self.clock()
        rows = table.to_pylist()
        ids = [str(row["sentence_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("V2 input contains duplicate sentence IDs")
        completed = self.store.completed_ids()
        if not completed.issubset(set(ids)):
            raise ValueError("V2 checkpoints contain IDs absent from input")
        pending = [row for row in rows if str(row["sentence_id"]) not in completed]
        batch_index = len(self.store.batch_indexes())
        inference_seconds = prior_inference
        interrupted = False
        for offset in range(0, len(pending), self.batch_size):
            if self.stop_requested():
                interrupted = True
                break
            batch = pending[offset : offset + self.batch_size]
            inputs = [_prompt(row) for row in batch]
            before = self.clock()
            records = self.engine.generate(
                [build_v2_messages(item) for item in inputs],
                sentence_ids=[item.sentence_id for item in inputs],
            )
            inference_seconds += max(0.0, self.clock() - before)
            if len(records) != len(batch):
                raise ValueError("V2 engine response count does not match input")
            expected_ids = {str(row["sentence_id"]) for row in batch}
            actual_ids = [record.sentence_id for record in records]
            if (
                len(actual_ids) != len(set(actual_ids))
                or set(actual_ids) != expected_ids
            ):
                raise ValueError("V2 engine response IDs do not match input")
            self.store.write_batch(batch_index, records)
            batch_index += 1
            completed.update(record.sentence_id for record in records)
            elapsed = prior_elapsed + max(0.0, self.clock() - started)
            _write_progress(
                self.store.root,
                completed=len(completed),
                total=len(rows),
                elapsed_seconds=elapsed,
                inference_seconds=inference_seconds,
            )
            if self.checkpoint_mirror is not None:
                self.checkpoint_mirror(batch_index - 1)
            if self.batch_callback is not None:
                rate = len(completed) / elapsed if elapsed else 0.0
                self.batch_callback(
                    {
                        "batch_index": batch_index - 1,
                        "batch_rows": len(records),
                        "completed_rows": len(completed),
                        "total_rows": len(rows),
                        "remaining_rows": len(rows) - len(completed),
                        "rows_per_second": rate,
                        "eta_seconds": (len(rows) - len(completed)) / rate
                        if rate
                        else None,
                        "inference_seconds": inference_seconds,
                    }
                )
        elapsed = prior_elapsed + max(0.0, self.clock() - started)
        _write_progress(
            self.store.root,
            completed=len(completed),
            total=len(rows),
            elapsed_seconds=elapsed,
            inference_seconds=inference_seconds,
        )
        _atomic_json(
            self.store.root / "timing.json",
            {
                "completed": len(completed),
                "total": len(rows),
                "interrupted": interrupted,
                "inference_seconds": inference_seconds,
                "total_wall_seconds": elapsed,
            },
        )
        return V2RunResult(
            completed=len(completed),
            total=len(rows),
            interrupted=interrupted,
            elapsed_seconds=elapsed,
            inference_seconds=inference_seconds,
        )


def _write_progress(
    root: Path,
    *,
    completed: int,
    total: int,
    elapsed_seconds: float,
    inference_seconds: float,
) -> None:
    rate = completed / elapsed_seconds if elapsed_seconds else 0.0
    payload = {
        "completed": completed,
        "total": total,
        "remaining": total - completed,
        "elapsed_seconds": elapsed_seconds,
        "inference_seconds": inference_seconds,
        "rows_per_second": rate,
        "eta_seconds": (total - completed) / rate if rate else None,
    }
    _atomic_json(root / "progress.json", payload)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write a progress artifact atomically and durably."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        temporary.unlink(missing_ok=True)
        raise


__all__ = ["V2LogitRunner", "V2RunResult"]
