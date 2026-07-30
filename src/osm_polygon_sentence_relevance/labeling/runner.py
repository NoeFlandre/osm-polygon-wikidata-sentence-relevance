"""Bounded, resumable labeling orchestration with factual timing."""

from __future__ import annotations

import json
import signal
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pyarrow as pa

from .checkpoint import CheckpointStore
from .contracts import LabelRecord, SentenceLabel
from .engine import LabelEngine
from .prompt import PromptInput, build_messages
from .repair import BoundedRepair, Messages, RepairEngine, RepairStats
from .validation import LabelValidationError


def _passthrough_engine(
    initial_response: str, fallback_engine: RepairEngine
) -> RepairEngine:
    """Return an engine that emits ``initial_response`` first then delegates to ``fallback_engine``.

    The first invocation returns the original batched response verbatim so
    the strict validator runs on the model's actual output. Subsequent
    invocations are routed to ``fallback_engine`` so the bounded repair can
    issue a single-prompt request. The fallback engine is expected to call
    the production server with the repaired message; the response is
    validated against the same strict schema.
    """

    state = {"called": 0}

    def call(messages: Sequence[Messages]) -> list[str]:
        state["called"] += 1
        if state["called"] == 1:
            return [initial_response]
        return fallback_engine(messages)

    return call


def _batched_to_single_engine(
    engine: LabelEngine,
    *,
    clock: Callable[[], float] | None = None,
    repair_time_holder: list[float] | None = None,
) -> RepairEngine:
    """Adapt a batched :class:`LabelEngine` to a single-message :class:`RepairEngine`.

    When ``clock`` and ``repair_time_holder`` are provided, the wall time
    spent on each single-row invocation is added to ``repair_time_holder``
    so the runner can report ``repair_inference_seconds`` independently
    of the batched ``initial_inference_seconds`` total.
    """

    def call(messages: Sequence[Messages]) -> list[str]:
        if clock is not None and repair_time_holder is not None:
            before = clock()
            result = list(engine.generate(list(messages)))
            repair_time_holder[0] += max(0.0, clock() - before)
            return result
        return list(engine.generate(list(messages)))

    return call


def build_timing_payload(
    *,
    initial_inference_seconds: float,
    repair_inference_seconds: float,
    checkpoint_and_validation_seconds: float,
    completed: int,
    total: int,
    interrupted: bool,
    repair_stats: Mapping[str, object],
    started_at: float,
    finished_at: float,
) -> dict[str, object]:
    """Assemble one atomic timing payload with split inference components.

    The contract is::

        inference_seconds = initial_inference_seconds + repair_inference_seconds
        total_wall_seconds = inference_seconds + checkpoint_and_validation_seconds

    All values are non-negative. ``interrupted`` is recorded verbatim so
    the resume path can distinguish completed batches from a SIGINT-stopped
    allocation.
    """

    initial = max(0.0, float(initial_inference_seconds))
    repair = max(0.0, float(repair_inference_seconds))
    checkpoint = max(0.0, float(checkpoint_and_validation_seconds))
    total_wall = max(0.0, float(finished_at) - float(started_at))
    return {
        "completed": int(completed),
        "total": int(total),
        "interrupted": bool(interrupted),
        "initial_inference_seconds": initial,
        "repair_inference_seconds": repair,
        "inference_seconds": initial + repair,
        "checkpoint_and_validation_seconds": checkpoint,
        "total_wall_seconds": total_wall,
        "repair_stats": dict(repair_stats),
    }


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of a complete or safely interrupted run."""

    completed: int
    total: int
    interrupted: bool
    elapsed_seconds: float
    inference_seconds: float
    initial_inference_seconds: float
    repair_inference_seconds: float
    repair_stats: RepairStats


class StopController:
    """Signal handler that requests a stop after the current batch."""

    def __init__(self) -> None:
        self.requested = False

    def __call__(self) -> bool:
        return self.requested

    def request(self, signum: int, frame: object) -> None:
        del signum, frame
        self.requested = True

    def install(self) -> None:
        signal.signal(signal.SIGINT, self.request)
        signal.signal(signal.SIGTERM, self.request)


class LabelingRunner:
    """Label unseen rows in bounded batches and checkpoint each batch."""

    def __init__(
        self,
        *,
        engine: LabelEngine,
        store: CheckpointStore,
        batch_size: int,
        stop_requested: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        repair: BoundedRepair | None = None,
        repair_log_path: Path | None = None,
        repair_max_attempts: int = 1,
    ) -> None:
        if isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if batch_size != store.identity.batch_size:
            raise ValueError("batch_size must match checkpoint identity")
        self.engine = engine
        self.store = store
        self.batch_size = batch_size
        self.stop_requested = stop_requested or (lambda: False)
        self.clock = clock
        self.repair = repair or BoundedRepair(max_attempts=repair_max_attempts)
        self.repair_log_path = repair_log_path

    def _log_repair_event(self, entry: dict[str, object]) -> None:
        if self.repair_log_path is None:
            return
        sanitized = {
            key: ("<redacted>" if key in {"prompt", "response"} else value)
            for key, value in entry.items()
        }
        self.repair_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.repair_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sanitized, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    def _label_one(
        self,
        *,
        engine_callable: RepairEngine,
        messages: Messages,
        target_sentence: str,
        sentence_id: str,
    ) -> SentenceLabel:
        initial_failures_before = self.repair.stats.initial_failures
        try:
            label = self.repair.call(
                engine=engine_callable,
                messages=list(messages),
                target_sentence=target_sentence,
            )
        except LabelValidationError as exc:
            self._log_repair_event(
                {
                    "sentence_id": sentence_id,
                    "reason": str(exc),
                    "attempt": 0,
                    "event": "label_repair_initial_failure",
                }
            )
            raise
        # Log the resolved state for any row whose initial response was
        # rejected, regardless of whether the repair succeeded.
        if self.repair.stats.initial_failures > initial_failures_before:
            self._log_repair_event(
                {
                    "sentence_id": sentence_id,
                    "event": "label_repair_recovered",
                    "attempts": self.repair.stats.to_dict(),
                }
            )
        return label

    @staticmethod
    def _prompt(row: dict[str, object]) -> PromptInput:
        def optional_text(field: str) -> str | None:
            value = row[field]
            if value is not None and not isinstance(value, str):
                raise ValueError("input row has invalid prompt context")
            return value

        tags = row["osm_tags"]
        section_path = row["section_path"]
        if not isinstance(tags, list) or not isinstance(section_path, list):
            raise ValueError("input row has invalid prompt context")
        return PromptInput(
            sentence_id=str(row["sentence_id"]),
            sentence_text=str(row["sentence_text_raw"]),
            previous_sentence=optional_text("previous_sentence"),
            next_sentence=optional_text("next_sentence"),
            polygon_name=optional_text("polygon_name"),
            region=str(row["region"]),
            osm_primary_tag=optional_text("osm_primary_tag"),
            osm_tags=tuple(cast(list[dict[str, str]], tags)),
            language=str(row["language"]),
            page_title=str(row["page_title"]),
            section_path=tuple(str(value) for value in section_path),
        )

    def run(self, table: pa.Table) -> RunResult:
        """Run until complete or until a stop is requested at a batch boundary."""

        started = self.clock()
        completed = self.store.completed_ids()
        ids = table.column("sentence_id").to_pylist()
        if len(ids) != len(set(ids)):
            raise ValueError("input contains duplicate sentence IDs")
        if not completed.issubset(set(ids)):
            raise ValueError("checkpoints contain sentence IDs absent from input")
        pending_indexes = [
            index for index, value in enumerate(ids) if value not in completed
        ]
        batch_index = len(self.store._batch_indexes())
        initial_inference_seconds = 0.0
        repair_time_holder: list[float] = [0.0]
        interrupted = False
        for offset in range(0, len(pending_indexes), self.batch_size):
            if self.stop_requested():
                interrupted = True
                break
            indexes = pending_indexes[offset : offset + self.batch_size]
            rows = table.take(pa.array(indexes, type=pa.int64())).to_pylist()
            prompt_inputs = [self._prompt(row) for row in rows]
            before = self.clock()
            responses = self.engine.generate(
                [build_messages(prompt_input) for prompt_input in prompt_inputs]
            )
            initial_inference_seconds += max(0.0, self.clock() - before)
            if len(responses) != len(rows):
                raise ValueError("engine response count does not match request count")
            records: list[LabelRecord] = []
            # Per-batch single-row engine used only for repair: timer
            # accumulates wall time on every invocation past the first.
            fallback = _batched_to_single_engine(
                self.engine,
                clock=self.clock,
                repair_time_holder=repair_time_holder,
            )
            for prompt_input, response, messages in zip(
                prompt_inputs,
                responses,
                [build_messages(prompt_input) for prompt_input in prompt_inputs],
                strict=True,
            ):
                try:
                    label = self._label_one(
                        engine_callable=_passthrough_engine(response, fallback),
                        messages=messages,
                        target_sentence=prompt_input.sentence_text,
                        sentence_id=prompt_input.sentence_id,
                    )
                except LabelValidationError:
                    # Fail the batch: do not write partial results.
                    raise
                records.append(
                    LabelRecord(
                        sentence_id=prompt_input.sentence_id,
                        landuse_relevance=label.landuse_relevance,
                        polygon_relevance=label.polygon_relevance,
                        landuse_reason=label.landuse_reason,
                        polygon_reason=label.polygon_reason,
                        evidence=label.evidence,
                    )
                )
            self.store.write_batch(batch_index, records)
            batch_index += 1
            completed.update(record.sentence_id for record in records)
            elapsed = max(0.0, self.clock() - started)
            self.store.write_progress(
                completed=len(completed), total=table.num_rows, elapsed_seconds=elapsed
            )
        total_elapsed = max(0.0, self.clock() - started)
        repair_inference_seconds = repair_time_holder[0]
        timing = build_timing_payload(
            initial_inference_seconds=initial_inference_seconds,
            repair_inference_seconds=repair_inference_seconds,
            checkpoint_and_validation_seconds=max(
                0.0,
                total_elapsed - initial_inference_seconds - repair_inference_seconds,
            ),
            completed=len(completed),
            total=table.num_rows,
            interrupted=interrupted,
            repair_stats=self.repair.stats.to_dict(),
            started_at=started,
            finished_at=total_elapsed + started,
        )
        self.store.write_timing(timing)
        return RunResult(
            completed=len(completed),
            total=table.num_rows,
            interrupted=interrupted,
            elapsed_seconds=total_elapsed,
            inference_seconds=initial_inference_seconds + repair_inference_seconds,
            initial_inference_seconds=initial_inference_seconds,
            repair_inference_seconds=repair_inference_seconds,
            repair_stats=self.repair.stats,
        )


__all__ = [
    "LabelingRunner",
    "RunResult",
    "StopController",
    "build_timing_payload",
]
