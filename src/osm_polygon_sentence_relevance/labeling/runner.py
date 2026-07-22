"""Bounded, resumable labeling orchestration with factual timing."""

from __future__ import annotations

import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pyarrow as pa

from .checkpoint import CheckpointStore
from .contracts import LabelRecord
from .engine import LabelEngine
from .prompt import PromptInput, build_messages
from .validation import parse_label_response


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of a complete or safely interrupted run."""

    completed: int
    total: int
    interrupted: bool
    elapsed_seconds: float
    inference_seconds: float


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

    @staticmethod
    def _prompt(row: dict[str, object]) -> PromptInput:
        tags = row["osm_tags"]
        section_path = row["section_path"]
        if not isinstance(tags, list) or not isinstance(section_path, list):
            raise ValueError("input row has invalid prompt context")
        return PromptInput(
            sentence_id=str(row["sentence_id"]),
            sentence_text=str(row["sentence_text_raw"]),
            previous_sentence=row["previous_sentence"],  # type: ignore[arg-type]
            next_sentence=row["next_sentence"],  # type: ignore[arg-type]
            polygon_name=row["polygon_name"],  # type: ignore[arg-type]
            region=str(row["region"]),
            osm_primary_tag=row["osm_primary_tag"],  # type: ignore[arg-type]
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
        inference_seconds = 0.0
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
            inference_seconds += max(0.0, self.clock() - before)
            if len(responses) != len(rows):
                raise ValueError("engine response count does not match request count")
            records: list[LabelRecord] = []
            for prompt_input, raw in zip(prompt_inputs, responses, strict=True):
                label = parse_label_response(
                    raw, target_sentence=prompt_input.sentence_text
                )
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
        timing: dict[str, float | int | bool] = {
            "completed": len(completed),
            "total": table.num_rows,
            "interrupted": interrupted,
            "inference_seconds": inference_seconds,
            "checkpoint_and_validation_seconds": max(
                0.0, total_elapsed - inference_seconds
            ),
            "total_wall_seconds": total_elapsed,
        }
        self.store.write_timing(timing)
        return RunResult(
            completed=len(completed),
            total=table.num_rows,
            interrupted=interrupted,
            elapsed_seconds=total_elapsed,
            inference_seconds=inference_seconds,
        )


__all__ = ["LabelingRunner", "RunResult", "StopController"]
