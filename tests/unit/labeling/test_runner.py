from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
from osm_polygon_sentence_relevance.labeling.contracts import RunIdentity
from osm_polygon_sentence_relevance.labeling.runner import LabelingRunner


def _identity() -> RunIdentity:
    return RunIdentity(
        input_sha256="a" * 64,
        input_dataset_revision="b" * 40,
        model_repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        model_revision="c" * 40,
        model_file="Qwen3.6-27B-Q4_K_M.gguf",
        model_file_sha256="d" * 64,
        prompt_version="afghanistan-landuse-polygon-v1",
        source_commit="e" * 40,
        engine="vllm",
        engine_version="0.21.0",
        batch_size=2,
    )


def _table(count: int = 5) -> pa.Table:
    rows = []
    for i in range(count):
        rows.append(
            {
                "sentence_id": f"s{i}",
                "sentence_text_raw": f"Sentence {i} describes farming.",
                "previous_sentence": None,
                "next_sentence": None,
                "polygon_name": "Place",
                "region": "afghanistan",
                "osm_primary_tag": "landuse=farmland",
                "osm_tags": [{"key": "landuse", "value": "farmland"}],
                "language": "en",
                "page_title": "Place",
                "section_path": ["Economy"],
            }
        )
    return pa.Table.from_pylist(rows)


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def generate(self, messages: list[list[dict[str, str]]]) -> list[str]:
        self.calls.append([m[1]["content"] for m in messages])
        return [
            json.dumps(
                {
                    "landuse_relevance": "yes",
                    "polygon_relevance": "yes",
                    "landuse_reason": "explicit_land_use",
                    "polygon_reason": "direct_polygon_reference",
                    "evidence": "farming",
                }
            )
            for _ in messages
        ]


def test_runs_bounded_batches_and_resumes_without_relabeling(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    engine = FakeEngine()
    first = LabelingRunner(engine=engine, store=store, batch_size=2).run(_table())
    assert first.completed == 5
    assert [len(call) for call in engine.calls] == [2, 2, 1]

    resumed_engine = FakeEngine()
    second = LabelingRunner(
        engine=resumed_engine,
        store=CheckpointStore(tmp_path, _identity()),
        batch_size=2,
    ).run(_table())
    assert second.completed == 5
    assert resumed_engine.calls == []


def test_stop_finishes_current_batch_and_is_resumable(tmp_path: Path) -> None:
    engine = FakeEngine()
    checks = iter([False, True])
    runner = LabelingRunner(
        engine=engine,
        store=CheckpointStore(tmp_path, _identity()),
        batch_size=2,
        stop_requested=lambda: next(checks, True),
    )
    result = runner.run(_table())

    assert result.interrupted is True
    assert result.completed == 2
    assert len(CheckpointStore(tmp_path, _identity()).load_all()) == 2


def test_progress_and_final_timing_are_written(tmp_path: Path) -> None:
    times = iter([10.0, 12.0, 14.0, 16.0, 18.0])
    result = LabelingRunner(
        engine=FakeEngine(),
        store=CheckpointStore(tmp_path, _identity()),
        batch_size=2,
        clock=lambda: next(times, 18.0),
    ).run(_table(2))

    progress = json.loads((tmp_path / "progress.json").read_text())
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert progress["completed"] == 2
    assert progress["remaining"] == 0
    assert timing["total_wall_seconds"] >= 0
    assert timing["inference_seconds"] >= 0
    assert result.elapsed_seconds == timing["total_wall_seconds"]


def test_output_order_is_input_order_even_after_resume(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    runner = LabelingRunner(engine=FakeEngine(), store=store, batch_size=2)
    runner.run(_table())
    assert [r.sentence_id for r in store.load_all()] == [f"s{i}" for i in range(5)]
