from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa

from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
from osm_polygon_sentence_relevance.labeling.contracts import RunIdentity
from osm_polygon_sentence_relevance.labeling.runner import LabelingRunner
from osm_polygon_sentence_relevance.labeling.sampling import select_stratified_rows


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
        engine="llama.cpp",
        engine_version="1",
        batch_size=2,
        llama_parallel=16,
        llama_per_slot_context=4096,
        llama_total_context=65536,
        request_concurrency=16,
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
                "lat": 34.0 + i,
                "lon": 69.0 + i,
            }
        )
    return pa.Table.from_pylist(rows)


def test_expanded_v2_target_reuses_checkpoints_and_labels_only_new_rows(
    tmp_path: Path,
) -> None:
    initial_identity = replace(
        _identity(),
        sampling_target=3,
        sampling_seed="sentence-relevance-v2",
        h3_resolution=3,
        sampling_version="labeling-v2-h3-language-osm-primary",
    )
    initial_table = select_stratified_rows(_table(), target=3)
    first_engine = FakeEngine()
    first = LabelingRunner(
        engine=first_engine,
        store=CheckpointStore(tmp_path, initial_identity),
        batch_size=2,
    ).run(initial_table)
    assert first.completed == 3

    expanded_identity = replace(initial_identity, sampling_target=5)
    expanded_table = select_stratified_rows(_table(), target=5)
    resumed_engine = FakeEngine()
    second = LabelingRunner(
        engine=resumed_engine,
        store=CheckpointStore(tmp_path, expanded_identity),
        batch_size=2,
    ).run(expanded_table)

    assert second.completed == 5
    assert [len(call) for call in resumed_engine.calls] == [2]
    assert len(CheckpointStore(tmp_path, expanded_identity).load_all()) == 5


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


def test_batch_tracker_receives_numeric_throughput_metrics(tmp_path: Path) -> None:
    metrics: list[dict[str, object]] = []
    times = iter([10.0, 10.0, 12.0, 12.0, 12.0])
    runner = LabelingRunner(
        engine=FakeEngine(),
        store=CheckpointStore(tmp_path, _identity()),
        batch_size=2,
        clock=lambda: next(times),
        batch_tracker=metrics.append,
    )

    runner.run(_table(2))

    assert len(metrics) == 1
    assert metrics[0]["completed_rows"] == 2
    assert metrics[0]["rows_per_second"] == 1.0
    assert metrics[0]["eta_seconds"] == 0.0


def test_output_order_is_input_order_even_after_resume(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    runner = LabelingRunner(engine=FakeEngine(), store=store, batch_size=2)
    runner.run(_table())
    assert [r.sentence_id for r in store.load_all()] == [f"s{i}" for i in range(5)]


def test_checkpoint_mirror_is_called_after_each_atomic_batch(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    calls: list[tuple[int, int]] = []

    def enqueue(index: int) -> None:
        calls.append((index, len(store.load_all())))

    result = LabelingRunner(
        engine=FakeEngine(),
        store=store,
        batch_size=2,
        checkpoint_mirror=enqueue,
    ).run(_table())

    assert result.completed == 5
    assert calls == [(0, 2), (1, 4), (2, 5)]
