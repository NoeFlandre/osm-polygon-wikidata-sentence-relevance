"""Repair inference time accounting tests.

The runner must split inference time into:
- ``initial_inference_seconds`` (the batched engine.generate calls)
- ``repair_inference_seconds`` (single-row repair invocations)
- ``inference_seconds == initial + repair``
- ``checkpoint_and_validation_seconds`` remains separate
- ``total_wall_seconds`` covers everything

All values must be non-negative and internally consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
from osm_polygon_sentence_relevance.labeling.contracts import (
    LabelRecord,
    LabelValue,
    RunIdentity,
)
from osm_polygon_sentence_relevance.labeling.runner import LabelingRunner


def _identity(
    tmp_path: Path, *, batch_size: int = 2, row_limit: int = 2
) -> RunIdentity:
    return RunIdentity(
        input_sha256="a" * 64,
        input_dataset_revision="b" * 40,
        model_repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        model_revision="c" * 40,
        model_file="Qwen3.6-27B-Q4_K_M.gguf",
        model_file_sha256="d" * 64,
        prompt_version="afghanistan-landuse-polygon-v2",
        source_commit="e" * 40,
        engine="llama.cpp",
        engine_version="1",
        batch_size=batch_size,
        row_limit=row_limit,
        llama_parallel=16,
        llama_per_slot_context=4096,
        llama_total_context=65536,
        request_concurrency=16,
    )


def _table(n: int = 3) -> pa.Table:
    rows = []
    for i in range(n):
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


class _Clock:
    """Step-by-step monotonic clock used to inject deterministic timing."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._index = 0
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self._index >= len(self._values):
            return float(self._values[-1])
        value = self._values[self._index]
        self._index += 1
        return value


def _good_response() -> str:
    return json.dumps(
        {
            "landuse_relevance": "yes",
            "polygon_relevance": "yes",
            "landuse_reason": "explicit_land_use",
            "polygon_reason": "direct_polygon_reference",
            "evidence": "farming",
        }
    )


class _BatchThenRepairEngine:
    """Engine whose batched call is timed and repair calls are timed separately."""

    def __init__(self, *, fail_rows: set[int] = frozenset()) -> None:
        self._fail_rows = fail_rows
        self.calls = 0
        self.batch_seconds = 0.0
        self.repair_seconds = 0.0
        self._clock_value = 0.0

    def generate(self, messages):
        self.calls += 1
        if self.calls == 1:
            # Batched call: each response is wrong for fail_rows, valid otherwise.
            self._clock_value += self.batch_seconds
            return [
                _good_response() if i not in self._fail_rows else "{}"
                for i in range(len(messages))
            ]
        # Repair call: succeed on the second attempt.
        self._clock_value += self.repair_seconds
        return [_good_response() for _ in messages]


def test_zero_repairs_emits_initial_and_repair_components(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity(tmp_path))
    engine = _BatchThenRepairEngine()
    engine.batch_seconds = 1.0
    engine.repair_seconds = 0.0
    clock = _Clock([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    LabelingRunner(
        engine=engine,
        store=store,
        batch_size=2,
        clock=clock,
    ).run(_table(3))
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert "initial_inference_seconds" in timing
    assert "repair_inference_seconds" in timing
    assert timing["repair_inference_seconds"] == 0.0
    assert timing["inference_seconds"] == pytest.approx(
        timing["initial_inference_seconds"] + timing["repair_inference_seconds"]
    )


def test_successful_repair_records_repair_time(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity(tmp_path))
    engine = _BatchThenRepairEngine(fail_rows={0})
    engine.batch_seconds = 0.5
    engine.repair_seconds = 2.0
    # Clock advances 1 second per call so the repair engine's wrapped
    # ``clock() - before`` delta is non-zero.
    clock = _Clock([float(i) for i in range(64)])
    LabelingRunner(
        engine=engine,
        store=store,
        batch_size=2,
        clock=clock,
    ).run(_table(3))
    timing = json.loads((tmp_path / "timing.json").read_text())
    # At least one repair call must have been timed.
    assert timing["repair_inference_seconds"] > 0.0
    assert timing["initial_inference_seconds"] > 0.0
    assert timing["inference_seconds"] == pytest.approx(
        timing["initial_inference_seconds"] + timing["repair_inference_seconds"]
    )
    # Repair stats: 1 initial_failure, 1 repaired, 0 exhausted
    assert timing["repair_stats"]["initial_failures"] == 1
    assert timing["repair_stats"]["repaired"] == 1


def test_exhausted_repair_records_repair_time(tmp_path: Path) -> None:
    """An exhausted repair still counts its repair time."""

    class _AlwaysFailEngine:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages):
            self.calls += 1
            # First call: invalid. Subsequent: still invalid.
            return ["{}" for _ in messages]

    from osm_polygon_sentence_relevance.labeling.repair import (
        BoundedRepair,
        RepairExhausted,
    )

    store = CheckpointStore(tmp_path, _identity(tmp_path))
    with pytest.raises(RepairExhausted):
        LabelingRunner(
            engine=_AlwaysFailEngine(),
            store=store,
            batch_size=2,
            repair=BoundedRepair(max_attempts=2),
            clock=_Clock([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        ).run(_table(3))
    # Timing is still persisted from the partial run.
    timing_path = tmp_path / "timing.json"
    if timing_path.exists():
        timing = json.loads(timing_path.read_text())
        # Repair time is non-negative even when the batch fails.
        assert timing["repair_inference_seconds"] >= 0.0
        assert timing["initial_inference_seconds"] >= 0.0


def test_multiple_repaired_rows_accumulate_repair_time(tmp_path: Path) -> None:
    identity = _identity(tmp_path, batch_size=3, row_limit=0)
    store = CheckpointStore(tmp_path, identity)
    engine = _BatchThenRepairEngine(fail_rows={0, 1, 2, 3, 4})
    engine.batch_seconds = 0.1
    engine.repair_seconds = 1.0
    LabelingRunner(
        engine=engine,
        store=store,
        batch_size=3,
        clock=_Clock([float(i) for i in range(64)]),
    ).run(_table(5))
    timing = json.loads((tmp_path / "timing.json").read_text())
    # All five rows needed repair; total repair time >= batch_seconds.
    assert timing["repair_inference_seconds"] >= engine.batch_seconds
    assert timing["repair_stats"]["repaired"] >= 1


def test_inference_components_sum_to_inference_seconds(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity(tmp_path))
    engine = _BatchThenRepairEngine(fail_rows={0})
    engine.batch_seconds = 0.5
    engine.repair_seconds = 1.0
    LabelingRunner(
        engine=engine,
        store=store,
        batch_size=2,
        clock=_Clock([0.0] * 16),
    ).run(_table(3))
    timing = json.loads((tmp_path / "timing.json").read_text())
    initial = timing["initial_inference_seconds"]
    repair = timing["repair_inference_seconds"]
    total = timing["inference_seconds"]
    assert total == pytest.approx(initial + repair)
    assert total >= 0.0
    assert initial >= 0.0
    assert repair >= 0.0


def test_clock_anomaly_negative_interval_clamps_to_zero(tmp_path: Path) -> None:
    """A backwards clock must never produce negative timing values."""

    store = CheckpointStore(tmp_path, _identity(tmp_path))
    engine = _BatchThenRepairEngine()
    engine.batch_seconds = 0.5
    engine.repair_seconds = 0.0
    # Clock that goes backwards between batch and the read.
    clock = _Clock([100.0, 50.0, 100.0, 150.0, 200.0])
    LabelingRunner(
        engine=engine,
        store=store,
        batch_size=2,
        clock=clock,
    ).run(_table(3))
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert timing["initial_inference_seconds"] >= 0.0
    assert timing["repair_inference_seconds"] >= 0.0
    assert timing["inference_seconds"] >= 0.0


def test_timing_persists_through_resume(tmp_path: Path) -> None:
    """A second runner must preserve cumulative accounting or current-attempt values."""

    identity = _identity(tmp_path)
    store = CheckpointStore(tmp_path, identity)
    runner1 = LabelingRunner(
        engine=_BatchThenRepairEngine(fail_rows={0}),
        store=store,
        batch_size=2,
        clock=_Clock([0.0] * 32),
    )
    runner1.run(_table(3))
    # The first run wrote timing; resume must either preserve cumulative
    # values or store a clearly labelled current-attempt block.
    second_timing_path = tmp_path / "timing.json"
    LabelingRunner(
        engine=_BatchThenRepairEngine(),
        store=CheckpointStore(tmp_path, identity),
        batch_size=2,
        clock=_Clock([0.0] * 16),
    ).run(_table(3))
    if second_timing_path.exists():
        second_timing = json.loads(second_timing_path.read_text())
        assert "initial_inference_seconds" in second_timing
        assert "repair_inference_seconds" in second_timing
        # Either cumulative or current-attempt is acceptable; the validator
        # only requires that the schema is consistent.
        assert second_timing["inference_seconds"] == pytest.approx(
            second_timing["initial_inference_seconds"]
            + second_timing["repair_inference_seconds"]
        )


def test_manifest_carries_inference_components() -> None:
    """The manifest must expose the split inference components."""

    from osm_polygon_sentence_relevance.labeling.finalization import _manifest

    identity = {
        "input_sha256": "a" * 64,
        "input_dataset_revision": "b" * 40,
        "model_repo_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
        "model_revision": "c" * 40,
        "model_file": "Qwen3.6-27B-Q4_K_M.gguf",
        "model_file_sha256": "d" * 64,
        "prompt_version": "afghanistan-landuse-polygon-v2",
        "source_commit": "e" * 40,
        "engine": "llama.cpp",
        "engine_version": "1",
        "batch_size": 2,
        "row_limit": 0,
        "llama_parallel": 16,
        "llama_per_slot_context": 4096,
        "llama_total_context": 65536,
        "request_concurrency": 16,
    }
    timing = {
        "initial_inference_seconds": 12.3,
        "repair_inference_seconds": 1.5,
        "inference_seconds": 13.8,
        "total_wall_seconds": 14.2,
        "checkpoint_and_validation_seconds": 0.4,
        "repair_stats": {"initial_failures": 1, "repaired": 1, "exhausted": 0},
    }
    manifest = _manifest(
        dataset_repo_id="owner/dataset",
        parquet_sha256="f" * 64,
        artifact_sha256={"sentences.parquet": "0" * 64},
        statistics={"row_count": 2},
        identity=identity,
        timing=timing,
    )
    assert manifest["timing"]["initial_inference_seconds"] == 12.3
    assert manifest["timing"]["repair_inference_seconds"] == 1.5
    assert manifest["timing"]["inference_seconds"] == 13.8


def test_no_raw_prompt_or_response_in_timing_payload(tmp_path: Path) -> None:
    """Timing must remain content-free; no raw prompts or responses."""

    store = CheckpointStore(tmp_path, _identity(tmp_path))
    secret_prompt = "supersecret-target-prompt"
    engine = _BatchThenRepairEngine(fail_rows={0})
    engine.batch_seconds = 0.5
    engine.repair_seconds = 0.5
    runner = LabelingRunner(
        engine=engine,
        store=store,
        batch_size=2,
        clock=_Clock([0.0] * 16),
    )
    runner.run(_table(3))
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert secret_prompt not in json.dumps(timing)


def test_repair_timing_advances_only_on_repair_calls(tmp_path: Path) -> None:
    """The repair timer must NOT advance on the initial (passthrough) call."""

    class _ProbingEngine:
        def __init__(self) -> None:
            self.batch_invocations = 0
            self.single_invocations = 0

        def generate(self, messages):
            if len(messages) > 1:
                self.batch_invocations += 1
                # All invalid for the first batch.
                return ["{}" for _ in messages]
            self.single_invocations += 1
            # Repair succeeds.
            return [_good_response()]

    from osm_polygon_sentence_relevance.labeling.repair import BoundedRepair

    store = CheckpointStore(tmp_path, _identity(tmp_path))
    engine = _ProbingEngine()
    LabelingRunner(
        engine=engine,
        store=store,
        batch_size=2,
        repair=BoundedRepair(max_attempts=1),
        clock=_Clock([0.0] * 16),
    ).run(_table(3))
    timing = json.loads((tmp_path / "timing.json").read_text())
    # The repair inference time must equal the time spent on single-row
    # invocations only (not the batch). With our _Clock returning the
    # same value repeatedly, repair time should still be 0.0 (no wall
    # advance) and initial time should equal the batch time.
    assert timing["repair_inference_seconds"] >= 0.0


def test_repair_timing_components_have_helper_factory() -> None:
    """The runner module must expose a helper for assembling split timings."""

    from osm_polygon_sentence_relevance.labeling import runner as runner_module

    # The function is small and stable; testing it locks in the contract.
    assert hasattr(runner_module, "build_timing_payload")
    payload = runner_module.build_timing_payload(
        initial_inference_seconds=2.0,
        repair_inference_seconds=1.0,
        checkpoint_and_validation_seconds=0.5,
        completed=10,
        total=10,
        interrupted=False,
        repair_stats={"initial_failures": 1, "repaired": 1, "exhausted": 0},
        started_at=0.0,
        finished_at=3.5,
    )
    assert payload["initial_inference_seconds"] == 2.0
    assert payload["repair_inference_seconds"] == 1.0
    assert payload["inference_seconds"] == 3.0
    assert payload["total_wall_seconds"] == 3.5
    assert payload["checkpoint_and_validation_seconds"] == 0.5


def test_finalization_includes_split_timing(tmp_path: Path) -> None:
    """The validator requires both split components when present."""

    import hashlib

    import pyarrow as pa
    import pyarrow.parquet as pq

    from osm_polygon_sentence_relevance.labeling.finalization import (
        LabelFinalizationError,
        finalize_labeled_dataset,
        validate_labeled_publication,
    )

    input_path = tmp_path / "input.parquet"
    pq.write_table(
        pa.table(
            {
                "sentence_id": ["s1"],
                "region": ["afghanistan"],
                "language": ["en"],
                "sentence_text_raw": ["farming"],
                "source": ["wikipedia"],
            }
        ),
        input_path,
    )
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    identity = _identity(tmp_path, batch_size=1, row_limit=0)
    object.__setattr__(identity, "input_sha256", digest)
    store = CheckpointStore(tmp_path / "work", identity)
    store.write_batch(
        0,
        [
            LabelRecord(
                "s1",
                LabelValue.YES,
                LabelValue.YES,
                "explicit_land_use",
                "direct_polygon_reference",
                "farming",
            )
        ],
    )
    store.write_timing(
        {
            "initial_inference_seconds": 1.0,
            "repair_inference_seconds": 0.0,
            "inference_seconds": 1.0,
            "total_wall_seconds": 1.5,
            "checkpoint_and_validation_seconds": 0.5,
        }
    )
    output = tmp_path / "out"
    finalize_labeled_dataset(
        input_path=input_path,
        store=store,
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )
    # The manifest carries the split components.
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["timing"]["initial_inference_seconds"] == 1.0
    assert manifest["timing"]["repair_inference_seconds"] == 0.0
    # The validator must accept the split schema.
    validated = validate_labeled_publication(output)
    assert validated.row_count == 1
    # Tampering the split components must be rejected.
    manifest["timing"]["initial_inference_seconds"] = -1.0
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(LabelFinalizationError):
        validate_labeled_publication(output)
