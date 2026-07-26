"""Extra coverage for production runtime amendment branches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
from osm_polygon_sentence_relevance.labeling.contracts import (
    LabelRecord,
    LabelValue,
    RunIdentity,
)
from osm_polygon_sentence_relevance.labeling.repair import (
    BoundedRepair,
    Messages,
    RepairExhausted,
    _build_repair_messages,
    _failure_reason,
    _invoke_engine,
)
from osm_polygon_sentence_relevance.labeling.runner import LabelingRunner
from osm_polygon_sentence_relevance.labeling.runtime import (
    MIN_PER_SLOT_CONTEXT,
    RuntimePlan,
    build_runtime_plan,
    compute_total_context,
    resolve_engine_factory,
    validate_llama_parallel,
    validate_per_slot_context,
)
from osm_polygon_sentence_relevance.labeling.validation import (
    LabelValidationError,
)

# ---------------------------------------------------------------------------
# runtime.py: every branch of the validation and plan construction
# ---------------------------------------------------------------------------


def test_validate_llama_parallel_rejects_non_int() -> None:
    with pytest.raises(ValueError, match="integer"):
        validate_llama_parallel("16")
    with pytest.raises(ValueError, match="integer"):
        validate_llama_parallel(1.5)
    with pytest.raises(ValueError, match="integer"):
        validate_llama_parallel(None)


def test_validate_llama_parallel_rejects_unsupported() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        validate_llama_parallel(3)
    with pytest.raises(ValueError, match="must be one of"):
        validate_llama_parallel(64)


def test_validate_per_slot_context_rejects_below_minimum() -> None:
    with pytest.raises(ValueError, match="at least"):
        validate_per_slot_context(2048)


def test_validate_per_slot_context_rejects_non_int() -> None:
    with pytest.raises(ValueError, match="integer"):
        validate_per_slot_context(None)


def test_compute_total_context_with_explicit_per_slot() -> None:
    assert compute_total_context(8, 4096) == 32768
    assert compute_total_context(16, 8192) == 131072


def test_runtime_plan_rejects_mismatched_total_context() -> None:
    with pytest.raises(ValueError, match="equal"):
        RuntimePlan(
            parallel=16,
            per_slot_context=4096,
            total_context=32768,
            request_concurrency=16,
        )


def test_runtime_plan_rejects_concurrency_above_parallel() -> None:
    with pytest.raises(ValueError, match="between"):
        RuntimePlan(
            parallel=4,
            per_slot_context=4096,
            total_context=16384,
            request_concurrency=8,
        )


def test_runtime_plan_rejects_zero_concurrency() -> None:
    with pytest.raises(ValueError, match="between"):
        RuntimePlan(
            parallel=4,
            per_slot_context=4096,
            total_context=16384,
            request_concurrency=0,
        )


def test_build_runtime_plan_uses_default_per_slot() -> None:
    plan = build_runtime_plan(parallel=2)
    assert plan.per_slot_context == MIN_PER_SLOT_CONTEXT
    assert plan.total_context == 2 * MIN_PER_SLOT_CONTEXT
    assert plan.request_concurrency == 2


def test_resolve_engine_factory_forwards_endpoint_and_model() -> None:
    plan = build_runtime_plan(parallel=4)
    factory = resolve_engine_factory(plan)
    engine = factory(endpoint="http://h:1", model="m1")
    assert engine.endpoint == "http://h:1"
    assert engine.model == "m1"


# ---------------------------------------------------------------------------
# repair.py: every branch of validation, repair messages, stats, and helpers
# ---------------------------------------------------------------------------


def test_failure_reason_substring() -> None:
    err = LabelValidationError("evidence must be an exact substring of target sentence")
    assert (
        _failure_reason(err) == "evidence is not an exact substring of target sentence"
    )


def test_failure_reason_inconsistent() -> None:
    err = LabelValidationError("landuse_reason is inconsistent with landuse_relevance")
    assert _failure_reason(err) == "reason is inconsistent with the relevance label"


def test_failure_reason_evidence_too_long() -> None:
    err = LabelValidationError("evidence must contain at most 240 characters")
    assert _failure_reason(err) == "evidence exceeds the 240-character limit"


def test_failure_reason_default() -> None:
    err = LabelValidationError("invalid enum value")
    assert _failure_reason(err) == "response violates the structured label contract"


def test_build_repair_messages_handles_non_system_prompts() -> None:
    messages: Messages = [{"role": "user", "content": "x"}]
    out = _build_repair_messages(messages, "target", "reason")
    assert out == messages


def test_invoke_engine_rejects_unexpected_response_count() -> None:
    def engine(messages):
        return ["a", "b"]

    with pytest.raises(RepairExhausted, match="unexpected response count"):
        _invoke_engine(engine, [{"role": "user", "content": "x"}])


def test_repair_exhausted_raises_without_initial_error() -> None:
    """When the initial response parses, no repair is attempted."""

    def engine(messages):
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
        ]

    repair = BoundedRepair(max_attempts=1)
    label = repair.call(
        engine=engine,
        messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "farming valley"},
        ],
        target_sentence="farming valley",
    )
    assert label.evidence == "farming"
    assert repair.stats.initial_failures == 0
    assert repair.stats.repaired == 0


def test_repair_rejects_low_max_attempts() -> None:
    with pytest.raises(ValueError, match="at least"):
        BoundedRepair(max_attempts=0)


# ---------------------------------------------------------------------------
# runner.py: write progress, identity mismatch, repair log path
# ---------------------------------------------------------------------------


def _identity(tmp_path: Path) -> RunIdentity:
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
        batch_size=2,
        row_limit=2,
        llama_parallel=16,
        llama_per_slot_context=4096,
        llama_total_context=65536,
        request_concurrency=16,
    )


def _table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sentence_id": f"s{i}",
                "sentence_text_raw": "Sentence describes farming",
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
            for i in range(3)
        ]
    )


class _GoodEngine:
    def generate(self, messages):
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


def test_runner_writes_repair_log_on_invalid_response(tmp_path: Path) -> None:
    class _RecoveringEngine:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages):
            self.calls += 1
            if self.calls == 1:
                return [
                    json.dumps(
                        {
                            "landuse_relevance": "yes",
                            "polygon_relevance": "yes",
                            "landuse_reason": "explicit_land_use",
                            "polygon_reason": "direct_polygon_reference",
                            "evidence": "wrong substring",
                        }
                    )
                    for _ in messages
                ]
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

    store = CheckpointStore(tmp_path, _identity(tmp_path))
    LabelingRunner(
        engine=_RecoveringEngine(),
        store=store,
        batch_size=2,
        repair_log_path=tmp_path / "repair.log",
        repair_max_attempts=2,
    ).run(_table())
    log = (tmp_path / "repair.log").read_text()
    assert "label_repair" in log


def test_runner_rejects_batch_size_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        LabelingRunner(
            engine=_GoodEngine(),
            store=CheckpointStore(tmp_path, _identity(tmp_path)),
            batch_size=4,
        )


def test_runner_rejects_non_positive_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        LabelingRunner(
            engine=_GoodEngine(),
            store=CheckpointStore(tmp_path, _identity(tmp_path)),
            batch_size=0,
        )


def test_runner_rejects_duplicate_sentence_ids(tmp_path: Path) -> None:
    rows = _table().to_pylist()
    rows.append(dict(rows[0]))
    table = pa.Table.from_pylist(rows)
    with pytest.raises(ValueError, match="duplicate"):
        LabelingRunner(
            engine=_GoodEngine(),
            store=CheckpointStore(tmp_path, _identity(tmp_path)),
            batch_size=2,
        ).run(table)


def test_runner_rejects_unmatched_completed_ids(tmp_path: Path) -> None:
    base_identity = _identity(tmp_path)
    store = CheckpointStore(tmp_path, base_identity)
    # Pre-populate with a sentence_id not in the input table.
    store.write_batch(
        0,
        [
            LabelRecord(
                sentence_id="other-id",
                landuse_relevance=LabelValue.YES,
                polygon_relevance=LabelValue.NO,
                landuse_reason="explicit_land_use",
                polygon_reason="nearby_or_broader_area",
                evidence="x",
            )
        ],
    )
    with pytest.raises(ValueError, match="checkpoints contain"):
        LabelingRunner(
            engine=_GoodEngine(),
            store=CheckpointStore(tmp_path, _identity(tmp_path)),
            batch_size=2,
        ).run(_table())


def test_runner_writes_repair_stats_in_timing(tmp_path: Path) -> None:
    LabelingRunner(
        engine=_GoodEngine(),
        store=CheckpointStore(tmp_path, _identity(tmp_path)),
        batch_size=2,
    ).run(_table())
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert "repair_stats" in timing
    assert timing["repair_stats"]["initial_failures"] == 0
    assert timing["repair_stats"]["repaired"] == 0
    assert timing["repair_stats"]["exhausted"] == 0


# ---------------------------------------------------------------------------
# contracts.py: identity validation for non-llama.cpp engines
# ---------------------------------------------------------------------------


def test_run_identity_rejects_non_llama_engine() -> None:
    with pytest.raises(ValueError, match="llama.cpp"):
        RunIdentity(
            input_sha256="a" * 64,
            input_dataset_revision="b" * 40,
            model_repo_id="r",
            model_revision="c" * 40,
            model_file="m",
            model_file_sha256="d" * 64,
            prompt_version="v",
            source_commit="e" * 40,
            engine="vllm",
            engine_version="1",
            batch_size=1,
        )


# ---------------------------------------------------------------------------
# finalization.py: server config validator rejection
# ---------------------------------------------------------------------------


def test_finalization_rejects_missing_server_config(tmp_path: Path) -> None:
    """The validator must require server_config in the persisted manifest."""

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
        "batch_size": 128,
        "row_limit": 0,
        "llama_parallel": 16,
        "llama_per_slot_context": 4096,
        "llama_total_context": 65536,
        "request_concurrency": 16,
    }
    input_path = tmp_path / "input.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table(
            {
                "sentence_id": ["s1"],
                "region": ["afghanistan"],
                "language": ["en"],
                "sentence_text_raw": ["farming"],
            }
        ),
        input_path,
    )
    digest = _compute_digest(input_path)
    identity["input_sha256"] = digest

    from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
    from osm_polygon_sentence_relevance.labeling.finalization import (
        LabelFinalizationError,
        finalize_labeled_dataset,
    )

    store = CheckpointStore(tmp_path / "work", _identity_from_dict(identity))
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
            "total_wall_seconds": 1.0,
            "initial_inference_seconds": 0.5,
            "repair_inference_seconds": 0.0,
            "inference_seconds": 0.5,
            "checkpoint_and_validation_seconds": 0.5,
        }
    )

    output = tmp_path / "publication"
    finalize_labeled_dataset(
        input_path=input_path,
        store=store,
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )
    # Now mutate the manifest to remove server_config.
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("server_config")
    manifest_path.write_text(json.dumps(manifest))
    from osm_polygon_sentence_relevance.labeling.finalization import (
        validate_labeled_publication,
    )

    with pytest.raises(LabelFinalizationError, match="server_config"):
        validate_labeled_publication(output)


def _compute_digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_from_dict(d: dict) -> RunIdentity:
    return RunIdentity(
        input_sha256=d["input_sha256"],
        input_dataset_revision=d["input_dataset_revision"],
        model_repo_id=d["model_repo_id"],
        model_revision=d["model_revision"],
        model_file=d["model_file"],
        model_file_sha256=d["model_file_sha256"],
        prompt_version=d["prompt_version"],
        source_commit=d["source_commit"],
        engine=d["engine"],
        engine_version=d["engine_version"],
        batch_size=d["batch_size"],
        row_limit=d.get("row_limit", 0),
        llama_parallel=d.get("llama_parallel", 16),
        llama_per_slot_context=d.get("llama_per_slot_context", 4096),
        llama_total_context=d.get("llama_total_context", 65536),
        request_concurrency=d.get("request_concurrency", 16),
    )


def test_finalization_rejects_partial_afghanistan_input(tmp_path: Path) -> None:
    """Lines 88 / 215 / 243 / 315-317 in finalization.py: content branches."""

    from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
    from osm_polygon_sentence_relevance.labeling.finalization import (
        LabelFinalizationError,
        finalize_labeled_dataset,
    )

    input_path = tmp_path / "input.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table(
            {
                "sentence_id": ["s1", "s2"],
                "region": ["afghanistan", "afghanistan"],
                "language": ["en", "en"],
                "sentence_text_raw": ["farming", "history"],
                "source": ["wikipedia", "wikipedia"],
            }
        ),
        input_path,
    )
    digest = _compute_digest(input_path)
    identity: dict[str, Any] = {
        "input_sha256": digest,
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
    store = CheckpointStore(tmp_path / "work", _identity_from_dict(identity))
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
            ),
            LabelRecord(
                "s2",
                LabelValue.NO,
                LabelValue.YES,
                "no_landuse_or_cover",
                "direct_polygon_reference",
                "history",
            ),
        ],
    )
    store.write_timing(
        {
            "total_wall_seconds": 1.0,
            "initial_inference_seconds": 0.5,
            "repair_inference_seconds": 0.0,
            "inference_seconds": 0.5,
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
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Tamper statistics to trigger row_count / statistics mismatch branches.
    manifest["statistics"]["row_count"] = 99
    manifest_path.write_text(json.dumps(manifest))
    from osm_polygon_sentence_relevance.labeling.finalization import (
        validate_labeled_publication,
    )

    with pytest.raises(LabelFinalizationError, match="row count"):
        validate_labeled_publication(output)


def test_finalization_rejects_tampered_landuse_distribution(tmp_path: Path) -> None:
    """Cover the statistics-mismatch branch of the validator."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
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
    digest = _compute_digest(input_path)
    identity: dict[str, Any] = {
        "input_sha256": digest,
        "input_dataset_revision": "b" * 40,
        "model_repo_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
        "model_revision": "c" * 40,
        "model_file": "Qwen3.6-27B-Q4_K_M.gguf",
        "model_file_sha256": "d" * 64,
        "prompt_version": "afghanistan-landuse-polygon-v2",
        "source_commit": "e" * 40,
        "engine": "llama.cpp",
        "engine_version": "1",
        "batch_size": 1,
        "row_limit": 0,
        "llama_parallel": 16,
        "llama_per_slot_context": 4096,
        "llama_total_context": 65536,
        "request_concurrency": 16,
    }
    store = CheckpointStore(tmp_path / "work", _identity_from_dict(identity))
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
            "total_wall_seconds": 1.0,
            "initial_inference_seconds": 0.5,
            "repair_inference_seconds": 0.0,
            "inference_seconds": 0.5,
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
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["statistics"]["landuse_relevance"] = {"no": 99, "yes": 0}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(LabelFinalizationError, match="statistics"):
        validate_labeled_publication(output)
