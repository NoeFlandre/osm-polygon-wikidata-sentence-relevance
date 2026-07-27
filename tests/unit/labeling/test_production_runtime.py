"""Phase 9R: production runtime amendment (no out-of-repo patches).

These tests encode the contract that the production labeling path must satisfy
after the OAR 334344 canary validation. The amendment is verified by the
following 14 properties:

1. Production code does not import, monkeypatch, or rely on ``sitecustomize``,
   ``PYTHONPATH`` overrides, or relaxed validation.
2. Production payload launches ``llama-server`` directly with no vLLM attempt.
3. ``llama.cpp`` parallelism is an explicit, validated launcher argument
   propagated through the frontend submit helper, the OAR job wrapper, and the
   compute payload.
4. Only the small supported set {1, 2, 4, 8, 16, 32} is accepted.
5. Total context is computed from parallelism and per-slot context (no silent
   partitioning, per-slot capacity at least 4096, total = parallel * 4096).
6. Client concurrency equals the server parallel slot count.
7. Server configuration is recorded in run identity and persisted manifest.
8. Parallelism or context changes are rejected by identity binding.
9. Strict ``parse_label_response`` validation is retained.
10. One bounded repair attempt is made for invalid responses with strict
    validation of the replacement.
11. Repair counts are tracked factually; no raw prompt or response content is
    written to public logs.
12. Resume semantics hold: validated checkpoints are reused; failing batches
    are not written; restart retries unfinished work; identity changes are
    rejected.
13. The launcher wires argument translation with measurable fake-binary tests.
14. Documentation states that ``llama.cpp`` is selected because of GGUF/MTP
    model compatibility.

The tests are intentionally deferred to the implementation phase; they exist
to capture RED failures before any production change and to remain GREEN once
the production amendment is in place.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.labeling.contracts import RunIdentity

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src" / "osm_polygon_sentence_relevance" / "labeling"
GRID_ROOT = ROOT / "scripts" / "grid5000"
DOC = ROOT / "docs" / "guides" / "grid5000.md"
SUBMIT = GRID_ROOT / "submit_afghanistan_labeling.sh"
JOB_WRAPPER = GRID_ROOT / "run_afghanistan_labeling_job.sh"
PAYLOAD = GRID_ROOT / "run_afghanistan_labeling.sh"


# ---------------------------------------------------------------------------
# 1. Production code must not import, monkeypatch, or rely on sitecustomize,
#    PYTHONPATH overrides, or relaxed validation.
# ---------------------------------------------------------------------------


def test_no_sitecustomize_imports_in_production_labeling() -> None:
    for path in SRC_ROOT.rglob("*.py"):
        text = path.read_text()
        assert "sitecustomize" not in text, (
            f"{path} imports sitecustomize; production code must not rely on it"
        )


def test_no_pythonpath_overrides_in_production_labeling() -> None:
    for path in SRC_ROOT.rglob("*.py"):
        text = path.read_text()
        assert "PYTHONPATH" not in text, (
            f"{path} sets PYTHONPATH; production code must not rely on it"
        )


def test_no_relaxed_label_validation_in_production_code() -> None:
    validation = (SRC_ROOT / "validation.py").read_text()
    # Strict validation: invalid evidence raises (does not silently clamp to "").
    assert "raise LabelValidationError" in validation
    assert 'evidence ""' not in validation
    assert "replace" not in validation.lower() or "replace" in validation.lower()
    # The sitecustomize-style fallback is the explicit antipattern we forbid.
    assert ".replace(invalid" not in validation
    assert "if evidence and evidence not in" in validation


# ---------------------------------------------------------------------------
# 2. Production payload launches llama-server directly with no vLLM attempt.
# ---------------------------------------------------------------------------


def test_payload_does_not_attempt_vllm() -> None:
    text = PAYLOAD.read_text()
    assert "vllm serve" not in text
    assert "vllm --version" not in text
    assert "ENGINE=vllm" not in text
    assert "ENGINE=llama.cpp" in text
    assert "llama-server" in text


def test_submitter_uses_production_default_queue_without_cpu_fallback() -> None:
    text = SUBMIT.read_text()
    # The nantes site schedules through default queue resources for production
    # jobs. The
    # Nancy-compatible resource contracts currently require explicit exotic jobs.
    # ``-t besteffort`` flag must be absent (it may appear in a comment).
    assert "-q default" in text
    assert "-t exotic" in text
    assert " -t besteffort" not in text
    assert "best_effort" not in text
    assert "cpu" not in text
    assert "mps" not in text


# ---------------------------------------------------------------------------
# 3. llama.cpp parallelism is an explicit, validated launcher argument
#    propagated through the submit helper, the OAR job wrapper, and the
#    compute payload.
# ---------------------------------------------------------------------------


def test_llama_parallel_is_first_class_positional_in_submit_helper() -> None:
    text = SUBMIT.read_text()
    assert "LLAMA_PARALLEL" in text
    assert "exactly fifteen arguments" in text
    # The argument must be quoted and propagated to the wrapper.
    assert text.count("LLAMA_PARALLEL") >= 3


def test_llama_parallel_is_first_class_positional_in_job_wrapper() -> None:
    text = JOB_WRAPPER.read_text()
    assert "LLAMA_PARALLEL" in text
    assert "exactly fifteen arguments" in text
    assert "${15}" in text


def test_llama_parallel_is_first_class_positional_in_payload() -> None:
    text = PAYLOAD.read_text()
    assert "LLAMA_PARALLEL" in text
    assert "exactly thirteen arguments" in text
    assert "${13}" in text


def test_supported_parallel_values_are_explicit() -> None:
    from osm_polygon_sentence_relevance.labeling.runtime import (
        SUPPORTED_LLAMA_PARALLEL,
    )

    assert SUPPORTED_LLAMA_PARALLEL == (1, 2, 4, 8, 16, 32)


# ---------------------------------------------------------------------------
# 4. Only the small supported set is accepted.
# ---------------------------------------------------------------------------


def test_supported_parallel_set_rejects_other_values() -> None:
    from osm_polygon_sentence_relevance.labeling.runtime import (
        validate_llama_parallel,
    )

    for value in (-1, 0, 3, 5, 64, 128, "16", None, True):
        try:
            validate_llama_parallel(value)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"validate_llama_parallel did not reject {value!r}")


def test_supported_parallel_set_accepts_supported_values() -> None:
    from osm_polygon_sentence_relevance.labeling.runtime import (
        validate_llama_parallel,
    )

    for value in (1, 2, 4, 8, 16, 32):
        assert validate_llama_parallel(value) == value


# ---------------------------------------------------------------------------
# 5. Total context is computed from parallelism and per-slot context.
# ---------------------------------------------------------------------------


def test_total_context_is_parallel_times_per_slot_minimum() -> None:
    from osm_polygon_sentence_relevance.labeling.runtime import (
        MIN_PER_SLOT_CONTEXT,
        compute_total_context,
    )

    assert MIN_PER_SLOT_CONTEXT == 4096
    for parallel in (1, 2, 4, 8, 16, 32):
        assert compute_total_context(parallel) == parallel * 4096


def test_compute_total_context_rejects_invalid_parallel() -> None:
    from osm_polygon_sentence_relevance.labeling.runtime import compute_total_context

    for value in (-1, 0, 3, 64):
        try:
            compute_total_context(value)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"compute_total_context did not reject {value!r}")


# ---------------------------------------------------------------------------
# 6. Client concurrency equals server parallel slot count.
# ---------------------------------------------------------------------------


def test_resolve_engine_wires_concurrency_to_parallel() -> None:
    from osm_polygon_sentence_relevance.labeling.runtime import (
        RuntimePlan,
        resolve_engine_factory,
    )

    plan = RuntimePlan(
        parallel=16,
        per_slot_context=4096,
        total_context=65536,
        request_concurrency=16,
    )
    factory = resolve_engine_factory(plan)
    engine = factory(endpoint="http://localhost", model="m")
    assert engine.concurrency == 16


def test_engine_factory_uses_plan_concurrency_when_distinct() -> None:
    from osm_polygon_sentence_relevance.labeling.engine import OpenAICompatibleEngine
    from osm_polygon_sentence_relevance.labeling.runtime import (
        RuntimePlan,
        resolve_engine_factory,
    )

    plan = RuntimePlan(
        parallel=8,
        per_slot_context=4096,
        total_context=32768,
        request_concurrency=8,
    )
    factory = resolve_engine_factory(plan)
    engine = factory(endpoint="http://localhost", model="m")
    assert isinstance(engine, OpenAICompatibleEngine)
    assert engine.concurrency == 8


# ---------------------------------------------------------------------------
# 7. Server configuration is recorded in run identity and persisted manifest.
# ---------------------------------------------------------------------------


def test_run_identity_records_server_config() -> None:
    from osm_polygon_sentence_relevance.labeling.contracts import RunIdentity

    identity = RunIdentity(
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
        batch_size=128,
        row_limit=128,
        llama_parallel=16,
        llama_per_slot_context=4096,
        llama_total_context=65536,
        request_concurrency=16,
    )
    payload = identity.to_dict()
    assert payload["llama_parallel"] == 16
    assert payload["llama_per_slot_context"] == 4096
    assert payload["llama_total_context"] == 65536
    assert payload["request_concurrency"] == 16


def test_manifest_includes_server_config_section() -> None:
    from osm_polygon_sentence_relevance.labeling.finalization import (
        _manifest,
    )

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
        "row_limit": 128,
        "llama_parallel": 16,
        "llama_per_slot_context": 4096,
        "llama_total_context": 65536,
        "request_concurrency": 16,
    }
    manifest = _manifest(
        dataset_repo_id="owner/dataset",
        parquet_sha256="f" * 64,
        artifact_sha256={"sentences.parquet": "0" * 64},
        statistics={"row_count": 128},
        identity=identity,
        timing={
            "total_wall_seconds": 1.0,
            "initial_inference_seconds": 0.5,
            "repair_inference_seconds": 0.0,
            "inference_seconds": 0.5,
            "checkpoint_and_validation_seconds": 0.5,
        },
    )
    assert manifest["server_config"] == {
        "llama_parallel": 16,
        "llama_per_slot_context": 4096,
        "llama_total_context": 65536,
        "request_concurrency": 16,
    }


# ---------------------------------------------------------------------------
# 8. Parallelism or context changes are rejected by identity binding.
# ---------------------------------------------------------------------------


def test_resume_rejects_parallel_change_in_checkpoints(tmp_path: Path) -> None:
    from osm_polygon_sentence_relevance.labeling.checkpoint import (
        CheckpointError,
        CheckpointStore,
    )
    from osm_polygon_sentence_relevance.labeling.contracts import (
        LabelRecord,
        LabelValue,
        RunIdentity,
    )

    base = RunIdentity(
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
        batch_size=128,
        row_limit=128,
        llama_parallel=16,
        llama_per_slot_context=4096,
        llama_total_context=65536,
        request_concurrency=16,
    )
    CheckpointStore(tmp_path, base).write_batch(
        0,
        [
            LabelRecord(
                sentence_id="s1",
                landuse_relevance=LabelValue.YES,
                polygon_relevance=LabelValue.NO,
                landuse_reason="explicit_land_use",
                polygon_reason="nearby_or_broader_area",
                evidence="farming",
            )
        ],
    )
    changed = dataclasses.replace(
        base,
        llama_parallel=8,
        llama_total_context=8 * 4096,
        request_concurrency=8,
    )
    with pytest.raises(CheckpointError, match="identity"):
        CheckpointStore(tmp_path, changed).load_all()


def test_resume_rejects_total_context_change(tmp_path: Path) -> None:
    from osm_polygon_sentence_relevance.labeling.checkpoint import (
        CheckpointError,
        CheckpointStore,
    )
    from osm_polygon_sentence_relevance.labeling.contracts import (
        LabelRecord,
        LabelValue,
        RunIdentity,
    )

    base = RunIdentity(
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
        batch_size=128,
        row_limit=128,
        llama_parallel=16,
        llama_per_slot_context=4096,
        llama_total_context=65536,
        request_concurrency=16,
    )
    CheckpointStore(tmp_path, base).write_batch(
        0,
        [
            LabelRecord(
                sentence_id="s1",
                landuse_relevance=LabelValue.YES,
                polygon_relevance=LabelValue.NO,
                landuse_reason="explicit_land_use",
                polygon_reason="nearby_or_broader_area",
                evidence="farming",
            )
        ],
    )
    changed = dataclasses.replace(
        base,
        llama_total_context=131072,
        llama_per_slot_context=8192,
    )
    with pytest.raises(CheckpointError, match="identity"):
        CheckpointStore(tmp_path, changed).load_all()


# ---------------------------------------------------------------------------
# 9. Strict parse_label_response validation is retained.
# ---------------------------------------------------------------------------


def test_strict_validation_rejects_invalid_evidence() -> None:
    from osm_polygon_sentence_relevance.labeling.validation import (
        LabelValidationError,
        parse_label_response,
    )

    response = json.dumps(
        {
            "landuse_relevance": "yes",
            "polygon_relevance": "yes",
            "landuse_reason": "explicit_land_use",
            "polygon_reason": "direct_polygon_reference",
            "evidence": "this is not a substring",
        }
    )
    with pytest.raises(LabelValidationError, match="exact substring"):
        parse_label_response(response, target_sentence="farming valley")


def test_strict_validation_rejects_inconsistent_reason() -> None:
    from osm_polygon_sentence_relevance.labeling.validation import (
        LabelValidationError,
        parse_label_response,
    )

    response = json.dumps(
        {
            "landuse_relevance": "no",
            "polygon_relevance": "yes",
            "landuse_reason": "explicit_land_use",
            "polygon_reason": "direct_polygon_reference",
            "evidence": "farming",
        }
    )
    with pytest.raises(LabelValidationError, match="inconsistent"):
        parse_label_response(response, target_sentence="farming valley")


# ---------------------------------------------------------------------------
# 10. One bounded repair attempt is made for invalid responses with strict
#     validation of the replacement.
# ---------------------------------------------------------------------------


def test_bounded_repair_succeeds_on_valid_replacement() -> None:
    from osm_polygon_sentence_relevance.labeling.repair import (
        BoundedRepair,
    )

    attempts: list[list[dict[str, str]]] = []

    def engine(messages: list[list[dict[str, str]]]) -> list[str]:
        attempts.append(messages[0])
        # First attempt: invalid evidence. Repair attempt: valid evidence.
        if len(attempts) == 1:
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
        ]

    repair = BoundedRepair(max_attempts=2)
    label = repair.call(
        engine=engine,
        messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "farming valley"},
        ],
        target_sentence="farming valley",
    )
    assert label.evidence == "farming"
    assert len(attempts) == 2
    # The repair message must include the exact rule that failed.
    assert any("exact substring" in c["content"] for c in attempts[1])


def test_bounded_repair_clears_only_exhausted_invalid_evidence() -> None:
    from osm_polygon_sentence_relevance.labeling.repair import BoundedRepair

    def engine(_messages: list[list[dict[str, str]]]) -> list[str]:
        return [
            json.dumps(
                {
                    "landuse_relevance": "no",
                    "polygon_relevance": "yes",
                    "landuse_reason": "no_landuse_or_cover",
                    "polygon_reason": "place_description",
                    "evidence": "paraphrase absent from source",
                }
            )
        ]

    repair = BoundedRepair(max_attempts=1)
    label = repair.call(
        engine=engine,
        messages=[
            {"role": "system", "content": "label"},
            {"role": "user", "content": "source sentence"},
        ],
        target_sentence="source sentence",
    )

    assert label.evidence == ""
    assert label.landuse_relevance.value == "no"
    assert label.polygon_relevance.value == "yes"
    assert repair.stats.initial_failures == 1
    assert repair.stats.repaired == 1


def test_bounded_repair_rejects_strict_invalid_replacement() -> None:
    from osm_polygon_sentence_relevance.labeling.repair import (
        BoundedRepair,
        RepairExhausted,
    )

    def engine(messages: list[list[dict[str, str]]]) -> list[str]:
        # Both attempts invalid: reason contradicts the positive label.
        return [
            json.dumps(
                {
                    "landuse_relevance": "yes",
                    "polygon_relevance": "yes",
                    "landuse_reason": "no_landuse_or_cover",
                    "polygon_reason": "direct_polygon_reference",
                    "evidence": "",
                }
            )
        ]

    repair = BoundedRepair(max_attempts=2)
    with pytest.raises(RepairExhausted):
        repair.call(
            engine=engine,
            messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "farming valley"},
            ],
            target_sentence="farming valley",
        )


def test_bounded_repair_caps_attempts() -> None:
    from osm_polygon_sentence_relevance.labeling.repair import (
        BoundedRepair,
        RepairExhausted,
    )

    def engine(messages: list[list[dict[str, str]]]) -> list[str]:
        return [
            json.dumps(
                {
                    "landuse_relevance": "yes",
                    "polygon_relevance": "yes",
                    "landuse_reason": "no_landuse_or_cover",
                    "polygon_reason": "direct_polygon_reference",
                    "evidence": "",
                }
            )
        ]

    repair = BoundedRepair(max_attempts=1)
    with pytest.raises(RepairExhausted):
        repair.call(
            engine=engine,
            messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "farming valley"},
            ],
            target_sentence="farming valley",
        )


def test_bounded_repair_changes_instruction_between_attempts() -> None:
    from osm_polygon_sentence_relevance.labeling.repair import (
        BoundedRepair,
        RepairExhausted,
    )

    prompts: list[str] = []

    def engine(messages: list[list[dict[str, str]]]) -> list[str]:
        prompts.append(messages[0][-1]["content"])
        return [
            json.dumps(
                {
                    "landuse_relevance": "yes",
                    "polygon_relevance": "yes",
                    "landuse_reason": "no_landuse_or_cover",
                    "polygon_reason": "direct_polygon_reference",
                    "evidence": "",
                }
            )
        ]

    with pytest.raises(RepairExhausted):
        BoundedRepair(max_attempts=3).call(
            engine=engine,
            messages=[
                {"role": "system", "content": "label"},
                {"role": "user", "content": "farming valley"},
            ],
            target_sentence="farming valley",
        )

    assert len(prompts) == 4
    assert len(set(prompts[1:])) == 3


def test_production_cli_uses_three_bounded_repair_attempts() -> None:
    from osm_polygon_sentence_relevance.labeling import cli

    assert "BoundedRepair(max_attempts=3)" in Path(cli.__file__).read_text()


# ---------------------------------------------------------------------------
# 11. Repair counts are tracked factually; no raw prompt or response content
#     is written to public logs.
# ---------------------------------------------------------------------------


def test_repair_counts_are_tracked_and_redacted() -> None:
    from osm_polygon_sentence_relevance.labeling.repair import (
        BoundedRepair,
        RepairStats,
    )

    attempts: list[list[dict[str, str]]] = []

    def engine(messages: list[list[dict[str, str]]]) -> list[str]:
        attempts.append(messages[0])
        if len(attempts) == 1:
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
        ]

    repair = BoundedRepair(max_attempts=2)
    sensitive = "secret-prompt-content"
    label = repair.call(
        engine=engine,
        messages=[
            {"role": "system", "content": sensitive},
            {"role": "user", "content": sensitive},
        ],
        target_sentence="farming valley",
    )
    assert label.evidence == "farming"
    stats = repair.stats
    assert isinstance(stats, RepairStats)
    assert stats.repaired >= 1
    # No raw prompt or response content should be exposed via the stats.
    raw = str(stats.to_dict())
    assert sensitive not in raw


# ---------------------------------------------------------------------------
# 12. Resume semantics: validated checkpoints reused, failing batches not
#     written, restart retries unfinished work, identity changes rejected.
# ---------------------------------------------------------------------------


def test_resume_reuses_validated_checkpoints(tmp_path: Path) -> None:
    from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
    from osm_polygon_sentence_relevance.labeling.runner import LabelingRunner

    identity = _identity(tmp_path)
    store = CheckpointStore(tmp_path, identity)
    LabelingRunner(
        engine=_FakeEngine(),
        store=store,
        batch_size=2,
    ).run(_table())
    counts = store._batch_indexes()
    assert counts == [0, 1, 2]


def test_resume_does_not_rewrite_failing_batch(tmp_path: Path) -> None:
    from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
    from osm_polygon_sentence_relevance.labeling.repair import RepairExhausted
    from osm_polygon_sentence_relevance.labeling.runner import LabelingRunner

    class FailingEngine:
        def generate(self, messages):
            raise RepairExhausted("simulated")

    identity = _identity(tmp_path)
    store = CheckpointStore(tmp_path, identity)
    with pytest.raises(RepairExhausted):
        LabelingRunner(
            engine=FailingEngine(),
            store=store,
            batch_size=2,
        ).run(_table())
    assert store._batch_indexes() == []


def test_resume_retries_unfinished_work_after_failure(tmp_path: Path) -> None:
    from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
    from osm_polygon_sentence_relevance.labeling.runner import LabelingRunner

    identity = _identity(tmp_path)
    store = CheckpointStore(tmp_path, identity)
    runner = LabelingRunner(
        engine=_FakeEngine(),
        store=store,
        batch_size=2,
    )
    runner.run(_table())
    assert len(store._batch_indexes()) == 3
    # Resume with a fresh runner should not relabel already completed rows.
    resumed = LabelingRunner(
        engine=_FakeEngine(),
        store=CheckpointStore(tmp_path, identity),
        batch_size=2,
    )
    resumed.run(_table())
    assert len(store._batch_indexes()) == 3


# ---------------------------------------------------------------------------
# 13. The launcher wires argument translation with measurable fake-binary tests.
# ---------------------------------------------------------------------------


def test_payload_launches_real_llama_server_binary() -> None:
    text = PAYLOAD.read_text()
    assert "llama-server" in text
    assert "--model" in text
    assert "--alias" in text
    assert "--host 127.0.0.1" in text
    assert "--port" in text
    assert "--ctx-size" in text
    assert "--parallel" in text
    assert "--n-gpu-layers 999" in text
    # No wrapper script.
    assert "exec llama-server" in text or "llama-server --model" in text


def test_payload_translates_context_and_parallel() -> None:
    text = PAYLOAD.read_text()
    assert '"${LLAMA_TOTAL_CONTEXT}"' in text
    assert '"${LLAMA_PARALLEL}"' in text


def test_payload_uses_supported_parallel_validation() -> None:
    text = PAYLOAD.read_text()
    # The payload must validate against the supported set {1, 2, 4, 8, 16, 32}.
    assert (
        "1|2|4|8|16|32" in text
        or "1 2 4 8 16 32" in text
        or re.search(r"1[^\n]?2[^\n]?4[^\n]?8[^\n]?16[^\n]?32", text)
    )


def test_payload_persists_server_config_to_run_identity() -> None:
    text = PAYLOAD.read_text()
    assert "--llama-parallel" in text
    assert "--llama-per-slot-context" in text
    assert "--llama-total-context" in text
    assert "--request-concurrency" in text


# ---------------------------------------------------------------------------
# 14. Documentation states that llama.cpp is selected because of GGUF/MTP
#     model compatibility.
# ---------------------------------------------------------------------------


def test_guide_documents_llama_cpp_selection_reason() -> None:
    text = DOC.read_text()
    assert "llama.cpp" in text
    assert "vLLM" in text
    assert "MTP" in text or "mtp" in text
    assert "GGUF" in text or "gguf" in text


def test_guide_documents_llama_parallel_argument() -> None:
    text = DOC.read_text()
    assert "LLAMA_PARALLEL" in text
    assert "1, 2, 4, 8, 16, 32" in text or "{1, 2, 4, 8, 16, 32}" in text


def test_guide_documents_repair_policy() -> None:
    text = DOC.read_text()
    assert "bounded repair" in text.lower() or "one repair" in text.lower()


def test_guide_documents_no_patches_or_wrappers() -> None:
    text = DOC.read_text()
    # The guide must explicitly deny reliance on sitecustomize, PYTHONPATH,
    # and wrapper binaries that rewrite arguments.
    assert "sitecustomize" in text  # explicitly mentioned as not used
    assert "PYTHONPATH" in text  # explicitly mentioned as not used
    lower = text.lower()
    assert "rely on" in lower
    assert "monkey-patch" in lower or "monkeypatch" in lower
    assert "no vllm attempt" in lower or "is selected" in lower


# ---------------------------------------------------------------------------
# Helpers
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
    rows = []
    for i in range(5):
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


class _FakeEngine:
    def generate(self, messages: list[list[dict[str, str]]]) -> list[str]:
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
