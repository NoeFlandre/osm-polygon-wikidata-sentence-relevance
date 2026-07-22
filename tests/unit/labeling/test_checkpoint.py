from __future__ import annotations

import json
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from osm_polygon_sentence_relevance.labeling.checkpoint import (
    CheckpointError,
    CheckpointStore,
)
from osm_polygon_sentence_relevance.labeling.contracts import (
    LabelRecord,
    LabelValue,
    RunIdentity,
)


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
        batch_size=128,
    )


def _record(sentence_id: str = "s1") -> LabelRecord:
    return LabelRecord(
        sentence_id=sentence_id,
        landuse_relevance=LabelValue.YES,
        polygon_relevance=LabelValue.NO,
        landuse_reason="explicit_land_use",
        polygon_reason="nearby_or_broader_area",
        evidence="farming",
    )


def test_writes_and_loads_atomic_checkpoint(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    store.write_batch(0, [_record()])

    assert store.load_all() == [_record()]
    assert store.completed_ids() == {"s1"}
    assert stat.S_IMODE((tmp_path / "checkpoints").stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in (tmp_path / "checkpoints").iterdir()
    )
    assert not list(tmp_path.rglob("*.tmp"))


def test_resume_rejects_identity_mismatch(tmp_path: Path) -> None:
    CheckpointStore(tmp_path, _identity()).write_batch(0, [_record()])
    changed = replace(_identity(), engine="llama.cpp")
    with pytest.raises(CheckpointError, match="identity"):
        CheckpointStore(tmp_path, changed).load_all()


def test_rejects_duplicate_sentence_ids_across_batches(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    store.write_batch(0, [_record()])
    store.write_batch(1, [_record()])
    with pytest.raises(CheckpointError, match="duplicate"):
        store.load_all()


def test_rejects_tampered_parquet(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    store.write_batch(0, [_record()])
    parquet = tmp_path / "checkpoints" / "batch-000000.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"tamper")
    with pytest.raises(CheckpointError, match="SHA-256"):
        store.load_all()


def test_rejects_unexpected_checkpoint_entry(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    store.write_batch(0, [_record()])
    (tmp_path / "checkpoints" / "debug.txt").write_text("x")
    with pytest.raises(CheckpointError, match="unexpected"):
        store.load_all()


def test_progress_is_atomic_and_identity_bound(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    store.write_progress(completed=10, total=100, elapsed_seconds=5.0)
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["completed"] == 10
    assert progress["rows_per_second"] == 2.0
    assert progress["eta_seconds"] == 45.0
    assert progress["identity"] == _identity().to_dict()


def test_rejects_empty_negative_and_existing_batches(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    with pytest.raises(CheckpointError, match="non-empty"):
        store.write_batch(-1, [])
    store.write_batch(0, [_record()])
    with pytest.raises(CheckpointError, match="already exists"):
        store.write_batch(0, [_record("s2")])


def test_zero_elapsed_progress_has_no_eta(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    store.write_progress(completed=0, total=10, elapsed_seconds=0)
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["rows_per_second"] == 0
    assert progress["eta_seconds"] is None
