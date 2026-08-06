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
        engine="llama.cpp",
        engine_version="1",
        batch_size=128,
        llama_parallel=16,
        llama_per_slot_context=4096,
        llama_total_context=65536,
        request_concurrency=16,
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
    changed = replace(_identity(), engine_version="different")
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


def test_v1_identity_shape_stays_legacy_and_v2_identity_is_explicit() -> None:
    legacy = _identity()
    assert "sampling_target" not in legacy.to_dict()

    v2 = replace(
        legacy,
        sampling_target=200_000,
        sampling_seed="sentence-relevance-v2",
        h3_resolution=3,
        sampling_version="labeling-v2-h3-language-osm-primary",
    )
    payload = v2.to_dict()
    assert payload["sampling_target"] == 200_000
    assert payload["sampling_seed"] == "sentence-relevance-v2"
    assert payload["h3_resolution"] == 3
    assert payload["sampling_version"] == "labeling-v2-h3-language-osm-primary"
    assert v2.to_dict() != legacy.to_dict()
    expanded = replace(v2, sampling_target=400_000)
    assert expanded.to_dict()["sampling_target"] == 400_000
    assert expanded.checkpoint_dict() == v2.checkpoint_dict()


def test_checkpoint_store_reuses_batches_when_v2_target_expands(
    tmp_path: Path,
) -> None:
    low = replace(
        _identity(),
        sampling_target=2,
        sampling_seed="sentence-relevance-v2",
        h3_resolution=3,
        sampling_version="labeling-v2-h3-language-osm-primary",
    )
    CheckpointStore(tmp_path, low).write_batch(0, [_record()])
    expanded = replace(low, sampling_target=4)
    assert CheckpointStore(tmp_path, expanded).load_all() == [_record()]


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


def test_write_batch_removes_parquet_when_metadata_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CheckpointStore(tmp_path, _identity())
    import osm_polygon_sentence_relevance.labeling.checkpoint as checkpoint

    original = checkpoint._atomic_bytes
    calls = 0

    def fail_metadata(path: Path, data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("metadata write failed")
        original(path, data)

    monkeypatch.setattr(checkpoint, "_atomic_bytes", fail_metadata)
    with pytest.raises(OSError, match="metadata write failed"):
        store.write_batch(0, [_record()])
    assert not (store.directory / "batch-000000.parquet").exists()
    assert not (store.directory / "batch-000000.json").exists()


def test_load_all_rejects_incomplete_and_malformed_batches(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    store.write_batch(0, [_record()])
    (store.directory / "batch-000000.json").unlink()
    with pytest.raises(CheckpointError, match="incomplete"):
        store.load_all()

    malformed = CheckpointStore(tmp_path / "malformed", _identity())
    malformed.write_batch(0, [_record()])
    (malformed.directory / "batch-000000.json").write_text("not json")
    with pytest.raises(CheckpointError, match="metadata is invalid"):
        malformed.load_all()


@pytest.mark.parametrize("field", ["schema", "row_count"])
def test_load_all_rejects_schema_and_row_count_mismatches(
    tmp_path: Path, field: str
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    import osm_polygon_sentence_relevance.labeling.checkpoint as checkpoint

    store = CheckpointStore(tmp_path, _identity())
    store.write_batch(0, [_record()])
    parquet = store.directory / "batch-000000.parquet"
    metadata = store.directory / "batch-000000.json"
    if field == "schema":
        pq.write_table(pa.table({"wrong": ["value"]}), parquet)
        payload = json.loads(metadata.read_text())
        payload["parquet_sha256"] = checkpoint._sha256(parquet)
        metadata.write_text(json.dumps(payload))
        with pytest.raises(CheckpointError, match="schema mismatch"):
            store.load_all()
    else:
        payload = json.loads(metadata.read_text())
        payload["row_count"] = 2
        metadata.write_text(json.dumps(payload))
        with pytest.raises(CheckpointError, match="row count mismatch"):
            store.load_all()


def test_atomic_bytes_cleans_temporary_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_sentence_relevance.labeling.checkpoint as checkpoint

    target = tmp_path / "target"

    def fail_replace(*_: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(checkpoint.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        checkpoint._atomic_bytes(target, b"payload")
    assert not list(tmp_path.glob(".*.tmp"))


def test_load_all_rejects_unexpected_entry_and_invalid_parquet(tmp_path: Path) -> None:
    import osm_polygon_sentence_relevance.labeling.checkpoint as checkpoint

    store = CheckpointStore(tmp_path / "entry", _identity())
    store.write_batch(0, [_record()])
    (store.directory / "unexpected").mkdir()
    with pytest.raises(CheckpointError, match="unexpected"):
        store.load_all()

    broken = CheckpointStore(tmp_path / "broken", _identity())
    broken.write_batch(0, [_record()])
    parquet = broken.directory / "batch-000000.parquet"
    parquet.write_bytes(b"not parquet")
    metadata = broken.directory / "batch-000000.json"
    payload = json.loads(metadata.read_text())
    payload["parquet_sha256"] = checkpoint._sha256(parquet)
    metadata.write_text(json.dumps(payload))
    with pytest.raises(CheckpointError, match="Parquet is invalid"):
        broken.load_all()
