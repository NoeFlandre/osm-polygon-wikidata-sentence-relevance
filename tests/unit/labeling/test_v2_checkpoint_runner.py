from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.labeling.contracts import RunIdentity
from osm_polygon_sentence_relevance.labeling.v2_checkpoint import (
    V2_LOGIT_SCHEMA,
    V2CheckpointStore,
)
from osm_polygon_sentence_relevance.labeling.v2_contracts import (
    V2_LOGIT_PROMPT_VERSION,
    V2LogitRecord,
)
from osm_polygon_sentence_relevance.labeling.v2_runner import V2LogitRunner


def _identity() -> RunIdentity:
    return RunIdentity(
        input_sha256="a" * 64,
        input_dataset_revision="b" * 40,
        model_repo_id="ggml-org/Qwen3.6-27B-GGUF",
        model_revision="c" * 40,
        model_file="Qwen3.6-27B-Q4_K_M.gguf",
        model_file_sha256="d" * 64,
        prompt_version=V2_LOGIT_PROMPT_VERSION,
        source_commit="e" * 40,
        engine="llama.cpp",
        engine_version="1",
        batch_size=2,
        sampling_target=2,
        sampling_seed="seed",
        h3_resolution=3,
        sampling_version="v2-area-h3-logit",
        release_lane="v2-worldwide",
    )


def test_v2_checkpoint_round_trip_is_binary_and_hash_validated(tmp_path: Path) -> None:
    store = V2CheckpointStore(tmp_path, _identity())
    record = V2LogitRecord(
        sentence_id="s1",
        place_relevance="yes",
        yes_logprob=-0.1,
        no_logprob=-1.1,
    )
    store.write_batch(0, [record])
    assert store.load_all() == [record]
    import pyarrow.parquet as pq

    assert pq.read_schema(store.directory / "batch-000000.parquet").equals(
        V2_LOGIT_SCHEMA
    )


def test_v2_checkpoint_write_fsyncs_files_and_directory(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd) or real_fsync(fd))
    store = V2CheckpointStore(tmp_path, _identity())
    store.write_batch(
        0,
        [V2LogitRecord("s1", "yes", -0.1, -1.1)],
    )

    assert len(calls) >= 4


def test_v2_checkpoint_rejects_duplicate_sentence_ids(tmp_path: Path) -> None:
    store = V2CheckpointStore(tmp_path, _identity())
    record = V2LogitRecord(
        sentence_id="s1",
        place_relevance="no",
        yes_logprob=-1.1,
        no_logprob=-0.1,
    )
    store.write_batch(0, [record])
    store.write_batch(1, [record])
    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        store.load_all()


def test_v2_checkpoint_rejects_duplicate_ids_within_a_batch(tmp_path: Path) -> None:
    store = V2CheckpointStore(tmp_path, _identity())
    record = V2LogitRecord(
        sentence_id="s1",
        place_relevance="no",
        yes_logprob=-1.1,
        no_logprob=-0.1,
    )

    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        store.write_batch(0, [record, record])


def test_v2_checkpoint_rejects_invalid_or_duplicate_batch_files(tmp_path: Path) -> None:
    store = V2CheckpointStore(tmp_path, _identity())
    with pytest.raises(ValueError, match="non-empty"):
        store.write_batch(-1, [])
    store.write_batch(0, [V2LogitRecord("s1", "yes", -0.1, -1.1)])
    with pytest.raises(ValueError, match="already exists"):
        store.write_batch(0, [V2LogitRecord("s2", "yes", -0.1, -1.1)])
    (store.directory / "notes.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="unexpected"):
        store.batch_indexes()


def test_v2_checkpoint_rejects_incomplete_batch(tmp_path: Path) -> None:
    store = V2CheckpointStore(tmp_path, _identity())
    (store.directory / "batch-000000.parquet").write_bytes(b"not parquet")
    with pytest.raises(ValueError, match="incomplete"):
        store.batch_indexes()


@pytest.mark.parametrize(
    ("metadata_update", "message"),
    [
        (lambda metadata: metadata.update(schema_version=2), "schema"),
        (lambda metadata: metadata.update(identity={}), "identity"),
        (lambda metadata: metadata.update(row_count=True), "row count"),
        (lambda metadata: metadata.update(row_count=2), "row count"),
        (lambda metadata: metadata.update(parquet_sha256="0" * 64), "SHA-256"),
    ],
)
def test_v2_checkpoint_rejects_tampered_metadata(
    tmp_path: Path, metadata_update, message: str
) -> None:
    store = V2CheckpointStore(tmp_path, _identity())
    store.write_batch(0, [V2LogitRecord("s1", "yes", -0.1, -1.1)])
    metadata_path = store.directory / "batch-000000.json"
    metadata = json.loads(metadata_path.read_text())
    metadata_update(metadata)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        store.load_all()


def test_v2_checkpoint_rejects_malformed_metadata_and_schema(tmp_path: Path) -> None:
    store = V2CheckpointStore(tmp_path, _identity())
    store.write_batch(0, [V2LogitRecord("s1", "yes", -0.1, -1.1)])
    metadata_path = store.directory / "batch-000000.json"
    metadata_path.write_text("not-json")
    with pytest.raises(ValueError, match="metadata"):
        store.load_all()

    store = V2CheckpointStore(tmp_path / "second", _identity())
    store.write_batch(0, [V2LogitRecord("s1", "yes", -0.1, -1.1)])
    metadata_path = store.directory / "batch-000000.json"
    metadata_path.write_text("[]")
    with pytest.raises(ValueError, match="metadata"):
        store.load_all()

    store = V2CheckpointStore(tmp_path / "third", _identity())
    store.write_batch(0, [V2LogitRecord("s1", "yes", -0.1, -1.1)])
    metadata = json.loads((store.directory / "batch-000000.json").read_text())
    metadata["row_count"] = 1
    pq.write_table(
        pa.table({"wrong": ["s1"]}), store.directory / "batch-000000.parquet"
    )
    metadata["parquet_sha256"] = (
        __import__("hashlib")
        .sha256((store.directory / "batch-000000.parquet").read_bytes())
        .hexdigest()
    )
    (store.directory / "batch-000000.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="schema"):
        store.load_all()


def test_v2_checkpoint_rejects_derived_score_tampering(tmp_path: Path) -> None:
    store = V2CheckpointStore(tmp_path, _identity())
    store.write_batch(0, [V2LogitRecord("s1", "yes", -0.1, -1.1)])
    parquet_path = store.directory / "batch-000000.parquet"
    table = pq.read_table(parquet_path)
    table = table.set_column(
        table.schema.get_field_index("logit_margin"),
        "logit_margin",
        pa.array([99.0]),
    )
    pq.write_table(table, parquet_path)
    metadata_path = store.directory / "batch-000000.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["parquet_sha256"] = "tampered"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="SHA-256"):
        store.load_all()


def test_v2_checkpoint_rejects_derived_score_mismatch_after_hash_update(
    tmp_path: Path,
) -> None:
    store = V2CheckpointStore(tmp_path, _identity())
    store.write_batch(0, [V2LogitRecord("s1", "yes", -0.1, -1.1)])
    parquet_path = store.directory / "batch-000000.parquet"
    table = pq.read_table(parquet_path).set_column(
        5,
        pa.field("two_class_probability", pa.float64(), nullable=False),
        pa.array([0.0]),
    )
    pq.write_table(table, parquet_path)
    metadata_path = store.directory / "batch-000000.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["parquet_sha256"] = (
        __import__("hashlib").sha256(parquet_path.read_bytes()).hexdigest()
    )
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="derived"):
        store.load_all()


def test_v2_checkpoint_rejects_symlinks_and_non_files(tmp_path: Path) -> None:
    store = V2CheckpointStore(tmp_path, _identity())
    target = tmp_path / "target"
    target.write_text("target")
    (store.directory / "link").symlink_to(target)
    with pytest.raises(ValueError, match="unexpected"):
        store.batch_indexes()


def test_v2_checkpoint_removes_parquet_if_metadata_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = V2CheckpointStore(tmp_path, _identity())
    import osm_polygon_sentence_relevance.labeling.v2_checkpoint as module

    real_atomic = module._atomic
    calls = 0

    def broken_atomic(path: Path, data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("metadata write failed")
        real_atomic(path, data)

    monkeypatch.setattr(module, "_atomic", broken_atomic)
    with pytest.raises(OSError, match="metadata"):
        store.write_batch(0, [V2LogitRecord("s1", "yes", -0.1, -1.1)])
    assert not (store.directory / "batch-000000.parquet").exists()


def test_v2_checkpoint_atomic_write_cleans_temporary_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_sentence_relevance.labeling.v2_checkpoint as module

    monkeypatch.setattr(
        module.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("replace"))
    )
    with pytest.raises(OSError, match="replace"):
        module._atomic(tmp_path / "artifact", b"data")
    assert list(tmp_path.glob(".artifact.*")) == []


class _Engine:
    def generate(
        self, messages: list[list[dict[str, str]]], *, sentence_ids: list[str]
    ) -> list[V2LogitRecord]:
        return [
            V2LogitRecord(sentence_id, "yes", -0.1, -1.1)
            for sentence_id in sentence_ids
        ]


def test_v2_runner_checkpoints_each_batch_and_resumes(tmp_path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "sentence_id": f"s{i}",
                "sentence_text_raw": "The valley has slopes.",
                "page_title": "Valley",
                "section_path": ["Geography"],
                "previous_sentence": None,
                "next_sentence": None,
            }
            for i in range(3)
        ]
    )
    store = V2CheckpointStore(tmp_path, _identity())
    first = V2LogitRunner(engine=_Engine(), store=store, batch_size=2).run(table)
    assert first.completed == 3
    assert len(list(store.directory.glob("batch-*.parquet"))) == 2
    second = V2LogitRunner(engine=_Engine(), store=store, batch_size=2).run(table)
    assert second.completed == 3
    assert len(list(store.directory.glob("batch-*.parquet"))) == 2


def test_v2_runner_enqueues_each_checkpoint_for_async_mirroring(tmp_path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "sentence_id": f"s{i}",
                "sentence_text_raw": "The valley has slopes.",
                "page_title": "Valley",
                "section_path": ["Geography"],
                "previous_sentence": None,
                "next_sentence": None,
            }
            for i in range(3)
        ]
    )
    store = V2CheckpointStore(tmp_path, _identity())
    mirrored: list[int] = []

    result = V2LogitRunner(
        engine=_Engine(),
        store=store,
        batch_size=2,
        checkpoint_mirror=mirrored.append,
    ).run(table)

    assert result.completed == 3
    assert mirrored == [0, 1]


def test_v2_runner_rejects_engine_records_for_different_ids(tmp_path: Path) -> None:
    class WrongIdEngine(_Engine):
        def generate(
            self, messages: list[list[dict[str, str]]], *, sentence_ids: list[str]
        ) -> list[V2LogitRecord]:
            return [V2LogitRecord("unexpected", "yes", -0.1, -1.1)]

    table = pa.Table.from_pylist(
        [
            {
                "sentence_id": "s1",
                "sentence_text_raw": "The valley has slopes.",
                "page_title": "Valley",
                "section_path": ["Geography"],
            }
        ]
    )
    store = V2CheckpointStore(tmp_path, _identity())

    import pytest

    with pytest.raises(ValueError, match="IDs"):
        V2LogitRunner(engine=WrongIdEngine(), store=store, batch_size=2).run(table)
    assert not list(store.directory.glob("batch-*"))


def _runner_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sentence_id": "s1",
                "sentence_text_raw": "A valley.",
                "page_title": "Valley",
                "section_path": "Geography",
                "previous_sentence": 3,
                "next_sentence": None,
            }
        ]
    )


def test_v2_runner_validates_constructor_and_input_contracts(tmp_path: Path) -> None:
    store = V2CheckpointStore(tmp_path, _identity())
    with pytest.raises(ValueError, match="positive"):
        V2LogitRunner(engine=_Engine(), store=store, batch_size=0)
    with pytest.raises(ValueError, match="identity"):
        V2LogitRunner(engine=_Engine(), store=store, batch_size=1)
    with pytest.raises(ValueError, match="missing"):
        V2LogitRunner(engine=_Engine(), store=store, batch_size=2).run(
            pa.table({"sentence_id": ["s1"]})
        )
    duplicate = pa.Table.from_pylist(
        [
            {"sentence_id": "s1", "page_title": "x", "section_path": []},
            {"sentence_id": "s1", "page_title": "x", "section_path": []},
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        V2LogitRunner(engine=_Engine(), store=store, batch_size=2).run(duplicate)


def test_v2_runner_rejects_invalid_context_and_stale_checkpoints(
    tmp_path: Path,
) -> None:
    store = V2CheckpointStore(tmp_path, _identity())
    with pytest.raises(ValueError, match="text or null"):
        V2LogitRunner(engine=_Engine(), store=store, batch_size=2).run(_runner_table())

    stale_store = V2CheckpointStore(tmp_path / "stale", _identity())
    stale_store.write_batch(0, [V2LogitRecord("missing", "yes", -0.1, -1.1)])
    with pytest.raises(ValueError, match="absent"):
        V2LogitRunner(engine=_Engine(), store=stale_store, batch_size=2).run(
            pa.Table.from_pylist(
                [{"sentence_id": "s1", "page_title": "x", "section_path": []}]
            )
        )
    invalid_section = (
        _runner_table()
        .set_column(3, "section_path", pa.array([3]))
        .set_column(4, "previous_sentence", pa.array([None]))
    )
    with pytest.raises(ValueError, match="section_path"):
        V2LogitRunner(engine=_Engine(), store=store, batch_size=2).run(invalid_section)


def test_v2_runner_can_stop_before_inference_and_reports_timing(tmp_path: Path) -> None:
    store = V2CheckpointStore(tmp_path, _identity())
    result = V2LogitRunner(
        engine=_Engine(),
        store=store,
        batch_size=2,
        stop_requested=lambda: True,
        clock=lambda: 0.0,
    ).run(_runner_table().set_column(4, "previous_sentence", pa.array([None])))
    assert result.interrupted is True
    assert result.completed == 0
    assert json.loads((tmp_path / "timing.json").read_text())["interrupted"] is True


def test_v2_runner_accumulates_timing_across_resumes(tmp_path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "sentence_id": f"s{i}",
                "sentence_text_raw": "A valley.",
                "page_title": "Valley",
                "section_path": [],
                "previous_sentence": None,
                "next_sentence": None,
            }
            for i in range(3)
        ]
    )
    store = V2CheckpointStore(tmp_path, _identity())
    stop_calls = 0

    def stop_after_one_batch() -> bool:
        nonlocal stop_calls
        stop_calls += 1
        return stop_calls > 1

    first = V2LogitRunner(
        engine=_Engine(),
        store=store,
        batch_size=2,
        stop_requested=stop_after_one_batch,
        clock=iter([0.0, 1.0, 2.0, 3.0, 4.0]).__next__,
    ).run(table)
    second = V2LogitRunner(
        engine=_Engine(),
        store=store,
        batch_size=2,
        clock=iter([0.0, 0.1, 0.2, 0.3, 0.4]).__next__,
    ).run(table)

    assert first.interrupted is True
    assert second.completed == 3
    assert second.inference_seconds > first.inference_seconds
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert timing["inference_seconds"] == second.inference_seconds


def test_v2_runner_rejects_wrong_record_count_and_reports_zero_rate_callback(
    tmp_path: Path,
) -> None:
    class EmptyEngine:
        def generate(self, messages, *, sentence_ids):
            return []

    store = V2CheckpointStore(tmp_path, _identity())
    with pytest.raises(ValueError, match="count"):
        V2LogitRunner(
            engine=EmptyEngine(), store=store, batch_size=2, clock=lambda: 0.0
        ).run(_runner_table().set_column(4, "previous_sentence", pa.array([None])))

    callbacks: list[dict[str, object]] = []
    store = V2CheckpointStore(tmp_path / "callback", _identity())
    result = V2LogitRunner(
        engine=_Engine(),
        store=store,
        batch_size=2,
        clock=lambda: 0.0,
        batch_callback=callbacks.append,
    ).run(_runner_table().set_column(4, "previous_sentence", pa.array([None])))
    assert result.completed == 1
    assert callbacks[0]["eta_seconds"] is None
