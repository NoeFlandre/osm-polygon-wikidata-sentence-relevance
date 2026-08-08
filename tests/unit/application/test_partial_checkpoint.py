"""Tests for crash-safe intra-shard segmentation progress."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.application._checkpoint import (
    partial as partial_module,
)
from osm_polygon_sentence_relevance.application._checkpoint.common import (
    CheckpointValidationError,
)
from osm_polygon_sentence_relevance.application._checkpoint.inventory import (
    SourceFileEntry,
)
from osm_polygon_sentence_relevance.application._checkpoint.partial import (
    append_partial_batch,
    create_partial_state,
    discard_partial_state,
    load_partial_state,
    merge_partial_reports,
    partial_shard_path,
    read_partial_table,
)
from osm_polygon_sentence_relevance.contracts.schemas import SEGMENTED_SENTENCES_SCHEMA
from osm_polygon_sentence_relevance.sentences.segmentation import SegmentationReport
from osm_polygon_sentence_relevance.sentences.table import SegmentedBatch

SOURCE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
INPUT_REVISION = "abcdefabcdefabcdefabcdefabcdefabcdefabcd"
INPUT_ROOT = Path("/data/input")
SOURCE_FILES = [SourceFileEntry("polygons/reg-a.parquet", 1, "a" * 64)]


def _report(index: int) -> SegmentationReport:
    return SegmentationReport(
        input_section_occurrence_count=1,
        emitted_segment_count=index,
        retained_sentence_occurrence_count=index,
        dropped_empty_raw_count=0,
        dropped_empty_normalized_count=0,
        wikipedia_sentence_occurrence_count=index,
        wikivoyage_sentence_occurrence_count=0,
    )


def _batch(start: int, end: int, index: int = 1) -> SegmentedBatch:
    return SegmentedBatch(
        start_index=start,
        end_index=end,
        table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
        report=_report(index),
    )


def _create(tmp_path: Path, *, total_sections: int = 2):
    return create_partial_state(
        tmp_path,
        shard_key="reg-a",
        source_commit=SOURCE_COMMIT,
        input_dataset_revision=INPUT_REVISION,
        pipeline_version="v1",
        model_name="mock",
        batch_size=1,
        input_root=INPUT_ROOT,
        source_files=SOURCE_FILES,
        total_sections=total_sections,
    )


def _load(tmp_path: Path, *, total_sections: int = 2):
    return load_partial_state(
        tmp_path,
        shard_key="reg-a",
        source_commit=SOURCE_COMMIT,
        input_dataset_revision=INPUT_REVISION,
        pipeline_version="v1",
        model_name="mock",
        batch_size=1,
        input_root=INPUT_ROOT,
        source_files=SOURCE_FILES,
        total_sections=total_sections,
    )


def test_round_trip_and_report_merge(tmp_path: Path) -> None:
    state = _create(tmp_path)
    state = append_partial_batch(state, _batch(0, 1, 1))
    state = append_partial_batch(state, _batch(1, 2, 2))

    loaded = _load(tmp_path)
    assert loaded is not None
    assert loaded.next_section_index == 2
    assert [batch.start_index for batch in loaded.batches] == [0, 1]
    assert read_partial_table(loaded).schema.equals(SEGMENTED_SENTENCES_SCHEMA)
    assert merge_partial_reports(loaded).retained_sentence_occurrence_count == 3
    discard_partial_state(tmp_path, "reg-a")
    assert not partial_shard_path(tmp_path, "reg-a").exists()


def test_missing_partial_returns_none(tmp_path: Path) -> None:
    assert (
        load_partial_state(
            tmp_path,
            shard_key="reg-a",
            source_commit=SOURCE_COMMIT,
            input_dataset_revision=INPUT_REVISION,
            pipeline_version="v1",
            model_name="mock",
            batch_size=1,
            input_root=INPUT_ROOT,
            source_files=SOURCE_FILES,
            total_sections=2,
        )
        is None
    )


def test_append_requires_next_batch(tmp_path: Path) -> None:
    state = _create(tmp_path)
    with pytest.raises(CheckpointValidationError, match="next expected"):
        append_partial_batch(state, _batch(1, 2))


def test_append_rejects_wrong_schema(tmp_path: Path) -> None:
    state = _create(tmp_path)
    batch = SegmentedBatch(
        start_index=0,
        end_index=1,
        table=pa.table({"wrong": ["x"]}),
        report=_report(1),
    )
    with pytest.raises(CheckpointValidationError, match="wrong schema"):
        append_partial_batch(state, batch)


def test_append_rejects_out_of_bounds_and_duplicate(tmp_path: Path) -> None:
    state = _create(tmp_path)
    with pytest.raises(CheckpointValidationError, match="out of bounds"):
        append_partial_batch(state, _batch(0, 3))
    state = append_partial_batch(state, _batch(0, 1))
    state = replace(state, next_section_index=0, batches=())
    with pytest.raises(CheckpointValidationError, match="already exists"):
        append_partial_batch(state, _batch(0, 1))


def test_append_cleans_batch_when_progress_write_fails(
    tmp_path: Path, monkeypatch
) -> None:
    state = _create(tmp_path)

    def fail(_state):
        raise OSError("progress unavailable")

    monkeypatch.setattr(partial_module, "_write_progress", fail)
    with pytest.raises(OSError, match="progress unavailable"):
        append_partial_batch(state, _batch(0, 1))
    assert not any(state.directory.glob("batch-*.parquet"))


def test_create_rejects_existing_partial(tmp_path: Path) -> None:
    _create(tmp_path)
    with pytest.raises(CheckpointValidationError, match="already exists"):
        _create(tmp_path)


@pytest.mark.parametrize("field", ["source_commit", "model_name", "batch_size"])
def test_load_rejects_identity_mismatch(tmp_path: Path, field: str) -> None:
    state = _create(tmp_path)
    append_partial_batch(state, _batch(0, 1))
    values = {
        "source_commit": SOURCE_COMMIT,
        "input_dataset_revision": INPUT_REVISION,
        "pipeline_version": "v1",
        "model_name": "mock",
        "batch_size": 1,
    }
    values[field] = 2 if field == "batch_size" else "different"
    with pytest.raises(CheckpointValidationError, match="mismatch"):
        load_partial_state(
            tmp_path,
            shard_key="reg-a",
            source_commit=values["source_commit"],
            input_dataset_revision=values["input_dataset_revision"],
            pipeline_version=values["pipeline_version"],
            model_name=values["model_name"],
            batch_size=values["batch_size"],
            input_root=INPUT_ROOT,
            source_files=SOURCE_FILES,
            total_sections=2,
        )


def test_empty_state_table_and_report(tmp_path: Path) -> None:
    _create(tmp_path, total_sections=0)
    loaded = _load(tmp_path, total_sections=0)
    assert loaded is not None
    assert read_partial_table(loaded).num_rows == 0
    assert merge_partial_reports(loaded).input_section_occurrence_count == 0


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda p: p.write_text("{"), "malformed"),
        (lambda p: p.write_text("[]"), "JSON object"),
        (lambda p: json.loads(p.read_text()).update({"source_files": []}), "manifest"),
    ],
)
def test_load_rejects_malformed_progress(tmp_path: Path, mutator, message: str) -> None:
    state = _create(tmp_path)
    progress = state.directory / "progress.json"
    if message == "manifest":
        payload = json.loads(progress.read_text())

        def mutator(path: Path) -> None:
            path.write_text(json.dumps({**payload, "source_files": []}))

    mutator(progress)
    os.chmod(progress, 0o600)
    with pytest.raises(CheckpointValidationError, match=message):
        _load(tmp_path)


def test_load_rejects_non_integer_next_index(tmp_path: Path) -> None:
    state = _create(tmp_path)
    progress = state.directory / "progress.json"
    payload = json.loads(progress.read_text())
    payload["next_section_index"] = True
    progress.write_text(json.dumps(payload))
    os.chmod(progress, 0o600)
    with pytest.raises(CheckpointValidationError, match="must be an integer"):
        _load(tmp_path)


@pytest.mark.parametrize(
    ("payload_edit", "message"),
    [
        (lambda item: item.update({"filename": "../escape.parquet"}), "unsafe"),
        (lambda item: item.update({"start_index": True}), "integers"),
        (lambda item: item.update({"sha256": "x" * 64}), "hash is invalid"),
        (lambda item: item.update({"report": {}}), "report is invalid"),
    ],
)
def test_load_rejects_invalid_batch_entries(
    tmp_path: Path, payload_edit, message: str
) -> None:
    state = append_partial_batch(_create(tmp_path), _batch(0, 1))
    progress = state.directory / "progress.json"
    payload = json.loads(progress.read_text())
    payload_edit(payload["batches"][0])
    progress.write_text(json.dumps(payload))
    os.chmod(progress, 0o600)
    with pytest.raises(CheckpointValidationError, match=message):
        _load(tmp_path)


def test_load_rejects_wrong_schema_and_row_count(tmp_path: Path) -> None:
    state = append_partial_batch(_create(tmp_path), _batch(0, 1))
    progress = state.directory / "progress.json"
    payload = json.loads(progress.read_text())
    payload["batches"][0]["rows"] = 1
    progress.write_text(json.dumps(payload))
    os.chmod(progress, 0o600)
    with pytest.raises(CheckpointValidationError, match="row count mismatch"):
        _load(tmp_path)


def test_load_rejects_invalid_batch_sequence(tmp_path: Path) -> None:
    state = append_partial_batch(_create(tmp_path), _batch(0, 1))
    progress = state.directory / "progress.json"
    payload = json.loads(progress.read_text())
    payload["batches"][0]["start_index"] = 1
    progress.write_text(json.dumps(payload))
    os.chmod(progress, 0o600)
    with pytest.raises(CheckpointValidationError, match="contiguous"):
        _load(tmp_path)


def test_load_rejects_extra_batch_and_negative_rows(tmp_path: Path) -> None:
    state = append_partial_batch(_create(tmp_path), _batch(0, 1))
    progress = state.directory / "progress.json"
    payload = json.loads(progress.read_text())
    payload["batches"][0]["rows"] = -1
    progress.write_text(json.dumps(payload))
    os.chmod(progress, 0o600)
    with pytest.raises(CheckpointValidationError, match="negative"):
        _load(tmp_path)


def test_load_rejects_wrong_batch_schema(tmp_path: Path) -> None:
    state = append_partial_batch(_create(tmp_path), _batch(0, 1))
    batch_path = state.directory / state.batches[0].filename
    partial_module.pq.write_table(pa.table({"wrong": ["x"]}), batch_path)
    os.chmod(batch_path, 0o600)
    payload = json.loads((state.directory / "progress.json").read_text())
    payload["batches"][0]["sha256"] = partial_module.sha256_file(batch_path)
    (state.directory / "progress.json").write_text(json.dumps(payload))
    os.chmod(state.directory / "progress.json", 0o600)
    with pytest.raises(CheckpointValidationError, match="wrong schema"):
        _load(tmp_path)


@pytest.mark.parametrize(
    ("batches", "message"),
    [
        ("not-a-list", "must be a list"),
        ([1], "entry must be an object"),
    ],
)
def test_load_rejects_invalid_batch_container(
    tmp_path: Path, batches, message: str
) -> None:
    state = _create(tmp_path)
    progress = state.directory / "progress.json"
    payload = json.loads(progress.read_text())
    payload["batches"] = batches
    progress.write_text(json.dumps(payload))
    os.chmod(progress, 0o600)
    with pytest.raises(CheckpointValidationError, match=message):
        _load(tmp_path)


def test_load_rejects_next_index_out_of_bounds(tmp_path: Path) -> None:
    state = _create(tmp_path)
    progress = state.directory / "progress.json"
    payload = json.loads(progress.read_text())
    payload["next_section_index"] = -1
    progress.write_text(json.dumps(payload))
    os.chmod(progress, 0o600)
    with pytest.raises(CheckpointValidationError, match="out of bounds"):
        _load(tmp_path)


def test_load_rejects_batch_read_error(tmp_path: Path, monkeypatch) -> None:
    append_partial_batch(_create(tmp_path), _batch(0, 1))
    monkeypatch.setattr(
        partial_module.pq, "read_table", lambda _: (_ for _ in ()).throw(OSError("bad"))
    )
    with pytest.raises(CheckpointValidationError, match="cannot be read"):
        _load(tmp_path)


def test_filesystem_guards_reject_bad_paths(tmp_path: Path, monkeypatch) -> None:
    directory = tmp_path / "directory"
    directory.mkdir(mode=0o700)
    os.chmod(directory, 0o755)
    with pytest.raises(CheckpointValidationError, match="unsafe mode"):
        partial_module._ensure_directory(directory)

    regular = tmp_path / "regular"
    regular.write_text("x")
    os.chmod(regular, 0o755)
    with pytest.raises(CheckpointValidationError, match="unsafe mode"):
        partial_module._ensure_regular(regular, 0o600)

    symlink = tmp_path / "symlink"
    symlink.symlink_to(regular)
    with pytest.raises(CheckpointValidationError, match="not a directory"):
        partial_module._ensure_directory(symlink)
    with pytest.raises(CheckpointValidationError, match="not regular"):
        partial_module._ensure_regular(symlink, 0o600)

    monkeypatch.setattr(
        partial_module.os, "lstat", lambda _: (_ for _ in ()).throw(OSError("denied"))
    )
    with pytest.raises(CheckpointValidationError, match="inaccessible"):
        partial_module._ensure_directory(tmp_path / "missing")
    with pytest.raises(CheckpointValidationError, match="inaccessible"):
        partial_module._ensure_regular(tmp_path / "missing-file", 0o600)


@pytest.mark.parametrize("mode", [0o755])
def test_load_rejects_unsafe_modes(tmp_path: Path, mode: int) -> None:
    state = _create(tmp_path)
    os.chmod(state.directory, mode)
    with pytest.raises(CheckpointValidationError, match="unsafe mode"):
        _load(tmp_path)


def test_discard_missing_is_noop(tmp_path: Path) -> None:
    discard_partial_state(tmp_path, "reg-a")


def test_load_rejects_tampered_batch(tmp_path: Path) -> None:
    state = _create(tmp_path)
    state = append_partial_batch(state, _batch(0, 1))
    (state.directory / state.batches[0].filename).write_bytes(b"tampered")
    os.chmod(state.directory / state.batches[0].filename, 0o600)
    with pytest.raises(CheckpointValidationError, match="hash mismatch"):
        load_partial_state(
            tmp_path,
            shard_key="reg-a",
            source_commit=SOURCE_COMMIT,
            input_dataset_revision=INPUT_REVISION,
            pipeline_version="v1",
            model_name="mock",
            batch_size=1,
            input_root=INPUT_ROOT,
            source_files=SOURCE_FILES,
            total_sections=2,
        )


def test_load_rejects_extra_entry(tmp_path: Path) -> None:
    state = _create(tmp_path)
    (state.directory / "extra").write_text("x")
    os.chmod(state.directory / "extra", 0o600)
    with pytest.raises(CheckpointValidationError, match="unexpected entries"):
        load_partial_state(
            tmp_path,
            shard_key="reg-a",
            source_commit=SOURCE_COMMIT,
            input_dataset_revision=INPUT_REVISION,
            pipeline_version="v1",
            model_name="mock",
            batch_size=1,
            input_root=INPUT_ROOT,
            source_files=SOURCE_FILES,
            total_sections=2,
        )


def test_load_removes_interrupted_atomic_temporary_files(tmp_path: Path) -> None:
    """A killed atomic write must not make durable partial progress unusable."""
    state = _create(tmp_path)
    progress_tmp = state.directory / ".progress.json.interrupted.tmp"
    batch_tmp = state.directory / ".batch-000000000-000000001.parquet.interrupted.tmp"
    progress_tmp.write_bytes(b"incomplete")
    batch_tmp.write_bytes(b"incomplete")
    os.chmod(progress_tmp, 0o600)
    os.chmod(batch_tmp, 0o600)

    loaded = _load(tmp_path)

    assert loaded is not None
    assert not progress_tmp.exists()
    assert not batch_tmp.exists()


def test_load_removes_interrupted_temporary_files_with_stale_mode(
    tmp_path: Path,
) -> None:
    """A stale temp file's mode must not block recovery of durable progress."""
    state = _create(tmp_path)
    progress_tmp = state.directory / ".progress.json.interrupted.tmp"
    progress_tmp.write_bytes(b"incomplete")
    os.chmod(progress_tmp, 0o644)

    loaded = _load(tmp_path)

    assert loaded is not None
    assert not progress_tmp.exists()


def test_load_rejects_noncontiguous_progress(tmp_path: Path) -> None:
    state = _create(tmp_path)
    payload = json.loads((state.directory / "progress.json").read_text())
    payload["next_section_index"] = 1
    (state.directory / "progress.json").write_text(json.dumps(payload))
    os.chmod(state.directory / "progress.json", 0o600)
    with pytest.raises(CheckpointValidationError, match="does not match batches"):
        load_partial_state(
            tmp_path,
            shard_key="reg-a",
            source_commit=SOURCE_COMMIT,
            input_dataset_revision=INPUT_REVISION,
            pipeline_version="v1",
            model_name="mock",
            batch_size=1,
            input_root=INPUT_ROOT,
            source_files=SOURCE_FILES,
            total_sections=2,
        )


def test_load_rejects_symlink_progress(tmp_path: Path) -> None:
    state = _create(tmp_path)
    progress = state.directory / "progress.json"
    progress.unlink()
    progress.symlink_to(tmp_path / "missing")
    with pytest.raises(CheckpointValidationError, match="not regular"):
        load_partial_state(
            tmp_path,
            shard_key="reg-a",
            source_commit=SOURCE_COMMIT,
            input_dataset_revision=INPUT_REVISION,
            pipeline_version="v1",
            model_name="mock",
            batch_size=1,
            input_root=INPUT_ROOT,
            source_files=SOURCE_FILES,
            total_sections=2,
        )


def test_partial_path_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(CheckpointValidationError, match="invalid shard_key"):
        partial_shard_path(tmp_path, "../bad")
