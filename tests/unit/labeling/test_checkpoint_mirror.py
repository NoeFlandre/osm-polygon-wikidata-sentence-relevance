"""Tests for the non-blocking Hugging Face checkpoint mirror."""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.labeling import checkpoint_mirror
from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
from osm_polygon_sentence_relevance.labeling.checkpoint_mirror import (
    CheckpointMirror,
    CheckpointMirrorError,
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
        batch_size=2,
        llama_parallel=16,
        llama_per_slot_context=4096,
        llama_total_context=65536,
        request_concurrency=16,
        sampling_target=200_000,
        sampling_seed="sentence-relevance-v2",
        h3_resolution=3,
        sampling_version="labeling-v2-h3-language-osm-primary",
    )


def _write_batch(store: CheckpointStore, index: int = 0) -> None:
    store.write_batch(
        index,
        [
            LabelRecord(
                sentence_id=f"s{index}",
                landuse_relevance=LabelValue.YES,
                polygon_relevance=LabelValue.YES,
                landuse_reason="explicit_land_use",
                polygon_reason="direct_polygon_reference",
                evidence="farming",
            )
        ],
    )


def test_enqueue_is_non_blocking_while_upload_is_in_flight(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    _write_batch(store, 0)
    _write_batch(store, 1)
    entered = threading.Event()
    release = threading.Event()
    calls: list[tuple[str, str, tuple[tuple[str, Path], ...]]] = []

    def upload(
        dataset_id: str, branch: str, files: tuple[tuple[str, Path], ...]
    ) -> str:
        calls.append((dataset_id, branch, files))
        entered.set()
        assert release.wait(5)
        return "commit"

    mirror = CheckpointMirror(
        store=store,
        dataset_id="NoeFlandre/osm-polygon-wikidata-sentence-relevance",
        branch="checkpoints/aaaaaaaaaaaaaaaaaaaa",
        uploader=upload,
    )
    mirror.start()
    started = time.monotonic()
    mirror.enqueue(0)
    assert entered.wait(5)
    mirror.enqueue(1)
    assert time.monotonic() - started < 1.0
    release.set()
    mirror.close(wait=True, timeout=5)
    assert [call[2][0][0] for call in calls] == [
        ".pipeline/checkpoints/aaaaaaaaaaaaaaaaaaaa/batch-000000.parquet",
        ".pipeline/checkpoints/aaaaaaaaaaaaaaaaaaaa/batch-000001.parquet",
    ]


def test_upload_failure_stays_pending_and_next_start_retries(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    _write_batch(store)
    attempts = 0

    def failing_upload(*_: object) -> str:
        nonlocal attempts
        attempts += 1
        raise OSError("network unavailable")

    first = CheckpointMirror(
        store=store,
        dataset_id="owner/dataset",
        branch="checkpoints/bbbbbbbbbbbbbbbbbbbb",
        uploader=failing_upload,
    )
    first.start()
    first.enqueue(0)
    first.close(wait=True, timeout=5)
    assert attempts == 1
    assert list((tmp_path / ".checkpoint-mirror" / "pending").glob("*.json"))
    status = json.loads((tmp_path / ".checkpoint-mirror" / "status.json").read_text())
    assert status["last_error"] == "upload-failed"

    succeeded: list[int] = []

    def successful_upload(*_: object) -> str:
        succeeded.append(1)
        return "commit"

    second = CheckpointMirror(
        store=store,
        dataset_id="owner/dataset",
        branch="checkpoints/bbbbbbbbbbbbbbbbbbbb",
        uploader=successful_upload,
    )
    second.start()
    second.close(wait=True, timeout=5)
    assert succeeded == [1]
    assert not list((tmp_path / ".checkpoint-mirror" / "pending").glob("*.json"))
    assert list((tmp_path / ".checkpoint-mirror" / "uploaded").glob("*.json"))


def test_enqueue_is_idempotent_for_a_batch(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    _write_batch(store)
    calls: list[int] = []
    mirror = CheckpointMirror(
        store=store,
        dataset_id="owner/dataset",
        branch="checkpoints/cccccccccccccccccccc",
        uploader=lambda *_: calls.append(1) or "commit",
    )
    mirror.enqueue(0)
    mirror.enqueue(0)
    mirror.start()
    mirror.close(wait=True, timeout=5)
    assert calls == [1]


def test_tampered_checkpoint_is_not_uploaded(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    _write_batch(store)
    calls: list[int] = []
    mirror = CheckpointMirror(
        store=store,
        dataset_id="owner/dataset",
        branch="checkpoints/dddddddddddddddddddd",
        uploader=lambda *_: calls.append(1) or "commit",
    )
    mirror.enqueue(0)
    (store.directory / "batch-000000.parquet").write_bytes(b"tampered")
    mirror.start()
    mirror.close(wait=True, timeout=5)
    assert calls == []
    status = json.loads((tmp_path / ".checkpoint-mirror" / "status.json").read_text())
    assert status["last_error"] == "checkpoint-drift"


def test_mirror_rejects_final_release_lanes_and_invalid_ids(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    with pytest.raises(ValueError, match="dataset ID"):
        CheckpointMirror(
            store=store, dataset_id="owner", branch="checkpoints/eeeeeeeeeeeeeeeeeeee"
        )
    with pytest.raises(ValueError, match="checkpoint namespace"):
        CheckpointMirror(store=store, dataset_id="owner/dataset", branch="main")
    with pytest.raises(ValueError, match="checkpoint namespace"):
        CheckpointMirror(
            store=store, dataset_id="owner/dataset", branch="checkpoints/not-a-run"
        )


def test_resume_identity_does_not_change_mirror_namespace(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    expanded = replace(_identity(), sampling_target=400_000)
    expanded_store = CheckpointStore(tmp_path, expanded)
    mirror = CheckpointMirror(
        store=expanded_store,
        dataset_id="owner/dataset",
        branch="checkpoints/ffffffffffffffffffff",
        uploader=lambda *_: "commit",
    )
    assert mirror.branch == "checkpoints/ffffffffffffffffffff"
    assert mirror.dataset_id == "owner/dataset"
    del store


def test_mirror_start_cleans_an_already_uploaded_pending_marker(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    _write_batch(store)
    mirror = CheckpointMirror(
        store=store,
        dataset_id="owner/dataset",
        branch="checkpoints/aaaaaaaaaaaaaaaaaaaa",
        uploader=lambda *_: "commit",
    )
    mirror.enqueue(0)
    pending = mirror.pending / "batch-000000.json"
    shutil.copy2(pending, mirror.uploaded / pending.name)
    mirror.start()
    mirror.start()
    mirror.close(wait=False)
    mirror.close(wait=False)
    assert not pending.exists()
    with pytest.raises(RuntimeError, match="closed"):
        mirror.start()


def test_mirror_rejects_invalid_enqueue_and_missing_batch(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    mirror = CheckpointMirror(
        store=store,
        dataset_id="owner/dataset",
        branch="checkpoints/aaaaaaaaaaaaaaaaaaaa",
        uploader=lambda *_: "commit",
    )
    with pytest.raises(ValueError, match="non-negative"):
        mirror.enqueue(-1)
    with pytest.raises(ValueError, match="non-negative"):
        mirror.enqueue(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        mirror.close(timeout=-1)
    mirror.enqueue(0)
    status = json.loads((tmp_path / ".checkpoint-mirror" / "status.json").read_text())
    assert status["last_error"] == "checkpoint-drift"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("identity", {"wrong": "identity"}),
        ("parquet_sha256", "bad"),
        ("row_count", 0),
    ],
)
def test_marker_payload_rejects_invalid_checkpoint_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    store = CheckpointStore(tmp_path, _identity())
    _write_batch(store)
    metadata_path = store.directory / "batch-000000.json"
    payload = json.loads(metadata_path.read_text())
    payload[field] = value
    metadata_path.write_text(json.dumps(payload))
    mirror = CheckpointMirror(
        store=store,
        dataset_id="owner/dataset",
        branch="checkpoints/aaaaaaaaaaaaaaaaaaaa",
        uploader=lambda *_: "commit",
    )
    mirror.enqueue(0)
    status = json.loads((tmp_path / ".checkpoint-mirror" / "status.json").read_text())
    assert status["last_error"] == "checkpoint-drift"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"schema_version": 1},
        {"schema_version": 1, "files": []},
        {
            "schema_version": 1,
            "files": [{"local_name": "unsafe", "remote_path": "main/unsafe"}],
        },
    ],
)
def test_pending_marker_validation_refuses_malformed_payload(
    tmp_path: Path, payload: object
) -> None:
    store = CheckpointStore(tmp_path, _identity())
    mirror = CheckpointMirror(
        store=store,
        dataset_id="owner/dataset",
        branch="checkpoints/aaaaaaaaaaaaaaaaaaaa",
        uploader=lambda *_: "commit",
    )
    marker = mirror.pending / "batch-000000.json"
    marker.write_text(json.dumps(payload))
    mirror.start()
    mirror.close(wait=True, timeout=5)
    status = json.loads((tmp_path / ".checkpoint-mirror" / "status.json").read_text())
    assert status["last_error"] == "checkpoint-drift"
    assert marker.exists()


def test_pending_marker_rejects_invalid_file_entries_and_missing_files(
    tmp_path: Path,
) -> None:
    store = CheckpointStore(tmp_path, _identity())
    mirror = CheckpointMirror(
        store=store,
        dataset_id="owner/dataset",
        branch="checkpoints/aaaaaaaaaaaaaaaaaaaa",
        uploader=lambda *_: "commit",
    )
    for payload in (
        {
            "schema_version": 1,
            "dataset_id": "owner/dataset",
            "branch": "checkpoints/aaaaaaaaaaaaaaaaaaaa",
            "identity": store.identity.checkpoint_dict(),
            "files": [None, None],
        },
        {
            "schema_version": 1,
            "dataset_id": "owner/dataset",
            "branch": "checkpoints/aaaaaaaaaaaaaaaaaaaa",
            "identity": store.identity.checkpoint_dict(),
            "files": [
                {"local_name": "unsafe", "remote_path": "main/unsafe"},
                {"local_name": "unsafe", "remote_path": "main/unsafe"},
            ],
        },
    ):
        marker = mirror.pending / "batch-000000.json"
        marker.write_text(json.dumps(payload))
        mirror.start()
        mirror.close(wait=True, timeout=5)
        assert marker.exists()
        mirror = CheckpointMirror(
            store=store,
            dataset_id="owner/dataset",
            branch="checkpoints/aaaaaaaaaaaaaaaaaaaa",
            uploader=lambda *_: "commit",
        )
    _write_batch(store)
    mirror.enqueue(0)
    (store.directory / "batch-000000.parquet").unlink()
    mirror.start()
    mirror.close(wait=True, timeout=5)
    assert (mirror.pending / "batch-000000.json").exists()


def test_marker_payload_rejects_invalid_json_metadata(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    _write_batch(store)
    (store.directory / "batch-000000.json").write_text("not json")
    mirror = CheckpointMirror(
        store=store,
        dataset_id="owner/dataset",
        branch="checkpoints/aaaaaaaaaaaaaaaaaaaa",
        uploader=lambda *_: "commit",
    )
    mirror.enqueue(0)
    status = json.loads((tmp_path / ".checkpoint-mirror" / "status.json").read_text())
    assert status["last_error"] == "checkpoint-drift"


def test_uploader_blank_commit_is_not_marked_uploaded(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    _write_batch(store)
    mirror = CheckpointMirror(
        store=store,
        dataset_id="owner/dataset",
        branch="checkpoints/aaaaaaaaaaaaaaaaaaaa",
        uploader=lambda *_: "",
    )
    mirror.start()
    mirror.enqueue(0)
    mirror.close(wait=True, timeout=5)
    assert list(mirror.pending.glob("*.json"))
    assert not list(mirror.uploaded.glob("*.json"))
    status = json.loads((tmp_path / ".checkpoint-mirror" / "status.json").read_text())
    assert status["last_error"] == "checkpoint-drift"


def test_default_uploader_commits_two_files_on_main(
    tmp_path: Path, monkeypatch
) -> None:
    calls: dict[str, object] = {}

    class FakeApi:
        def create_branch(self, **kwargs: object) -> None:
            calls["branch"] = kwargs

        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            calls["commit"] = kwargs
            return SimpleNamespace(oid="f" * 40)

    class FakeOperation:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, CommitOperationAdd=FakeOperation),
    )
    first = tmp_path / "batch-000000.parquet"
    second = tmp_path / "batch-000000.json"
    first.write_bytes(b"data")
    second.write_text("{}")
    result = checkpoint_mirror._default_uploader(
        "owner/dataset",
        "checkpoints/aaaaaaaaaaaaaaaaaaaa",
        (("remote/data", first), ("remote/meta", second)),
    )
    assert result == "f" * 40
    commit = calls["commit"]
    assert isinstance(commit, dict)
    assert commit["revision"] == "main"
    assert len(commit["operations"]) == 2


def test_status_write_is_best_effort_and_preserves_existing_facts(
    tmp_path: Path, monkeypatch
) -> None:
    store = CheckpointStore(tmp_path, _identity())
    mirror = CheckpointMirror(
        store=store,
        dataset_id="owner/dataset",
        branch="checkpoints/aaaaaaaaaaaaaaaaaaaa",
        uploader=lambda *_: "commit",
    )
    mirror._write_status(first="fact")
    mirror._write_status(second="fact")
    original = checkpoint_mirror._atomic_json
    monkeypatch.setattr(
        checkpoint_mirror, "_atomic_json", lambda *_: (_ for _ in ()).throw(OSError())
    )
    mirror._write_status(third="fact")
    monkeypatch.setattr(checkpoint_mirror, "_atomic_json", original)
    status = json.loads((tmp_path / ".checkpoint-mirror" / "status.json").read_text())
    assert status["first"] == "fact"
    assert status["second"] == "fact"


def test_default_uploader_rejects_missing_commit_id(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeApi:
        def create_branch(self, **_: object) -> None:
            return None

        def create_commit(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace()

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, CommitOperationAdd=lambda **kwargs: kwargs),
    )
    with pytest.raises(RuntimeError, match="no checkpoint commit ID"):
        checkpoint_mirror._default_uploader(
            "owner/dataset",
            "checkpoints/aaaaaaaaaaaaaaaaaaaa",
            (("remote/data", tmp_path / "data"),),
        )


def test_enqueue_missing_checkpoint_records_drift_without_upload(
    tmp_path: Path,
) -> None:
    store = CheckpointStore(tmp_path, _identity())
    calls: list[int] = []
    mirror = CheckpointMirror(
        store=store,
        dataset_id="owner/dataset",
        branch="checkpoints/aaaaaaaaaaaaaaaaaaaa",
        uploader=lambda *_: calls.append(1) or "commit",
    )
    mirror.enqueue(0)
    status = json.loads((tmp_path / ".checkpoint-mirror" / "status.json").read_text())
    assert status["last_error"] == "checkpoint-drift"
    assert calls == []


def test_upload_marker_validates_file_shape_and_paths(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, _identity())
    _write_batch(store)
    mirror = CheckpointMirror(
        store=store,
        dataset_id="owner/dataset",
        branch="checkpoints/aaaaaaaaaaaaaaaaaaaa",
        uploader=lambda *_: "commit",
    )
    base = mirror._marker_payload(0)
    variants = [
        {**base, "files": []},
        {**base, "files": [None, base["files"][1]]},
        {
            **base,
            "files": [
                {**base["files"][0], "local_name": "unsafe"},
                base["files"][1],
            ],
        },
        {**base, "batch_index": 1},
    ]
    for variant in variants:
        marker = mirror.pending / "batch-000000.json"
        marker.write_text(json.dumps(variant))
        with pytest.raises(CheckpointMirrorError):
            mirror._upload_marker(marker)
    (store.directory / "batch-000000.parquet").unlink()
    marker = mirror.pending / "batch-000000.json"
    marker.write_text(json.dumps(base))
    with pytest.raises(CheckpointMirrorError, match="regular file"):
        mirror._upload_marker(marker)
