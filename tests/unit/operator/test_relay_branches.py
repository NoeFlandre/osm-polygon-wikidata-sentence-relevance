"""Focused branch tests for the checkpoint relay safety boundary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator import relay
from osm_polygon_sentence_relevance.operator.relay_transport import RemoteEntry
from tests.unit.operator.test_relay import (
    _build_real_checkpoint_set,
    _FakeRemote,
    _identity,
    _patch_subprocess,
)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "not a mapping"),
        ({"identity": {}}, "identity mismatch"),
        (
            {"identity": _identity().to_dict(), "completed": "bad", "total": 4},
            "completed is invalid",
        ),
        (
            {"identity": _identity().to_dict(), "completed": 1, "total": "bad"},
            "total is invalid",
        ),
        (
            {
                "identity": _identity().to_dict(),
                "completed": 1,
                "remaining": "bad",
            },
            "remaining is invalid",
        ),
        (
            {"identity": _identity().to_dict(), "completed": 1},
            "missing total",
        ),
        (
            {"identity": _identity().to_dict(), "completed": 5, "total": 4},
            "counters are inconsistent",
        ),
    ],
)
def test_validate_progress_rejects_invalid_payloads(
    payload: object, message: str
) -> None:
    with pytest.raises(relay.RelayError, match=message):
        relay._validate_progress_payload(payload, _identity().to_dict())


def test_validate_progress_accepts_remaining_counter() -> None:
    identity = _identity().to_dict()
    assert relay._validate_progress_payload(
        {"identity": identity, "completed": 2, "remaining": 3}, identity
    ) == (2, 5)


def test_next_generation_skips_malformed_names_and_increments(tmp_path: Path) -> None:
    (tmp_path / "relay.generation-nope").mkdir()
    (tmp_path / "relay.generation-0003").mkdir()
    (tmp_path / "ordinary").mkdir()
    assert relay._next_generation(tmp_path).name == "relay.generation-0004"


def test_list_local_dir_reports_all_entry_types(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x")
    (tmp_path / "dir").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "file")
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    assert {(entry.name, entry.kind) for entry in relay._list_local_dir(tmp_path)} == {
        ("file", "file"),
        ("dir", "dir"),
        ("link", "symlink"),
        ("fifo", "other"),
    }


def test_inventory_ordered_indexes_sorts(tmp_path: Path) -> None:
    inventory = relay.RelayInventory(
        root=tmp_path,
        progress=tmp_path / "progress.json",
        timing=None,
        parquet_paths=(
            tmp_path / "batch-000002.parquet",
            tmp_path / "batch-000000.parquet",
        ),
        metadata_paths={},
        completed=2,
        total=3,
    )
    assert inventory.ordered_indexes == (0, 2)


def test_overlapping_identity_reports_first_mismatch() -> None:
    canonical = _identity().to_dict()
    expected = dict(canonical)
    expected["prompt_version"] = "different"
    with pytest.raises(relay.RelayError, match="prompt_version"):
        relay._enforce_overlapping_identity(canonical, expected)


def test_stage_local_rejects_empty_symlink_and_bad_run_root(tmp_path: Path) -> None:
    destination = tmp_path / "Seagate"
    destination.mkdir()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(relay.RelayError, match="empty"):
        relay._stage_relay_local(
            source_dir=empty,
            destination_root=destination,
            run_id="a" * 20,
            expected_run_identity=None,
        )
    link = tmp_path / "source-link"
    link.symlink_to(empty)
    with pytest.raises(relay.RelayError, match="real directory"):
        relay._stage_relay_local(
            source_dir=link,
            destination_root=destination,
            run_id="b" * 20,
            expected_run_identity=None,
        )
    source = tmp_path / "source"
    _build_real_checkpoint_set(source)
    (destination / ("c" * 20)).write_text("not a directory")
    with pytest.raises(relay.RelayError, match="run root"):
        relay._stage_relay_local(
            source_dir=source,
            destination_root=destination,
            run_id="c" * 20,
            expected_run_identity=_identity().to_dict(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("remove_progress", "missing progress"),
        ("remove_batches", "missing any checkpoint"),
        ("bad_metadata", "metadata is malformed"),
        ("metadata_list", "metadata is not a mapping"),
        ("missing_identity", "identity missing"),
        ("missing_sha", "parquet_sha256 missing"),
        ("bad_sha", "SHA-256 mismatch"),
    ],
)
def test_stage_local_rejects_corrupt_checkpoint_sets(
    tmp_path: Path, mutation: str, message: str
) -> None:
    source = tmp_path / "source"
    _build_real_checkpoint_set(source)
    if mutation == "remove_progress":
        (source / "progress.json").unlink()
    elif mutation == "remove_batches":
        for path in (source / "checkpoints").iterdir():
            path.unlink()
    else:
        metadata = next((source / "checkpoints").glob("*.json"))
        payload = json.loads(metadata.read_text())
        if mutation == "bad_metadata":
            metadata.write_text("{")
        elif mutation == "metadata_list":
            metadata.write_text("[]")
        elif mutation == "missing_identity":
            payload.pop("identity")
            metadata.write_text(json.dumps(payload))
        elif mutation == "missing_sha":
            payload.pop("parquet_sha256")
            metadata.write_text(json.dumps(payload))
        else:
            payload["parquet_sha256"] = "0" * 64
            metadata.write_text(json.dumps(payload))
    destination = tmp_path / "Seagate"
    destination.mkdir()
    with pytest.raises(relay.RelayError, match=message):
        relay._stage_relay_local(
            source_dir=source,
            destination_root=destination,
            run_id="d" * 20,
            expected_run_identity=_identity().to_dict(),
        )


def test_retrieve_requires_progress_and_rejects_remote_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _build_real_checkpoint_set(source)
    fake = _FakeRemote(tmp_path / "remote")
    fake.root.mkdir()
    _patch_subprocess(monkeypatch, fake)
    destination = tmp_path / "Seagate"
    destination.mkdir()
    monkeypatch.setattr(
        relay,
        "list_remote_dir",
        lambda *_a: [RemoteEntry("fifo", "other")],
    )
    with pytest.raises(relay.RelayError, match="missing progress"):
        relay.retrieve_to_seagate(
            source=relay.RemoteTransfer("nancy"),
            source_checkpoint_root="/home/u/work",
            destination_root=destination,
            run_id="e" * 20,
        )


def test_destination_root_rejects_symlink_and_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(relay.RelayError, match="real directory"):
        relay._validate_destination_root(missing)
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(relay.RelayError, match="symlink"):
        relay._validate_destination_root(link)


@pytest.mark.parametrize(
    ("entry_name", "kind", "message"),
    [
        ("link", "symlink", "relay symlink"),
        ("extra", "dir", "unexpected relay subdirectory"),
        ("fifo", "other", "relay non-file"),
        ("unexpected.txt", "file", "unexpected relay entry"),
    ],
)
def test_stage_local_rejects_unsafe_top_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_name: str,
    kind: str,
    message: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "progress.json").write_text("{}")
    destination = tmp_path / "Seagate"
    destination.mkdir()
    monkeypatch.setattr(
        relay,
        "_list_local_dir",
        lambda _path: [RemoteEntry(entry_name, kind)],
    )
    with pytest.raises(relay.RelayError, match=message):
        relay._stage_relay_local(
            source_dir=source,
            destination_root=destination,
            run_id="f" * 20,
            expected_run_identity=None,
        )


@pytest.mark.parametrize(
    ("kind", "name", "message"),
    [
        ("symlink", "batch-000000.json", "relay symlink"),
        ("dir", "batch-000000.json", "relay non-file"),
        ("file", "unexpected.txt", "unexpected relay entry"),
    ],
)
def test_stage_local_rejects_unsafe_checkpoint_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    name: str,
    message: str,
) -> None:
    source = tmp_path / "source"
    _build_real_checkpoint_set(source)
    destination = tmp_path / "Seagate"
    destination.mkdir()
    original = relay._list_local_dir

    def listing(path: Path) -> list[RemoteEntry]:
        if path.name == "checkpoints":
            return [RemoteEntry(name, kind)]
        return original(path)

    monkeypatch.setattr(relay, "_list_local_dir", listing)
    with pytest.raises(relay.RelayError, match=message):
        relay._stage_relay_local(
            source_dir=source,
            destination_root=destination,
            run_id="1" * 20,
            expected_run_identity=None,
        )


def test_stage_local_rejects_missing_pair_and_progress_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _build_real_checkpoint_set(source)
    next((source / "checkpoints").glob("*.json")).unlink()
    destination = tmp_path / "Seagate"
    destination.mkdir()
    with pytest.raises(relay.RelayError, match="missing metadata"):
        relay._stage_relay_local(
            source_dir=source,
            destination_root=destination,
            run_id="2" * 20,
            expected_run_identity=None,
        )

    source2 = tmp_path / "source2"
    _build_real_checkpoint_set(source2)
    progress = json.loads((source2 / "progress.json").read_text())
    progress.pop("identity")
    (source2 / "progress.json").write_text(json.dumps(progress))
    with pytest.raises(relay.RelayError, match="progress.json identity is missing"):
        relay._stage_relay_local(
            source_dir=source2,
            destination_root=destination,
            run_id="3" * 20,
            expected_run_identity=None,
        )


def test_stage_local_preserves_previous_generation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _build_real_checkpoint_set(source)
    destination = tmp_path / "Seagate"
    destination.mkdir()
    first = relay._stage_relay_local(
        source_dir=source,
        destination_root=destination,
        run_id="4" * 20,
        expected_run_identity=_identity().to_dict(),
    )
    second = relay._stage_relay_local(
        source_dir=source,
        destination_root=destination,
        run_id="4" * 20,
        expected_run_identity=_identity().to_dict(),
    )
    assert first.root != second.root
    assert (destination / ("4" * 20) / "prev").is_symlink()


@pytest.mark.parametrize(
    ("top_entry", "checkpoint_entry", "message"),
    [
        (RemoteEntry("link", "symlink"), None, "remote symlink"),
        (RemoteEntry("extra", "dir"), None, "unexpected remote subdirectory"),
        (RemoteEntry("fifo", "other"), None, "remote non-file"),
        (RemoteEntry("extra.txt", "file"), None, "unexpected remote top-level"),
        (
            None,
            RemoteEntry("batch-000000.json", "symlink"),
            "remote symlink",
        ),
        (
            None,
            RemoteEntry("batch-000000.json", "dir"),
            "remote non-file checkpoint",
        ),
        (
            None,
            RemoteEntry("unexpected.txt", "file"),
            "unexpected remote checkpoint",
        ),
    ],
)
def test_retrieve_rejects_unsafe_remote_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    top_entry: RemoteEntry | None,
    checkpoint_entry: RemoteEntry | None,
    message: str,
) -> None:
    destination = tmp_path / "Seagate"
    destination.mkdir()
    top = [RemoteEntry("progress.json", "file")]
    if top_entry is not None:
        top.append(top_entry)
    monkeypatch.setattr(relay, "list_remote_dir", lambda *_a: top)
    monkeypatch.setattr(
        relay,
        "list_remote_checkpoints",
        lambda *_a: [checkpoint_entry] if checkpoint_entry is not None else [],
    )

    class NoFetch(relay.RemoteTransfer):
        def fetch(self, _remote_path: str, local_path: Path) -> None:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text("{}")

    with pytest.raises(relay.RelayError, match=message):
        relay.retrieve_to_seagate(
            source=NoFetch("nancy"),
            source_checkpoint_root="/home/u/work",
            destination_root=destination,
            run_id="5" * 20,
        )


def test_stage_destination_rejects_nonempty_and_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = relay.RelayInventory(
        root=tmp_path,
        progress=tmp_path / "progress.json",
        timing=None,
        parquet_paths=(),
        metadata_paths={},
        completed=1,
        total=2,
    )
    inventory.progress.write_text("{}")
    monkeypatch.setattr(
        relay.subprocess,
        "run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout="occupied\n"),
    )
    with pytest.raises(relay.RelayError, match="non-empty"):
        relay.stage_to_destination(
            inventory=inventory,
            destination=relay.RemoteTransfer("nancy"),
            destination_checkpoint_root="/home/u/work",
        )


def test_stage_local_accepts_flat_batch_layout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _build_real_checkpoint_set(source)
    for checkpoint in list((source / "checkpoints").iterdir()):
        checkpoint.replace(source / checkpoint.name)
    (source / "checkpoints").rmdir()
    destination = tmp_path / "Seagate"
    destination.mkdir()
    inventory = relay._stage_relay_local(
        source_dir=source,
        destination_root=destination,
        run_id="6" * 20,
        expected_run_identity=_identity().to_dict(),
    )
    assert inventory.completed > 0


def test_stage_local_rejects_cross_batch_and_progress_identity_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _build_real_checkpoint_set(source)
    checkpoint_dir = source / "checkpoints"
    first_parquet = next(checkpoint_dir.glob("*.parquet"))
    first_metadata = next(checkpoint_dir.glob("*.json"))
    (checkpoint_dir / "batch-000001.parquet").write_bytes(first_parquet.read_bytes())
    metadata = json.loads(first_metadata.read_text())
    metadata["identity"] = {**metadata["identity"], "prompt_version": "different"}
    metadata["parquet_sha256"] = hashlib.sha256(first_parquet.read_bytes()).hexdigest()
    (checkpoint_dir / "batch-000001.json").write_text(json.dumps(metadata))
    destination = tmp_path / "Seagate"
    destination.mkdir()
    with pytest.raises(relay.RelayError, match="differs across batches"):
        relay._stage_relay_local(
            source_dir=source,
            destination_root=destination,
            run_id="7" * 20,
            expected_run_identity=None,
        )

    source2 = tmp_path / "source2"
    _build_real_checkpoint_set(source2)
    progress = json.loads((source2 / "progress.json").read_text())
    progress["identity"] = {**progress["identity"], "prompt_version": "different"}
    (source2 / "progress.json").write_text(json.dumps(progress))
    with pytest.raises(relay.RelayError, match="does not match checkpoints"):
        relay._stage_relay_local(
            source_dir=source2,
            destination_root=destination,
            run_id="8" * 20,
            expected_run_identity=None,
        )


def test_stage_destination_rejects_corrupt_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _build_real_checkpoint_set(source)
    destination_root = tmp_path / "Seagate"
    destination_root.mkdir()
    inventory = relay._stage_relay_local(
        source_dir=source,
        destination_root=destination_root,
        run_id="9" * 20,
        expected_run_identity=_identity().to_dict(),
    )

    class CorruptTransfer:
        ssh_target = "nancy"

        def ssh_mkdir_0700(self, _path: str) -> None:
            pass

        def push(self, _local: Path, _remote: str) -> None:
            pass

        def fetch(self, _remote: str, local: Path) -> None:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(b"corrupt")

        def ssh_atomic_rename(self, _src: str, _dst: str) -> None:
            raise AssertionError("rename must not happen")

    monkeypatch.setattr(
        relay.subprocess,
        "run",
        lambda *_a, **_kw: SimpleNamespace(returncode=1, stdout=""),
    )
    with pytest.raises(relay.RelayError, match="readback hash mismatch"):
        relay.stage_to_destination(
            inventory=inventory,
            destination=CorruptTransfer(),  # type: ignore[arg-type]
            destination_checkpoint_root="/home/u/work",
        )
