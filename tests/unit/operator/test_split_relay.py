"""Transfer contracts for content-addressed split resume bundles."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from osm_polygon_sentence_relevance.operator import split_relay
from osm_polygon_sentence_relevance.operator.relay_transport import RemoteEntry
from tests.unit.scripts.streaming.test_resume_bundle import (
    IDENTITY,
    SHARD,
    _write_partial,
    _write_state,
)


class _LocalTransfer:
    def __init__(self, remote_root: Path) -> None:
        self.remote_root = remote_root
        self.ssh_target = "local-test"
        self.renames: list[tuple[str, str]] = []

    def _path(self, remote: str) -> Path:
        return self.remote_root / remote.lstrip("/")

    def fetch(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(remote), local)

    def push(self, local: Path, remote: str) -> None:
        target = self._path(remote)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)

    def ssh_mkdir_0700(self, remote: str) -> None:
        self._path(remote).mkdir(parents=True, exist_ok=True, mode=0o700)

    def ssh_atomic_rename(self, source: str, destination: str) -> None:
        source_path = self._path(source)
        destination_path = self._path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(destination_path)
        self.renames.append((source, destination))


def _entries(path: Path) -> list[RemoteEntry]:
    if not path.is_dir():
        return []
    result: list[RemoteEntry] = []
    for child in sorted(path.iterdir()):
        kind = "symlink" if child.is_symlink() else "dir" if child.is_dir() else "file"
        result.append(RemoteEntry(child.name, kind))
    return result


def test_retrieve_and_stage_split_resume_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_remote = tmp_path / "source-remote"
    source_work = source_remote / "home/u/run/work"
    _write_state(source_work, "afghanistan-latest")
    source = _LocalTransfer(source_remote)
    seagate = tmp_path / "Seagate"
    seagate.mkdir()

    def source_list(_target: str, remote: str) -> list[RemoteEntry]:
        return _entries(source._path(remote))

    monkeypatch.setattr(split_relay, "list_remote_dir", source_list)
    inventory = split_relay.retrieve_to_seagate(
        source=source,  # type: ignore[arg-type]
        source_work_root="/home/u/run/work",
        destination_root=seagate,
        run_id=str(IDENTITY["run_id"]),
        expected_identity=IDENTITY,
    )

    destination_remote = tmp_path / "destination-remote"
    destination = _LocalTransfer(destination_remote)
    monkeypatch.setattr(
        split_relay,
        "list_remote_dir",
        lambda _target, remote: _entries(destination._path(remote)),
    )
    remote_bundle = split_relay.stage_to_destination(
        inventory=inventory,
        destination=destination,  # type: ignore[arg-type]
        destination_resume_root="/home/u/run/split-resume",
        expected_identity=IDENTITY,
    )

    assert remote_bundle.endswith(inventory.snapshot_id)
    assert destination.renames
    assert (destination._path(remote_bundle) / "inventory.json").is_file()
    assert (destination._path(remote_bundle) / "state.json").is_file()


def test_retrieve_rejects_remote_state_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seagate = tmp_path / "Seagate"
    seagate.mkdir()
    monkeypatch.setattr(
        split_relay,
        "list_remote_dir",
        lambda *_args: [RemoteEntry("state.json", "symlink")],
    )

    with pytest.raises(split_relay.SplitRelayError, match="state.json"):
        split_relay.retrieve_to_seagate(
            source=_LocalTransfer(tmp_path),  # type: ignore[arg-type]
            source_work_root="/home/u/run/work",
            destination_root=seagate,
            run_id=str(IDENTITY["run_id"]),
            expected_identity=IDENTITY,
        )


def test_retrieve_includes_partial_files_and_reuses_content_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_remote = tmp_path / "source"
    work = source_remote / "home/u/run/work"
    _write_state(work)
    _write_partial(work)
    transfer = _LocalTransfer(source_remote)
    seagate = tmp_path / "Seagate"
    seagate.mkdir()
    monkeypatch.setattr(
        split_relay,
        "list_remote_dir",
        lambda _target, remote: _entries(transfer._path(remote)),
    )

    first = split_relay.retrieve_to_seagate(
        source=transfer,  # type: ignore[arg-type]
        source_work_root="/home/u/run/work",
        destination_root=seagate,
        run_id=str(IDENTITY["run_id"]),
        expected_identity=IDENTITY,
    )
    second = split_relay.retrieve_to_seagate(
        source=transfer,  # type: ignore[arg-type]
        source_work_root="/home/u/run/work",
        destination_root=seagate,
        run_id=str(IDENTITY["run_id"]),
        expected_identity=IDENTITY,
    )

    assert first.root == second.root
    assert first.partial_shard == SHARD


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        ([[]], "missing state"),
        (
            [
                [RemoteEntry("state.json", "file"), RemoteEntry("shards", "dir")],
                [RemoteEntry("partial", "symlink")],
            ],
            "partial root is unsafe",
        ),
        (
            [
                [RemoteEntry("state.json", "file"), RemoteEntry("shards", "dir")],
                [RemoteEntry("partial", "dir")],
                [RemoteEntry("one", "dir"), RemoteEntry("two", "dir")],
            ],
            "at most one partial shard",
        ),
        (
            [
                [RemoteEntry("state.json", "file"), RemoteEntry("shards", "dir")],
                [RemoteEntry("partial", "dir")],
                [RemoteEntry(SHARD, "dir")],
                [RemoteEntry("progress.json", "symlink")],
            ],
            "unsafe entries",
        ),
    ],
)
def test_retrieve_rejects_remote_layout_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    responses: list[list[RemoteEntry]],
    message: str,
) -> None:
    remaining = list(responses)
    monkeypatch.setattr(
        split_relay,
        "list_remote_dir",
        lambda *_args: remaining.pop(0),
    )
    seagate = tmp_path / "Seagate"
    seagate.mkdir()
    with pytest.raises(split_relay.SplitRelayError, match=message):
        split_relay.retrieve_to_seagate(
            source=_LocalTransfer(tmp_path),  # type: ignore[arg-type]
            source_work_root="/home/u/run/work",
            destination_root=seagate,
            run_id=str(IDENTITY["run_id"]),
            expected_identity=IDENTITY,
        )


def test_partial_listing_handles_missing_or_empty_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transfer = _LocalTransfer(tmp_path)
    monkeypatch.setattr(
        split_relay,
        "list_remote_dir",
        lambda *_args: (_ for _ in ()).throw(subprocess.CalledProcessError(1, [])),
    )
    assert split_relay._partial_files(
        source=transfer,
        source_work_root="/home/u/run/work",  # type: ignore[arg-type]
    ) == (None, ())

    monkeypatch.setattr(split_relay, "list_remote_dir", lambda *_args: [])
    assert split_relay._partial_files(
        source=transfer,
        source_work_root="/home/u/run/work",  # type: ignore[arg-type]
    ) == (None, ())


def test_stage_reuses_matching_existing_snapshot_and_rejects_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _write_state(source)
    from scripts.streaming.resume_bundle import create_resume_bundle

    inventory = create_resume_bundle(source, tmp_path / "bundle", IDENTITY)
    remote = tmp_path / "remote"
    transfer = _LocalTransfer(remote)
    monkeypatch.setattr(
        split_relay,
        "list_remote_dir",
        lambda _target, path: _entries(transfer._path(path)),
    )
    first = split_relay.stage_to_destination(
        inventory=inventory,
        destination=transfer,  # type: ignore[arg-type]
        destination_resume_root="/home/u/run/split-resume",
        expected_identity=IDENTITY,
    )
    assert (
        split_relay.stage_to_destination(
            inventory=inventory,
            destination=transfer,  # type: ignore[arg-type]
            destination_resume_root="/home/u/run/split-resume",
            expected_identity=IDENTITY,
        )
        == first
    )
    transfer._path(first + "/inventory.json").write_text("different")
    with pytest.raises(split_relay.SplitRelayError, match="non-empty"):
        split_relay.stage_to_destination(
            inventory=inventory,
            destination=transfer,  # type: ignore[arg-type]
            destination_resume_root="/home/u/run/split-resume",
            expected_identity=IDENTITY,
        )


def test_stage_wraps_transfer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _write_state(source)
    from scripts.streaming.resume_bundle import create_resume_bundle

    inventory = create_resume_bundle(source, tmp_path / "bundle", IDENTITY)

    class BrokenTransfer(_LocalTransfer):
        def push(self, local: Path, remote: str) -> None:
            raise OSError("disk full")

    transfer = BrokenTransfer(tmp_path / "remote")
    monkeypatch.setattr(split_relay, "list_remote_dir", lambda *_args: [])
    with pytest.raises(split_relay.SplitRelayError, match="staging failed"):
        split_relay.stage_to_destination(
            inventory=inventory,
            destination=transfer,  # type: ignore[arg-type]
            destination_resume_root="/home/u/run/split-resume",
            expected_identity=IDENTITY,
        )


def test_partial_listing_accepts_empty_partial_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transfer = _LocalTransfer(tmp_path)
    responses = [[RemoteEntry("partial", "dir")], []]
    monkeypatch.setattr(split_relay, "list_remote_dir", lambda *_args: responses.pop(0))
    assert split_relay._partial_files(
        source=transfer,
        source_work_root="/home/u/run/work",  # type: ignore[arg-type]
    ) == (None, ())


def test_retrieve_wraps_remote_inspection_and_copy_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seagate = tmp_path / "Seagate"
    seagate.mkdir()
    transfer = _LocalTransfer(tmp_path)
    monkeypatch.setattr(
        split_relay,
        "list_remote_dir",
        lambda *_args: (_ for _ in ()).throw(subprocess.CalledProcessError(1, [])),
    )
    with pytest.raises(split_relay.SplitRelayError, match="inspect"):
        split_relay.retrieve_to_seagate(
            source=transfer,  # type: ignore[arg-type]
            source_work_root="/home/u/run/work",
            destination_root=seagate,
            run_id=str(IDENTITY["run_id"]),
            expected_identity=IDENTITY,
        )

    monkeypatch.setattr(
        split_relay,
        "list_remote_dir",
        lambda _target, remote: (
            [RemoteEntry("state.json", "file")] if remote.endswith("/work") else []
        ),
    )
    with pytest.raises(split_relay.SplitRelayError, match="retrieval failed"):
        split_relay.retrieve_to_seagate(
            source=transfer,  # type: ignore[arg-type]
            source_work_root="/home/u/run/work",
            destination_root=seagate,
            run_id=str(IDENTITY["run_id"]),
            expected_identity=IDENTITY,
        )


def test_stage_handles_missing_remote_path_and_manifest_fetch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _write_state(source)
    from scripts.streaming.resume_bundle import create_resume_bundle

    inventory = create_resume_bundle(source, tmp_path / "bundle", IDENTITY)
    transfer = _LocalTransfer(tmp_path / "remote")
    calls = 0

    def list_once(*_args: object) -> list[RemoteEntry]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.CalledProcessError(1, [])
        return []

    monkeypatch.setattr(split_relay, "list_remote_dir", list_once)
    remote_path = split_relay.stage_to_destination(
        inventory=inventory,
        destination=transfer,  # type: ignore[arg-type]
        destination_resume_root="/home/u/run/split-resume",
        expected_identity=IDENTITY,
    )
    assert remote_path.endswith(inventory.snapshot_id)

    class BrokenFetch(_LocalTransfer):
        def fetch(self, remote: str, local: Path) -> None:
            raise OSError("unreadable")

    broken = BrokenFetch(tmp_path / "remote")
    assert not split_relay._remote_manifest_matches(broken, remote_path, inventory)  # type: ignore[arg-type]
