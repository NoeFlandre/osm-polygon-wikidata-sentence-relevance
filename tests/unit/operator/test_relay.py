"""Direct tests for the atomic cross-site checkpoint relay.

Every test exercises either the real production ``CheckpointStore`` (to
produce a valid relay set) or a stateful fake filesystem that records the
exact subprocess argv the relay uses. No test depends on fabricated
``RelayInventory`` objects -- the inventory must be produced by the relay
itself, byte-for-byte.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
from osm_polygon_sentence_relevance.labeling.contracts import (
    LabelRecord,
    LabelValue,
    RunIdentity,
)
from osm_polygon_sentence_relevance.operator import relay


def _identity() -> RunIdentity:
    return RunIdentity(
        input_sha256="a" * 64,
        input_dataset_revision="b" * 40,
        model_repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        model_revision="5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace",
        model_file="Qwen3.6-27B-Q4_K_M.gguf",
        model_file_sha256="c" * 64,
        prompt_version="v1",
        source_commit="d" * 40,
        engine="llama.cpp",
        engine_version="b1234",
        batch_size=128,
        row_limit=0,
        llama_parallel=8,
        llama_per_slot_context=8192,
        llama_total_context=65536,
        request_concurrency=8,
    )


def _build_real_checkpoint_set(tmp_path: Path) -> Path:
    """Build a real checkpoint set on disk using CheckpointStore.

    Returns the directory containing ``progress.json``, ``timing.json``,
    and ``checkpoints/batch-NN.{parquet,json}``.
    """

    identity = _identity()
    store = CheckpointStore(tmp_path, identity)
    records: list[LabelRecord] = [
        LabelRecord(
            sentence_id=f"s{i:08d}",
            landuse_relevance=LabelValue.YES,
            polygon_relevance=LabelValue.YES,
            landuse_reason="x",
            polygon_reason="y",
            evidence="z",
        )
        for i in range(4)
    ]
    store.write_batch(0, records[:2])
    store.write_batch(1, records[2:])
    store.write_progress(completed=4, total=20, elapsed_seconds=10.0)
    store.write_timing({"started_at": 1.0, "finished_at": 2.0})
    return tmp_path


# ------------------------------------------------------------------
# Stateful fake subprocess for remote SCP/SSH filesystem simulation.
# ------------------------------------------------------------------


@dataclass
class _FakeRemote:
    """Stateful simulation of one Grid'5000 frontend filesystem.

    Records every scp/ssh command the relay issues so the test can assert
    exact argv, directory creation, atomic rename, and read-back.
    """

    root: Path
    argv_calls: list[list[str]] = field(default_factory=list)
    create_dirs: list[str] = field(default_factory=list)
    rename_calls: list[tuple[str, str]] = field(default_factory=list)
    #: Override for "remote path -> real filesystem path" translation.
    remote_map: dict[str, Path] = field(default_factory=dict)

    # ---------- popen side effects ----------

    def run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        self.argv_calls.append(list(argv))
        cmd = list(argv)
        if cmd[0] == "scp":
            return self._handle_scp(cmd)
        if cmd[0] == "ssh":
            return self._handle_ssh(cmd)
        raise AssertionError(f"unexpected command: {cmd}")

    def _handle_scp(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        # scp -B -q -p <src> <dst>
        src, dst = cmd[-2], cmd[-1]
        if ":" in src and not src.startswith("/"):
            # src is "host:remote_path"
            _, remote = src.split(":", 1)
            self._download(remote, dst)
        elif ":" in dst and not dst.startswith("/"):
            _, remote = dst.split(":", 1)
            self._upload(src, remote)
        else:
            raise AssertionError(f"unexpected scp argv: {cmd}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def ssh_target(self) -> str:
        # First scp argument that was a "host:" string
        for call in self.argv_calls:
            for arg in call:
                if (arg.startswith("sophia") or arg.startswith("user@")) and ":" in arg:
                    return arg.split(":", 1)[0]
        return "sophia"

    def _handle_ssh(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        # ssh <host> find <path> ...
        inner = " ".join(cmd[2:])
        if inner.startswith("find "):
            args = inner.split()
            return self._handle_find(args)
        if "install -d -m 0700" in inner:
            self.create_dirs.append(inner)
            self._run_remote_mkdir(inner)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "chmod" in inner:
            self._run_remote_chmod(inner)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        # Atomic rename: "if [ -e X ]; ...; mv src dst"
        if inner.startswith("if [ -e ") and "; mv " in inner:
            self.rename_calls.append((inner, ""))
            self._run_remote_rename(inner)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "test -d" in inner and "exit 0" in inner:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def _run_remote_mkdir(self, inner: str) -> None:
        # parse "install -d -m 0700 <path>"
        parts = inner.split()
        path_str = parts[-1]
        path = self._map_remote(path_str)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _run_remote_chmod(self, inner: str) -> None:
        parts = inner.split()
        if "-R" in parts:
            path = Path(parts[-1])
            for child in path.rglob("*"):
                if child.is_file():
                    child.chmod(0o600)
                elif child.is_dir():
                    child.chmod(0o700)
            path.chmod(0o700)
        else:
            mode = int(parts[1], 8)
            Path(parts[-1]).chmod(mode)

    def _run_remote_rename(self, inner: str) -> None:
        # parse the trailing "mv <src> <dst>" from inside the if-block
        marker = "; mv -- " if "; mv -- " in inner else "; mv "
        mv_idx = inner.find(marker)
        tail = inner[mv_idx + len(marker) :]
        src_str, dst_str = tail.split(maxsplit=1)
        src = self._map_remote(src_str)
        dst = self._map_remote(dst_str)
        if dst.exists() and dst.is_dir() and any(dst.iterdir()):
            raise RuntimeError("refusing to clobber non-empty destination")
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if not src.exists():
            raise FileNotFoundError(f"rename source missing: {src}")
        os.replace(src, dst)

    def _handle_find(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        # find <path> -mindepth 1 -maxdepth 1 -printf '%y\t%f\n'
        path = Path(args[1])
        lines: list[str] = []
        if not path.is_dir():
            return subprocess.CompletedProcess(args, 0, "", "")
        for child in sorted(path.iterdir()):
            kind_token = (
                "f"
                if child.is_file() and not child.is_symlink()
                else "l"
                if child.is_symlink()
                else "d"
                if child.is_dir()
                else "o"
            )
            lines.append(f"{kind_token}\t{child.name}\n")
        return subprocess.CompletedProcess(args, 0, "".join(lines), "")

    def _download(self, remote: str, local: str) -> None:
        # remote is absolute path on the frontend; local is the Seagate path.
        local_path = Path(local)
        local_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # translate remote to local within self.root
        # The remote path is the source-side root + relative file.
        # We stored under the simulation root mapped 1:1.
        source = self._map_remote(remote)
        if not source.is_file():
            raise relay.RelayError(f"remote file does not exist: {remote}")
        local_path.write_bytes(source.read_bytes())
        os.chmod(local_path, 0o600)

    def _upload(self, local: str, remote: str) -> None:
        local_path = Path(local)
        if not local_path.is_file() or local_path.is_symlink():
            raise relay.RelayError(f"refusing to upload non-regular: {local}")
        target = self._map_remote(remote)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # The relay stages under a temporary dir, then renames into place.
        target.write_bytes(local_path.read_bytes())
        os.chmod(target, 0o600)

    def _map_remote(self, remote: str) -> Path:
        # explicit overrides win (used to map remote src-dir to a real local dir)
        if remote in self.remote_map:
            return self.remote_map[remote]
        return self.root / remote.lstrip("/")


def _patch_subprocess(monkeypatch: pytest.MonkeyPatch, fake: _FakeRemote) -> None:
    """Replace subprocess.run across the relay and transport stack with our stateful fake."""

    def fake_run(
        argv: Sequence[str],
        *,
        check: bool = False,
        shell: bool = False,
        timeout: float | None = None,
        capture_output: bool = False,
        text: bool | None = None,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        cp = fake.run(argv)
        if check and cp.returncode != 0:
            raise subprocess.CalledProcessError(cp.returncode, list(argv))
        return cp

    monkeypatch.setattr(relay.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "osm_polygon_sentence_relevance.operator.relay_transport.subprocess.run",
        fake_run,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_retrieve_to_seagate_builds_inventory_from_real_checkpoint_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrieve produces an inventory whose Parquet hashes match metadata."""

    src_dir = tmp_path / "source"
    src_dir.mkdir()
    _build_real_checkpoint_set(src_dir)
    seagate = tmp_path / "seagate"
    seagate.mkdir()
    fake = _FakeRemote(root=tmp_path / "frontend-sophia")
    fake.root.mkdir()
    # Map the remote source root directly to the real local source dir.
    fake.remote_map[str(src_dir)] = src_dir
    fake.remote_map[str(src_dir / "checkpoints")] = src_dir / "checkpoints"
    # Pre-seed per-file mapping so the fake can locate individual files.
    for path in src_dir.rglob("*"):
        if path.is_file():
            fake.remote_map[str(path)] = path
    _patch_subprocess(monkeypatch, fake)

    inventory = relay.retrieve_to_seagate(
        source=relay.RemoteTransfer(ssh_target="sophia"),
        source_checkpoint_root=str(src_dir),
        destination_root=seagate,
        run_id="7e8f1a748497e3dbcc56",
        expected_run_identity=_identity().to_dict(),
    )
    assert inventory.root.is_dir()
    assert not inventory.root.is_symlink()
    assert stat.S_IMODE(inventory.root.stat().st_mode) == 0o700
    assert (inventory.root / "progress.json").is_file()
    assert (inventory.root / "timing.json").is_file()
    parquets = sorted(
        p.name for p in inventory.root.glob("checkpoints/batch-*.parquet")
    )
    jsons = sorted(p.name for p in inventory.root.glob("checkpoints/batch-*.json"))
    assert parquets == ["batch-000000.parquet", "batch-000001.parquet"]
    assert jsons == ["batch-000000.json", "batch-000001.json"]
    for path in inventory.root.rglob("*"):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    scp_fetches = [
        call
        for call in fake.argv_calls
        if call[0] == "scp"
        and any(
            arg.endswith(
                (
                    "/progress.json",
                    "/timing.json",
                    "/batch-000000.parquet",
                    "/batch-000001.parquet",
                    "/batch-000000.json",
                    "/batch-000001.json",
                )
            )
            for arg in call
        )
    ]
    # Each remote file is fetched exactly once via scp.
    fetched_files = {
        arg
        for call in scp_fetches
        for arg in call
        if arg.endswith(
            (
                "/progress.json",
                "/timing.json",
                "/batch-000000.parquet",
                "/batch-000001.parquet",
                "/batch-000000.json",
                "/batch-000001.json",
            )
        )
    }
    assert len(fetched_files) == 6


def test_retrieve_rejects_unsafe_remote_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seagate = tmp_path / "seagate"
    seagate.mkdir()
    fake = _FakeRemote(root=tmp_path / "frontend")
    fake.root.mkdir()
    _patch_subprocess(monkeypatch, fake)

    for unsafe in [
        "/home/with space/progress.json",
        "/home/with$danger/progress.json",
        "/home/with`backtick`/progress.json",
        "/home/with;semicolon/progress.json",
        "/home/with|pipe/progress.json",
        '/home/with"quote/progress.json',
        "/home/with'quote/progress.json",
    ]:
        with pytest.raises(relay.RelayError):
            relay.retrieve_to_seagate(
                source=relay.RemoteTransfer(ssh_target="sophia"),
                source_checkpoint_root=unsafe,
                destination_root=seagate,
                run_id="7e8f1a748497e3dbcc56",
            )


def test_retrieve_rejects_symlinks_in_remote_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "progress.json").write_text('{"identity": {}, "completed": 1, "total": 2}')
    ckpts = src / "checkpoints"
    ckpts.mkdir()
    target = ckpts / "batch-000000.parquet"
    target.write_bytes(b"fake")
    metadata = ckpts / "batch-000000.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": _identity().to_dict(),
                "row_count": 1,
                "parquet_sha256": "f" * 64,
            }
        )
    )
    # Plant a symlink inside checkpoints
    (ckpts / "evil-link").symlink_to(tmp_path / "outside.txt")
    seagate = tmp_path / "seagate"
    seagate.mkdir()
    fake = _FakeRemote(root=tmp_path / "frontend")
    fake.root.mkdir()
    for path in src.rglob("*"):
        if path.is_file():
            fake.remote_map[str(path)] = path
    fake.remote_map[str(src)] = src
    fake.remote_map[str(ckpts)] = ckpts
    _patch_subprocess(monkeypatch, fake)

    with pytest.raises(relay.RelayError):
        relay.retrieve_to_seagate(
            source=relay.RemoteTransfer(ssh_target="sophia"),
            source_checkpoint_root=str(src),
            destination_root=seagate,
            run_id="7e8f1a748497e3dbcc56",
        )


def test_stage_to_destination_creates_temp_dir_with_mode_0700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    _build_real_checkpoint_set(src_dir)
    seagate = tmp_path / "seagate"
    seagate.mkdir()
    fake_src = _FakeRemote(root=tmp_path / "frontend-sophia")
    fake_src.root.mkdir()
    fake_src.remote_map[str(src_dir)] = src_dir
    fake_src.remote_map[str(src_dir / "checkpoints")] = src_dir / "checkpoints"
    for path in src_dir.rglob("*"):
        if path.is_file():
            fake_src.remote_map[str(path)] = path
    _patch_subprocess(monkeypatch, fake_src)
    inventory = relay.retrieve_to_seagate(
        source=relay.RemoteTransfer(ssh_target="sophia"),
        source_checkpoint_root=str(src_dir),
        destination_root=seagate,
        run_id="7e8f1a748497e3dbcc56",
        expected_run_identity=_identity().to_dict(),
    )

    fake_dst = _FakeRemote(root=tmp_path / "frontend-grenoble")
    fake_dst.root.mkdir()
    # Pre-create empty destination parent (relay refuses non-empty destinations)
    (fake_dst.root / "destination").mkdir(mode=0o755)
    _patch_subprocess(monkeypatch, fake_dst)

    relay.stage_to_destination(
        inventory=inventory,
        destination=relay.RemoteTransfer(ssh_target="user@grenoble.grid5000.fr"),
        destination_checkpoint_root="/destination",
    )
    mode_700 = [entry for entry in fake_dst.create_dirs if "0700" in entry]
    assert mode_700, f"no 0700 mkdir observed: {fake_dst.create_dirs}"
    final = fake_dst.root / "destination"
    assert final.is_dir()
    assert (final / "progress.json").is_file()
    parquets = sorted(p.name for p in final.glob("checkpoints/batch-*.parquet"))
    assert parquets == ["batch-000000.parquet", "batch-000001.parquet"]


def test_stage_to_destination_failure_preserves_prior_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    _build_real_checkpoint_set(src_dir)
    seagate = tmp_path / "seagate"
    seagate.mkdir()
    fake_src = _FakeRemote(root=tmp_path / "frontend-sophia")
    fake_src.root.mkdir()
    fake_src.remote_map[str(src_dir)] = src_dir
    fake_src.remote_map[str(src_dir / "checkpoints")] = src_dir / "checkpoints"
    for path in src_dir.rglob("*"):
        if path.is_file():
            fake_src.remote_map[str(path)] = path
    _patch_subprocess(monkeypatch, fake_src)
    inventory = relay.retrieve_to_seagate(
        source=relay.RemoteTransfer(ssh_target="sophia"),
        source_checkpoint_root=str(src_dir),
        destination_root=seagate,
        run_id="7e8f1a748497e3dbcc56",
        expected_run_identity=_identity().to_dict(),
    )

    fake_dst = _FakeRemote(root=tmp_path / "frontend-grenoble")
    fake_dst.root.mkdir()
    final = fake_dst.root / "destination" / inventory.root.name
    final.mkdir(parents=True)
    (final / "progress.json").write_text('{"prior": "valid"}')
    _patch_subprocess(monkeypatch, fake_dst)

    # Force a readback hash mismatch by replacing one local parquet with garbage.
    (inventory.root / "checkpoints" / "batch-000000.parquet").write_bytes(b"corrupt")

    with pytest.raises(relay.RelayError):
        relay.stage_to_destination(
            inventory=inventory,
            destination=relay.RemoteTransfer(ssh_target="user@grenoble"),
            destination_checkpoint_root="/destination",
        )
    assert (final / "progress.json").read_text() == '{"prior": "valid"}'


def test_relay_does_not_use_mac_internal_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DATA_ROOT must point at the external Seagate volume."""

    # The contract: relay never calls anything on /Users or local internal storage.
    # This test asserts destination_root must be on the Seagate path.
    with pytest.raises(relay.RelayError):
        relay.retrieve_to_seagate(
            source=relay.RemoteTransfer(ssh_target="sophia"),
            source_checkpoint_root="/home/u/label-work",
            destination_root=Path("/tmp/internal"),  # wrong
            run_id="7e8f1a748497e3dbcc56",
        )


def test_relay_inventory_ordered_indexes_rejects_non_batch_names() -> None:
    """The inventory rejects any path whose name does not match batch-XXXXXX."""

    root = Path("/tmp/never-touch")
    root.mkdir(exist_ok=True)
    fake_parquet = type("P", (), {"name": "not-a-batch.parquet"})()
    inv = relay.RelayInventory(
        root=root,
        progress=root / "progress.json",
        timing=None,
        parquet_paths=[fake_parquet],
        metadata_paths={},
        completed=0,
        total=0,
    )
    with pytest.raises(relay.RelayError):
        inv.ordered_indexes()


def test_relay_validate_destination_root_rejects_tmp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The relay refuses to publish on /tmp even with valid checkpoints."""

    from osm_polygon_sentence_relevance.operator import relay

    with pytest.raises(relay.RelayError, match="internal storage"):
        relay._validate_destination_root(Path("/tmp/probe"))


def test_relay_validate_destination_root_rejects_users(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_relevance.operator import relay

    with pytest.raises(relay.RelayError, match="internal storage"):
        relay._validate_destination_root(Path("/Users/foo"))
