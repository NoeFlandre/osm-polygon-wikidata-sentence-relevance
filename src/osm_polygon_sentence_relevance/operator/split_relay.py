"""Cross-site transport for small, validated split resume bundles."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path

from scripts.streaming.resume_bundle import (
    MANIFEST_NAME,
    ResumeBundle,
    ResumeBundleError,
    create_resume_bundle,
    validate_resume_bundle,
)

from osm_polygon_sentence_relevance.operator.relay import _validate_destination_root
from osm_polygon_sentence_relevance.operator.relay_transport import (
    FILE_MODE,
    RELAY_DIR_MODE,
    RemoteEntry,
    RemoteTransfer,
    list_remote_dir,
    validate_safe_remote_path,
)


class SplitRelayError(RuntimeError):
    """A split resume snapshot could not be transferred safely."""


def _entry(entries: list[RemoteEntry], name: str) -> RemoteEntry | None:
    return next((item for item in entries if item.name == name), None)


def _require_remote_state(entries: list[RemoteEntry]) -> None:
    state = _entry(entries, "state.json")
    if state is None:
        raise SplitRelayError("remote split work is missing state.json")
    if state.kind != "file":
        raise SplitRelayError("remote split state.json is not a regular file")


def _partial_files(
    *, source: RemoteTransfer, source_work_root: str
) -> tuple[str | None, tuple[str, ...]]:
    shards_root = f"{source_work_root.rstrip('/')}/shards"
    try:
        shard_entries = list_remote_dir(source.ssh_target, shards_root)
    except subprocess.CalledProcessError:
        return None, ()
    partial_entry = _entry(shard_entries, "partial")
    if partial_entry is None:
        return None, ()
    if partial_entry.kind != "dir":
        raise SplitRelayError("remote partial root is unsafe")
    partial_root = f"{shards_root}/partial"
    partial_entries = list_remote_dir(source.ssh_target, partial_root)
    if not partial_entries:
        return None, ()
    if len(partial_entries) != 1 or partial_entries[0].kind != "dir":
        raise SplitRelayError(
            "remote split work must contain at most one partial shard"
        )
    shard_key = partial_entries[0].name
    shard_root = f"{partial_root}/{shard_key}"
    entries = list_remote_dir(source.ssh_target, shard_root)
    if not entries or any(item.kind != "file" for item in entries):
        raise SplitRelayError("remote partial shard contains unsafe entries")
    return shard_key, tuple(sorted(item.name for item in entries))


def retrieve_to_seagate(
    *,
    source: RemoteTransfer,
    source_work_root: str,
    destination_root: Path,
    run_id: str,
    expected_identity: Mapping[str, str | int],
) -> ResumeBundle:
    """Fetch the split ledger and active partial shard into a validated bundle."""

    validate_safe_remote_path(source_work_root)
    _validate_destination_root(destination_root)
    try:
        top_entries = list_remote_dir(source.ssh_target, source_work_root)
        _require_remote_state(top_entries)
        partial_shard, partial_names = _partial_files(
            source=source, source_work_root=source_work_root
        )
    except (subprocess.CalledProcessError, ResumeBundleError) as exc:
        raise SplitRelayError("could not inspect remote split work") from exc

    run_root = destination_root / run_id
    run_root.mkdir(parents=True, mode=RELAY_DIR_MODE, exist_ok=True)
    os.chmod(run_root, RELAY_DIR_MODE)
    fetched = Path(tempfile.mkdtemp(prefix=".split-fetch.", dir=run_root))
    os.chmod(fetched, RELAY_DIR_MODE)
    pending = run_root / f".split-bundle-{uuid.uuid4().hex}"
    try:
        state_path = fetched / "state.json"
        source.fetch(f"{source_work_root.rstrip('/')}/state.json", state_path)
        os.chmod(state_path, FILE_MODE)
        if partial_shard is not None:
            local_partial = fetched / "shards" / "partial" / partial_shard
            local_partial.mkdir(parents=True, mode=RELAY_DIR_MODE)
            os.chmod(local_partial, RELAY_DIR_MODE)
            for name in partial_names:
                target = local_partial / name
                source.fetch(
                    f"{source_work_root.rstrip('/')}/shards/partial/"
                    f"{partial_shard}/{name}",
                    target,
                )
                os.chmod(target, FILE_MODE)
        created = create_resume_bundle(fetched, pending, expected_identity)
        final_parent = run_root / "split-relay"
        final_parent.mkdir(mode=RELAY_DIR_MODE, exist_ok=True)
        os.chmod(final_parent, RELAY_DIR_MODE)
        final = final_parent / created.snapshot_id
        if final.exists():
            existing = validate_resume_bundle(final, expected_identity)
            if existing.snapshot_id != created.snapshot_id:
                raise SplitRelayError("content-addressed split bundle collision")
            shutil.rmtree(created.root)
        else:
            os.replace(created.root, final)
        return validate_resume_bundle(final, expected_identity)
    except (OSError, ResumeBundleError) as exc:
        raise SplitRelayError(f"split resume retrieval failed: {exc}") from exc
    finally:
        shutil.rmtree(fetched, ignore_errors=True)
        if pending.exists():
            shutil.rmtree(pending, ignore_errors=True)


def stage_to_destination(
    *,
    inventory: ResumeBundle,
    destination: RemoteTransfer,
    destination_resume_root: str,
    expected_identity: Mapping[str, str | int],
) -> str:
    """Push, read back, and atomically publish one remote resume bundle."""

    validate_safe_remote_path(destination_resume_root)
    inventory = validate_resume_bundle(inventory.root, expected_identity)
    final = f"{destination_resume_root.rstrip('/')}/{inventory.snapshot_id}"
    try:
        existing = list_remote_dir(destination.ssh_target, final)
    except subprocess.CalledProcessError:
        existing = []
    if existing:
        if _remote_manifest_matches(destination, final, inventory):
            return final
        raise SplitRelayError("destination split bundle path is non-empty")

    temp = f"{destination_resume_root.rstrip('/')}/.staging-{uuid.uuid4().hex}"
    destination.ssh_mkdir_0700(destination_resume_root)
    destination.ssh_mkdir_0700(temp)
    local_paths = [inventory.root / MANIFEST_NAME]
    local_paths.extend(inventory.root / path for path in inventory.relative_files)
    try:
        for local_path in local_paths:
            relative = local_path.relative_to(inventory.root).as_posix()
            remote_path = f"{temp}/{relative}"
            remote_parent = remote_path.rsplit("/", 1)[0]
            destination.ssh_mkdir_0700(remote_parent)
            destination.push(local_path, remote_path)
        verification = Path(
            tempfile.mkdtemp(prefix=".split-readback.", dir=inventory.root.parent)
        )
        os.chmod(verification, RELAY_DIR_MODE)
        try:
            for local_path in local_paths:
                relative = local_path.relative_to(inventory.root)
                fetched = verification / relative
                fetched.parent.mkdir(parents=True, mode=RELAY_DIR_MODE, exist_ok=True)
                os.chmod(fetched.parent, RELAY_DIR_MODE)
                destination.fetch(f"{temp}/{relative.as_posix()}", fetched)
                os.chmod(fetched, FILE_MODE)
            validate_resume_bundle(verification, expected_identity)
        finally:
            shutil.rmtree(verification, ignore_errors=True)
        destination.ssh_atomic_rename(temp, final)
        return final
    except (OSError, ResumeBundleError, subprocess.CalledProcessError) as exc:
        raise SplitRelayError(f"split resume staging failed: {exc}") from exc


def _remote_manifest_matches(
    destination: RemoteTransfer,
    final: str,
    inventory: ResumeBundle,
) -> bool:
    with tempfile.TemporaryDirectory(
        prefix=".split-existing.", dir=inventory.root.parent
    ) as directory:
        fetched = Path(directory) / MANIFEST_NAME
        try:
            destination.fetch(f"{final}/{MANIFEST_NAME}", fetched)
        except Exception:
            return False
        return fetched.read_bytes() == (inventory.root / MANIFEST_NAME).read_bytes()


__all__ = [
    "SplitRelayError",
    "retrieve_to_seagate",
    "stage_to_destination",
]
