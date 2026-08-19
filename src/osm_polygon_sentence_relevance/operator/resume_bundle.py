"""Validated, content-addressed cross-site split resume bundles.

The authoritative completed-shard data remains on Hugging Face.  This module
moves only the small local acceleration state: the verified-checkpoint ledger
and at most one crash-safe partial shard.  Bundles are identity-bound, hashed,
safe to merge repeatedly, and never contain input Parquet files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_sentence_relevance.application._checkpoint.common import (
    CheckpointValidationError,
)
from osm_polygon_sentence_relevance.application._checkpoint.inventory import (
    SourceFileEntry,
)
from osm_polygon_sentence_relevance.application._checkpoint.partial import (
    PartialShardState,
    load_partial_state,
)

DIR_MODE = 0o700
FILE_MODE = 0o600
MANIFEST_NAME = "inventory.json"
STATE_NAME = "state.json"
_SCHEMA_VERSION = 1


class ResumeBundleError(RuntimeError):
    """A split resume bundle is malformed, unsafe, or divergent."""


@dataclass(frozen=True, slots=True)
class ResumeBundle:
    """A validated local snapshot ready for cross-site transport."""

    root: Path
    snapshot_id: str
    state_path: Path
    partial_shard: str | None
    relative_files: tuple[Path, ...]
    completed_shards: int


@dataclass(frozen=True, slots=True)
class ResumeMergeResult:
    """Outcome of importing one bundle into a persistent split work root."""

    imported: bool
    completed_shards: int
    partial_shard: str | None


def valid_streaming_shard_key(value: object) -> bool:
    """Return whether ``value`` follows the production shard-key grammar."""

    return bool(
        isinstance(value, str)
        and value
        and all(
            (char.isascii() and char.isalnum() and char == char.lower())
            or char in "-_."
            for char in value
        )
    )


def validate_streaming_state(
    payload: object,
    expected_identity: Mapping[str, str | int],
) -> dict[str, dict[str, str | int]]:
    """Validate a driver state payload and return its checkpoint ledger."""

    if not isinstance(payload, dict):
        raise ResumeBundleError("state.json must contain a JSON object")
    for field, expected in expected_identity.items():
        if payload.get(field) != expected:
            raise ResumeBundleError(f"state.json {field} does not match this run")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ResumeBundleError("state.json schema version is invalid")
    raw = payload.get("verified_checkpoints")
    if not isinstance(raw, dict):
        raise ResumeBundleError("state.json verified_checkpoints must be an object")
    result: dict[str, dict[str, str | int]] = {}
    for shard_key, descriptor in raw.items():
        if not valid_streaming_shard_key(shard_key) or not isinstance(descriptor, dict):
            raise ResumeBundleError("state.json checkpoint ledger is malformed")
        if set(descriptor) != {
            "segmented_table_sha256",
            "segmented_table_bytes",
        }:
            raise ResumeBundleError("state.json checkpoint descriptor is malformed")
        digest = descriptor.get("segmented_table_sha256")
        size = descriptor.get("segmented_table_bytes")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise ResumeBundleError("state.json checkpoint descriptor is malformed")
        result[str(shard_key)] = {
            "segmented_table_sha256": digest,
            "segmented_table_bytes": size,
        }
    return result


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    _ensure_regular(path, FILE_MODE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeBundleError(f"{label} is malformed") from exc
    if not isinstance(payload, dict):
        raise ResumeBundleError(f"{label} must contain a JSON object")
    return payload


def _load_state(
    root: Path, expected_identity: Mapping[str, str | int]
) -> tuple[dict[str, Any], dict[str, dict[str, str | int]]]:
    payload = _load_json_mapping(root / STATE_NAME, STATE_NAME)
    return payload, validate_streaming_state(payload, expected_identity)


def _source_entries(value: object) -> list[SourceFileEntry]:
    if not isinstance(value, list) or not value:
        raise ResumeBundleError("partial source manifest is malformed")
    entries: list[SourceFileEntry] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise ResumeBundleError("partial source manifest is malformed")
        path = item.get("path")
        size = item.get("size")
        digest = item.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ResumeBundleError("partial source manifest is malformed")
        entries.append(SourceFileEntry(path, size, digest))
    return entries


def _partial_directories(work_root: Path) -> list[Path]:
    parent = work_root / "shards" / "partial"
    if not parent.exists():
        return []
    _ensure_directory(parent)
    result: list[Path] = []
    for path in parent.iterdir():
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ResumeBundleError("partial root contains an unsafe entry")
        if not valid_streaming_shard_key(path.name):
            raise ResumeBundleError("partial root contains an invalid shard key")
        result.append(path)
    if len(result) > 1:
        raise ResumeBundleError("split work may contain at most one partial shard")
    return result


def _validate_partial(
    work_root: Path,
    expected_identity: Mapping[str, str | int],
) -> PartialShardState | None:
    directories = _partial_directories(work_root)
    if not directories:
        return None
    shard_key = directories[0].name
    progress = _load_json_mapping(directories[0] / "progress.json", "partial progress")
    input_root = progress.get("input_root")
    total_sections = progress.get("total_sections")
    if not isinstance(input_root, str) or not Path(input_root).is_absolute():
        raise ResumeBundleError("partial input_root is invalid")
    if (
        isinstance(total_sections, bool)
        or not isinstance(total_sections, int)
        or total_sections < 0
    ):
        raise ResumeBundleError("partial total_sections is invalid")
    source_files = _source_entries(progress.get("source_files"))
    try:
        state = load_partial_state(
            work_root,
            shard_key=shard_key,
            source_commit=str(expected_identity["source_commit"]),
            input_dataset_revision=str(expected_identity["resolved_revision"]),
            pipeline_version=str(expected_identity["pipeline_version"]),
            model_name=str(expected_identity["model_name"]),
            batch_size=int(expected_identity["batch_size"]),
            input_root=Path(input_root),
            source_files=source_files,
            total_sections=total_sections,
        )
    except (CheckpointValidationError, KeyError, TypeError, ValueError) as exc:
        raise ResumeBundleError(f"partial checkpoint is invalid: {exc}") from exc
    assert state is not None
    return state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_source_files(source_work: Path) -> tuple[Path, ...]:
    files = [source_work / STATE_NAME]
    partials = _partial_directories(source_work)
    if partials:
        files.extend(sorted(path for path in partials[0].iterdir()))
    return tuple(files)


def _relative_source_path(source_work: Path, path: Path) -> Path:
    relative = path.relative_to(source_work)
    if relative == Path(STATE_NAME):
        return relative
    if len(relative.parts) != 4 or relative.parts[:2] != ("shards", "partial"):
        raise ResumeBundleError(f"unexpected split resume file: {relative}")
    if not valid_streaming_shard_key(relative.parts[2]):
        raise ResumeBundleError("invalid partial shard path")
    return relative


def _manifest_payload(
    expected_identity: Mapping[str, str | int], files: Mapping[str, str]
) -> tuple[dict[str, object], str]:
    canonical = {
        "schema_version": _SCHEMA_VERSION,
        "identity": dict(expected_identity),
        "files": dict(sorted(files.items())),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    snapshot_id = hashlib.sha256(encoded).hexdigest()[:20]
    return {**canonical, "snapshot_id": snapshot_id}, snapshot_id


def create_resume_bundle(
    source_work: Path,
    destination: Path,
    expected_identity: Mapping[str, str | int],
) -> ResumeBundle:
    """Create and validate one immutable bundle from persistent split work."""

    _ensure_directory(source_work)
    _state, ledger = _load_state(source_work, expected_identity)
    partial = _validate_partial(source_work, expected_identity)
    if partial is not None and partial.shard_key in ledger:
        raise ResumeBundleError("partial shard is already present in completed ledger")
    if destination.exists() or destination.is_symlink():
        raise ResumeBundleError("resume bundle destination already exists")
    destination.parent.mkdir(parents=True, mode=DIR_MODE, exist_ok=True)
    temp = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    os.chmod(temp, DIR_MODE)
    try:
        file_hashes: dict[str, str] = {}
        for source in _bundle_source_files(source_work):
            _ensure_regular(source, FILE_MODE)
            relative = _relative_source_path(source_work, source)
            target = temp / relative
            target.parent.mkdir(parents=True, mode=DIR_MODE, exist_ok=True)
            os.chmod(target.parent, DIR_MODE)
            shutil.copyfile(source, target, follow_symlinks=False)
            os.chmod(target, FILE_MODE)
            file_hashes[relative.as_posix()] = _sha256(target)
        manifest, _snapshot_id = _manifest_payload(expected_identity, file_hashes)
        _atomic_write_json(temp / MANIFEST_NAME, manifest)
        os.replace(temp, destination)
        return validate_resume_bundle(destination, expected_identity)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def validate_resume_bundle(
    root: Path,
    expected_identity: Mapping[str, str | int],
) -> ResumeBundle:
    """Validate bundle identity, inventory, hashes, state, and partial bytes."""

    _ensure_directory(root)
    manifest = _load_json_mapping(root / MANIFEST_NAME, MANIFEST_NAME)
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ResumeBundleError("resume bundle schema version is invalid")
    if manifest.get("identity") != dict(expected_identity):
        raise ResumeBundleError("resume bundle identity does not match this run")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, dict) or STATE_NAME not in raw_files:
        raise ResumeBundleError("resume bundle file inventory is malformed")
    files: dict[str, str] = {}
    for name, digest in raw_files.items():
        relative = Path(name) if isinstance(name, str) else Path(".")
        if (
            not isinstance(name, str)
            or relative.is_absolute()
            or ".." in relative.parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ResumeBundleError("resume bundle file inventory is malformed")
        files[name] = digest
    expected_manifest, snapshot_id = _manifest_payload(expected_identity, files)
    if manifest != expected_manifest:
        raise ResumeBundleError("resume bundle snapshot identity is invalid")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    expected_files = set(files) | {MANIFEST_NAME}
    if actual_files != expected_files:
        raise ResumeBundleError("resume bundle contains unexpected or missing files")
    for relative, expected_hash in files.items():
        path = root / relative
        _ensure_regular(path, FILE_MODE)
        if _sha256(path) != expected_hash:
            raise ResumeBundleError(f"resume bundle hash mismatch: {relative}")
    _state, ledger = _load_state(root, expected_identity)
    partial = _validate_partial(root, expected_identity)
    if partial is not None and partial.shard_key in ledger:
        raise ResumeBundleError("partial shard is already present in completed ledger")
    return ResumeBundle(
        root=root,
        snapshot_id=snapshot_id,
        state_path=root / STATE_NAME,
        partial_shard=partial.shard_key if partial is not None else None,
        relative_files=tuple(Path(name) for name in sorted(files)),
        completed_shards=len(ledger),
    )


def merge_resume_bundle(
    work_root: Path,
    bundle_root: Path,
    expected_identity: Mapping[str, str | int],
) -> ResumeMergeResult:
    """Merge one validated bundle into destination work without losing progress."""

    bundle = validate_resume_bundle(bundle_root, expected_identity)
    bundle_state, imported_ledger = _load_state(bundle.root, expected_identity)
    imported_partial = _validate_partial(bundle.root, expected_identity)
    work_root.mkdir(parents=True, mode=DIR_MODE, exist_ok=True)
    os.chmod(work_root, DIR_MODE)
    destination_state_path = work_root / STATE_NAME
    if destination_state_path.exists() or destination_state_path.is_symlink():
        destination_state, destination_ledger = _load_state(
            work_root, expected_identity
        )
    else:
        destination_state = dict(bundle_state)
        destination_ledger = {}
    merged_ledger = dict(destination_ledger)
    for shard_key, descriptor in imported_ledger.items():
        existing = merged_ledger.get(shard_key)
        if existing is not None and existing != descriptor:
            raise ResumeBundleError(
                f"conflicting checkpoint descriptor for {shard_key}"
            )
        merged_ledger[shard_key] = descriptor

    destination_partial = _validate_partial(work_root, expected_identity)
    final_partial = _merge_partial(
        work_root,
        imported_partial=imported_partial,
        destination_partial=destination_partial,
        merged_ledger=merged_ledger,
    )
    destination_state.update(dict(expected_identity))
    destination_state["schema_version"] = _SCHEMA_VERSION
    destination_state["verified_checkpoints"] = merged_ledger
    destination_state["last_updated"] = bool(
        destination_state.get("last_updated") or bundle_state.get("last_updated")
    )
    _atomic_write_json(destination_state_path, destination_state)
    shutil.rmtree(bundle.root, ignore_errors=True)
    return ResumeMergeResult(
        imported=True,
        completed_shards=len(merged_ledger),
        partial_shard=final_partial,
    )


def _merge_partial(
    work_root: Path,
    *,
    imported_partial: PartialShardState | None,
    destination_partial: PartialShardState | None,
    merged_ledger: Mapping[str, object],
) -> str | None:
    if (
        destination_partial is not None
        and destination_partial.shard_key in merged_ledger
    ):
        shutil.rmtree(destination_partial.directory)
        destination_partial = None
    if imported_partial is None:
        return (
            destination_partial.shard_key if destination_partial is not None else None
        )
    if imported_partial.shard_key in merged_ledger:
        return (
            destination_partial.shard_key if destination_partial is not None else None
        )
    if destination_partial is not None:
        if destination_partial.shard_key != imported_partial.shard_key:
            raise ResumeBundleError("split work contains divergent partial shards")
        destination_batches = [
            (batch.start_index, batch.end_index, batch.sha256)
            for batch in destination_partial.batches
        ]
        imported_batches = [
            (batch.start_index, batch.end_index, batch.sha256)
            for batch in imported_partial.batches
        ]
        overlap = min(len(destination_batches), len(imported_batches))
        if destination_batches[:overlap] != imported_batches[:overlap]:
            raise ResumeBundleError("matching partial shard histories diverge")
        if (
            destination_partial.next_section_index
            >= imported_partial.next_section_index
        ):
            return destination_partial.shard_key
        shutil.rmtree(destination_partial.directory)
    target = work_root / "shards" / "partial" / imported_partial.shard_key
    target.parent.mkdir(parents=True, mode=DIR_MODE, exist_ok=True)
    os.chmod(target.parent, DIR_MODE)
    temp = target.with_name(f".{target.name}.import-{os.getpid()}")
    if temp.exists():
        shutil.rmtree(temp)
    shutil.copytree(imported_partial.directory, temp, symlinks=False)
    for directory in [temp, *[path for path in temp.rglob("*") if path.is_dir()]]:
        os.chmod(directory, DIR_MODE)
    for path in temp.rglob("*"):
        if path.is_file():
            os.chmod(path, FILE_MODE)
    os.replace(temp, target)
    return imported_partial.shard_key


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, mode=DIR_MODE, exist_ok=True)
    os.chmod(path.parent, DIR_MODE)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        descriptor = os.open(
            temp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            FILE_MODE,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)


def _ensure_directory(path: Path) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ResumeBundleError(f"resume directory is inaccessible: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ResumeBundleError(f"resume path is not a real directory: {path}")


def _ensure_regular(path: Path, expected_mode: int) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ResumeBundleError(f"resume file is inaccessible: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ResumeBundleError(f"resume file is not regular: {path}")
    if info.st_mode & 0o777 != expected_mode:
        raise ResumeBundleError(f"resume file has unsafe mode: {path}")


__all__ = [
    "ResumeBundle",
    "ResumeBundleError",
    "ResumeMergeResult",
    "create_resume_bundle",
    "merge_resume_bundle",
    "valid_streaming_shard_key",
    "validate_resume_bundle",
    "validate_streaming_state",
]
