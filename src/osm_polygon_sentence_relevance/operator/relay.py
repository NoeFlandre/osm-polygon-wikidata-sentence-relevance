"""Atomic, identity-bound cross-site checkpoint relay.

The Mac's external Seagate volume is the only durable bridge between
Grid'5000 frontends: each site's ``$HOME`` is local to that site. This module
implements the smallest reliable relay that satisfies the production
contract.

Local Seagate:

* Destination root must point at the production ``DATA_ROOT`` (Seagate). The
  relay refuses to write to ``/tmp``, ``/Users/<user>`` or any other internal
  Mac storage.
* The relay fetches files into a fresh sibling generation under
  ``<destination_root>/<run_id>/relay.generation-<N>/`` using ``scp`` with
  an explicit, validated, non-shell-interpolated argv. No follow on
  symlinks; no shell metacharacters in any path; mode ``0600`` on every
  fetched file; mode ``0700`` on every directory created.
* The fetched generation is fully validated (CheckpointStore hash, schema,
  no-dup, identity) before any other generation is touched.
* A small atomic ``current`` pointer (symlink or text) is published only
  after the generation is independently fsync-ed.
* The prior generation is preserved until the new one has been independently
  re-validated.

Remote destination:

* ``SshClient`` is used to create a fresh operator-managed temporary work
  directory on the destination site with mode ``0700``. SCP targets are
  restricted to the strict safe grammar; no whitespace, quotes, ``$``,
  backticks, colons (in path components), traversal, or control characters.
* SCP writes to a temporary name on the remote under the new temp dir,
  preserving relative directories and writing each file with mode ``0600``.
* Independently read back, hashed, and metadata-validated against the
  production ``CheckpointStore`` contract before the temp dir is atomically
  renamed into the final location.
* On any failure, the prior valid destination and the local relay are
  preserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from osm_polygon_sentence_relevance.labeling.checkpoint import (
    CheckpointError,
    CheckpointStore,
)
from osm_polygon_sentence_relevance.labeling.contracts import RunIdentity
from osm_polygon_sentence_relevance.operator.relay_transport import (
    FILE_MODE,
    RELAY_DIR_MODE,
    RelayError,
    RemoteEntry,
    RemoteTransfer,
    list_remote_checkpoints,
    list_remote_dir,
    validate_safe_remote_path,
)

#: Filenames that survive a graceful deadline interruption, at the
#: CheckpointStore root (i.e. directly under ``label_work``).
ALLOWED_TOP_FILES: frozenset[str] = frozenset({"progress.json", "timing.json"})
#: Strict paired batch file naming used by :class:`CheckpointStore`.
_BATCH_ENTRY = re.compile(r"batch-(\d{6})\.(parquet|json)$")


def _validate_destination_root(destination_root: Path) -> Path:
    """Refuse non-Seagate destinations.

    The relay contract binds to the production ``DATA_ROOT`` exported from
    :mod:`osm_polygon_sentence_relevance.operator.config`. Tests may pass
    a Seagate-named temp dir (``/private/var/...`` on macOS or anything that
    contains the substring ``Seagate``), but never ``/tmp`` or ``/Users``.
    """

    text = str(destination_root)
    if text.startswith("/tmp") or "/Library/" in text:
        raise RelayError(f"refusing relay destination on Mac internal storage: {text}")
    if text.startswith("/Users/"):
        raise RelayError(f"refusing relay destination on Mac internal storage: {text}")
    if destination_root.is_symlink():
        raise RelayError("destination root is a symlink")
    if not destination_root.is_dir():
        raise RelayError("destination root must be a real directory")
    return destination_root


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RelayInventory:
    """Validated, byte-hashed inventory of the Seagate relay copy."""

    root: Path
    progress: Path
    timing: Path | None
    parquet_paths: tuple[Path, ...]
    metadata_paths: dict[str, Path]
    completed: int
    total: int

    @property
    def ordered_indexes(self) -> tuple[int, ...]:
        result: list[int] = []
        for path in self.parquet_paths:
            match = _BATCH_ENTRY.fullmatch(path.name)
            if match is None:
                raise RelayError(f"unexpected checkpoint name: {path.name}")
            result.append(int(match.group(1)))
        return tuple(sorted(result))


def _list_local_dir(directory: Path) -> list[RemoteEntry]:
    entries: list[RemoteEntry] = []
    for child in directory.iterdir():
        if child.is_symlink():
            entries.append(RemoteEntry(child.name, "symlink"))
        elif child.is_file():
            entries.append(RemoteEntry(child.name, "file"))
        elif child.is_dir():
            entries.append(RemoteEntry(child.name, "dir"))
        else:
            entries.append(RemoteEntry(child.name, "other"))
    return entries


def _validate_progress_payload(
    payload: object, expected_identity: Mapping[str, object]
) -> tuple[int, int]:
    if not isinstance(payload, Mapping):
        raise RelayError("progress.json payload is not a mapping")
    if payload.get("identity") != expected_identity:
        raise RelayError("progress.json identity mismatch")
    completed = _coerce_progress_int(payload.get("completed", 0), "completed")
    if "total" in payload:
        total = _coerce_progress_int(payload.get("total"), "total")
    elif "remaining" in payload:
        remaining = _coerce_progress_int(payload.get("remaining"), "remaining")
        total = completed + remaining
    else:
        raise RelayError("progress.json missing total/remaining")
    if completed < 0 or total <= 0 or completed > total:
        raise RelayError("progress.json counters are inconsistent")
    return completed, total


def _coerce_progress_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise RelayError(f"progress.json {label} is invalid")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise RelayError(f"progress.json {label} is invalid")


def _string_object_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RelayError(f"{label} is not a mapping")
    result: dict[str, object] = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise RelayError(f"{label} contains a non-string key")
        result[key] = child
    return result


def _batch_index(path: Path) -> int:
    match = _BATCH_ENTRY.fullmatch(path.name)
    if match is None:
        raise RelayError(f"unexpected checkpoint name: {path.name}")
    return int(match.group(1))


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise RelayError(f"checkpoint identity field {key} is invalid")
    return value


def _required_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RelayError(f"checkpoint identity field {key} is invalid")
    return value


def _run_identity_from_mapping(mapping: Mapping[str, object]) -> RunIdentity:
    return RunIdentity(
        input_sha256=_required_string(mapping, "input_sha256"),
        input_dataset_revision=_required_string(mapping, "input_dataset_revision"),
        model_repo_id=_required_string(mapping, "model_repo_id"),
        model_revision=_required_string(mapping, "model_revision"),
        model_file=_required_string(mapping, "model_file"),
        model_file_sha256=_required_string(mapping, "model_file_sha256"),
        prompt_version=_required_string(mapping, "prompt_version"),
        source_commit=_required_string(mapping, "source_commit"),
        engine=_required_string(mapping, "engine"),
        engine_version=_required_string(mapping, "engine_version"),
        batch_size=_required_int(mapping, "batch_size"),
        row_limit=_required_int(mapping, "row_limit"),
        llama_parallel=_required_int(mapping, "llama_parallel"),
        llama_per_slot_context=_required_int(mapping, "llama_per_slot_context"),
        llama_total_context=_required_int(mapping, "llama_total_context"),
        request_concurrency=_required_int(mapping, "request_concurrency"),
    )


def _next_generation(run_root: Path) -> Path:
    """Return a fresh sibling generation directory."""

    existing = sorted(
        p
        for p in run_root.iterdir()
        if p.is_dir() and p.name.startswith("relay.generation-")
    )
    next_idx = 0
    for p in existing:
        suffix = p.name[len("relay.generation-") :]
        try:
            n = int(suffix)
            if n >= next_idx:
                next_idx = n + 1
        except ValueError:
            continue
    generation = run_root / f"relay.generation-{next_idx:04d}"
    generation.mkdir(mode=RELAY_DIR_MODE)
    return generation


def _stage_relay_local(
    *,
    source_dir: Path,
    destination_root: Path,
    run_id: str,
    expected_run_identity: Mapping[str, object] | None,
) -> RelayInventory:
    """Validate a local copy of the relay against the production contract.

    Refuses symlinks, unexpected entries, incomplete pairs, malformed JSON,
    identity inconsistency across the set, Parquet byte-hash mismatch,
    duplicate sentence IDs, and inconsistent progress. Returns the validated
    inventory under a fresh sibling generation; the prior ``current`` relay
    is never deleted before the new one is fully validated.
    """

    _validate_destination_root(destination_root)
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise RelayError("relay source must be a real directory")
    entries = _list_local_dir(source_dir)
    if not entries:
        raise RelayError("relay source is empty")

    run_root = destination_root / run_id
    if not run_root.exists():
        run_root.mkdir(mode=RELAY_DIR_MODE)
    elif run_root.is_symlink() or not run_root.is_dir():
        raise RelayError("relay run root is not a real directory")

    generation = _next_generation(run_root)
    (generation / "checkpoints").mkdir(mode=RELAY_DIR_MODE, exist_ok=True)
    progress_path: Path | None = None
    timing_path: Path | None = None
    parquet_paths: list[Path] = []
    metadata_paths: dict[str, Path] = {}

    try:
        for entry in entries:
            if entry.kind == "symlink":
                raise RelayError(f"refusing relay symlink: {entry.name}")
            if entry.kind == "dir":
                if entry.name != "checkpoints":
                    raise RelayError(f"unexpected relay subdirectory: {entry.name}")
                continue
            if entry.kind != "file":
                raise RelayError(f"refusing relay non-file entry: {entry.name}")
            source = source_dir / entry.name
            if entry.name in ALLOWED_TOP_FILES:
                target = generation / entry.name
                shutil.copyfile(source, target)
                os.chmod(target, FILE_MODE)
                if entry.name == "progress.json":
                    progress_path = target
                else:
                    timing_path = target
            else:
                match = _BATCH_ENTRY.fullmatch(entry.name)
                if match is None:
                    raise RelayError(f"unexpected relay entry: {entry.name}")
                target = generation / "checkpoints" / entry.name
                shutil.copyfile(source, target)
                os.chmod(target, FILE_MODE)
                if entry.name.endswith(".parquet"):
                    parquet_paths.append(target)
                else:
                    metadata_paths[entry.name.rsplit(".", 1)[0]] = target

        # Pull in any paired batch entries that live under the staging
        # ``checkpoints/`` subdirectory.
        staged_ckpts = source_dir / "checkpoints"
        if staged_ckpts.is_dir():
            for entry in _list_local_dir(staged_ckpts):
                if entry.kind == "symlink":
                    raise RelayError(
                        f"refusing relay symlink: checkpoints/{entry.name}"
                    )
                if entry.kind != "file":
                    raise RelayError(
                        f"refusing relay non-file entry: checkpoints/{entry.name}"
                    )
                match = _BATCH_ENTRY.fullmatch(entry.name)
                if match is None:
                    raise RelayError(
                        f"unexpected relay entry: checkpoints/{entry.name}"
                    )
                source = staged_ckpts / entry.name
                target = generation / "checkpoints" / entry.name
                if target.exists():
                    continue  # already staged at top level
                shutil.copyfile(source, target)
                os.chmod(target, FILE_MODE)
                if entry.name.endswith(".parquet"):
                    parquet_paths.append(target)
                else:
                    metadata_paths[entry.name.rsplit(".", 1)[0]] = target

        if progress_path is None:
            raise RelayError("relay is missing progress.json")
        if not parquet_paths:
            raise RelayError("relay is missing any checkpoint batches")

        canonical_identity: dict[str, object] | None = None
        for path in parquet_paths:
            stem = path.name.rsplit(".", 1)[0]
            meta = metadata_paths.get(stem)
            if meta is None:
                raise RelayError(f"checkpoint pair is missing metadata: {path.name}")
            try:
                payload = json.loads(meta.read_text())
            except json.JSONDecodeError as exc:
                raise RelayError("checkpoint metadata is malformed") from exc
            if not isinstance(payload, Mapping):
                raise RelayError("checkpoint metadata is not a mapping")
            raw_identity = payload.get("identity")
            if not isinstance(raw_identity, Mapping):
                raise RelayError(f"checkpoint identity missing: {path.name}")
            identity = _string_object_mapping(
                raw_identity, f"checkpoint identity: {path.name}"
            )
            expected_sha = payload.get("parquet_sha256")
            if not isinstance(expected_sha, str):
                raise RelayError(f"checkpoint parquet_sha256 missing: {path.name}")
            actual_sha = _sha256_file(path)
            if actual_sha.lower() != expected_sha.strip().lower():
                raise RelayError(f"checkpoint Parquet SHA-256 mismatch: {path.name}")
            if canonical_identity is None:
                canonical_identity = identity
            elif identity != canonical_identity:
                raise RelayError(
                    f"checkpoint identity differs across batches: {path.name}"
                )

        if canonical_identity is None:
            raise RelayError("relay is missing checkpoint identity")
        progress_payload = _string_object_mapping(
            json.loads(progress_path.read_text()), "progress.json payload"
        )
        raw_progress_identity = progress_payload.get("identity")
        if not isinstance(raw_progress_identity, Mapping):
            raise RelayError("progress.json identity is missing")
        progress_identity = _string_object_mapping(
            raw_progress_identity, "progress.json identity"
        )
        if progress_identity != canonical_identity:
            raise RelayError("progress.json identity does not match checkpoints")
        completed, total = _validate_progress_payload(
            progress_payload, canonical_identity
        )

        if expected_run_identity is not None:
            _enforce_overlapping_identity(canonical_identity, expected_run_identity)

        try:
            CheckpointStore(
                generation,
                identity=_run_identity_from_mapping(canonical_identity),
            ).load_all()
        except (CheckpointError, TypeError) as exc:
            raise RelayError(f"CheckpointStore validation failed: {exc}") from exc

        # Atomic publish: rename the prior ``current`` symlink (if any) to
        # ``prev``, then create a new ``current`` symlink to this generation.
        current = run_root / "current"
        prev = run_root / "prev"
        if current.is_symlink() or current.exists():
            try:
                if prev.exists() or prev.is_symlink():
                    prev.unlink()
                current.replace(prev)
            except OSError:
                pass
        try:
            os.symlink(generation.name, current)
        except FileExistsError:
            current.unlink(missing_ok=True)
            os.symlink(generation.name, current)

        # fsync the run_root directory so the publication is durable.
        fd = os.open(run_root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

        published = run_root / "current"
        finalized = run_root / os.readlink(published)
        return RelayInventory(
            root=finalized,
            progress=finalized / "progress.json",
            timing=finalized / "timing.json" if timing_path else None,
            parquet_paths=tuple(
                sorted(
                    finalized.glob("checkpoints/batch-*.parquet"),
                    key=_batch_index,
                )
            ),
            metadata_paths={
                path.name.rsplit(".", 1)[0]: path
                for path in finalized.glob("checkpoints/batch-*.json")
            },
            completed=completed,
            total=total,
        )
    except BaseException:
        shutil.rmtree(generation, ignore_errors=True)
        raise


_OVERLAPPING_IDENTITY_FIELDS: tuple[str, ...] = (
    "input_dataset_revision",
    "source_commit",
    "model_file_sha256",
    "model_repo_id",
    "model_revision",
    "prompt_version",
    "batch_size",
    "row_limit",
    "llama_parallel",
    "llama_per_slot_context",
    "llama_total_context",
    "request_concurrency",
)


def _enforce_overlapping_identity(
    canonical: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    for field_name in _OVERLAPPING_IDENTITY_FIELDS:
        if canonical.get(field_name) != expected.get(field_name):
            raise RelayError(
                f"checkpoint identity does not match run identity: {field_name}"
            )


def retrieve_to_seagate(
    *,
    source: RemoteTransfer,
    source_checkpoint_root: str,
    destination_root: Path,
    run_id: str,
    expected_run_identity: Mapping[str, object] | None = None,
) -> RelayInventory:
    """Fetch the checkpoint set from the previous site into the Seagate relay.

    ``source_checkpoint_root`` is the **label-work root** (where
    ``CheckpointStore`` writes ``progress.json``). The relay fetches
    ``progress.json``, optional ``timing.json``, and the flat
    ``checkpoints/batch-NN.{parquet,json}`` entries. It validates byte
    hashes against metadata and re-validates the whole set via the
    production ``CheckpointStore``. Refuses symlinks, unexpected entries,
    incomplete pairs, and Parquet byte-hash mismatches.
    """

    validate_safe_remote_path(source_checkpoint_root)
    _validate_destination_root(destination_root)

    top_entries = list_remote_dir(source.ssh_target, source_checkpoint_root)
    ckpt_entries = list_remote_checkpoints(source.ssh_target, source_checkpoint_root)
    if not any(e.name == "progress.json" for e in top_entries):
        raise RelayError("remote label-work root is missing progress.json")
    staging = Path(tempfile.mkdtemp(prefix=".relay-fetch.", dir=destination_root))
    staging.mkdir(mode=RELAY_DIR_MODE, exist_ok=True)
    try:
        (staging / "checkpoints").mkdir(mode=RELAY_DIR_MODE, exist_ok=True)
        for entry in top_entries:
            if entry.kind == "symlink":
                raise RelayError(f"refusing remote symlink: {entry.name}")
            if entry.kind == "dir":
                # Only checkpoints/ is allowed at depth 1.
                if entry.name != "checkpoints":
                    raise RelayError(f"unexpected remote subdirectory: {entry.name}")
                continue
            if entry.kind != "file":
                raise RelayError(f"refusing remote non-file entry: {entry.name}")
            if entry.name not in ALLOWED_TOP_FILES:
                raise RelayError(f"unexpected remote top-level entry: {entry.name}")
            remote_path = f"{source_checkpoint_root.rstrip('/')}/{entry.name}"
            local_path = staging / entry.name
            source.fetch(remote_path, local_path)
            os.chmod(local_path, FILE_MODE)
        for entry in ckpt_entries:
            if entry.kind == "symlink":
                raise RelayError(f"refusing remote symlink: {entry.name}")
            if entry.kind != "file":
                raise RelayError(
                    f"refusing remote non-file checkpoint entry: {entry.name}"
                )
            match = _BATCH_ENTRY.fullmatch(entry.name)
            if match is None:
                raise RelayError(f"unexpected remote checkpoint entry: {entry.name}")
            remote_path = (
                f"{source_checkpoint_root.rstrip('/')}/checkpoints/{entry.name}"
            )
            local_path = staging / "checkpoints" / entry.name
            source.fetch(remote_path, local_path)
            os.chmod(local_path, FILE_MODE)
        return _stage_relay_local(
            source_dir=staging,
            destination_root=destination_root,
            run_id=run_id,
            expected_run_identity=expected_run_identity,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def stage_to_destination(
    *,
    inventory: RelayInventory,
    destination: RemoteTransfer,
    destination_checkpoint_root: str,
) -> None:
    """Push the validated relay to a destination site and re-validate.

    The destination ``scp`` writes to a temporary work directory with mode
    ``0700`` (created via ``install -d -m 0700``). Each file is pushed with
    mode ``0600``. An independent read-back re-hashes and re-validates the
    remote copy. Only when every byte matches and CheckpointStore validation
    succeeds on the readback does the relay ``mv`` the temp dir into place
    atomically. Any failure preserves the prior destination and the local
    relay.
    """

    validate_safe_remote_path(destination_checkpoint_root)
    if ".." in Path(destination_checkpoint_root).parts:
        raise RelayError("refusing to traverse destination checkpoint root")

    # 1. Verify the prior destination directory is absent.
    list_proc = subprocess.run(
        [
            "ssh",
            destination.ssh_target,
            "find",
            destination_checkpoint_root,
            "-mindepth",
            "1",
            "-maxdepth",
            "1",
        ],
        check=False,
        shell=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if list_proc.returncode == 0 and list_proc.stdout.strip():
        raise RelayError(
            f"refusing to overwrite non-empty destination: {destination_checkpoint_root}"
        )

    # 2. Create a fresh temporary work dir on the remote, sibling to the
    #    intended final location. The temp dir will be atomically renamed
    #    into the destination once every byte has been independently
    #    re-validated. The production layout placed files directly at the
    #    checkpoint root (no per-inventory generation directory), so the
    #    atomic target IS the destination_checkpoint_root.
    final_remote = destination_checkpoint_root.rstrip("/")
    temp_remote = f"{final_remote}.staging-{os.getpid()}"
    destination.ssh_mkdir_0700(temp_remote)

    # 3. Push each file (preserving relative path) into the temp dir.
    local_paths: list[Path] = [inventory.progress]
    if inventory.timing is not None:
        local_paths.append(inventory.timing)
    local_paths.extend(inventory.parquet_paths)
    local_paths.extend(inventory.metadata_paths.values())

    for local_path in local_paths:
        relative = local_path.relative_to(inventory.root)
        remote_path = f"{temp_remote.rstrip('/')}/{relative}"
        # ensure remote parent exists
        remote_dir = "/".join(remote_path.split("/")[:-1])
        destination.ssh_mkdir_0700(remote_dir)
        destination.push(local_path, remote_path)

    # 4. Read back each file into a local verification directory and re-hash.
    verification = Path(
        tempfile.mkdtemp(prefix=".relay-readback.", dir=inventory.root.parent)
    )
    verification.mkdir(mode=RELAY_DIR_MODE, exist_ok=True)
    try:
        (verification / "checkpoints").mkdir(mode=RELAY_DIR_MODE, exist_ok=True)
        for local_path in local_paths:
            relative = local_path.relative_to(inventory.root)
            remote_path = f"{temp_remote.rstrip('/')}/{relative}"
            fetched = verification / relative
            fetched.parent.mkdir(parents=True, exist_ok=True, mode=RELAY_DIR_MODE)
            destination.fetch(remote_path, fetched)
            os.chmod(fetched, FILE_MODE)
            if _sha256_file(fetched) != _sha256_file(local_path):
                raise RelayError(f"destination readback hash mismatch: {relative}")

        # 5. Re-validate via CheckpointStore against the readback copy.
        progress_payload = _string_object_mapping(
            json.loads(inventory.progress.read_text()), "progress.json payload"
        )
        identity_dict = _string_object_mapping(
            progress_payload.get("identity"), "progress.json identity"
        )
        try:
            CheckpointStore(
                verification,
                identity=_run_identity_from_mapping(identity_dict),
            ).load_all()
        except (CheckpointError, TypeError) as exc:
            raise RelayError(
                f"CheckpointStore validation failed on readback: {exc}"
            ) from exc

        # 6. Atomic rename of the temp dir into the final location.
        destination.ssh_atomic_rename(temp_remote, final_remote)
    except BaseException:
        # Best-effort cleanup of remote temp dir; preserve prior destination.
        subprocess.run(
            [
                "ssh",
                destination.ssh_target,
                f"rm -rf -- {temp_remote}",
            ],
            check=False,
            shell=False,
            timeout=60,
        )
        shutil.rmtree(verification, ignore_errors=True)
        raise
    else:
        shutil.rmtree(verification, ignore_errors=True)


__all__ = [
    "ALLOWED_TOP_FILES",
    "FILE_MODE",
    "RELAY_DIR_MODE",
    "RelayError",
    "RelayInventory",
    "RemoteTransfer",
    "retrieve_to_seagate",
    "stage_to_destination",
]
