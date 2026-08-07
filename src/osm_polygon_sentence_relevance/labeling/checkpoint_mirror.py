"""Non-blocking, durable Hugging Face mirroring for label checkpoints.

The local :class:`~.checkpoint.CheckpointStore` remains authoritative.  A
mirror writes a small outbox marker after each successful local batch and a
single daemon worker uploads that batch to a run-specific path on the dataset's
main revision. A failed upload leaves its marker in place for the next resumed
allocation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Protocol

from .releases import checkpoint_prefix


class CheckpointStoreLike(Protocol):
    """Common durable-store surface shared by V1 and V2 checkpoints."""

    root: Path
    directory: Path
    identity: Any


type UploadFiles = tuple[tuple[str, Path], ...]
type UploadCallable = Callable[[str, str, UploadFiles], str]

_BATCH_NAME = re.compile(r"batch-(\d{6})")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_DATASET_ID = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_MARKER_SCHEMA = 1


class CheckpointMirrorError(RuntimeError):
    """Raised for invalid mirror configuration or local marker data."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _validate_dataset_id(dataset_id: str) -> None:
    if not isinstance(dataset_id, str) or not _DATASET_ID.fullmatch(dataset_id):
        raise ValueError("dataset ID must be an owner/name value")


def _validate_branch(branch: str) -> None:
    """Validate the legacy argument carrying a main-tree namespace."""
    if not isinstance(branch, str) or not re.fullmatch(
        r"checkpoints/[0-9a-f]{20}", branch
    ):
        raise ValueError("checkpoint namespace must be checkpoints/<20 lowercase hex>")


def _batch_paths(store: CheckpointStoreLike, index: int) -> tuple[Path, Path]:
    stem = f"batch-{index:06d}"
    return store.directory / f"{stem}.parquet", store.directory / f"{stem}.json"


class CheckpointMirror:
    """Queue local batches for one-at-a-time remote staging uploads."""

    def __init__(
        self,
        *,
        store: CheckpointStoreLike,
        dataset_id: str,
        branch: str,
        uploader: UploadCallable | None = None,
    ) -> None:
        _validate_dataset_id(dataset_id)
        _validate_branch(branch)
        self.store = store
        self.dataset_id = dataset_id
        self.branch = branch
        self._uploader = uploader
        self.root = store.root / ".checkpoint-mirror"
        self.pending = self.root / "pending"
        self.uploaded = self.root / "uploaded"
        for directory in (self.root, self.pending, self.uploaded):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(directory, 0o700)
        self._queue: Queue[Path] = Queue()
        self._stop = threading.Event()
        self._status_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False

    def start(self) -> None:
        """Start the worker and enqueue durable markers from an earlier run."""

        if self._closed:
            raise CheckpointMirrorError("checkpoint mirror is closed")
        if self._thread is not None:
            return
        for marker in sorted(self.pending.glob("batch-*.json")):
            uploaded = self.uploaded / marker.name
            if uploaded.exists():
                marker.unlink(missing_ok=True)
            else:
                self._queue.put_nowait(marker)
        self._thread = threading.Thread(
            target=self._worker,
            name="checkpoint-mirror",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, index: int) -> None:
        """Persist and queue one completed batch without waiting for network I/O."""

        if isinstance(index, bool) or index < 0:
            raise ValueError("checkpoint batch index must be a non-negative integer")
        marker = self.pending / f"batch-{index:06d}.json"
        if marker.exists() or (self.uploaded / marker.name).exists():
            return
        try:
            payload = self._marker_payload(index)
            _atomic_json(marker, payload)
        except (CheckpointMirrorError, OSError, ValueError):
            self._record_error("checkpoint-drift", index)
            return
        self._queue.put_nowait(marker)

    def close(self, *, wait: bool = True, timeout: float = 30.0) -> None:
        """Stop the worker, draining queued uploads only up to ``timeout`` seconds."""

        if timeout < 0:
            raise ValueError("checkpoint mirror timeout must be non-negative")
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + timeout
        if wait:
            while self._queue.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            remaining = max(0.0, deadline - time.monotonic()) if wait else 1.0
            self._thread.join(timeout=remaining)

    def _marker_payload(self, index: int) -> dict[str, Any]:
        parquet, metadata = _batch_paths(self.store, index)
        for path in (parquet, metadata):
            if path.is_symlink() or not path.is_file():
                raise CheckpointMirrorError("checkpoint file is not a regular file")
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointMirrorError("checkpoint metadata is invalid") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("identity") != self.store.identity.checkpoint_dict()
        ):
            raise CheckpointMirrorError("checkpoint identity mismatch")
        parquet_sha = payload.get("parquet_sha256")
        row_count = payload.get("row_count")
        if not isinstance(parquet_sha, str) or not _HEX64.fullmatch(parquet_sha):
            raise CheckpointMirrorError("checkpoint hash is invalid")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 1
        ):
            raise CheckpointMirrorError("checkpoint row count is invalid")
        run_id = self.branch.removeprefix("checkpoints/")
        files = (
            {
                "local_name": parquet.name,
                "remote_path": f"{checkpoint_prefix(run_id)}/{parquet.name}",
                "sha256": parquet_sha,
                "bytes": parquet.stat().st_size,
            },
            {
                "local_name": metadata.name,
                "remote_path": f"{checkpoint_prefix(run_id)}/{metadata.name}",
                "sha256": _sha256(metadata),
                "bytes": metadata.stat().st_size,
            },
        )
        return {
            "schema_version": _MARKER_SCHEMA,
            "batch_index": index,
            "dataset_id": self.dataset_id,
            "branch": self.branch,
            "identity": self.store.identity.checkpoint_dict(),
            "files": files,
        }

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                marker = self._queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                self._upload_marker(marker)
            except CheckpointMirrorError:
                self._record_error("checkpoint-drift", self._index_from_marker(marker))
            except Exception:
                self._record_error("upload-failed", self._index_from_marker(marker))
            finally:
                self._queue.task_done()

    def _index_from_marker(self, marker: Path) -> int | None:
        match = _BATCH_NAME.fullmatch(marker.stem)
        return int(match.group(1)) if match else None

    def _upload_marker(self, marker: Path) -> None:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointMirrorError("mirror marker is invalid") from exc
        if not isinstance(payload, dict):
            raise CheckpointMirrorError("mirror marker is invalid")
        if (
            payload.get("schema_version") != _MARKER_SCHEMA
            or payload.get("dataset_id") != self.dataset_id
            or payload.get("branch") != self.branch
            or payload.get("identity") != self.store.identity.checkpoint_dict()
        ):
            raise CheckpointMirrorError("mirror marker identity mismatch")
        batch_index = payload.get("batch_index")
        if (
            isinstance(batch_index, bool)
            or not isinstance(batch_index, int)
            or batch_index < 0
        ):
            raise CheckpointMirrorError("mirror marker batch index is invalid")
        files = payload.get("files")
        if not isinstance(files, list) or len(files) != 2:
            raise CheckpointMirrorError("mirror marker files are invalid")
        upload_files: list[tuple[str, Path]] = []
        for item in files:
            if not isinstance(item, dict):
                raise CheckpointMirrorError("mirror marker file is invalid")
            local_name = item.get("local_name")
            remote_path = item.get("remote_path")
            expected_sha = item.get("sha256")
            expected_bytes = item.get("bytes")
            if (
                not isinstance(local_name, str)
                or Path(local_name).name != local_name
                or re.fullmatch(r"batch-\d{6}\.(?:parquet|json)", local_name) is None
                or not isinstance(remote_path, str)
                or remote_path
                != f"{checkpoint_prefix(self.branch.removeprefix('checkpoints/'))}/{local_name}"
                or not isinstance(expected_sha, str)
                or not _HEX64.fullmatch(expected_sha)
                or isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
            ):
                raise CheckpointMirrorError("mirror marker file is invalid")
            path = self.store.directory / local_name
            if path.is_symlink() or not path.is_file():
                raise CheckpointMirrorError("checkpoint file is not a regular file")
            if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha:
                raise CheckpointMirrorError("checkpoint file changed after enqueue")
            upload_files.append((remote_path, path))
        expected_stem = f"batch-{batch_index:06d}"
        if {path.name for _, path in upload_files} != {
            f"{expected_stem}.parquet",
            f"{expected_stem}.json",
        }:
            raise CheckpointMirrorError("mirror marker batch files are inconsistent")
        uploader = self._uploader or _default_uploader
        commit_id = uploader(self.dataset_id, self.branch, tuple(upload_files))
        if not isinstance(commit_id, str) or not commit_id:
            raise CheckpointMirrorError("uploader returned no commit ID")
        payload["commit_id"] = commit_id
        _atomic_json(self.uploaded / marker.name, payload)
        marker.unlink(missing_ok=True)
        self._record_success(self._index_from_marker(marker))

    def _record_success(self, index: int | None) -> None:
        self._write_status(last_error=None, last_successful_batch=index)

    def _record_error(self, token: str, index: int | None) -> None:
        self._write_status(last_error=token, last_failed_batch=index)

    def _write_status(self, **values: object) -> None:
        payload: dict[str, Any] = {"schema_version": _MARKER_SCHEMA}
        with self._status_lock:
            status_path = self.root / "status.json"
            try:
                if status_path.exists():
                    existing = json.loads(status_path.read_text(encoding="utf-8"))
                    if isinstance(existing, dict):
                        payload.update(existing)
                payload.update(values)
                _atomic_json(status_path, payload)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                # The checkpoint and its outbox marker remain authoritative if
                # a best-effort status write cannot be completed.
                return


def _default_uploader(dataset_id: str, branch: str, files: UploadFiles) -> str:
    """Upload two deterministic files to the run path on Hub ``main``."""

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise CheckpointMirrorError(
            "install the hub extra to mirror checkpoints"
        ) from exc
    api = HfApi()
    operations = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
        for remote, local in files
    ]
    info = api.create_commit(
        repo_id=dataset_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Mirror labeling checkpoint {Path(files[0][1]).stem}",
        revision="main",
    )
    commit_id = getattr(info, "oid", None)
    if not isinstance(commit_id, str) or not commit_id:
        raise CheckpointMirrorError("Hugging Face returned no checkpoint commit ID")
    return commit_id


__all__ = [
    "CheckpointMirror",
    "CheckpointMirrorError",
    "CheckpointStoreLike",
    "UploadCallable",
    "UploadFiles",
]
