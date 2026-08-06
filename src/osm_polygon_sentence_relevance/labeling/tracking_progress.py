"""Durable, non-blocking batch progress logging for Trackio.

Local checkpoints remain authoritative. Each completed batch first becomes a
small JSON outbox marker. A daemon worker replays those markers into one
resumable Trackio run, so network latency and temporary Hub failures do not
hold the inference loop or lose progress.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from .releases import ReleaseLane, trackio_space_id


class BatchTrackingError(RuntimeError):
    """Raised for invalid local batch-tracking configuration."""


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


class TrackioBatchLogger:
    """Queue one immutable Trackio step per completed local checkpoint."""

    def __init__(
        self,
        *,
        work_dir: Path,
        project: str,
        run_name: str,
        lane: ReleaseLane,
        space_id: str | None = None,
        trackio_module: Any | None = None,
    ) -> None:
        if not project.strip() or not run_name.strip():
            raise ValueError("Trackio project and run name must be non-blank")
        self.work_dir = Path(work_dir)
        self.project = project.strip()
        self.run_name = run_name.strip()
        self.lane = lane
        self.space_id = space_id or trackio_space_id(lane)
        self._trackio_module = trackio_module
        self.root = self.work_dir / ".trackio"
        self.pending = self.root / "pending"
        self.uploaded = self.root / "uploaded"
        for path in (self.root, self.pending, self.uploaded):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(path, 0o700)
        self._queue: Queue[Path] = Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._started = False
        self._finished = False

    def start(self) -> None:
        """Start the worker and enqueue markers left by an earlier allocation."""

        if self._closed:
            raise BatchTrackingError("Trackio batch logger is closed")
        if self._thread is not None:
            return
        for marker in sorted(self.pending.glob("batch-*.json")):
            if (self.uploaded / marker.name).exists():
                marker.unlink(missing_ok=True)
            else:
                self._queue.put_nowait(marker)
        self._thread = threading.Thread(
            target=self._worker,
            name="trackio-batch-logger",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, metrics: Mapping[str, object]) -> None:
        """Persist and queue a sanitized metric snapshot without network I/O."""

        step = metrics.get("batch_index")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Trackio batch_index must be a non-negative integer")
        marker = self.pending / f"batch-{step:06d}.json"
        if marker.exists() or (self.uploaded / marker.name).exists():
            return
        payload = dict(metrics)
        payload["batch_index"] = step
        payload["release_lane"] = self.lane.value
        _atomic_json(marker, payload)
        self._queue.put_nowait(marker)

    def close(self, *, wait: bool = True, timeout: float = 30.0) -> None:
        """Drain the bounded queue and finish only after all steps are logged."""

        if timeout < 0:
            raise ValueError("Trackio drain timeout must be non-negative")
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + timeout
        if wait:
            while self._queue.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(
                timeout=max(0.0, deadline - time.monotonic()) if wait else 1.0
            )
        pending_markers = any(self.pending.glob("batch-*.json"))
        if (
            self._started
            and not self._finished
            and self._queue.unfinished_tasks == 0
            and not pending_markers
        ):
            try:
                self._trackio().finish()
            except Exception as exc:
                raise BatchTrackingError("Trackio run could not finish") from exc
            self._finished = True

    def _trackio(self) -> Any:
        if self._trackio_module is None:
            try:
                self._trackio_module = importlib.import_module("trackio")
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise BatchTrackingError(
                    "install the tracking extra to log batch progress"
                ) from exc
        return self._trackio_module

    def _ensure_started(self) -> Any:
        trackio = self._trackio()
        if not self._started:
            trackio.init(
                project=self.project,
                name=self.run_name,
                space_id=self.space_id,
                resume="allow",
                config={"release_lane": self.lane.value},
                embed=False,
                auto_log_gpu=False,
            )
            self._started = True
        return trackio

    def _worker(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                marker = self._queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise BatchTrackingError("Trackio marker is not an object")
                self._ensure_started().log(payload, step=int(payload["batch_index"]))
                _atomic_json(self.uploaded / marker.name, payload)
                marker.unlink(missing_ok=True)
            except Exception as exc:
                status = self.root / "status.json"
                _atomic_json(
                    status,
                    {
                        "schema_version": 1,
                        "last_error": type(exc).__name__,
                        "pending_batches": self._queue.unfinished_tasks,
                    },
                )
            finally:
                self._queue.task_done()


__all__ = ["BatchTrackingError", "TrackioBatchLogger"]
