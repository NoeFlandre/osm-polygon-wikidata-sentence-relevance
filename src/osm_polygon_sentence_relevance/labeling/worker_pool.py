"""Reusable worker-pool lifecycle shared by inference engines."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


class _ReusableWorkerPool:
    """Lazily create, reuse, and safely close an inference worker pool."""

    def __init__(
        self,
        *,
        max_workers: int,
        error_type: type[RuntimeError],
        executor_factory: Callable[..., ThreadPoolExecutor] | None = None,
    ) -> None:
        self._max_workers = max_workers
        self._error_type = error_type
        self._executor_factory = executor_factory or ThreadPoolExecutor
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False

    def get(self) -> ThreadPoolExecutor:
        if self._closed:
            raise self._error_type("inference engine is closed")
        if self._executor is None:
            self._executor = self._executor_factory(max_workers=self._max_workers)
        return self._executor

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
