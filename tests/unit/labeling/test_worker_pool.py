from __future__ import annotations

import pytest

from osm_polygon_sentence_relevance.labeling import worker_pool
from osm_polygon_sentence_relevance.labeling.worker_pool import _ReusableWorkerPool


def test_reusable_worker_pool_lazily_reuses_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []

    class _Executor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers
            self.shutdown_calls: list[bool] = []
            created.append(self)

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            self.shutdown_calls.append(wait and cancel_futures)

    monkeypatch.setattr(worker_pool, "ThreadPoolExecutor", _Executor)
    pool = _ReusableWorkerPool(max_workers=2, error_type=RuntimeError)

    first = pool.get()
    second = pool.get()

    assert first is second
    assert len(created) == 1
    assert created[0].max_workers == 2

    pool.close()
    pool.close()
    assert created[0].shutdown_calls == [True]
    with pytest.raises(RuntimeError, match="closed"):
        pool.get()
