from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from osm_polygon_sentence_relevance.labeling.releases import ReleaseLane
from osm_polygon_sentence_relevance.labeling.tracking_progress import (
    BatchTrackingError,
    TrackioBatchLogger,
)


class _Trackio:
    def __init__(self) -> None:
        self.init_calls: list[dict[str, object]] = []
        self.log_calls: list[tuple[dict[str, object], int]] = []
        self.finish_calls = 0

    def init(self, **kwargs: object) -> None:
        self.init_calls.append(kwargs)

    def log(self, metrics: dict[str, object], *, step: int) -> None:
        self.log_calls.append((metrics, step))

    def finish(self) -> None:
        self.finish_calls += 1


class _FailingTrackio(_Trackio):
    def log(self, metrics: dict[str, object], *, step: int) -> None:
        raise OSError("temporary Hub failure")


class _FinishFailingTrackio(_Trackio):
    def finish(self) -> None:
        raise OSError("finish failed")


def _wait(logger: TrackioBatchLogger, trackio: _Trackio) -> None:
    deadline = time.monotonic() + 2
    while len(trackio.log_calls) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    logger.close(timeout=2)


def test_batch_logger_is_async_durable_and_uses_worldwide_space(tmp_path: Path) -> None:
    fake = _Trackio()
    logger = TrackioBatchLogger(
        work_dir=tmp_path,
        project="worldwide-stratified-labeling",
        run_name="run-abc",
        lane=ReleaseLane.V2_WORLDWIDE,
        trackio_module=fake,
    )
    logger.start()
    logger.enqueue({"batch_index": 3, "completed_rows": 512, "eta_seconds": 10.0})

    _wait(logger, fake)

    assert (
        fake.init_calls[0]["space_id"]
        == "NoeFlandre/worldwide-stratified-labeling-trackio"
    )
    assert fake.init_calls[0]["resume"] == "allow"
    assert fake.log_calls[0][1] == 3
    assert fake.log_calls[0][0]["release_lane"] == "v2-worldwide"
    assert fake.finish_calls == 1
    assert not list((tmp_path / ".trackio" / "pending").glob("*.json"))


def test_pending_marker_is_replayed_on_next_logger_start(tmp_path: Path) -> None:
    pending = tmp_path / ".trackio" / "pending"
    pending.mkdir(parents=True)
    (pending / "batch-000001.json").write_text(
        json.dumps({"batch_index": 1, "completed_rows": 128})
    )
    fake = _Trackio()
    logger = TrackioBatchLogger(
        work_dir=tmp_path,
        project="project",
        run_name="run",
        lane=ReleaseLane.V2_WORLDWIDE,
        trackio_module=fake,
    )
    logger.start()
    _wait(logger, fake)

    assert [step for _, step in fake.log_calls] == [1]


def test_failed_upload_stays_pending_and_does_not_finish_run(tmp_path: Path) -> None:
    fake = _FailingTrackio()
    logger = TrackioBatchLogger(
        work_dir=tmp_path,
        project="worldwide",
        run_name="run",
        lane=ReleaseLane.V2_WORLDWIDE,
        trackio_module=fake,
    )
    logger.start()
    logger.enqueue({"batch_index": 0, "completed_rows": 128})
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and logger._queue.unfinished_tasks:
        time.sleep(0.01)
    logger.close(timeout=1)

    assert list((tmp_path / ".trackio" / "pending").glob("batch-*.json"))
    assert fake.finish_calls == 0


def test_batch_logger_rejects_invalid_lifecycle_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-blank"):
        TrackioBatchLogger(
            work_dir=tmp_path,
            project=" ",
            run_name="run",
            lane=ReleaseLane.V2_WORLDWIDE,
            trackio_module=_Trackio(),
        )
    logger = TrackioBatchLogger(
        work_dir=tmp_path,
        project="project",
        run_name="run",
        lane=ReleaseLane.V2_WORLDWIDE,
        trackio_module=_Trackio(),
    )
    with pytest.raises(ValueError, match="batch_index"):
        logger.enqueue({"batch_index": -1})
    with pytest.raises(ValueError, match="timeout"):
        logger.close(timeout=-1)
    logger.close()
    with pytest.raises(BatchTrackingError, match="closed"):
        logger.start()


def test_batch_logger_deduplicates_markers_and_start_is_idempotent(
    tmp_path: Path,
) -> None:
    fake = _Trackio()
    logger = TrackioBatchLogger(
        work_dir=tmp_path,
        project="project",
        run_name="run",
        lane=ReleaseLane.V2_WORLDWIDE,
        trackio_module=fake,
    )
    logger.start()
    logger.start()
    logger.enqueue({"batch_index": 0, "completed_rows": 1})
    logger.enqueue({"batch_index": 0, "completed_rows": 1})
    _wait(logger, fake)
    assert [step for _, step in fake.log_calls] == [0]


def test_batch_logger_discards_marker_already_uploaded(
    tmp_path: Path,
) -> None:
    fake = _Trackio()
    pending = tmp_path / ".trackio" / "pending"
    uploaded = tmp_path / ".trackio" / "uploaded"
    pending.mkdir(parents=True)
    uploaded.mkdir(parents=True)
    marker = pending / "batch-000000.json"
    marker.write_text(json.dumps({"batch_index": 0}))
    (uploaded / marker.name).write_text(marker.read_text())
    logger = TrackioBatchLogger(
        work_dir=tmp_path,
        project="project",
        run_name="run",
        lane=ReleaseLane.V2_WORLDWIDE,
        trackio_module=fake,
    )
    logger.start()
    logger.close(wait=False)
    assert not marker.exists()


def test_batch_logger_records_marker_that_is_not_an_object(tmp_path: Path) -> None:
    fake = _Trackio()
    pending = tmp_path / ".trackio" / "pending"
    pending.mkdir(parents=True)
    (pending / "batch-000000.json").write_text("[]")
    logger = TrackioBatchLogger(
        work_dir=tmp_path,
        project="project",
        run_name="run",
        lane=ReleaseLane.V2_WORLDWIDE,
        trackio_module=fake,
    )
    logger.start()
    logger.close(timeout=2)
    status = json.loads((tmp_path / ".trackio" / "status.json").read_text())
    assert status["last_error"] == "BatchTrackingError"
    assert pending.joinpath("batch-000000.json").exists()


def test_batch_logger_surfaces_finish_failure(tmp_path: Path) -> None:
    fake = _FinishFailingTrackio()
    logger = TrackioBatchLogger(
        work_dir=tmp_path,
        project="project",
        run_name="run",
        lane=ReleaseLane.V2_WORLDWIDE,
        trackio_module=fake,
    )
    logger.start()
    logger.enqueue({"batch_index": 0, "completed_rows": 1})
    deadline = time.monotonic() + 2
    while logger._queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(BatchTrackingError, match="finish"):
        logger.close(timeout=2)
