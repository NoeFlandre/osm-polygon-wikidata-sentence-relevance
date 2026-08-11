from __future__ import annotations

import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.labeling import v2_resumable_sampling
from osm_polygon_sentence_relevance.labeling.v2_resumable_sampling import (
    FinalizedShard,
    ResumableSamplingError,
    _build_plan,
    _heaps_from_database,
    _load_state,
    _materialize_heaps,
    _PersistentPlanner,
    _retain_shard,
    select_v2_shards_resumable,
)
from osm_polygon_sentence_relevance.labeling.v2_sampling import select_v2_rows


def _table(rows: int = 48) -> pa.Table:
    values: list[dict[str, object]] = []
    buckets = (("tiny", 0.05), ("small", 0.5), ("medium", 5.0), ("large", 20.0))
    for index in range(rows):
        bucket, area = buckets[index % len(buckets)]
        polygon = index % 13
        values.append(
            {
                "sentence_id": f"sentence-{index:04d}",
                "polygon_id": f"polygon-{polygon:03d}",
                "lat": float(-60 + polygon * 8),
                "lon": float(-150 + polygon * 20),
                "area_km2": area,
                "area_bucket": bucket,
                "language": ("en", "fr", "ar")[index % 3],
                "osm_primary_tag": ("natural=wood", "landuse=forest")[index % 2],
                "sentence_text_raw": f"Row {index}",
            }
        )
    first: dict[str, tuple[str, float]] = {}
    for value in values:
        polygon_id = str(value["polygon_id"])
        first.setdefault(
            polygon_id, (str(value["area_bucket"]), float(value["area_km2"]))
        )
        value["area_bucket"], value["area_km2"] = first[polygon_id]
    return pa.Table.from_pylist(values)


def _shards(tmp_path: Path) -> tuple[list[FinalizedShard], pa.Table]:
    source = _table()
    result: list[FinalizedShard] = []
    for index, table in enumerate((source.slice(0, 23), source.slice(23))):
        path = tmp_path / f"shard-{index}.parquet"
        pq.write_table(table, path, row_group_size=5)
        result.append(FinalizedShard(f"shard-{index}", path, str(index)))
    return result, source


@pytest.mark.parametrize("target", [1, 17, 48, 80])
def test_resumable_shards_match_in_memory_reference(
    tmp_path: Path, target: int
) -> None:
    shards, source = _shards(tmp_path)
    result = select_v2_shards_resumable(
        shards,
        tmp_path / "selected.parquet",
        target=target,
        seed="seed",
        state_dir=tmp_path / "state",
        batch_size=7,
    )

    expected = select_v2_rows(source, target=target, seed="seed")
    actual = pq.read_table(result)
    assert actual["sentence_id"].to_pylist() == expected["sentence_id"].to_pylist()
    assert actual.equals(expected)
    assert (tmp_path / "state" / "state.json").exists()
    assert (tmp_path / "state" / "sampling.sqlite3").exists()


def test_resumable_scans_project_only_columns_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shards, _ = _shards(tmp_path)
    real_parquet_file = v2_resumable_sampling.pq.ParquetFile
    calls: list[tuple[str, ...] | None] = []

    class TrackingParquetFile:
        def __init__(self, path: Path) -> None:
            self._inner = real_parquet_file(path)

        def iter_batches(self, *, batch_size: int, columns: list[str] | None = None):
            calls.append(tuple(columns) if columns is not None else None)
            return self._inner.iter_batches(batch_size=batch_size, columns=columns)

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    monkeypatch.setattr(v2_resumable_sampling.pq, "ParquetFile", TrackingParquetFile)
    select_v2_shards_resumable(
        shards,
        tmp_path / "selected.parquet",
        target=11,
        seed="seed",
        state_dir=tmp_path / "state",
        batch_size=7,
    )

    assert tuple(v2_resumable_sampling._PLANNING_COLUMNS) in calls
    assert tuple(v2_resumable_sampling._RETENTION_COLUMNS) in calls
    assert all(
        columns
        in {
            tuple(v2_resumable_sampling._PLANNING_COLUMNS),
            tuple(v2_resumable_sampling._RETENTION_COLUMNS),
        }
        for columns in calls
    )


def test_materialize_heaps_reads_only_row_groups_with_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shards, _ = _shards(tmp_path)
    planner = _PersistentPlanner(tmp_path / "sampling.sqlite3")
    try:
        offset = 0
        for shard in shards:
            planner.begin()
            row_count = pq.ParquetFile(shard.path).metadata.num_rows
            for batch in pq.ParquetFile(shard.path).iter_batches(batch_size=64):
                planner.observe(batch)
            planner.finish_planning_shard(shard, row_count, offset)
            offset += row_count
        plan = _build_plan(planner, target=1, seed="seed")
        stratum = next(iter(plan.quotas))
        planner.connection.execute(
            "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?)",
            (stratum, 0, "0" * 64, 0, shards[0].shard_key, 0),
        )
        planner.connection.commit()

        real_parquet_file = v2_resumable_sampling.pq.ParquetFile
        read_groups: list[int] = []

        class TrackingParquetFile:
            def __init__(self, path: Path) -> None:
                self._inner = real_parquet_file(path)

            def read_row_group(self, index: int):
                read_groups.append(index)
                return self._inner.read_row_group(index)

            def __getattr__(self, name: str) -> object:
                return getattr(self._inner, name)

        monkeypatch.setattr(
            v2_resumable_sampling.pq, "ParquetFile", TrackingParquetFile
        )
        v2_resumable_sampling._materialize_heaps(
            planner,
            shards,
            plan,
            batch_size=4,
            materialize_shard=None,
        )

        assert read_groups == [0]
    finally:
        planner.close()


def test_resumable_sampling_reuses_plan_when_target_expands(tmp_path: Path) -> None:
    shards, source = _shards(tmp_path)
    state_dir = tmp_path / "state"
    select_v2_shards_resumable(
        shards,
        tmp_path / "first.parquet",
        target=17,
        seed="seed",
        state_dir=state_dir,
        batch_size=7,
    )
    result = select_v2_shards_resumable(
        shards,
        tmp_path / "expanded.parquet",
        target=31,
        seed="seed",
        state_dir=state_dir,
        batch_size=7,
    )

    expected = select_v2_rows(source, target=31, seed="seed")
    assert (
        pq.read_table(result)["sentence_id"].to_pylist()
        == expected["sentence_id"].to_pylist()
    )


def test_resumable_sampling_rejects_shard_identity_change(tmp_path: Path) -> None:
    shards, _ = _shards(tmp_path)
    state_dir = tmp_path / "state"
    select_v2_shards_resumable(
        shards,
        tmp_path / "first.parquet",
        target=5,
        seed="seed",
        state_dir=state_dir,
    )
    changed = [
        FinalizedShard(shards[0].shard_key, shards[0].path, "changed"),
        shards[1],
    ]
    with pytest.raises(ValueError, match="shard identity"):
        select_v2_shards_resumable(
            changed,
            tmp_path / "changed.parquet",
            target=5,
            seed="seed",
            state_dir=state_dir,
        )


def test_resumable_sampling_keeps_completed_scans_after_output_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shards, _ = _shards(tmp_path)
    state_dir = tmp_path / "state"
    original = v2_resumable_sampling._write_selected
    monkeypatch.setattr(
        v2_resumable_sampling,
        "_write_selected",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("stop after scans")),
    )
    with pytest.raises(OSError, match="stop after scans"):
        select_v2_shards_resumable(
            shards,
            tmp_path / "failed.parquet",
            target=11,
            seed="seed",
            state_dir=state_dir,
        )

    calls: list[str] = []
    real_retain = v2_resumable_sampling._retain_shard

    def counted_retain(*args: object, **kwargs: object) -> None:
        calls.append(str(args[1].shard_key))
        real_retain(*args, **kwargs)

    monkeypatch.setattr(v2_resumable_sampling, "_write_selected", original)
    monkeypatch.setattr(v2_resumable_sampling, "_retain_shard", counted_retain)
    result = select_v2_shards_resumable(
        shards,
        tmp_path / "recovered.parquet",
        target=11,
        seed="seed",
        state_dir=state_dir,
    )
    assert result.exists()
    assert calls == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": True}, "target"),
        ({"target": 0}, "target"),
        ({"batch_size": True}, "batch_size"),
        ({"batch_size": 0}, "batch_size"),
        ({"seed": ""}, "seed"),
    ],
)
def test_resumable_sampling_rejects_invalid_configuration(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    shards, _ = _shards(tmp_path)
    values: dict[str, object] = {
        "target": 3,
        "seed": "seed",
        "batch_size": 4,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        select_v2_shards_resumable(
            shards,
            tmp_path / "selected-state-link.parquet",
            state_dir=tmp_path / "state",
            **values,
        )


def test_resumable_sampling_rejects_invalid_shard_inventory(tmp_path: Path) -> None:
    shards, _ = _shards(tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        select_v2_shards_resumable(
            [],
            tmp_path / "selected-state-link.parquet",
            target=1,
            seed="s",
            state_dir=tmp_path / "state",
        )
    with pytest.raises(ValueError, match="unique"):
        select_v2_shards_resumable(
            [shards[0], shards[0]],
            tmp_path / "selected.parquet",
            target=1,
            seed="s",
            state_dir=tmp_path / "state",
        )
    with pytest.raises(ValueError, match="regular file"):
        select_v2_shards_resumable(
            [FinalizedShard("bad", tmp_path / "missing.parquet", "id")],
            tmp_path / "selected.parquet",
            target=1,
            seed="s",
            state_dir=tmp_path / "state",
        )
    with pytest.raises(ValueError, match="identity"):
        select_v2_shards_resumable(
            [FinalizedShard("", None, "id")],
            tmp_path / "selected-missing-identity.parquet",
            target=1,
            seed="s",
            state_dir=tmp_path / "state-missing-identity",
        )


def test_resumable_sampling_rejects_corrupt_state_and_orphan_database(
    tmp_path: Path,
) -> None:
    shards, _ = _shards(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "state.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(ResumableSamplingError, match="state.json"):
        select_v2_shards_resumable(
            shards, tmp_path / "one.parquet", target=1, seed="s", state_dir=state
        )
    (state / "state.json").unlink()
    (state / "sampling.sqlite3").write_bytes(b"orphan")
    with pytest.raises(ResumableSamplingError, match="without state"):
        select_v2_shards_resumable(
            shards, tmp_path / "two.parquet", target=1, seed="s", state_dir=state
        )


def test_resumable_sampling_rejects_state_parameter_changes(tmp_path: Path) -> None:
    shards, _ = _shards(tmp_path)
    state = tmp_path / "state"
    select_v2_shards_resumable(
        shards, tmp_path / "first.parquet", target=3, seed="seed", state_dir=state
    )
    with pytest.raises(ValueError, match="seed mismatch"):
        select_v2_shards_resumable(
            shards, tmp_path / "seed.parquet", target=3, seed="other", state_dir=state
        )
    with pytest.raises(ValueError, match="cannot shrink"):
        select_v2_shards_resumable(
            shards, tmp_path / "shrink.parquet", target=2, seed="seed", state_dir=state
        )
    state.joinpath("state.json").write_text('{"schema_version": 999}', encoding="utf-8")
    with pytest.raises(ResumableSamplingError, match="schema"):
        select_v2_shards_resumable(
            shards, tmp_path / "schema.parquet", target=3, seed="seed", state_dir=state
        )


def test_persistent_planner_rejects_bad_rows_and_duplicate_ids(tmp_path: Path) -> None:
    planner = _PersistentPlanner(tmp_path / "sampling.sqlite3")
    try:
        planner.begin()
        missing = pa.record_batch([pa.array(["s1"])], names=["sentence_id"])
        with pytest.raises(ValueError, match="missing required"):
            planner.observe(missing)
        planner.rollback()
        table = _table(1)
        missing_coordinates = table.to_pylist()[0]
        missing_coordinates["lat"] = None
        planner.begin()
        assert (
            planner.observe(pa.Table.from_pylist([missing_coordinates]).to_batches()[0])
            == 1
        )
        planner.rollback()
        duplicate = pa.Table.from_pylist(table.to_pylist() * 2)
        planner.begin()
        with pytest.raises(ValueError, match="duplicate"):
            planner.observe(duplicate.to_batches()[0])
        planner.rollback()
        planner.begin()
        bad = table.to_pylist()[0]
        bad["sentence_id"] = ""
        with pytest.raises(ValueError, match="sentence_id"):
            planner.observe(pa.Table.from_pylist([bad]).to_batches()[0])
        planner.rollback()
    finally:
        planner.close()


def test_persistent_planner_rejects_nested_polygon_metadata_change(
    tmp_path: Path,
) -> None:
    planner = _PersistentPlanner(tmp_path / "sampling.sqlite3")
    try:
        planner.begin()
        rows = _table(2).to_pylist()
        rows[1]["polygon_id"] = rows[0]["polygon_id"]
        rows[1]["area_bucket"] = "small"
        with pytest.raises(ValueError, match="metadata"):
            planner.observe(pa.Table.from_pylist(rows).to_batches()[0])
        planner.rollback()
    finally:
        planner.close()


def test_materialize_heaps_rejects_missing_source_and_schema_changes(
    tmp_path: Path,
) -> None:
    shards, _ = _shards(tmp_path)
    planner = _PersistentPlanner(tmp_path / "sampling.sqlite3")
    try:
        planner.begin()
        for index, shard in enumerate(shards):
            planner.finish_planning_shard(
                shard, 23 if index == 0 else 25, 0 if index == 0 else 23
            )
        plan = _build_plan(planner, target=4, seed="seed")
        with pytest.raises(ResumableSamplingError, match="missing source"):
            _materialize_heaps(
                planner,
                [FinalizedShard("missing", None, "id")],
                plan,
                batch_size=4,
                materialize_shard=None,
            )
    finally:
        planner.close()


def test_retain_shard_requires_planning_checkpoint(tmp_path: Path) -> None:
    shards, _ = _shards(tmp_path)
    planner = _PersistentPlanner(tmp_path / "sampling.sqlite3")
    try:
        plan = _build_plan(planner, target=1, seed="seed")
        with pytest.raises(ResumableSamplingError, match="planning checkpoint"):
            _retain_shard(
                planner,
                shards[0],
                plan,
                _heaps_from_database(planner, plan),
                seed="seed",
                batch_size=4,
                materialize_shard=None,
            )
    finally:
        planner.close()


def test_retain_shard_checkpoints_within_a_shard_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shards, _ = _shards(tmp_path)
    planner = _PersistentPlanner(tmp_path / "sampling.sqlite3")
    try:
        offset = 0
        for shard in shards:
            planner.begin()
            row_count = pq.ParquetFile(shard.path).metadata.num_rows
            for batch in pq.ParquetFile(shard.path).iter_batches(batch_size=64):
                planner.observe(batch)
            planner.finish_planning_shard(shard, row_count, offset)
            offset += row_count
        plan = _build_plan(planner, target=11, seed="seed")
        heaps = _heaps_from_database(planner, plan)
        real_parquet_file = v2_resumable_sampling.pq.ParquetFile

        class FailingParquetFile:
            def __init__(self, path: Path) -> None:
                self._inner = real_parquet_file(path)

            def iter_batches(
                self, *, batch_size: int, columns: list[str] | None = None
            ):
                for index, batch in enumerate(
                    self._inner.iter_batches(batch_size=batch_size, columns=columns)
                ):
                    yield batch
                    if index == 0:
                        raise RuntimeError("interrupt retention")

        monkeypatch.setattr(v2_resumable_sampling.pq, "ParquetFile", FailingParquetFile)
        with pytest.raises(RuntimeError, match="interrupt retention"):
            _retain_shard(
                planner,
                shards[0],
                plan,
                heaps,
                seed="seed",
                batch_size=4,
                materialize_shard=None,
            )
        retained_offset, retained = planner.retention_progress()[shards[0].shard_key]
        assert retained_offset == 4
        assert not retained

        monkeypatch.setattr(v2_resumable_sampling.pq, "ParquetFile", real_parquet_file)
        _retain_shard(
            planner,
            shards[0],
            plan,
            _heaps_from_database(planner, plan),
            seed="seed",
            batch_size=4,
            materialize_shard=None,
        )
        retained_offset, retained = planner.retention_progress()[shards[0].shard_key]
        assert retained_offset == 23
        assert retained
    finally:
        planner.close()


def test_retention_does_not_recompute_h3_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shards, _ = _shards(tmp_path)
    planner = _PersistentPlanner(tmp_path / "sampling.sqlite3")
    try:
        planner.begin()
        for batch in pq.ParquetFile(shards[0].path).iter_batches(batch_size=64):
            planner.observe(batch)
        planner.finish_planning_shard(shards[0], 23, 0)
        plan = _build_plan(planner, target=1, seed="seed")
        monkeypatch.setattr(
            v2_resumable_sampling,
            "_cell",
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("retention should not recompute H3 cells")
            ),
        )

        _retain_shard(
            planner,
            shards[0],
            plan,
            _heaps_from_database(planner, plan),
            seed="seed",
            batch_size=4,
            materialize_shard=None,
        )
        assert planner.retention_progress()[shards[0].shard_key][1]
    finally:
        planner.close()


def test_retention_checkpoint_validates_offsets_and_migrates_old_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "old.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE shard_progress ("
        "shard_key TEXT PRIMARY KEY, row_count INTEGER NOT NULL DEFAULT 0, "
        "row_offset INTEGER NOT NULL DEFAULT 0, planned INTEGER NOT NULL DEFAULT 0, "
        "retained INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID"
    )
    connection.commit()
    connection.close()
    planner = _PersistentPlanner(database)
    try:
        assert "retain_offset" in {
            str(row[1])
            for row in planner.connection.execute("PRAGMA table_info(shard_progress)")
        }
        with pytest.raises(ValueError, match="integer"):
            planner.checkpoint_retaining_shard("missing", True)
        with pytest.raises(ValueError, match="non-negative"):
            planner.checkpoint_retaining_shard("missing", -1)
    finally:
        planner.close()


def test_retain_shard_rejects_invalid_checkpoint_and_skips_completed_shard(
    tmp_path: Path,
) -> None:
    shards, _ = _shards(tmp_path)
    planner = _PersistentPlanner(tmp_path / "sampling.sqlite3")
    try:
        planner.begin()
        planner.finish_planning_shard(shards[0], 23, 0)
        plan = _build_plan(planner, target=1, seed="seed")
        planner.connection.execute(
            "UPDATE shard_progress SET retain_offset = row_count, retained = 1 "
            "WHERE shard_key = ?",
            (shards[0].shard_key,),
        )
        planner.connection.commit()
        _retain_shard(
            planner,
            shards[0],
            plan,
            {},
            seed="seed",
            batch_size=4,
            materialize_shard=None,
        )
        planner.connection.execute(
            "UPDATE shard_progress SET retain_offset = row_count + 1, retained = 0 "
            "WHERE shard_key = ?",
            (shards[0].shard_key,),
        )
        planner.connection.commit()
        with pytest.raises(ResumableSamplingError, match="invalid retention"):
            _retain_shard(
                planner,
                shards[0],
                plan,
                {},
                seed="seed",
                batch_size=4,
                materialize_shard=None,
            )
    finally:
        planner.close()


def test_retain_shard_rolls_back_finish_failure_and_detects_source_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shards, _ = _shards(tmp_path)
    planner = _PersistentPlanner(tmp_path / "sampling.sqlite3")
    try:
        planner.begin()
        for batch in pq.ParquetFile(shards[0].path).iter_batches(batch_size=64):
            planner.observe(batch)
        planner.finish_planning_shard(shards[0], 22, 0)
        plan = _build_plan(planner, target=1, seed="seed")
        with pytest.raises(ValueError, match="source changed"):
            _retain_shard(
                planner,
                shards[0],
                plan,
                _heaps_from_database(planner, plan),
                seed="seed",
                batch_size=64,
                materialize_shard=None,
            )

        planner.connection.execute(
            "UPDATE shard_progress SET retain_offset = 0, row_count = 23 "
            "WHERE shard_key = ?",
            (shards[0].shard_key,),
        )
        planner.connection.commit()
        real_finish = planner.finish_retaining_shard

        def fail_finish(shard_key: str) -> None:
            raise RuntimeError(f"finish {shard_key}")

        monkeypatch.setattr(planner, "finish_retaining_shard", fail_finish)
        with pytest.raises(RuntimeError, match="finish"):
            _retain_shard(
                planner,
                shards[0],
                plan,
                _heaps_from_database(planner, plan),
                seed="seed",
                batch_size=64,
                materialize_shard=None,
            )
        assert not planner.connection.in_transaction
        monkeypatch.setattr(planner, "finish_retaining_shard", real_finish)
    finally:
        planner.close()


def test_public_sampler_rejects_missing_planning_source(tmp_path: Path) -> None:
    with pytest.raises(ResumableSamplingError, match="missing source"):
        select_v2_shards_resumable(
            [FinalizedShard("missing", None, "identity")],
            tmp_path / "selected.parquet",
            target=1,
            seed="seed",
            state_dir=tmp_path / "state",
        )


def test_load_state_rejects_unsupported_schema(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text('{"schema_version": 99}', encoding="utf-8")
    with pytest.raises(ResumableSamplingError, match="unsupported"):
        _load_state(state)


def test_resumable_sampling_rejects_symlinked_paths(tmp_path: Path) -> None:
    shards, _ = _shards(tmp_path)
    output_target = tmp_path / "output-target"
    output_target.write_text("x", encoding="utf-8")
    output_link = tmp_path / "selected.parquet"
    output_link.symlink_to(output_target)
    with pytest.raises(ValueError, match="output"):
        select_v2_shards_resumable(
            shards, output_link, target=1, seed="seed", state_dir=tmp_path / "state"
        )
    state_target = tmp_path / "state-target"
    state_target.mkdir()
    state_link = tmp_path / "state-link"
    state_link.symlink_to(state_target, target_is_directory=True)
    with pytest.raises(ValueError, match="state directory"):
        select_v2_shards_resumable(
            shards,
            tmp_path / "selected-state-link.parquet",
            target=1,
            seed="seed",
            state_dir=state_link,
        )


def test_persistent_planner_rejects_open_transaction_and_missing_meta(
    tmp_path: Path,
) -> None:
    planner = _PersistentPlanner(tmp_path / "sampling.sqlite3")
    try:
        planner.begin()
        with pytest.raises(ResumableSamplingError, match="already open"):
            planner.begin()
        planner.rollback()
        with pytest.raises(ResumableSamplingError, match="missing"):
            planner._meta("missing")
    finally:
        planner.close()


def test_persistent_planner_rejects_blank_polygon_id(tmp_path: Path) -> None:
    planner = _PersistentPlanner(tmp_path / "sampling.sqlite3")
    try:
        row = _table(1).to_pylist()[0]
        row["polygon_id"] = ""
        planner.begin()
        with pytest.raises(ValueError, match="polygon_id"):
            planner.observe(pa.Table.from_pylist([row]).to_batches()[0])
        planner.rollback()
    finally:
        planner.close()


def test_retain_shard_rejects_missing_materialized_source(tmp_path: Path) -> None:
    shards, _ = _shards(tmp_path)
    planner = _PersistentPlanner(tmp_path / "sampling.sqlite3")
    try:
        planner.begin()
        planner.finish_planning_shard(shards[0], 23, 0)
        plan = _build_plan(planner, target=1, seed="seed")
        with pytest.raises(ResumableSamplingError, match="missing source"):
            _retain_shard(
                planner,
                FinalizedShard(shards[0].shard_key, None, shards[0].identity),
                plan,
                {},
                seed="seed",
                batch_size=4,
                materialize_shard=None,
            )
    finally:
        planner.close()


def test_materialize_heaps_rejects_schema_and_candidate_changes(tmp_path: Path) -> None:
    shards, source = _shards(tmp_path)
    planner = _PersistentPlanner(tmp_path / "sampling.sqlite3")
    try:
        offset = 0
        for shard in shards:
            planner.begin()
            row_count = pq.ParquetFile(shard.path).metadata.num_rows
            for batch in pq.ParquetFile(shard.path).iter_batches(batch_size=64):
                planner.observe(batch)
            planner.finish_planning_shard(shard, row_count, offset)
            offset += row_count
        plan = _build_plan(planner, target=4, seed="seed")
        planner.connection.execute(
            "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?)",
            (next(iter(plan.quotas)), 0, "0" * 64, 9999, shards[0].shard_key, 9999),
        )
        planner.connection.commit()
        with pytest.raises(ValueError, match="no longer exist"):
            _materialize_heaps(
                planner,
                shards,
                plan,
                batch_size=4,
                materialize_shard=None,
            )
        planner.connection.execute("DELETE FROM candidates WHERE global_index = 9999")
        planner.connection.commit()
        wrong = tmp_path / "wrong.parquet"
        pq.write_table(pa.table({"wrong": [1]}), wrong)

        def materialize(shard: FinalizedShard):
            from contextlib import nullcontext

            return nullcontext(
                wrong if shard.shard_key == shards[1].shard_key else shard.path
            )

        with pytest.raises(ValueError, match="schemas do not match"):
            _materialize_heaps(
                planner,
                shards,
                plan,
                batch_size=4,
                materialize_shard=materialize,
            )
    finally:
        planner.close()
