"""Durable, shard-by-shard V2 sampling.

The bounded selector keeps the deterministic in-memory contract from
``v2_sampling`` while making the two expensive scans restartable.  A small
SQLite ledger records sentence IDs, polygon metadata, per-shard progress, and
the bounded candidate heaps.  The ledger lives outside allocation scratch, so
an interrupted OAR job can continue without reprocessing completed shards.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .v2_bounded_sampling import (
    _REQUIRED,
    SamplingPlan,
    _write_selected,
)
from .v2_sampling import (
    _MISSING_CELL,
    _cell,
    _rank,
    _weighted_schedule,
    canonical_area_bucket,
    ordered_polygon_ids,
)

_SCHEMA_VERSION = 1
_STATE_FILENAME = "state.json"
_DATABASE_FILENAME = "sampling.sqlite3"
_MATERIALIZED_DIRNAME = "materialized"
_MATERIALIZED_CACHE_VERSION = 1
_PLANNING_COLUMNS: tuple[str, ...] = (
    "sentence_id",
    "polygon_id",
    "lat",
    "lon",
    "area_km2",
    "area_bucket",
)
_RETENTION_COLUMNS: tuple[str, ...] = (
    "sentence_id",
    "polygon_id",
    "lat",
    "lon",
)


class ResumableSamplingError(RuntimeError):
    """The durable V2 sampling checkpoint is invalid or unavailable."""


@dataclass(frozen=True, slots=True)
class FinalizedShard:
    """One immutable, already-finalized shard available to the sampler."""

    shard_key: str
    path: Path | None
    identity: str


@dataclass(frozen=True, slots=True)
class _CandidateLocator:
    stratum: str
    polygon_order: int
    rank_hex: str
    global_index: int
    shard_key: str
    row_index: int

    @property
    def heap_value(self) -> tuple[int, int, int, str, int]:
        return (
            -self.polygon_order,
            -int(self.rank_hex, 16),
            -self.global_index,
            self.shard_key,
            self.row_index,
        )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _validate_shards(shards: Sequence[FinalizedShard]) -> tuple[FinalizedShard, ...]:
    ordered = tuple(sorted(shards, key=lambda item: item.shard_key))
    if not ordered:
        raise ValueError("at least one finalized shard is required")
    if len({item.shard_key for item in ordered}) != len(ordered):
        raise ValueError("finalized shard keys must be unique")
    for item in ordered:
        if not item.shard_key or not item.identity:
            raise ValueError("finalized shard identity is required")
        if item.path is not None and (
            item.path.is_symlink() or not item.path.is_file()
        ):
            raise ValueError(f"finalized shard is not a regular file: {item.shard_key}")
    return ordered


class _PersistentPlanner:
    def __init__(self, database: Path) -> None:
        self.database = database
        self.connection = sqlite3.connect(database)
        # The checkpoint is stored on Grid'5000 home/NFS.  The rollback
        # journal is safer there than WAL's sidecar files and still gives us
        # atomic shard transactions when an allocation is terminated.
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sentence_ids (
                value TEXT PRIMARY KEY
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS polygon_metadata (
                polygon_id TEXT PRIMARY KEY,
                area_bucket TEXT NOT NULL,
                h3_cell TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS stratum_sizes (
                stratum TEXT PRIMARY KEY,
                row_count INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS shard_progress (
                shard_key TEXT PRIMARY KEY,
                row_count INTEGER NOT NULL DEFAULT 0,
                row_offset INTEGER NOT NULL DEFAULT 0,
                planned INTEGER NOT NULL DEFAULT 0,
                retained INTEGER NOT NULL DEFAULT 0,
                retain_offset INTEGER NOT NULL DEFAULT 0
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS candidates (
                stratum TEXT NOT NULL,
                polygon_order INTEGER NOT NULL,
                rank_hex TEXT NOT NULL,
                global_index INTEGER NOT NULL,
                shard_key TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                PRIMARY KEY (stratum, global_index)
            ) WITHOUT ROWID;
            """
        )
        progress_columns = {
            str(row[1])
            for row in self.connection.execute("PRAGMA table_info(shard_progress)")
        }
        if "retain_offset" not in progress_columns:
            self.connection.execute(
                "ALTER TABLE shard_progress ADD COLUMN retain_offset INTEGER NOT NULL DEFAULT 0"
            )
        self.connection.commit()
        self.total_rows = int(self._meta("total_rows", "0"))
        self.polygon_metadata = {
            polygon: (bucket, cell)
            for polygon, bucket, cell in self.connection.execute(
                "SELECT polygon_id, area_bucket, h3_cell FROM polygon_metadata"
            )
        }
        self.stratum_sizes = dict(
            self.connection.execute("SELECT stratum, row_count FROM stratum_sizes")
        )

    def _meta(self, key: str, default: str | None = None) -> str:
        row = self.connection.execute(
            "SELECT value FROM run_meta WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            if default is None:
                raise ResumableSamplingError(f"sampling metadata is missing {key}")
            return default
        return str(row[0])

    def begin(self) -> None:
        if self.connection.in_transaction:
            raise ResumableSamplingError("sampling transaction is already open")
        self.connection.execute("BEGIN IMMEDIATE")

    def observe(self, batch: pa.RecordBatch, *, validate_schema: bool = True) -> int:
        required = _REQUIRED if validate_schema else set(_PLANNING_COLUMNS)
        missing = required.difference(batch.schema.names)
        if missing:
            raise ValueError(
                f"V2 sampling input is missing required columns: {sorted(missing)}"
            )
        sentence_ids = batch["sentence_id"].to_pylist()
        polygon_ids = batch["polygon_id"].to_pylist()
        latitudes = batch["lat"].to_pylist()
        longitudes = batch["lon"].to_pylist()
        areas = batch["area_km2"].to_pylist()
        buckets = batch["area_bucket"].to_pylist()
        identifiers: list[tuple[str]] = []
        metadata_rows: dict[str, tuple[str, str]] = {}
        counts: dict[str, int] = {}
        for sentence_id, polygon_id, lat, lon, area, bucket in zip(
            sentence_ids,
            polygon_ids,
            latitudes,
            longitudes,
            areas,
            buckets,
            strict=True,
        ):
            if not isinstance(sentence_id, str) or not sentence_id:
                raise ValueError("sentence_id must be non-empty")
            if not isinstance(polygon_id, str) or not polygon_id:
                raise ValueError("polygon_id must be non-empty")
            identifiers.append((sentence_id,))
            cell = _cell(lat, lon)
            if cell == _MISSING_CELL:
                continue
            metadata = (
                canonical_area_bucket(area, bucket),
                cell,
            )
            prior = metadata_rows.get(polygon_id, self.polygon_metadata.get(polygon_id))
            if prior is not None and prior != metadata:
                raise ValueError("polygon metadata is inconsistent")
            metadata_rows[polygon_id] = metadata
            counts[cell] = counts.get(cell, 0) + 1
        try:
            self.connection.executemany(
                "INSERT INTO sentence_ids(value) VALUES (?)", identifiers
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("sampling input contains duplicate sentence IDs") from exc
        self.connection.executemany(
            "INSERT OR IGNORE INTO polygon_metadata VALUES (?, ?, ?)",
            (
                (polygon_id, bucket, cell)
                for polygon_id, (bucket, cell) in metadata_rows.items()
            ),
        )
        for polygon_id, (bucket, cell) in metadata_rows.items():
            self.polygon_metadata[polygon_id] = (bucket, cell)
        stratum_rows = list(counts.items())
        for stratum, count in stratum_rows:
            self.stratum_sizes[stratum] = self.stratum_sizes.get(stratum, 0) + count
        self.connection.executemany(
            "INSERT INTO stratum_sizes(stratum, row_count) VALUES (?, ?) "
            "ON CONFLICT(stratum) DO UPDATE SET row_count = row_count + excluded.row_count",
            stratum_rows,
        )
        return len(sentence_ids)

    def finish_planning_shard(
        self, shard: FinalizedShard, row_count: int, offset: int
    ) -> None:
        self.connection.execute(
            "INSERT INTO shard_progress(shard_key, row_count, row_offset, planned) "
            "VALUES (?, ?, ?, 1) "
            "ON CONFLICT(shard_key) DO UPDATE SET row_count=excluded.row_count, "
            "row_offset=excluded.row_offset, planned=1",
            (shard.shard_key, row_count, offset),
        )
        self.total_rows = offset + row_count
        self.connection.execute(
            "INSERT INTO run_meta(key, value) VALUES ('total_rows', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(self.total_rows),),
        )
        self.connection.commit()

    def finish_retaining_shard(self, shard_key: str) -> None:
        self.connection.execute(
            "UPDATE shard_progress SET retained=1, retain_offset=row_count "
            "WHERE shard_key = ?",
            (shard_key,),
        )
        self.connection.commit()

    def checkpoint_retaining_shard(self, shard_key: str, row_offset: int) -> None:
        """Persist progress after one bounded retention batch."""

        if isinstance(row_offset, bool) or not isinstance(row_offset, int):
            raise ValueError("retention row offset must be an integer")
        if row_offset < 0:
            raise ValueError("retention row offset must be non-negative")
        self.connection.execute(
            "UPDATE shard_progress SET retain_offset = ? WHERE shard_key = ?",
            (row_offset, shard_key),
        )
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()
        self.total_rows = int(self._meta("total_rows", "0"))
        self.polygon_metadata = {
            polygon: (bucket, cell)
            for polygon, bucket, cell in self.connection.execute(
                "SELECT polygon_id, area_bucket, h3_cell FROM polygon_metadata"
            )
        }
        self.stratum_sizes = dict(
            self.connection.execute("SELECT stratum, row_count FROM stratum_sizes")
        )

    def progress(self) -> dict[str, tuple[int, int, bool, bool]]:
        return {
            key: (rows, offset, bool(planned), bool(retained))
            for key, rows, offset, planned, retained in self.connection.execute(
                "SELECT shard_key, row_count, row_offset, planned, retained "
                "FROM shard_progress"
            )
        }

    def retention_progress(self) -> dict[str, tuple[int, bool]]:
        """Return the local row offset and completion flag for each shard."""

        return {
            key: (offset, bool(retained))
            for key, offset, retained in self.connection.execute(
                "SELECT shard_key, retain_offset, retained FROM shard_progress"
            )
        }

    def clear_candidates(self) -> None:
        self.connection.execute("DELETE FROM candidates")
        self.connection.execute("UPDATE shard_progress SET retained=0, retain_offset=0")
        self.connection.commit()

    def iter_candidate_locators(self) -> Iterator[_CandidateLocator]:
        for stratum, order, rank, index, shard, row in self.connection.execute(
            "SELECT stratum, polygon_order, rank_hex, global_index, shard_key, row_index "
            "FROM candidates"
        ):
            yield _CandidateLocator(stratum, order, rank, index, shard, row)

    def compact_candidates(
        self,
        heaps: Mapping[str, Sequence[tuple[int, int, int, str, int]]],
    ) -> None:
        """Persist only the quota-best candidates recovered from the ledger."""

        keep = [
            (stratum, -candidate[2])
            for stratum, heap in heaps.items()
            for candidate in heap
        ]
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "CREATE TEMP TABLE candidate_keep ("
                "stratum TEXT NOT NULL, global_index INTEGER NOT NULL, "
                "PRIMARY KEY (stratum, global_index)"
                ") WITHOUT ROWID"
            )
            self.connection.executemany(
                "INSERT INTO candidate_keep VALUES (?, ?)", keep
            )
            self.connection.execute(
                "DELETE FROM candidates WHERE NOT EXISTS ("
                "SELECT 1 FROM candidate_keep "
                "WHERE candidate_keep.stratum = candidates.stratum "
                "AND candidate_keep.global_index = candidates.global_index"
                ")"
            )
            self.connection.execute("DROP TABLE candidate_keep")
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def close(self) -> None:
        self.connection.close()


def _build_plan(planner: _PersistentPlanner, *, target: int, seed: str) -> SamplingPlan:
    ordered = ordered_polygon_ids(planner.polygon_metadata, seed=seed)
    eligible_rows = sum(planner.stratum_sizes.values())
    schedule = _weighted_schedule(
        planner.stratum_sizes,
        rank=lambda key: _rank(seed, key),
        limit=min(target, eligible_rows),
    )
    return SamplingPlan(
        polygon_order={value: index for index, value in enumerate(ordered)},
        polygon_metadata=dict(planner.polygon_metadata),
        schedule=tuple(schedule),
        quotas=dict(Counter(schedule)),
        total_rows=planner.total_rows,
    )


def _heaps_from_database(
    planner: _PersistentPlanner, plan: SamplingPlan
) -> dict[str, list[tuple[int, int, int, str, int]]]:
    import heapq

    heaps: dict[str, list[tuple[int, int, int, str, int]]] = {
        stratum: [] for stratum in plan.quotas
    }
    for item in planner.iter_candidate_locators():
        quota = plan.quotas.get(item.stratum)
        if quota is None:
            continue
        candidate = item.heap_value
        heap = heaps[item.stratum]
        if len(heap) < quota:
            heapq.heappush(heap, candidate)
        elif candidate[:3] > heap[0][:3]:
            heapq.heapreplace(heap, candidate)
    for heap in heaps.values():
        heapq.heapify(heap)
    planner.compact_candidates(heaps)
    return heaps


def _retain_shard(
    planner: _PersistentPlanner,
    shard: FinalizedShard,
    plan: SamplingPlan,
    heaps: dict[str, list[tuple[int, int, int, str, int]]],
    *,
    seed: str,
    batch_size: int,
    materialize_shard: Callable[[FinalizedShard], AbstractContextManager[Path]] | None,
) -> None:
    import heapq

    progress = planner.progress().get(shard.shard_key)
    if progress is None or not progress[2]:
        raise ResumableSamplingError(
            f"missing planning checkpoint for {shard.shard_key}"
        )
    offset = progress[1]
    retain_offset, retained = planner.retention_progress().get(
        shard.shard_key, (0, False)
    )
    if retained:
        return
    if retain_offset < 0 or retain_offset > progress[0]:
        raise ResumableSamplingError(
            f"invalid retention checkpoint for {shard.shard_key}"
        )
    source = (
        materialize_shard(shard)
        if materialize_shard is not None
        else nullcontext(shard.path)
    )
    try:
        with source as path:
            if path is None:
                raise ResumableSamplingError(
                    f"missing source path for {shard.shard_key}"
                )
            local_row_index = 0
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=batch_size, columns=list(_RETENTION_COLUMNS)
            ):
                batch_start = local_row_index
                batch_end = batch_start + batch.num_rows
                local_row_index = batch_end
                if batch_end <= retain_offset:
                    continue
                start = max(0, retain_offset - batch_start)
                sentence_ids = batch["sentence_id"].to_pylist()[start:]
                polygon_ids = batch["polygon_id"].to_pylist()[start:]
                latitudes = batch["lat"].to_pylist()[start:]
                longitudes = batch["lon"].to_pylist()[start:]
                pending_deletes: list[tuple[str, int]] = []
                pending_inserts: list[tuple[str, int, str, int, str, int]] = []
                planner.begin()
                try:
                    for relative_index, (
                        sentence_id,
                        polygon_id,
                        lat,
                        lon,
                    ) in enumerate(
                        zip(
                            sentence_ids,
                            polygon_ids,
                            latitudes,
                            longitudes,
                            strict=True,
                        ),
                        start=max(retain_offset, batch_start),
                    ):
                        row_index = relative_index
                        global_index = offset + row_index
                        # Planning already validates immutable source coordinates
                        # and records the H3 stratum. Retention only needs to
                        # preserve the contract that missing coordinates are
                        # excluded, so avoid recomputing H3 for every row.
                        if lat is None or lon is None:
                            continue
                        polygon_id = str(polygon_id)
                        stratum = plan.polygon_metadata[polygon_id][1]
                        quota = plan.quotas.get(stratum, 0)
                        if quota:
                            rank_hex = _rank(seed, str(sentence_id))
                            candidate = (
                                -plan.polygon_order[polygon_id],
                                -int(rank_hex, 16),
                                -global_index,
                                shard.shard_key,
                                row_index,
                            )
                            heap = heaps[stratum]
                            if len(heap) < quota:
                                heapq.heappush(heap, candidate)
                                pending_inserts.append(
                                    (
                                        stratum,
                                        -candidate[0],
                                        rank_hex,
                                        global_index,
                                        shard.shard_key,
                                        row_index,
                                    )
                                )
                            elif candidate[:3] > heap[0][:3]:
                                old = heapq.heapreplace(heap, candidate)
                                pending_deletes.append((stratum, -old[2]))
                                pending_inserts.append(
                                    (
                                        stratum,
                                        -candidate[0],
                                        rank_hex,
                                        global_index,
                                        shard.shard_key,
                                        row_index,
                                    )
                                )
                    planner.connection.executemany(
                        "DELETE FROM candidates WHERE stratum=? AND global_index=?",
                        pending_deletes,
                    )
                    planner.connection.executemany(
                        "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?)",
                        pending_inserts,
                    )
                    planner.checkpoint_retaining_shard(shard.shard_key, batch_end)
                except BaseException:
                    planner.rollback()
                    raise
            if local_row_index != progress[0]:
                raise ValueError("V2 source changed between sampling passes")
            planner.begin()
            try:
                planner.finish_retaining_shard(shard.shard_key)
            except BaseException:
                planner.rollback()
                raise
    except BaseException:
        raise


def _materialize_heaps(
    planner: _PersistentPlanner,
    shards: Sequence[FinalizedShard],
    plan: SamplingPlan,
    *,
    batch_size: int,
    materialize_shard: Callable[[FinalizedShard], AbstractContextManager[Path]] | None,
    materialized_cache_dir: Path | None = None,
) -> tuple[dict[str, list[tuple[int, int, int, dict[str, Any]]]], pa.Schema]:
    by_shard: dict[str, list[_CandidateLocator]] = defaultdict(list)
    for item in planner.iter_candidate_locators():
        by_shard[item.shard_key].append(item)
    for shard_locators in by_shard.values():
        shard_locators.sort(key=lambda item: item.row_index)
    heaps: dict[str, list[tuple[int, int, int, dict[str, Any]]]] = {
        stratum: [] for stratum in plan.quotas
    }
    schema: pa.Schema | None = None
    for shard in shards:
        shard_locators = by_shard.get(shard.shard_key, [])
        if materialized_cache_dir is not None and shard_locators:
            cached = _load_materialized_rows(
                materialized_cache_dir,
                shard,
                shard_locators,
                schema,
            )
            if cached is not None:
                if schema is None:
                    schema = cached.schema
                _append_materialized_rows(heaps, shard_locators, cached)
                continue
        source = (
            materialize_shard(shard)
            if materialize_shard is not None
            else nullcontext(shard.path)
        )
        with source as path:
            if path is None:
                raise ResumableSamplingError(
                    f"missing source path for {shard.shard_key}"
                )
            current_schema = pq.read_schema(path)
            if schema is None:
                schema = current_schema
            elif current_schema != schema:
                raise ValueError("V2 source shard schemas do not match")
            parquet = pq.ParquetFile(path)
            locator_index = 0
            row_offset = 0
            selected_tables: list[pa.Table] = []
            for row_group_index in range(parquet.num_row_groups):
                row_group_rows = parquet.metadata.row_group(row_group_index).num_rows
                row_group_end = row_offset + row_group_rows
                selected: list[_CandidateLocator] = []
                while locator_index < len(shard_locators):
                    item = shard_locators[locator_index]
                    if item.row_index >= row_group_end:
                        break
                    if item.row_index < row_offset:
                        raise ValueError(
                            "V2 sampling candidates no longer exist in their source"
                        )
                    selected.append(item)
                    locator_index += 1
                if selected:
                    row_group = parquet.read_row_group(row_group_index)
                    selected_offsets = pa.array(
                        [item.row_index - row_offset for item in selected],
                        type=pa.int64(),
                    )
                    selected_tables.append(row_group.take(selected_offsets))
                row_offset = row_group_end
            if row_offset != planner.progress()[shard.shard_key][0]:
                raise ValueError("V2 source changed between sampling passes")
            if locator_index != len(shard_locators):
                raise ValueError(
                    "V2 sampling candidates no longer exist in their source"
                )
            if selected_tables:
                selected_table = pa.concat_tables(selected_tables)
                if materialized_cache_dir is not None:
                    _write_materialized_rows(
                        materialized_cache_dir,
                        shard,
                        shard_locators,
                        selected_table,
                    )
                _append_materialized_rows(heaps, shard_locators, selected_table)
    if schema is None:
        raise ResumableSamplingError("V2 sampling has no source schema")
    return heaps, schema


def _state_payload(
    *,
    seed: str,
    target: int,
    phase: str,
    shards: Sequence[FinalizedShard],
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "sampling_seed": seed,
        "sampling_target": target,
        "phase": phase,
        "shards": {item.shard_key: {"identity": item.identity} for item in shards},
    }


def _load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumableSamplingError("sampling state.json is invalid") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _SCHEMA_VERSION
    ):
        raise ResumableSamplingError("sampling state schema is unsupported")
    return payload


def _materialized_cache_paths(cache_dir: Path, shard_key: str) -> tuple[Path, Path]:
    digest = hashlib.sha256(shard_key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.parquet", cache_dir / f"{digest}.json"


def _schema_fingerprint(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _locator_fingerprint(locators: Sequence[_CandidateLocator]) -> str:
    digest = hashlib.sha256()
    for item in locators:
        digest.update(
            f"{item.stratum}\0{item.polygon_order}\0{item.rank_hex}\0"
            f"{item.global_index}\0{item.shard_key}\0{item.row_index}\n".encode()
        )
    return digest.hexdigest()


def _load_materialized_rows(
    cache_dir: Path,
    shard: FinalizedShard,
    locators: Sequence[_CandidateLocator],
    expected_schema: pa.Schema | None,
) -> pa.Table | None:
    table_path, metadata_path = _materialized_cache_paths(cache_dir, shard.shard_key)
    if not table_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(metadata, dict)
        or metadata.get("cache_version") != _MATERIALIZED_CACHE_VERSION
        or metadata.get("shard_key") != shard.shard_key
        or metadata.get("identity") != shard.identity
        or metadata.get("row_count") != len(locators)
        or metadata.get("locator_sha256") != _locator_fingerprint(locators)
    ):
        return None
    try:
        table = pq.read_table(table_path)
    except (OSError, pa.ArrowException):
        return None
    if table.num_rows != len(locators):
        return None
    if metadata.get("schema_sha256") != _schema_fingerprint(table.schema):
        return None
    if expected_schema is not None and not table.schema.equals(expected_schema):
        raise ValueError("V2 source shard schemas do not match")
    return table


def _write_materialized_rows(
    cache_dir: Path,
    shard: FinalizedShard,
    locators: Sequence[_CandidateLocator],
    table: pa.Table,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cache_dir, 0o700)
    table_path, metadata_path = _materialized_cache_paths(cache_dir, shard.shard_key)
    temporary = table_path.with_name(f".{table_path.name}.tmp-{os.getpid()}")
    pq.write_table(table, temporary, compression="zstd")
    os.chmod(temporary, 0o600)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, table_path)
    _atomic_json(
        metadata_path,
        {
            "cache_version": _MATERIALIZED_CACHE_VERSION,
            "shard_key": shard.shard_key,
            "identity": shard.identity,
            "row_count": len(locators),
            "locator_sha256": _locator_fingerprint(locators),
            "schema_sha256": _schema_fingerprint(table.schema),
        },
    )


def _append_materialized_rows(
    heaps: dict[str, list[tuple[int, int, int, dict[str, Any]]]],
    locators: Sequence[_CandidateLocator],
    table: pa.Table,
) -> None:
    for item, row in zip(locators, table.to_pylist(), strict=True):
        heaps[item.stratum].append((*item.heap_value[:3], row))


def select_v2_shards_resumable(
    shards: Sequence[FinalizedShard],
    output_path: Path,
    *,
    target: int,
    seed: str,
    state_dir: Path,
    batch_size: int = 65_536,
    materialize_shard: Callable[[FinalizedShard], AbstractContextManager[Path]]
    | None = None,
) -> Path:
    """Select the deterministic V2 prefix while checkpointing every shard."""

    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("target must be a positive integer")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be non-blank")
    ordered = _validate_shards(shards)
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise ValueError("V2 output must be fresh")
    state_root = Path(state_dir)
    if state_root.exists() and state_root.is_symlink():
        raise ValueError("sampling state directory must not be a symlink")
    state_root.mkdir(parents=True, exist_ok=True)
    os.chmod(state_root, 0o700)
    state_path = state_root / _STATE_FILENAME
    database_path = state_root / _DATABASE_FILENAME
    state = _load_state(state_path)
    expected_shards = {item.shard_key: item.identity for item in ordered}
    if state is not None:
        if state.get("sampling_seed") != seed:
            raise ValueError("sampling state seed mismatch")
        old_shards = state.get("shards")
        if (
            not isinstance(old_shards, dict)
            or {
                key: value.get("identity")
                for key, value in old_shards.items()
                if isinstance(value, dict)
            }
            != expected_shards
        ):
            raise ValueError("sampling shard identity mismatch")
        old_target = state.get("sampling_target")
        if not isinstance(old_target, int) or target < old_target:
            raise ValueError("sampling target cannot shrink")
    elif database_path.exists():
        raise ResumableSamplingError("sampling database exists without state.json")

    planner = _PersistentPlanner(database_path)
    try:
        current_target = int(state["sampling_target"]) if state else target
        if state is None:
            state = _state_payload(
                seed=seed, target=target, phase="planning", shards=ordered
            )
            _atomic_json(state_path, state)
        elif target > current_target:
            planner.clear_candidates()
            state = _state_payload(
                seed=seed, target=target, phase="planned", shards=ordered
            )
            _atomic_json(state_path, state)
        else:
            target = current_target

        progress = planner.progress()
        offset = sum(item[0] for item in progress.values())
        for shard in ordered:
            existing = planner.progress().get(shard.shard_key)
            if existing is not None and existing[2]:
                continue
            planner.begin()
            try:
                row_count = 0
                source = (
                    materialize_shard(shard)
                    if materialize_shard is not None
                    else nullcontext(shard.path)
                )
                with source as path:
                    if path is None:
                        raise ResumableSamplingError(
                            f"missing source path for {shard.shard_key}"
                        )
                    source_schema = pq.read_schema(path)
                    missing = _REQUIRED.difference(source_schema.names)
                    if missing:
                        raise ValueError(
                            "V2 sampling input is missing required columns: "
                            f"{sorted(missing)}"
                        )
                    for batch in pq.ParquetFile(path).iter_batches(
                        batch_size=batch_size, columns=list(_PLANNING_COLUMNS)
                    ):
                        row_count += planner.observe(batch, validate_schema=False)
                planner.finish_planning_shard(shard, row_count, offset)
            except BaseException:
                planner.rollback()
                raise
            offset += row_count
            state = _state_payload(
                seed=seed, target=target, phase="planning", shards=ordered
            )
            _atomic_json(state_path, state)

        plan = _build_plan(planner, target=target, seed=seed)
        state = _state_payload(
            seed=seed, target=target, phase="retaining", shards=ordered
        )
        _atomic_json(state_path, state)
        heaps = _heaps_from_database(planner, plan)
        for shard in ordered:
            existing = planner.progress().get(shard.shard_key)
            if existing is not None and existing[3]:
                continue
            _retain_shard(
                planner,
                shard,
                plan,
                heaps,
                seed=seed,
                batch_size=batch_size,
                materialize_shard=materialize_shard,
            )
            _atomic_json(
                state_path,
                _state_payload(
                    seed=seed, target=target, phase="retaining", shards=ordered
                ),
            )

        # Retention can leave superseded candidates in the ledger. Compact
        # before materialization so the durable row cache represents exactly
        # the final deterministic sample and can be reused after a restart.
        planner.compact_candidates(heaps)
        _atomic_json(
            state_path,
            _state_payload(
                seed=seed, target=target, phase="materializing", shards=ordered
            ),
        )
        heaps_with_rows, output_schema = _materialize_heaps(
            planner,
            ordered,
            plan,
            batch_size=batch_size,
            materialize_shard=materialize_shard,
            materialized_cache_dir=state_root / _MATERIALIZED_DIRNAME,
        )
        _atomic_json(
            state_path,
            _state_payload(seed=seed, target=target, phase="writing", shards=ordered),
        )
        result = _write_selected(
            output,
            schema=output_schema,
            plan=plan,
            heaps=heaps_with_rows,
            batch_size=batch_size,
        )
        _atomic_json(
            state_path,
            _state_payload(seed=seed, target=target, phase="complete", shards=ordered),
        )
        return result
    finally:
        planner.close()


__all__ = [
    "FinalizedShard",
    "ResumableSamplingError",
    "select_v2_shards_resumable",
]
