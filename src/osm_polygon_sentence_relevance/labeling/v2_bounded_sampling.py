"""Bounded exact V2 sampling from an enriched Parquet stream.

The in-memory :func:`v2_sampling.select_v2_rows` function remains the
reference contract. This module reproduces its exact sentence order while
holding only polygon metadata, stratum counts, and at most ``target`` rows in
memory. The only stratum is the H3 cell; language and OSM primary tag columns
are retained metadata. Sentence-ID uniqueness is enforced with a temporary
SQLite index.
"""

from __future__ import annotations

import heapq
import os
import sqlite3
import tempfile
from collections import Counter, deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .v2_sampling import (
    _cell,
    _rank,
    _weighted_schedule,
    canonical_area_bucket,
    ordered_polygon_ids,
)

_REQUIRED = frozenset(
    {
        "sentence_id",
        "polygon_id",
        "lat",
        "lon",
        "area_km2",
        "area_bucket",
        "language",
        "osm_primary_tag",
    }
)
_Stratum = str
_Candidate = tuple[int, int, int, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    """The bounded global information needed by the selection scan."""

    polygon_order: dict[str, int]
    polygon_metadata: dict[str, tuple[str, str]]
    schedule: tuple[_Stratum, ...]
    quotas: dict[_Stratum, int]
    total_rows: int


class _Planner:
    def __init__(self, *, target: int, seed: str, scratch_dir: Path) -> None:
        self.target = target
        self.seed = seed
        self.polygon_metadata: dict[str, tuple[str, str]] = {}
        self.stratum_sizes: dict[_Stratum, int] = {}
        descriptor, name = tempfile.mkstemp(
            prefix="v2-sampling-", suffix=".sqlite3", dir=scratch_dir
        )
        os.close(descriptor)
        self.database = Path(name)
        self.connection = sqlite3.connect(self.database)
        self.connection.execute(
            "CREATE TABLE sentence_ids (value TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        self.total_rows = 0

    def close(self) -> None:
        self.connection.close()
        self.database.unlink(missing_ok=True)

    def observe(self, batch: pa.RecordBatch) -> None:
        missing = _REQUIRED.difference(batch.schema.names)
        if missing:
            raise ValueError(
                f"V2 sampling input is missing required columns: {sorted(missing)}"
            )
        rows = batch.to_pylist()
        identifiers: list[tuple[str]] = []
        counts: Counter[_Stratum] = Counter()
        for row in rows:
            sentence_id = row["sentence_id"]
            polygon_id = row["polygon_id"]
            if not isinstance(sentence_id, str) or not sentence_id:
                raise ValueError("sentence_id must be non-empty")
            if not isinstance(polygon_id, str) or not polygon_id:
                raise ValueError("polygon_id must be non-empty")
            identifiers.append((sentence_id,))
            metadata = (
                canonical_area_bucket(row["area_km2"], row["area_bucket"]),
                _cell(row["lat"], row["lon"]),
            )
            prior = self.polygon_metadata.get(polygon_id)
            if prior is not None and prior != metadata:
                raise ValueError("polygon metadata is inconsistent")
            self.polygon_metadata.setdefault(polygon_id, metadata)
            counts[metadata[1]] += 1
        try:
            self.connection.executemany(
                "INSERT INTO sentence_ids(value) VALUES (?)", identifiers
            )
            self.connection.commit()
        except sqlite3.IntegrityError as exc:
            self.connection.rollback()
            raise ValueError("sampling input contains duplicate sentence IDs") from exc
        for stratum, count in counts.items():
            self.stratum_sizes[stratum] = self.stratum_sizes.get(stratum, 0) + count
        self.total_rows += len(rows)

    def build(self) -> SamplingPlan:
        ordered = ordered_polygon_ids(self.polygon_metadata, seed=self.seed)
        schedule = _weighted_schedule(
            self.stratum_sizes,
            rank=lambda key: _rank(self.seed, key),
            limit=min(self.target, self.total_rows),
        )
        return SamplingPlan(
            polygon_order={value: index for index, value in enumerate(ordered)},
            polygon_metadata=dict(self.polygon_metadata),
            schedule=tuple(schedule),
            quotas=dict(Counter(schedule)),
            total_rows=self.total_rows,
        )


def _validate_inputs(
    source_path: Path,
    output_path: Path,
    *,
    target: int,
    batch_size: int,
) -> None:
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("V2 source must be a regular file")
    if output_path.is_symlink():
        raise ValueError("V2 output must not be a symlink")
    if output_path.exists():
        raise ValueError("V2 output must be fresh")
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("target must be a positive integer")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")


def _retain_candidates(
    parquet: pq.ParquetFile,
    plan: SamplingPlan,
    *,
    seed: str,
    batch_size: int,
) -> dict[_Stratum, list[_Candidate]]:
    heaps: dict[_Stratum, list[_Candidate]] = {stratum: [] for stratum in plan.quotas}
    global_index = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            polygon_id = str(row["polygon_id"])
            metadata = plan.polygon_metadata[polygon_id]
            stratum = metadata[1]
            quota = plan.quotas.get(stratum, 0)
            if quota:
                candidate: _Candidate = (
                    -plan.polygon_order[polygon_id],
                    -int(_rank(seed, str(row["sentence_id"])), 16),
                    -global_index,
                    row,
                )
                heap = heaps[stratum]
                if len(heap) < quota:
                    heapq.heappush(heap, candidate)
                elif candidate[:3] > heap[0][:3]:
                    heapq.heapreplace(heap, candidate)
            global_index += 1
    if global_index != plan.total_rows:
        raise ValueError("V2 source changed between sampling passes")
    if any(len(heaps[key]) != quota for key, quota in plan.quotas.items()):
        raise ValueError("V2 sampling plan could not be fulfilled")
    return heaps


def _write_selected(
    output_path: Path,
    *,
    schema: pa.Schema,
    plan: SamplingPlan,
    heaps: dict[_Stratum, list[_Candidate]],
    batch_size: int,
) -> Path:
    queues = {
        stratum: deque(
            item[3]
            for item in sorted(
                heap,
                key=lambda item: (-item[0], -item[1], -item[2]),
            )
        )
        for stratum, heap in heaps.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(raw)
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    pending: list[dict[str, Any]] = []
    try:
        for stratum in plan.schedule:
            pending.append(queues[stratum].popleft())
            if len(pending) == batch_size:
                writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                pending.clear()
        if pending:
            writer.write_table(pa.Table.from_pylist(pending, schema=schema))
        writer.close()
        os.chmod(temporary, 0o600)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
        directory = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return output_path
    except BaseException:
        with suppress(Exception):
            writer.close()
        temporary.unlink(missing_ok=True)
        raise


def select_v2_parquet_bounded(
    source_path: Path,
    output_path: Path,
    *,
    target: int,
    seed: str,
    scratch_dir: Path,
    batch_size: int = 65_536,
) -> Path:
    """Select the exact V2 prefix without loading the full source table."""

    source = Path(source_path)
    output = Path(output_path)
    scratch = Path(scratch_dir)
    _validate_inputs(source, output, target=target, batch_size=batch_size)
    scratch.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(source)
    planner = _Planner(target=target, seed=seed, scratch_dir=scratch)
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            planner.observe(batch)
        plan = planner.build()
        heaps = _retain_candidates(
            parquet,
            plan,
            seed=seed,
            batch_size=batch_size,
        )
        return _write_selected(
            output,
            schema=parquet.schema_arrow,
            plan=plan,
            heaps=heaps,
            batch_size=batch_size,
        )
    finally:
        planner.close()


__all__ = ["SamplingPlan", "select_v2_parquet_bounded"]
