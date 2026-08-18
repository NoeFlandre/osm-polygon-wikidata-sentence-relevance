"""Geometry-only augmentation for an already finalized V2 table.

The V2 label table is immutable evidence: this module never resamples rows or
touches model scores.  It projects the geometry already stored in the pinned
upstream ``polygons/{region}.parquet`` shards onto the existing rows by
``polygon_id`` and writes the result atomically.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .v2_input import _polygon_shard_key, _regular_file

GEOMETRY_COLUMN = "geometry"
_SOURCE_REQUIRED = frozenset({"sentence_id", "polygon_id"})
_DOWNLOAD_REQUIRED = frozenset({"sentence_id", "polygon_id", "region"})
_GEOMETRY_REQUIRED = frozenset({"polygon_id", GEOMETRY_COLUMN})
_RANGE_BLOCK_SIZE = 4 * 1024 * 1024
_DEFAULT_BATCH_SIZE = 8192
_WRITE_BATCH_SIZE = 256
_MAX_ROWS_PER_PAGE = 256


def _select_polygon_rows(table: pa.Table, polygon_ids: set[str]) -> pa.Table:
    """Select requested polygons while preserving the source shard order."""

    # ``pyarrow.compute.is_in`` is available at runtime but missing from the
    # installed type definitions, so keep the compatibility cast local.
    is_in = cast(Callable[..., pa.Array], pc.__dict__["is_in"])
    return table.filter(
        is_in(
            table["polygon_id"],
            value_set=pa.array(sorted(polygon_ids)),
        )
    )


def _validate_geometry(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("geometry must be non-empty")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("geometry must be valid GeoJSON") from exc
    if not isinstance(parsed, dict) or parsed.get("type") not in {
        "Polygon",
        "MultiPolygon",
    }:
        raise ValueError("geometry must be a Polygon or MultiPolygon GeoJSON object")
    if "coordinates" not in parsed:
        raise ValueError("geometry GeoJSON is missing coordinates")
    return value


def _geometry_by_polygon(
    metadata_tables: Mapping[str, pa.Table],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for region in sorted(metadata_tables):
        table = metadata_tables[region]
        missing = _GEOMETRY_REQUIRED.difference(table.column_names)
        if missing:
            raise ValueError(
                f"polygon geometry for {region} is missing columns: {sorted(missing)}"
            )
        for row in table.select(sorted(_GEOMETRY_REQUIRED)).to_pylist():
            polygon_id = row["polygon_id"]
            if not isinstance(polygon_id, str) or not polygon_id:
                raise ValueError("polygon geometry polygon_id must be non-empty")
            if polygon_id in result:
                raise ValueError("duplicate polygon geometry")
            result[polygon_id] = _validate_geometry(row[GEOMETRY_COLUMN])
    return result


def add_v2_geometry(
    source: pa.Table,
    metadata_tables: Mapping[str, pa.Table],
) -> pa.Table:
    """Append geometry to an existing V2 table using a strict polygon join.

    Existing columns and row order are preserved byte-for-value.  Repeating
    the operation is idempotent when an existing geometry column matches the
    pinned source; a disagreement fails closed instead of silently replacing
    public data.
    """

    missing = _SOURCE_REQUIRED.difference(source.column_names)
    if missing:
        raise ValueError(f"V2 table is missing columns: {sorted(missing)}")
    sentence_ids = source["sentence_id"].to_pylist()
    if len(sentence_ids) != len(set(sentence_ids)):
        raise ValueError("V2 table contains duplicate sentence IDs")
    geometry_by_polygon = _geometry_by_polygon(metadata_tables)
    polygon_ids = source["polygon_id"].to_pylist()
    geometry_values: list[str] = []
    for polygon_id in polygon_ids:
        if polygon_id not in geometry_by_polygon:
            raise ValueError(f"missing polygon geometry for {polygon_id}")
        geometry_values.append(geometry_by_polygon[polygon_id])

    if GEOMETRY_COLUMN in source.column_names:
        existing = source[GEOMETRY_COLUMN].to_pylist()
        for index, (old, new) in enumerate(zip(existing, geometry_values, strict=True)):
            if not isinstance(old, str) or not old.strip():
                raise ValueError(f"existing geometry is empty at row {index}")
            if old != new:
                raise ValueError(f"existing geometry differs at row {index}")
        return source
    return source.append_column(
        GEOMETRY_COLUMN, pa.array(geometry_values, type=pa.string())
    )


def add_v2_geometry_from_paths(
    source_path: Path,
    output_path: Path,
    metadata_paths: Mapping[str, Path],
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> Path:
    """Read local geometry shards and stream an augmented table atomically."""

    source_path = Path(source_path)
    output_path = Path(output_path)
    _regular_file(source_path, "V2 table")
    if output_path.is_symlink():
        raise ValueError("V2 geometry output must not be a symlink")
    _validate_batch_size(batch_size)
    geometry_by_polygon: dict[str, str] = {}
    for region in sorted(metadata_paths):
        path = Path(metadata_paths[region])
        _regular_file(path, f"polygon geometry for {region}")
        table = pq.read_table(path, columns=sorted(_GEOMETRY_REQUIRED))
        region_geometry = _geometry_by_polygon({region: table})
        for polygon_id, geometry in region_geometry.items():
            if polygon_id in geometry_by_polygon:
                raise ValueError("duplicate polygon geometry")
            geometry_by_polygon[polygon_id] = geometry
    return _stream_augmented_file(
        source_path,
        output_path,
        geometry_by_polygon,
        batch_size=batch_size,
    )


def _default_filesystem() -> Any:
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as exc:  # pragma: no cover - optional hub boundary
        raise RuntimeError(
            "install the hub extra to retrieve polygon geometry"
        ) from exc
    return HfFileSystem()


def _read_geometry_shard(filesystem: Any, path: str) -> pa.Table:
    with filesystem.open(path, "rb", block_size=_RANGE_BLOCK_SIZE) as stream:
        return pq.read_table(stream, columns=sorted(_GEOMETRY_REQUIRED))


def _validate_batch_size(batch_size: int) -> None:
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")


def _stream_augmented_file(
    source_path: Path,
    output_path: Path,
    geometry_by_polygon: Mapping[str, str],
    *,
    batch_size: int,
) -> Path:
    """Write one geometry column with bounded memory and compressed pages."""

    _validate_batch_size(batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_symlink():
        raise ValueError("V2 geometry output must not be a symlink")
    source = pq.ParquetFile(source_path)

    # Re-validating an already augmented file should not expand a dictionary
    # encoded geometry column into many copies or rewrite it unnecessarily.
    if GEOMETRY_COLUMN in source.schema.names:
        seen_sentence_ids: set[str] = set()
        for batch in source.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            sentence_ids = table["sentence_id"].to_pylist()
            if seen_sentence_ids.intersection(sentence_ids):
                raise ValueError("V2 table contains duplicate sentence IDs")
            seen_sentence_ids.update(sentence_ids)
            for index, (polygon_id, existing) in enumerate(
                zip(
                    table["polygon_id"].to_pylist(),
                    table[GEOMETRY_COLUMN].to_pylist(),
                    strict=True,
                )
            ):
                expected = geometry_by_polygon.get(polygon_id)
                if expected is None:
                    raise ValueError(f"missing polygon geometry for {polygon_id}")
                if not isinstance(existing, str) or not existing.strip():
                    raise ValueError(f"existing geometry is empty at row {index}")
                if existing != expected:
                    raise ValueError(f"existing geometry differs at row {index}")
        if source_path.resolve() == output_path.resolve():
            return output_path
        _copy_atomic(source_path, output_path)
        return output_path

    fd, raw = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    os.close(fd)
    temporary = Path(raw)
    writer: pq.ParquetWriter | None = None
    seen_sentence_ids: set[str] = set()
    rows_written = 0
    try:
        read_batch_size = min(batch_size, _WRITE_BATCH_SIZE)
        for batch in source.iter_batches(batch_size=read_batch_size):
            table = pa.Table.from_batches([batch])
            sentence_ids = table["sentence_id"].to_pylist()
            if seen_sentence_ids.intersection(sentence_ids):
                raise ValueError("V2 table contains duplicate sentence IDs")
            seen_sentence_ids.update(sentence_ids)
            polygon_ids = table["polygon_id"].to_pylist()
            geometry_values: list[str] = []
            for polygon_id in polygon_ids:
                if polygon_id not in geometry_by_polygon:
                    raise ValueError(f"missing polygon geometry for {polygon_id}")
                geometry_values.append(geometry_by_polygon[polygon_id])
            augmented = table.append_column(
                GEOMETRY_COLUMN, pa.array(geometry_values, type=pa.string())
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    augmented.schema,
                    compression="zstd",
                    compression_level=19,
                    use_dictionary=False,
                    write_batch_size=64,
                    max_rows_per_page=_MAX_ROWS_PER_PAGE,
                )
            writer.write_table(augmented, row_group_size=augmented.num_rows)
            rows_written += augmented.num_rows
        if writer is None or rows_written == 0:
            raise ValueError("V2 table must contain at least one row")
        writer.close()
        os.replace(temporary, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def _copy_atomic(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    os.close(fd)
    temporary = Path(raw)
    try:
        shutil.copyfile(source_path, temporary)
        os.replace(temporary, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def download_and_add_v2_geometry(
    source_path: Path,
    output_path: Path,
    *,
    dataset_id: str,
    revision: str,
    filesystem: Any | None = None,
    max_workers: int = 8,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> Path:
    """Range-read needed upstream geometry columns at a pinned revision.

    Shards are fetched with bounded concurrency, while their results are
    assembled in sorted region order for deterministic validation and output.
    """

    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be non-empty")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be non-empty")
    if (
        not isinstance(max_workers, int)
        or isinstance(max_workers, bool)
        or max_workers < 1
    ):
        raise ValueError("max_workers must be a positive integer")
    _validate_batch_size(batch_size)
    source_path = Path(source_path)
    output_path = Path(output_path)
    _regular_file(source_path, "V2 table")
    source = pq.read_table(source_path, columns=sorted(_DOWNLOAD_REQUIRED))
    missing = _DOWNLOAD_REQUIRED.difference(source.column_names)
    if missing:
        raise ValueError(f"V2 table is missing columns: {sorted(missing)}")
    supplied_filesystem = filesystem
    needed_by_region: dict[str, set[str]] = {}
    for row in source.select(["polygon_id", "region"]).to_pylist():
        region = _polygon_shard_key(str(row["region"]))
        needed_by_region.setdefault(region, set()).add(row["polygon_id"])
    del source
    regions = sorted(needed_by_region)

    def read_region(region: str) -> tuple[str, dict[str, str]]:
        path = f"datasets/{dataset_id}@{revision}/polygons/{region}.parquet"
        # HfFileSystem owns an HTTP session and is not safe to share between
        # concurrent range readers.  Keep injected test doubles shareable,
        # but give each worker its own real client.
        reader = supplied_filesystem or _default_filesystem()
        table = _read_geometry_shard(reader, path)
        selected = _select_polygon_rows(table, needed_by_region[region])
        expected = needed_by_region[region]
        found = set(selected["polygon_id"].to_pylist())
        if found != expected:
            missing = sorted(expected - found)
            raise ValueError(f"{region} is missing selected polygons: {missing[:5]}")
        result = _geometry_by_polygon({region: selected})
        del table, selected
        return region, result

    with ThreadPoolExecutor(max_workers=min(max_workers, len(regions) or 1)) as pool:
        metadata = dict(pool.map(read_region, regions))
    geometry_by_polygon: dict[str, str] = {}
    for region in regions:
        for polygon_id, geometry in metadata[region].items():
            if polygon_id in geometry_by_polygon:
                raise ValueError("duplicate polygon geometry")
            geometry_by_polygon[polygon_id] = geometry
    del metadata
    return _stream_augmented_file(
        source_path,
        output_path,
        geometry_by_polygon,
        batch_size=batch_size,
    )


__all__ = [
    "GEOMETRY_COLUMN",
    "add_v2_geometry",
    "add_v2_geometry_from_paths",
    "download_and_add_v2_geometry",
]
