"""Prepare the V2 sampling input from split sentences and polygon metadata.

The historical V1 sentence output intentionally has a stable public schema
without polygon-area fields. V2 therefore enriches that output in a separate,
atomic file after splitting. Only the six area/identity columns from each
upstream ``polygons/{region}.parquet`` file are read, and the source revision
is pinned by the caller.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .v2_sampling import canonical_area_bucket

_SOURCE_REQUIRED = frozenset({"sentence_id", "polygon_id", "region"})
_METADATA_REQUIRED = frozenset({"polygon_id", "area_km2", "area_bucket"})


def hf_hub_download(**kwargs: Any) -> str:
    """Lazy wrapper kept injectable for deterministic tests and dry runs."""

    try:
        from huggingface_hub import hf_hub_download as download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the hub extra to prepare V2 input") from exc
    return str(download(**kwargs))


def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")


def _metadata_by_polygon(
    metadata_tables: Mapping[str, pa.Table],
) -> dict[str, tuple[float, str]]:
    result: dict[str, tuple[float, str]] = {}
    for region in sorted(metadata_tables):
        table = metadata_tables[region]
        missing = _METADATA_REQUIRED.difference(table.column_names)
        if missing:
            raise ValueError(
                f"polygon metadata for {region} is missing columns: {sorted(missing)}"
            )
        for row in table.select(sorted(_METADATA_REQUIRED)).to_pylist():
            polygon_id = row["polygon_id"]
            if not isinstance(polygon_id, str) or not polygon_id:
                raise ValueError("polygon metadata polygon_id must be non-empty")
            if polygon_id in result:
                raise ValueError("duplicate polygon metadata")
            area = row["area_km2"]
            bucket = canonical_area_bucket(area, row["area_bucket"])
            result[polygon_id] = (float(area), bucket)
    return result


def enrich_v2_table(
    source: pa.Table,
    metadata_tables: Mapping[str, pa.Table],
) -> pa.Table:
    """Append canonical area fields to a split sentence table.

    All source columns and row order are preserved. Every source polygon must
    have exactly one metadata row; this makes missing or duplicate upstream
    joins fail before a remote labeling job can consume them.
    """

    missing = _SOURCE_REQUIRED.difference(source.column_names)
    if missing:
        raise ValueError(f"V2 split input is missing columns: {sorted(missing)}")
    sentence_ids = source["sentence_id"].to_pylist()
    if len(sentence_ids) != len(set(sentence_ids)):
        raise ValueError("V2 split input contains duplicate sentence IDs")
    metadata = _metadata_by_polygon(metadata_tables)
    rows = source.select(["polygon_id", "region"]).to_pylist()
    area_values: list[float] = []
    bucket_values: list[str] = []
    for row in rows:
        polygon_id = row["polygon_id"]
        if polygon_id not in metadata:
            raise ValueError(f"missing polygon metadata for {polygon_id}")
        area, bucket = metadata[polygon_id]
        area_values.append(area)
        bucket_values.append(bucket)
    result = source
    for name, values, dtype in (
        ("area_km2", area_values, pa.float64()),
        ("area_bucket", bucket_values, pa.string()),
    ):
        if name in result.column_names:
            result = result.set_column(
                result.column_names.index(name), name, pa.array(values, type=dtype)
            )
        else:
            result = result.append_column(name, pa.array(values, type=dtype))
    return result


def enrich_v2_input(
    source_path: Path,
    output_path: Path,
    *,
    metadata_paths: Mapping[str, Path],
) -> Path:
    """Read local metadata files and atomically write the V2 input parquet."""

    source_path = Path(source_path)
    output_path = Path(output_path)
    _regular_file(source_path, "V2 split input")
    if output_path.is_symlink():
        raise ValueError("V2 output must not be a symlink")
    source = pq.read_table(source_path)
    tables: dict[str, pa.Table] = {}
    for region in sorted(metadata_paths):
        path = Path(metadata_paths[region])
        _regular_file(path, f"polygon metadata for {region}")
        tables[region] = pq.read_table(path, columns=sorted(_METADATA_REQUIRED))
    result = enrich_v2_table(source, tables)
    return _write_table_atomically(result, output_path)


def _download_polygon_metadata(
    source: pa.Table,
    *,
    dataset_id: str,
    revision: str,
    cache_dir: Path,
) -> dict[str, pa.Table]:
    """Download one pinned polygon file per source region in sorted order."""

    regions = sorted({str(value) for value in source["region"].to_pylist()})
    return {
        region: download_v2_polygon_metadata(
            dataset_id=dataset_id,
            revision=revision,
            shard_key=region,
            cache_dir=cache_dir,
        )
        for region in regions
    }


def download_v2_polygon_metadata(
    *,
    dataset_id: str,
    revision: str,
    shard_key: str,
    cache_dir: Path,
) -> pa.Table:
    """Read one shard's area metadata from an immutable upstream revision."""

    path = hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        filename=f"polygons/{shard_key}.parquet",
        cache_dir=cache_dir,
    )
    return pq.read_table(Path(path), columns=sorted(_METADATA_REQUIRED))


def download_and_enrich_v2_input(
    source_path: Path,
    output_path: Path,
    *,
    dataset_id: str,
    revision: str,
    cache_dir: Path,
) -> Path:
    """Enrich a split output using pinned upstream polygon files."""

    _regular_file(Path(source_path), "V2 split input")
    source = pq.read_table(source_path)
    tables = _download_polygon_metadata(
        source,
        dataset_id=dataset_id,
        revision=revision,
        cache_dir=Path(cache_dir),
    )
    result = enrich_v2_table(source, tables)
    return _write_table_atomically(result, Path(output_path))


def _write_table_atomically(table: pa.Table, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_symlink():
        raise ValueError("V2 output must not be a symlink")
    fd, raw = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temporary = Path(raw)
    try:
        os.fchmod(fd, 0o600)
        os.close(fd)
        pq.write_table(table, temporary, compression="zstd")
        os.chmod(temporary, 0o600)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args(argv)
    source = Path(args.source)
    output = Path(args.output)
    source_table = pq.read_table(source)
    tables = _download_polygon_metadata(
        source_table,
        dataset_id=args.dataset_id,
        revision=args.revision,
        cache_dir=Path(args.cache_dir),
    )
    _write_table_atomically(enrich_v2_table(source_table, tables), output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "download_v2_polygon_metadata",
    "download_and_enrich_v2_input",
    "enrich_v2_input",
    "enrich_v2_table",
]
