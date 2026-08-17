from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.labeling.v2_geometry import (
    add_v2_geometry,
    add_v2_geometry_from_paths,
    download_and_add_v2_geometry,
)


def _source() -> pa.Table:
    return pa.table(
        {
            "sentence_id": ["s1", "s2", "s3"],
            "polygon_id": ["p2", "p1", "p2"],
            "region": ["beta-latest", "alpha-latest", "beta-latest"],
            "place_relevance": ["yes", "no", "yes"],
            "yes_logprob": [-0.1, -1.0, -0.2],
        }
    )


def _metadata(
    *,
    p1: str = '{"type":"Polygon","coordinates":[]}',
    p2: str = '{"type":"MultiPolygon","coordinates":[]}',
) -> dict[str, pa.Table]:
    return {
        "alpha-latest": pa.table({"polygon_id": ["p1"], "geometry": [p1]}),
        "beta-latest": pa.table({"polygon_id": ["p2"], "geometry": [p2]}),
    }


def test_add_v2_geometry_preserves_rows_and_appends_keyed_geometry() -> None:
    source = _source()

    result = add_v2_geometry(source, _metadata())

    assert result.column_names == [*source.column_names, "geometry"]
    assert result.drop_columns(["geometry"]).equals(source)
    assert result["geometry"].to_pylist() == [
        '{"type":"MultiPolygon","coordinates":[]}',
        '{"type":"Polygon","coordinates":[]}',
        '{"type":"MultiPolygon","coordinates":[]}',
    ]


def test_add_v2_geometry_rejects_missing_duplicate_and_blank_metadata() -> None:
    with pytest.raises(ValueError, match="missing polygon geometry"):
        add_v2_geometry(
            _source(),
            {"alpha-latest": _metadata()["alpha-latest"]},
        )

    duplicate = pa.table(
        {
            "polygon_id": ["p2", "p2"],
            "geometry": [
                '{"type":"Polygon","coordinates":[]}',
                '{"type":"Polygon","coordinates":[]}',
            ],
        }
    )
    with pytest.raises(ValueError, match="duplicate polygon geometry"):
        add_v2_geometry(
            _source(),
            {"alpha-latest": _metadata()["alpha-latest"], "beta-latest": duplicate},
        )

    with pytest.raises(ValueError, match="geometry must be non-empty"):
        add_v2_geometry(
            _source(),
            {
                "alpha-latest": _metadata()["alpha-latest"],
                "beta-latest": pa.table({"polygon_id": ["p2"], "geometry": [None]}),
            },
        )


def test_add_v2_geometry_rejects_missing_source_columns_and_existing_mismatch() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        add_v2_geometry(pa.table({"sentence_id": ["s1"]}), _metadata())

    existing = _source().append_column(
        "geometry",
        pa.array(
            [
                '{"type":"MultiPolygon","coordinates":[]}',
                '{"type":"Polygon","coordinates":[]}',
                '{"type":"MultiPolygon","coordinates":[]}',
            ]
        ),
    )
    assert add_v2_geometry(existing, _metadata()).equals(existing)

    mismatched = existing.set_column(
        existing.column_names.index("geometry"),
        "geometry",
        pa.array(
            ["wrong", existing["geometry"][1].as_py(), existing["geometry"][2].as_py()]
        ),
    )
    with pytest.raises(ValueError, match="existing geometry differs"):
        add_v2_geometry(mismatched, _metadata())


def test_add_v2_geometry_from_paths_writes_atomically(tmp_path: Path) -> None:
    source_path = tmp_path / "source.parquet"
    output_path = tmp_path / "nested" / "augmented.parquet"
    pq.write_table(_source(), source_path)
    metadata_paths: dict[str, Path] = {}
    for region, table in _metadata().items():
        path = tmp_path / f"{region}.parquet"
        pq.write_table(table, path)
        metadata_paths[region] = path

    result = add_v2_geometry_from_paths(
        source_path, output_path, metadata_paths, batch_size=8192
    )

    assert result == output_path
    assert (
        pq.read_table(output_path)["geometry"]
        .to_pylist()[0]
        .startswith('{"type":"MultiPolygon"')
    )
    assert pq.ParquetFile(output_path).metadata.num_row_groups == 1
    assert not list(output_path.parent.glob(".augmented.parquet.*"))


class _FakeFileSystem:
    def __init__(self, shards: dict[str, bytes]) -> None:
        self.shards = shards
        self.calls: list[tuple[str, str, int | None]] = []

    def open(self, path: str, mode: str = "rb", *, block_size: int | None = None):
        self.calls.append((path, mode, block_size))
        return io.BytesIO(self.shards[path])


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def test_download_and_add_v2_geometry_uses_sorted_pinned_range_reads(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.parquet"
    output_path = tmp_path / "output.parquet"
    pq.write_table(_source(), source_path)
    shards = {
        "datasets/owner/input@rev/polygons/alpha-latest.parquet": _parquet_bytes(
            pa.table(
                {
                    "polygon_id": ["p1"],
                    "geometry": ['{"type":"Polygon","coordinates":[]}'],
                    "ignored": ["not read by the projection"],
                }
            )
        ),
        "datasets/owner/input@rev/polygons/beta-latest.parquet": _parquet_bytes(
            pa.table(
                {
                    "polygon_id": ["p2"],
                    "geometry": ['{"type":"MultiPolygon","coordinates":[]}'],
                    "ignored": ["not read by the projection"],
                }
            )
        ),
    }
    filesystem = _FakeFileSystem(shards)

    result = download_and_add_v2_geometry(
        source_path,
        output_path,
        dataset_id="owner/input",
        revision="rev",
        filesystem=filesystem,
        max_workers=1,
    )

    assert result == output_path
    assert [call[0] for call in filesystem.calls] == [
        "datasets/owner/input@rev/polygons/alpha-latest.parquet",
        "datasets/owner/input@rev/polygons/beta-latest.parquet",
    ]
    assert all(call[1] == "rb" for call in filesystem.calls)
    assert all(call[2] is not None and call[2] > 0 for call in filesystem.calls)
    assert pq.read_table(output_path)["geometry"].to_pylist() == [
        '{"type":"MultiPolygon","coordinates":[]}',
        '{"type":"Polygon","coordinates":[]}',
        '{"type":"MultiPolygon","coordinates":[]}',
    ]
