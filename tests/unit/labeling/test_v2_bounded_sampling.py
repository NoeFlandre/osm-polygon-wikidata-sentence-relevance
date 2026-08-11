from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.labeling import v2_bounded_sampling
from osm_polygon_sentence_relevance.labeling.v2_bounded_sampling import (
    SamplingPlan,
    select_v2_parquet_bounded,
)
from osm_polygon_sentence_relevance.labeling.v2_sampling import select_v2_rows


def _table(rows: int = 48) -> pa.Table:
    values: list[dict[str, object]] = []
    buckets = (
        ("tiny", 0.05),
        ("small", 0.5),
        ("medium", 5.0),
        ("large", 20.0),
    )
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
    # A polygon's metadata must be stable. Assign its bucket from its first row.
    first: dict[str, tuple[str, float]] = {}
    for value in values:
        polygon_id = str(value["polygon_id"])
        first.setdefault(
            polygon_id,
            (str(value["area_bucket"]), float(value["area_km2"])),
        )
        value["area_bucket"], value["area_km2"] = first[polygon_id]
    return pa.Table.from_pylist(values).replace_schema_metadata({b"identity": b"test"})


@pytest.mark.parametrize("target", [1, 7, 31, 48, 80])
@pytest.mark.parametrize("seed", ["alpha", "beta"])
def test_bounded_parquet_selection_matches_in_memory_reference(
    tmp_path: Path, target: int, seed: str
) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "selected.parquet"
    pq.write_table(_table(), source, row_group_size=5)

    result = select_v2_parquet_bounded(
        source,
        output,
        target=target,
        seed=seed,
        scratch_dir=tmp_path / "scratch",
        batch_size=7,
    )

    expected = select_v2_rows(_table(), target=target, seed=seed)
    actual = pq.read_table(result)
    assert actual["sentence_id"].to_pylist() == expected["sentence_id"].to_pylist()
    assert actual.schema == expected.schema
    assert actual.schema.metadata == {b"identity": b"test"}
    assert not list((tmp_path / "scratch").glob("*.sqlite3"))


def test_bounded_selection_is_invariant_to_language_and_primary_tag_values(
    tmp_path: Path,
) -> None:
    table = _table()
    metadata_only = table.set_column(
        table.schema.get_field_index("language"),
        "language",
        pa.array(["same-language"] * table.num_rows),
    ).set_column(
        table.schema.get_field_index("osm_primary_tag"),
        "osm_primary_tag",
        pa.array(["same-tag"] * table.num_rows),
    )
    source = tmp_path / "source.parquet"
    changed_source = tmp_path / "changed-source.parquet"
    output = tmp_path / "selected.parquet"
    changed_output = tmp_path / "selected-changed.parquet"
    pq.write_table(table, source, row_group_size=5)
    pq.write_table(metadata_only, changed_source, row_group_size=5)

    select_v2_parquet_bounded(
        source,
        output,
        target=17,
        seed="seed",
        scratch_dir=tmp_path / "scratch",
        batch_size=7,
    )
    select_v2_parquet_bounded(
        changed_source,
        changed_output,
        target=17,
        seed="seed",
        scratch_dir=tmp_path / "scratch-changed",
        batch_size=7,
    )

    assert (
        pq.read_table(output)["sentence_id"].to_pylist()
        == pq.read_table(changed_output)["sentence_id"].to_pylist()
    )


def test_bounded_selection_discards_missing_coordinate_rows(
    tmp_path: Path,
) -> None:
    table = _table()
    table = table.set_column(
        table.schema.get_field_index("lat"),
        "lat",
        pa.array([None] + table["lat"].to_pylist()[1:], type=pa.float64()),
    )
    source = tmp_path / "source.parquet"
    output = tmp_path / "selected.parquet"
    pq.write_table(table, source, row_group_size=5)

    result = select_v2_parquet_bounded(
        source,
        output,
        target=table.num_rows,
        seed="seed",
        scratch_dir=tmp_path / "scratch",
        batch_size=7,
    )

    selected = pq.read_table(result)
    assert selected.num_rows == table.num_rows - 1
    assert "sentence-0000" not in selected["sentence_id"].to_pylist()


def test_bounded_selection_rejects_duplicate_ids_across_batches(
    tmp_path: Path,
) -> None:
    table = _table(12)
    ids = table["sentence_id"].to_pylist()
    ids[-1] = ids[0]
    source = tmp_path / "source.parquet"
    pq.write_table(table.set_column(0, "sentence_id", pa.array(ids)), source)

    with pytest.raises(ValueError, match="duplicate sentence IDs"):
        select_v2_parquet_bounded(
            source,
            tmp_path / "selected.parquet",
            target=5,
            seed="seed",
            scratch_dir=tmp_path / "scratch",
            batch_size=4,
        )


def test_bounded_selection_refuses_symlink_output(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(_table(8), source)
    target = tmp_path / "target"
    target.write_text("do not replace", encoding="utf-8")
    output = tmp_path / "selected.parquet"
    output.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        select_v2_parquet_bounded(
            source,
            output,
            target=4,
            seed="seed",
            scratch_dir=tmp_path / "scratch",
        )
    assert target.read_text(encoding="utf-8") == "do not replace"


@pytest.mark.parametrize("target", [0, -1, True])
def test_bounded_selection_rejects_invalid_target(
    tmp_path: Path, target: object
) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(_table(8), source)
    with pytest.raises(ValueError, match="target"):
        select_v2_parquet_bounded(
            source,
            tmp_path / "selected.parquet",
            target=target,  # type: ignore[arg-type]
            seed="seed",
            scratch_dir=tmp_path / "scratch",
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("sentence_id", "", "sentence_id must be non-empty"),
        ("polygon_id", "", "polygon_id must be non-empty"),
    ],
)
def test_bounded_selection_rejects_blank_identifiers(
    tmp_path: Path, column: str, value: str, message: str
) -> None:
    source = tmp_path / "source.parquet"
    table = _table(4)
    index = table.schema.get_field_index(column)
    values = table[column].to_pylist()
    values[0] = value
    pq.write_table(table.set_column(index, column, pa.array(values)), source)

    with pytest.raises(ValueError, match=message):
        select_v2_parquet_bounded(
            source,
            tmp_path / "selected.parquet",
            target=2,
            seed="seed",
            scratch_dir=tmp_path / "scratch",
        )


def test_bounded_selection_rejects_missing_columns(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(_table(4).drop(["language"]), source)

    with pytest.raises(ValueError, match="missing required columns.*language"):
        select_v2_parquet_bounded(
            source,
            tmp_path / "selected.parquet",
            target=2,
            seed="seed",
            scratch_dir=tmp_path / "scratch",
        )


def test_bounded_selection_rejects_inconsistent_polygon_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    table = _table(14)
    latitudes = table["lat"].to_pylist()
    latitudes[-1] = 80.0
    pq.write_table(
        table.set_column(
            table.schema.get_field_index("lat"), "lat", pa.array(latitudes)
        ),
        source,
    )

    with pytest.raises(ValueError, match="polygon metadata is inconsistent"):
        select_v2_parquet_bounded(
            source,
            tmp_path / "selected.parquet",
            target=2,
            seed="seed",
            scratch_dir=tmp_path / "scratch",
        )


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_bounded_selection_rejects_invalid_batch_size(
    tmp_path: Path, batch_size: object
) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(_table(4), source)

    with pytest.raises(ValueError, match="batch_size"):
        select_v2_parquet_bounded(
            source,
            tmp_path / "selected.parquet",
            target=2,
            seed="seed",
            scratch_dir=tmp_path / "scratch",
            batch_size=batch_size,  # type: ignore[arg-type]
        )


def test_bounded_selection_requires_fresh_regular_paths(tmp_path: Path) -> None:
    output = tmp_path / "selected.parquet"
    output.write_bytes(b"existing")
    with pytest.raises(ValueError, match="regular file"):
        select_v2_parquet_bounded(
            tmp_path / "missing.parquet",
            output,
            target=2,
            seed="seed",
            scratch_dir=tmp_path / "scratch",
        )

    source = tmp_path / "source.parquet"
    pq.write_table(_table(4), source)
    with pytest.raises(ValueError, match="fresh"):
        select_v2_parquet_bounded(
            source,
            output,
            target=2,
            seed="seed",
            scratch_dir=tmp_path / "scratch",
        )


class _Batches:
    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        self._batches = batches

    def iter_batches(self, *, batch_size: int) -> list[pa.RecordBatch]:
        assert batch_size > 0
        return self._batches


def test_candidate_scan_detects_source_change_and_unfulfillable_plan() -> None:
    batch = _table(2).to_batches()[0]
    metadata = {
        str(row["polygon_id"]): (
            str(row["area_bucket"]),
            v2_bounded_sampling._cell(row["lat"], row["lon"]),
        )
        for row in batch.to_pylist()
    }
    polygon_order = {polygon_id: index for index, polygon_id in enumerate(metadata)}
    stratum = next(iter(metadata.values()))[1]

    changed = SamplingPlan(polygon_order, metadata, (), {}, total_rows=3)
    with pytest.raises(ValueError, match="changed between sampling passes"):
        v2_bounded_sampling._retain_candidates(
            _Batches([batch]),
            changed,
            seed="seed",
            batch_size=2,  # type: ignore[arg-type]
        )

    impossible = SamplingPlan(
        polygon_order, metadata, (stratum, stratum, stratum), {stratum: 3}, total_rows=2
    )
    with pytest.raises(ValueError, match="could not be fulfilled"):
        v2_bounded_sampling._retain_candidates(
            _Batches([batch]),
            impossible,
            seed="seed",
            batch_size=2,  # type: ignore[arg-type]
        )


def test_write_failure_removes_temporary_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "selected.parquet"
    pq.write_table(_table(4), source)

    class _FailingWriter:
        def __init__(self, path: Path, schema: pa.Schema, **_: Any) -> None:
            del schema
            self.path = Path(path)

        def write_table(self, table: pa.Table) -> None:
            del table
            raise OSError("synthetic write failure")

        def close(self) -> None:
            pass

    monkeypatch.setattr(v2_bounded_sampling.pq, "ParquetWriter", _FailingWriter)
    with pytest.raises(OSError, match="synthetic write failure"):
        select_v2_parquet_bounded(
            source,
            output,
            target=2,
            seed="seed",
            scratch_dir=tmp_path / "scratch",
            batch_size=1,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".selected.parquet.*"))
