from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import osm_polygon_sentence_relevance.labeling.v2_input as v2_input
from osm_polygon_sentence_relevance.labeling.v2_input import (
    _write_table_atomically,
    download_v2_polygon_metadata,
    enrich_v2_input,
    enrich_v2_table,
)


def _source() -> pa.Table:
    return pa.table(
        {
            "sentence_id": ["s1", "s2"],
            "polygon_id": ["p1", "p2"],
            "region": ["alpha-latest", "beta-latest"],
            "sentence_text_raw": ["A", "B"],
            "area_bucket": [None, None],
        }
    ).drop_columns(["area_bucket"])


def _metadata(path: Path, rows: list[dict[str, object]]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_enrich_v2_table_adds_canonical_area_metadata_deterministically() -> None:
    result = enrich_v2_table(
        _source(),
        {
            "alpha-latest": pa.table(
                {
                    "polygon_id": ["p1"],
                    "area_km2": [0.00005],
                    "area_bucket": ["<100m2"],
                }
            ),
            "beta-latest": pa.table(
                {
                    "polygon_id": ["p2"],
                    "area_km2": [125.0],
                    "area_bucket": [">100km2"],
                }
            ),
        },
    )

    assert result.column_names == [
        "sentence_id",
        "polygon_id",
        "region",
        "sentence_text_raw",
        "area_km2",
        "area_bucket",
    ]
    assert result["area_km2"].to_pylist() == [0.00005, 125.0]
    assert result["area_bucket"].to_pylist() == ["tiny", "large"]


def test_enrich_v2_table_rejects_missing_polygon_metadata() -> None:
    with pytest.raises(ValueError, match="missing polygon metadata"):
        enrich_v2_table(
            _source(),
            {
                "alpha-latest": pa.table(
                    {
                        "polygon_id": ["p1"],
                        "area_km2": [0.00005],
                        "area_bucket": ["<100m2"],
                    }
                )
            },
        )


def test_enrich_v2_table_rejects_invalid_source_and_metadata_shapes() -> None:
    with pytest.raises(ValueError, match="split input is missing"):
        enrich_v2_table(pa.table({"sentence_id": ["s"]}), {})
    duplicate = _source().set_column(0, "sentence_id", pa.array(["same", "same"]))
    with pytest.raises(ValueError, match="duplicate sentence"):
        enrich_v2_table(duplicate, {})
    with pytest.raises(ValueError, match="missing columns"):
        enrich_v2_table(
            _source(),
            {"alpha-latest": pa.table({"polygon_id": ["p1"]})},
        )
    with pytest.raises(ValueError, match="non-empty"):
        enrich_v2_table(
            _source(),
            {
                "alpha-latest": pa.table(
                    {
                        "polygon_id": [""],
                        "area_km2": [0.05],
                        "area_bucket": ["10k_m2-100k_m2"],
                    }
                )
            },
        )


def test_enrich_v2_table_replaces_existing_area_columns() -> None:
    source = _source().append_column("area_km2", pa.array([99.0, 99.0]))
    source = source.append_column("area_bucket", pa.array(["large", "large"]))
    result = enrich_v2_table(
        source,
        {
            "alpha-latest": pa.table(
                {
                    "polygon_id": ["p1"],
                    "area_km2": [0.5],
                    "area_bucket": ["0.1-1km2"],
                }
            ),
            "beta-latest": pa.table(
                {
                    "polygon_id": ["p2"],
                    "area_km2": [5.0],
                    "area_bucket": ["1-10km2"],
                }
            ),
        },
    )
    assert result["area_km2"].to_pylist() == [0.5, 5.0]
    assert result["area_bucket"].to_pylist() == ["small", "medium"]


def test_enrich_v2_table_rejects_duplicate_or_inconsistent_polygon_metadata() -> None:
    metadata = pa.table(
        {
            "polygon_id": ["p1", "p1"],
            "area_km2": [0.00005, 0.00006],
            "area_bucket": ["<100m2", "<100m2"],
        }
    )
    with pytest.raises(ValueError, match="duplicate polygon metadata"):
        enrich_v2_table(_source(), {"alpha-latest": metadata, "beta-latest": metadata})


def test_enrich_v2_input_reads_metadata_and_writes_atomically(tmp_path: Path) -> None:
    source_path = tmp_path / "source.parquet"
    output_path = tmp_path / "nested" / "v2.parquet"
    pq.write_table(_source(), source_path)
    metadata = {
        "alpha-latest": _metadata(
            tmp_path / "alpha.parquet",
            [
                {
                    "polygon_id": "p1",
                    "area_km2": 0.5,
                    "area_bucket": "0.1-1km2",
                }
            ],
        ),
        "beta-latest": _metadata(
            tmp_path / "beta.parquet",
            [
                {
                    "polygon_id": "p2",
                    "area_km2": 5.0,
                    "area_bucket": "1-10km2",
                }
            ],
        ),
    }

    result = enrich_v2_input(source_path, output_path, metadata_paths=metadata)

    assert result == output_path
    assert pq.read_table(output_path)["area_bucket"].to_pylist() == ["small", "medium"]
    assert not list(output_path.parent.glob(".v2.parquet.*"))


def test_enrich_v2_input_rejects_symlink_source(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"not used")
    link = tmp_path / "link.parquet"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="regular"):
        enrich_v2_input(link, tmp_path / "output.parquet", metadata_paths={})


def test_enrich_v2_input_rejects_symlink_output(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pq.write_table(_source(), source)
    target = tmp_path / "target.parquet"
    target.write_bytes(b"target")
    link = tmp_path / "output.parquet"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        enrich_v2_input(source, link, metadata_paths={})


def test_downloaded_metadata_uses_sorted_regions_and_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.parquet"
    pq.write_table(_source(), source_path)
    calls: list[tuple[str, str, str, str]] = []
    paths = {
        "alpha-latest": _metadata(
            tmp_path / "alpha.parquet",
            [{"polygon_id": "p1", "area_km2": 0.5, "area_bucket": "small"}],
        ),
        "beta-latest": _metadata(
            tmp_path / "beta.parquet",
            [{"polygon_id": "p2", "area_km2": 5.0, "area_bucket": "medium"}],
        ),
    }

    def fake_download(
        *, repo_id: str, repo_type: str, revision: str, filename: str, cache_dir: Path
    ) -> str:
        calls.append((repo_id, repo_type, revision, filename))
        return str(paths[filename.removeprefix("polygons/").removesuffix(".parquet")])

    monkeypatch.setattr(
        "osm_polygon_sentence_relevance.labeling.v2_input.hf_hub_download",
        fake_download,
    )
    output = tmp_path / "output.parquet"
    from osm_polygon_sentence_relevance.labeling.v2_input import (
        download_and_enrich_v2_input,
    )

    download_and_enrich_v2_input(
        source_path,
        output,
        dataset_id="owner/input",
        revision="a" * 40,
        cache_dir=tmp_path / "cache",
    )
    assert [call[-1] for call in calls] == [
        "polygons/alpha-latest.parquet",
        "polygons/beta-latest.parquet",
    ]
    assert all(call[:3] == ("owner/input", "dataset", "a" * 40) for call in calls)
    assert hashlib.sha256(output.read_bytes()).hexdigest()


def test_download_one_shard_metadata_uses_pinned_upstream_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata(
        tmp_path / "alpha.parquet",
        [{"polygon_id": "p1", "area_km2": 0.5, "area_bucket": "0.1-1km2"}],
    )
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(metadata)

    monkeypatch.setattr(v2_input, "hf_hub_download", fake_download)

    result = download_v2_polygon_metadata(
        dataset_id="owner/input",
        revision="a" * 40,
        shard_key="alpha-latest",
        cache_dir=tmp_path / "cache",
    )

    assert result.column_names == ["area_bucket", "area_km2", "polygon_id"]
    assert calls == [
        {
            "repo_id": "owner/input",
            "repo_type": "dataset",
            "revision": "a" * 40,
            "filename": "polygons/alpha-latest.parquet",
            "cache_dir": tmp_path / "cache",
        }
    ]


def test_atomic_writer_cleans_temp_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.parquet"
    monkeypatch.setattr(
        v2_input.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace"):
        _write_table_atomically(_source(), output)
    assert not output.exists()
    assert not list(tmp_path.glob(".output.parquet.*"))


def test_atomic_writer_rejects_symlink_output(tmp_path: Path) -> None:
    target = tmp_path / "target.parquet"
    target.write_bytes(b"target")
    output = tmp_path / "output.parquet"
    output.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        _write_table_atomically(_source(), output)


def test_hf_hub_download_wrapper_is_lazy_and_injectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = SimpleNamespace(hf_hub_download=lambda **_kwargs: "cached.parquet")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    assert v2_input.hf_hub_download(repo_id="owner/input") == "cached.parquet"


def test_v2_input_module_main_uses_pinned_downloader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "output.parquet"
    pq.write_table(_source(), source)
    monkeypatch.setattr(
        v2_input,
        "_download_polygon_metadata",
        lambda *_args, **_kwargs: {
            "alpha-latest": pa.table(
                {
                    "polygon_id": ["p1"],
                    "area_km2": [0.5],
                    "area_bucket": ["0.1-1km2"],
                }
            ),
            "beta-latest": pa.table(
                {
                    "polygon_id": ["p2"],
                    "area_km2": [5.0],
                    "area_bucket": ["1-10km2"],
                }
            ),
        },
    )
    assert (
        v2_input.main(
            [
                "--source",
                str(source),
                "--output",
                str(output),
                "--dataset-id",
                "owner/input",
                "--revision",
                "a" * 40,
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )
        == 0
    )
    assert output.is_file()
