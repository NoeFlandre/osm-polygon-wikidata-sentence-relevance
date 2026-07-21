"""RED/GREEN contract tests for bounded streamed finalization."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.streaming.finalization import (
    StreamingFinalizationError,
    _aggregate_reports,
    _evict_materialized,
    _validate_inventory,
    _verify_shard_namespace,
    finalize_streamed_run,
    main,
)
from scripts.streaming.offload import OffloadHandle

from osm_polygon_sentence_relevance.contracts.schemas import (
    OUTPUT_SENTENCE_SCHEMA,
    SEGMENTED_SENTENCES_SCHEMA,
)
from osm_polygon_sentence_relevance.output.validation import (
    validate_export_directory,
)

REVISION = "a" * 40
SOURCE_COMMIT = "b" * 40


def _segmented(shard: str, text: str) -> pa.Table:
    values: dict[str, pa.Array] = {}
    for field in SEGMENTED_SENTENCES_SCHEMA:
        if field.name == "polygon_id":
            raw = [f"{shard}:1"]
        elif field.name == "wikidata":
            raw = ["Q1"]
        elif field.name == "document_id":
            raw = [f"doc-{shard}"]
        elif field.name == "article_id":
            raw = [None]
        elif field.name == "source":
            raw = ["wikipedia"]
        elif field.name == "language":
            raw = ["en"]
        elif field.name == "site":
            raw = ["en.wikipedia.org"]
        elif field.name == "page_title":
            raw = [shard]
        elif field.name == "url":
            raw = [f"https://example.test/{shard}"]
        elif field.name in {
            "page_id",
            "revision_id",
            "section_index",
            "sentence_index",
        }:
            raw = [1]
        elif field.name == "revision_timestamp":
            raw = ["2026-01-01T00:00:00Z"]
        elif field.name in {"document_content_hash", "section_content_hash"}:
            raw = ["0" * 64]
        elif field.name == "section_id":
            raw = ["section-1"]
        elif field.name == "section_path":
            raw = [["Lead"]]
        elif field.name in {"sentence_text_raw", "sentence_text_normalized"}:
            raw = [text]
        elif field.name == "polygon_name":
            raw = [shard]
        elif field.name == "osm_primary_tag":
            raw = ["boundary"]
        elif field.name == "osm_tags":
            raw = [[("name", shard)]]
        elif field.name == "region":
            raw = [shard]
        elif field.name in {"lat", "lon"}:
            raw = [1.0]
        else:  # pragma: no cover - schema change tripwire
            raise AssertionError(field.name)
        values[field.name] = pa.array(raw, type=field.type)
    return pa.table(values, schema=SEGMENTED_SENTENCES_SCHEMA)


def _handle(tmp_path: Path, shard: str) -> OffloadHandle:
    directory = tmp_path / "cache" / shard
    directory.mkdir(parents=True)
    table_path = directory / "segmented.parquet"
    metadata_path = directory / "metadata.json"
    pq.write_table(_segmented(shard, f"Sentence for {shard}."), table_path)
    metadata_path.write_text("{}", encoding="utf-8")
    return OffloadHandle(
        repo_id="owner/output",
        run_id="run-1",
        shard_key=shard,
        staging_revision="checkpoints/run-1",
        folder_path=f"checkpoints/run-1/{shard}",
        expected_table_sha256="0" * 64,
        computed_table_sha256="0" * 64,
        table_bytes=table_path.stat().st_size,
        metadata={
            "source_commit": SOURCE_COMMIT,
            "input_dataset_revision": REVISION,
            "pipeline_version": "0.1.0",
            "model_name": "sat-3l-sm",
            "batch_size": 128,
        },
        local_table_path=table_path,
        local_metadata_path=metadata_path,
    )


def test_inventory_requires_exact_set(tmp_path: Path) -> None:
    handle = _handle(tmp_path, "a-latest")
    with pytest.raises(StreamingFinalizationError, match="inventory"):
        _validate_inventory([handle], ["a-latest", "b-latest"])


def test_inventory_rejects_duplicate_keys(tmp_path: Path) -> None:
    handle = _handle(tmp_path, "a-latest")
    with pytest.raises(StreamingFinalizationError, match="duplicate"):
        _validate_inventory([handle, handle], ["a-latest"])


def test_namespace_rejects_cross_shard_polygon() -> None:
    with pytest.raises(StreamingFinalizationError, match="another shard"):
        _verify_shard_namespace(_segmented("a-latest", "x"), "b-latest")


def test_evict_refuses_path_outside_cache(tmp_path: Path) -> None:
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"x")
    handle = mock.Mock(local_table_path=outside, local_metadata_path=None)
    with pytest.raises(StreamingFinalizationError, match="outside"):
        _evict_materialized(handle, tmp_path / "cache")


def test_evict_ignores_absent_files_and_stops_at_nonempty_parent(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    directory = cache / "run" / "a-latest"
    directory.mkdir(parents=True)
    keep = directory / "keep"
    keep.write_text("keep", encoding="utf-8")
    handle = mock.Mock(
        local_table_path=directory / "missing.parquet",
        local_metadata_path=None,
    )
    _evict_materialized(handle, cache)
    assert keep.exists()


def test_aggregate_reports() -> None:
    from osm_polygon_sentence_relevance.sentences.finalization import (
        FinalizationReport,
    )

    result = _aggregate_reports(
        [FinalizationReport(3, 2, 1, 1), FinalizationReport(4, 3, 1, 0)]
    )
    assert result == FinalizationReport(7, 5, 2, 1)


def test_finalization_is_bounded_and_produces_valid_public_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "123")
    handles = [_handle(tmp_path, "b-latest"), _handle(tmp_path, "a-latest")]
    seen: list[str] = []

    def materialize(handle: OffloadHandle, **_: object) -> OffloadHandle:
        seen.append(handle.shard_key)
        return handle

    monkeypatch.setattr(
        "scripts.streaming.finalization.discover_run", lambda **_: handles
    )
    monkeypatch.setattr(
        "scripts.streaming.finalization.materialize_checkpoint", materialize
    )
    monkeypatch.delattr(pa, "concat_tables")

    output = finalize_streamed_run(
        hub_api=mock.Mock(),
        repo_id="owner/output",
        upstream_repo_id="owner/input",
        run_id="run-1",
        staging_revision="checkpoints/run-1",
        source_commit=SOURCE_COMMIT,
        input_dataset_revision=REVISION,
        pipeline_version="0.1.0",
        model_name="sat-3l-sm",
        batch_size=128,
        local_cache_dir=tmp_path / "cache",
        scratch_dir=tmp_path / "scratch",
        output_dir=tmp_path / "output",
        expected_shard_keys=["a-latest", "b-latest"],
    )

    assert seen == ["a-latest", "b-latest"]
    assert sorted(path.name for path in output.iterdir()) == [
        "README.md",
        "manifest.json",
        "sentences.parquet",
    ]
    assert pq.read_schema(output / "sentences.parquet").equals(OUTPUT_SENTENCE_SCHEMA)
    validated = validate_export_directory(output)
    assert validated.row_count == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["statistics"]["row_count"] == 2
    assert "GENERATED AUTOMATICALLY" in (output / "README.md").read_text()
    assert not any((tmp_path / "cache").rglob("segmented.parquet"))


def test_finalization_requires_oar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OAR_JOB_ID", raising=False)
    with pytest.raises(StreamingFinalizationError, match="OAR"):
        finalize_streamed_run(
            hub_api=mock.Mock(),
            repo_id="owner/output",
            upstream_repo_id="owner/input",
            run_id="run-1",
            staging_revision="checkpoints/run-1",
            source_commit=SOURCE_COMMIT,
            input_dataset_revision=REVISION,
            pipeline_version="0.1.0",
            model_name="sat-3l-sm",
            batch_size=128,
            local_cache_dir=tmp_path / "cache",
            scratch_dir=tmp_path / "scratch",
            output_dir=tmp_path / "output",
            expected_shard_keys=[],
        )


def test_finalization_refuses_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "1")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(StreamingFinalizationError, match="fresh"):
        finalize_streamed_run(
            hub_api=mock.Mock(),
            repo_id="owner/output",
            upstream_repo_id="owner/input",
            run_id="r",
            staging_revision="checkpoints/r",
            source_commit=SOURCE_COMMIT,
            input_dataset_revision=REVISION,
            pipeline_version="0.1.0",
            model_name="sat-3l-sm",
            batch_size=128,
            local_cache_dir=tmp_path / "cache",
            scratch_dir=tmp_path / "scratch",
            output_dir=output,
            expected_shard_keys=[],
        )


def test_finalization_discovers_expected_upstream_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "1")
    inventory = mock.Mock(return_value=["a-latest"])
    monkeypatch.setattr(
        "scripts.streaming.finalization.list_remote_shard_keys", inventory
    )
    monkeypatch.setattr("scripts.streaming.finalization.discover_run", lambda **_: [])
    with pytest.raises(StreamingFinalizationError, match="inventory"):
        finalize_streamed_run(
            hub_api=mock.Mock(),
            repo_id="owner/output",
            upstream_repo_id="owner/input",
            run_id="r",
            staging_revision="checkpoints/r",
            source_commit=SOURCE_COMMIT,
            input_dataset_revision=REVISION,
            pipeline_version="0.1.0",
            model_name="sat-3l-sm",
            batch_size=128,
            local_cache_dir=tmp_path / "cache",
            scratch_dir=tmp_path / "scratch",
            output_dir=tmp_path / "output",
        )
    inventory.assert_called_once()


def test_finalization_cleans_partial_output_on_materialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "1")
    handle = _handle(tmp_path, "a-latest")
    handle = replace(handle, local_table_path=None)
    monkeypatch.setattr(
        "scripts.streaming.finalization.discover_run", lambda **_: [handle]
    )
    monkeypatch.setattr(
        "scripts.streaming.finalization.materialize_checkpoint", lambda *_, **__: handle
    )
    output = tmp_path / "output"
    with pytest.raises(StreamingFinalizationError, match="no table"):
        finalize_streamed_run(
            hub_api=mock.Mock(),
            repo_id="owner/output",
            upstream_repo_id="owner/input",
            run_id="r",
            staging_revision="checkpoints/r",
            source_commit=SOURCE_COMMIT,
            input_dataset_revision=REVISION,
            pipeline_version="0.1.0",
            model_name="sat-3l-sm",
            batch_size=128,
            local_cache_dir=tmp_path / "cache",
            scratch_dir=tmp_path / "scratch",
            output_dir=output,
            expected_shard_keys=["a-latest"],
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".finalizing-*"))


def test_main_accepts_expected_shard_for_one_shard_canary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI must accept ``--expected-shard afghanistan-latest`` and
    pass it through to ``finalize_streamed_run`` as a one-element
    ``expected_shard_keys`` list, so the bounded finalizer can require
    exactly one expected shard instead of the full upstream inventory.
    Without the flag the CLI cannot drive a canary: it would either
    attempt to download the entire 303-shard inventory or refuse the
    one-shard staging checkpoint set.
    """
    monkeypatch.setenv("OAR_JOB_ID", "1")
    monkeypatch.setattr("huggingface_hub.HfApi", mock.Mock)
    captured: dict[str, object] = {}
    target = tmp_path / "output"

    def fake_finalize(**kwargs: object) -> Path:
        captured.update(kwargs)
        return target

    monkeypatch.setattr(
        "scripts.streaming.finalization.finalize_streamed_run", fake_finalize
    )
    rc = main(
        [
            "--repo-id",
            "owner/output",
            "--upstream-repo-id",
            "owner/input",
            "--run-id",
            "r",
            "--staging-revision",
            "checkpoints/r",
            "--source-commit",
            SOURCE_COMMIT,
            "--input-revision",
            REVISION,
            "--cache-dir",
            str(tmp_path / "cache"),
            "--scratch-dir",
            str(tmp_path / "scratch"),
            "--output-dir",
            str(target),
            "--expected-shard",
            "afghanistan-latest",
        ]
    )
    assert rc == 0
    assert captured["expected_shard_keys"] == ["afghanistan-latest"]


def test_main_threads_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("huggingface_hub.HfApi", mock.Mock)
    target = tmp_path / "output"
    called: dict[str, object] = {}

    def fake_finalize(**kwargs: object) -> Path:
        called.update(kwargs)
        return target

    monkeypatch.setattr(
        "scripts.streaming.finalization.finalize_streamed_run", fake_finalize
    )
    assert (
        main(
            [
                "--repo-id",
                "owner/output",
                "--upstream-repo-id",
                "owner/input",
                "--run-id",
                "r",
                "--staging-revision",
                "checkpoints/r",
                "--source-commit",
                SOURCE_COMMIT,
                "--input-revision",
                REVISION,
                "--cache-dir",
                str(tmp_path / "cache"),
                "--scratch-dir",
                str(tmp_path / "scratch"),
                "--output-dir",
                str(target),
            ]
        )
        == 0
    )
    assert called["repo_id"] == "owner/output"
