"""Exact contracts for small streaming-finalization helpers."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
import scripts.streaming.finalization as finalization
import scripts.streaming.v2_finalization as v2_finalization
from scripts.streaming.finalization import (
    StreamingFinalizationError,
    _evict_materialized_file,
    _expected_finalization_keys,
    _identity,
    _remove_empty_materialized_parents,
    _remove_empty_parent,
    _validate_inventory,
    _validate_sampling_request,
    _validate_sampling_seed,
    _validate_sampling_target,
)
from scripts.streaming.v2_finalization import (
    _report_from_metadata,
    _report_values,
    _v2_manifest,
    _v2_readme,
)

from osm_polygon_sentence_relevance.sentences.finalization import FinalizationReport


def test_identity_preserves_every_run_identity_field() -> None:
    assert _identity(
        source_commit="source-commit",
        input_dataset_revision="input-revision",
        pipeline_version="pipeline-v2",
        model_name="model-name",
        batch_size=37,
    ) == {
        "source_commit": "source-commit",
        "input_dataset_revision": "input-revision",
        "pipeline_version": "pipeline-v2",
        "model_name": "model-name",
        "batch_size": 37,
    }


def test_sampling_request_validates_target_then_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        finalization,
        "_validate_sampling_target",
        lambda target: calls.append(("target", target)),
    )
    monkeypatch.setattr(
        finalization,
        "_validate_sampling_seed",
        lambda seed: calls.append(("seed", seed)),
    )

    _validate_sampling_request(17, "seed")

    assert calls == [("target", 17), ("seed", "seed")]


@pytest.mark.parametrize("target", [True, False, 0, -1, 1.5, "1"])
def test_sampling_target_rejects_non_positive_or_non_integer_values(
    target: object,
) -> None:
    with pytest.raises(
        StreamingFinalizationError,
        match=r"^sampling target must be a positive integer$",
    ):
        _validate_sampling_target(target)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", [None, 1, 200_000])
def test_sampling_target_accepts_none_or_positive_integers(target: int | None) -> None:
    _validate_sampling_target(target)


@pytest.mark.parametrize("seed", [None, "", "   ", 12])
def test_sampling_seed_rejects_blank_or_non_string_values(seed: object) -> None:
    with pytest.raises(
        StreamingFinalizationError,
        match=r"^sampling seed must be non-blank$",
    ):
        _validate_sampling_seed(seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", ["seed", " seed "])
def test_sampling_seed_accepts_non_blank_strings(seed: str) -> None:
    _validate_sampling_seed(seed)


def test_inventory_returns_shards_in_deterministic_order() -> None:
    handles = [
        type("Handle", (), {"shard_key": "z-latest"})(),
        type("Handle", (), {"shard_key": "a-latest"})(),
    ]

    assert [handle.shard_key for handle in _validate_inventory(handles, ["a-latest", "z-latest"])] == [
        "a-latest",
        "z-latest",
    ]


def test_inventory_rejects_duplicate_shards_with_exact_error() -> None:
    handle = type("Handle", (), {"shard_key": "a-latest"})()

    with pytest.raises(
        StreamingFinalizationError,
        match=r"^staging run contains duplicate shard keys$",
    ):
        _validate_inventory([handle, handle], ["a-latest"])


def test_inventory_reports_sorted_missing_and_unexpected_shards() -> None:
    handles = [
        type("Handle", (), {"shard_key": "z-latest"})(),
        type("Handle", (), {"shard_key": "b-latest"})(),
    ]

    with pytest.raises(StreamingFinalizationError) as error:
        _validate_inventory(handles, ["a-latest", "b-latest", "c-latest"])

    assert str(error.value) == (
        "staging checkpoint inventory is incomplete or unexpected: "
        "missing=['a-latest', 'c-latest'], unexpected=['z-latest']"
    )


def test_expected_finalization_keys_preserves_explicit_inventory() -> None:
    expected = ("a-latest", "b-latest")

    assert (
        _expected_finalization_keys(
            hub_api=object(),
            upstream_repo_id="owner/input",
            revision="revision",
            expected=expected,
        )
        is expected
    )


def test_expected_finalization_keys_fetches_remote_inventory_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_list_remote_shard_keys(**kwargs: object) -> list[str]:
        calls.append(kwargs)
        return ["b-latest", "a-latest"]

    monkeypatch.setattr(finalization, "list_remote_shard_keys", fake_list_remote_shard_keys)

    assert _expected_finalization_keys(
        hub_api="api",
        upstream_repo_id="owner/input",
        revision="revision",
        expected=None,
    ) == ["b-latest", "a-latest"]
    assert calls == [
        {
            "hub_api": "api",
            "repo_id": "owner/input",
            "revision": "revision",
        }
    ]


def test_v2_report_helpers_use_exact_public_error_messages() -> None:
    with pytest.raises(ValueError, match=r"^finalized artifact has no finalization report$"):
        _report_from_metadata({})

    with pytest.raises(ValueError, match=r"^finalized artifact report is invalid$"):
        _report_values(
            {
                "input_sentence_occurrence_count": 1,
                "output_sentence_count": True,
                "duplicate_occurrence_count_removed": 0,
                "cross_source_duplicate_group_count": 0,
            }
        )


def test_v2_artifact_metadata_preserves_all_integrity_fields(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table_path = tmp_path / "finalized.parquet"
    table_path.write_bytes(b"artifact")
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        v2_finalization,
        "sha256_file",
        lambda path: calls.append(("file", path)) or "table-digest",
    )
    monkeypatch.setattr(
        v2_finalization,
        "schema_sha256",
        lambda schema: calls.append(("schema", schema)) or "schema-digest",
    )
    shard = type("Shard", (), {"shard_key": "a-latest"})()
    output_table = pa.table({"sentence_id": ["s1", "s2"]})
    stream_schema = output_table.schema
    report = FinalizationReport(5, 2, 3, 1)

    metadata = v2_finalization._artifact_metadata(
        table_path,
        output_table,
        shard,
        stream_schema,
        {"source_commit": "commit", "pipeline_version": "v2"},
        report,
    )

    assert metadata == {
        "schema_version": 1,
        "shard_key": "a-latest",
        "table_sha256": "table-digest",
        "table_bytes": 8,
        "row_count": 2,
        "schema_sha256": "schema-digest",
        "source_commit": "commit",
        "pipeline_version": "v2",
        "finalization_report": {
            "input_sentence_occurrence_count": 5,
            "output_sentence_count": 2,
            "duplicate_occurrence_count_removed": 3,
            "cross_source_duplicate_group_count": 1,
        },
    }
    assert calls == [("file", table_path), ("schema", stream_schema)]


def test_v2_manifest_preserves_sampling_and_provenance_contract() -> None:
    report = FinalizationReport(20, 12, 8, 2)

    assert _v2_manifest(
        10,
        "digest",
        10,
        "seed",
        "owner/input",
        "revision",
        "commit",
        "pipeline-v2",
        report,
    ) == {
        "manifest_version": 1,
        "purpose": "v2-worldwide-label-input",
        "row_count": 10,
        "sha256": "digest",
        "input_dataset_id": "owner/input",
        "input_dataset_revision": "revision",
        "source_commit": "commit",
        "pipeline_version": "pipeline-v2",
        "sampling": {
            "target": 10,
            "seed": "seed",
            "source_finalized_rows": 12,
        },
    }


def test_v2_readme_preserves_public_metadata_text() -> None:
    assert _v2_readme(
        10,
        FinalizationReport(20, 12, 8, 2),
        "seed",
        "revision",
        "digest",
    ) == (
        "# Worldwide V2 labeling input\n\n"
        "This internal artifact was generated deterministically from the "
        "complete validated split checkpoints.\n\n"
        "- Selected sentences: 10\n"
        "- Source finalized sentences: 12\n"
        "- Sampling seed: `seed`\n"
        "- Input revision: `revision`\n"
        "- SHA-256: `digest`\n"
    )


def test_evict_outside_cache_uses_exact_safety_error(tmp_path) -> None:
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"x")

    with pytest.raises(
        StreamingFinalizationError,
        match=r"^refusing to evict a checkpoint outside the finalization cache$",
    ):
        _evict_materialized_file(outside, tmp_path / "cache")


def test_empty_parent_cleanup_visits_deepest_parents_first(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    deep = root / "a" / "a-latest"
    shallow = root / "z"
    calls: list[Path] = []
    monkeypatch.setattr(
        finalization,
        "_remove_empty_parent",
        lambda parent, root: calls.append(parent),
    )

    _remove_empty_materialized_parents(
        [deep / "table.parquet", shallow / "metadata.json", None], root
    )

    assert calls == [deep, shallow]


def test_empty_parent_cleanup_removes_only_descendants_of_root(tmp_path) -> None:
    root = tmp_path / "cache"
    nested = root / "run" / "a-latest"
    nested.mkdir(parents=True)

    _remove_empty_parent(nested, root)

    assert not nested.exists()
    assert not (root / "run").exists()
    assert root.exists()
