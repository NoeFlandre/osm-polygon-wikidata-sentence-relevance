from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.labeling.contracts import RunIdentity
from osm_polygon_sentence_relevance.labeling.v2_checkpoint import V2CheckpointStore
from osm_polygon_sentence_relevance.labeling.v2_contracts import (
    V2_LOGIT_PROMPT_VERSION,
    V2LogitRecord,
)
from osm_polygon_sentence_relevance.labeling.v2_finalization import (
    V2_PUBLICATION_FILES,
    finalize_v2_dataset,
    validate_v2_publication,
)
from osm_polygon_sentence_relevance.labeling.v2_sampling import select_v2_rows


def _identity(input_sha256: str, *, row_limit: int = 0) -> RunIdentity:
    return RunIdentity(
        input_sha256=input_sha256,
        input_dataset_revision="b" * 40,
        model_repo_id="ggml-org/Qwen3.6-27B-GGUF",
        model_revision="c" * 40,
        model_file="Qwen3.6-27B-Q4_K_M.gguf",
        model_file_sha256="d" * 64,
        prompt_version=V2_LOGIT_PROMPT_VERSION,
        source_commit="e" * 40,
        engine="llama.cpp",
        engine_version="1",
        batch_size=2,
        row_limit=row_limit,
        sampling_target=2,
        sampling_seed="seed",
        h3_resolution=3,
        sampling_version="v2-area-h3-logit",
        release_lane="v2-worldwide",
    )


def _make_release(tmp_path: Path) -> tuple[Path, Path, V2CheckpointStore]:
    input_path = tmp_path / "input.parquet"
    rows = [
        {
            "sentence_id": "s0",
            "sentence_text_raw": "The valley has steep slopes.",
            "previous_sentence": None,
            "next_sentence": None,
            "page_title": "Valley",
            "section_path": ["Geography"],
            "polygon_id": "p0",
            "osm_primary_tag": "natural=valley",
            "language": "en",
            "lat": 45.0,
            "lon": 2.0,
            "area_km2": 20.0,
            "area_bucket": "large",
        },
        {
            "sentence_id": "s1",
            "sentence_text_raw": "The place was founded in 1900.",
            "previous_sentence": None,
            "next_sentence": None,
            "page_title": "Valley",
            "section_path": ["History"],
            "polygon_id": "p0",
            "osm_primary_tag": "natural=valley",
            "language": "fr",
            "lat": 45.0,
            "lon": 2.0,
            "area_km2": 20.0,
            "area_bucket": "large",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), input_path)
    identity = _identity(hashlib.sha256(input_path.read_bytes()).hexdigest())
    store = V2CheckpointStore(tmp_path / "work", identity)
    store.write_batch(
        0,
        [
            V2LogitRecord("s0", "yes", -0.1, -1.1),
            V2LogitRecord("s1", "no", -1.1, -0.1),
        ],
    )
    output = tmp_path / "output"
    finalize_v2_dataset(
        input_path=input_path,
        store=store,
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )
    return input_path, output, store


def test_v2_finalization_writes_only_binary_score_release(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    rows = [
        {
            "sentence_id": "s0",
            "sentence_text_raw": "The valley has steep slopes.",
            "previous_sentence": None,
            "next_sentence": None,
            "page_title": "Valley",
            "section_path": ["Geography"],
            "polygon_id": "p0",
            "osm_primary_tag": "natural=valley",
            "language": "en",
            "lat": 45.0,
            "lon": 2.0,
            "area_km2": 20.0,
            "area_bucket": "large",
        },
        {
            "sentence_id": "s1",
            "sentence_text_raw": "The place was founded in 1900.",
            "previous_sentence": None,
            "next_sentence": None,
            "page_title": "Valley",
            "section_path": ["History"],
            "polygon_id": "p0",
            "osm_primary_tag": "natural=valley",
            "language": "fr",
            "lat": 45.0,
            "lon": 2.0,
            "area_km2": 20.0,
            "area_bucket": "large",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), input_path)
    identity = _identity(hashlib.sha256(input_path.read_bytes()).hexdigest())
    store = V2CheckpointStore(tmp_path / "work", identity)
    store.write_batch(
        0,
        [
            V2LogitRecord("s0", "yes", -0.1, -1.1),
            V2LogitRecord("s1", "no", -1.1, -0.1),
        ],
    )
    output = tmp_path / "output"
    result = finalize_v2_dataset(
        input_path=input_path,
        store=store,
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )

    assert result.row_count == 2
    assert {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    } == set(V2_PUBLICATION_FILES)
    table = pq.read_table(output / "sentences.parquet")
    assert {
        "place_relevance",
        "yes_logprob",
        "no_logprob",
        "logit_margin",
        "two_class_probability",
    }.issubset(table.column_names)
    assert "place_reason" not in table.column_names
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["statistics"]["place_counts"] == {"no": 1, "yes": 1}
    validate_v2_publication(output)


def test_v2_validation_accepts_float_round_trip_difference(tmp_path: Path) -> None:
    """Parquet float serialization may change a derived score by one ULP."""

    _input_path, output, _store = _make_release(tmp_path)
    table = pq.read_table(output / "sentences.parquet")
    original_size = (output / "sentences.parquet").stat().st_size
    original_manifest = json.loads((output / "manifest.json").read_text())
    original_digest = original_manifest["parquet_sha256"]
    probabilities = table["two_class_probability"].to_pylist()
    probabilities[0] = math.nextafter(probabilities[0], math.inf)
    table = table.set_column(
        table.schema.get_field_index("two_class_probability"),
        pa.field("two_class_probability", pa.float64(), nullable=False),
        pa.array(probabilities),
    )
    pq.write_table(table, output / "sentences.parquet", compression="zstd")
    updated_size = (output / "sentences.parquet").stat().st_size
    digest = hashlib.sha256((output / "sentences.parquet").read_bytes()).hexdigest()
    readme = (output / "README.md").read_text()
    (output / "README.md").write_text(
        readme.replace(
            f"num_bytes: {original_size}", f"num_bytes: {updated_size}"
        ).replace(f"`{original_digest}`", f"`{digest}`")
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parquet_sha256"] = digest
    manifest["artifact_sha256"]["sentences.parquet"] = digest
    manifest_path.write_text(json.dumps(manifest))

    validate_v2_publication(output)


def test_v2_card_is_public_facing_and_navigable(tmp_path: Path) -> None:
    """The generated V2 card exposes provenance, citation, and dashboards."""

    _input_path, output, _store = _make_release(tmp_path)
    card = (output / "README.md").read_text()
    for token in (
        "github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance/blob/main/README.md",
        "worldwide-stratified-labeling-trackio",
        "citation.cff",
        "model-generated",
        "decoded token itself is",
        "missing_coordinate_count",
        "@dataset",
    ):
        assert token in card


def test_v2_finalization_uses_smoke_row_limit_before_sampling_target(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.parquet"
    rows = [
        {
            "sentence_id": f"s{index}",
            "sentence_text_raw": f"Sentence {index}.",
            "previous_sentence": None,
            "next_sentence": None,
            "page_title": "Valley",
            "section_path": ["Geography"],
            "polygon_id": f"p{index}",
            "osm_primary_tag": "natural=valley",
            "language": "en",
            "lat": 45.0 + index,
            "lon": 2.0,
            "area_km2": 20.0,
            "area_bucket": "large",
        }
        for index in range(2)
    ]
    pq.write_table(pa.Table.from_pylist(rows), input_path)
    identity = _identity(
        hashlib.sha256(input_path.read_bytes()).hexdigest(), row_limit=1
    )
    store = V2CheckpointStore(tmp_path / "work", identity)
    selected_id = select_v2_rows(pq.read_table(input_path), target=1, seed="seed")[
        "sentence_id"
    ][0].as_py()
    store.write_batch(0, [V2LogitRecord(selected_id, "yes", -0.1, -1.1)])

    result = finalize_v2_dataset(
        input_path=input_path,
        store=store,
        output_dir=tmp_path / "output",
        dataset_repo_id="owner/dataset",
    )

    assert result.row_count == 1


def test_v2_finalization_replaces_existing_output_atomically(tmp_path: Path) -> None:
    input_path, output, store = _make_release(tmp_path)
    result = finalize_v2_dataset(
        input_path=input_path,
        store=store,
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )
    assert result.row_count == 2
    assert not output.with_name(".output.backup").exists()


def test_v2_finalization_rejects_input_hash_and_score_set_mismatches(
    tmp_path: Path,
) -> None:
    input_path, output, store = _make_release(tmp_path)
    mismatched_store = V2CheckpointStore(tmp_path / "wrong-work", _identity("0" * 64))
    with pytest.raises(ValueError, match="input SHA"):
        finalize_v2_dataset(
            input_path=input_path,
            store=mismatched_store,
            output_dir=tmp_path / "wrong-output",
            dataset_repo_id="owner/dataset",
        )
    (output / "manifest.json").write_text((output / "manifest.json").read_text())
    empty_store = V2CheckpointStore(tmp_path / "empty-work", store.identity)
    with pytest.raises(ValueError, match="exactly one score"):
        finalize_v2_dataset(
            input_path=input_path,
            store=empty_store,
            output_dir=tmp_path / "empty-output",
            dataset_repo_id="owner/dataset",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("layout", "layout"),
        ("release", "release lane"),
        ("parquet", "Parquet hash"),
        ("artifact_keys", "artifact hashes"),
        ("artifact_digest", "artifact hash"),
        ("missing_score", "score columns"),
        ("derived", "derived score"),
        ("prompt", "prompt version"),
        ("model_repo", "model repository"),
        ("model_file", "model file"),
        ("analytics", "analytics"),
        ("readme", "README"),
    ],
)
def test_v2_publication_validation_rejects_tampering(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _input_path, output, _store = _make_release(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "layout":
        (output / "extra.txt").write_text("unexpected")
    elif mutation == "release":
        manifest["release_lane"] = "v1"
    elif mutation == "parquet":
        manifest["parquet_sha256"] = "0" * 64
    elif mutation == "artifact_keys":
        manifest["artifact_sha256"] = {}
    elif mutation == "artifact_digest":
        manifest["artifact_sha256"]["assets/label_distribution.png"] = "0" * 64
    elif mutation == "missing_score":
        table = pq.read_table(output / "sentences.parquet").drop_columns(
            ["place_relevance"]
        )
        pq.write_table(table, output / "sentences.parquet", compression="zstd")
        manifest["parquet_sha256"] = hashlib.sha256(
            (output / "sentences.parquet").read_bytes()
        ).hexdigest()
        manifest["artifact_sha256"]["sentences.parquet"] = manifest["parquet_sha256"]
    elif mutation == "derived":
        table = pq.read_table(output / "sentences.parquet")
        table = table.set_column(
            table.schema.get_field_index("logit_margin"),
            pa.field("logit_margin", pa.float64(), nullable=False),
            pa.array([99.0, 99.0]),
        )
        pq.write_table(table, output / "sentences.parquet", compression="zstd")
        manifest["parquet_sha256"] = hashlib.sha256(
            (output / "sentences.parquet").read_bytes()
        ).hexdigest()
        manifest["artifact_sha256"]["sentences.parquet"] = manifest["parquet_sha256"]
    elif mutation in {"prompt", "model_repo", "model_file"}:
        key = {
            "prompt": "prompt_version",
            "model_repo": "model_repo_id",
            "model_file": "model_file",
        }[mutation]
        manifest["run_identity"][key] = "wrong"
    elif mutation == "analytics":
        manifest["statistics"]["unique_polygons"] = 999
    elif mutation == "readme":
        (output / "README.md").write_text("stale")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        validate_v2_publication(output)


def test_v2_finalization_cleans_staging_after_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.parquet"
    rows = [
        {
            "sentence_id": "s0",
            "sentence_text_raw": "A valley.",
            "previous_sentence": None,
            "next_sentence": None,
            "page_title": "Valley",
            "section_path": ["Geography"],
            "polygon_id": "p0",
            "osm_primary_tag": "natural=valley",
            "language": "en",
            "lat": 45.0,
            "lon": 2.0,
            "area_km2": 20.0,
            "area_bucket": "large",
        }
    ]
    pq.write_table(pa.Table.from_pylist(rows), input_path)
    identity = _identity(hashlib.sha256(input_path.read_bytes()).hexdigest())
    store = V2CheckpointStore(tmp_path / "work", identity)
    store.write_batch(0, [V2LogitRecord("s0", "yes", -0.1, -1.1)])
    import osm_polygon_sentence_relevance.labeling.v2_finalization as module

    monkeypatch.setattr(
        module,
        "write_v2_manual_eval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("manual eval")),
    )
    with pytest.raises(OSError, match="manual eval"):
        finalize_v2_dataset(
            input_path=input_path,
            store=store,
            output_dir=tmp_path / "output",
            dataset_repo_id="owner/dataset",
        )
    assert not list(tmp_path.glob(".output.*"))
