from __future__ import annotations

import hashlib
import json
import shutil
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.labeling.canary import select_canary_rows
from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
from osm_polygon_sentence_relevance.labeling.contracts import (
    LabelRecord,
    LabelValue,
    RunIdentity,
)
from osm_polygon_sentence_relevance.labeling.finalization import (
    LabelFinalizationError,
    _render_card,
    _scope_label,
    _validate_replaceable_v2_output,
    finalize_labeled_dataset,
    validate_labeled_publication,
)
from osm_polygon_sentence_relevance.labeling.publication import (
    LabelPublicationError,
    publish_labeled_dataset,
)


def _identity(input_sha256: str = "a" * 64) -> RunIdentity:
    return RunIdentity(
        input_sha256=input_sha256,
        input_dataset_revision="b" * 40,
        model_repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        model_revision="c" * 40,
        model_file="Qwen3.6-27B-Q4_K_M.gguf",
        model_file_sha256="d" * 64,
        prompt_version="afghanistan-landuse-polygon-v1",
        source_commit="e" * 40,
        engine="llama.cpp",
        engine_version="1",
        batch_size=2,
        llama_parallel=16,
        llama_per_slot_context=4096,
        llama_total_context=65536,
        request_concurrency=16,
    )


def _input(path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "sentence_id": ["s1", "s2", "s3"],
                "region": ["afghanistan"] * 3,
                "language": ["en", "fa", "en"],
                "sentence_text_raw": ["farming", "history", "forest"],
            }
        ),
        path,
    )


def _v1_remote_snapshot(tmp_path: Path, output: Path) -> Path:
    """Materialize a finalized V1 output below its public remote prefix."""

    snapshot = tmp_path / "remote-snapshot"
    shutil.copytree(output, snapshot / "v1-afghanistan")
    return snapshot


def _store(
    path: Path,
    input_path: Path,
    *,
    complete: bool = True,
    identity: RunIdentity | None = None,
) -> CheckpointStore:
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    store = CheckpointStore(path, identity or _identity(digest))
    records = [
        LabelRecord(
            "s1",
            LabelValue.YES,
            LabelValue.YES,
            "explicit_land_use",
            "direct_polygon_reference",
            "farming",
        ),
        LabelRecord(
            "s2",
            LabelValue.NO,
            LabelValue.YES,
            "no_landuse_or_cover",
            "direct_polygon_reference",
            "history",
        ),
    ]
    if complete:
        records.append(
            LabelRecord(
                "s3",
                LabelValue.YES,
                LabelValue.UNCERTAIN,
                "explicit_land_cover",
                "insufficient_evidence",
                "forest",
            )
        )
    store.write_batch(0, records)
    store.write_timing(
        {
            "total_wall_seconds": 12.5,
            "initial_inference_seconds": 10.0,
            "repair_inference_seconds": 0.0,
            "inference_seconds": 10.0,
            "checkpoint_and_validation_seconds": 2.5,
        }
    )
    return store


def test_finalization_generates_factual_card_manifest_and_plots(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    _input(input_path)
    output = tmp_path / "publication"
    finalize_labeled_dataset(
        input_path=input_path,
        store=_store(tmp_path / "work", input_path),
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )

    validated = validate_labeled_publication(output)
    assert validated.row_count == 3
    table = pq.read_table(output / "sentences.parquet")
    assert table["landuse_relevance"].to_pylist() == ["yes", "no", "yes"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["release_lane"] == "v1-afghanistan"
    assert manifest["release_prefix"] == "v1-afghanistan"
    assert manifest["statistics"]["landuse_relevance"] == {"no": 1, "yes": 2}
    assert manifest["statistics"]["polygon_relevance"] == {
        "uncertain": 1,
        "yes": 2,
    }
    assert manifest["statistics"]["positive_languages"] == {"en": 2}
    assert manifest["statistics"]["joint_labels"] == {
        "no|yes": 1,
        "yes|uncertain": 1,
        "yes|yes": 1,
    }
    assert set(manifest["artifact_sha256"]) == {
        "assets/label_distribution.png",
        "assets/positive_languages.png",
        "assets/joint_label_heatmap.png",
        "assets/polygon_coverage_funnel.png",
        "assets/reason_code_distribution.png",
        "sentences.parquet",
    }
    card = (output / "README.md").read_text()
    assert "3 labeled sentences" in card
    assert "2 (66.67%)" in card
    assert "unsloth/Qwen3.6-27B-MTP-GGUF" in card
    assert (
        "https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only" in card
    )
    assert (
        "https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance" in card
    )
    assert "## Sentence preparation" in card
    assert "wtpsplit` SaT model (`sat-12l-sm`)" in card
    assert card.index("## Dataset metrics") < card.index("## Sentence preparation")
    assert card.index("## Sentence preparation") < card.index("## Sentence labeling")
    assert card.index("## Sentence labeling") < card.index("## Label summary")
    assert "| Land use / land cover | Polygon relevance | Count | Share |" in card
    assert "| Stage | Polygons |" in card
    assert "![Joint label heatmap]" not in card
    assert "![Polygon coverage funnel]" not in card
    assert "slice_yield.html" not in card
    assert "public Trackio dashboard provides an interactive slice table" in card
    assert "## Label and reason codes" not in card
    assert "![Label distributions]" not in card
    assert "## Language coverage" in card
    assert "![Positive-label languages]" in card
    assert "resolve/main/v1-afghanistan/assets/positive_languages.png" in card
    assert "## Repair" not in card
    assert "## Runtime" not in card
    assert "Initial inference:" not in card
    assert (
        "https://huggingface.co/spaces/NoeFlandre/afghanistan-labeling-trackio" in card
    )
    assert (
        "https://noeflandre.github.io/osm-polygon-wikidata-sentence-relevance/"
        "presentations/afghanistan-dataset-overview/index.html" in card
    )
    assert (
        (output / "assets" / "label_distribution.png")
        .read_bytes()
        .startswith(b"\x89PNG")
    )
    assert (
        (output / "assets" / "positive_languages.png")
        .read_bytes()
        .startswith(b"\x89PNG")
    )


def test_refuses_partial_and_accepts_multiple_regions(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    _input(input_path)
    with pytest.raises(LabelFinalizationError, match="exactly one"):
        finalize_labeled_dataset(
            input_path=input_path,
            store=_store(tmp_path / "partial", input_path, complete=False),
            output_dir=tmp_path / "out",
            dataset_repo_id="owner/dataset",
        )
    table = pq.read_table(input_path).set_column(
        1, "region", pa.array(["afghanistan", "other", "afghanistan"])
    )
    pq.write_table(table, input_path)
    with pytest.raises(LabelFinalizationError, match="V1 Afghanistan"):
        finalize_labeled_dataset(
            input_path=input_path,
            store=_store(tmp_path / "complete", input_path),
            output_dir=tmp_path / "out2",
            dataset_repo_id="owner/dataset",
        )


def test_scope_label_and_v2_card_handle_global_and_empty_inputs() -> None:
    assert _scope_label(pa.table({"sentence_id": ["s1"]})) == "Dataset"
    assert _scope_label(pa.table({"region": ["afghanistan-latest"]})) == "Afghanistan"
    assert _scope_label(pa.table({"region": ["afghanistan", "france"]})) == "Global"

    identity = _identity().to_dict()
    identity.update(
        {
            "sampling_target": 200_000,
            "sampling_seed": "sentence-relevance-v2",
            "h3_resolution": 3,
            "sampling_version": "labeling-v2-h3-language-osm-primary",
        }
    )
    empty_counts = {"yes": 0, "no": 0, "uncertain": 0}
    analytics = {
        "total_labeled_sentences": 0,
        "unique_polygons": 0,
        "unique_languages": 0,
        "strong_positive_yield": 0.0,
        "joint_counts": {
            f"{land}|{polygon}": 0 for land in empty_counts for polygon in empty_counts
        },
        "joint_percentages": {
            f"{land}|{polygon}": 0.0
            for land in empty_counts
            for polygon in empty_counts
        },
        "coverage_funnel": {
            "all_polygons": 0,
            "polygon_relevant_polygons": 0,
            "landuse_relevant_polygons": 0,
            "both_yes_polygons": 0,
        },
        "slices": [],
    }
    card = _render_card(
        dataset_repo_id="owner/dataset",
        row_count=0,
        stats={
            "landuse_relevance": empty_counts,
            "polygon_relevance": empty_counts,
            "analytics": analytics,
        },
        identity=identity,
        timing={
            "initial_inference_seconds": 0.0,
            "repair_inference_seconds": 0.0,
            "inference_seconds": 0.0,
        },
        scope_label="Global",
        publication_revision="v2",
    )
    assert "deterministic stratified sample" in card
    assert "0.00%" in card


def test_v2_finalization_replaces_only_a_previous_v2_target_output(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.parquet"
    _input(input_path)
    table = pq.read_table(input_path).set_column(
        1, "region", pa.array(["afghanistan", "france", "japan"])
    )
    pq.write_table(table, input_path)
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    initial_identity = replace(
        _identity(digest),
        sampling_target=0,
        sampling_seed="sentence-relevance-v2",
        h3_resolution=3,
        sampling_version="labeling-v2-h3-language-osm-primary",
    )
    output = tmp_path / "publication"
    finalize_labeled_dataset(
        input_path=input_path,
        store=_store(tmp_path / "work", input_path, identity=initial_identity),
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )

    expanded_identity = replace(initial_identity, sampling_target=3)
    result = finalize_labeled_dataset(
        input_path=input_path,
        store=CheckpointStore(tmp_path / "work", expanded_identity),
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )

    assert result.row_count == 3
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["run_identity"]["sampling_target"] == 3
    assert manifest["publication_revision"] == "main"
    assert manifest["release_lane"] == "v2-worldwide"
    assert manifest["release_prefix"] == "v2-worldwide"
    assert "assets/h3_sentence_distribution.png" in manifest["artifact_sha256"]
    assert (
        manifest["statistics"]["h3_sentence_distribution"]["missing_coordinate_count"]
        == 3
    )
    card = (output / "README.md").read_text()
    assert "## Geographic distribution" in card
    assert "resolve/main/v2-worldwide/assets/h3_sentence_distribution.png" in card


def test_v2_replacement_guard_rejects_untrusted_previous_outputs(
    tmp_path: Path,
) -> None:
    current = _identity().to_dict()
    current.update(
        {
            "sampling_target": 400,
            "sampling_seed": "sentence-relevance-v2",
            "h3_resolution": 3,
            "sampling_version": "labeling-v2-h3-language-osm-primary",
        }
    )

    invalid_manifest = tmp_path / "invalid"
    invalid_manifest.mkdir()
    (invalid_manifest / "manifest.json").write_text("{")
    with pytest.raises(LabelFinalizationError, match="manifest is invalid"):
        _validate_replaceable_v2_output(invalid_manifest, current)

    missing_identity = tmp_path / "missing-identity"
    missing_identity.mkdir()
    (missing_identity / "manifest.json").write_text("{}")
    with pytest.raises(LabelFinalizationError, match="identity is missing"):
        _validate_replaceable_v2_output(missing_identity, current)

    v1_identity = _identity().to_dict()
    non_v2 = tmp_path / "v1"
    non_v2.mkdir()
    (non_v2 / "manifest.json").write_text(json.dumps({"run_identity": v1_identity}))
    with pytest.raises(LabelFinalizationError, match="not a V2"):
        _validate_replaceable_v2_output(non_v2, current)

    mismatch_identity = dict(current)
    mismatch_identity["sampling_target"] = 200
    mismatch_identity["source_commit"] = "f" * 40
    mismatch = tmp_path / "mismatch"
    mismatch.mkdir()
    (mismatch / "manifest.json").write_text(
        json.dumps({"run_identity": mismatch_identity})
    )
    with pytest.raises(LabelFinalizationError, match="does not match"):
        _validate_replaceable_v2_output(mismatch, current)

    same_target = tmp_path / "same-target"
    same_target.mkdir()
    previous = dict(current)
    previous["sampling_target"] = 400
    (same_target / "manifest.json").write_text(json.dumps({"run_identity": previous}))
    with pytest.raises(LabelFinalizationError, match="not expandable"):
        _validate_replaceable_v2_output(same_target, current)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    previous["sampling_target"] = 200
    (incomplete / "manifest.json").write_text(json.dumps({"run_identity": previous}))
    with pytest.raises(LabelFinalizationError, match="failed validation"):
        _validate_replaceable_v2_output(incomplete, current)

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(LabelFinalizationError, match="regular V2"):
        _validate_replaceable_v2_output(symlink, current)


def test_v1_finalization_never_replaces_an_existing_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    _input(input_path)
    output = tmp_path / "publication"
    output.mkdir()
    with pytest.raises(LabelFinalizationError, match="only be replaced by a V2"):
        finalize_labeled_dataset(
            input_path=input_path,
            store=_store(tmp_path / "work", input_path),
            output_dir=output,
            dataset_repo_id="owner/dataset",
        )


def test_finalization_rejects_a_dangling_output_symlink(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    _input(input_path)
    output = tmp_path / "publication"
    output.symlink_to(tmp_path / "missing-publication", target_is_directory=True)
    with pytest.raises(LabelFinalizationError, match="existing output"):
        finalize_labeled_dataset(
            input_path=input_path,
            store=_store(tmp_path / "work", input_path),
            output_dir=output,
            dataset_repo_id="owner/dataset",
        )


def test_finalization_rejects_input_hash_drift(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    _input(input_path)
    store = _store(tmp_path / "work", input_path)
    input_path.write_bytes(input_path.read_bytes() + b"drift")
    with pytest.raises(LabelFinalizationError, match="SHA-256"):
        finalize_labeled_dataset(
            input_path=input_path,
            store=store,
            output_dir=tmp_path / "publication",
            dataset_repo_id="owner/dataset",
        )


def test_finalizes_exact_deterministic_canary_subset(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sentence_id": f"s{index}",
                    "region": "afghanistan",
                    "language": language,
                    "source": source,
                    "sentence_text_raw": "text",
                }
                for index, (language, source) in enumerate(
                    [
                        ("en", "wikipedia"),
                        ("fa", "wikipedia"),
                        ("ps", "wikipedia"),
                        ("fr", "wikivoyage"),
                    ]
                )
            ]
        ),
        input_path,
    )
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    identity = RunIdentity(
        input_sha256=digest,
        input_dataset_revision="b" * 40,
        model_repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        model_revision="c" * 40,
        model_file="Qwen3.6-27B-Q4_K_M.gguf",
        model_file_sha256="d" * 64,
        prompt_version="afghanistan-landuse-polygon-v2",
        source_commit="e" * 40,
        engine="llama.cpp",
        engine_version="1",
        batch_size=2,
        row_limit=2,
        llama_parallel=16,
        llama_per_slot_context=4096,
        llama_total_context=65536,
        request_concurrency=16,
    )
    store = CheckpointStore(tmp_path / "work", identity)
    selected = select_canary_rows(pq.read_table(input_path), 2)
    store.write_batch(
        0,
        [
            LabelRecord(
                sentence_id,
                LabelValue.NO,
                LabelValue.YES,
                "no_landuse_or_cover",
                "direct_polygon_reference",
                "text",
            )
            for sentence_id in selected["sentence_id"].to_pylist()
        ],
    )
    store.write_timing(
        {
            "total_wall_seconds": 1.0,
            "initial_inference_seconds": 0.5,
            "repair_inference_seconds": 0.0,
            "inference_seconds": 0.5,
            "checkpoint_and_validation_seconds": 0.5,
        }
    )

    result = finalize_labeled_dataset(
        input_path=input_path,
        store=store,
        output_dir=tmp_path / "out",
        dataset_repo_id="owner/dataset",
    )

    assert result.row_count == 2
    assert (
        "representative **2-row canary**"
        in (result.directory / "README.md").read_text()
    )


def test_publication_is_one_commit_and_includes_all_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    _input(input_path)
    output = tmp_path / "publication"
    finalize_labeled_dataset(
        input_path=input_path,
        store=_store(tmp_path / "work", input_path),
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )
    calls: list[dict[str, object]] = []
    readbacks: list[tuple[str, str]] = []

    class Api:
        def list_repo_files(self, **kwargs: object) -> list[str]:
            return [".gitattributes"]

        def create_commit(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(oid="f" * 40, commit_url="https://example/commit")

    result = publish_labeled_dataset(
        output,
        "owner/dataset",
        hub_api=Api(),
        operation_factory=lambda **kwargs: kwargs,
        readback_downloader=lambda dataset_id, revision: (
            readbacks.append((dataset_id, revision))
            or _v1_remote_snapshot(tmp_path, output)
        ),
    )
    assert result.commit_id == "f" * 40
    assert len(calls) == 1
    assert readbacks == [("owner/dataset", "f" * 40)]
    paths = {op["path_in_repo"] for op in calls[0]["operations"]}  # type: ignore[index]
    assert paths == {
        "v1-afghanistan/sentences.parquet",
        "v1-afghanistan/manifest.json",
        "v1-afghanistan/README.md",
        "v1-afghanistan/assets/label_distribution.png",
        "v1-afghanistan/assets/positive_languages.png",
        "v1-afghanistan/assets/joint_label_heatmap.png",
        "v1-afghanistan/assets/polygon_coverage_funnel.png",
        "v1-afghanistan/assets/reason_code_distribution.png",
    }


def test_publication_rejects_invalid_independent_readback(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    _input(input_path)
    output = tmp_path / "publication"
    finalize_labeled_dataset(
        input_path=input_path,
        store=_store(tmp_path / "work", input_path),
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )
    invalid_readback = tmp_path / "readback"
    invalid_readback.mkdir()

    class Api:
        def list_repo_files(self, **kwargs: object) -> list[str]:
            return []

        def create_commit(self, **kwargs: object) -> object:
            return SimpleNamespace(oid="f" * 40, commit_url="https://example/commit")

    with pytest.raises(LabelPublicationError, match="readback validation failed"):
        publish_labeled_dataset(
            output,
            "owner/dataset",
            hub_api=Api(),
            operation_factory=lambda **kwargs: kwargs,
            readback_downloader=lambda dataset_id, revision: invalid_readback,
        )


def test_publication_default_hub_integration_verifies_exact_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.parquet"
    _input(input_path)
    output = tmp_path / "publication"
    finalize_labeled_dataset(
        input_path=input_path,
        store=_store(tmp_path / "work", input_path),
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )
    downloads: list[dict[str, object]] = []

    class Api:
        def list_repo_files(self, **kwargs: object) -> list[str]:
            return [".gitattributes"]

        def create_commit(self, **kwargs: object) -> object:
            return SimpleNamespace(oid="f" * 40, commit_url="https://example/commit")

    def snapshot_download(**kwargs: object) -> str:
        downloads.append(kwargs)
        return str(_v1_remote_snapshot(tmp_path, output))

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = Api  # type: ignore[attr-defined]
    fake_hub.CommitOperationAdd = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    fake_hub.CommitOperationDelete = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    fake_hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    result = publish_labeled_dataset(output, "owner/dataset")

    assert result.commit_id == "f" * 40
    assert downloads == [
        {
            "repo_id": "owner/dataset",
            "repo_type": "dataset",
            "revision": "f" * 40,
            "allow_patterns": [
                "v1-afghanistan/sentences.parquet",
                "v1-afghanistan/manifest.json",
                "v1-afghanistan/README.md",
                "v1-afghanistan/assets/label_distribution.png",
                "v1-afghanistan/assets/positive_languages.png",
                "v1-afghanistan/assets/joint_label_heatmap.png",
                "v1-afghanistan/assets/polygon_coverage_funnel.png",
                "v1-afghanistan/assets/reason_code_distribution.png",
                "README.md",
                ".gitattributes",
            ],
        }
    ]


def test_publication_rejects_blank_target_and_remote_failures(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    _input(input_path)
    output = tmp_path / "publication"
    finalize_labeled_dataset(
        input_path=input_path,
        store=_store(tmp_path / "work", input_path),
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )
    with pytest.raises(LabelPublicationError, match="non-blank"):
        publish_labeled_dataset(
            output, " ", hub_api=object(), operation_factory=lambda **kwargs: kwargs
        )

    class FailingApi:
        def list_repo_files(self, **kwargs: object) -> list[str]:
            return []

        def create_commit(self, **kwargs: object) -> object:
            raise OSError("remote failed")

    with pytest.raises(LabelPublicationError, match="publication failed"):
        publish_labeled_dataset(
            output,
            "owner/dataset",
            hub_api=FailingApi(),
            operation_factory=lambda **kwargs: kwargs,
        )


def test_validator_rejects_layout_hash_and_statistics_tampering(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    _input(input_path)
    output = tmp_path / "publication"
    finalize_labeled_dataset(
        input_path=input_path,
        store=_store(tmp_path / "work", input_path),
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )
    extra = output / "debug.txt"
    extra.write_text("x")
    with pytest.raises(LabelFinalizationError, match="layout"):
        validate_labeled_publication(output)
    extra.unlink()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parquet_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(LabelFinalizationError, match="SHA-256"):
        validate_labeled_publication(output)


def test_validator_rejects_tampered_plot(tmp_path: Path) -> None:
    input_path = tmp_path / "input.parquet"
    _input(input_path)
    output = tmp_path / "publication"
    finalize_labeled_dataset(
        input_path=input_path,
        store=_store(tmp_path / "work", input_path),
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )
    plot = output / "assets" / "label_distribution.png"
    plot.write_bytes(plot.read_bytes() + b"tamper")
    with pytest.raises(LabelFinalizationError, match="artifact SHA-256"):
        validate_labeled_publication(output)
