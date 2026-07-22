from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
from osm_polygon_sentence_relevance.labeling.contracts import (
    LabelRecord,
    LabelValue,
    RunIdentity,
)
from osm_polygon_sentence_relevance.labeling.finalization import (
    LabelFinalizationError,
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
        engine="vllm",
        engine_version="0.21.0",
        batch_size=2,
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


def _store(path: Path, input_path: Path, *, complete: bool = True) -> CheckpointStore:
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    store = CheckpointStore(path, _identity(digest))
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
    store.write_timing({"total_wall_seconds": 12.5, "inference_seconds": 10.0})
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
        "sentences.parquet",
    }
    card = (output / "README.md").read_text()
    assert "3 labeled sentences" in card
    assert "2 (66.67%)" in card
    assert "12.50 seconds" in card
    assert "unsloth/Qwen3.6-27B-MTP-GGUF" in card
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


def test_refuses_partial_or_non_afghanistan_finalization(tmp_path: Path) -> None:
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
    with pytest.raises(LabelFinalizationError, match="Afghanistan"):
        finalize_labeled_dataset(
            input_path=input_path,
            store=_store(tmp_path / "complete", input_path),
            output_dir=tmp_path / "out2",
            dataset_repo_id="owner/dataset",
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
        def create_commit(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(oid="f" * 40, commit_url="https://example/commit")

    result = publish_labeled_dataset(
        output,
        "owner/dataset",
        hub_api=Api(),
        operation_factory=lambda **kwargs: kwargs,
        readback_downloader=lambda dataset_id, revision: (
            readbacks.append((dataset_id, revision)) or output
        ),
    )
    assert result.commit_id == "f" * 40
    assert len(calls) == 1
    assert readbacks == [("owner/dataset", "f" * 40)]
    paths = {op["path_in_repo"] for op in calls[0]["operations"]}  # type: ignore[index]
    assert paths == {
        "sentences.parquet",
        "manifest.json",
        "README.md",
        "assets/label_distribution.png",
        "assets/positive_languages.png",
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
        def create_commit(self, **kwargs: object) -> object:
            return SimpleNamespace(oid="f" * 40, commit_url="https://example/commit")

    def snapshot_download(**kwargs: object) -> str:
        downloads.append(kwargs)
        return str(output)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = Api  # type: ignore[attr-defined]
    fake_hub.CommitOperationAdd = lambda **kwargs: kwargs  # type: ignore[attr-defined]
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
                "sentences.parquet",
                "manifest.json",
                "README.md",
                "assets/label_distribution.png",
                "assets/positive_languages.png",
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
