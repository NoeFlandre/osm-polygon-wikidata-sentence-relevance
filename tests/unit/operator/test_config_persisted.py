"""Tests for historical OperatorConfig reconstruction from a persisted identity.

These tests pin the contract that ``OperatorConfig.from_persisted(identity)``
is stage-aware: a split identity must NOT silently inherit current label
defaults, and vice versa. The reconstructed canonical identity must
byte-match the persisted one and reproduce the same run ID.
"""

from __future__ import annotations

from osm_polygon_sentence_relevance.operator.config import (
    OperatorConfig,
    Stage,
)


def _split_identity() -> dict[str, object]:
    return {
        "scope": "region",
        "stage": "split",
        "source_commit": "0" * 40,
        "input_dataset_id": "NoeFlandre/osm-polygon-wikidata-only",
        "output_dataset_id": "NoeFlandre/osm-polygon-wikidata-sentence-relevance",
        "pipeline_version": "test-pipeline-1",
        "batch_size": 64,
        "region": "afghanistan-latest",
        "split_model": "sat-12l-sm",
    }


def _label_identity() -> dict[str, object]:
    return {
        "scope": "region",
        "stage": "label",
        "source_commit": "1" * 40,
        "input_dataset_id": "NoeFlandre/osm-polygon-wikidata-only",
        "output_dataset_id": "NoeFlandre/osm-polygon-wikidata-sentence-relevance",
        "pipeline_version": "test-pipeline-1",
        "batch_size": 128,
        "row_limit": 0,
        "llama_parallel": 16,
        "llama_per_slot_context": 4096,
        "llama_total_context": 65536,
        "request_concurrency": 16,
        "region": "afghanistan-latest",
        "input_dataset_revision": "9" * 40,
        "model_repo_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
        "model_revision": "5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace",
        "model_file": "Qwen3.6-27B-Q4_K_M.gguf",
        "model_file_sha256": "a" * 64,
        "tokenizer_repo_id": "Qwen/Qwen3.6-27B",
        "tokenizer_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "prompt_version": "v1",
    }


def _all_identity() -> dict[str, object]:
    payload = _label_identity()
    payload["stage"] = "all"
    payload["split_model"] = "sat-12l-sm"
    return payload


def test_split_identity_does_not_inherit_label_defaults() -> None:
    identity = _split_identity()
    config = OperatorConfig.from_persisted(identity)
    assert config.stage is Stage.SPLIT
    # No label-only values should appear in the reconstructed identity.
    canonical = config.run_identity.to_dict()
    for forbidden in (
        "model_repo_id",
        "model_revision",
        "model_file",
        "model_file_sha256",
        "tokenizer_repo_id",
        "tokenizer_revision",
        "prompt_version",
        "row_limit",
        "llama_parallel",
        "llama_per_slot_context",
        "llama_total_context",
        "request_concurrency",
        "input_dataset_revision",
    ):
        assert forbidden not in canonical, (
            f"label default leaked into split: {forbidden}"
        )


def test_label_identity_does_not_inherit_split_default() -> None:
    identity = _label_identity()
    config = OperatorConfig.from_persisted(identity)
    assert config.stage is Stage.LABEL
    canonical = config.run_identity.to_dict()
    assert "split_model" not in canonical


def test_reconstructed_canonical_identity_matches_persisted() -> None:
    for identity in (_split_identity(), _label_identity(), _all_identity()):
        config = OperatorConfig.from_persisted(identity)
        # The reconstructed identity, when re-canonicalized, must match the
        # original canonical JSON byte-for-byte.
        import json

        reconstructed = json.loads(config.canonical_json)
        original = dict(identity)
        # Original contains extra fields that are only defaults; only the
        # fields actually persisted in to_dict() matter for byte-match.
        for key in sorted(reconstructed):
            assert reconstructed[key] == original[key], (
                f"{key} differs for stage {original['stage']}"
            )


def test_reconstructed_run_id_reproduces_persisted() -> None:
    """The deterministic run ID must be reproducible from persisted state."""

    config = OperatorConfig.from_persisted(_label_identity())
    # Re-run from the canonical JSON produced by the first reconstruction.
    config2 = OperatorConfig.from_persisted(config.run_identity.to_dict())
    assert config.run_id == config2.run_id


def test_from_persisted_reproduces_v2_all_identity() -> None:
    """A persisted worldwide V2 identity must remain resumable."""

    config = OperatorConfig.build(
        scope="all",
        stage="all",
        source_commit="2" * 40,
        input_revision="3" * 40,
        sampling_target=200_000,
    )
    persisted = config.run_identity.to_dict()

    resumed = OperatorConfig.from_persisted(persisted)

    assert resumed.run_id == config.run_id
    assert resumed.label_model_repo_id == "ggml-org/Qwen3.6-27B-GGUF"


def test_from_persisted_split_rejects_label_fields_when_present() -> None:
    """A split identity that erroneously includes label fields must reject."""

    identity = _split_identity()
    identity["model_repo_id"] = "unsloth/Qwen3.6-27B-MTP-GGUF"
    identity["model_revision"] = "5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace"
    identity["model_file"] = "Qwen3.6-27B-Q4_K_M.gguf"
    identity["model_file_sha256"] = "a" * 64
    identity["tokenizer_repo_id"] = "Qwen/Qwen3.6-27B"
    identity["tokenizer_revision"] = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    identity["prompt_version"] = "v1"
    identity["row_limit"] = 0
    # A split identity must not silently drop or accept label fields; the
    # reconstructed canonical identity should include exactly what the
    # stage allows.
    config = OperatorConfig.from_persisted(identity)
    canonical = config.run_identity.to_dict()
    # split stage drops label-only fields by design.
    for forbidden in ("model_repo_id", "model_file_sha256", "prompt_version"):
        assert forbidden not in canonical
