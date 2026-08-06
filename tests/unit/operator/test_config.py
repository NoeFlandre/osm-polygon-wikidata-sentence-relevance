"""Strict TDD coverage for operator configuration contracts."""

from __future__ import annotations

import json
import re

import pytest

from osm_polygon_sentence_relevance.labeling.runtime import (
    SUPPORTED_LLAMA_PARALLEL as RUNTIME_SUPPORTED_LLAMA_PARALLEL,
)
from osm_polygon_sentence_relevance.operator import (
    DATA_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LABEL_MODEL_FILE,
    DEFAULT_LABEL_MODEL_FILE_SHA256,
    DEFAULT_LABEL_MODEL_REPO_ID,
    DEFAULT_ROW_LIMIT,
    DEFAULT_TOKENIZER_REPO_ID,
    DEFAULT_TOKENIZER_REVISION,
    INPUT_DATASET_ID,
    OUTPUT_DATASET_ID,
    SUPPORTED_LLAMA_PARALLEL,
    OperatorConfig,
    Scope,
    Stage,
)
from osm_polygon_sentence_relevance.operator.config import (
    DEFAULT_SPLIT_MODEL,
    PROMPT_VERSION,
    Grid5000Requirements,
)


def _base_kwargs() -> dict[str, object]:
    return {
        "scope": "region",
        "region": "afghanistan-latest",
        "stage": "all",
        "source_commit": "a" * 40,
        "input_revision": "b" * 40,
        "batch_size": DEFAULT_BATCH_SIZE,
        "row_limit": DEFAULT_ROW_LIMIT,
    }


def test_split_identity_ignores_labeling_runtime_only_settings() -> None:
    base = {
        "scope": "region",
        "region": "afghanistan-latest",
        "stage": "split",
        "source_commit": "a" * 40,
        "input_revision": "b" * 40,
    }
    first = OperatorConfig.build(**base)
    second = OperatorConfig.build(
        **base,
        row_limit=128,
        llama_parallel=16,
        llama_per_slot_context=4096,
        llama_total_context=65536,
        request_concurrency=16,
        model_repo_id=DEFAULT_LABEL_MODEL_REPO_ID,
        model_revision="c" * 40,
        model_file=DEFAULT_LABEL_MODEL_FILE,
        model_file_sha256=DEFAULT_LABEL_MODEL_FILE_SHA256,
        tokenizer_repo_id=DEFAULT_TOKENIZER_REPO_ID,
        tokenizer_revision=DEFAULT_TOKENIZER_REVISION,
        prompt_version="afghanistan-landuse-polygon-v2",
    )
    assert first.run_id == second.run_id


def test_label_identity_ignores_split_model() -> None:
    base = {
        "scope": "region",
        "region": "afghanistan-latest",
        "stage": "label",
        "source_commit": "a" * 40,
        "input_revision": "b" * 40,
    }
    first = OperatorConfig.build(**base)
    second = OperatorConfig.build(**base, split_model="sat-12l-sm-alt")
    assert first.run_id == second.run_id


def test_split_canonical_json_excludes_label_fields() -> None:
    config = OperatorConfig.build(
        scope=Scope.REGION,
        region="afghanistan-latest",
        stage=Stage.SPLIT,
        source_commit="a" * 40,
        input_revision="b" * 40,
    )
    payload = json.loads(config.canonical_json)
    assert "row_limit" not in payload
    assert "llama_parallel" not in payload
    assert "llama_per_slot_context" not in payload
    assert "llama_total_context" not in payload
    assert "request_concurrency" not in payload
    assert "model_repo_id" not in payload
    assert "model_revision" not in payload
    assert "model_file" not in payload
    assert "model_file_sha256" not in payload
    assert "tokenizer_repo_id" not in payload
    assert "tokenizer_revision" not in payload
    assert "prompt_version" not in payload


def test_label_canonical_json_excludes_split_field() -> None:
    config = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )
    payload = json.loads(config.canonical_json)
    assert "split_model" not in payload


def test_all_identity_changes_for_split_and_label_fields() -> None:
    base = OperatorConfig.build(**_base_kwargs())
    variants: list[dict[str, object]] = [
        {"source_commit": "b" * 40},
        {"input_revision": "c" * 40},
        {"split_model": "sat-12l-sm-alt"},
        {"model_revision": "c" * 40},
        {"prompt_version": "afghanistan-landuse-polygon-v3"},
        {"batch_size": 256},
        {"row_limit": 128},
        {"llama_parallel": 8},
        {"request_concurrency": 8},
        {"llama_parallel": 8, "llama_total_context": 32768},
    ]
    for patch in variants:
        args = _base_kwargs()
        args["stage"] = "all"
        args.update(patch)
        candidate = OperatorConfig.build(**args)
        assert candidate.run_id != base.run_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_commit", " " + ("a" * 40)),
        ("input_revision", " " + ("b" * 40)),
        ("pipeline_version", " 0.1.0"),
        ("split_model", f" {DEFAULT_SPLIT_MODEL} "),
        ("model_repo_id", f" {DEFAULT_LABEL_MODEL_REPO_ID} "),
        ("model_file", f" {DEFAULT_LABEL_MODEL_FILE} "),
        ("tokenizer_repo_id", f" {DEFAULT_TOKENIZER_REPO_ID} "),
        ("prompt_version", f" {PROMPT_VERSION} "),
        ("region", " afghanistan-latest"),
    ],
)
def test_surrounding_whitespace_is_rejected(field: str, value: str) -> None:
    kwargs = _base_kwargs()
    kwargs["stage"] = "label"
    if field in ("source_commit", "input_revision"):
        kwargs[field] = value
    elif field == "region":
        kwargs["region"] = value
    elif field == "split_model":
        kwargs["split_model"] = value
    elif field == "model_repo_id":
        kwargs["model_repo_id"] = value
    elif field == "model_file":
        kwargs["model_file"] = value
    elif field == "tokenizer_repo_id":
        kwargs["tokenizer_repo_id"] = value
    elif field == "pipeline_version":
        kwargs["pipeline_version"] = value
    else:
        kwargs["prompt_version"] = value
    with pytest.raises(
        ValueError, match=r"whitespace|blank|must be|invalid scope|invalid stage"
    ):
        OperatorConfig.build(**kwargs)


@pytest.mark.parametrize(
    "region",
    [
        "afghanistan",
        "latest",
        "Afghanistan-latest",
        "afghanistan_latest",
        "afghanistan/latest",
        "../afghanistan-latest",
        "afghanistan.later",
        "../afghanistan-latest/",
    ],
)
def test_region_requires_canonical_latest(region: str) -> None:
    with pytest.raises(ValueError, match=r"region"):
        OperatorConfig.build(
            scope="region",
            region=region,
            stage="split",
            source_commit="a" * 40,
        )


@pytest.mark.parametrize("region", ["afghanistan-latest", "united-states-latest"])
def test_region_grammar_allows_multi_component_latest_shards(region: str) -> None:
    config = OperatorConfig.build(
        scope=Scope.REGION,
        region=region,
        stage=Stage.SPLIT,
        source_commit="a" * 40,
        input_revision="b" * 40,
    )
    assert config.region == region


@pytest.mark.parametrize(
    "filename",
    [
        "../model.gguf",
        "dir/model.gguf",
        r"dir\\model.gguf",
        "model/../model.gguf",
        "../../model.gguf",
        "..",
        "./model.gguf",
        "model\0.gguf",
    ],
)
def test_unsafe_model_filenames_rejected(filename: str) -> None:
    with pytest.raises(ValueError, match=r"safe filename"):
        OperatorConfig.build(
            scope="region",
            region="afghanistan-latest",
            stage="label",
            source_commit="a" * 40,
            model_file=filename,
            input_revision="b" * 40,
        )


def test_safe_model_filename_rejected_without_repository_dependency() -> None:
    with pytest.raises(ValueError, match=r"safe filename"):
        OperatorConfig.build(
            scope="region",
            region="afghanistan-latest",
            stage="label",
            source_commit="a" * 40,
            input_revision="b" * 40,
            model_file="../unsafe.gguf",
        )


def test_invalid_input_dataset_ids_rejected() -> None:
    for bad_value in [
        "NoeFlandre repo/osm-polygon-wikidata-only",
        "NoeFlandre/osm-polygon-wikidata-only/extra",
        "NoeFlandre/ .bad",
    ]:
        with pytest.raises(
            ValueError, match=r"expected repository|owner/name|whitespace"
        ):
            OperatorConfig.build(
                scope="region",
                region="afghanistan-latest",
                stage="label",
                source_commit="a" * 40,
                input_revision="b" * 40,
                input_dataset_id=bad_value,
            )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model_repo_id", "NoeFlandre/WrongRepo"),
        ("tokenizer_repo_id", "NoeFlandre/wrong"),
    ],
)
def test_invalid_model_and_tokenizer_repo_ids_rejected(
    field: str,
    bad_value: str,
) -> None:
    kwargs = _base_kwargs()
    kwargs["stage"] = "label"
    kwargs[field] = bad_value
    with pytest.raises(ValueError, match=r"expected repository|owner/name"):
        OperatorConfig.build(**kwargs)


def test_invalid_repo_id_with_forced_ownercase() -> None:
    with pytest.raises(ValueError, match=r"owner/name"):
        OperatorConfig.build(
            scope="region",
            region="afghanistan-latest",
            stage="label",
            source_commit="a" * 40,
            input_revision="b" * 40,
            model_repo_id="../bad/repo",
        )


def test_invalid_scope_rejected() -> None:
    for scope in ["region_only", "", "  ", None]:
        with pytest.raises(ValueError, match=r"invalid scope|must be a string|scope"):
            OperatorConfig.build(
                scope=scope,  # type: ignore[arg-type]
                stage="label",
                source_commit="a" * 40,
                region="afghanistan-latest",
                input_revision="b" * 40,
            )


def test_invalid_stage_rejected() -> None:
    for stage in ["label_only", "", "  ", None]:
        with pytest.raises(ValueError, match=r"invalid stage|must be a string|stage"):
            OperatorConfig.build(
                scope="region",
                stage=stage,  # type: ignore[arg-type]
                source_commit="a" * 40,
                region="afghanistan-latest",
                input_revision="b" * 40,
            )


def test_scope_and_region_relationship_validation() -> None:
    with pytest.raises(ValueError, match=r"region"):
        OperatorConfig.build(
            scope="region",
            stage="split",
            source_commit="a" * 40,
            input_revision="b" * 40,
            region=None,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match=r"region"):
        OperatorConfig.build(
            scope="all",
            stage="split",
            source_commit="a" * 40,
            input_revision="b" * 40,
            region="afghanistan-latest",
        )


def test_invalid_revision_and_sha_validation() -> None:
    with pytest.raises(ValueError, match=r"lowercase hexadecimal"):
        OperatorConfig.build(
            scope="region",
            region="afghanistan-latest",
            stage="split",
            source_commit="A" * 40,
            input_revision="b" * 40,
        )

    with pytest.raises(
        ValueError, match=r"lowercase hexadecimal|exactly 40 characters"
    ):
        OperatorConfig.build(
            scope="region",
            region="afghanistan-latest",
            stage="split",
            source_commit="a" * 39,
            input_revision="b" * 40,
        )


def test_numeric_validation_branches() -> None:
    kwargs = {
        "scope": "region",
        "region": "afghanistan-latest",
        "stage": "label",
        "source_commit": "a" * 40,
        "input_revision": "b" * 40,
    }
    with pytest.raises(ValueError, match=r"positive integer"):
        OperatorConfig.build(**{**kwargs, "batch_size": 0})
    with pytest.raises(ValueError, match=r"non-negative"):
        OperatorConfig.build(**{**kwargs, "row_limit": -1})
    with pytest.raises(ValueError, match=r"leading zeroes"):
        OperatorConfig.build(**{**kwargs, "row_limit": "000"})
    with pytest.raises(ValueError, match=r"must be one of"):
        OperatorConfig.build(**{**kwargs, "llama_parallel": 3})
    with pytest.raises(ValueError, match=r"between 1 and the parallel slot count"):
        OperatorConfig.build(
            **{**kwargs, "llama_parallel": 8, "request_concurrency": 16}
        )
    with pytest.raises(ValueError, match=r"integer"):
        OperatorConfig.build(**{**kwargs, "request_concurrency": True})


def test_requirements_builder_rejects_type_and_mismatch() -> None:
    requirements = Grid5000Requirements.build(
        row_limit=10,
        llama_parallel=8,
        llama_per_slot_context=4096,
        request_concurrency=4,
    )
    assert requirements.llama_total_context == 32768

    with pytest.raises(ValueError, match=r"llama_total_context"):
        Grid5000Requirements.build(
            llama_total_context=100,
            llama_parallel=8,
            llama_per_slot_context=4096,
        )

    with pytest.raises(ValueError, match=r"between 1 and the parallel slot count"):
        Grid5000Requirements.build(
            llama_parallel=4,
            llama_per_slot_context=4096,
            request_concurrency=8,
        )

    with pytest.raises(ValueError, match=r"positive integer|integer"):
        Grid5000Requirements.build(llama_parallel=True)


def test_requirements_minimum_and_default_total_context() -> None:
    with pytest.raises(ValueError, match=r"at least 4096"):
        Grid5000Requirements.build(llama_per_slot_context=4095)

    assert (
        Grid5000Requirements.build(llama_per_slot_context=4096).llama_per_slot_context
        == 4096
    )
    assert (
        Grid5000Requirements.build(llama_per_slot_context=8192).llama_per_slot_context
        == 8192
    )

    requirements = Grid5000Requirements.build(
        llama_parallel=4, llama_per_slot_context=8192
    )
    assert requirements.llama_total_context == 32768

    with pytest.raises(
        ValueError, match=r"llama_total_context must equal llama_parallel"
    ):
        Grid5000Requirements.build(
            llama_parallel=8,
            llama_per_slot_context=4096,
            llama_total_context=10000,
        )


def test_supported_parallel_values_follow_labeling_runtime_contract() -> None:
    assert SUPPORTED_LLAMA_PARALLEL == RUNTIME_SUPPORTED_LLAMA_PARALLEL


def test_direct_requirements_construction_rejects_invalid_values() -> None:
    base = {
        "batch_size": 128,
        "row_limit": 0,
        "llama_parallel": 16,
        "llama_per_slot_context": 4096,
        "llama_total_context": 65536,
    }
    invalid_values = [
        ({"batch_size": 0}, r"must be a positive integer"),
        ({"row_limit": -1}, r"must be a non-negative integer"),
        ({"llama_parallel": 3}, r"must be one of"),
        ({"llama_per_slot_context": 4095}, r"at least 4096"),
        (
            {"llama_total_context": 1},
            r"must equal llama_parallel \* llama_per_slot_context",
        ),
        ({"request_concurrency": 0}, r"must be a positive integer"),
        ({"request_concurrency": 17}, r"between 1 and the parallel slot count"),
    ]
    for kwargs, message in invalid_values:
        candidate = base.copy()
        candidate.update(kwargs)
        with pytest.raises(ValueError, match=message):
            Grid5000Requirements(**candidate)  # type: ignore[arg-type]


def test_cov_branches_for_primitive_validation_helpers() -> None:
    with pytest.raises(ValueError, match=r"must be a string"):
        OperatorConfig.build(
            scope="region",
            region="afghanistan-latest",
            stage="label",
            source_commit=12,  # type: ignore[arg-type]
            input_revision="b" * 40,
        )

    with pytest.raises(ValueError, match=r"must be"):
        Grid5000Requirements.build(batch_size=1.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"cannot use leading zeroes"):
        OperatorConfig.build(
            scope="region",
            region="afghanistan-latest",
            stage="label",
            source_commit="a" * 40,
            input_revision="b" * 40,
            batch_size="012",
        )


def test_fixed_data_root_and_repos_are_bound() -> None:
    config = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="split",
        source_commit="a" * 40,
    )
    assert config.data_root == DATA_ROOT
    assert config.input_dataset_id == INPUT_DATASET_ID
    assert config.output_dataset_id == OUTPUT_DATASET_ID


def test_canonical_json_stable_despite_mapping_order() -> None:
    args_a = {
        "source_commit": "a" * 40,
        "input_revision": "b" * 40,
        "scope": "region",
        "region": "afghanistan-latest",
        "stage": "label",
    }
    args_b = {
        "region": "afghanistan-latest",
        "stage": "label",
        "source_commit": "a" * 40,
        "input_revision": "b" * 40,
        "scope": "region",
    }
    first = OperatorConfig.build(**args_a)
    second = OperatorConfig.build(**args_b)
    assert first.canonical_json == second.canonical_json
    assert first.run_id == second.run_id


def test_v2_target_expansion_reuses_the_same_immutable_run_identity() -> None:
    base = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="a" * 40,
        input_revision="b" * 40,
        sampling_target=200_000,
    )
    expanded = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="label",
        source_commit="a" * 40,
        input_revision="b" * 40,
        sampling_target=400_000,
    )
    assert base.run_id == expanded.run_id
    assert (
        base.run_identity.checkpoint_dict() == expanded.run_identity.checkpoint_dict()
    )
    assert base.run_identity.to_dict()["sampling_target"] == 200_000
    assert expanded.run_identity.to_dict()["sampling_target"] == 400_000
    resumed = OperatorConfig.from_persisted(
        {**base.run_identity.checkpoint_dict(), "sampling_target": 400_000}
    )
    assert resumed.run_id == base.run_id
    assert resumed.requirements.sampling_target == 400_000


def test_run_id_is_twenty_lowercase_hex_chars() -> None:
    config = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="split",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )
    assert re.fullmatch(r"[0-9a-f]{20}", config.run_id) is not None
    assert len(config.run_id) == 20


def test_public_exports_are_stable() -> None:
    import osm_polygon_sentence_relevance.operator as operator

    expected = {
        "DATA_ROOT",
        "INPUT_DATASET_ID",
        "OUTPUT_DATASET_ID",
        "Scope",
        "Stage",
        "Grid5000Requirements",
        "RunIdentity",
        "OperatorConfig",
    }
    assert expected.issubset(set(operator.__all__))


def test_no_filesystem_side_effects(tmp_path) -> None:
    before = sorted(entry.name for entry in tmp_path.iterdir())
    OperatorConfig.build(**_base_kwargs())
    after = sorted(entry.name for entry in tmp_path.iterdir())
    assert before == after


def test_default_row_limit_is_preserved_for_split_and_excluded_from_identity() -> None:
    defaulted = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="split",
        source_commit="a" * 40,
        input_revision="b" * 40,
    )
    assert defaulted.run_identity.row_limit is None
    assert defaulted.requirements.row_limit == DEFAULT_ROW_LIMIT
