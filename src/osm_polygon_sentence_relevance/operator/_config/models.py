"""Immutable operator configuration and run-identity contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from osm_polygon_sentence_relevance.contracts.constants import PIPELINE_VERSION
from osm_polygon_sentence_relevance.operator._config.defaults import (
    DATA_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LABEL_MODEL_FILE,
    DEFAULT_LABEL_MODEL_FILE_SHA256,
    DEFAULT_LABEL_MODEL_REPO_ID,
    DEFAULT_LABEL_MODEL_REVISION,
    DEFAULT_LLAMA_PARALLEL,
    DEFAULT_LLAMA_PER_SLOT_CONTEXT,
    DEFAULT_ROW_LIMIT,
    DEFAULT_SAMPLING_H3_RESOLUTION,
    DEFAULT_SAMPLING_SEED,
    DEFAULT_SAMPLING_TARGET,
    DEFAULT_SPLIT_MODEL,
    DEFAULT_TOKENIZER_REPO_ID,
    DEFAULT_TOKENIZER_REVISION,
    DEFAULT_V2_LABEL_MODEL_FILE,
    DEFAULT_V2_LABEL_MODEL_FILE_SHA256,
    DEFAULT_V2_LABEL_MODEL_REPO_ID,
    DEFAULT_V2_LABEL_MODEL_REVISION,
    INPUT_DATASET_ID,
    OUTPUT_DATASET_ID,
    PROMPT_VERSION,
    SAMPLING_VERSION,
    V2_LOGIT_PROMPT_VERSION,
    V2_SAMPLING_VERSION,
)
from osm_polygon_sentence_relevance.operator._config.enums import Scope, Stage
from osm_polygon_sentence_relevance.operator._config.identity import RunIdentity
from osm_polygon_sentence_relevance.operator._config.requirements import (
    Grid5000Requirements,
)
from osm_polygon_sentence_relevance.operator._config.validation import (
    canonicalize_scope,
    canonicalize_stage,
    require_run_fields_for_scope,
    validate_hex,
    validate_model_file,
    validate_nonblank_no_ws,
    validate_repo_id,
)


@dataclass(frozen=True, slots=True)
class OperatorConfig:
    """Validated production configuration for one operator run."""

    scope: Scope
    stage: Stage
    source_commit: str
    requirements: Grid5000Requirements
    # The checkout used to execute the pipeline.  This is deliberately not
    # part of RunIdentity: source_commit identifies the data contract, while
    # execution_commit may advance for behavior-preserving runtime fixes.
    execution_commit: str | None = None
    data_root: Path = DATA_ROOT
    input_dataset_id: str = INPUT_DATASET_ID
    output_dataset_id: str = OUTPUT_DATASET_ID
    pipeline_version: str = PIPELINE_VERSION
    input_dataset_revision: str | None = None
    region: str | None = None
    split_model: str = DEFAULT_SPLIT_MODEL
    label_model_repo_id: str = DEFAULT_LABEL_MODEL_REPO_ID
    label_model_revision: str = DEFAULT_LABEL_MODEL_REVISION
    label_model_file: str = DEFAULT_LABEL_MODEL_FILE
    label_model_file_sha256: str = DEFAULT_LABEL_MODEL_FILE_SHA256
    tokenizer_repo_id: str = DEFAULT_TOKENIZER_REPO_ID
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION
    prompt_version: str = PROMPT_VERSION
    run_identity: RunIdentity = field(init=False)

    @classmethod
    def build(
        cls,
        *,
        scope: Scope | str,
        stage: Stage | str,
        source_commit: str,
        batch_size: int | str = DEFAULT_BATCH_SIZE,
        row_limit: int | str = DEFAULT_ROW_LIMIT,
        llama_parallel: int | str = DEFAULT_LLAMA_PARALLEL,
        llama_per_slot_context: int | str = DEFAULT_LLAMA_PER_SLOT_CONTEXT,
        llama_total_context: int | str | None = None,
        request_concurrency: int | str | None = None,
        sampling_target: int | str | None = None,
        sampling_seed: str = DEFAULT_SAMPLING_SEED,
        sampling_h3_resolution: int = DEFAULT_SAMPLING_H3_RESOLUTION,
        sampling_version: str | None = None,
        region: str | None = None,
        input_revision: str | None = None,
        input_dataset_id: str = INPUT_DATASET_ID,
        output_dataset_id: str = OUTPUT_DATASET_ID,
        pipeline_version: str = PIPELINE_VERSION,
        split_model: str = DEFAULT_SPLIT_MODEL,
        model_repo_id: str | None = None,
        model_revision: str | None = None,
        model_file: str | None = None,
        model_file_sha256: str | None = None,
        tokenizer_repo_id: str | None = None,
        tokenizer_revision: str | None = None,
        prompt_version: str | None = None,
    ) -> OperatorConfig:
        canonical_scope = canonicalize_scope(scope)
        canonical_stage = canonicalize_stage(stage)
        canonical_region = require_run_fields_for_scope(
            canonical_scope, canonical_stage, region
        )

        v2 = canonical_scope is Scope.ALL and (
            prompt_version is None or prompt_version == V2_LOGIT_PROMPT_VERSION
        )
        effective_model_repo_id = model_repo_id or (
            DEFAULT_V2_LABEL_MODEL_REPO_ID if v2 else DEFAULT_LABEL_MODEL_REPO_ID
        )
        effective_model_revision = model_revision or (
            DEFAULT_V2_LABEL_MODEL_REVISION if v2 else DEFAULT_LABEL_MODEL_REVISION
        )
        effective_model_file = model_file or (
            DEFAULT_V2_LABEL_MODEL_FILE if v2 else DEFAULT_LABEL_MODEL_FILE
        )
        effective_model_file_sha256 = model_file_sha256 or (
            DEFAULT_V2_LABEL_MODEL_FILE_SHA256
            if v2
            else DEFAULT_LABEL_MODEL_FILE_SHA256
        )
        effective_tokenizer_repo_id = tokenizer_repo_id or DEFAULT_TOKENIZER_REPO_ID
        effective_tokenizer_revision = tokenizer_revision or DEFAULT_TOKENIZER_REVISION
        effective_prompt_version = (
            V2_LOGIT_PROMPT_VERSION if v2 else prompt_version or PROMPT_VERSION
        )
        effective_sampling_version = sampling_version or (
            V2_SAMPLING_VERSION if v2 else SAMPLING_VERSION
        )
        validated_source_commit = validate_hex(
            source_commit, length=40, field="source_commit"
        )
        validated_input_revision = (
            validate_hex(input_revision, length=40, field="input_revision")
            if input_revision is not None
            else None
        )
        validated_pipeline_version = validate_nonblank_no_ws(
            pipeline_version, "pipeline_version"
        )
        validated_split_model = validate_nonblank_no_ws(split_model, "split_model")
        validated_model_repo = validate_repo_id(
            effective_model_repo_id,
            "model_repo_id",
            expected=DEFAULT_V2_LABEL_MODEL_REPO_ID
            if v2
            else DEFAULT_LABEL_MODEL_REPO_ID,
        )
        validated_model_revision = validate_hex(
            effective_model_revision, length=40, field="model_revision"
        )
        validated_model_file = validate_model_file(effective_model_file, "model_file")
        validated_model_file_sha256 = validate_hex(
            effective_model_file_sha256, length=64, field="model_file_sha256"
        )
        validated_tokenizer_repo = validate_repo_id(
            effective_tokenizer_repo_id,
            "tokenizer_repo_id",
            expected=DEFAULT_TOKENIZER_REPO_ID,
        )
        validated_tokenizer_revision = validate_hex(
            effective_tokenizer_revision, length=40, field="tokenizer_revision"
        )
        validated_prompt_version = validate_nonblank_no_ws(
            effective_prompt_version, "prompt_version"
        )
        validated_input_dataset_id = validate_repo_id(
            input_dataset_id,
            "input_dataset_id",
            expected=INPUT_DATASET_ID,
        )
        validated_output_dataset_id = validate_repo_id(
            output_dataset_id,
            "output_dataset_id",
            expected=OUTPUT_DATASET_ID,
        )

        effective_sampling_target = (
            DEFAULT_SAMPLING_TARGET
            if canonical_scope is Scope.ALL and sampling_target is None
            else sampling_target
        )
        requirements = Grid5000Requirements.build(
            batch_size=batch_size,
            row_limit=row_limit,
            llama_parallel=llama_parallel,
            llama_per_slot_context=llama_per_slot_context,
            llama_total_context=llama_total_context,
            request_concurrency=request_concurrency,
            sampling_target=effective_sampling_target,
            sampling_seed=sampling_seed,
            sampling_h3_resolution=sampling_h3_resolution,
            sampling_version=effective_sampling_version,
        )
        return cls(
            scope=canonical_scope,
            stage=canonical_stage,
            source_commit=validated_source_commit,
            execution_commit=validated_source_commit,
            requirements=requirements,
            region=canonical_region,
            input_dataset_revision=validated_input_revision,
            input_dataset_id=validated_input_dataset_id,
            output_dataset_id=validated_output_dataset_id,
            pipeline_version=validated_pipeline_version,
            split_model=validated_split_model,
            label_model_repo_id=validated_model_repo,
            label_model_revision=validated_model_revision,
            label_model_file=validated_model_file,
            label_model_file_sha256=validated_model_file_sha256,
            tokenizer_repo_id=validated_tokenizer_repo,
            tokenizer_revision=validated_tokenizer_revision,
            prompt_version=validated_prompt_version,
        )

    @classmethod
    def from_persisted(cls, identity: Mapping[str, object]) -> OperatorConfig:
        """Reconstruct a validated configuration from a persisted run_identity.

        ``resume RUN_ID`` uses this to reattach or continue a historical run
        without requiring the current local Git HEAD to match the run's
        recorded source commit. The reconstruction is stage-aware: a split
        identity never inherits label-only defaults, and vice versa.

        Every field that was persisted in the original ``run_identity`` is
        validated exactly as in :meth:`build`. The reconstructed
        ``canonical_json`` therefore byte-matches the persisted one and the
        ``run_id`` reproduces exactly.
        """

        def _opt(key: str, fallback: object) -> object:
            value = identity.get(key)
            return value if value not in (None, "") else fallback

        identity_dict = cast("Mapping[str, str | int]", identity)
        stage = canonicalize_stage(cast(str, identity_dict["stage"]))
        sampling_fields_present = any(
            key in identity_dict
            for key in ("sampling_seed", "sampling_h3_resolution", "sampling_version")
        )
        persisted_sampling_target: int | None = cast(
            "int | None", identity_dict.get("sampling_target")
        )
        if persisted_sampling_target is None and sampling_fields_present:
            # A stable checkpoint identity intentionally omits the mutable
            # target. Zero reconstructs the explicit V2 full-input mode when
            # no target fact is available; the operator resume path injects a
            # positive target from state facts before calling this method.
            persisted_sampling_target = 0
        # Split identities never persisted label-only fields; we must NOT
        # silently fall back to current defaults for those, because doing
        # so would change the canonical JSON and break the run ID.
        if stage is Stage.SPLIT:
            return cls.build(
                scope=cast(str, identity_dict["scope"]),
                stage=Stage.SPLIT,
                source_commit=cast(str, identity_dict["source_commit"]),
                batch_size=identity_dict["batch_size"],
                region=cast("str | None", identity_dict.get("region")),
                input_revision=cast(
                    "str | None", identity_dict.get("input_dataset_revision")
                ),
                input_dataset_id=cast(str, identity_dict["input_dataset_id"]),
                output_dataset_id=cast(str, identity_dict["output_dataset_id"]),
                pipeline_version=cast(str, _opt("pipeline_version", PIPELINE_VERSION)),
                split_model=cast(str, _opt("split_model", DEFAULT_SPLIT_MODEL)),
            )
        if stage is Stage.LABEL:
            return cls.build(
                scope=cast(str, identity_dict["scope"]),
                stage=Stage.LABEL,
                source_commit=cast(str, identity_dict["source_commit"]),
                batch_size=identity_dict["batch_size"],
                row_limit=identity_dict["row_limit"],
                llama_parallel=identity_dict["llama_parallel"],
                llama_per_slot_context=identity_dict["llama_per_slot_context"],
                llama_total_context=identity_dict["llama_total_context"],
                request_concurrency=identity_dict["request_concurrency"],
                sampling_target=persisted_sampling_target,
                sampling_seed=cast(
                    str,
                    _opt("sampling_seed", DEFAULT_SAMPLING_SEED),
                ),
                sampling_h3_resolution=cast(
                    int,
                    _opt("sampling_h3_resolution", DEFAULT_SAMPLING_H3_RESOLUTION),
                ),
                sampling_version=cast(str, _opt("sampling_version", SAMPLING_VERSION)),
                region=cast("str | None", identity_dict.get("region")),
                input_revision=cast(
                    "str | None", identity_dict.get("input_dataset_revision")
                ),
                input_dataset_id=cast(str, identity_dict["input_dataset_id"]),
                output_dataset_id=cast(str, identity_dict["output_dataset_id"]),
                pipeline_version=cast(str, _opt("pipeline_version", PIPELINE_VERSION)),
                model_repo_id=cast(
                    str, _opt("model_repo_id", DEFAULT_LABEL_MODEL_REPO_ID)
                ),
                model_revision=cast(
                    str, _opt("model_revision", DEFAULT_LABEL_MODEL_REVISION)
                ),
                model_file=cast(str, _opt("model_file", DEFAULT_LABEL_MODEL_FILE)),
                model_file_sha256=cast(
                    str, _opt("model_file_sha256", DEFAULT_LABEL_MODEL_FILE_SHA256)
                ),
                tokenizer_repo_id=cast(
                    str, _opt("tokenizer_repo_id", DEFAULT_TOKENIZER_REPO_ID)
                ),
                tokenizer_revision=cast(
                    str, _opt("tokenizer_revision", DEFAULT_TOKENIZER_REVISION)
                ),
                prompt_version=cast(str, _opt("prompt_version", PROMPT_VERSION)),
            )
        # Stage.ALL requires both split and label identity fields.
        return cls.build(
            scope=cast(str, identity_dict["scope"]),
            stage=Stage.ALL,
            source_commit=cast(str, identity_dict["source_commit"]),
            batch_size=identity_dict["batch_size"],
            row_limit=identity_dict["row_limit"],
            llama_parallel=identity_dict["llama_parallel"],
            llama_per_slot_context=identity_dict["llama_per_slot_context"],
            llama_total_context=identity_dict["llama_total_context"],
            request_concurrency=identity_dict["request_concurrency"],
            sampling_target=persisted_sampling_target,
            sampling_seed=cast(str, _opt("sampling_seed", DEFAULT_SAMPLING_SEED)),
            sampling_h3_resolution=cast(
                int, _opt("sampling_h3_resolution", DEFAULT_SAMPLING_H3_RESOLUTION)
            ),
            sampling_version=cast(str, _opt("sampling_version", SAMPLING_VERSION)),
            region=cast("str | None", identity_dict.get("region")),
            input_revision=cast(
                "str | None", identity_dict.get("input_dataset_revision")
            ),
            input_dataset_id=cast(str, identity_dict["input_dataset_id"]),
            output_dataset_id=cast(str, identity_dict["output_dataset_id"]),
            pipeline_version=cast(str, _opt("pipeline_version", PIPELINE_VERSION)),
            split_model=cast(str, _opt("split_model", DEFAULT_SPLIT_MODEL)),
            model_repo_id=cast(str, _opt("model_repo_id", DEFAULT_LABEL_MODEL_REPO_ID)),
            model_revision=cast(
                str, _opt("model_revision", DEFAULT_LABEL_MODEL_REVISION)
            ),
            model_file=cast(str, _opt("model_file", DEFAULT_LABEL_MODEL_FILE)),
            model_file_sha256=cast(
                str, _opt("model_file_sha256", DEFAULT_LABEL_MODEL_FILE_SHA256)
            ),
            tokenizer_repo_id=cast(
                str, _opt("tokenizer_repo_id", DEFAULT_TOKENIZER_REPO_ID)
            ),
            tokenizer_revision=cast(
                str, _opt("tokenizer_revision", DEFAULT_TOKENIZER_REVISION)
            ),
            prompt_version=cast(str, _opt("prompt_version", PROMPT_VERSION)),
        )

    @property
    def run_id(self) -> str:
        """Short hash identifying this deterministic run."""

        return self.run_identity.run_id

    @property
    def canonical_json(self) -> str:
        """Canonical JSON representation of the result-affecting identity."""

        return self.run_identity.canonical_json

    def __post_init__(self) -> None:
        if self.execution_commit is None:
            object.__setattr__(self, "execution_commit", self.source_commit)
        is_split = self.stage in (Stage.SPLIT, Stage.ALL)
        is_label = self.stage in (Stage.LABEL, Stage.ALL)

        split_model = self.split_model if is_split else None
        model_repo_id = self.label_model_repo_id if is_label else None
        model_revision = self.label_model_revision if is_label else None
        model_file = self.label_model_file if is_label else None
        model_file_sha = self.label_model_file_sha256 if is_label else None
        tokenizer_repo_id = self.tokenizer_repo_id if is_label else None
        tokenizer_revision = self.tokenizer_revision if is_label else None
        prompt_version = self.prompt_version if is_label else None
        row_limit = self.requirements.row_limit if is_label else None
        llama_parallel = self.requirements.llama_parallel if is_label else None
        llama_per_slot_context = (
            self.requirements.llama_per_slot_context if is_label else None
        )
        llama_total_context = (
            self.requirements.llama_total_context if is_label else None
        )
        request_concurrency = (
            self.requirements.request_concurrency if is_label else None
        )
        sampling_target = self.requirements.sampling_target if is_label else None
        sampling_enabled = is_label and sampling_target is not None
        sampling_seed = self.requirements.sampling_seed if sampling_enabled else None
        sampling_h3_resolution = (
            self.requirements.sampling_h3_resolution if sampling_enabled else None
        )
        sampling_version = (
            self.requirements.sampling_version if sampling_enabled else None
        )

        object.__setattr__(
            self,
            "run_identity",
            RunIdentity(
                scope=self.scope,
                stage=self.stage,
                source_commit=self.source_commit,
                input_dataset_id=self.input_dataset_id,
                output_dataset_id=self.output_dataset_id,
                pipeline_version=self.pipeline_version,
                batch_size=self.requirements.batch_size,
                row_limit=row_limit,
                llama_parallel=llama_parallel,
                llama_per_slot_context=llama_per_slot_context,
                llama_total_context=llama_total_context,
                request_concurrency=request_concurrency,
                region=self.region,
                input_dataset_revision=self.input_dataset_revision,
                split_model=split_model,
                model_repo_id=model_repo_id,
                model_revision=model_revision,
                model_file=model_file,
                model_file_sha256=model_file_sha,
                tokenizer_repo_id=tokenizer_repo_id,
                tokenizer_revision=tokenizer_revision,
                prompt_version=prompt_version,
                sampling_target=sampling_target,
                sampling_seed=sampling_seed,
                sampling_h3_resolution=sampling_h3_resolution,
                sampling_version=sampling_version,
            ),
        )
