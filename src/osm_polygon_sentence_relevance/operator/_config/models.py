"""Immutable operator configuration and run-identity contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from json import dumps
from pathlib import Path
from typing import cast

from osm_polygon_sentence_relevance.contracts.constants import (
    PIPELINE_VERSION,
)
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
    DEFAULT_SPLIT_MODEL,
    DEFAULT_TOKENIZER_REPO_ID,
    DEFAULT_TOKENIZER_REVISION,
    INPUT_DATASET_ID,
    OUTPUT_DATASET_ID,
    PROMPT_VERSION,
)
from osm_polygon_sentence_relevance.operator._config.enums import Scope, Stage
from osm_polygon_sentence_relevance.operator._config.validation import (
    canonicalize_scope,
    canonicalize_stage,
    normalize_runtime_requirements,
    require_run_fields_for_scope,
    validate_hex,
    validate_model_file,
    validate_nonblank_no_ws,
    validate_repo_id,
)


@dataclass(frozen=True, slots=True)
class Grid5000Requirements:
    """Validated runtime settings that affect resumable checkpoints."""

    batch_size: int
    row_limit: int
    llama_parallel: int
    llama_per_slot_context: int
    llama_total_context: int | None = None
    request_concurrency: int | None = None

    def __post_init__(self) -> None:
        (
            normalized_batch_size,
            normalized_row_limit,
            normalized_parallel,
            normalized_per_slot,
            normalized_total,
            normalized_concurrency,
        ) = normalize_runtime_requirements(
            batch_size=self.batch_size,
            row_limit=self.row_limit,
            llama_parallel=self.llama_parallel,
            llama_per_slot_context=self.llama_per_slot_context,
            llama_total_context=self.llama_total_context,
            request_concurrency=self.request_concurrency,
        )
        object.__setattr__(self, "batch_size", normalized_batch_size)
        object.__setattr__(self, "row_limit", normalized_row_limit)
        object.__setattr__(self, "llama_parallel", normalized_parallel)
        object.__setattr__(self, "llama_per_slot_context", normalized_per_slot)
        object.__setattr__(self, "llama_total_context", normalized_total)
        object.__setattr__(self, "request_concurrency", normalized_concurrency)

    @classmethod
    def build(
        cls,
        *,
        batch_size: int | str = DEFAULT_BATCH_SIZE,
        row_limit: int | str = DEFAULT_ROW_LIMIT,
        llama_parallel: int | str = DEFAULT_LLAMA_PARALLEL,
        llama_per_slot_context: int | str = DEFAULT_LLAMA_PER_SLOT_CONTEXT,
        llama_total_context: int | str | None = None,
        request_concurrency: int | str | None = None,
    ) -> Grid5000Requirements:
        return cls(
            batch_size=cast(int, batch_size),
            row_limit=cast(int, row_limit),
            llama_parallel=cast(int, llama_parallel),
            llama_per_slot_context=cast(int, llama_per_slot_context),
            llama_total_context=cast(int | None, llama_total_context),
            request_concurrency=cast(int | None, request_concurrency),
        )


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Deterministic identity for a complete operator run."""

    scope: Scope
    stage: Stage
    source_commit: str
    input_dataset_id: str
    output_dataset_id: str
    pipeline_version: str
    batch_size: int
    row_limit: int | None
    llama_parallel: int | None
    llama_per_slot_context: int | None
    llama_total_context: int | None
    request_concurrency: int | None
    region: str | None = None
    input_dataset_revision: str | None = None
    split_model: str | None = None
    model_repo_id: str | None = None
    model_revision: str | None = None
    model_file: str | None = None
    model_file_sha256: str | None = None
    tokenizer_repo_id: str | None = None
    tokenizer_revision: str | None = None
    prompt_version: str | None = None
    _canonical_json: str = field(default="", init=False, repr=False)
    run_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        payload = self.to_dict()
        canonical = dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        digest = sha256(canonical.encode("utf-8")).hexdigest()[:20]
        object.__setattr__(self, "_canonical_json", canonical)
        object.__setattr__(self, "run_id", digest)

    @property
    def canonical_json(self) -> str:
        """Canonical JSON used to derive run identity."""

        return self._canonical_json

    def to_dict(self) -> dict[str, str | int]:
        """Return the stable subset of result-affecting fields."""

        payload: dict[str, str | int] = {
            "scope": str(self.scope),
            "stage": str(self.stage),
            "source_commit": self.source_commit,
            "input_dataset_id": self.input_dataset_id,
            "output_dataset_id": self.output_dataset_id,
            "pipeline_version": self.pipeline_version,
            "batch_size": self.batch_size,
        }
        if self.region is not None:
            payload["region"] = self.region
        if self.input_dataset_revision is not None:
            payload["input_dataset_revision"] = self.input_dataset_revision
        if self.split_model is not None:
            payload["split_model"] = self.split_model
        if self.model_repo_id is not None:
            payload["model_repo_id"] = self.model_repo_id
        if self.model_revision is not None:
            payload["model_revision"] = self.model_revision
        if self.model_file is not None:
            payload["model_file"] = self.model_file
        if self.model_file_sha256 is not None:
            payload["model_file_sha256"] = self.model_file_sha256
        if self.tokenizer_repo_id is not None:
            payload["tokenizer_repo_id"] = self.tokenizer_repo_id
        if self.tokenizer_revision is not None:
            payload["tokenizer_revision"] = self.tokenizer_revision
        if self.prompt_version is not None:
            payload["prompt_version"] = self.prompt_version
        if self.row_limit is not None:
            payload["row_limit"] = self.row_limit
        if self.llama_parallel is not None:
            payload["llama_parallel"] = self.llama_parallel
        if self.llama_per_slot_context is not None:
            payload["llama_per_slot_context"] = self.llama_per_slot_context
        if self.llama_total_context is not None:
            payload["llama_total_context"] = self.llama_total_context
        if self.request_concurrency is not None:
            payload["request_concurrency"] = self.request_concurrency
        return payload


@dataclass(frozen=True, slots=True)
class OperatorConfig:
    """Validated production configuration for one operator run."""

    scope: Scope
    stage: Stage
    source_commit: str
    requirements: Grid5000Requirements
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
        region: str | None = None,
        input_revision: str | None = None,
        input_dataset_id: str = INPUT_DATASET_ID,
        output_dataset_id: str = OUTPUT_DATASET_ID,
        pipeline_version: str = PIPELINE_VERSION,
        split_model: str = DEFAULT_SPLIT_MODEL,
        model_repo_id: str = DEFAULT_LABEL_MODEL_REPO_ID,
        model_revision: str = DEFAULT_LABEL_MODEL_REVISION,
        model_file: str = DEFAULT_LABEL_MODEL_FILE,
        model_file_sha256: str = DEFAULT_LABEL_MODEL_FILE_SHA256,
        tokenizer_repo_id: str = DEFAULT_TOKENIZER_REPO_ID,
        tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
        prompt_version: str = PROMPT_VERSION,
    ) -> OperatorConfig:
        canonical_scope = canonicalize_scope(scope)
        canonical_stage = canonicalize_stage(stage)
        canonical_region = require_run_fields_for_scope(
            canonical_scope, canonical_stage, region
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
            model_repo_id, "model_repo_id", expected=DEFAULT_LABEL_MODEL_REPO_ID
        )
        validated_model_revision = validate_hex(
            model_revision, length=40, field="model_revision"
        )
        validated_model_file = validate_model_file(model_file, "model_file")
        validated_model_file_sha256 = validate_hex(
            model_file_sha256, length=64, field="model_file_sha256"
        )
        validated_tokenizer_repo = validate_repo_id(
            tokenizer_repo_id,
            "tokenizer_repo_id",
            expected=DEFAULT_TOKENIZER_REPO_ID,
        )
        validated_tokenizer_revision = validate_hex(
            tokenizer_revision, length=40, field="tokenizer_revision"
        )
        validated_prompt_version = validate_nonblank_no_ws(
            prompt_version, "prompt_version"
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

        requirements = Grid5000Requirements.build(
            batch_size=batch_size,
            row_limit=row_limit,
            llama_parallel=llama_parallel,
            llama_per_slot_context=llama_per_slot_context,
            llama_total_context=llama_total_context,
            request_concurrency=request_concurrency,
        )
        return cls(
            scope=canonical_scope,
            stage=canonical_stage,
            source_commit=validated_source_commit,
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
            ),
        )
