"""Autonomous operator public configuration and run-identity contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from json import dumps
from pathlib import Path
from typing import Final, cast

from osm_polygon_sentence_relevance.contracts.constants import (
    INPUT_DATASET_ID as _INPUT_DATASET_ID,
)
from osm_polygon_sentence_relevance.contracts.constants import (
    OUTPUT_DATASET_ID as _OUTPUT_DATASET_ID,
)
from osm_polygon_sentence_relevance.contracts.constants import (
    PIPELINE_VERSION,
)
from osm_polygon_sentence_relevance.labeling.prompt import PROMPT_VERSION
from osm_polygon_sentence_relevance.labeling.runtime import (
    MIN_PER_SLOT_CONTEXT,
    compute_total_context,
    validate_llama_parallel,
    validate_per_slot_context,
)
from osm_polygon_sentence_relevance.labeling.runtime import (
    SUPPORTED_LLAMA_PARALLEL as _RUNTIME_SUPPORTED_LLAMA_PARALLEL,
)

DATA_ROOT: Final[Path] = Path(
    "/Volumes/Seagate M3/projects/osm-polygon-wikidata-sentence-relevance"
)
INPUT_DATASET_ID: Final[str] = _INPUT_DATASET_ID
OUTPUT_DATASET_ID: Final[str] = _OUTPUT_DATASET_ID

DEFAULT_SPLIT_MODEL: Final[str] = "sat-12l-sm"
DEFAULT_BATCH_SIZE: Final[int] = 128
DEFAULT_ROW_LIMIT: Final[int] = 0
DEFAULT_LLAMA_PARALLEL: Final[int] = 16
DEFAULT_LLAMA_PER_SLOT_CONTEXT: Final[int] = MIN_PER_SLOT_CONTEXT

DEFAULT_LABEL_MODEL_REPO_ID: Final[str] = "unsloth/Qwen3.6-27B-MTP-GGUF"
DEFAULT_LABEL_MODEL_REVISION: Final[str] = "5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace"
DEFAULT_LABEL_MODEL_FILE: Final[str] = "Qwen3.6-27B-Q4_K_M.gguf"
DEFAULT_LABEL_MODEL_FILE_SHA256: Final[str] = (
    "a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f"
)
DEFAULT_TOKENIZER_REPO_ID: Final[str] = "Qwen/Qwen3.6-27B"
DEFAULT_TOKENIZER_REVISION: Final[str] = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

SUPPORTED_LLAMA_PARALLEL: Final[tuple[int, ...]] = _RUNTIME_SUPPORTED_LLAMA_PARALLEL
"""The production-parallelism values accepted by Grid5000 and labeling."""

_NON_NEGATIVE_INT_RE = re.compile(r"^(0|[1-9][0-9]*)$")
_POSITIVE_INT_RE = re.compile(r"^[1-9][0-9]*$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_REGION_RE = re.compile(r"^(?:[a-z0-9]+(?:-[a-z0-9]+)*)-latest$")
_REPO_SEGMENT_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,98}[A-Za-z0-9])?$")


class Scope(StrEnum):
    """Operator execution scope."""

    REGION = "region"
    ALL = "all"


class Stage(StrEnum):
    """Operator stage within the end-to-end pipeline."""

    SPLIT = "split"
    LABEL = "label"
    ALL = "all"


def _coerce_int(
    value: int | str,
    field: str,
    *,
    allow_zero: bool = False,
) -> int:
    """Parse an integer argument with strict numeric validation."""
    if isinstance(value, bool):
        raise ValueError(
            f"{field} must be {'a non-negative' if allow_zero else 'a positive'} integer"
        )
    if isinstance(value, int):
        if not allow_zero and value <= 0:
            raise ValueError(f"{field} must be a positive integer")
        if allow_zero and value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        return value
    if isinstance(value, str):
        if value == "":
            raise ValueError(
                f"{field} must be {'a non-negative' if allow_zero else 'a positive'} integer"
            )
        pattern = _NON_NEGATIVE_INT_RE if allow_zero else _POSITIVE_INT_RE
        if not pattern.fullmatch(value):
            raise ValueError(
                f"{field} must be {'a non-negative' if allow_zero else 'a positive'} integer "
                "and cannot use leading zeroes"
            )
        parsed = int(value)
        if not allow_zero and parsed <= 0:
            raise ValueError(f"{field} must be a positive integer")
        return parsed
    raise ValueError(
        f"{field} must be {'a non-negative' if allow_zero else 'a positive'} integer"
    )


def _validate_nonblank_no_ws(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} cannot be blank")
    if value != value.strip():
        raise ValueError(f"{field} has surrounding whitespace")
    return value


def _validate_repo_id(value: str, field: str, *, expected: str | None = None) -> str:
    validated = _validate_nonblank_no_ws(value, field)
    if validated.count("/") != 1:
        raise ValueError(f"{field} must be exactly owner/name")
    owner, repo = validated.split("/")
    if not owner or not repo:
        raise ValueError(f"{field} must be exactly owner/name")
    if not _REPO_SEGMENT_RE.fullmatch(owner) or not _REPO_SEGMENT_RE.fullmatch(repo):
        raise ValueError(f"{field} must be a valid owner/name identifier")
    if expected is not None and validated != expected:
        raise ValueError(f"{field} must match the expected repository: {expected!r}")
    return validated


def _validate_model_file(value: str, field: str) -> str:
    validated = _validate_nonblank_no_ws(value, field)
    if (
        validated in {"", ".", ".."}
        or ".." in validated
        or "/" in validated
        or "\\" in validated
        or "\x00" in validated
    ):
        raise ValueError(f"{field} must be a safe filename")
    return validated


def _validate_hex(value: str, *, length: int, field: str) -> str:
    validated = _validate_nonblank_no_ws(value, field)
    if len(validated) != length:
        raise ValueError(f"{field} must be exactly {length} characters")
    pattern = _HEX40_RE if length == 40 else _HEX64_RE
    if not pattern.fullmatch(validated):
        raise ValueError(f"{field} must be lowercase hexadecimal of length {length}")
    return validated


def _validate_region(value: str) -> str:
    validated = _validate_nonblank_no_ws(value, "region")
    if not _REGION_RE.fullmatch(validated):
        raise ValueError("region has malformed canonical shard syntax")
    return validated


def _canonicalize_scope(scope: Scope | str) -> Scope:
    if isinstance(scope, Scope):
        return scope
    if not isinstance(scope, str):
        raise ValueError("scope must be a string")
    try:
        return Scope(scope)
    except ValueError as exc:
        raise ValueError(f"invalid scope: {scope!r}") from exc


def _canonicalize_stage(stage: Stage | str) -> Stage:
    if isinstance(stage, Stage):
        return stage
    if not isinstance(stage, str):
        raise ValueError("stage must be a string")
    try:
        return Stage(stage)
    except ValueError as exc:
        raise ValueError(f"invalid stage: {stage!r}") from exc


def _require_run_fields_for_scope(
    scope: Scope, stage: Stage, region: str | None
) -> str | None:
    if scope is Scope.ALL:
        if region is not None:
            raise ValueError("region is not allowed when scope is all")
        return None
    if region is None:
        raise ValueError("region is required when scope is region")
    _ = stage
    return _validate_region(region)


def _normalize_runtime_requirements(
    batch_size: int | str,
    row_limit: int | str,
    llama_parallel: int | str,
    llama_per_slot_context: int | str,
    llama_total_context: int | str | None = None,
    request_concurrency: int | str | None = None,
) -> tuple[int, int, int, int, int, int]:
    parsed_batch_size = _coerce_int(batch_size, "batch_size")
    parsed_row_limit = _coerce_int(row_limit, "row_limit", allow_zero=True)
    parsed_parallel = validate_llama_parallel(
        _coerce_int(llama_parallel, "llama_parallel")
    )
    parsed_per_slot = validate_per_slot_context(
        _coerce_int(llama_per_slot_context, "llama_per_slot_context")
    )
    parsed_total = (
        _coerce_int(llama_total_context, "llama_total_context")
        if llama_total_context is not None
        else compute_total_context(parsed_parallel, parsed_per_slot)
    )
    if parsed_total != parsed_parallel * parsed_per_slot:
        raise ValueError(
            "llama_total_context must equal llama_parallel * llama_per_slot_context"
        )
    parsed_concurrency = (
        parsed_parallel
        if request_concurrency is None
        else _coerce_int(request_concurrency, "request_concurrency")
    )
    if parsed_concurrency < 1 or parsed_concurrency > parsed_parallel:
        raise ValueError(
            "request concurrency must be between 1 and the parallel slot count"
        )
    return (
        parsed_batch_size,
        parsed_row_limit,
        parsed_parallel,
        parsed_per_slot,
        parsed_total,
        parsed_concurrency,
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
        ) = _normalize_runtime_requirements(
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
        canonical_scope = _canonicalize_scope(scope)
        canonical_stage = _canonicalize_stage(stage)
        canonical_region = _require_run_fields_for_scope(
            canonical_scope, canonical_stage, region
        )

        validated_source_commit = _validate_hex(
            source_commit, length=40, field="source_commit"
        )
        validated_input_revision = (
            _validate_hex(input_revision, length=40, field="input_revision")
            if input_revision is not None
            else None
        )
        validated_pipeline_version = _validate_nonblank_no_ws(
            pipeline_version, "pipeline_version"
        )
        validated_split_model = _validate_nonblank_no_ws(split_model, "split_model")
        validated_model_repo = _validate_repo_id(
            model_repo_id, "model_repo_id", expected=DEFAULT_LABEL_MODEL_REPO_ID
        )
        validated_model_revision = _validate_hex(
            model_revision, length=40, field="model_revision"
        )
        validated_model_file = _validate_model_file(model_file, "model_file")
        validated_model_file_sha256 = _validate_hex(
            model_file_sha256, length=64, field="model_file_sha256"
        )
        validated_tokenizer_repo = _validate_repo_id(
            tokenizer_repo_id,
            "tokenizer_repo_id",
            expected=DEFAULT_TOKENIZER_REPO_ID,
        )
        validated_tokenizer_revision = _validate_hex(
            tokenizer_revision, length=40, field="tokenizer_revision"
        )
        validated_prompt_version = _validate_nonblank_no_ws(
            prompt_version, "prompt_version"
        )
        validated_input_dataset_id = _validate_repo_id(
            input_dataset_id,
            "input_dataset_id",
            expected=INPUT_DATASET_ID,
        )
        validated_output_dataset_id = _validate_repo_id(
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


__all__ = [
    "Scope",
    "Stage",
    "Grid5000Requirements",
    "RunIdentity",
    "OperatorConfig",
    "DATA_ROOT",
    "INPUT_DATASET_ID",
    "OUTPUT_DATASET_ID",
    "DEFAULT_SPLIT_MODEL",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LLAMA_PARALLEL",
    "DEFAULT_LLAMA_PER_SLOT_CONTEXT",
    "DEFAULT_ROW_LIMIT",
    "DEFAULT_LABEL_MODEL_REPO_ID",
    "DEFAULT_LABEL_MODEL_REVISION",
    "DEFAULT_LABEL_MODEL_FILE",
    "DEFAULT_LABEL_MODEL_FILE_SHA256",
    "DEFAULT_TOKENIZER_REPO_ID",
    "DEFAULT_TOKENIZER_REVISION",
    "SUPPORTED_LLAMA_PARALLEL",
]
