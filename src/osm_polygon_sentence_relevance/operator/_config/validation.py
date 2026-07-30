"""Private validation for operator configuration contracts."""

from __future__ import annotations

import re

from osm_polygon_sentence_relevance.labeling.runtime import (
    compute_total_context,
    validate_llama_parallel,
    validate_per_slot_context,
)
from osm_polygon_sentence_relevance.operator._config.enums import Scope, Stage

_NON_NEGATIVE_INT_RE = re.compile(r"^(0|[1-9][0-9]*)$")
_POSITIVE_INT_RE = re.compile(r"^[1-9][0-9]*$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_REGION_RE = re.compile(r"^(?:[a-z0-9]+(?:-[a-z0-9]+)*)-latest$")
_REPO_SEGMENT_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,98}[A-Za-z0-9])?$")


def coerce_int(
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


def validate_nonblank_no_ws(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} cannot be blank")
    if value != value.strip():
        raise ValueError(f"{field} has surrounding whitespace")
    return value


def validate_repo_id(value: str, field: str, *, expected: str | None = None) -> str:
    validated = validate_nonblank_no_ws(value, field)
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


def validate_model_file(value: str, field: str) -> str:
    validated = validate_nonblank_no_ws(value, field)
    if (
        validated in {"", ".", ".."}
        or ".." in validated
        or "/" in validated
        or "\\" in validated
        or "\x00" in validated
    ):
        raise ValueError(f"{field} must be a safe filename")
    return validated


def validate_hex(value: str, *, length: int, field: str) -> str:
    validated = validate_nonblank_no_ws(value, field)
    if len(validated) != length:
        raise ValueError(f"{field} must be exactly {length} characters")
    pattern = _HEX40_RE if length == 40 else _HEX64_RE
    if not pattern.fullmatch(validated):
        raise ValueError(f"{field} must be lowercase hexadecimal of length {length}")
    return validated


def validate_region(value: str) -> str:
    validated = validate_nonblank_no_ws(value, "region")
    if not _REGION_RE.fullmatch(validated):
        raise ValueError("region has malformed canonical shard syntax")
    return validated


def canonicalize_scope(scope: Scope | str) -> Scope:
    if isinstance(scope, Scope):
        return scope
    if not isinstance(scope, str):
        raise ValueError("scope must be a string")
    try:
        return Scope(scope)
    except ValueError as exc:
        raise ValueError(f"invalid scope: {scope!r}") from exc


def canonicalize_stage(stage: Stage | str) -> Stage:
    if isinstance(stage, Stage):
        return stage
    if not isinstance(stage, str):
        raise ValueError("stage must be a string")
    try:
        return Stage(stage)
    except ValueError as exc:
        raise ValueError(f"invalid stage: {stage!r}") from exc


def require_run_fields_for_scope(
    scope: Scope, stage: Stage, region: str | None
) -> str | None:
    if scope is Scope.ALL:
        if region is not None:
            raise ValueError("region is not allowed when scope is all")
        return None
    if region is None:
        raise ValueError("region is required when scope is region")
    _ = stage
    return validate_region(region)


def normalize_runtime_requirements(
    batch_size: int | str,
    row_limit: int | str,
    llama_parallel: int | str,
    llama_per_slot_context: int | str,
    llama_total_context: int | str | None = None,
    request_concurrency: int | str | None = None,
) -> tuple[int, int, int, int, int, int]:
    parsed_batch_size = coerce_int(batch_size, "batch_size")
    parsed_row_limit = coerce_int(row_limit, "row_limit", allow_zero=True)
    parsed_parallel = validate_llama_parallel(
        coerce_int(llama_parallel, "llama_parallel")
    )
    parsed_per_slot = validate_per_slot_context(
        coerce_int(llama_per_slot_context, "llama_per_slot_context")
    )
    parsed_total = (
        coerce_int(llama_total_context, "llama_total_context")
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
        else coerce_int(request_concurrency, "request_concurrency")
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
