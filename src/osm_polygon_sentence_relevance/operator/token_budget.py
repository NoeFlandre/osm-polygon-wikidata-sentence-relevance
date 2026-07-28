"""Deterministic llama.cpp context planning from measured prompt tokens."""

from __future__ import annotations

from dataclasses import dataclass

from osm_polygon_sentence_relevance.labeling.runtime import (
    SUPPORTED_LLAMA_PARALLEL,
)

SUPPORTED_SLOT_CONTEXTS = (4096, 8192, 12288, 16384, 32768)


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    """GPU-safe llama.cpp slot and request concurrency plan."""

    max_prompt_tokens: int
    response_tokens: int
    per_slot_context: int
    parallel: int
    total_context: int
    request_concurrency: int


class TokenBudgetError(ValueError):
    """No supported context plan can hold the measured request."""


def _positive(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise TokenBudgetError(f"{name} must be a positive integer")
    return value


def plan_runtime(
    *,
    max_prompt_tokens: int,
    response_tokens: int,
    gpu_memory_mb: int,
    max_total_context: int,
    model_memory_mb: int = 20_000,
    context_memory_bytes_per_token: int = 256,
) -> RuntimePlan:
    """Choose the fastest supported plan that fits context and GPU memory."""

    prompt = _positive(max_prompt_tokens, "max_prompt_tokens")
    response = _positive(response_tokens, "response_tokens")
    gpu_mb = _positive(gpu_memory_mb, "gpu_memory_mb")
    total_limit = _positive(max_total_context, "max_total_context")
    model_mb = _positive(model_memory_mb, "model_memory_mb")
    bytes_per_token = _positive(
        context_memory_bytes_per_token, "context_memory_bytes_per_token"
    )
    required = prompt + response
    slot = next((size for size in SUPPORTED_SLOT_CONTEXTS if size >= required), None)
    if slot is None:
        raise TokenBudgetError("measured request exceeds supported slot contexts")

    available_bytes = (gpu_mb - model_mb) * 1024 * 1024
    if available_bytes <= 0:
        raise TokenBudgetError("model does not fit available GPU memory")
    for parallel in sorted(SUPPORTED_LLAMA_PARALLEL, reverse=True):
        total = parallel * slot
        if total > total_limit:
            continue
        if total * bytes_per_token > available_bytes:
            continue
        return RuntimePlan(
            max_prompt_tokens=prompt,
            response_tokens=response,
            per_slot_context=slot,
            parallel=parallel,
            total_context=total,
            request_concurrency=parallel,
        )
    raise TokenBudgetError("no supported runtime plan fits the GPU budget")


__all__ = [
    "RuntimePlan",
    "SUPPORTED_SLOT_CONTEXTS",
    "TokenBudgetError",
    "plan_runtime",
]
