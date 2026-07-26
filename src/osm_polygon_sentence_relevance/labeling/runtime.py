"""Runtime configuration for the local llama.cpp inference server.

This module owns the small, validated set of parallelism choices that the
production labeling payload supports. It is the single source of truth for
how parallel slots, per-slot context, and total context are related; the Grid'5000
launchers and the labeling CLI both depend on its predicates so that a
misconfiguration is rejected before any binary is invoked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .engine import OpenAICompatibleEngine

SUPPORTED_LLAMA_PARALLEL: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
"""The only parallelism values the launcher accepts.

Choosing a value outside this set would either underuse the GPU or exceed
the per-slot context budget we have validated. The set is small and
deliberately enumerated so the production payload cannot silently fall
back to a partially supported configuration.
"""

MIN_PER_SLOT_CONTEXT: int = 4096
"""Minimum per-slot context for the labeling prompt and response."""


def validate_llama_parallel(value: object) -> int:
    """Return ``value`` as ``int`` when it is a supported parallelism; else raise."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("llama parallel must be an integer")
    if value not in SUPPORTED_LLAMA_PARALLEL:
        raise ValueError(
            "llama parallel must be one of "
            f"{', '.join(str(item) for item in SUPPORTED_LLAMA_PARALLEL)}"
        )
    return value


def validate_per_slot_context(value: object) -> int:
    """Return ``value`` as ``int`` when it is at least the minimum per-slot context."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("per-slot context must be an integer")
    if value < MIN_PER_SLOT_CONTEXT:
        raise ValueError(f"per-slot context must be at least {MIN_PER_SLOT_CONTEXT}")
    return value


def compute_total_context(parallel: int, per_slot_context: int | None = None) -> int:
    """Compute the total context as ``parallel * per_slot_context``.

    The per-slot argument is optional; when omitted the minimum is used. The
    function never silently partitions a fixed total into more slots than the
    server can support.
    """

    parallel_int = validate_llama_parallel(parallel)
    per_slot = (
        validate_per_slot_context(per_slot_context)
        if per_slot_context is not None
        else MIN_PER_SLOT_CONTEXT
    )
    return parallel_int * per_slot


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    """Validated, identity-bound runtime configuration for one labeling run."""

    parallel: int
    per_slot_context: int
    total_context: int
    request_concurrency: int

    def __post_init__(self) -> None:
        validate_llama_parallel(self.parallel)
        validate_per_slot_context(self.per_slot_context)
        if self.total_context != self.parallel * self.per_slot_context:
            raise ValueError("total context must equal parallel times per-slot context")
        if self.request_concurrency < 1 or self.request_concurrency > self.parallel:
            raise ValueError(
                "request concurrency must be between 1 and the parallel slot count"
            )


def build_runtime_plan(
    *, parallel: int, per_slot_context: int | None = None
) -> RuntimePlan:
    """Construct a :class:`RuntimePlan` with derived total context and concurrency."""

    parallel_int = validate_llama_parallel(parallel)
    per_slot = (
        validate_per_slot_context(per_slot_context)
        if per_slot_context is not None
        else MIN_PER_SLOT_CONTEXT
    )
    total = parallel_int * per_slot
    return RuntimePlan(
        parallel=parallel_int,
        per_slot_context=per_slot,
        total_context=total,
        request_concurrency=parallel_int,
    )


def resolve_engine_factory(
    plan: RuntimePlan,
) -> Callable[..., OpenAICompatibleEngine]:
    """Return an engine factory that pins concurrency to the parallel slot count."""

    plan_parallel = plan.parallel

    def factory(
        *, endpoint: str, model: str, **kwargs: object
    ) -> OpenAICompatibleEngine:
        del kwargs  # any explicit concurrency is overridden by the runtime plan
        return OpenAICompatibleEngine(
            endpoint=endpoint,
            model=model,
            concurrency=plan_parallel,
        )

    return factory


__all__ = [
    "MIN_PER_SLOT_CONTEXT",
    "RuntimePlan",
    "SUPPORTED_LLAMA_PARALLEL",
    "build_runtime_plan",
    "compute_total_context",
    "resolve_engine_factory",
    "validate_llama_parallel",
    "validate_per_slot_context",
]
