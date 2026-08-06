"""Validated Grid'5000 runtime requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from osm_polygon_sentence_relevance.operator._config.defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LLAMA_PARALLEL,
    DEFAULT_LLAMA_PER_SLOT_CONTEXT,
    DEFAULT_ROW_LIMIT,
    DEFAULT_SAMPLING_H3_RESOLUTION,
    DEFAULT_SAMPLING_SEED,
    DEFAULT_SAMPLING_TARGET,
    SAMPLING_VERSION,
)
from osm_polygon_sentence_relevance.operator._config.validation import (
    coerce_int,
    normalize_runtime_requirements,
)


@dataclass(frozen=True, slots=True)
class Grid5000Requirements:
    """Validated settings that affect a resumable allocation."""

    batch_size: int
    row_limit: int
    llama_parallel: int
    llama_per_slot_context: int
    llama_total_context: int | None = None
    request_concurrency: int | None = None
    sampling_target: int | None = DEFAULT_SAMPLING_TARGET
    sampling_seed: str = DEFAULT_SAMPLING_SEED
    sampling_h3_resolution: int = DEFAULT_SAMPLING_H3_RESOLUTION
    sampling_version: str = SAMPLING_VERSION

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
        if self.sampling_target is not None:
            from osm_polygon_sentence_relevance.labeling.sampling import SamplingConfig

            SamplingConfig(
                target=self.sampling_target or 1,
                seed=self.sampling_seed,
                h3_resolution=self.sampling_h3_resolution,
                version=self.sampling_version,
            )

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
        sampling_target: int | str | None = DEFAULT_SAMPLING_TARGET,
        sampling_seed: str = DEFAULT_SAMPLING_SEED,
        sampling_h3_resolution: int = DEFAULT_SAMPLING_H3_RESOLUTION,
        sampling_version: str = SAMPLING_VERSION,
    ) -> Grid5000Requirements:
        parsed_sampling_target = (
            None
            if sampling_target is None
            else coerce_int(sampling_target, "sampling_target", allow_zero=True)
        )
        return cls(
            batch_size=cast(int, batch_size),
            row_limit=cast(int, row_limit),
            llama_parallel=cast(int, llama_parallel),
            llama_per_slot_context=cast(int, llama_per_slot_context),
            llama_total_context=cast(int | None, llama_total_context),
            request_concurrency=cast(int | None, request_concurrency),
            sampling_target=parsed_sampling_target,
            sampling_seed=sampling_seed,
            sampling_h3_resolution=coerce_int(
                sampling_h3_resolution, "sampling_h3_resolution", allow_zero=True
            ),
            sampling_version=sampling_version,
        )


__all__ = ["Grid5000Requirements"]
