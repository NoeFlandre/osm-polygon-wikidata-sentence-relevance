"""Small immutable contracts shared by the labeling pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class LabelValue(str, Enum):
    """Permitted relevance decisions."""

    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class SentenceLabel:
    """Validated labels returned by the model."""

    landuse_relevance: LabelValue
    polygon_relevance: LabelValue
    landuse_reason: str
    polygon_reason: str
    evidence: str


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """One label bound to its source sentence identifier."""

    sentence_id: str
    landuse_relevance: LabelValue
    polygon_relevance: LabelValue
    landuse_reason: str
    polygon_reason: str
    evidence: str


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Complete identity that makes checkpoint reuse safe."""

    input_sha256: str
    input_dataset_revision: str
    model_repo_id: str
    model_revision: str
    model_file: str
    model_file_sha256: str
    prompt_version: str
    source_commit: str
    engine: str
    engine_version: str
    batch_size: int
    row_limit: int = 0
    llama_parallel: int = 16
    llama_per_slot_context: int = 4096
    llama_total_context: int = 65536
    request_concurrency: int = 16

    def __post_init__(self) -> None:
        from .runtime import (
            SUPPORTED_LLAMA_PARALLEL,
            validate_llama_parallel,
            validate_per_slot_context,
        )

        if self.engine != "llama.cpp":
            raise ValueError("run identity is bound to the llama.cpp engine")
        validate_llama_parallel(self.llama_parallel)
        validate_per_slot_context(self.llama_per_slot_context)
        if (
            self.llama_total_context
            != self.llama_parallel * self.llama_per_slot_context
        ):
            raise ValueError(
                "llama total context must equal parallel times per-slot context"
            )
        if (
            self.request_concurrency < 1
            or self.request_concurrency > self.llama_parallel
        ):
            raise ValueError(
                "request concurrency must be between 1 and the parallel slot count"
            )
        # Silence unused-import lint for SUPPORTED_LLAMA_PARALLEL while keeping
        # the public constant reachable from the runtime module.
        del SUPPORTED_LLAMA_PARALLEL

    def to_dict(self) -> dict[str, str | int]:
        """Return the stable JSON representation."""

        return asdict(self)
