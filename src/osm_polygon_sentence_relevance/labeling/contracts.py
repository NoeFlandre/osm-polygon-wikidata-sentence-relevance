"""Small immutable contracts shared by the labeling pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from .sampling import (
    DEFAULT_H3_RESOLUTION,
    DEFAULT_SAMPLE_SEED,
    SAMPLING_VERSION,
)


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
    # ``None`` preserves the pre-v2 identity contract for callers that build a
    # legacy identity directly. The CLI always records an explicit v2 target.
    sampling_target: int | None = None
    sampling_seed: str | None = None
    h3_resolution: int | None = None
    sampling_version: str | None = None
    release_lane: str | None = None

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
        sampling_enabled = any(
            value is not None
            for value in (
                self.sampling_target,
                self.sampling_seed,
                self.h3_resolution,
                self.sampling_version,
            )
        )
        if sampling_enabled:
            from .sampling import SamplingConfig

            SamplingConfig(
                # The target is a mutable continuation budget. Validate the
                # immutable sampling dimensions even when a persisted
                # checkpoint identity has no current target field.
                target=self.sampling_target or 1,
                seed=self.sampling_seed or DEFAULT_SAMPLE_SEED,
                h3_resolution=(
                    self.h3_resolution
                    if self.h3_resolution is not None
                    else DEFAULT_H3_RESOLUTION
                ),
                version=self.sampling_version or SAMPLING_VERSION,
            )
        if self.release_lane not in {None, "v1-afghanistan", "v2-worldwide"}:
            raise ValueError("release lane is invalid")

    def to_dict(self) -> dict[str, str | int]:
        """Return the stable JSON representation."""

        payload = asdict(self)
        sampling_enabled = any(
            value is not None
            for value in (
                self.sampling_target,
                self.sampling_seed,
                self.h3_resolution,
                self.sampling_version,
            )
        )
        if not sampling_enabled:
            for field_name in (
                "sampling_target",
                "sampling_seed",
                "h3_resolution",
                "sampling_version",
            ):
                payload.pop(field_name, None)
        else:
            if self.sampling_target is None:
                payload.pop("sampling_target", None)
            payload["sampling_seed"] = self.sampling_seed or DEFAULT_SAMPLE_SEED
            payload["h3_resolution"] = (
                self.h3_resolution
                if self.h3_resolution is not None
                else DEFAULT_H3_RESOLUTION
            )
            payload["sampling_version"] = self.sampling_version or SAMPLING_VERSION
        if self.release_lane is None:
            payload.pop("release_lane", None)
        return payload

    def checkpoint_dict(self) -> dict[str, str | int]:
        """Return identity fields that are immutable across target expansion."""

        payload = self.to_dict()
        payload.pop("sampling_target", None)
        return payload
