"""Immutable operator identity and target-expansion contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from json import dumps

from osm_polygon_sentence_relevance.operator._config.defaults import (
    DEFAULT_SAMPLING_H3_RESOLUTION,
    DEFAULT_SAMPLING_SEED,
    SAMPLING_VERSION,
)
from osm_polygon_sentence_relevance.operator._config.enums import Scope, Stage


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Deterministic identity for a complete operator run.

    The V2 sample target is intentionally provenance, not identity: increasing
    it reuses the same validated checkpoints because the selector is a nested
    deterministic prefix. The input, seed, H3 resolution, and sampling
    version remain immutable identity fields.
    """

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
    sampling_target: int | None = None
    sampling_seed: str | None = None
    sampling_h3_resolution: int | None = None
    sampling_version: str | None = None
    _canonical_json: str = field(default="", init=False, repr=False)
    run_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        canonical = dumps(
            self.checkpoint_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        object.__setattr__(self, "_canonical_json", canonical)
        object.__setattr__(self, "run_id", sha256(canonical.encode()).hexdigest()[:20])

    @property
    def canonical_json(self) -> str:
        """Canonical JSON used to derive run identity."""

        return self._canonical_json

    def to_dict(self) -> dict[str, str | int]:
        """Return public provenance, including the current sample target."""

        payload: dict[str, str | int] = {
            "scope": str(self.scope),
            "stage": str(self.stage),
            "source_commit": self.source_commit,
            "input_dataset_id": self.input_dataset_id,
            "output_dataset_id": self.output_dataset_id,
            "pipeline_version": self.pipeline_version,
            "batch_size": self.batch_size,
        }
        optional_fields = (
            ("region", self.region),
            ("input_dataset_revision", self.input_dataset_revision),
            ("split_model", self.split_model),
            ("model_repo_id", self.model_repo_id),
            ("model_revision", self.model_revision),
            ("model_file", self.model_file),
            ("model_file_sha256", self.model_file_sha256),
            ("tokenizer_repo_id", self.tokenizer_repo_id),
            ("tokenizer_revision", self.tokenizer_revision),
            ("prompt_version", self.prompt_version),
        )
        payload.update(
            {key: value for key, value in optional_fields if value is not None}
        )
        sampling_enabled = any(
            value is not None
            for value in (
                self.sampling_target,
                self.sampling_seed,
                self.sampling_h3_resolution,
                self.sampling_version,
            )
        )
        if sampling_enabled:
            if self.sampling_target is not None:
                payload["sampling_target"] = self.sampling_target
            payload["sampling_seed"] = self.sampling_seed or DEFAULT_SAMPLING_SEED
            payload["sampling_h3_resolution"] = (
                self.sampling_h3_resolution
                if self.sampling_h3_resolution is not None
                else DEFAULT_SAMPLING_H3_RESOLUTION
            )
            payload["sampling_version"] = self.sampling_version or SAMPLING_VERSION
        runtime_fields = (
            ("row_limit", self.row_limit),
            ("llama_parallel", self.llama_parallel),
            ("llama_per_slot_context", self.llama_per_slot_context),
            ("llama_total_context", self.llama_total_context),
            ("request_concurrency", self.request_concurrency),
        )
        payload.update(
            {key: value for key, value in runtime_fields if value is not None}
        )
        return payload

    def checkpoint_dict(self) -> dict[str, str | int]:
        """Return immutable identity fields shared by target expansions."""

        payload = self.to_dict()
        payload.pop("sampling_target", None)
        return payload


__all__ = ["RunIdentity"]
