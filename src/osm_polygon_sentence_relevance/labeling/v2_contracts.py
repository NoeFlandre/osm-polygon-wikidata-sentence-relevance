"""Immutable contracts for the V2 binary place-description label lane.

V1 keeps its historical two-question JSON contract.  V2 deliberately uses a
separate record type because its decision is a single binary classification
whose evidence is the pair of model scores, not generated text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

V2_LOGIT_PROMPT_VERSION = "worldwide-place-description-logit-v2"
V2_MODEL_REPO_ID = "ggml-org/Qwen3.6-27B-GGUF"
V2_MODEL_REVISION = "4c8d89a3b10d66695ded02bacee44f9dcf64848b"
V2_MODEL_FILE = "Qwen3.6-27B-Q4_K_M.gguf"
V2_MODEL_FILE_SHA256 = (
    "65b753ea835627f7b511143c6ceb976525c7f21f5df8c664bc0a9c23d1c49921"
)

BinaryLabel = Literal["yes", "no"]


@dataclass(frozen=True, slots=True)
class V2LogitRecord:
    """One binary decision and its two-class score evidence."""

    sentence_id: str
    place_relevance: BinaryLabel
    yes_logprob: float
    no_logprob: float

    def __post_init__(self) -> None:
        if not isinstance(self.sentence_id, str) or not self.sentence_id:
            raise ValueError("sentence_id must be a non-empty string")
        if self.place_relevance not in {"yes", "no"}:
            raise ValueError("place_relevance must be yes or no")
        for name, value in (
            ("yes_logprob", self.yes_logprob),
            ("no_logprob", self.no_logprob),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

    @property
    def logit_margin(self) -> float:
        """Return the yes-minus-no score margin."""

        return float(self.yes_logprob) - float(self.no_logprob)

    @property
    def two_class_probability(self) -> float:
        """Return sigmoid(margin), a relative yes-vs-no probability."""

        margin = self.logit_margin
        if margin >= 0:
            scale = math.exp(-margin)
            return 1.0 / (1.0 + scale)
        scale = math.exp(margin)
        return scale / (1.0 + scale)


__all__ = [
    "BinaryLabel",
    "V2_LOGIT_PROMPT_VERSION",
    "V2_MODEL_FILE",
    "V2_MODEL_FILE_SHA256",
    "V2_MODEL_REPO_ID",
    "V2_MODEL_REVISION",
    "V2LogitRecord",
]
