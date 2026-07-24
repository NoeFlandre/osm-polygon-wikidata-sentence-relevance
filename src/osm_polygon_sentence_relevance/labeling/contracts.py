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

    def to_dict(self) -> dict[str, str | int]:
        """Return the stable JSON representation."""

        return asdict(self)
