"""Immutable defaults used by operator configuration models."""

from pathlib import Path
from typing import Final

from osm_polygon_sentence_relevance.contracts.constants import (
    INPUT_DATASET_ID as _INPUT_DATASET_ID,
)
from osm_polygon_sentence_relevance.contracts.constants import (
    OUTPUT_DATASET_ID as _OUTPUT_DATASET_ID,
)
from osm_polygon_sentence_relevance.labeling.prompt import PROMPT_VERSION
from osm_polygon_sentence_relevance.labeling.runtime import (
    MIN_PER_SLOT_CONTEXT,
)
from osm_polygon_sentence_relevance.labeling.runtime import (
    SUPPORTED_LLAMA_PARALLEL as _RUNTIME_SUPPORTED_LLAMA_PARALLEL,
)
from osm_polygon_sentence_relevance.labeling.sampling import (
    DEFAULT_H3_RESOLUTION,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_TARGET,
)
from osm_polygon_sentence_relevance.labeling.sampling import (
    SAMPLING_VERSION as _SAMPLING_VERSION,
)

DATA_ROOT: Final[Path] = Path(
    "/Volumes/Seagate M3/projects/osm-polygon-wikidata-sentence-relevance"
)
INPUT_DATASET_ID: Final[str] = _INPUT_DATASET_ID
OUTPUT_DATASET_ID: Final[str] = _OUTPUT_DATASET_ID

DEFAULT_SPLIT_MODEL: Final[str] = "sat-12l-sm"
DEFAULT_BATCH_SIZE: Final[int] = 128
DEFAULT_ROW_LIMIT: Final[int] = 0
DEFAULT_SAMPLING_TARGET: Final[int] = DEFAULT_SAMPLE_TARGET
DEFAULT_SAMPLING_SEED: Final[str] = DEFAULT_SAMPLE_SEED
DEFAULT_SAMPLING_H3_RESOLUTION: Final[int] = DEFAULT_H3_RESOLUTION
SAMPLING_VERSION: Final[str] = _SAMPLING_VERSION
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

__all__ = [
    "DATA_ROOT",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LABEL_MODEL_FILE",
    "DEFAULT_LABEL_MODEL_FILE_SHA256",
    "DEFAULT_LABEL_MODEL_REPO_ID",
    "DEFAULT_LABEL_MODEL_REVISION",
    "DEFAULT_LLAMA_PARALLEL",
    "DEFAULT_LLAMA_PER_SLOT_CONTEXT",
    "DEFAULT_ROW_LIMIT",
    "DEFAULT_SAMPLING_TARGET",
    "DEFAULT_SAMPLING_SEED",
    "DEFAULT_SAMPLING_H3_RESOLUTION",
    "SAMPLING_VERSION",
    "DEFAULT_SPLIT_MODEL",
    "DEFAULT_TOKENIZER_REPO_ID",
    "DEFAULT_TOKENIZER_REVISION",
    "INPUT_DATASET_ID",
    "OUTPUT_DATASET_ID",
    "PROMPT_VERSION",
    "SUPPORTED_LLAMA_PARALLEL",
]
