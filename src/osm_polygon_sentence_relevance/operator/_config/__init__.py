"""Focused implementation package for operator configuration."""

from osm_polygon_sentence_relevance.operator._config.defaults import (
    DATA_ROOT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LABEL_MODEL_FILE,
    DEFAULT_LABEL_MODEL_FILE_SHA256,
    DEFAULT_LABEL_MODEL_REPO_ID,
    DEFAULT_LABEL_MODEL_REVISION,
    DEFAULT_LLAMA_PARALLEL,
    DEFAULT_LLAMA_PER_SLOT_CONTEXT,
    DEFAULT_ROW_LIMIT,
    DEFAULT_SPLIT_MODEL,
    DEFAULT_TOKENIZER_REPO_ID,
    DEFAULT_TOKENIZER_REVISION,
    INPUT_DATASET_ID,
    OUTPUT_DATASET_ID,
    SUPPORTED_LLAMA_PARALLEL,
)
from osm_polygon_sentence_relevance.operator._config.defaults import (
    PROMPT_VERSION as PROMPT_VERSION,
)
from osm_polygon_sentence_relevance.operator._config.enums import Scope, Stage
from osm_polygon_sentence_relevance.operator._config.models import (
    Grid5000Requirements,
    OperatorConfig,
    RunIdentity,
)

__all__ = [
    "Scope",
    "Stage",
    "Grid5000Requirements",
    "RunIdentity",
    "OperatorConfig",
    "DATA_ROOT",
    "INPUT_DATASET_ID",
    "OUTPUT_DATASET_ID",
    "DEFAULT_SPLIT_MODEL",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LLAMA_PARALLEL",
    "DEFAULT_LLAMA_PER_SLOT_CONTEXT",
    "DEFAULT_ROW_LIMIT",
    "DEFAULT_LABEL_MODEL_REPO_ID",
    "DEFAULT_LABEL_MODEL_REVISION",
    "DEFAULT_LABEL_MODEL_FILE",
    "DEFAULT_LABEL_MODEL_FILE_SHA256",
    "DEFAULT_TOKENIZER_REPO_ID",
    "DEFAULT_TOKENIZER_REVISION",
    "SUPPORTED_LLAMA_PARALLEL",
]
