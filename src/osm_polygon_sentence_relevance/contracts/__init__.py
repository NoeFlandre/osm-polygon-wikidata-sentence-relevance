"""Canonical home for cross-cutting pipeline contracts.

Exposes the stable public surface for constants, errors, and schemas. New
code should import from the submodules directly
(``contracts.constants``, ``contracts.errors``, ``contracts.schemas``);
this package re-exports the most commonly used names for convenience.
"""

from osm_polygon_sentence_relevance.contracts.constants import (
    ALLOWED_INPUT_PATHS,
    ALLOWED_SOURCES,
    DEFAULT_INPUT_REVISION,
    INPUT_DATASET_ID,
    OUTPUT_DATASET_ID,
    PIPELINE_VERSION,
    SCHEMA_NAMES,
)
from osm_polygon_sentence_relevance.contracts.errors import (
    AcquisitionError,
    ConfigurationError,
    ExportError,
    FinalizationError,
    IncompatibleTypesError,
    JoinIntegrityError,
    MissingColumnsError,
    PreprocessingError,
    PublicationError,
    SchemaContractError,
    SegmentationError,
    ShardDiscoveryError,
    UnknownTableError,
)
from osm_polygon_sentence_relevance.contracts.schemas import (
    JOINED_SECTIONS_SCHEMA,
    OUTPUT_SENTENCE_SCHEMA,
    SCHEMA_REGISTRY,
    SEGMENTED_SENTENCES_SCHEMA,
    validate_table_schema,
)

__all__ = [
    # constants
    "INPUT_DATASET_ID",
    "OUTPUT_DATASET_ID",
    "DEFAULT_INPUT_REVISION",
    "PIPELINE_VERSION",
    "ALLOWED_SOURCES",
    "SCHEMA_NAMES",
    "ALLOWED_INPUT_PATHS",
    # errors
    "ConfigurationError",
    "SchemaContractError",
    "UnknownTableError",
    "MissingColumnsError",
    "IncompatibleTypesError",
    "PreprocessingError",
    "SegmentationError",
    "ShardDiscoveryError",
    "JoinIntegrityError",
    "FinalizationError",
    "ExportError",
    "AcquisitionError",
    "PublicationError",
    # schemas
    "SCHEMA_REGISTRY",
    "OUTPUT_SENTENCE_SCHEMA",
    "JOINED_SECTIONS_SCHEMA",
    "SEGMENTED_SENTENCES_SCHEMA",
    "validate_table_schema",
]
