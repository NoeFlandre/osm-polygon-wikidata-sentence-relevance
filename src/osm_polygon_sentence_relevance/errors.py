"""Compatibility facade: ``osm_polygon_sentence_relevance.errors``.

The canonical hierarchy now lives in
:mod:`osm_polygon_sentence_relevance.contracts.errors`. Import from there in
new code; this module re-exports the stable public symbols so existing
imports keep working.
"""

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

__all__ = [
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
]
