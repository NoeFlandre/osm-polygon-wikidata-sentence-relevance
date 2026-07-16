"""Schema registry and validation API.

Keyed by logical table name; each entry is an immutable PyArrow schema
from :mod:`osm_polygon_sentence_relevance.contracts.schemas.input` or
:mod:`osm_polygon_sentence_relevance.contracts.schemas.pipeline`.
"""

from __future__ import annotations

import pyarrow as pa

from osm_polygon_sentence_relevance.contracts.errors import (
    IncompatibleTypesError,
    MissingColumnsError,
    UnknownTableError,
)
from osm_polygon_sentence_relevance.contracts.schemas.input import (
    POLYGON_ARTICLES_SCHEMA,
    POLYGONS_SCHEMA,
    SECTIONS_SCHEMA,
    WIKIPEDIA_DOCUMENTS_SCHEMA,
    WIKIVOYAGE_DOCUMENTS_SCHEMA,
)
from osm_polygon_sentence_relevance.contracts.schemas.pipeline import (
    JOINED_SECTIONS_SCHEMA,
    OUTPUT_SENTENCE_SCHEMA,
    SEGMENTED_SENTENCES_SCHEMA,
)

# The registry reuses the section schema for both Wikipedia and Wikivoyage.
SCHEMA_REGISTRY: dict[str, pa.Schema] = {
    "polygons": POLYGONS_SCHEMA,
    "polygon_articles": POLYGON_ARTICLES_SCHEMA,
    "wikipedia_documents": WIKIPEDIA_DOCUMENTS_SCHEMA,
    "wikivoyage_documents": WIKIVOYAGE_DOCUMENTS_SCHEMA,
    "wikipedia_sections": SECTIONS_SCHEMA,
    "wikivoyage_sections": SECTIONS_SCHEMA,
}

# Canonical pipeline schemas exposed for direct reference by stages.
__all__ = [
    "SCHEMA_REGISTRY",
    "OUTPUT_SENTENCE_SCHEMA",
    "JOINED_SECTIONS_SCHEMA",
    "SEGMENTED_SENTENCES_SCHEMA",
    "POLYGONS_SCHEMA",
    "POLYGON_ARTICLES_SCHEMA",
    "WIKIPEDIA_DOCUMENTS_SCHEMA",
    "WIKIVOYAGE_DOCUMENTS_SCHEMA",
    "SECTIONS_SCHEMA",
    "validate_table_schema",
]


def validate_table_schema(table_name: str, actual_schema: pa.Schema) -> None:
    """Validate *actual_schema* against the registered contract for *table_name*.

    Raises
    ------
    UnknownTableError
        If *table_name* is not in :data:`SCHEMA_REGISTRY`.
    MissingColumnsError
        If any required columns are absent from *actual_schema*.
    IncompatibleTypesError
        If any present columns have a type incompatible with the contract.

    Notes
    -----
    * Extra columns in *actual_schema* are silently allowed so the pipeline
      remains forward-compatible with upstream additions.
    * Column order in *actual_schema* need not match the contract.
    """
    if table_name not in SCHEMA_REGISTRY:
        raise UnknownTableError(table_name)

    expected = SCHEMA_REGISTRY[table_name]
    actual_names = {f.name for f in actual_schema}

    # --- missing columns ---
    expected_names = {f.name for f in expected}
    missing = sorted(expected_names - actual_names)
    if missing:
        raise MissingColumnsError(table_name, missing)

    # --- type mismatches ---
    mismatches: list[tuple[str, str, str]] = []
    for field in expected:
        actual_field = actual_schema.field(field.name)
        if actual_field.type != field.type:
            mismatches.append((field.name, str(field.type), str(actual_field.type)))
    if mismatches:
        raise IncompatibleTypesError(table_name, mismatches)
