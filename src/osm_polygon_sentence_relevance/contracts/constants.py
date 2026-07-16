"""Pipeline constants.

Centralises dataset identifiers, version strings, allowed source names,
schema names, and the allowlist of input subdirectory paths.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Dataset identifiers
# ---------------------------------------------------------------------------
INPUT_DATASET_ID: str = "NoeFlandre/osm-polygon-wikidata-only"
OUTPUT_DATASET_ID: str = "NoeFlandre/osm-polygon-wikidata-sentence-relevance"
DEFAULT_INPUT_REVISION: str = "main"

# ---------------------------------------------------------------------------
# Pipeline version (matches package __version__ for now)
# ---------------------------------------------------------------------------
PIPELINE_VERSION: str = "0.1.0"

# ---------------------------------------------------------------------------
# Allowed source labels for the sentence output table
# ---------------------------------------------------------------------------
ALLOWED_SOURCES: frozenset[str] = frozenset({"wikipedia", "wikivoyage"})

# ---------------------------------------------------------------------------
# Logical table names recognised by the schema registry
# ---------------------------------------------------------------------------
SCHEMA_NAMES: tuple[str, ...] = (
    "polygons",
    "polygon_articles",
    "wikipedia_documents",
    "wikivoyage_documents",
    "wikipedia_sections",
    "wikivoyage_sections",
)

# ---------------------------------------------------------------------------
# Allowed input subdirectory paths (relative to the dataset root).
# The obsolete ``articles/`` directory is intentionally excluded.
# ---------------------------------------------------------------------------
ALLOWED_INPUT_PATHS: tuple[str, ...] = (
    "polygons",
    "polygon_articles",
    "wikipedia/documents",
    "wikipedia/sections",
    "wikivoyage/documents",
    "wikivoyage/sections",
)

__all__ = [
    "INPUT_DATASET_ID",
    "OUTPUT_DATASET_ID",
    "DEFAULT_INPUT_REVISION",
    "PIPELINE_VERSION",
    "ALLOWED_SOURCES",
    "SCHEMA_NAMES",
    "ALLOWED_INPUT_PATHS",
]
