"""Canonical publishing layer.

Provides programmatic one-commit publishing of a locally validated
export directory to an existing Hugging Face dataset repository. The
CLI does not currently expose publishing flags; this package is the
single programmatic entry point.

Public API
----------

- ``PublicationError``  -- dedicated error for publish failures.
- ``PublicationResult`` -- frozen, slotted dataclass with verified facts.
- ``publish_export_directory(...)`` -- validate then publish in one
  ``create_commit`` call, two add operations only, no deletes, no
  repository creation, no token handling.
"""

from osm_polygon_sentence_relevance.contracts.errors import PublicationError
from osm_polygon_sentence_relevance.publishing.huggingface import (
    PublicationResult,
    publish_export_directory,
)

__all__ = [
    "PublicationError",
    "PublicationResult",
    "publish_export_directory",
]
