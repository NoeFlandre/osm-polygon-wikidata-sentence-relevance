"""Public-import smoke tests for the publishing package (Phase 7C).

Asserts that:
- The base package and the ``publishing`` package import cleanly
  without pulling in ``huggingface_hub``.
- The public symbols are reachable via the canonical
  ``osm_polygon_sentence_relevance.publishing`` path.
- The ``publishing`` package ships a ``huggingface.py`` module
  implementing the public API.
"""

from __future__ import annotations

import importlib
import sys

import pytest


class TestPublicImportSmoke:
    def test_base_package_import_no_optional_deps(self):
        """Importing the base package must not pull in optional extras."""
        assert "huggingface_hub" not in sys.modules
        assert "wtpsplit" not in sys.modules
        importlib.import_module("osm_polygon_sentence_relevance")
        assert "huggingface_hub" not in sys.modules
        assert "wtpsplit" not in sys.modules

    def test_publishing_package_import_no_optional_deps(self):
        """Importing the publishing package must not pull in
        ``huggingface_hub``.
        """
        mod = importlib.import_module("osm_polygon_sentence_relevance.publishing")
        assert "huggingface_hub" not in sys.modules
        assert mod is not None

    def test_publishing_huggingface_module_import_no_optional_deps(self):
        """Importing the Hugging Face implementation module must not
        pull in ``huggingface_hub`` either.
        """
        importlib.import_module("osm_polygon_sentence_relevance.publishing.huggingface")
        assert "huggingface_hub" not in sys.modules

    def test_publishing_public_symbols_reachable(self):
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            PublicationResult,
            publish_export_directory,
        )

        assert PublicationError is not None
        assert PublicationResult is not None
        assert callable(publish_export_directory)

    def test_publication_error_inherits_value_error(self):
        from osm_polygon_sentence_relevance.errors import PublicationError

        assert issubclass(PublicationError, ValueError)

    def test_publication_result_is_frozen_and_slotted(self):
        from dataclasses import FrozenInstanceError

        from osm_polygon_sentence_relevance.publishing import PublicationResult

        assert PublicationResult.__slots__ is not None
        r = PublicationResult(
            dataset_id="x",
            target_revision="main",
            commit_id="oid",
            commit_url="url",
            row_count=0,
            sha256="0" * 64,
        )
        with pytest.raises(FrozenInstanceError):
            r.commit_id = "tampered"  # type: ignore[misc]
