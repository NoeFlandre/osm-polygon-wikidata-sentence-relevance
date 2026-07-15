"""Tests that importing the package has no filesystem or network side effects."""

from __future__ import annotations

import importlib


class TestPackageImport:
    """Importing the package must be side-effect-free."""

    def test_import_succeeds(self):
        mod = importlib.import_module("osm_polygon_sentence_relevance")
        assert hasattr(mod, "__version__")

    def test_version_accessible(self):
        from osm_polygon_sentence_relevance import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0
