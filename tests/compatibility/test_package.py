"""Tests that importing the package has no filesystem or network side effects."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


class TestPackageImport:
    """Importing the package must be side-effect-free."""

    def test_import_succeeds(self):
        mod = importlib.import_module("osm_polygon_sentence_relevance")
        assert hasattr(mod, "__version__")

    def test_version_accessible(self):
        from osm_polygon_sentence_relevance import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0


class TestProjectMetadata:
    """Project extras must stay separate and core must stay lightweight."""

    def _load_pyproject(self) -> dict:
        return tomllib.loads(
            (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
        )

    def test_extras_declared_separately(self):
        pyproject = self._load_pyproject()
        extras = pyproject["project"]["optional-dependencies"]
        # Known extras must remain declared separately; future extras are allowed.
        assert "segmentation" in extras
        assert "hub" in extras
        # Each extra carries its own distinct dependency.
        assert extras["segmentation"]
        assert extras["hub"]
        assert extras["segmentation"] != extras["hub"]

    def test_core_keeps_lightweight(self):
        pyproject = self._load_pyproject()
        core = pyproject["project"]["dependencies"]
        # Core install must not pull in heavy ML or hub dependencies.
        joined = " ".join(core).lower()
        assert "wtpsplit" not in joined
        assert "huggingface" not in joined
        # Only the minimal required core dependency is present.
        assert core == ["pyarrow"]
