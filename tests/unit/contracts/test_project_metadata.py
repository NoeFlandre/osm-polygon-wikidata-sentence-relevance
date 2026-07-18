"""Tests for pyproject.toml project metadata.

Validates that the build configuration follows uv conventions so plain
``uv sync`` works without ``--extra dev`` and that the ``segmentation``
extra installs the SaT runtime the default segmenter needs.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT_PATH = Path(__file__).resolve().parents[3] / "pyproject.toml"


def _load_pyproject() -> dict:
    with open(PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


class TestDevelopmentDependencies:
    """pytest must be available after plain ``uv sync``."""

    def test_dependency_groups_table_exists(self):
        data = _load_pyproject()
        assert "dependency-groups" in data, (
            "pyproject.toml must have a [dependency-groups] table"
        )

    def test_pytest_in_dev_group(self):
        groups = _load_pyproject().get("dependency-groups", {})
        dev = groups.get("dev", [])
        assert any(isinstance(d, str) and "pytest" in d for d in dev), (
            "pytest must be declared in [dependency-groups] dev"
        )

    def test_pytest_not_only_in_optional_dependencies(self):
        opt_deps = _load_pyproject().get("project", {}).get("optional-dependencies", {})
        dev_opt = opt_deps.get("dev", [])
        assert not any(isinstance(d, str) and "pytest" in d for d in dev_opt), (
            "pytest must not be declared under [project.optional-dependencies] dev; "
            "use [dependency-groups] dev instead so plain 'uv sync' includes it"
        )


class TestSegmentationExtra:
    """The ``segmentation`` extra installs the SaT runtime directly.

    ``wtpsplit`` brings the SaT adapter and ``torch`` is its required
    PyTorch runtime; core and ``hub``-only installs stay lightweight.
    SaT model weights themselves are still downloaded separately on
    first model construction.
    """

    @staticmethod
    def _extra(name: str) -> list[str]:
        return (
            _load_pyproject()
            .get("project", {})
            .get("optional-dependencies", {})
            .get(name, [])
        )

    def test_segmentation_declares_wtpsplit(self):
        assert "wtpsplit>=2.2.1,<3" in self._extra("segmentation"), (
            "'segmentation' extra must declare 'wtpsplit>=2.2.1,<3'"
        )

    def test_segmentation_declares_torch(self):
        assert "torch>=2.2,<3" in self._extra("segmentation"), (
            "'segmentation' extra must declare 'torch>=2.2,<3' so the "
            "SaT adapter can construct its PyTorch-backed model"
        )


class TestCoreAndHubStayLightweight:
    """Core stays exactly ``pyarrow``; the ``hub`` extra supports
    acquisition and publishing but stays separate from segmentation."""

    @staticmethod
    def _dependencies() -> list[str]:
        return _load_pyproject().get("project", {}).get("dependencies", [])

    @staticmethod
    def _extra(name: str) -> list[str]:
        return (
            _load_pyproject()
            .get("project", {})
            .get("optional-dependencies", {})
            .get(name, [])
        )

    def test_core_is_exactly_pyarrow(self):
        assert self._dependencies() == ["pyarrow"], (
            "Core dependencies must remain exactly ['pyarrow']; "
            f"found {self._dependencies()}"
        )

    def test_hub_extra_does_not_declare_segmentation_runtime(self):
        hub = self._extra("hub")
        assert not any(d.startswith("torch") for d in hub), (
            f"'hub' extra must not directly declare torch; found: {hub}"
        )
        assert not any(d.startswith("wtpsplit") for d in hub), (
            f"'hub' extra must not directly declare wtpsplit; found: {hub}"
        )
