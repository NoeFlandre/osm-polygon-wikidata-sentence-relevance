"""Tests for pyproject.toml project metadata.

Validates that the build configuration follows uv conventions so that
``uv sync && uv run pytest -q`` works without ``--extra dev``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


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
        data = _load_pyproject()
        groups = data.get("dependency-groups", {})
        dev = groups.get("dev", [])
        pytest_entries = [d for d in dev if isinstance(d, str) and "pytest" in d]
        assert pytest_entries, (
            "pytest must be declared in [dependency-groups] dev"
        )

    def test_pytest_not_only_in_optional_dependencies(self):
        data = _load_pyproject()
        # pytest should NOT depend on installing an optional extra
        opt_deps = data.get("project", {}).get("optional-dependencies", {})
        dev_opt = opt_deps.get("dev", [])
        pytest_in_opt = [d for d in dev_opt if isinstance(d, str) and "pytest" in d]
        assert not pytest_in_opt, (
            "pytest must not be declared under [project.optional-dependencies] dev; "
            "use [dependency-groups] dev instead so plain 'uv sync' includes it"
        )
