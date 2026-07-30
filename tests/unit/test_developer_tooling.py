"""Repository-level contracts for the supported developer toolchain."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _names(requirements: list[str]) -> set[str]:
    return {
        requirement.split("[", 1)[0]
        .split("<", 1)[0]
        .split(">", 1)[0]
        .split("=", 1)[0]
        .strip()
        .lower()
        for requirement in requirements
    }


def test_required_runtime_and_development_tools_are_direct() -> None:
    project = _pyproject()
    runtime = _names(project["project"]["dependencies"])
    development = _names(project["dependency-groups"]["dev"])
    assert {"typer", "rich", "tqdm"} <= runtime
    assert {"pytest", "pytest-cov", "ruff", "ty", "pre-commit"} <= development
    assert "mypy" not in runtime | development
