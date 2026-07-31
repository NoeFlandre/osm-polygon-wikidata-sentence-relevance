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
    operator = _names(project["project"]["optional-dependencies"]["operator"])
    development = _names(project["dependency-groups"]["dev"])
    assert runtime == {"pyarrow"}
    assert {"click", "typer", "rich", "tqdm"} <= operator
    assert {"pytest", "pytest-cov", "ruff", "ty", "pre-commit"} <= development
    assert "mypy" not in runtime | operator | development


def test_justfile_exposes_required_recipes() -> None:
    text = (ROOT / "justfile").read_text(encoding="utf-8")
    for recipe in (
        "sync:",
        "format:",
        "format-check:",
        "lint:",
        "typecheck:",
        "test:",
        "check:",
        "build:",
        "verify-dist:",
        "ci:",
    ):
        assert recipe in text
    assert "mypy" not in text


def test_precommit_uses_locked_project_commands() -> None:
    text = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    for entry in (
        "entry: uv run ruff format --check .",
        "entry: uv run ruff check .",
        "entry: uv run ty check",
        "tests/unit/operator/test_console.py",
        "tests/unit/operator/test_cli.py",
        "tests/unit/test_developer_tooling.py",
    ):
        assert entry in text
    assert text.count("language: system") == 4
    assert text.count("pass_filenames: false") == 4


def test_ci_uses_just_recipes_and_keeps_locked_sync() -> None:
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "uv sync --locked --all-extras --dev" in text
    assert "cargo install just --locked --version 1.40.0" in text
    assert "run: just check" in text
    assert "run: just verify-dist" in text
    assert "osm-polygon-grid5000 --help" in text
    assert "--with click==8.4.2 --with typer==0.26.8" in text
    assert "--with rich==15.0.0 --with tqdm==4.68.4" in text
    assert "mypy" not in text


def test_contributing_documents_one_supported_toolchain() -> None:
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    for command in (
        "uv sync --locked --all-extras --dev",
        "just check",
        "just ci",
        "uv run pre-commit install",
        "uv run ty check",
    ):
        assert command in text
    assert "mypy" not in text.lower()


def test_readme_names_operator_and_primary_quality_command() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "osm-polygon-grid5000" in text
    assert "just check" in text


def test_development_guide_matches_shared_quality_workflow() -> None:
    text = (ROOT / "docs/guides/development.md").read_text(encoding="utf-8")
    assert "uv sync --locked --all-extras --dev" in text
    assert "just check" in text
    assert "just ci" in text
    assert "uv run pre-commit install" in text
    assert "mypy" not in text.lower()


def test_mkdocs_site_and_pages_workflow_are_locked_and_strict() -> None:
    project = _pyproject()
    docs_extra = _names(project["project"]["optional-dependencies"]["docs"])
    assert "mkdocs-material" in docs_extra
    assert (
        project["project"]["urls"]["Documentation"] == "https://noeflandre.github.io/"
        "osm-polygon-wikidata-sentence-relevance/"
    )

    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "site_name: OSM Polygon Sentence Relevance" in config
    assert (
        "site_url: https://noeflandre.github.io/"
        "osm-polygon-wikidata-sentence-relevance/"
    ) in config
    assert "nav:" in config
    assert "- Getting started: guides/getting-started.md" in config
    assert "- CLI reference: reference/cli.md" in config
    assert "- Data contract: reference/data-contract.md" in config

    workflow = (ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    assert "branches: [main]" in workflow
    assert "workflow_dispatch:" in workflow
    assert "pages: write" in workflow
    assert "id-token: write" in workflow
    assert "mkdocs build --strict" in workflow
    assert "actions/upload-pages-artifact" in workflow
    assert "actions/deploy-pages" in workflow

    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    assert "docs-build:" in justfile
    assert "mkdocs build --strict" in justfile
