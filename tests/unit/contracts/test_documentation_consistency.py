"""Focused consistency checks between documentation, package metadata,
and project configuration. Validates structural / stable facts only; it
does not assert prose.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text()


def _resolve(rel: str) -> Path:
    return ROOT / rel


# ---------------------------------------------------------------------------
# Markdown link resolution (relative file paths only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "doc_rel",
    [
        "README.md",
        "docs/index.md",
        "docs/architecture/overview.md",
        "docs/architecture/decisions/0001-domain-package-layout.md",
        "docs/guides/getting-started.md",
        "docs/guides/development.md",
        "docs/guides/reproducibility.md",
        "docs/reference/api.md",
        "docs/reference/cli.md",
        "docs/reference/data-contract.md",
    ],
)
def test_doc_exists(doc_rel: str) -> None:
    assert _resolve(doc_rel).is_file()


_MD_LINK_RE = re.compile(r"(?<!\!)\[[^\]]*\]\((?!https?://|#|mailto:)([^)\s]+)\)")


@pytest.mark.parametrize(
    "doc_rel",
    [
        "README.md",
        "docs/index.md",
        "docs/architecture/overview.md",
        "docs/architecture/decisions/0001-domain-package-layout.md",
        "docs/guides/getting-started.md",
        "docs/guides/development.md",
        "docs/guides/reproducibility.md",
        "docs/reference/api.md",
        "docs/reference/cli.md",
        "docs/reference/data-contract.md",
    ],
)
def test_relative_markdown_links_resolve(doc_rel: str) -> None:
    """Every relative Markdown link target inside a doc must resolve."""
    doc = _resolve(doc_rel)
    text = _read(doc)
    base_dir = doc.parent
    for match in _MD_LINK_RE.finditer(text):
        link = match.group(1).split("#", 1)[0]
        if not link:
            continue
        target = (base_dir / link).resolve()
        # Permit ".md" link to existing file or directory.
        assert target.exists(), f"{doc_rel}: broken relative link -> {link}"


# ---------------------------------------------------------------------------
# Dataset ID consistency
# ---------------------------------------------------------------------------


def test_readme_dataset_ids_match_constants() -> None:
    """README dataset links must use the canonical constants."""
    constants = _read(_resolve("src/osm_polygon_sentence_relevance/constants.py"))
    readme = _read(_resolve("README.md"))

    input_id_match = re.search(r'INPUT_DATASET_ID:\s*str\s*=\s*"([^"]+)"', constants)
    output_id_match = re.search(r'OUTPUT_DATASET_ID:\s*str\s*=\s*"([^"]+)"', constants)
    assert input_id_match, "INPUT_DATASET_ID not found in constants.py"
    assert output_id_match, "OUTPUT_DATASET_ID not found in constants.py"

    assert input_id_match.group(1) in readme
    assert output_id_match.group(1) in readme


# ---------------------------------------------------------------------------
# pyproject version == package __version__
# ---------------------------------------------------------------------------


def test_package_version_matches_pyproject() -> None:
    pkg_init = _read(_resolve("src/osm_polygon_sentence_relevance/__init__.py"))
    pyproject = tomllib.loads(_read(_resolve("pyproject.toml")))

    match = re.search(r'__version__\s*=\s*"([^"]+)"', pkg_init)
    assert match, "package __version__ not found"
    pkg_version = match.group(1)
    toml_version = pyproject["project"]["version"]
    assert pkg_version == toml_version


# ---------------------------------------------------------------------------
# py.typed is shipped
# ---------------------------------------------------------------------------


def test_py_typed_marker_exists() -> None:
    assert (_resolve("src/osm_polygon_sentence_relevance/py.typed")).is_file()


def test_py_typed_is_in_package_data() -> None:
    pyproject = tomllib.loads(_read(_resolve("pyproject.toml")))
    pd = pyproject["tool"]["setuptools"]["package-data"][
        "osm_polygon_sentence_relevance"
    ]
    assert "py.typed" in pd


# ---------------------------------------------------------------------------
# CLI documentation consistency
# ---------------------------------------------------------------------------


def _cli_help_lines() -> list[str]:
    import subprocess

    proc = subprocess.run(
        ["uv", "run", "osm-polygon-sentence-relevance", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.splitlines()


def _documented_cli_flags() -> set[str]:
    """Extract every ``--foo`` flag referenced inside docs/reference/cli.md."""
    cli_ref = _read(_resolve("docs/reference/cli.md"))
    return set(re.findall(r"--[a-z][a-z0-9-]+", cli_ref))


def _actual_cli_flags() -> set[str]:
    """Extract every ``--foo`` flag surfaced by the real ``--help`` output.

    Argparse's own ``-h/--help`` entries are ignored because they are
    produced by argparse itself rather than the project's CLI surface.
    """
    return {
        flag
        for flag in re.findall(r"--[a-z][a-z0-9-]+", "\n".join(_cli_help_lines()))
        if flag not in {"--help"}
    }


def test_cli_reference_flags_match_parser() -> None:
    """Documented CLI flags must match the parser's actual flags exactly.

    After excluding argparse's own ``--help`` entry, the set of flags
    referenced in ``docs/reference/cli.md`` must equal the set of flags
    surfaced by the real ``--help`` output. Newly added but undocumented
    flags and stale-documented flags both fail the test, with both sets
    listed in the failure message for easy triage.
    """
    documented = _documented_cli_flags()
    actual = _actual_cli_flags()
    stale = sorted(documented - actual)  # in docs, missing from parser
    undocumented = sorted(actual - documented)  # in parser, missing from docs
    assert not stale, (
        f"flags documented in docs/reference/cli.md but no longer surfaced "
        f"by --help: {stale}\n"
        f"Update docs/reference/cli.md and the parser together."
    )
    assert not undocumented, (
        f"flags surfaced by --help but not documented in "
        f"docs/reference/cli.md: {undocumented}\n"
        f"Update docs/reference/cli.md and the parser together."
    )


# ---------------------------------------------------------------------------
# Local-only path / stale doc / dataset ID hygiene
# ---------------------------------------------------------------------------


LOCAL_DOC_MARKERS = (
    ".local-docs/",
    "/Volumes/",
    "/tmp/",
    "Seagate",
    "main.py",
    "config.py",
)


@pytest.mark.parametrize(
    "doc_rel",
    [
        "README.md",
        "docs/index.md",
        "docs/architecture/overview.md",
        "docs/architecture/decisions/0001-domain-package-layout.md",
        "docs/guides/getting-started.md",
        "docs/guides/development.md",
        "docs/guides/reproducibility.md",
        "docs/reference/api.md",
        "docs/reference/cli.md",
        "docs/reference/data-contract.md",
    ],
)
def test_no_local_machine_paths_or_stale_root_scripts(doc_rel: str) -> None:
    text = _read(_resolve(doc_rel))
    for marker in LOCAL_DOC_MARKERS:
        assert marker not in text, (
            f"{doc_rel}: must not contain local/private marker '{marker}'"
        )


@pytest.mark.parametrize(
    "doc_rel",
    [
        "README.md",
        "docs/index.md",
        "docs/guides/getting-started.md",
        "docs/guides/reproducibility.md",
        "docs/reference/cli.md",
        "docs/reference/api.md",
    ],
)
def test_no_unpublished_output_dataset_claim(doc_rel: str) -> None:
    text = _read(_resolve(doc_rel))
    # The output dataset link is allowed in README (as a link) but
    # documentation must not claim it has been *published*.
    forbidden = "the output dataset is published"
    assert forbidden not in text.lower(), (
        f"{doc_rel}: must not claim the output dataset is already published"
    )
