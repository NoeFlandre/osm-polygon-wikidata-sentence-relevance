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
    constants = _read(
        _resolve("src/osm_polygon_sentence_relevance/contracts/constants.py")
    )
    readme = _read(_resolve("README.md"))

    input_id_match = re.search(r'INPUT_DATASET_ID:\s*str\s*=\s*"([^"]+)"', constants)
    output_id_match = re.search(r'OUTPUT_DATASET_ID:\s*str\s*=\s*"([^"]+)"', constants)
    assert input_id_match, "INPUT_DATASET_ID not found in contracts/constants.py"
    assert output_id_match, "OUTPUT_DATASET_ID not found in contracts/constants.py"

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


# ---------------------------------------------------------------------------
# API reference accuracy: canonical module paths and ownership
# ---------------------------------------------------------------------------


def test_api_reference_uses_contracts_canonical_paths() -> None:
    """docs/reference/api.md must point at the canonical contracts modules."""
    api = _read(_resolve("docs/reference/api.md"))
    assert "contracts.constants" in api
    assert "contracts.errors" in api
    assert "contracts.schemas" in api
    # Settings ownership is canonical at application/settings.
    assert "application/settings" in api or "application.settings" in api


def test_api_reference_sat_segmenter_uses_segmentation_extra() -> None:
    """SaTSentenceSegmenter documentation must name the canonical
    ``segmentation`` extra (which installs ``wtpsplit`` plus its required
    PyTorch runtime), and must not refer to a ``wtpsplit`` extra.

    This guards the install contract:

    - The canonical extra name is ``segmentation`` (the literal pyproject
      extra in ``pyproject.toml``).
    - The extra installs ``wtpsplit`` plus PyTorch (``torch``).
    - The model weights are downloaded separately on first model
      construction; the extra itself does not bundle them.

    The test is intentionally structural: it scans the paragraph that
    mentions ``SaTSentenceSegmenter`` in ``docs/reference/api.md`` and
    asserts stable contracts (presence / absence of substrings) rather
    than exact prose.
    """
    api = _read(_resolve("docs/reference/api.md"))
    # Locate the SaTSentenceSegmenter paragraph (the bullet listing it
    # plus any indented continuation lines belonging to the same bullet).
    lines = api.splitlines()
    sat_indices = [i for i, line in enumerate(lines) if "SaTSentenceSegmenter" in line]
    assert sat_indices, (
        "docs/reference/api.md must reference 'SaTSentenceSegmenter' "
        "in its API listing."
    )
    # Collect the bullet line and subsequent indented continuation
    # lines until the next top-level bullet (a line starting with '- ')
    # or end-of-file.
    collected: list[str] = []
    for start in sat_indices:
        collected.append(lines[start])
        for j in range(start + 1, len(lines)):
            line = lines[j]
            if line.startswith("- "):
                break
            if line.strip() == "":
                break
            collected.append(line)
    paragraph = "\n".join(collected)

    # Must name the canonical `segmentation` extra.
    assert "segmentation" in paragraph, (
        "SaTSentenceSegmenter documentation must reference the "
        "`segmentation` extra (the canonical install target)."
    )
    # Must mention both wtpsplit and torch / PyTorch (the runtime).
    lower = paragraph.lower()
    assert "wtpsplit" in lower, (
        "SaTSentenceSegmenter documentation must mention 'wtpsplit'."
    )
    assert "torch" in lower or "pytorch" in lower, (
        "SaTSentenceSegmenter documentation must mention the required "
        "PyTorch runtime (torch or PyTorch)."
    )
    # Must NOT call it the 'wtpsplit extra'.
    assert "wtpsplit extra" not in lower, (
        "SaTSentenceSegmenter documentation must not refer to the "
        "`wtpsplit` extra; the extra is named `segmentation`."
    )


def test_api_reference_does_not_put_allow_patterns_under_constants() -> None:
    """The API reference must not claim ALLOW_PATTERNS/IGNORE_PATTERNS live
    on ``constants``; they belong to ``ingestion.acquisition``."""
    api = _read(_resolve("docs/reference/api.md"))
    # The erroneous 'constants — ... ALLOW_PATTERNS, IGNORE_PATTERNS ...'
    # phrasing must not appear.
    assert "ALLOW_PATTERNS`,\n  `IGNORE_PATTERNS`" not in api.replace(" ", "")
    # Positive: the correct location is documented.
    assert "acquisition.ALLOW_PATTERNS" in api
    assert "acquisition.IGNORE_PATTERNS" in api


def test_no_stale_seagate_or_external_drive_claims() -> None:
    """No doc may still claim the legacy Seagate/external-drive data dir."""
    markers = ["/Volumes/", "Seagate", "external-drive", "external drive path"]
    for doc_rel in [
        "README.md",
        "docs/index.md",
        "docs/architecture/overview.md",
        "docs/guides/development.md",
        "docs/guides/getting-started.md",
        "docs/guides/reproducibility.md",
        "docs/reference/api.md",
        "docs/reference/cli.md",
        "docs/reference/data-contract.md",
    ]:
        text = _read(_resolve(doc_rel))
        for marker in markers:
            assert marker not in text, (
                f"{doc_rel}: stale machine-path marker '{marker}' must not appear"
            )


# ---------------------------------------------------------------------------
# Public scope & architecture claim hygiene
# ---------------------------------------------------------------------------


def _changelog_unreleased_block(text: str) -> str:
    """Return the text under the first ``## [Unreleased]`` heading in
    CHANGELOG.md, stopping at the next ``## [...]`` heading."""
    match = re.search(
        r"^## \[Unreleased\]\s*$\n(?P<body>.*?)(?=^## \[)", text, re.M | re.S
    )
    assert match, "CHANGELOG.md must define a '## [Unreleased]' section"
    return match.group("body")


def test_changelog_does_not_claim_all_publishing_unimplemented() -> None:
    """The blanket claim that *all* Hugging Face dataset publishing /
    upload is unimplemented is stale now that ``publishing/`` exists.
    Legitimate statements about CLI publishing and repository creation
    remaining unimplemented are still allowed.
    """
    text = _read(_resolve("CHANGELOG.md"))
    stale = "Hugging Face dataset publishing / upload."
    assert stale not in text, (
        "CHANGELOG.md still blanket-claims that all Hugging Face "
        "publishing is unimplemented; replace with precise remaining "
        "boundaries (CLI publishing, repository creation)."
    )


def test_changelog_records_implemented_publishing_and_validator() -> None:
    """[Unreleased] must record: read-only export validator; programmatic
    publishing of validated exports; the ``publishing/`` domain package
    and ``PublicationError``.
    """
    text = _read(_resolve("CHANGELOG.md"))
    unreleased = _changelog_unreleased_block(text)
    # Read-only export validator must be recorded in [Unreleased]/Added.
    assert "validate_export_directory" in unreleased, (
        "CHANGELOG.md [Unreleased] must record the read-only export "
        "validator (validate_export_directory)."
    )
    # Programmatic publishing must be recorded.
    assert re.search(
        r"programmatic[^\n]*publishing|publishing[^\n]*validated",
        unreleased,
        re.IGNORECASE,
    ), (
        "CHANGELOG.md [Unreleased] must describe the implemented "
        "programmatic publishing of validated exports."
    )
    # Domain package + PublicationError must both be mentioned.
    assert "publishing/" in unreleased, (
        "CHANGELOG.md [Unreleased] must mention the dedicated `publishing/` domain package."
    )
    assert "PublicationError" in unreleased, (
        "CHANGELOG.md [Unreleased] must mention PublicationError."
    )


def test_changelog_states_precise_remaining_publishing_boundaries() -> None:
    """[Unreleased] distinguishes supported publishing from remaining scope."""
    text = _read(_resolve("CHANGELOG.md"))
    unreleased = _changelog_unreleased_block(text)
    assert re.search(r"CLI\s+publishing", unreleased)
    assert re.search(r"repository\s+creation", unreleased, re.IGNORECASE)
    assert "classification" in unreleased
    assert "parallel shard processing" in unreleased


def test_contributing_scope_is_accurate_about_publishing() -> None:
    """CONTRIBUTING.md must state that publishing (programmatic and CLI)
    is implemented while repository creation, classification, and
    concurrency/resumability remain out of scope; it must not
    blanket-claim all publishing is out.
    """
    text = _read(_resolve("CONTRIBUTING.md"))
    stale = "Hugging Face dataset publishing / upload."
    assert stale not in text, (
        "CONTRIBUTING.md must not blanket-claim that all Hugging Face "
        "publishing is out of scope; publishing is now implemented "
        "under publishing/."
    )
    # Programmatic publishing is implemented.
    assert re.search(r"programmatic[^\n]*publishing", text, re.IGNORECASE), (
        "CONTRIBUTING.md must state that programmatic publishing is implemented."
    )
    # CLI publishing is implemented (or referenced as an actual surface).
    assert re.search(r"CLI[^\n]*publish|--publish-dataset-id", text, re.IGNORECASE), (
        "CONTRIBUTING.md must acknowledge that CLI publishing is now implemented."
    )
    # Repository creation remains out of scope (precise boundary).
    assert re.search(r"repository\s+creation", text, re.IGNORECASE), (
        "CONTRIBUTING.md must keep repository creation flagged as out of scope."
    )


# Current maintained docs that must not deny the existing CLI publishing
# surface. Historical ADRs (docs/architecture/decisions/) are intentionally
# excluded because they describe a superseded prior layout.
_CURRENT_PUBLISHING_DOCS = [
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "docs/index.md",
    "docs/architecture/overview.md",
    "docs/guides/getting-started.md",
    "docs/guides/development.md",
    "docs/guides/reproducibility.md",
    "docs/reference/api.md",
    "docs/reference/cli.md",
    "docs/reference/data-contract.md",
]

_STALE_NO_CLI_PUBLISHING_PHRASES = (
    "no CLI flag",
    "no CLI publish",
    "CLI publishing is not implemented",
    "CLI publishing is unimplemented",
    "CLI publishing is unavailable",
)


def test_current_docs_do_not_deny_cli_publishing() -> None:
    """Maintained current documentation must not claim that no CLI
    publishing flag exists. Phrases like "no CLI flag" or "CLI publishing
    is not implemented" are stale now that ``--publish-dataset-id`` exists.
    Historical ADRs are excluded as superseded.
    """
    for doc_rel in _CURRENT_PUBLISHING_DOCS:
        text = _read(_resolve(doc_rel)).lower()
        for phrase in _STALE_NO_CLI_PUBLISHING_PHRASES:
            assert phrase.lower() not in text, (
                f"{doc_rel}: stale current-scope phrase {phrase!r} must "
                f"be removed; CLI publishing exists via --publish-dataset-id."
            )


def test_contributing_documents_canonical_contracts_layout() -> None:
    """CONTRIBUTING.md must reflect the canonical layout: cross-cutting
    contracts live under ``contracts/``, settings under
    ``application/settings.py``, ``publishing/`` is an operational
    domain package, and the top-level contract modules are
    compatibility facades.
    """
    text = _read(_resolve("CONTRIBUTING.md"))
    # Stale claim: contracts live at the package root.
    forbidden = "live at the package root"
    assert forbidden not in text, (
        "CONTRIBUTING.md still claims cross-cutting contracts live at "
        "the package root; canonical ownership is under contracts/."
    )
    # Positive: canonical contracts/ location.
    assert "contracts/" in text, (
        "CONTRIBUTING.md must reference the contracts/ package as the "
        "canonical home for cross-cutting contracts."
    )
    # Positive: settings ownership is under application/.
    assert re.search(r"application/settings\.py|application/settings\b", text), (
        "CONTRIBUTING.md must point at application/settings.py as the "
        "canonical settings ownership."
    )
    # Positive: publishing/ is listed as an operational domain package.
    assert "publishing/" in text, (
        "CONTRIBUTING.md must list publishing/ among the operational domain packages."
    )
    # Positive: top-level facades are not canonical ownership.
    assert re.search(
        r"(compatibilit(?:y|ies) facades?|facade).*ownership|"
        r"ownership.*compatibilit(?:y|ies) facades?|"
        r"facades?.*canonical",
        text,
        re.IGNORECASE,
    ), (
        "CONTRIBUTING.md must describe the top-level contract modules "
        "as compatibility facades rather than canonical ownership."
    )
    # Positive: the four legacy root compatibility facades are named as
    # modules (``.py``), not directories, so each maps to an actual file.
    root_modules = [
        "constants.py",
        "errors.py",
        "schemas.py",
        "settings.py",
    ]
    missing = [m for m in root_modules if m not in text]
    assert not missing, (
        "CONTRIBUTING.md must name the legacy root compatibility facade "
        f"modules with their .py extension; missing: {missing}"
    )


def test_contributing_hub_extra_supports_acquisition_and_publishing() -> None:
    """CONTRIBUTING.md's ``uv sync --extra hub`` line must not be
    confined to "read-only acquisition" only; it now also powers
    programmatic publishing.
    """
    text = _read(_resolve("CONTRIBUTING.md"))
    # Locate the 'uv sync --extra hub' line.
    lines = [line for line in text.splitlines() if "uv sync --extra hub" in line]
    assert lines, "CONTRIBUTING.md must mention the 'hub' extra install command"
    line = lines[0]
    assert re.search(r"acquisition", line), (
        "CONTRIBUTING.md 'hub' extra comment must still mention acquisition."
    )
    # Must also mention publishing somehow.
    assert re.search(r"publish|publishing", line, re.IGNORECASE), (
        "CONTRIBUTING.md 'hub' extra comment must mention publishing as "
        "well as read-only acquisition."
    )


def test_current_docs_contain_no_historical_or_superseded_workflows() -> None:
    current_docs = (
        "README.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "docs/index.md",
        "docs/architecture/overview.md",
        "docs/guides/development.md",
        "docs/guides/getting-started.md",
        "docs/guides/grid5000.md",
        "docs/guides/reproducibility.md",
        "docs/reference/api.md",
        "docs/reference/cli.md",
        "docs/reference/data-contract.md",
    )
    stale_terms = (
        "Phase 9",
        "amendment",
        "smoke test",
        "submit_gpu_build.sh",
        "run_gpu_build.sh",
        "submit_gpu_smoke.sh",
        "run_gpu_smoke.sh",
    )
    offenders = {
        f"{path}:{term}"
        for path in current_docs
        for term in stale_terms
        if term.casefold() in _read(_resolve(path)).casefold()
    }
    assert offenders == set()
