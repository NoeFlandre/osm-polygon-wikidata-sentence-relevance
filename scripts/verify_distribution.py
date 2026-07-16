#!/usr/bin/env python3
"""Verify the built wheel and sdist contents.

Stdlib-only. Asserts that the wheel ships the canonical domain packages,
the compatibility facades, the joins package, ``py.typed``, and the license
metadata, while excluding docs and the local-only ``.local-docs/`` guide;
and that the sdist ships the public docs, governance files, source, and
tests while excluding local docs, caches, data, credentials, Parquet files,
model weights, and build/dist output.

Usage:
    python scripts/verify_distribution.py <wheel> <sdist>

Exits non-zero with a precise message on the first mismatch.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

WHEEL_PKG = "osm_polygon_sentence_relevance"
DOMAIN_PACKAGES = [
    "application",
    "contracts",
    "ingestion",
    "joins",
    "output",
    "sentences",
]
# Subpackages that must ship inside the wheel.
WHEEL_REQUIRED_SUBDIRS = [
    "contracts/schemas",
]
COMPAT_FACADES = [
    "acquisition",
    "cli",
    "constants",
    "discovery",
    "errors",
    "exporter",
    "finalization",
    "loading",
    "pipeline",
    "preprocessing",
    "sat_adapter",
    "schemas",
    "segmentation",
    "sentence_table",
    "settings",
]
SDIST_PUBLIC_DOCS = [
    "docs/architecture/overview.md",
    "docs/guides/getting-started.md",
    "docs/guides/development.md",
    "docs/guides/reproducibility.md",
    "docs/reference/api.md",
    "docs/reference/cli.md",
    "docs/reference/data-contract.md",
    "docs/index.md",
]
SDIST_GOVERNANCE = [
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "MANIFEST.in",
    "pyproject.toml",
]
SDIST_FORBIDDEN = [
    ".local-docs/",
    "build/",
    "dist/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".coverage",
    "coverage.xml",
]
WHEEL_FORBIDDEN_PREFIXES = ("docs/", ".local-docs/", "tests/", ".git")


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _wheel_names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def _sdist_names(sdist: Path) -> list[str]:
    with tarfile.open(sdist) as tf:
        return tf.getnames()


def verify_wheel(wheel: Path) -> None:
    names = _wheel_names(wheel)
    prefix = f"{WHEEL_PKG}/"

    if not any(n == f"{prefix}py.typed" for n in names):
        _fail("wheel is missing py.typed marker")

    for pkg in DOMAIN_PACKAGES:
        if not any(n.startswith(f"{prefix}{pkg}/") for n in names):
            _fail(f"wheel is missing domain package: {pkg}")

    for sub in WHEEL_REQUIRED_SUBDIRS:
        if not any(n.startswith(f"{prefix}{sub}/") for n in names):
            _fail(f"wheel is missing required subpackage: {sub}")

    for facade in COMPAT_FACADES:
        if not any(n == f"{prefix}{facade}.py" for n in names):
            _fail(f"wheel is missing compatibility facade: {facade}.py")

    if not any(
        n.endswith("METADATA")
        and "License-Expression: MIT" in (zipfile.ZipFile(wheel).read(n).decode())
        for n in names
    ):
        _fail("wheel metadata is missing 'License-Expression: MIT'")

    for n in names:
        if n.endswith(".py") and "/tests/" in n:
            _fail(f"wheel contains a test module: {n}")
        for forbidden in WHEEL_FORBIDDEN_PREFIXES:
            if n.startswith(forbidden) or f"/{forbidden}" in n:
                _fail(f"wheel contains forbidden path: {n}")
        if n.startswith("docs/") or "local-docs" in n:
            _fail(f"wheel contains documentation: {n}")


def verify_sdist(sdist: Path) -> None:
    names = _sdist_names(sdist)
    # sdist names are prefixed with <pkg>-<version>/
    roots = {n.split("/", 1)[0] for n in names if "/" in n}
    root = next(iter(roots), "")
    if not root:
        _fail("sdist has unexpected layout (no top-level directory)")

    def in_sdist(rel: str) -> bool:
        return any(
            n == f"{root}/{rel}" or n.startswith(f"{root}/{rel}/") for n in names
        )

    for doc in SDIST_PUBLIC_DOCS:
        if not in_sdist(doc):
            _fail(f"sdist is missing public doc: {doc}")

    for gov in SDIST_GOVERNANCE:
        if not in_sdist(gov):
            _fail(f"sdist is missing governance file: {gov}")

    if not in_sdist("src"):
        _fail("sdist is missing source tree (src/)")
    if not in_sdist("tests"):
        _fail("sdist is missing tests/")

    flat = "\n".join(names)
    for forbidden in SDIST_FORBIDDEN:
        if forbidden in flat:
            _fail(f"sdist contains forbidden path: {forbidden}")
    if "local-docs" in flat:
        _fail("sdist contains .local-docs/")
    if ".parquet" in flat:
        _fail("sdist contains Parquet data files")
    if any(f in flat for f in (".onnx", ".safetensors", ".bin")):
        _fail("sdist contains model weights")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: verify_distribution.py <wheel> <sdist>",
            file=sys.stderr,
        )
        return 2

    wheel = Path(argv[1])
    sdist = Path(argv[2])
    if not wheel.is_file():
        print(f"wheel not found: {wheel}", file=sys.stderr)
        return 2
    if not sdist.is_file():
        print(f"sdist not found: {sdist}", file=sys.stderr)
        return 2

    verify_wheel(wheel)
    verify_sdist(sdist)
    print("OK: distribution contents verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
