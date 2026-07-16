"""Focused tests for the distribution verifier's facade-omission detection.

These construct in-memory fake wheels that are missing exactly one required
facade and assert the verifier rejects each. They do NOT compare constants
to themselves; they exercise the actual zip-scanning logic.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "verify_distribution.py"


def _load_verifier():
    """Import the verifier script as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("verify_distribution", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _make_fake_wheel(
    tmp_path: Path,
    *,
    omit_facade: str | None = None,
    omit_domain: str | None = None,
    omit_py_typed: bool = False,
    include_license: bool = True,
) -> Path:
    """Build a minimal fake wheel zip on disk and return its path."""
    wheel = tmp_path / "fake-0.1.0-py3-none-any.whl"
    prefix = "osm_polygon_sentence_relevance/"

    # Build the name list.
    names: list[str] = []
    if not omit_py_typed:
        names.append(f"{prefix}py.typed")
    for pkg in [
        "application",
        "contracts",
        "ingestion",
        "joins",
        "output",
        "sentences",
    ]:
        if omit_domain == pkg:
            continue
        names.append(f"{prefix}{pkg}/__init__.py")
    names.append(f"{prefix}contracts/schemas/__init__.py")
    if omit_domain == "contracts":
        # Also drop the subdir so the verifier really sees no contracts/ prefix.
        names = [n for n in names if not n.startswith(f"{prefix}contracts/")]

    facades = [
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
    for f in facades:
        if omit_facade == f:
            continue
        names.append(f"{prefix}{f}.py")

    # METADATA with license expression.
    metadata_lines = ["Metadata-Version: 2.1"]
    if include_license:
        metadata_lines.append("License-Expression: MIT")
    metadata_content = "\n".join(metadata_lines) + "\n"

    with zipfile.ZipFile(wheel, "w") as zf:
        for n in names:
            zf.writestr(n, b"# stub\n")
        zf.writestr(
            "osm_polygon_sentence_relevance-0.1.0.dist-info/METADATA", metadata_content
        )

    return wheel


class TestVerifierRejectsMissingFacade:
    """Omission of any required facade must cause verify_wheel to fail."""

    ALL_FACADES = [
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

    @pytest.mark.parametrize("missing", ALL_FACADES)
    def test_missing_facade_rejected(self, tmp_path: Path, missing: str) -> None:
        v = _load_verifier()
        wheel = _make_fake_wheel(tmp_path, omit_facade=missing)
        with pytest.raises(SystemExit) as exc_info:
            v.verify_wheel(wheel)
        assert exc_info.value.code == 1

    def test_missing_domain_package_rejected(self, tmp_path: Path) -> None:
        v = _load_verifier()
        wheel = _make_fake_wheel(tmp_path, omit_domain="contracts")
        with pytest.raises(SystemExit) as exc_info:
            v.verify_wheel(wheel)
        assert exc_info.value.code == 1

    def test_missing_subpackage_rejected(self, tmp_path: Path) -> None:
        """contracts/schemas/ must be present."""
        v = _load_verifier()
        # Build a wheel with contracts/ but no contracts/schemas/.
        wheel = tmp_path / "fake-nosub-0.1.0-py3-none-any.whl"
        prefix = "osm_polygon_sentence_relevance/"
        with zipfile.ZipFile(wheel, "w") as zf:
            zf.writestr(f"{prefix}py.typed", b"")
            for pkg in [
                "application",
                "contracts",
                "ingestion",
                "joins",
                "output",
                "sentences",
            ]:
                zf.writestr(f"{prefix}{pkg}/__init__.py", b"")
            # NOTE: deliberately NO contracts/schemas/__init__.py
            for f in self.ALL_FACADES:
                zf.writestr(f"{prefix}{f}.py", b"")
            zf.writestr(
                "osm_polygon_sentence_relevance-0.1.0.dist-info/METADATA",
                b"License-Expression: MIT\n",
            )
        with pytest.raises(SystemExit) as exc_info:
            v.verify_wheel(wheel)
        assert exc_info.value.code == 1

    def test_missing_py_typed_rejected(self, tmp_path: Path) -> None:
        v = _load_verifier()
        wheel = _make_fake_wheel(tmp_path, omit_py_typed=True)
        with pytest.raises(SystemExit) as exc_info:
            v.verify_wheel(wheel)
        assert exc_info.value.code == 1

    def test_missing_license_rejected(self, tmp_path: Path) -> None:
        v = _load_verifier()
        wheel = _make_fake_wheel(tmp_path, include_license=False)
        with pytest.raises(SystemExit) as exc_info:
            v.verify_wheel(wheel)
        assert exc_info.value.code == 1

    def test_complete_wheel_accepted(self, tmp_path: Path) -> None:
        """A wheel with all required components must NOT raise."""
        v = _load_verifier()
        wheel = _make_fake_wheel(tmp_path)
        # Should not raise.
        v.verify_wheel(wheel)
