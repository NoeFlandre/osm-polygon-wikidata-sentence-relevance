"""Focused tests for the distribution verifier's facade-omission detection.

These construct in-memory fake wheels that are missing exactly one required
facade and assert the verifier rejects each. They do NOT compare constants
to themselves; they exercise the actual zip-scanning logic.
"""

from __future__ import annotations

import io
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
        "publishing",
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
                "publishing",
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


class TestVerifierRejectsOperationalScriptsInWheel:
    """Operational shell scripts and the scripts/ package must never
    leak into the installed wheel."""

    def test_shell_script_in_wheel_rejected(self, tmp_path: Path) -> None:
        v = _load_verifier()
        wheel = _make_fake_wheel(tmp_path)
        prefix = "osm_polygon_sentence_relevance/"
        with zipfile.ZipFile(wheel, "a") as zf:
            zf.writestr(
                f"{prefix}scripts/grid5000/run_gpu_smoke_job.sh", b"#!/bin/sh\n"
            )
        with pytest.raises(SystemExit) as exc_info:
            v.verify_wheel(wheel)
        assert exc_info.value.code == 1

    def test_scripts_package_dir_in_wheel_rejected(self, tmp_path: Path) -> None:
        v = _load_verifier()
        wheel = _make_fake_wheel(tmp_path)
        prefix = "osm_polygon_sentence_relevance/"
        with zipfile.ZipFile(wheel, "a") as zf:
            zf.writestr(f"{prefix}scripts/verify_distribution.py", b"# stub\n")
        with pytest.raises(SystemExit) as exc_info:
            v.verify_wheel(wheel)
        assert exc_info.value.code == 1


def _make_fake_sdist(
    tmp_path: Path,
    *,
    omit_script: str | None = None,
    shell_script_modes: dict[str, int] | None = None,
) -> Path:
    """Build a minimal fake sdist tarball on disk and return its path.

    ``shell_script_modes`` lets tests stamp each public shell
    script with a custom TarInfo mode (default 0o755 for all).
    """
    import tarfile

    sdist = tmp_path / "fake-0.1.0.tar.gz"
    root = "osm_polygon_sentence_relevance-0.1.0"

    public_scripts = [
        "scripts/grid5000/run_gpu_smoke.sh",
        "scripts/grid5000/run_gpu_smoke_job.sh",
        "scripts/grid5000/submit_gpu_smoke.sh",
        "scripts/grid5000/run_gpu_build.sh",
        "scripts/grid5000/run_gpu_build_job.sh",
        "scripts/grid5000/submit_gpu_build.sh",
        "scripts/grid5000/gpu_preflight.py",
        "scripts/grid5000/_validate_artifact.py",
        "scripts/grid5000/_run_metadata.py",
    ]
    shell_only = set(public_scripts[:6])
    modes = shell_script_modes or {}
    with tarfile.open(sdist, "w:gz") as tf:
        # Minimal required sdist layout.
        for doc in [
            "docs/architecture/overview.md",
            "docs/guides/getting-started.md",
            "docs/guides/development.md",
            "docs/guides/reproducibility.md",
            "docs/reference/api.md",
            "docs/reference/cli.md",
            "docs/reference/data-contract.md",
            "docs/index.md",
        ]:
            info = tarfile.TarInfo(f"{root}/{doc}")
            info.size = 0
            tf.addfile(info)
        for gov in [
            "README.md",
            "LICENSE",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "MANIFEST.in",
            "pyproject.toml",
        ]:
            info = tarfile.TarInfo(f"{root}/{gov}")
            info.size = 0
            tf.addfile(info)
        # src/ and tests/ minimal placeholders.
        for rel in ["src/dummy.py", "tests/dummy.py"]:
            data = b"# stub\n"
            info = tarfile.TarInfo(f"{root}/{rel}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        # Public scripts.
        for script in public_scripts:
            if omit_script == script:
                continue
            data = b"# stub\n"
            info = tarfile.TarInfo(f"{root}/{script}")
            info.size = len(data)
            # Stamp shell scripts with their configured mode so
            # tests can exercise the mode-0755 contract.
            if script in shell_only:
                info.mode = modes.get(script, 0o755)
            tf.addfile(info, io.BytesIO(data))
    return sdist


class TestVerifierRequiresPublicScriptsInSdist:
    """The public Grid'5000 operational scripts must ship in the sdist."""

    @pytest.mark.parametrize(
        "missing",
        [
            "scripts/grid5000/run_gpu_smoke.sh",
            "scripts/grid5000/run_gpu_smoke_job.sh",
            "scripts/grid5000/submit_gpu_smoke.sh",
            "scripts/grid5000/run_gpu_build.sh",
            "scripts/grid5000/run_gpu_build_job.sh",
            "scripts/grid5000/submit_gpu_build.sh",
            "scripts/grid5000/gpu_preflight.py",
            "scripts/grid5000/_validate_artifact.py",
            "scripts/grid5000/_run_metadata.py",
        ],
    )
    def test_missing_public_script_rejected(self, tmp_path: Path, missing: str) -> None:
        v = _load_verifier()
        sdist = _make_fake_sdist(tmp_path, omit_script=missing)
        with pytest.raises(SystemExit) as exc_info:
            v.verify_sdist(sdist)
        assert exc_info.value.code == 1

    def test_complete_sdist_accepted(self, tmp_path: Path) -> None:
        v = _load_verifier()
        sdist = _make_fake_sdist(tmp_path)
        # Should not raise.
        v.verify_sdist(sdist)


class TestVerifierRequiresShellScriptMode0755:
    """Each public shell script in the sdist must be a regular file
    with mode 0o755. Stamping it with any other permission (e.g.
    the historical 0o711) must fail verification so a downstream
    operator never has to restore the executable bit manually."""

    @pytest.mark.parametrize(
        ("script", "bad_mode"),
        [
            ("scripts/grid5000/run_gpu_smoke.sh", 0o711),
            ("scripts/grid5000/run_gpu_smoke.sh", 0o644),
            ("scripts/grid5000/run_gpu_smoke_job.sh", 0o711),
            ("scripts/grid5000/run_gpu_smoke_job.sh", 0o600),
            ("scripts/grid5000/submit_gpu_smoke.sh", 0o711),
            ("scripts/grid5000/submit_gpu_smoke.sh", 0o755 ^ 0o001),
        ],
    )
    def test_incorrectly_permissioned_shell_script_rejected(
        self, tmp_path: Path, script: str, bad_mode: int
    ) -> None:
        v = _load_verifier()
        sdist = _make_fake_sdist(
            tmp_path,
            shell_script_modes={script: bad_mode},
        )
        with pytest.raises(SystemExit) as exc_info:
            v.verify_sdist(sdist)
        assert exc_info.value.code == 1
