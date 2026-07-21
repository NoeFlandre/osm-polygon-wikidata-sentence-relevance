"""Unit tests for the persistent-storage copy helper (the implementation hardening).

Contract (the public surface of
``scripts/grid5000/_finalize_persist.sh``):

* ``finalize_persist_artifacts SCRATCH_OUT_DIR LOG_ROOT OAR_JOB_ID``
  copies the three required finalization artifacts (sentences.parquet,
  manifest.json, README.md) from ``SCRATCH_OUT_DIR`` (compute-node
  scratch) to ``${LOG_ROOT}/${OAR_JOB_ID}/output`` on the operator's
  persistent NFS mount.
* Requires the scratch directory to be real (not a symlink) and to
  contain exactly the three required artifacts as regular files.
* Creates a fresh mode-0700 staging directory under
  ``${LOG_ROOT}/${OAR_JOB_ID}.persist.XXXXXX``.
* Copies each artifact with mode 0600 and verifies the staging
  directory contains exactly three regular files with no symlinks or
  extra entries.
* Atomically renames the staging directory to
  ``${LOG_ROOT}/${OAR_JOB_ID}/output``.
* Refuses overwrite/reuse when the target already exists.
* Returns 0 on success and prints the persistent path on stdout.
* Returns non-zero on any failure with a single diagnostic line on
  stderr.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts" / "grid5000" / "_finalize_persist.sh"


def _bash() -> str:
    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover - bash is a build dependency
        pytest.skip("bash not available")
    return bash


def _call_helper(
    scratch: Path, log_root: Path, oar_job_id: str = "12345"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _bash(),
            "-c",
            (
                f"set -euo pipefail; source {HELPER}; "
                f"finalize_persist_artifacts '{scratch}' '{log_root}' '{oar_job_id}'"
            ),
        ],
        capture_output=True,
        text=True,
    )


def _make_scratch(tmp_path: Path) -> Path:
    scratch = tmp_path / "out"
    scratch.mkdir()
    (scratch / "sentences.parquet").write_bytes(b"PARQUET-BYTES")
    (scratch / "manifest.json").write_text('{"ok": true}', encoding="utf-8")
    (scratch / "README.md").write_text("# card", encoding="utf-8")
    return scratch


def test_helper_exists_and_is_executable() -> None:
    assert HELPER.exists()
    assert os.access(HELPER, os.X_OK)


def test_persist_copies_three_artifacts_with_mode_0600(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir()
    scratch = _make_scratch(tmp_path)

    result = _call_helper(scratch, log_root, oar_job_id="6789")

    assert result.returncode == 0, result.stderr
    target = log_root / "6789" / "output"
    assert target.is_dir()
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    files = sorted(p.name for p in target.iterdir())
    assert files == ["README.md", "manifest.json", "sentences.parquet"]
    for entry in target.iterdir():
        assert stat.S_IMODE(entry.stat().st_mode) == 0o600
        assert entry.read_bytes() == (scratch / entry.name).read_bytes()
    # staging directory must be gone (atomic rename consumed it)
    assert not (log_root / "6789.persist.XXXXXX").exists()


def test_persist_refuses_overwrite(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir()
    scratch = _make_scratch(tmp_path)

    first = _call_helper(scratch, log_root, oar_job_id="1111")
    assert first.returncode == 0, first.stderr

    second = _call_helper(scratch, log_root, oar_job_id="1111")
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr


def test_persist_refuses_missing_artifact(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir()
    scratch = tmp_path / "out"
    scratch.mkdir()
    (scratch / "sentences.parquet").write_bytes(b"PARQUET")
    (scratch / "manifest.json").write_text("{}", encoding="utf-8")
    # README.md intentionally missing

    result = _call_helper(scratch, log_root)
    assert result.returncode != 0
    assert "missing required artifact" in result.stderr
    assert "README.md" in result.stderr


def test_persist_refuses_symlink_artifact(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir()
    scratch = tmp_path / "out"
    scratch.mkdir()
    (scratch / "sentences.parquet").write_bytes(b"PARQUET")
    (scratch / "manifest.json").write_text("{}", encoding="utf-8")
    (scratch / "README.md").write_text("# card", encoding="utf-8")
    target = scratch / "sentences.parquet"
    target.unlink()
    target.symlink_to(scratch / "README.md")

    result = _call_helper(scratch, log_root)
    assert result.returncode != 0
    assert "refused symlink" in result.stderr


def test_persist_refuses_extra_entry(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir()
    scratch = _make_scratch(tmp_path)
    (scratch / "extra.txt").write_text("junk", encoding="utf-8")

    result = _call_helper(scratch, log_root)
    assert result.returncode != 0
    assert "expected exactly 3" in result.stderr


def test_persist_refuses_symlinked_scratch_dir(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir()
    real = tmp_path / "real_out"
    real.mkdir()
    (real / "sentences.parquet").write_bytes(b"PARQUET")
    (real / "manifest.json").write_text("{}", encoding="utf-8")
    (real / "README.md").write_text("# card", encoding="utf-8")
    link = tmp_path / "link_out"
    link.symlink_to(real)

    result = _call_helper(link, log_root)
    assert result.returncode != 0
    assert "real directory" in result.stderr


def test_persist_requires_absolute_paths(tmp_path: Path) -> None:
    scratch = _make_scratch(tmp_path)
    result = subprocess.run(
        [
            _bash(),
            "-c",
            (
                f"set -euo pipefail; source {HELPER}; "
                f"finalize_persist_artifacts '{scratch}' 'relative/logs' '99'"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must be absolute" in result.stderr
