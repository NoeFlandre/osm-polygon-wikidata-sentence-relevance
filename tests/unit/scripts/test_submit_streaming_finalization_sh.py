"""Unit tests for the bounded one-shard OAR finalization adapter (the implementation).

The submission adapter ``scripts/grid5000/submit_streaming_finalization.sh``
runs on the Grid'5000 frontend and validates its positional arguments
before invoking ``oarsub``.  These tests assert the public contract:

* Exactly 12 positional arguments are required.
* All persistent paths must be absolute real directories.
* Revision SHAs must be 40 lowercase hexadecimal characters.
* Repository IDs must be owner/name without spaces or slashes in
  the components.
* Run ID, expected shard, and walltime must match the documented
  patterns.
* ``node-type`` must be ``cpu`` or ``gpu``.
* Invalid arguments fail fast (exit 2) without contacting OAR.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SUBMIT = ROOT / "scripts" / "grid5000" / "submit_streaming_finalization.sh"


def _make_real_dir(tmp_path: Path) -> Path:
    """Create a real directory; reject symlinks because the adapter does."""
    real = tmp_path / "real"
    real.mkdir(parents=True, exist_ok=True)
    return real


def _good_args(tmp_path: Path) -> list[str]:
    repo = _make_real_dir(tmp_path / "repo")
    hf = _make_real_dir(tmp_path / "hf")
    logs = _make_real_dir(tmp_path / "logs")
    return [
        str(repo),
        str(hf),
        str(logs),
        "owner/output",
        "owner/input",
        "c9eb3c3ee107bee036a93097b2cc473d62fc93bd",
        "84b3ca0ff33a3d3fba44c093eb4ac49ee0b5ef90",
        "afghanistan-20260721t070917z",
        "checkpoints/afghanistan-20260721t070917z",
        "afghanistan-latest",
        "00:15:00",
        "cpu",
    ]


def test_submit_script_exists_and_is_executable() -> None:
    assert SUBMIT.exists()
    assert os.access(SUBMIT, os.X_OK)


def test_submit_rejects_wrong_arity(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover - bash is a build dependency
        pytest.skip("bash not available")
    result = subprocess.run(
        [bash, str(SUBMIT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "exactly twelve positional arguments" in result.stderr


def test_submit_rejects_non_absolute_repo_path(tmp_path: Path) -> None:
    args = _good_args(tmp_path)
    args[0] = "relative/path"
    result = subprocess.run(
        [str(SUBMIT), *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "persistent path must be absolute" in result.stderr


def test_submit_rejects_symlinked_repo_path(tmp_path: Path) -> None:
    real = _make_real_dir(tmp_path / "real_repo")
    link = tmp_path / "link_repo"
    link.symlink_to(real)
    args = _good_args(tmp_path)
    args[0] = str(link)
    result = subprocess.run(
        [str(SUBMIT), *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "real directory" in result.stderr


def test_submit_rejects_non_hex_source_commit(tmp_path: Path) -> None:
    args = _good_args(tmp_path)
    args[5] = "NOT_A_HEX_SHA"
    result = subprocess.run(
        [str(SUBMIT), *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "revisions must be 40 lowercase hex" in result.stderr


def test_submit_rejects_run_id_with_uppercase(tmp_path: Path) -> None:
    args = _good_args(tmp_path)
    args[7] = "Afghanistan-RUN"
    result = subprocess.run(
        [str(SUBMIT), *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "invalid run-id" in result.stderr


def test_submit_rejects_malformed_walltime(tmp_path: Path) -> None:
    args = _good_args(tmp_path)
    args[10] = "15 minutes"
    result = subprocess.run(
        [str(SUBMIT), *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "invalid run-id/shard/walltime" in result.stderr


def test_submit_rejects_unknown_node_type(tmp_path: Path) -> None:
    args = _good_args(tmp_path)
    args[11] = "tpu"
    result = subprocess.run(
        [str(SUBMIT), *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "node-type must be cpu|gpu" in result.stderr


def test_submit_rejects_repo_id_with_space(tmp_path: Path) -> None:
    args = _good_args(tmp_path)
    args[3] = "owner name/repo"
    result = subprocess.run(
        [str(SUBMIT), *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "owner/name" in result.stderr
