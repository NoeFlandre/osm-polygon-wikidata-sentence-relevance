"""RED tests for Phase 9P data-root guard.

`scripts.streaming.data_root.check_data_root(path, *, role, min_free_bytes)`
must reject any path that physically resolves to:
  - /tmp
  - /var/tmp
  - /dev/shm
including symlink escape (resolve the real path first).

It must also enforce a soft free-bytes ceiling and refuse non-regular
directories.
"""

from __future__ import annotations

import collections
import contextlib
import os
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "streaming"
sys.path.insert(0, str(SCRIPTS_DIR.parent))

_NAMED_TUPLE = collections.namedtuple("usage", ["total", "used", "free"])

from scripts.streaming.data_root import DataRootRejected, check_data_root  # noqa: E402

# ---------------------------------------------------------------------------
# /tmp /var/tmp /dev/shm rejections -- unconditional, post-resolve
# ---------------------------------------------------------------------------


def test_check_data_root_rejects_explicit_tmp_subdir() -> None:
    """Any path under /tmp (after physical resolution) is rejected.

    This test creates a real directory under the OS-level /tmp and
    asserts it is rejected, regardless of whether the OS resolves
    /tmp through a symlink (e.g. macOS -> /private/tmp). The denylist
    matches both the canonical and the resolved path.
    """
    import tempfile

    rejected_path = Path(tempfile.mkdtemp(prefix="streaming-reject-", dir="/tmp"))
    try:
        with pytest.raises(DataRootRejected) as exc_info:
            check_data_root(rejected_path, role="scratch", min_free_bytes=0)
        assert exc_info.value.reason == "TMP_FORBIDDEN"
    finally:
        with contextlib.suppress(OSError):
            rejected_path.rmdir()


def test_check_data_root_rejects_var_tmp_subdir() -> None:
    import tempfile

    rejected_path = Path(tempfile.mkdtemp(prefix="streaming-reject-", dir="/var/tmp"))
    try:
        with pytest.raises(DataRootRejected) as exc_info:
            check_data_root(rejected_path, role="scratch", min_free_bytes=0)
        assert exc_info.value.reason == "TMP_FORBIDDEN"
    finally:
        with contextlib.suppress(OSError):
            rejected_path.rmdir()


def test_check_data_root_rejects_dev_shm_if_present() -> None:
    """If /dev/shm exists (Linux), any subdir is rejected. Otherwise skip."""
    if not Path("/dev/shm").exists():
        pytest.skip("/dev/shm not present on this host")
    import tempfile

    rejected_path = Path(tempfile.mkdtemp(prefix="streaming-reject-", dir="/dev/shm"))
    try:
        with pytest.raises(DataRootRejected) as exc_info:
            check_data_root(rejected_path, role="scratch", min_free_bytes=0)
        assert exc_info.value.reason == "TMP_FORBIDDEN"
    finally:
        with contextlib.suppress(OSError):
            rejected_path.rmdir()


def test_check_data_root_resolves_through_symlink_to_tmp(tmp_path: Path) -> None:
    """A symlink whose physical target lies in /tmp is rejected."""
    import tempfile

    target = Path(tempfile.mkdtemp(prefix="streaming-sym-", dir="/tmp"))
    link_parent = tmp_path / "links"
    link_parent.mkdir()
    symlink = link_parent / "into_tmp"
    try:
        os.symlink(target, symlink, target_is_directory=True)
        with pytest.raises(DataRootRejected) as exc_info:
            check_data_root(symlink, role="scratch", min_free_bytes=0)
        assert exc_info.value.reason == "TMP_FORBIDDEN"
    finally:
        with contextlib.suppress(OSError):
            target.rmdir()
        with contextlib.suppress(OSError):
            symlink.unlink()


# ---------------------------------------------------------------------------
# Min-free-bytes ceiling
# ---------------------------------------------------------------------------


def test_check_data_root_enforces_min_free_bytes(tmp_path: Path, monkeypatch) -> None:
    """A min_free_bytes value above the reported free space must reject.

    We monkey-patch ``shutil.disk_usage`` to return a controlled value
    so the assertion is hermetic and not subject to host FS state.
    """
    # Use a real disk-backed path that won't be /tmp.
    real = tmp_path / "normal_root"
    real.mkdir()
    monkeypatch.setattr(
        "scripts.streaming.data_root.shutil.disk_usage",
        lambda p: _NAMED_TUPLE(total=1 << 30, used=1 << 20, free=0),
    )
    with pytest.raises(DataRootRejected) as exc_info:
        check_data_root(real, role="scratch", min_free_bytes=1 << 32)
    assert exc_info.value.reason == "BELOW_CEILING"


# ---------------------------------------------------------------------------
# Non-regular-dir rejection
# ---------------------------------------------------------------------------


def test_check_data_root_rejects_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-root"
    with pytest.raises(DataRootRejected) as exc_info:
        check_data_root(missing, role="scratch", min_free_bytes=0)
    assert exc_info.value.reason == "NOT_REGULAR_DIR"


# ---------------------------------------------------------------------------
# GREEN: a normal /home path is accepted.
# ---------------------------------------------------------------------------


def test_check_data_root_accepts_normal_home_path(tmp_path: Path) -> None:
    real = tmp_path / "ok"
    real.mkdir()
    result = check_data_root(real, role="scratch", min_free_bytes=0)
    assert result == real.resolve()
