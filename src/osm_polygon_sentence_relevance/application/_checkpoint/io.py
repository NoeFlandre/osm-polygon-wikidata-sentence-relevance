"""Crash-consistent atomic checkpoint file installation."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .common import _FILE_MODE


def _fsync_dir_strict(path: Path) -> None:
    """``fsync`` a directory entry; ``OSError`` propagates."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically inside ``path.parent``.

    Durability invariants:

    * data is ``flush()``-ed and the file descriptor is ``fsync``-ed
      *before* the rename — a crash between the rename and the parent
      ``fsync`` cannot lose the file;
    * the file mode is set to ``0o600`` *before* the rename;
    * the parent directory is ``fsync``-ed after the rename so the
      directory entry is durable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_name, _FILE_MODE)
        os.replace(tmp_name, path)
        os.chmod(path, _FILE_MODE)
    except Exception:
        with contextlib.suppress(OSError):  # pragma: no cover (best-effort cleanup)
            os.unlink(tmp_name)
        raise
    _fsync_dir_strict(path.parent)


def _atomic_write_parquet(table: Any, path: Path) -> None:
    """Write a PyArrow ``table`` to ``path`` atomically inside ``path.parent``.

    Same durability invariants as :func:`_atomic_write_bytes`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        pq.write_table(table, tmp_name)
        os.chmod(tmp_name, _FILE_MODE)
        # Force a fsync on the file so that the bytes hit disk before
        # the rename; this matches the bytes-then-rename-then-fsync-dir
        # pattern documented above.
        with open(tmp_name, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
        os.chmod(path, _FILE_MODE)
    except Exception:
        with contextlib.suppress(OSError):  # pragma: no cover (best-effort cleanup)
            os.unlink(tmp_name)
        raise
    _fsync_dir_strict(path.parent)
