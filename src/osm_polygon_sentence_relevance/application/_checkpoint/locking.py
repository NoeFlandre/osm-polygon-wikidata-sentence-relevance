"""Exclusive work-directory lock ownership."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from .common import (
    _DIR_MODE,
    _FILE_MODE,
    WORK_DIR_LOCK_NAME,
    CheckpointValidationError,
)


@dataclass(frozen=True, slots=True)
class WorkDirLock:
    """An open file descriptor on the work-dir lock file. The lock is
    held until :func:`release_work_dir_lock` is called; the file is
    not unlinked by the holder."""

    fd: int
    path: Path


def acquire_work_dir_lock(work_dir: Path) -> WorkDirLock:
    """Acquire a non-blocking exclusive ``flock`` on
    ``${work_dir}/shards/.lock``.

    The lock file is hardened before flock is taken:

    * it is opened with ``O_NOFOLLOW`` so a symlink at ``.lock`` can
      never be followed;
    * the ``fstat`` of the resulting descriptor must describe a
      regular file owned by the current user with mode ``0o600``;
    * any permissive mode or non-regular entry is rejected with
      :class:`CheckpointValidationError`.

    Raises :class:`CheckpointValidationError` if the lock is already
    held by another process or the lock file fails hardening. On
    POSIX systems the lock is held until the file descriptor is
    closed; this function never unlinks the lock file.
    """
    work_dir = work_dir.expanduser().resolve(strict=False)
    shards_root = work_dir / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):  # pragma: no cover (read-only work_dir)
        os.chmod(shards_root, _DIR_MODE)
    lock_path = shards_root / WORK_DIR_LOCK_NAME
    open_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(lock_path), open_flags, _FILE_MODE)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise CheckpointValidationError(
                f"work_dir lock file {lock_path} is a symlink; refusing"
            ) from exc
        raise
    try:
        st = os.fstat(fd)
    except OSError:
        os.close(fd)
        raise
    mode_bits = st.st_mode & 0o777
    if (
        not stat.S_ISREG(st.st_mode)
        or st.st_uid != os.getuid()
        or mode_bits != _FILE_MODE
    ):
        os.close(fd)
        raise CheckpointValidationError(
            f"work_dir lock file {lock_path} has unexpected "
            f"type/owner/mode: regular={stat.S_ISREG(st.st_mode)} "
            f"uid={st.st_uid} (expected {os.getuid()}) mode={mode_bits:o} "
            f"(expected {_FILE_MODE:o})"
        )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            raise CheckpointValidationError(
                f"work_dir {work_dir} is already locked by another run"
            ) from exc
        raise
    return WorkDirLock(fd=fd, path=lock_path)


def release_work_dir_lock(ctx: WorkDirLock) -> None:
    """Release the lock acquired by :func:`acquire_work_dir_lock`.

    The lock file is **not** unlinked: ownership of the evidence file
    remains with the work_dir and a stale lock check can still detect
    a process that died while holding it.
    """
    try:
        fcntl.flock(ctx.fd, fcntl.LOCK_UN)
    finally:
        os.close(ctx.fd)
