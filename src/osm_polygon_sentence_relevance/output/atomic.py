"""Rollback-safe directory installation for exported datasets.

Implements the atomic swap: build the new directory fully elsewhere, then
rename the existing output aside into a backup, rename the new directory
into place, and only remove the backup after the swap succeeds. Any failure
during the swap restores the backup; if even restoration fails, the backup
is explicitly preserved and surfaced via :class:`ExportError`.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import uuid
from pathlib import Path

from osm_polygon_sentence_relevance.contracts.errors import ExportError


def install_atomic(
    tmp_dir: Path,
    output_path: Path,
) -> Path | None:
    """Swap *tmp_dir* into *output_path* with a rollback-safe backup.

    Returns the backup directory path if one was created (caller is
    responsible for removing it after a successful swap), else ``None``.
    """
    backup_dir: Path | None = None

    # 2. Rename existing output to a temporary backup (if it exists).
    if output_path.exists():
        backup_dir = output_path.parent / f".backup_{uuid.uuid4().hex}"
        os.rename(output_path, backup_dir)

    try:
        # 3. Rename the new directory into place.
        os.rename(tmp_dir, output_path)
    except Exception as rename_err:
        # 4. If step 3 fails, restore the backup.
        if backup_dir is not None and backup_dir.exists():
            if output_path.exists():
                if output_path.is_dir():
                    shutil.rmtree(output_path)
                else:
                    os.remove(output_path)
            try:
                os.rename(backup_dir, output_path)
            except Exception as restore_err:
                saved_backup = backup_dir
                backup_dir = None
                raise ExportError(
                    f"Atomic replacement failed, and backup restoration also failed. "
                    f"Previous dataset is preserved at {saved_backup}"
                ) from restore_err
        raise rename_err

    return backup_dir


def remove_backup(backup_dir: Path) -> None:
    """Remove a successfully-replaced backup, erroring if cleanup fails."""
    try:
        shutil.rmtree(backup_dir)
    except Exception as rmtree_err:
        raise ExportError(
            f"New dataset successfully exported, but failed to delete backup "
            f"directory: {backup_dir}"
        ) from rmtree_err


def cleanup_on_failure(tmp_dir: Path | None, backup_dir: Path | None) -> None:
    """Best-effort removal of a partial temp dir and any leftover backup."""
    if tmp_dir is not None and tmp_dir.exists():
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir)
    if backup_dir is not None and backup_dir.exists():
        with contextlib.suppress(Exception):
            shutil.rmtree(backup_dir)
