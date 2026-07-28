"""Safe planning for pipeline-owned Grid'5000 storage reclamation."""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ManagedStatus(StrEnum):
    """Lifecycle status recorded in the operator inventory."""

    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"
    CACHE = "cache"
    PROTECTED = "protected"


@dataclass(frozen=True, slots=True)
class ManagedEntry:
    """One inventory-bound directory beneath the managed root."""

    path: Path
    status: ManagedStatus
    bytes_used: int
    updated_epoch: int
    pipeline_owned: bool = True


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    """Ordered deletion candidates and expected reclaimed bytes."""

    managed_root: Path
    candidates: tuple[ManagedEntry, ...]
    expected_reclaimed_bytes: int


class StorageSafetyError(RuntimeError):
    """A cleanup candidate violates the managed-storage boundary."""


_PROTECTED_NAMES = frozenset({".ssh", ".bashrc", ".profile", ".bash_profile"})


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def plan_cleanup(
    managed_root: Path,
    entries: list[ManagedEntry] | tuple[ManagedEntry, ...],
    required_bytes: int,
) -> CleanupPlan:
    """Plan oldest safe deletions without mutating the filesystem."""

    if required_bytes < 0:
        raise ValueError("required_bytes must be non-negative")
    root = managed_root.resolve(strict=True)
    if managed_root.is_symlink() or not root.is_dir():
        raise StorageSafetyError("managed root must be a real directory")

    eligible: list[ManagedEntry] = []
    for entry in entries:
        if not entry.pipeline_owned:
            continue
        if entry.status not in {ManagedStatus.COMPLETE, ManagedStatus.FAILED}:
            continue
        if entry.path.name in _PROTECTED_NAMES or entry.path.is_symlink():
            continue
        try:
            candidate = entry.path.resolve(strict=True)
        except OSError:
            continue
        if not _is_beneath(candidate, root) or not candidate.is_dir():
            continue
        eligible.append(entry)

    selected: list[ManagedEntry] = []
    reclaimed = 0
    for entry in sorted(
        eligible, key=lambda item: (item.updated_epoch, str(item.path))
    ):
        if reclaimed >= required_bytes:
            break
        selected.append(entry)
        reclaimed += max(0, entry.bytes_used)
    return CleanupPlan(root, tuple(selected), reclaimed)


def execute_cleanup(plan: CleanupPlan) -> int:
    """Execute a precomputed plan after immediate containment revalidation."""

    root = plan.managed_root.resolve(strict=True)
    reclaimed = 0
    for entry in plan.candidates:
        if entry.path.is_symlink():
            raise StorageSafetyError("cleanup candidate became a symlink")
        candidate = entry.path.resolve(strict=True)
        if not _is_beneath(candidate, root):
            raise StorageSafetyError("cleanup candidate escaped managed root")
        metadata = os.lstat(candidate)
        if not stat.S_ISDIR(metadata.st_mode):
            raise StorageSafetyError("cleanup candidate is not a directory")
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise StorageSafetyError("cleanup candidate owner changed")
        shutil.rmtree(candidate)
        reclaimed += max(0, entry.bytes_used)
    return reclaimed


__all__ = [
    "CleanupPlan",
    "ManagedEntry",
    "ManagedStatus",
    "StorageSafetyError",
    "execute_cleanup",
    "plan_cleanup",
]
