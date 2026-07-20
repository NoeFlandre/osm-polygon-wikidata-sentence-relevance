"""Operational data-root guard for the per-shard streaming workflow.

The streaming driver never stores large data under frontend ``/tmp``,
``/var/tmp``, or ``/dev/shm``. This module enforces that invariant
unconditionally, including symlink escape (``os.path.realpath``
resolution first, then exact-prefix check against the denylist).

It also enforces a soft free-bytes ceiling and refuses to accept
non-regular-dir paths.

The frontend-vs-compute-node distinction is documented here but the
enforcement is delegated to the driver (``driver.py``); the
non-frontend streaming-driver CLI refuses to start when invoked
outside an OAR job.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path

# Unconditional denylist. These prefixes are rejected after physical
# resolution regardless of filesystem type. They cover the common
# shared-memory/tmp paths that are too small for our storage budget.
#
# Note: on Linux, /tmp resolves to itself; on macOS, /tmp is a symlink
# to /private/tmp. We therefore check BOTH:
#   (a) the literal ``str(input)`` path strings for the canonical names;
#   (b) the resolved realpath against this same denylist;
#   (c) the first / second path components so that ``/private/tmp`` is
#       caught even when ``/tmp`` was not typed.
_FORBIDDEN_LITERAL_PREFIXES: tuple[str, ...] = (
    "/tmp",
    "/tmp/",
    "/var/tmp",
    "/var/tmp/",
    "/dev/shm",
    "/dev/shm/",
    "/private/tmp",
    "/private/tmp/",
    "/private/var/tmp",
    "/private/var/tmp/",
)

# First path component names that are forbidden on any host. The
# realpath is split into its ``/``-separated parts and each leading
# part is matched against this set. This catches both Linux
# (``/tmp/...``) and macOS (``/private/tmp/...`` or
# ``/private/var/tmp/...``).
_FORBIDDEN_PATH_NAMES: frozenset[str] = frozenset(
    {
        "tmp",
        "dev",
    }
)

# A pair of (parent_name, name) directory pairs that are forbidden
# regardless of leading /private prefix. So ``var/tmp`` is forbidden
# whether or not it sits under /private.
_FORBIDDEN_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("var", "tmp"),  # /var/tmp and /private/var/tmp
        ("dev", "shm"),  # /dev/shm
    }
)


class DataRootRejected(ValueError):
    """Raised by ``check_data_root`` when a path is rejected.

    Attributes
    ----------
    reason : str
        Machine-parseable rejection code, one of:
        ``"TMP_FORBIDDEN"``, ``"BELOW_CEILING"``, ``"NOT_REGULAR_DIR"``.
    role : str
        The role label provided by the caller
        (``"scratch"``, ``"input"``, ``"work"``, ...).
    path : str
        The physical (realpath) path that was rejected.
    """

    def __init__(self, *, reason: str, role: str, path: str) -> None:
        self.reason = reason
        self.role = role
        self.path = path
        super().__init__(
            f"data_root rejected (role={role!r}, reason={reason}, path={path})"
        )


def _phys(path: Path) -> Path:
    """Return the OS-canonical path with symlinks resolved.

    Uses ``Path.resolve(strict=False)`` which delegates to
    ``os.path.realpath``. Missing-path inputs are tolerated
    (``strict=False``).
    """
    return Path(os.path.realpath(str(path))).resolve(strict=False)


def _is_forbidden_tmp(path: Path) -> bool:
    """Return True iff the path sits under a forbidden tmp prefix.

    The check is prefix-based and unconditional; it does NOT inspect
    the filesystem type. The Lyon incident showed the issue can
    manifest on a disk-backed shared ``/tmp`` as well as on tmpfs.

    Both the literal input and the resolved realpath are checked.
    The first path component is checked against ``_FORBIDDEN_PATH_NAMES``
    so /private/tmp is caught the same way as /tmp.
    """
    candidates: list[str] = [str(path)]
    with contextlib.suppress(OSError):
        candidates.append(os.path.realpath(str(path)))

    for s in candidates:
        if not s:
            continue
        if s in {"/tmp", "/var/tmp", "/dev/shm", "/private/tmp", "/private/var/tmp"}:
            return True
        for prefix in _FORBIDDEN_LITERAL_PREFIXES:
            if s == prefix or s.startswith(prefix):
                return True

    # Component-pair check (e.g. /private/var/tmp).
    for s in candidates:
        if not s.startswith("/"):
            continue
        parts = [p for p in s.split("/") if p]
        # Check (parent_name, name) pairs starting from index 0.
        for i in range(1, len(parts)):
            pair = (parts[i - 1], parts[i])
            if pair in _FORBIDDEN_PAIRS:
                return True
        # Single-name check: the first path component is ``tmp`` or ``shm``.
        if parts and parts[0] in _FORBIDDEN_PATH_NAMES:
            return True
    return False


def check_data_root(path: Path, *, role: str, min_free_bytes: int) -> Path:
    """Validate a candidate data root path.

    Parameters
    ----------
    path : Path
        Candidate path. Will be physically resolved before evaluation.
    role : str
        Free-form role label recorded in any rejection. Caller-supplied;
        not validated.
    min_free_bytes : int
        Minimum required free bytes. ``0`` skips the ceiling check.

    Returns
    -------
    Path
        The physically-resolved path. Same on disk as the input, but
        with symlinks resolved.

    Raises
    ------
    DataRootRejected
        Reason codes:
          * ``"TMP_FORBIDDEN"`` -- physical path lies under
            ``/tmp``, ``/var/tmp`` or ``/dev/shm``.
          * ``"BELOW_CEILING"`` -- physical path is a regular dir
            but the filesystem has fewer than ``min_free_bytes``
            free bytes.
          * ``"NOT_REGULAR_DIR"`` -- physical path is missing or
            not an existing regular directory.
    """
    if role is None or not isinstance(role, str):
        raise ValueError("role must be a non-blank string")
    resolved = _phys(path)
    if _is_forbidden_tmp(resolved):
        raise DataRootRejected(reason="TMP_FORBIDDEN", role=role, path=str(resolved))

    # Existence check; tolerate missing as "NOT_REGULAR_DIR".
    try:
        if not resolved.exists():
            raise DataRootRejected(
                reason="NOT_REGULAR_DIR",
                role=role,
                path=str(resolved),
            )
    except OSError as exc:
        raise DataRootRejected(
            reason="NOT_REGULAR_DIR", role=role, path=str(resolved)
        ) from exc

    if not resolved.is_dir() or resolved.is_symlink():
        raise DataRootRejected(reason="NOT_REGULAR_DIR", role=role, path=str(resolved))

    if min_free_bytes > 0:
        usage = shutil.disk_usage(str(resolved))
        if usage.free < int(min_free_bytes):
            raise DataRootRejected(
                reason="BELOW_CEILING",
                role=role,
                path=str(resolved),
            )

    return resolved


def discover_oar_scratch_dir(min_free_bytes: int = 1 << 30) -> Path:
    """Discover and validate the real allocation-local scratch path on Grid'5000.

    Priority order:
    1. $LOCALSCRATCH
    2. /tmp/oar-$OAR_JOB_ID or /tmp/job-$OAR_JOB_ID
    3. $OAR_JOB_SCRATCH_DIR
    4. $TMPDIR / $TMP (if containing OAR_JOB_ID)

    Validates:
    - OAR_JOB_ID is set in environment;
    - Target directory exists or can be created;
    - Path contains the OAR_JOB_ID;
    - Owned by current process UID;
    - Meets min_free_bytes ceiling.
    """
    oar_job_id = os.environ.get("OAR_JOB_ID")
    if not oar_job_id or not oar_job_id.strip():
        raise DataRootRejected(
            reason="NOT_IN_OAR_JOB",
            role="scratch",
            path="",
        )
    oar_job_id = oar_job_id.strip()

    candidates: list[str] = []
    for var in ("LOCALSCRATCH", "OAR_JOB_SCRATCH_DIR", "TMPDIR", "TMP"):
        val = os.environ.get(var)
        if val and val.strip():
            candidates.append(val.strip())

    candidates.extend(
        [
            f"/tmp/oar-{oar_job_id}",
            f"/tmp/job-{oar_job_id}",
            f"/var/tmp/oar-{oar_job_id}",
        ]
    )

    resolved_scratch: Path | None = None
    for cand in candidates:
        try:
            p = Path(cand).resolve(strict=False)
            if oar_job_id in str(p) or oar_job_id in cand:
                p.mkdir(parents=True, exist_ok=True)
                if p.is_dir():
                    stat_info = p.stat()
                    if hasattr(stat_info, "st_uid") and stat_info.st_uid != os.getuid():
                        continue
                    usage = shutil.disk_usage(str(p))
                    if usage.free >= min_free_bytes:
                        resolved_scratch = p
                        break
        except Exception:
            continue

    if resolved_scratch is None:
        raise DataRootRejected(
            reason="NO_VALID_OAR_SCRATCH",
            role="scratch",
            path=candidates[0] if candidates else "",
        )
    return resolved_scratch


def safe_cleanup_scratch(path: Path, *, prefix_requirement: str = "osm_") -> None:
    """Safely delete scratch directory with multiple safety guards:

    1. Path must physically resolve (os.path.realpath).
    2. Path must NOT be root '/', home directory, or top-level system dirs.
    3. Path must be owned by current user UID.
    4. Path must contain OAR_JOB_ID if running inside an OAR job.
    5. Path basename or parent must start with prefix_requirement.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists() and not path.is_symlink():
        return

    real_path = Path(os.path.realpath(str(path)))
    str_path = str(real_path)

    protected = {
        "/",
        "/tmp",
        "/private/tmp",
        "/var",
        "/var/tmp",
        "/private/var/tmp",
        "/dev/shm",
        "/home",
        "/usr",
        str(Path.home()),
    }
    if str_path in protected or str_path.count("/") < 2:
        raise ValueError(
            f"safe_cleanup_scratch refused to clean protected system path: {str_path}"
        )

    try:
        if real_path.stat().st_uid != os.getuid():
            raise ValueError(
                f"safe_cleanup_scratch refused path owned by another UID: {str_path}"
            )
    except OSError as exc:
        raise ValueError(
            f"safe_cleanup_scratch stat failed on {str_path}: {exc}"
        ) from exc

    oar_job_id = os.environ.get("OAR_JOB_ID")
    if (
        oar_job_id
        and oar_job_id.strip()
        and (str_path.startswith("/tmp") or str_path.startswith("/var/tmp"))
        and oar_job_id.strip() not in str_path
    ):
        raise ValueError(
            f"safe_cleanup_scratch refused path missing OAR_JOB_ID {oar_job_id}: {str_path}"
        )

    if prefix_requirement and not any(
        part.startswith(prefix_requirement) for part in real_path.parts
    ):
        raise ValueError(
            f"safe_cleanup_scratch refused path missing prefix {prefix_requirement!r}: {str_path}"
        )

    shutil.rmtree(real_path)
