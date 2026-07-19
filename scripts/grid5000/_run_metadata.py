#!/usr/bin/env python3
r"""Atomic run-metadata writer for the Grid'5000 GPU smoke (Phase 9B).

This helper writes a small, deterministic JSON file describing the
provenance of a single smoke invocation. It is invoked by the
smoke shell payload *before* the SaT inference phase.

Atomic-install contract:

- the parent directory must already exist; no creation;
- the temporary file is unique (created via :func:`tempfile.mkstemp`
  inside the parent directory), so two concurrent calls cannot
  collide;
- ``os.fchmod(fd, 0o600)`` is invoked on the open descriptor
  before close, so the temp file is mode-restrictive even under
  a permissive umask;
- the JSON payload is written, ``flush``\ ed, and ``fsync``\ ed
  before the atomic install;
- the destination is installed via :func:`os.link` (POSIX atomic
  on the same filesystem, ``FileExistsError`` if destination
  already exists); the temp file is unlinked only after a
  successful link;
- on any *premature* failure (before the link succeeds) the temp
  file is cleaned up and the helper raises a path-free
  :class:`RunMetadataError`;
- failure to unlink the temp file *after* a successful link
  raises ``RunMetadataError("artifact installed but temporary
  cleanup failed")`` so the operator knows to investigate;
- the destination is **never** removed or overwritten by this
  helper.

Errors are stable, path-free messages:

- ``"temporary creation failed"``
- ``"temporary permission setup failed"``
- ``"temporary write/sync failed"``
- ``"atomic install failed"``
- ``"destination already exists"``
- ``"artifact installed but temporary cleanup failed"``

No filesystem path is ever embedded in the error string; only the
failing kind is named. ``__cause__`` is preserved so operators can
still introspect the underlying OSError when needed.

Schema contract: exactly the seven keys listed in
:data:`RUN_METADATA_KEYS`; any missing or extra key is rejected.

Field-format contract:

- ``source_commit``, ``model_revision``, ``tokenizer_revision``
  must each be exactly 40 lowercase hexadecimal characters;
- ``model_name`` must equal ``"sat-3l-sm"``;
- ``tokenizer_name`` must equal ``"facebookAI/xlm-roberta-base"``;
- ``oar_job_id`` and ``hostname`` must be non-blank strings.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path


class RunMetadataError(ValueError):
    """Raised when a run-metadata payload fails its contract.

    Error messages are path-free; only the failing kind is named
    (e.g. ``"atomic install failed"``). The underlying OSError is
    preserved via ``__cause__``.
    """


RUN_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "source_commit",
        "model_name",
        "model_revision",
        "tokenizer_name",
        "tokenizer_revision",
        "oar_job_id",
        "hostname",
    }
)

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_MODEL_NAME = "sat-3l-sm"
EXPECTED_TOKENIZER_NAME = "facebookAI/xlm-roberta-base"

_TEMP_MODE = 0o600


def _require_hex_40(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or _HEX_40.match(value) is None:
        raise RunMetadataError(
            f"field {field!r} must be exactly 40 lowercase hex characters"
        )
    return value


def _require_equal(value: object, expected: str, field: str) -> str:
    if not isinstance(value, str) or value != expected:
        raise RunMetadataError(f"field {field!r} must equal {expected!r}")
    return value


def _require_nonblank_str(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RunMetadataError(f"field {field!r} must be a non-blank string")
    return value


def _validate_payload_shape(payload: Mapping[str, object]) -> None:
    """Strict-schema shape validation."""
    if set(payload.keys()) != RUN_METADATA_KEYS:
        missing = sorted(RUN_METADATA_KEYS - set(payload.keys()))
        extra = sorted(set(payload.keys()) - RUN_METADATA_KEYS)
        raise RunMetadataError(
            f"unexpected key set: missing={missing!r} extra={extra!r}"
        )
    _require_hex_40(payload, "source_commit")
    _require_hex_40(payload, "model_revision")
    _require_hex_40(payload, "tokenizer_revision")
    _require_equal(payload.get("model_name"), EXPECTED_MODEL_NAME, "model_name")
    _require_equal(
        payload.get("tokenizer_name"),
        EXPECTED_TOKENIZER_NAME,
        "tokenizer_name",
    )
    _require_nonblank_str(payload, "oar_job_id")
    _require_nonblank_str(payload, "hostname")


def _safe_unlink(path: Path) -> bool:
    """Best-effort ``os.unlink``. Returns ``True`` on success,
    ``False`` on any OSError (the caller may want to surface
    post-link cleanup failures)."""
    try:
        os.unlink(path)
    except OSError:
        return False
    return True


def write_run_metadata(
    dst: os.PathLike[str] | str,
    source_commit: str,
    model_name: str,
    model_revision: str,
    tokenizer_name: str,
    tokenizer_revision: str,
    oar_job_id: str,
    hostname: str,
) -> None:
    """Atomically install the seven-key metadata JSON at ``dst``
    with mode ``0600``. Refuses to overwrite (the destination
    must not exist; ``os.link`` will fail on collision and
    ``FileExistsError`` is converted to ``RunMetadataError``).
    """
    dst_path = Path(dst)
    parent = dst_path.parent
    if not parent.is_dir():
        raise RunMetadataError("destination parent directory is missing")

    payload = {
        "source_commit": source_commit,
        "model_name": model_name,
        "model_revision": model_revision,
        "tokenizer_name": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "oar_job_id": oar_job_id,
        "hostname": hostname,
    }
    _validate_payload_shape(payload)

    # mkstemp yields a unique temp path AND an open fd. The path is
    # already in the parent directory so os.link is atomic on the
    # same filesystem (no cross-device move).
    try:
        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(parent),
            prefix=f".{dst_path.name}.",
            suffix=".tmp",
        )
    except OSError as exc:
        raise RunMetadataError("temporary creation failed") from exc
    tmp_path = Path(tmp_path_str)

    # From here on, every failure must (a) close ``fd`` exactly
    # once, (b) unlink the temp file, and (c) raise a
    # path-free RunMetadataError whose ``__cause__`` is the
    # original OSError. We use a per-phase ``try/except`` so
    # fchmod failures, fdopen/write/flush/fsync failures, and
    # ``os.link`` failures each translate to a distinct,
    # documented label.
    try:
        try:
            os.fchmod(fd, _TEMP_MODE)
        except OSError as exc:
            # The fd is still open; close it before propagating.
            with contextlib.suppress(OSError):
                os.close(fd)
            raise RunMetadataError("temporary permission setup failed") from exc

        # From here the fd is owned by the ``fdopen`` context
        # manager. ``os.fdopen`` itself may raise (e.g. on
        # invalid mode), in which case ownership has not yet
        # transferred; we close the fd manually.
        try:
            fh_cm = os.fdopen(fd, "w", encoding="utf-8")
        except OSError as exc:
            with contextlib.suppress(OSError):
                os.close(fd)
            raise RunMetadataError("temporary write/sync failed") from exc

        try:
            with fh_cm as fh:
                json.dump(payload, fh, sort_keys=True)
                fh.write("\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError as exc:
                    raise RunMetadataError("temporary write/sync failed") from exc
        except RunMetadataError:
            raise
        except OSError as exc:
            raise RunMetadataError("temporary write/sync failed") from exc
    except BaseException:
        _safe_unlink(tmp_path)
        raise

    # Atomic install (POSIX ``os.link`` is atomic on the same
    # filesystem). The destination must not pre-exist; on
    # collision we surface a stable, path-free error.
    try:
        os.link(tmp_path, dst_path)
    except FileExistsError as exc:
        _safe_unlink(tmp_path)
        raise RunMetadataError("destination already exists; not overwriting") from exc
    except OSError as exc:
        _safe_unlink(tmp_path)
        raise RunMetadataError("atomic install failed") from exc

    # Atomic install succeeded. If temp cleanup fails the artifact
    # has already been written; surface the cleanup failure so the
    # operator can investigate. The destination stays intact and
    # the underlying OSError is preserved on ``__cause__`` (the
    # whole point of the contract is that operators can introspect
    # the original failure without it having been swallowed).
    try:
        os.unlink(tmp_path)
    except OSError as exc:
        raise RunMetadataError(
            "artifact installed but temporary cleanup failed"
        ) from exc


__all__ = [
    "EXPECTED_MODEL_NAME",
    "EXPECTED_TOKENIZER_NAME",
    "RUN_METADATA_KEYS",
    "RunMetadataError",
    "write_run_metadata",
]
