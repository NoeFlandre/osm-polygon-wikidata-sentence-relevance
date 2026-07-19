#!/usr/bin/env python3
"""Reusable artifact validators and atomic install helper
(Grid'5000 GPU smoke, Phase 9B).

The smoke shell payload validates JSON artifacts before installing
them with an atomic link-based install, and never prints the
supplied artifact paths. These helpers are private to
``scripts/grid5000/``; they never contact the network and never log
paths or sensitive values.

Public surface:

- ``ArtifactValidationError`` -- raised on any contract violation.
- ``validate_preflight(payload)`` -- strict schema for
  ``gpu_preflight.json``. Requires an *exact* set of keys.
- ``validate_smoke_result(payload)`` -- strict schema for
  ``smoke_result.json``. Requires an *exact* set of keys.
- ``install_artifact(src, dst)`` -- atomic no-clobber install
  using ``os.link`` on the same filesystem. Raises
  ``ArtifactValidationError`` on collision or non-matching
  parents. The error message references field labels and the
  failure kind, never the supplied paths.

All error messages are path-free. The caller is responsible for
translating these exceptions into shell-side labels.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ArtifactValidationError(ValueError):
    """Raised when a smoke artifact fails its contract or when an
    atomic install cannot proceed."""


# --- exact key sets (the strict schema) ------------------------------

PREFLIGHT_KEYS: frozenset[str] = frozenset(
    {
        "oar_job_id",
        "hostname",
        "torch_version",
        "torch_cuda_runtime_version",
        "visible_cuda_device_count",
        "device_0_name",
    }
)

SMOKE_KEYS: frozenset[str] = frozenset(
    {
        "resolved_device",
        "model_name",
        "input_count",
        "sentence_counts",
        "elapsed_seconds",
        "torch_version",
        "torch_cuda_runtime_version",
        "cuda_device_name",
    }
)


def _exact_keys(payload: Mapping[str, Any], allowed: frozenset[str]) -> None:
    """Raise if ``payload`` has any key not in ``allowed``, or if
    any key in ``allowed`` is missing. Both directions are
    rejected with a stable, key-only message."""
    actual = set(payload.keys())
    missing = allowed - actual
    if missing:
        raise ArtifactValidationError(f"missing required field(s): {sorted(missing)!r}")
    extra = actual - allowed
    if extra:
        raise ArtifactValidationError(f"unexpected field(s): {sorted(extra)!r}")


# --- value-level requirements ---------------------------------------


def _require_nonblank_str(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"field {field!r} must be a non-blank string")
    return value


def _require_int_eq(payload: Mapping[str, Any], field: str, expected: int) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactValidationError(
            f"field {field!r} must equal {expected} (integer)"
        )
    if value != expected:
        raise ArtifactValidationError(
            f"field {field!r} must equal {expected}; got {value!r}"
        )
    return value


def _require_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactValidationError(f"field {field!r} must be an integer")
    return value


def _require_str_eq(payload: Mapping[str, Any], field: str, expected: str) -> str:
    value = payload.get(field)
    if value != expected:
        raise ArtifactValidationError(
            f"field {field!r} must equal {expected!r}; got {value!r}"
        )
    return value


def _require_finite_nonneg_number(payload: Mapping[str, Any], field: str) -> float:
    """Reject NaN, ±inf, booleans, and negative values."""
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError(
            f"field {field!r} must be a finite, non-negative number"
        )
    fvalue = float(value)
    if not math.isfinite(fvalue):
        raise ArtifactValidationError(
            f"field {field!r} must be a finite, non-negative number"
        )
    if fvalue < 0.0:
        raise ArtifactValidationError(
            f"field {field!r} must be a finite, non-negative number"
        )
    return fvalue


def _require_positive_int_list(
    payload: Mapping[str, Any], field: str, length: int
) -> list[int]:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, list):
        raise ArtifactValidationError(
            f"field {field!r} must be a list of positive integers"
        )
    if len(value) != length:
        raise ArtifactValidationError(
            f"field {field!r} must have exactly {length} entries; got {len(value)}"
        )
    out: list[int] = []
    for idx, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ArtifactValidationError(
                f"{field}[{idx}] must be a positive integer; got {item!r}"
            )
        if item < 1:
            raise ArtifactValidationError(
                f"{field}[{idx}] must be a positive integer; got {item!r}"
            )
        out.append(item)
    return out


# --- top-level validators -------------------------------------------


def validate_preflight(payload: object) -> None:
    """Validate the gpu_preflight.json contract.

    The schema is *strict*: only the keys listed in
    :data:`PREFLIGHT_KEYS` are accepted; any missing or extra key
    is rejected with a stable message.
    """
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("preflight payload must be a JSON object")
    _exact_keys(payload, PREFLIGHT_KEYS)
    _require_nonblank_str(payload, "oar_job_id")
    _require_nonblank_str(payload, "hostname")
    _require_nonblank_str(payload, "torch_version")
    _require_nonblank_str(payload, "torch_cuda_runtime_version")
    _require_nonblank_str(payload, "device_0_name")
    _require_int_eq(payload, "visible_cuda_device_count", 1)


def validate_smoke_result(payload: object) -> None:
    """Validate the smoke_result.json contract.

    The schema is *strict*: only the keys listed in
    :data:`SMOKE_KEYS` are accepted; any missing or extra key is
    rejected with a stable message.
    """
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("smoke payload must be a JSON object")
    _exact_keys(payload, SMOKE_KEYS)
    _require_str_eq(payload, "resolved_device", "cuda")
    _require_str_eq(payload, "model_name", "sat-3l-sm")
    _require_int_eq(payload, "input_count", 3)
    _require_positive_int_list(payload, "sentence_counts", 3)
    _require_finite_nonneg_number(payload, "elapsed_seconds")
    _require_nonblank_str(payload, "torch_version")
    _require_nonblank_str(payload, "torch_cuda_runtime_version")
    _require_nonblank_str(payload, "cuda_device_name")


# --- atomic install helper (no-clobber) -----------------------------


def install_artifact(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
    """Atomically install the temporary artifact at ``src`` as the
    final artifact at ``dst``, refusing to overwrite any existing
    destination.

    The contract is implemented with ``os.link`` on the same
    filesystem: ``os.link`` raises :class:`FileExistsError` if the
    destination already exists, which the helper converts into a
    path-free :class:`ArtifactValidationError`. On success, the
    source is unlinked; if unlink fails,
    ``install_artifact`` raises
    ``ArtifactValidationError("artifact installed but temporary
    cleanup failed")`` so the caller knows the final artifact is
    in place but the source must be cleaned up by the next shell
    trap. The final artifact is never removed or overwritten.

    Both ``src`` and ``dst`` must already share a parent directory
    (the ``SMOKE_LOG_DIR`` established by the smoke script), and
    both must already exist or be representable on the local
    filesystem; this helper does not perform any path creation.

    The error message references only the failure kind; the
    supplied paths are never echoed.
    """
    src_path = Path(src)
    dst_path = Path(dst)

    # Parent-directory parity guard. The smoke only ever passes
    # temp and final paths inside the same SMOKE_LOG_DIR, so any
    # cross-directory call is a contract violation.
    if src_path.parent != dst_path.parent:
        raise ArtifactValidationError(
            "source and destination parent directories do not match"
        )
    if not src_path.is_file():
        raise ArtifactValidationError("source temporary file is not present")

    # os.link is atomic on POSIX (same-filesystem requirement
    # satisfied by the parent check above). On FileExistsError
    # we preserve the existing destination byte-for-byte by
    # never attempting any destructive operation.
    try:
        os.link(src_path, dst_path)
    except FileExistsError as exc:
        # The destination already exists -- the contract forbids
        # overwriting. We convert to a stable path-free error.
        raise ArtifactValidationError(
            "destination artifact already exists; not overwriting"
        ) from exc
    except OSError as exc:
        # Any other OS-level failure (cross-device, permission,
        # etc.) is reported as a stable label.
        raise ArtifactValidationError(
            "atomic install failed (filesystem error)"
        ) from exc

    # Source cleanup -- only after a successful atomic install.
    # If unlink fails, the final artifact must remain in place;
    # we surface a stable path-free error so the caller can take
    # a deterministic action (the active shell trap will retry).
    try:
        os.unlink(src_path)
    except OSError as exc:
        raise ArtifactValidationError(
            "artifact installed but temporary cleanup failed"
        ) from exc


__all__ = [
    "ArtifactValidationError",
    "PREFLIGHT_KEYS",
    "SMOKE_KEYS",
    "install_artifact",
    "validate_preflight",
    "validate_smoke_result",
]


# --- CLI ----------------------------------------------------------
#
# Usage from the smoke shell payload:
#
#   python <abs-path-to-this-file> preflight <json-file>
#   python <abs-path-to-this-file> smoke-result <json-file>
#
# Usage for the install helper:
#
#   python <abs-path-to-this-file> install <src-path> <dst-path>
#
# All error messages from this CLI are path-free. The shell wraps
# the call and emits only its own stable label when validation
# fails.


def _cli(argv: list[str]) -> int:
    import json as _json
    import sys as _sys

    if len(argv) < 2:
        _sys.stderr.write(
            "usage: _validate_artifact.py {preflight|smoke-result|install} ...\n"
        )
        return 2

    command = argv[1]
    try:
        if command == "preflight":
            if len(argv) != 3:
                _sys.stderr.write("preflight: expected one argument\n")
                return 2
            with open(argv[2], encoding="utf-8") as fh:
                payload = _json.load(fh)
            validate_preflight(payload)
            return 0
        if command == "smoke-result":
            if len(argv) != 3:
                _sys.stderr.write("smoke-result: expected one argument\n")
                return 2
            with open(argv[2], encoding="utf-8") as fh:
                payload = _json.load(fh)
            validate_smoke_result(payload)
            return 0
        if command == "install":
            if len(argv) != 4:
                _sys.stderr.write("install: expected two arguments\n")
                return 2
            install_artifact(argv[2], argv[3])
            return 0
    except ArtifactValidationError as exc:
        _sys.stderr.write(f"validation failed: {exc}\n")
        return 1
    except FileNotFoundError:
        _sys.stderr.write("validation failed: source or json file is missing\n")
        return 1
    except _json.JSONDecodeError:
        _sys.stderr.write("validation failed: json is not parseable\n")
        return 1

    _sys.stderr.write(f"unknown subcommand: {command!r}\n")
    return 2


if __name__ == "__main__":  # pragma: no cover -- CLI dispatcher
    import sys as _sys_main

    raise SystemExit(_cli(_sys_main.argv))
