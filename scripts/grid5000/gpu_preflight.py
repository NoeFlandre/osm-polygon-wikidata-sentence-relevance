#!/usr/bin/env python3
"""Grid'5000 CUDA pre-flight check (Phase 9B + Phase 9H).

This script runs *inside* an allocated Grid'5000 compute node. It is a
narrow invariant gate that proves the runtime can actually use a CUDA
GPU before any SaT model is constructed. It is intentionally tiny and
has no dependency on the project package.

Design constraints (Phase 9B + Phase 9H):

- Linux only (Grid'5000 compute nodes are Linux).
- Requires a non-blank ``OAR_JOB_ID`` (we are inside an OAR
  allocation).
- ``CUDA_VISIBLE_DEVICES`` is **informational only**: Grid'5000
  scopes reserved GPUs through its resource isolation and does
  not guarantee ``CUDA_VISIBLE_DEVICES`` is set. The preflight
  does NOT read it. The authoritative runtime proof of GPU
  scoping is the combination of:
    - ``torch.cuda.is_available() is True``, and
    - ``torch.cuda.device_count() == 1``,
    - ``torch.cuda.get_device_name(0)`` succeeds.
- Imports Torch and requires ``torch.cuda.is_available() is True``.
- Requires exactly one visible CUDA device.
- Emits a single stable JSON object on stdout with:
  - ``oar_job_id``
  - ``hostname``
  - ``torch_version``
  - ``torch_cuda_runtime_version``
  - ``visible_cuda_device_count``
  - ``device_0_name``
- The result schema is unchanged from Phase 9B: no new fields.
- Never mutates ``os.environ``.
- Never prints environment variables wholesale, tokens, credentials,
  cache contents, or usernames.
- Exits non-zero with a concise actionable message if any invariant
  fails.

Testability: every runtime touch (platform, environment, torch) is
injected through the ``PreflightEnv`` protocol so unit tests run on the
Mac without a GPU. The module never contacts the network.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Protocol


class PreflightEnv(Protocol):
    """Injectable environment/platform/torch facade for tests."""

    @property
    def system(self) -> str:
        """``platform.system()`` value, e.g. ``"Linux"``."""
        ...

    @property
    def node_name(self) -> str:
        """Stable node identifier (``socket.gethostname()`` in prod)."""
        ...

    def getenv(self, name: str, default: str | None = None) -> str | None:
        """Read an environment variable (``os.getenv`` in prod)."""
        ...

    def torch_factory(self) -> Any:
        """Return the torch module (imported lazily in prod).

        May raise ``ImportError`` on hosts without Torch.
        """
        ...


@dataclass
class _RealPreflightEnv:
    """Production preflight environment backed by the real platform/os."""

    _torch: Any = field(default=None, init=False)

    @property
    def system(self) -> str:
        import platform

        return platform.system()

    @property
    def node_name(self) -> str:
        import socket

        return socket.gethostname()

    def getenv(self, name: str, default: str | None = None) -> str | None:
        return os.getenv(name, default)

    def torch_factory(self) -> Any:
        import torch  # imported lazily; may raise ImportError

        return torch


@dataclass
class PreflightResult:
    """Structured, JSON-serialisable preflight report.

    The schema is unchanged from Phase 9B. The six documented keys
    are the only fields ever emitted.
    """

    oar_job_id: str
    hostname: str
    torch_version: str
    torch_cuda_runtime_version: str
    visible_cuda_device_count: int
    device_0_name: str

    def to_json(self) -> str:
        import json

        return json.dumps(
            {
                "oar_job_id": self.oar_job_id,
                "hostname": self.hostname,
                "torch_version": self.torch_version,
                "torch_cuda_runtime_version": self.torch_cuda_runtime_version,
                "visible_cuda_device_count": self.visible_cuda_device_count,
                "device_0_name": self.device_0_name,
            },
            sort_keys=True,
        )


class PreflightError(RuntimeError):
    """Raised when a pre-flight invariant is not satisfied."""


def _fail(message: str) -> PreflightError:
    return PreflightError(f"gpu_preflight: {message}")


def run_preflight(
    env: PreflightEnv | None = None,
    *,
    torch_mod: Any | None = None,
) -> PreflightResult:
    """Execute the CUDA pre-flight check.

    Parameters
    ----------
    env:
        Injectable environment facade. When ``None``, the real
        platform/os-backed facade is used (production).
    torch_mod:
        Injectable torch module. When provided it is used instead of
        calling ``env.torch_factory()``; this keeps tests free of real
        Torch imports. In production ``torch_mod`` is ``None`` and the
        module is imported through ``env.torch_factory()``.

    Returns
    -------
    PreflightResult

    Raises
    ------
    PreflightError
        If any invariant (Linux, OAR job, CUDA availability,
        exactly-one-device) fails.
    """
    if env is None:
        env = _RealPreflightEnv()

    # 1. Linux only.
    if env.system != "Linux":
        raise _fail(
            f"expected Linux compute node, got platform {env.system!r}; "
            "gpu_preflight must run inside a Grid'5000 OAR allocation"
        )

    # 2. Non-blank OAR_JOB_ID.
    oar_job_id = env.getenv("OAR_JOB_ID")
    if not oar_job_id or not oar_job_id.strip():
        raise _fail(
            "OAR_JOB_ID is unset; gpu_preflight must run inside an "
            "allocated OAR job (oarsub)"
        )

    # 3. Torch present and CUDA available.
    if torch_mod is None:
        try:
            torch_mod = env.torch_factory()
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise _fail("torch is not importable in this environment") from exc

    if not torch_mod.cuda.is_available():
        raise _fail("torch.cuda.is_available() is False; no usable CUDA device")

    # 4. Exactly one visible CUDA device (the project deliberately
    # requests a single GPU; multi-GPU is not implemented and
    # would mask an incorrectly scoped OAR allocation). This is
    # the authoritative runtime proof that Grid'5000 scoped the
    # ``gpu=1`` request correctly. CUDA_VISIBLE_DEVICES is not
    # read here on purpose: it is not part of the guaranteed
    # scheduler contract.
    device_count = torch_mod.cuda.device_count()
    if device_count != 1:
        raise _fail(
            f"expected exactly one visible CUDA device, got "
            f"{device_count}; the OAR request must scope a single GPU"
        )

    device_0_name = torch_mod.cuda.get_device_name(0)

    return PreflightResult(
        oar_job_id=oar_job_id,
        hostname=env.node_name,
        torch_version=torch_mod.__version__,
        torch_cuda_runtime_version=torch_mod.version.cuda,
        visible_cuda_device_count=device_count,
        device_0_name=device_0_name,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code.

    Reads only the environment (through the real platform/os facade).
    Emits the JSON report on success; writes a concise error to
    stderr and returns 1 on any invariant failure.

    Tests inject a fake environment through :func:`run_with`.
    """
    return run_with(_RealPreflightEnv(), argv=argv)


def run_with(
    env: PreflightEnv,
    *,
    torch_mod: Any | None = None,
    argv: list[str] | None = None,
) -> int:
    """Execute :func:`main` logic against an injectable environment.

    Returns a process exit code (0 on success, 1 on any
    invariant failure). The JSON report is written to stdout on
    success; a concise error is written to stderr on failure.
    """
    _ = argv  # reserved; the script reads only the environment.
    try:
        result = run_preflight(env, torch_mod=torch_mod)
    except PreflightError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # broad: surface any unexpected failure
        print(f"gpu_preflight: unexpected error: {exc}", file=sys.stderr)
        return 1
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
