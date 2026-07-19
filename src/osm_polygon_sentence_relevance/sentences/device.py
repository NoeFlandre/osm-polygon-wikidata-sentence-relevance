"""Device resolver for SaT inference (Phase 9A).

The resolver is pure logic over a small capability snapshot. It does not
import Torch and does not perform any model construction or movement. The
capability object exposes two boolean fields:

- ``cuda_available``
- ``mps_available``

These match the names returned by ``torch.cuda.is_available()`` and
``torch.backends.mps.is_available()``; the public function accepts an
injected ``caps`` object so tests do not need a real accelerator.

Public device values:

- ``"auto"`` — picks the first available in priority order
  CUDA → MPS → CPU;
- ``"cpu"`` — always resolves to ``"cpu"``;
- ``"cuda"`` — resolves to ``"cuda"`` only when ``cuda_available`` is True;
- ``"mps"`` — resolves to ``"mps"`` only when ``mps_available`` is True.

Explicit ``"cuda"`` or ``"mps"`` requests are never silently downgraded.
Failures raise :class:`~.contracts.errors.SegmentationError`.
"""

from __future__ import annotations

from typing import Protocol

from osm_polygon_sentence_relevance.contracts.errors import SegmentationError

PUBLIC_DEVICE_VALUES: frozenset[str] = frozenset({"auto", "cpu", "cuda", "mps"})


class TorchCapabilities(Protocol):
    """Minimal Torch capability snapshot.

    Implementations are expected to expose the two booleans below; in
    production these come from ``torch.cuda.is_available()`` and
    ``torch.backends.mps.is_available()``.
    """

    cuda_available: bool
    mps_available: bool


def resolve_device(
    value: object,
    *,
    caps: TorchCapabilities,
) -> str:
    """Resolve a requested device value against a capability snapshot.

    Parameters
    ----------
    value:
        One of the four public device strings. Non-strings, blank strings,
        and unknown strings are rejected with ``SegmentationError``.
    caps:
        An object exposing ``cuda_available`` and ``mps_available``.

    Returns
    -------
    str
        One of ``"cpu"``, ``"cuda"``, ``"mps"``.
    """
    if not isinstance(value, str):
        raise SegmentationError(
            f"device must be one of {sorted(PUBLIC_DEVICE_VALUES)}; got "
            f"{type(value).__name__}"
        )
    if not value.strip():
        raise SegmentationError("device cannot be blank")
    if value not in PUBLIC_DEVICE_VALUES:
        raise SegmentationError(
            f"device must be one of {sorted(PUBLIC_DEVICE_VALUES)}; got {value!r}"
        )

    cuda_ok = bool(getattr(caps, "cuda_available", False))
    mps_ok = bool(getattr(caps, "mps_available", False))

    if value == "cpu":
        return "cpu"
    if value == "cuda":
        if not cuda_ok:
            raise SegmentationError(
                "device 'cuda' was requested but no CUDA backend is "
                "available on this host"
            )
        return "cuda"
    if value == "mps":
        if not mps_ok:
            raise SegmentationError(
                "device 'mps' was requested but the MPS backend is not "
                "available on this host"
            )
        return "mps"

    # value == "auto"
    if cuda_ok:
        return "cuda"
    if mps_ok:
        return "mps"
    return "cpu"


__all__ = [
    "PUBLIC_DEVICE_VALUES",
    "TorchCapabilities",
    "resolve_device",
    "default_caps",
]


class _DefaultCaps:
    """Production capability snapshot: queries Torch lazily."""

    @property
    def cuda_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @property
    def mps_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        # ``torch.backends.mps`` is only present on macOS builds.
        mps_mod = getattr(torch.backends, "mps", None)
        if mps_mod is None:
            return False
        try:
            return bool(mps_mod.is_available())
        except Exception:
            return False


def default_caps() -> TorchCapabilities:
    """Return the default production capability snapshot.

    Imports Torch lazily on first access; never raises.
    """
    # ``_DefaultCaps`` exposes read-only ``@property`` attributes, which
    # is intentional (capabilities are queried lazily and never mutated).
    # The Protocol type is structurally satisfied.
    return _DefaultCaps()  # type: ignore[return-value]
