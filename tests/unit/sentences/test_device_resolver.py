"""Tests for the SaT device resolver (Phase 9A).

The resolver must:

- accept only ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``;
- reject non-string, blank, and unknown values with ``SegmentationError``;
- resolve ``"auto"`` in priority order ``cuda`` → ``mps`` → ``cpu``;
- fail early and clearly for explicit ``cuda`` / ``mps`` when unavailable;
- be testable with an injected Torch capability object (no real CUDA/MPS).
"""

from __future__ import annotations

import pytest

from osm_polygon_sentence_relevance.contracts.errors import SegmentationError
from osm_polygon_sentence_relevance.sentences.device import (
    PUBLIC_DEVICE_VALUES,
    resolve_device,
)


class _Caps:
    """Minimal fake of a torch capability snapshot."""

    def __init__(self, *, cuda=False, mps=False):
        self.cuda_available = cuda
        self.mps_available = mps


class TestDevicePublicValues:
    def test_public_values_match_documented_set(self):
        assert set(PUBLIC_DEVICE_VALUES) == {"auto", "cpu", "cuda", "mps"}


class TestResolveAuto:
    def test_auto_prefers_cuda_when_available(self):
        caps = _Caps(cuda=True, mps=True)
        assert resolve_device("auto", caps=caps) == "cuda"

    def test_auto_falls_back_to_mps_when_cuda_unavailable(self):
        caps = _Caps(cuda=False, mps=True)
        assert resolve_device("auto", caps=caps) == "mps"

    def test_auto_falls_back_to_cpu_when_neither_available(self):
        caps = _Caps(cuda=False, mps=False)
        assert resolve_device("auto", caps=caps) == "cpu"

    def test_auto_cuda_only(self):
        caps = _Caps(cuda=True, mps=False)
        assert resolve_device("auto", caps=caps) == "cuda"


class TestResolveExplicit:
    def test_explicit_cpu_always_resolves_to_cpu(self):
        caps = _Caps(cuda=True, mps=True)
        assert resolve_device("cpu", caps=caps) == "cpu"

    def test_explicit_cuda_succeeds_when_available(self):
        caps = _Caps(cuda=True, mps=True)
        assert resolve_device("cuda", caps=caps) == "cuda"

    def test_explicit_mps_succeeds_when_available(self):
        caps = _Caps(cuda=False, mps=True)
        assert resolve_device("mps", caps=caps) == "mps"

    def test_explicit_cuda_fails_when_unavailable(self):
        caps = _Caps(cuda=False, mps=True)
        with pytest.raises(SegmentationError, match="cuda"):
            resolve_device("cuda", caps=caps)

    def test_explicit_mps_fails_when_unavailable(self):
        caps = _Caps(cuda=True, mps=False)
        with pytest.raises(SegmentationError, match="mps"):
            resolve_device("mps", caps=caps)

    def test_explicit_cuda_fails_when_neither_available(self):
        caps = _Caps(cuda=False, mps=False)
        with pytest.raises(SegmentationError, match="cuda"):
            resolve_device("cuda", caps=caps)

    def test_explicit_mps_fails_when_neither_available(self):
        caps = _Caps(cuda=False, mps=False)
        with pytest.raises(SegmentationError, match="mps"):
            resolve_device("mps", caps=caps)

    def test_explicit_unavailable_does_not_silently_fall_back(self):
        # Only CPU is available. Explicit cuda/mps must each raise.
        caps = _Caps(cuda=False, mps=False)
        # cpu must NOT be returned when mps is requested but unavailable
        with pytest.raises(SegmentationError):
            resolve_device("mps", caps=caps)
        # cpu must NOT be returned when cuda is requested but unavailable
        with pytest.raises(SegmentationError):
            resolve_device("cuda", caps=caps)


class TestResolveRejects:
    @pytest.mark.parametrize("bad", [None, 0, 1, b"cpu", [], {}])
    def test_non_string_rejected(self, bad):
        with pytest.raises(SegmentationError, match="device"):
            resolve_device(bad, caps=_Caps())

    @pytest.mark.parametrize(
        "bad", ["", " ", "\t\n", "CPU", "Cuda", "MPS", "gpu", "tpu", "auto "]
    )
    def test_blank_or_unknown_rejected(self, bad):
        with pytest.raises(SegmentationError, match="device"):
            resolve_device(bad, caps=_Caps())


class TestResolveErrorShape:
    def test_error_message_mentions_requested_value(self):
        caps = _Caps(cuda=False, mps=False)
        with pytest.raises(SegmentationError, match="cuda"):
            resolve_device("cuda", caps=caps)

    def test_error_preserves_no_internal_cause(self):
        # Resolver is pure logic; failures have no underlying exception.
        with pytest.raises(SegmentationError) as exc:
            resolve_device("gpu", caps=_Caps())
        assert exc.value.__cause__ is None


class TestDefaultCaps:
    """The production ``_DefaultCaps`` lazily probes Torch.

    Tests in this class do NOT inject caps; they exercise the actual
    production probe. Each property must return ``bool`` and must
    never raise.
    """

    def test_default_caps_lazy_probes_torch(self):
        from osm_polygon_sentence_relevance.sentences.device import default_caps

        caps = default_caps()
        assert isinstance(caps.cuda_available, bool)
        assert isinstance(caps.mps_available, bool)

    def test_default_caps_handles_missing_torch(self, monkeypatch):
        """When Torch is not installed, the default caps must return
        ``False`` for both flags rather than raising.
        """
        import builtins

        from osm_polygon_sentence_relevance.sentences.device import _DefaultCaps

        real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)
        caps = _DefaultCaps()
        assert caps.cuda_available is False
        assert caps.mps_available is False

    def test_default_caps_handles_mps_module_absent(self, monkeypatch):
        """On non-macOS builds ``torch.backends.mps`` is absent; the
        default caps must return ``False`` for ``mps_available``.
        """
        from osm_polygon_sentence_relevance.sentences.device import _DefaultCaps

        class _FakeBackends:
            pass

        import torch

        real_backends = torch.backends
        monkeypatch.setattr(torch, "backends", _FakeBackends())
        try:
            caps = _DefaultCaps()
            assert caps.mps_available is False
        finally:
            monkeypatch.setattr(torch, "backends", real_backends)

    def test_default_caps_handles_mps_is_available_exception(self, monkeypatch):
        """If ``torch.backends.mps.is_available()`` raises, the default
        caps must catch and return ``False``.
        """

        class _BoomMPS:
            def is_available(self):
                raise RuntimeError("boom")

        class _BackendsWithMPS:
            mps = _BoomMPS()

        import torch

        real_backends = torch.backends
        monkeypatch.setattr(torch, "backends", _BackendsWithMPS())
        try:
            from osm_polygon_sentence_relevance.sentences.device import _DefaultCaps

            caps = _DefaultCaps()
            assert caps.mps_available is False
        finally:
            monkeypatch.setattr(torch, "backends", real_backends)

    def test_default_caps_handles_cuda_is_available_exception(self, monkeypatch):
        """If ``torch.cuda.is_available()`` raises, the default caps
        must catch and return ``False``.
        """
        from osm_polygon_sentence_relevance.sentences.device import _DefaultCaps

        class _BoomCuda:
            def is_available(self):
                raise RuntimeError("boom")

        import torch

        real_cuda = torch.cuda
        monkeypatch.setattr(torch, "cuda", _BoomCuda())
        try:
            caps = _DefaultCaps()
            assert caps.cuda_available is False
        finally:
            monkeypatch.setattr(torch, "cuda", real_cuda)
