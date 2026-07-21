"""Tests for ``SaTSentenceSegmenter`` device selection (the implementation).

The segmenter must:

- default to ``auto`` and remain backward compatible;
- construct the model lazily (empty batch ⇒ no model);
- resolve the device exactly once, immediately before first model
  construction;
- place the *complete classifier* (owned by the
  ``wtpsplit.extract.PyTorchWrapper``) on the resolved device via one
  ``.to(resolved_device)`` call on the classifier;
- verify placement by reading every parameter/buffer device of the
  classifier rather than assume ``.to(...)`` succeeded;
- wrap any placement / construction / inference error as
  ``SegmentationError`` with ``__cause__`` preserved;
- never silently fall back when the user explicitly requested CUDA or MPS;
- reuse the placed model across batches.

These tests do NOT import the real ``wtpsplit`` package; they patch the
helper's class loader to return a stand-in class. The optional-dep
contract (``wtpsplit`` not loaded on a host without the segmentation
extra) is preserved.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from osm_polygon_sentence_relevance.contracts.errors import SegmentationError
from osm_polygon_sentence_relevance.sentences import _wtpsplit_device as _wtpsplit_mod
from osm_polygon_sentence_relevance.sentences.sat import SaTSentenceSegmenter


class _MockPyTorchWrapper:
    """Stand-in for ``wtpsplit.extract.PyTorchWrapper``.

    Holds ``self.model`` pointing at the classifier and delegates every
    other attribute to it via ``__getattr__``.
    """

    def __init__(self, inner: nn.Module) -> None:
        self.model = inner

    def __getattr__(self, name: str) -> object:
        if name in {"model"}:
            raise AttributeError(name)
        return getattr(self.model, name)


@pytest.fixture(autouse=True)
def _patch_pyTorchWrapper_class(monkeypatch):
    """Patch the helper's PyTorchWrapper loader to return our stand-in."""
    monkeypatch.setattr(
        _wtpsplit_mod,
        "_load_wtpsplit_pytorch_wrapper_class",
        lambda: _MockPyTorchWrapper,
    )
    return _MockPyTorchWrapper


class _Classifier(nn.Module):
    """Stand-in for ``SubwordXLMForTokenClassification``.

    Mirrors the real layout: a ``.model`` backbone and a separate
    ``.classifier`` head. Both live on this object; both must move
    together when the classifier is placed on a device.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(8, 8)
        self.classifier = nn.Linear(8, 2)
        # ``wtpsplit.extract.PyTorchWrapper`` reads ``model.config`` in
        # its ``__init__``; provide a dummy config attribute for the fake.
        self.config = type("Cfg", (), {"model_type": "xlm-roberta"})()


class _ClassifierFacade:
    """Mimics ``wtpsplit.SaT``: façade -> PyTorchWrapper -> classifier."""

    def __init__(self, model_name: str, **kwargs: object) -> None:
        self.model_name = model_name
        self.kwargs = dict(kwargs)
        self.classifier = _Classifier()
        self.model = _MockPyTorchWrapper(self.classifier)

    def split(self, texts, **kwargs):
        # The split entry point in real SaT lives on the façade and uses
        # the wrapper internally. For these tests we just emit trivial
        # groupings so we can observe placement without exercising the
        # full inference path.
        return [t.split("|") for t in texts]


def _factory():
    def factory(model_name, **kwargs):
        return _ClassifierFacade(model_name, **kwargs)

    return factory


class _Caps:
    def __init__(self, *, cuda=False, mps=False):
        self.cuda_available = cuda
        self.mps_available = mps


@pytest.fixture
def caps_cpu_only():
    return _Caps(cuda=False, mps=False)


class TestDeviceDefaultAndCompatibility:
    def test_default_device_is_auto(self):
        seg = SaTSentenceSegmenter(model_factory=_factory())
        assert seg.requested_device == "auto"

    def test_explicit_auto_works_with_caps_cpu(self, caps_cpu_only):
        seg = SaTSentenceSegmenter(
            model_factory=_factory(),
            caps=caps_cpu_only,
        )
        out = seg.split_batch(["a|b"], ["en"])
        assert out == (("a", "b"),)
        assert seg.resolved_device == "cpu"

    def test_explicit_cpu_resolves_to_cpu(self, caps_cpu_only):
        seg = SaTSentenceSegmenter(
            model_factory=_factory(),
            caps=caps_cpu_only,
            device="cpu",
        )
        seg.split_batch(["a|b"], ["en"])
        assert seg.resolved_device == "cpu"

    def test_explicit_cuda_fails_when_unavailable(self, caps_cpu_only):
        seg = SaTSentenceSegmenter(
            model_factory=_factory(),
            caps=caps_cpu_only,
            device="cuda",
        )
        with pytest.raises(SegmentationError, match="cuda"):
            seg.split_batch(["a|b"], ["en"])

    def test_explicit_mps_fails_when_unavailable(self, caps_cpu_only):
        seg = SaTSentenceSegmenter(
            model_factory=_factory(),
            caps=caps_cpu_only,
            device="mps",
        )
        with pytest.raises(SegmentationError, match="mps"):
            seg.split_batch(["a|b"], ["en"])

    def test_explicit_unavailable_does_not_silently_fall_back(self, caps_cpu_only):
        # Must raise even when CPU is available; never silently degrade.
        seg = SaTSentenceSegmenter(
            model_factory=_factory(),
            caps=caps_cpu_only,
            device="cuda",
        )
        with pytest.raises(SegmentationError):
            seg.split_batch(["a|b"], ["en"])


class TestDeviceAutoPriority:
    def test_auto_picks_cpu_when_neither_available(self, caps_cpu_only):
        seg = SaTSentenceSegmenter(model_factory=_factory(), caps=caps_cpu_only)
        seg.split_batch(["a|b"], ["en"])
        assert seg.resolved_device == "cpu"

    def test_explicit_cpu_places_complete_classifier_on_cpu(self, caps_cpu_only):
        seg = SaTSentenceSegmenter(
            model_factory=_factory(),
            caps=caps_cpu_only,
            device="cpu",
        )
        seg.split_batch(["a|b"], ["en"])
        sat = seg._model  # type: ignore[attr-defined]
        classifier = sat.classifier  # type: ignore[attr-defined]
        for tensor in classifier.parameters():
            assert tensor.device.type == "cpu"
        for tensor in classifier.buffers():
            assert tensor.device.type == "cpu"


class TestLazyConstructionAndEmpty:
    def test_no_model_until_first_non_empty_batch(self, caps_cpu_only):
        seg = SaTSentenceSegmenter(model_factory=_factory(), caps=caps_cpu_only)
        assert seg._model is None  # type: ignore[attr-defined]
        assert seg.split_batch([], []) == ()
        # Still no model after empty batch
        assert seg._model is None  # type: ignore[attr-defined]


class TestPlacementInvocation:
    def test_model_reused_across_batches(self, caps_cpu_only):
        seg = SaTSentenceSegmenter(model_factory=_factory(), caps=caps_cpu_only)
        seg.split_batch(["a|b"], ["en"])
        sat_first = seg._model  # type: ignore[attr-defined]
        seg.split_batch(["c|d"], ["en"])
        sat_second = seg._model  # type: ignore[attr-defined]
        assert sat_first is sat_second


class TestPlacementVerification:
    def test_failed_to_raises_actionable_error(self, monkeypatch):
        """A classifier whose ``.to(device)`` is a no-op must be detected
        by the placement-verification step. The segmenter must surface
        the requested and the observed device along with the model name.

        We simulate the no-op by monkey-patching the verification
        helper to report a device that does not match the requested
        one.
        """

        # Force the verification helper to report ``"xpu"`` regardless
        # of the actual parameter devices. The placement helper calls
        # ``.to("cpu")`` then verifies; with the patched observer, the
        # verification observes a mismatch and raises.
        def _fake_observed(_classifier: object) -> str:
            return "xpu"

        monkeypatch.setattr(
            _wtpsplit_mod, "_classifier_observed_device", _fake_observed
        )

        def factory(model_name, **kwargs):
            return _ClassifierFacade(model_name, **kwargs)

        caps = _Caps(cuda=False, mps=False)
        seg = SaTSentenceSegmenter(model_factory=factory, caps=caps, device="cpu")
        with pytest.raises(SegmentationError) as exc:
            seg.split_batch(["a|b"], ["en"])
        msg = str(exc.value)
        assert "cpu" in msg
        assert "xpu" in msg


class TestErrorCausality:
    def test_construction_error_causality_preserved(self, caps_cpu_only):
        def failing_factory(model_name, **kwargs):
            raise RuntimeError("model-load-boom")

        seg = SaTSentenceSegmenter(model_factory=failing_factory, caps=caps_cpu_only)
        with pytest.raises(SegmentationError) as exc:
            seg.split_batch(["a|b"], ["en"])
        assert exc.value.__cause__ is not None
        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_inference_error_causality_preserved(self, caps_cpu_only):
        class _BoomFacade(_ClassifierFacade):
            def split(self, texts, **kwargs):
                raise RuntimeError("infer-boom")

        def factory(model_name, **kwargs):
            return _BoomFacade(model_name, **kwargs)

        seg = SaTSentenceSegmenter(model_factory=factory, caps=caps_cpu_only)
        with pytest.raises(SegmentationError) as exc:
            seg.split_batch(["a|b"], ["en"])
        assert exc.value.__cause__ is not None
        assert isinstance(exc.value.__cause__, RuntimeError)


class TestRejectsInvalidDevice:
    def test_unknown_device_rejected_at_construction(self, caps_cpu_only):
        with pytest.raises(SegmentationError, match="device"):
            SaTSentenceSegmenter(
                model_factory=_factory(), caps=caps_cpu_only, device="gpu"
            )

    def test_blank_device_rejected_at_construction(self, caps_cpu_only):
        with pytest.raises(SegmentationError, match="device"):
            SaTSentenceSegmenter(
                model_factory=_factory(), caps=caps_cpu_only, device="  "
            )


class TestDeviceResolvedOnce:
    def test_resolved_device_is_set_after_first_batch(self, caps_cpu_only):
        seg = SaTSentenceSegmenter(model_factory=_factory(), caps=caps_cpu_only)
        assert seg.resolved_device is None
        seg.split_batch(["a|b"], ["en"])
        assert seg.resolved_device == "cpu"

    def test_second_batch_does_not_re_resolve(self, caps_cpu_only):
        seg = SaTSentenceSegmenter(model_factory=_factory(), caps=caps_cpu_only)
        seg.split_batch(["a|b"], ["en"])
        first = seg.resolved_device
        seg.split_batch(["c|d"], ["en"])
        # Re-resolution is forbidden by the cached ``resolved_device`` guard.
        assert seg.resolved_device == first
