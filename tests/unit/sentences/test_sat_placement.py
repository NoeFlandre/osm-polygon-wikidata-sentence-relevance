"""Regression tests for the wtpsplit placement boundary (the implementation hardening; the implementation placement-shape correction).

The real wtpsplit 2.2.1 wraps a Hugging Face token-classification model in
a custom ``PyTorchWrapper`` whose ``__getattr__`` delegates to the
*complete* classifier (not just its backbone):

    SaT                                        (façade, exposes ``split``)
      .model -> PyTorchWrapper                  (delegation shell)
          .model -> SubwordXLMForTokenClassification   <-- placement target
              .roberta     -> XLM-R backbone   (must NOT be touched)
              .classifier  -> nn.Linear head   (must NOT be touched)

Naively recursing through ``.model`` would land on the backbone; calling
``.to(device)`` there leaves the classifier head on CPU and risks a
mixed-device inference failure.

The placement helper MUST select the *complete classifier* owned by the
PyTorchWrapper, verify all of its parameters and buffers are on the
requested device, and refuse to operate on a wrapper shape it does not
recognize.

The contract is purely "complete ``torch.nn.Module`` at ``wrapper.model``";
the names of the backbone (``roberta`` / ``bert`` / ``xlm_roberta`` / …)
and of the classification head (``classifier`` / ``score`` / ``head`` /
…) are owned by the encoder family and are NOT preconditions. A naïve
check for ``.model`` on the inner classifier (the implementation) wrongly rejected
the real ``SubwordXLMForTokenClassification`` whose backbone is named
``.roberta``.

These tests do NOT import the real ``wtpsplit`` package so that the
``wtpsplit`` symbol never enters ``sys.modules`` during collection and
the package's optional-dependency contract stays clean. We exercise the
helper via a stand-in class (``_MockPyTorchWrapper``) and patch the
helper's class loader to return it.
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
    other attribute to it via ``__getattr__``. Real ``wtpsplit`` ships
    the same shape; we only need the structural contract here, so we
    do not import the real package.
    """

    def __init__(self, inner: nn.Module) -> None:
        self.model = inner

    def __getattr__(self, name: str) -> object:
        if name in {"model"}:
            raise AttributeError(name)
        return getattr(self.model, name)


@pytest.fixture(autouse=True)
def _patch_pyTorchWrapper_class(monkeypatch):
    """Patch the helper's PyTorchWrapper loader to return our stand-in.

    Without this, importing the real ``wtpsplit`` would be required.
    """
    monkeypatch.setattr(
        _wtpsplit_mod,
        "_load_wtpsplit_pytorch_wrapper_class",
        lambda: _MockPyTorchWrapper,
    )
    return _MockPyTorchWrapper


@pytest.fixture
def _real_loader():
    """Yield the real (un-patched)
    :func:`_wtpsplit_device._load_wtpsplit_pytorch_wrapper_class`.

    Reloads the private module so the autouse patch on the same
    attribute is bypassed for the duration of one test. Used by tests
    that exercise the real loader directly.
    """
    import importlib

    import osm_polygon_sentence_relevance.sentences._wtpsplit_device as wpd

    real_wpd = importlib.reload(wpd)
    try:
        yield real_wpd._load_wtpsplit_pytorch_wrapper_class
    finally:
        importlib.reload(wpd)


class _Backbone(nn.Module):
    """Stand-in for the XLM-R encoder body."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8)


class _Classifier(nn.Module):
    """Stand-in for ``SubwordXLMForTokenClassification``.

    Mirrors the real layout: a ``.model`` backbone and a separate
    ``.classifier`` head. Both live on this object; both must move
    together when the classifier is placed on a device.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()
        self.classifier = nn.Linear(8, 2)
        # ``wtpsplit.extract.PyTorchWrapper`` reads ``model.config`` in
        # its ``__init__``; provide a dummy config attribute for the fake.
        self.config = type("Cfg", (), {"model_type": "xlm-roberta"})()


class _CompleteSaT:
    """Faithful mimic of ``wtpsplit.SaT``: façade + real PyTorchWrapper + classifier."""

    def __init__(self, model_name: str, **kwargs: object) -> None:
        self.model_name = model_name
        self.kwargs = dict(kwargs)
        self.classifier = _Classifier()
        self.model = _MockPyTorchWrapper(self.classifier)

    def split(self, texts, **kwargs):
        return [[t.split("|")] for t in texts]


def _factory():
    def f(model_name: str, **kwargs: object) -> object:
        return _CompleteSaT(model_name, **kwargs)

    return f


class _Caps:
    def __init__(self, *, cuda: bool = False, mps: bool = False) -> None:
        self.cuda_available = cuda
        self.mps_available = mps


@pytest.fixture
def caps_cpu_only() -> _Caps:
    return _Caps(cuda=False, mps=False)


class TestClassifierIsMovedAtomically:
    def test_classifier_not_backbone_is_placed(self, caps_cpu_only):
        """The placement helper must call ``.to(device)`` on the
        *classifier* (the PyTorchWrapper-owned module), not on its
        inner backbone.
        """
        seg = SaTSentenceSegmenter(model_factory=_factory(), caps=caps_cpu_only)
        seg.split_batch(["a|b"], ["en"])
        sat = seg._model  # type: ignore[attr-defined]
        classifier = sat.classifier  # type: ignore[attr-defined]
        assert isinstance(classifier, _Classifier)
        for tensor in classifier.parameters():
            assert tensor.device.type == "cpu"
        for tensor in classifier.buffers():
            assert tensor.device.type == "cpu"

    def test_classifier_is_the_placement_target_not_backbone(self, caps_cpu_only):
        """Sanity: the placement target is identified structurally
        (``wtpsplit.extract.PyTorchWrapper``'s ``.model`` attribute). It
        must NOT be discovered by descending through ``.model``
        recursively (which would land on ``SubwordXLMRobertaModel``).
        """

        captured: dict[str, object] = {}

        class _Spy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Linear(2, 2)
                self.classifier = nn.Linear(2, 2)
                self.config = type("Cfg", (), {})()

            def to(self, device):  # type: ignore[override]
                captured["target"] = self
                return self

        classifier = _Spy()

        class _Facade:
            def __init__(self) -> None:
                self.model = _MockPyTorchWrapper(classifier)

            def split(self, texts, **kwargs):
                return [[t] for t in texts]

        _wtpsplit_mod.place_classifier(_Facade(), "cpu")

        assert captured.get("target") is classifier
        assert captured.get("target") is not classifier.model


class TestClassifierPlacementMismatchedShapeFails:
    def test_facade_without_wrapper_model_rejected(self, caps_cpu_only):
        """A SaT-like façade whose ``.model`` is not a
        ``wtpsplit.extract.PyTorchWrapper`` must be rejected when an
        accelerator is requested; on CPU the placement is a no-op
        (legacy CPU-only test-double path).
        """

        class _BareFacade:
            def __init__(self) -> None:
                # No ``.model`` attribute at all.
                self.classifier = _Classifier()

            def split(self, texts, **kwargs):
                return [[t] for t in texts]

        # CPU path: silently no-ops (legacy test-double path).
        result = _wtpsplit_mod.place_classifier(_BareFacade(), "cpu")
        assert result is not None
        # CUDA path: must raise; never silently degrade.
        with pytest.raises(SegmentationError, match="torch.nn.Module"):
            _wtpsplit_mod.place_classifier(_BareFacade(), "cuda")
        # MPS path: must raise too.
        with pytest.raises(SegmentationError, match="torch.nn.Module"):
            _wtpsplit_mod.place_classifier(_BareFacade(), "mps")

    def test_wrapper_with_torch_inner_accepted_regardless_of_submodule_names(
        self, caps_cpu_only
    ):
        """A wrapper whose inner is *any* complete ``torch.nn.Module``
        (regardless of whether it carries ``.classifier`` / ``.score``
        or a backbone named ``.model``) is now accepted on CPU; the
        new placement contract is purely "complete ``torch.nn.Module``
        at ``wrapper.model``". The classification head's name is
        version-fragile and is no longer a precondition.
        """

        class _BackboneOnly(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # No ``.classifier`` head, no ``.model`` backbone.
                self.linear = nn.Linear(4, 4)
                self.config = type("Cfg", (), {})()

        class _Facade:
            def __init__(self) -> None:
                self.model = _MockPyTorchWrapper(_BackboneOnly())

            def split(self, texts, **kwargs):
                return [[t] for t in texts]

        # CPU legacy path: silent no-op (the inner is a torch module
        # but has no parameters that move on CPU; placement is a no-op
        # in practice).
        result = _wtpsplit_mod.place_classifier(_Facade(), "cpu")
        assert result is not None


class TestVerificationReadsClassifierNotFaçade:
    def test_verification_reads_classifier_device(self, caps_cpu_only, monkeypatch):
        """The verification step must read every parameter/buffer
        device of the *complete classifier*. A classifier whose ``.to``
        silently no-ops must be detected and rejected.

        We simulate the no-op by monkey-patching the verification
        helper to report a device that does not match the requested
        one; the placement helper must observe the mismatch and raise.
        """

        def _fake_observed(_classifier: object) -> str:
            return "xpu"

        monkeypatch.setattr(
            _wtpsplit_mod, "_classifier_observed_device", _fake_observed
        )

        class _Facade:
            def __init__(self) -> None:
                self.classifier = _Classifier()
                self.model = _MockPyTorchWrapper(self.classifier)

            def split(self, texts, **kwargs):
                return [[t] for t in texts]

        with pytest.raises(SegmentationError) as exc:
            _wtpsplit_mod.place_classifier(_Facade(), "cpu")
        msg = str(exc.value)
        assert "cpu" in msg
        assert "xpu" in msg


class TestPublicSegmenterDoesNotMisplace:
    def test_segmenter_placement_targets_complete_classifier(self, caps_cpu_only):
        """End-to-end: the segmenter's first ``split_batch`` must place
        the complete classifier (the PyTorchWrapper-owned module) on the
        requested device.
        """
        seg = SaTSentenceSegmenter(model_factory=_factory(), caps=caps_cpu_only)
        seg.split_batch(["a|b"], ["en"])
        sat = seg._model  # type: ignore[attr-defined]
        classifier = sat.classifier  # type: ignore[attr-defined]
        for tensor in classifier.parameters():
            assert tensor.device.type == "cpu"

    def test_second_batch_reuses_resolved_device_without_re_probe(self, caps_cpu_only):
        """The device must be resolved exactly once; subsequent batches
        reuse both the resolved device and the placed model.
        """
        seg = SaTSentenceSegmenter(model_factory=_factory(), caps=caps_cpu_only)
        seg.split_batch(["a|b"], ["en"])
        first = seg.resolved_device
        sat_first = seg._model  # type: ignore[attr-defined]
        seg.split_batch(["c|d"], ["en"])
        assert seg.resolved_device == first
        assert seg._model is sat_first  # type: ignore[attr-defined]

    def test_explicit_cuda_rejected_for_non_torch_facade(self):
        """Explicit CUDA on a non-torch façade (no ``.to``) must fail."""

        class _BadFacade:
            def __init__(self) -> None:
                self.model = None

            def split(self, texts, **kwargs):
                return [[t] for t in texts]

        def factory(model_name, **kwargs):
            return _BadFacade()

        caps = _Caps(cuda=True, mps=False)
        seg = SaTSentenceSegmenter(model_factory=factory, caps=caps, device="cuda")
        with pytest.raises(SegmentationError):
            seg.split_batch(["a|b"], ["en"])

    def test_explicit_cuda_rejected_when_wrapper_inner_not_torch(self):
        """Explicit CUDA on a wrapper whose inner is not a torch module
        must fail with a clear SegmentationError.
        """

        class _InnerNotTorch:
            # No ``.to`` attribute — not a torch module.
            pass

        class _Facade:
            def __init__(self) -> None:
                self.model = _MockPyTorchWrapper(_InnerNotTorch())  # type: ignore[arg-type]

            def split(self, texts, **kwargs):
                return [[t] for t in texts]

        def factory(model_name, **kwargs):
            return _Facade()

        caps = _Caps(cuda=True, mps=False)
        seg = SaTSentenceSegmenter(model_factory=factory, caps=caps, device="cuda")
        with pytest.raises(SegmentationError, match="torch.nn.Module"):
            seg.split_batch(["a|b"], ["en"])

    def test_generic_placement_error_wrapped_with_model_name(self):
        """Any non-SegmentationError exception from ``_do_place`` is
        wrapped with the model name and the requested device.
        """

        class _BoomInner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Linear(2, 2)
                self.classifier = nn.Linear(2, 2)
                self.config = type("Cfg", (), {})()

            def to(self, device):  # type: ignore[override]
                raise RuntimeError("placement-boom")

        class _BoomFacade:
            def __init__(self) -> None:
                self.classifier = _BoomInner()
                self.model = _MockPyTorchWrapper(self.classifier)

            def split(self, texts, **kwargs):
                return [[t] for t in texts]

        def factory(model_name, **kwargs):
            return _BoomFacade()

        caps = _Caps(cuda=False, mps=False)
        seg = SaTSentenceSegmenter(model_factory=factory, caps=caps, device="cpu")
        with pytest.raises(SegmentationError) as exc:
            seg.split_batch(["a|b"], ["en"])
        msg = str(exc.value)
        assert "sat-3l-sm" in msg


class TestAutoDeviceResolutionPreservesAccelerator:
    """Regression: when ``device="auto"`` resolves to CUDA or MPS, the
    *complete classifier* owned by the ``PyTorchWrapper`` must be moved
    onto that accelerator -- never silently downgraded to CPU.

    The real wtpsplit ``SaT.model`` is a ``PyTorchWrapper``, not a
    ``torch.nn.Module``; a naive ``_is_torch_module(saT.model)`` returns
    ``False``, which previously caused the segmenter to silently rewrite
    the resolved device to ``"cpu"``. On Grid'5000 that would route an
    expensive SaT run onto the compute-node CPU. On the Mac it would
    silently compute on the local CPU when an accelerator is in fact
    available.

    These tests inject a faithful facade (real-shape ``PyTorchWrapper``
    -> real-shape complete ``nn.Module`` classifier) and a spy on the
    placement adapter so we can record exactly which device is passed
    into it and against which classifier. The host itself need not have
    real CUDA / MPS available -- the placement adapter's call is
    intercepted, not executed.
    """

    def _caps_cuda(self) -> _Caps:
        return _Caps(cuda=True, mps=False)

    def _caps_mps(self) -> _Caps:
        return _Caps(cuda=False, mps=True)

    @staticmethod
    def _spy_place(monkeypatch) -> dict[str, object]:
        """Patch ``_wtpsplit_device.place_classifier`` so it records
        the (classifier, device) call instead of moving real tensors.
        """
        captured: dict[str, object] = {"calls": []}

        def _fake_place(model: object, device: str) -> object:
            classifier = model.model.model  # type: ignore[attr-defined]
            captured["calls"].append((classifier, device))
            return classifier

        monkeypatch.setattr(_wtpsplit_mod, "place_classifier", _fake_place)
        return captured

    def test_auto_cuda_resolves_cuda_and_calls_classifier_to_once(self, monkeypatch):
        captured = self._spy_place(monkeypatch)

        class _Facade:
            def __init__(self) -> None:
                self.classifier = _Classifier()
                self.model = _MockPyTorchWrapper(self.classifier)

            def split(self, texts, **kwargs):
                return [[t.split("|")] for t in texts]

        def factory(model_name, **kwargs):
            return _Facade()

        seg = SaTSentenceSegmenter(
            model_factory=factory, caps=self._caps_cuda(), device="auto"
        )
        seg.split_batch(["a|b"], ["en"])

        assert seg.resolved_device == "cuda", (
            f"auto should have resolved to cuda but got {seg.resolved_device!r}"
        )
        assert len(captured["calls"]) == 1
        classifier_arg, device_arg = captured["calls"][0]
        assert device_arg == "cuda"
        assert isinstance(classifier_arg, _Classifier)

    def test_auto_mps_resolves_mps_and_calls_classifier_to_once(self, monkeypatch):
        captured = self._spy_place(monkeypatch)

        class _Facade:
            def __init__(self) -> None:
                self.classifier = _Classifier()
                self.model = _MockPyTorchWrapper(self.classifier)

            def split(self, texts, **kwargs):
                return [[t.split("|")] for t in texts]

        def factory(model_name, **kwargs):
            return _Facade()

        seg = SaTSentenceSegmenter(
            model_factory=factory, caps=self._caps_mps(), device="auto"
        )
        seg.split_batch(["a|b"], ["en"])

        assert seg.resolved_device == "mps"
        assert len(captured["calls"]) == 1
        classifier_arg, device_arg = captured["calls"][0]
        assert device_arg == "mps"
        assert isinstance(classifier_arg, _Classifier)

    def test_auto_cuda_does_not_fallback_to_cpu(self, monkeypatch):
        self._spy_place(monkeypatch)

        class _Facade:
            def __init__(self) -> None:
                self.classifier = _Classifier()
                self.model = _MockPyTorchWrapper(self.classifier)

            def split(self, texts, **kwargs):
                return [[t.split("|")] for t in texts]

        def factory(model_name, **kwargs):
            return _Facade()

        seg = SaTSentenceSegmenter(
            model_factory=factory, caps=self._caps_cuda(), device="auto"
        )
        seg.split_batch(["a|b"], ["en"])
        assert seg.resolved_device == "cuda"

    def test_auto_mps_does_not_fallback_to_cpu(self, monkeypatch):
        self._spy_place(monkeypatch)

        class _Facade:
            def __init__(self) -> None:
                self.classifier = _Classifier()
                self.model = _MockPyTorchWrapper(self.classifier)

            def split(self, texts, **kwargs):
                return [[t.split("|")] for t in texts]

        def factory(model_name, **kwargs):
            return _Facade()

        seg = SaTSentenceSegmenter(
            model_factory=factory, caps=self._caps_mps(), device="auto"
        )
        seg.split_batch(["a|b"], ["en"])
        assert seg.resolved_device == "mps"


class TestAcceleratorPlacementFailurePreventsInference:
    """GPU-only execution safety contract.

    When capabilities report CUDA (or MPS) available AND the request is
    either explicit ``"cuda"`` / ``"mps"`` or auto-resolves to that
    backend, a placement or verification failure must:

    1. raise :class:`SegmentationError`;
    2. never call ``model.split`` (inference must be skipped);
    3. never silently fall back to CPU.

    This is the guard that prevents a future Grid'5000 job from silently
    computing on CPU after an accelerator was selected.
    """

    def test_cuda_placement_failure_raises_and_skips_inference(self):
        """A classifier whose ``.to("cuda")`` silently no-ops (because
        the verifier sees the original device) must surface as
        :class:`SegmentationError`, never call ``model.split``, and must
        not resolve to CPU.
        """

        inference_calls: list[object] = []

        class _Classifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = _Backbone()
                self.classifier = nn.Linear(8, 2)
                self.config = type("Cfg", (), {})()

            def to(self, device):  # type: ignore[override]
                # Silently no-op (simulates a real GPU whose placement
                # fails for some reason — e.g. OOM, driver mismatch).
                return self

        class _Facade:
            def __init__(self) -> None:
                self.classifier = _Classifier()
                self.model = _MockPyTorchWrapper(self.classifier)

            def split(self, texts, **kwargs):
                inference_calls.append(texts)
                return [[t.split("|")] for t in texts]

        def factory(model_name, **kwargs):
            return _Facade()

        seg = SaTSentenceSegmenter(
            model_factory=factory, caps=_Caps(cuda=True, mps=False), device="cuda"
        )
        with pytest.raises(SegmentationError):
            seg.split_batch(["a|b"], ["en"])
        assert inference_calls == [], (
            "model.split must not be called when placement failed"
        )
        assert seg.resolved_device is None, (
            f"resolved_device must remain unset on placement failure; "
            f"got {seg.resolved_device!r}"
        )

    def test_auto_cuda_placement_failure_raises_and_skips_inference(self):
        """Same as above but with ``device="auto"`` and CUDA-capable
        caps. The accelerator must still be honored; no CPU fallback.
        """
        inference_calls: list[object] = []

        class _Classifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = _Backbone()
                self.classifier = nn.Linear(8, 2)
                self.config = type("Cfg", (), {})()

            def to(self, device):  # type: ignore[override]
                # Silently no-op.
                return self

        class _Facade:
            def __init__(self) -> None:
                self.classifier = _Classifier()
                self.model = _MockPyTorchWrapper(self.classifier)

            def split(self, texts, **kwargs):
                inference_calls.append(texts)
                return [[t.split("|")] for t in texts]

        def factory(model_name, **kwargs):
            return _Facade()

        seg = SaTSentenceSegmenter(
            model_factory=factory, caps=_Caps(cuda=True, mps=False), device="auto"
        )
        with pytest.raises(SegmentationError):
            seg.split_batch(["a|b"], ["en"])
        assert inference_calls == []
        assert seg.resolved_device is None

    def test_auto_mps_placement_failure_raises_and_skips_inference(self):
        inference_calls: list[object] = []

        class _Classifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = _Backbone()
                self.classifier = nn.Linear(8, 2)
                self.config = type("Cfg", (), {})()

            def to(self, device):  # type: ignore[override]
                return self

        class _Facade:
            def __init__(self) -> None:
                self.classifier = _Classifier()
                self.model = _MockPyTorchWrapper(self.classifier)

            def split(self, texts, **kwargs):
                inference_calls.append(texts)
                return [[t.split("|")] for t in texts]

        def factory(model_name, **kwargs):
            return _Facade()

        seg = SaTSentenceSegmenter(
            model_factory=factory, caps=_Caps(cuda=False, mps=True), device="auto"
        )
        with pytest.raises(SegmentationError):
            seg.split_batch(["a|b"], ["en"])
        assert inference_calls == []
        assert seg.resolved_device is None

    def test_placement_failure_does_not_silently_use_cpu(self):
        """The resolved device must never be silently rewritten to
        ``"cpu"`` after an accelerator placement failure. The model is
        left un-placed and the user is told the placement failed.
        """

        class _Classifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = _Backbone()
                self.classifier = nn.Linear(8, 2)
                self.config = type("Cfg", (), {})()

            def to(self, device):  # type: ignore[override]
                return self  # no-op

        class _Facade:
            def __init__(self) -> None:
                self.classifier = _Classifier()
                self.model = _MockPyTorchWrapper(self.classifier)

            def split(self, texts, **kwargs):
                return [[t.split("|")] for t in texts]

        def factory(model_name, **kwargs):
            return _Facade()

        seg = SaTSentenceSegmenter(
            model_factory=factory, caps=_Caps(cuda=True, mps=False), device="cuda"
        )
        with pytest.raises(SegmentationError):
            seg.split_batch(["a|b"], ["en"])
        # ``resolved_device`` MUST NOT be "cpu" after a failed CUDA
        # placement. The model is not placed; a future call would
        # re-attempt placement.
        assert seg.resolved_device != "cpu"


class TestLazyImportGuards:
    """The lazy ``wtpsplit`` import must guard against missing or
    wrong-version dependencies. These tests patch ``wtpsplit`` in
    ``sys.modules`` so the real package is not required.
    """

    def test_missing_wtpsplit_raises_actionable_error(self, monkeypatch):
        from osm_polygon_sentence_relevance.sentences import sat as sat_mod

        # Patch the lazy importer to raise ImportError as if wtpsplit
        # were not installed.
        def _no_wtpsplit():
            raise ImportError("No module named 'wtpsplit'")

        monkeypatch.setattr(sat_mod, "_lazy_import_sat", _no_wtpsplit)
        seg = SaTSentenceSegmenter()
        with pytest.raises(SegmentationError) as exc:
            seg.split_batch(["a|b"], ["en"])
        # The wrapped error preserves the underlying ImportError as
        # ``__cause__`` so callers can introspect it.
        assert isinstance(exc.value.__cause__, ImportError)
        assert "wtpsplit" in str(exc.value.__cause__)

    def test_unsupported_wtpsplit_version_raises(self, monkeypatch):
        # The version check is in ``_lazy_import_sat`` itself. We
        # exercise it by patching ``wtpsplit`` in ``sys.modules`` to
        # a fake with the wrong version, then calling the real
        # ``_lazy_import_sat``.
        import sys

        from osm_polygon_sentence_relevance.sentences import sat as sat_mod

        class _BadVersionMod:
            __version__ = "9.9.9"

        real_wtpsplit = sys.modules.get("wtpsplit")
        sys.modules["wtpsplit"] = _BadVersionMod()
        try:
            with pytest.raises(SegmentationError, match="unsupported"):
                sat_mod._lazy_import_sat()
        finally:
            if real_wtpsplit is None:
                sys.modules.pop("wtpsplit", None)
            else:
                sys.modules["wtpsplit"] = real_wtpsplit

    def test_supported_wtpsplit_version_returns_class(self, monkeypatch):
        """With ``wtpsplit.__version__ == "2.2.1"`` and a fake ``SaT``
        attribute, ``_lazy_import_sat`` returns the class without
        raising.
        """
        from osm_polygon_sentence_relevance.sentences import sat as sat_mod

        class _FakeSaT:
            pass

        class _GoodVersionMod:
            __version__ = "2.2.1"
            SaT = _FakeSaT

        import sys

        real_wtpsplit = sys.modules.get("wtpsplit")
        sys.modules["wtpsplit"] = _GoodVersionMod()
        try:
            result = sat_mod._lazy_import_sat()
            assert result is _FakeSaT
        finally:
            if real_wtpsplit is None:
                sys.modules.pop("wtpsplit", None)
            else:
                sys.modules["wtpsplit"] = real_wtpsplit

    def test_missing_wtpsplit_class_loader_raises(self, monkeypatch):
        """If ``wtpsplit`` is importable but lacks ``extract.PyTorchWrapper``,
        the class loader raises a clear SegmentationError.

        We un-patch the autouse ``_wtpsplit_mod._load_wtpsplit_pytorch_wrapper_class`` patch
        for this single test by calling a non-autouse-influenced
        variant of the loader directly via ``importlib.reload``.
        """

        class _NoExtractMod:
            __version__ = "2.2.1"

        import sys

        real_wtpsplit = sys.modules.get("wtpsplit")
        real_wtpsplit_extract = sys.modules.get("wtpsplit.extract")
        sys.modules["wtpsplit"] = _NoExtractMod()
        sys.modules.pop("wtpsplit.extract", None)
        # Undo all autouse + per-test patches for the duration of this
        # test. ``monkeypatch.undo()`` restores the patched attribute
        # values; the autouse fixture's monkeypatch will be restored
        # automatically when the test exits.
        monkeypatch.undo()
        try:
            with pytest.raises(SegmentationError, match="PyTorchWrapper"):
                _wtpsplit_mod._load_wtpsplit_pytorch_wrapper_class()
        finally:
            sys.modules.pop("wtpsplit", None)
            if real_wtpsplit is not None:
                sys.modules["wtpsplit"] = real_wtpsplit
            if real_wtpsplit_extract is not None:
                sys.modules["wtpsplit.extract"] = real_wtpsplit_extract

    def test_supported_loader_returns_pytorch_wrapper_class(self, monkeypatch):
        """With wtpsplit installed (the segmentation extra), the loader
        returns the real ``PyTorchWrapper`` class. We un-patch the
        autouse fixture to use the real loader.
        """
        pytest.importorskip("wtpsplit")

        # Un-patch to use the real loader.
        monkeypatch.undo()
        cls = _wtpsplit_mod._load_wtpsplit_pytorch_wrapper_class()
        from wtpsplit.extract import PyTorchWrapper  # real class

        assert cls is PyTorchWrapper


class TestExtractClassifierShape:
    """Targeted coverage for ``_wtpsplit_mod._extract_classifier`` error paths."""

    def test_facade_without_model_attribute(self, caps_cpu_only):
        class _NoModel:
            def split(self, texts, **kwargs):
                return [[t] for t in texts]

        with pytest.raises(SegmentationError, match="façade"):
            _wtpsplit_mod._extract_classifier(_NoModel())

    def test_wrapper_model_is_none(self, caps_cpu_only):
        class _WrapperWithNone:
            model = None

        class _Facade:
            def __init__(self):
                self.model = _WrapperWithNone()

            def split(self, texts, **kwargs):
                return [[t] for t in texts]

        with pytest.raises(SegmentationError, match="wrapper"):
            _wtpsplit_mod._extract_classifier(_Facade())

    def test_classifier_missing_inner_rejected(self, caps_cpu_only):
        """A wrapper whose inner is missing ``.model`` must be
        rejected at ``_extract_classifier`` time. We retain the
        ``wrapper.model is None`` rejection path because it is a
        legitimate sanity check (the wtpsplit wrapper must own a
        model object).
        """

        class _WrapperWithNoneModel:
            model = None

        class _Facade:
            def __init__(self):
                self.model = _WrapperWithNoneModel()

            def split(self, texts, **kwargs):
                return [[t] for t in texts]

        with pytest.raises(SegmentationError, match="wrapper"):
            _wtpsplit_mod._extract_classifier(_Facade())


class TestObservedDeviceEdgeCases:
    """Targeted coverage for ``_wtpsplit_mod._classifier_observed_device``
    edge cases: missing device attribute on a parameter/buffer,
    partial placement, empty classifier.
    """

    def test_parameter_missing_device_raises(self):
        class _ParamNoDev:
            pass

        class _Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(2, 2)
                self.classifier = nn.Linear(2, 2)
                self.config = type("Cfg", (), {})()

            def parameters(self):  # type: ignore[override]
                return iter([_ParamNoDev()])

        c = _Classifier()
        with pytest.raises(SegmentationError, match="no device"):
            _wtpsplit_mod._classifier_observed_device(c)

    def test_buffer_missing_device_raises(self):
        class _ParamWithDev:
            device = type("D", (), {"type": "cpu"})()

        class _BufferNoDev:
            pass

        class _Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(2, 2)
                self.classifier = nn.Linear(2, 2)
                self.config = type("Cfg", (), {})()

            def parameters(self):  # type: ignore[override]
                return iter([_ParamWithDev()])

            def buffers(self):  # type: ignore[override]
                return iter([_BufferNoDev()])

        c = _Classifier()
        with pytest.raises(SegmentationError, match="no device"):
            _wtpsplit_mod._classifier_observed_device(c)

    def test_empty_classifier_raises(self):
        class _Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Cfg", (), {})()

            def parameters(self):  # type: ignore[override]
                return iter([])

            def buffers(self):  # type: ignore[override]
                return iter([])

        c = _Classifier()
        with pytest.raises(SegmentationError, match="no parameters"):
            _wtpsplit_mod._classifier_observed_device(c)

    def test_partial_placement_raises(self):
        class _P1:
            device = type("D", (), {"type": "cpu"})()

        class _P2:
            device = type("D", (), {"type": "cuda"})()

        class _Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Cfg", (), {})()

            def parameters(self):  # type: ignore[override]
                return iter([_P1(), _P2()])

            def buffers(self):  # type: ignore[override]
                return iter([])

        c = _Classifier()
        with pytest.raises(SegmentationError, match="multiple devices"):
            _wtpsplit_mod._classifier_observed_device(c)


class TestHasSupportedShape:
    def test_unsupported_shape_returns_false(self):
        class _Facade:
            model = object()  # not a PyTorchWrapper

        assert _wtpsplit_mod.has_supported_shape(_Facade()) is False

    def test_supported_shape_returns_true(self, caps_cpu_only):
        class _Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(2, 2)
                self.classifier = nn.Linear(2, 2)
                self.config = type("Cfg", (), {})()

        class _Facade:
            def __init__(self):
                self.model = _MockPyTorchWrapper(_Classifier())

        assert _wtpsplit_mod.has_supported_shape(_Facade()) is True


class TestFaithfulWtpsplitShape:
    """Faithful regression: real wtpsplit classifier uses a registered
    backbone attribute that is NOT ``.model``.

    The real ``SubwordXLMForTokenClassification`` exposes its backbone
    under the registered-submodule name chosen by the underlying
    encoder: ``.roberta`` for XLM-RoBERTa backbones, ``.xlm_roberta`` /
    ``.bert`` / ``.deberta`` for other encoder families. A complete
    classifier may therefore call its backbone ``.roberta`` (or any
    other name) and have no ``.model`` attribute of its own.

    The placement adapter MUST:

    1. Accept a wrapper whose inner classifier is a complete
       ``torch.nn.Module`` with a registered backbone under any name
       (``.roberta`` here).
    2. Place the *complete classifier* (i.e. ``wrapper.model``)
       exactly once -- never descend into the backbone.
    3. Verify every recursively-registered parameter and buffer lives
       on the requested device after ``.to(device)`` returns.
    4. Refuse a wrapper whose inner is not a ``torch.nn.Module``.
    5. Refuse a façade whose ``.model`` is not a
       ``PyTorchWrapper``.
    6. Refuse partial / no-op placement.
    7. Refuse mixed-device parameters / buffers.
    8. Treat ``has_supported_shape`` and ``_extract_classifier`` as
       agreeing on the same structural contract.

    These tests do NOT inspect or rely on the internal backbone name;
    the contract is purely "complete ``torch.nn.Module`` at
    ``wrapper.model``".
    """

    def _make_real_shape_classifier(self) -> nn.Module:
        """Build a complete ``nn.Module`` whose backbone is named
        ``.roberta`` (as in the real ``SubwordXLMForTokenClassification``).
        """

        class _RealBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dense = nn.Linear(8, 8)

        class _RealClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Registered backbone under a *non*-``.model`` name.
                self.roberta = _RealBackbone()
                # No ``.classifier`` / ``.score`` head required.
                self.head = nn.Linear(8, 2)
                self.config = type("Cfg", (), {"model_type": "xlm-roberta"})()

        return _RealClassifier()

    def _make_facade(self, inner: nn.Module) -> object:
        class _Facade:
            def __init__(self) -> None:
                self.model = _MockPyTorchWrapper(inner)

            def split(self, texts, **kwargs):
                return [[t] for t in texts]

        return _Facade()

    # ------------------------------------------------------------------
    # 1) Faithful regression: real-shape classifier with `.roberta`
    # ------------------------------------------------------------------

    def test_real_shape_with_roberta_backbone_is_accepted(self, caps_cpu_only):
        """A real-shape ``SubwordXLMForTokenClassification`` (backbone
        ``.roberta``, no ``.model`` of its own) must be accepted and
        placed as a unit. With the old backbone-name assertion, the
        helper raised ``"no backbone attribute (`.model`)"`` and the
        smoke failed.
        """
        classifier = self._make_real_shape_classifier()
        facade = self._make_facade(classifier)

        # Sanity: the classifier indeed has no ``.model`` attribute.
        assert not hasattr(classifier, "model")
        # Sanity: it has a registered backbone under ``.roberta``.
        assert isinstance(classifier.roberta, nn.Module)

        placed = _wtpsplit_mod.place_classifier(facade, "cpu")
        assert placed is classifier

    def test_real_shape_classifier_not_backbone_is_placed(self, caps_cpu_only):
        """The placement helper must place the *complete classifier*
        (``wrapper.model``) and never descend into ``.roberta``. We
        spy on the classifier's ``.to()`` to record exactly which
        object it was called on.
        """
        captured: dict[str, object] = {}

        classifier = self._make_real_shape_classifier()
        original_to = classifier.to

        def _spy_to(device):  # type: ignore[no-untyped-def]
            captured["target"] = classifier
            captured["device"] = device
            return original_to(device)

        classifier.to = _spy_to  # type: ignore[method-assign]
        facade = self._make_facade(classifier)
        _wtpsplit_mod.place_classifier(facade, "cpu")

        assert captured.get("target") is classifier
        assert captured.get("target") is not classifier.roberta
        assert captured.get("device") == "cpu"

    def test_real_shape_recursive_parameters_all_on_requested_device(
        self, caps_cpu_only
    ):
        """Every registered parameter and buffer of the complete
        classifier (including the ones inside ``.roberta`` and the
        head) must be on the requested device after placement.
        """
        classifier = self._make_real_shape_classifier()
        facade = self._make_facade(classifier)
        _wtpsplit_mod.place_classifier(facade, "cpu")

        # Walk every parameter recursively.
        for name, tensor in classifier.named_parameters():
            assert tensor.device.type == "cpu", (
                f"parameter {name!r} device type {tensor.device.type!r} != 'cpu'"
            )
        for name, tensor in classifier.named_buffers():
            assert tensor.device.type == "cpu", (
                f"buffer {name!r} device type {tensor.device.type!r} != 'cpu'"
            )

    # ------------------------------------------------------------------
    # 2) Preserved rejections: unrecognised wrapper shape
    # ------------------------------------------------------------------

    def test_facade_without_real_wrapper_rejected(self, caps_cpu_only):
        """A façade whose ``.model`` is not a ``PyTorchWrapper`` is
        still rejected. CPU path: silently no-ops (legacy test-double
        path). Accelerator path: must raise.
        """

        class _Facade:
            model = object()  # not a PyTorchWrapper

            def split(self, texts, **kwargs):
                return [[t] for t in texts]

        # CPU legacy path: silent no-op.
        result = _wtpsplit_mod.place_classifier(_Facade(), "cpu")
        assert result is not None

        # CUDA accelerator: must raise, never silently degrade.
        with pytest.raises(SegmentationError):
            _wtpsplit_mod.place_classifier(_Facade(), "cuda")

        # MPS accelerator: must raise.
        with pytest.raises(SegmentationError):
            _wtpsplit_mod.place_classifier(_Facade(), "mps")

    def test_wrapper_with_non_torch_inner_rejected(self, caps_cpu_only):
        """If ``wrapper.model`` is not a ``torch.nn.Module``, the
        helper must refuse it. CPU path: legacy no-op. Accelerator:
        raise.
        """

        class _InnerNotTorch:
            pass

        class _Facade:
            def __init__(self) -> None:
                self.model = _MockPyTorchWrapper(_InnerNotTorch())  # type: ignore[arg-type]

            def split(self, texts, **kwargs):
                return [[t] for t in texts]

        # CPU legacy: silent no-op.
        _wtpsplit_mod.place_classifier(_Facade(), "cpu")
        # CUDA accelerator: raise with a torch-module hint.
        with pytest.raises(SegmentationError, match="torch.nn.Module"):
            _wtpsplit_mod.place_classifier(_Facade(), "cuda")

    # ------------------------------------------------------------------
    # 3) Preserved rejections: partial / no-op / mixed-device placement
    # ------------------------------------------------------------------

    def test_partial_placement_silently_no_op_raises_on_cuda(self):
        """A classifier whose ``.to("cuda")`` silently no-ops is
        detected by the verification step (which reads every
        parameter/buffer device) and surfaced as
        :class:`SegmentationError`. Inference must not be called.
        """
        inference_calls: list[object] = []

        class _NoOpClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.roberta = nn.Linear(2, 2)
                self.head = nn.Linear(2, 2)
                self.config = type("Cfg", (), {})()

            def to(self, device):  # type: ignore[override]
                return self

        class _Facade:
            def __init__(self) -> None:
                self.model = _MockPyTorchWrapper(_NoOpClassifier())

            def split(self, texts, **kwargs):
                inference_calls.append(texts)
                return [[t] for t in texts]

        seg = SaTSentenceSegmenter(
            model_factory=lambda *_a, **_kw: _Facade(),
            caps=_Caps(cuda=True, mps=False),
            device="cuda",
        )
        with pytest.raises(SegmentationError):
            seg.split_batch(["a|b"], ["en"])
        assert inference_calls == [], (
            "model.split must not be called when placement failed"
        )
        assert seg.resolved_device is None

    def test_mixed_device_parameters_raises(self, monkeypatch):
        """A classifier whose parameters straddle two device types
        after ``.to`` is detected by the verification step.
        """

        class _CpuParam:
            device = type("D", (), {"type": "cpu"})()

        class _CudaParam:
            device = type("D", (), {"type": "cuda"})()

        class _MixedClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.roberta = nn.Linear(2, 2)
                self.head = nn.Linear(2, 2)
                self.config = type("Cfg", (), {})()

            def to(self, device):  # type: ignore[override]
                return self

            def parameters(self):  # type: ignore[override]
                return iter([_CpuParam(), _CudaParam()])

        class _Facade:
            def __init__(self) -> None:
                self.model = _MockPyTorchWrapper(_MixedClassifier())

            def split(self, texts, **kwargs):
                return [[t] for t in texts]

        with pytest.raises(SegmentationError, match="multiple devices"):
            _wtpsplit_mod.place_classifier(_Facade(), "cpu")

    # ------------------------------------------------------------------
    # 4) ``has_supported_shape`` and ``_extract_classifier`` agree
    # ------------------------------------------------------------------

    def test_has_supported_shape_and_extract_classifier_agree_real(self):
        """For a real-shape ``SubwordXLMForTokenClassification``
        (backbone ``.roberta``, no ``.model``), both
        ``has_supported_shape`` and ``_extract_classifier`` must
        accept the same wrapper.
        """
        classifier = self._make_real_shape_classifier()
        facade = self._make_facade(classifier)
        assert _wtpsplit_mod.has_supported_shape(facade) is True
        extracted = _wtpsplit_mod._extract_classifier(facade)
        assert extracted is classifier

    def test_has_supported_shape_false_for_non_torch_inner(self):
        """A façade whose ``wrapper.model`` is not a ``torch.nn.Module``
        must be rejected by *both* helpers.
        """

        class _InnerNotTorch:
            pass

        class _Facade:
            def __init__(self) -> None:
                self.model = _MockPyTorchWrapper(_InnerNotTorch())  # type: ignore[arg-type]

        assert _wtpsplit_mod.has_supported_shape(_Facade()) is False
        with pytest.raises(SegmentationError, match="torch.nn.Module"):
            _wtpsplit_mod._extract_classifier(_Facade())


class TestIsTorchModule:
    """Targeted coverage for ``_is_torch_module``."""

    def test_returns_true_for_real_module(self):
        from osm_polygon_sentence_relevance.sentences import sat as sat_mod

        m = nn.Linear(2, 2)
        assert sat_mod._is_torch_module(m) is True

    def test_returns_false_for_plain_object(self):
        from osm_polygon_sentence_relevance.sentences import sat as sat_mod

        assert sat_mod._is_torch_module(object()) is False

    def test_returns_false_when_torch_missing(self, monkeypatch):
        """When ``torch`` cannot be imported, the helper must return
        ``False`` rather than raising.
        """
        import builtins

        from osm_polygon_sentence_relevance.sentences import sat as sat_mod

        real_import = builtins.__import__

        def _blocking(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking)
        assert sat_mod._is_torch_module(object()) is False


class TestDeclaredVersionAgreement:
    """Metadata test: the declared segmentation-extra pin in
    ``pyproject.toml`` and the lockfile must agree with the runtime
    supported-version constant. The placement adapter refuses any
    version other than its pinned constant, so the declared range
    must be ``wtpsplit==<version>`` (not ``>=``, not ``<3``).
    """

    @staticmethod
    def _pyproject_text() -> str:
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        return (repo_root / "pyproject.toml").read_text(encoding="utf-8")

    @staticmethod
    def _lockfile_text() -> str:
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        return (repo_root / "uv.lock").read_text(encoding="utf-8")

    def test_runtime_constant_is_stable(self):
        """The runtime supported-version constant must be a known string
        we can match against in the lockfile / pyproject.
        """
        from osm_polygon_sentence_relevance.sentences.sat import (
            WTPSPLIT_SUPPORTED_VERSION,
        )

        assert WTPSPLIT_SUPPORTED_VERSION == "2.2.1"
        # The private adapter module must agree.
        from osm_polygon_sentence_relevance.sentences import _wtpsplit_device

        assert _wtpsplit_device.supported_version() == "2.2.1"

    def test_pyproject_segmentation_extra_is_pinned(self):
        """The segmentation extra must declare ``wtpsplit==<version>``
        (an exact pin), not a range. A range would let a future wtpsplit
        release through that the adapter has not been tested against.
        """
        text = self._pyproject_text()
        # Locate the segmentation extra block.
        marker = "segmentation = ["
        start = text.find(marker)
        assert start >= 0, "segmentation extra block missing from pyproject.toml"
        end = text.find("]", start)
        block = text[start:end]
        # Must contain an exact pin, not a range.
        assert "wtpsplit==2.2.1" in block, (
            f"segmentation extra must pin wtpsplit to 2.2.1; got:\n{block}"
        )
        assert "wtpsplit>=" not in block
        assert "wtpsplit<" not in block

    def test_lockfile_records_wtpsplit_pinned_for_segmentation_extra(self):
        """``uv.lock`` must record ``wtpsplit==2.2.1`` for the
        segmentation extra. This proves the declared pin survives
        resolution into a concrete lockfile entry.
        """
        import re

        text = self._lockfile_text()
        # The lockfile records each extra with its marker; we look for
        # the segmentation extra's specifier entry.
        assert 'specifier = "==2.2.1"' in text, "uv.lock must pin wtpsplit to ==2.2.1"
        # The segmentation extra's wtpsplit entry must be exact-pinned,
        # not a range. Search for the segment-tagged line specifically.
        seg_match = re.search(
            r'name = "wtpsplit", marker = "extra == \'segmentation\'", '
            r'specifier = "==2\.2\.1"',
            text,
        )
        assert seg_match is not None, (
            "uv.lock must pin wtpsplit to ==2.2.1 under the segmentation extra"
        )
        # The package version itself must also be 2.2.1. Locate the
        # package's [[package]] block by scanning for "name = \"wtpsplit\""
        # followed by "version =".
        pkg_match = re.search(
            r'\[\[package\]\]\nname = "wtpsplit"\nversion = "([^"]+)"',
            text,
        )
        assert pkg_match is not None, (
            "uv.lock must contain a [[package]] block for wtpsplit"
        )
        assert pkg_match.group(1) == "2.2.1", (
            f"wtpsplit package version must be 2.2.1, got {pkg_match.group(1)!r}"
        )
