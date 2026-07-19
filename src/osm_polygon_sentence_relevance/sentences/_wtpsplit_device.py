"""Placement adapter for wtpsplit 2.2.1 (Phase 9A amendment; Phase 9K placement-shape correction).

This module isolates *every* wtpsplit-specific concern behind a small,
narrowly-versioned surface:

- The :func:`_load_wtpsplit_pytorch_wrapper_class` loader imports
  ``wtpsplit.extract.PyTorchWrapper`` lazily (using
  ``importlib.import_module`` so the function-name shadowing of the
  submodule is bypassed).
- :func:`_extract_classifier` descends *exactly one level* into the
  wrapper and returns the *complete* classifier (the
  ``SubwordXLMForTokenClassification`` instance). It refuses to
  recurse and refuses shapes it does not recognise. It does NOT
  assume the inner classifier calls its backbone ``.model`` or its
  head ``.classifier`` / ``.score``; complete Transformers
  classifiers register those submodules under encoder-family names
  (``.roberta`` / ``.bert`` / etc.). The placement contract is
  purely "complete ``torch.nn.Module`` at ``wrapper.model``".
- :func:`_place_classifier` calls ``classifier.to(device)`` once and
  verifies every parameter and buffer now reports ``device.type ==
  device``. Partial placements raise.

Device resolution is a separate concern handled in
:mod:`osm_polygon_sentence_relevance.sentences.device` and in
:mod:`osm_polygon_sentence_relevance.sentences.sat`. The two layers
never call each other: this module does **not** query torch
capabilities, does **not** know about the ``"auto"`` / ``"cpu"`` /
``"cuda"`` / ``"mps"`` strings, and does **not** perform any silent
downgrade. The caller passes a concrete, already-resolved device and
this module either places the model on it or raises.

This separation is what makes the Grid'5000 story safe: a production
``run_pipeline`` that selects ``"cuda"`` and constructs a real wtpsplit
model cannot silently land on CPU; either placement succeeds on the
accelerator or the run fails with a clear error.

The supported wtpsplit version is :data:`_SUPPORTED_VERSION` (== the
declared extra pin in ``pyproject.toml``). The loader rejects any
other version with an actionable message rather than guessing at a
shape it has not verified.
"""

from __future__ import annotations

import importlib

from osm_polygon_sentence_relevance.contracts.errors import SegmentationError

#: The wtpsplit version this adapter is structurally tested against.
#: The declared extra pin in ``pyproject.toml`` must agree with this
#: constant; the metadata test
#: ``tests/unit/sentences/test_sat_placement.py::TestDeclaredVersionAgreement``
#: enforces that contract.
_SUPPORTED_VERSION: str = "2.2.1"


def supported_version() -> str:
    """Return the wtpsplit version this adapter is pinned to."""
    return _SUPPORTED_VERSION


def _is_torch_module(obj: object) -> bool:
    """Return ``True`` iff ``obj`` is a ``torch.nn.Module`` instance.

    A real ``isinstance`` check is performed when Torch is importable;
    this is the only reliable way to recognise a torch module,
    regardless of where the class was defined (the real
    ``SubwordXLMForTokenClassification`` is defined inside
    ``transformers``; test doubles may live anywhere). We never infer
    module identity from a class-name substring.
    """
    try:
        import torch.nn as nn
    except ImportError:
        return False
    return isinstance(obj, nn.Module)


def _ensure_supported_wtpsplit(wtpsplit_module: object) -> None:
    """Raise :class:`SegmentationError` if ``wtpsplit_module``'s
    declared version does not match :data:`_SUPPORTED_VERSION`.

    Centralising the check here means the lazy importers and the
    placement adapter all agree on the same version rule.
    """
    installed = getattr(wtpsplit_module, "__version__", None)
    if installed != _SUPPORTED_VERSION:
        raise SegmentationError(
            "SaTSentenceSegmenter: unsupported wtpsplit version "
            f"{installed!r}; this build requires "
            f"wtpsplit=={_SUPPORTED_VERSION}"
        )


def _load_wtpsplit_pytorch_wrapper_class() -> type:
    """Return the wtpsplit ``PyTorchWrapper`` class, imported once.

    Imported lazily because the helper that needs it only runs when a
    real wtpsplit model is in play. Raises
    :class:`SegmentationError` if wtpsplit is missing, the declared
    version does not match the adapter's pinned version, or the
    ``extract.PyTorchWrapper`` class cannot be located.

    The ``wtpsplit.extract`` symbol is *both* a function (re-exported
    in ``wtpsplit/__init__.py``) and a submodule. ``from wtpsplit.extract
    import PyTorchWrapper`` resolves to the function, not the class.
    ``importlib.import_module("wtpsplit.extract")`` loads the actual
    submodule and lets us grab the class attribute off it.
    """
    try:
        import wtpsplit
    except ImportError as exc:
        raise SegmentationError(
            "SaTSentenceSegmenter: wtpsplit is not installed; "
            "install with `uv sync --extra segmentation`"
        ) from exc
    _ensure_supported_wtpsplit(wtpsplit)
    try:
        extract_module = importlib.import_module("wtpsplit.extract")
        PyTorchWrapper = extract_module.PyTorchWrapper
    except ImportError as exc:  # pragma: no cover - wtpsplit 2.2.1 ships it
        raise SegmentationError(
            "SaTSentenceSegmenter: cannot locate wtpsplit.extract.PyTorchWrapper; "
            f"this build requires wtpsplit=={_SUPPORTED_VERSION}"
        ) from exc
    return PyTorchWrapper


def _extract_classifier(model: object) -> object:
    """Locate the *complete* classifier owned by the wtpsplit wrapper.

    The structural contract, narrowly versioned to wtpsplit 2.2.1::

        SaT                                       (façade)
          .model -> wtpsplit.extract.PyTorchWrapper
              .model -> <complete torch.nn.Module>     <-- target
                  .<backbone-name> -> encoder body
                      (e.g. ``.roberta``, ``.bert``,
                       ``.xlm_roberta``, ``.deberta``)
                  .<head-name>     -> nn.Linear
                      (e.g. ``.classifier``, ``.score``,
                       ``.head``, or registered head name)

    ``wrapper.model`` is the *complete* classifier used for inference.
    The adapter must NOT descend into the backbone, must NOT inspect
    the backbone's registered-submodule name (``roberta`` /
    ``bert`` / etc.), and must NOT assume the classification head is
    called ``classifier`` / ``score``. Naming of those internal
    submodules is owned by the encoder family and changes between
    wtpsplit / transformers releases; the placement contract is
    purely "complete ``torch.nn.Module`` at ``wrapper.model``".

    A naïve ``.model`` recursion would land on the backbone and leave
    the head on its original device. We refuse to recurse past the
    wrapper: the *complete* classifier is the object held at
    ``wrapper.model``, full stop.
    """
    PyTorchWrapper = _load_wtpsplit_pytorch_wrapper_class()
    facade = model
    wrapper = getattr(facade, "model", None)
    if wrapper is None:
        raise SegmentationError(
            "SaTSentenceSegmenter: the wtpsplit SaT façade does not expose "
            "a `.model` attribute; the installed wrapper shape is "
            "unsupported"
        )
    if not isinstance(wrapper, PyTorchWrapper):
        raise SegmentationError(
            "SaTSentenceSegmenter: the object at `saT.model` is not a "
            "wtpsplit.extract.PyTorchWrapper; the installed wrapper shape "
            f"is unsupported (got {type(wrapper).__name__!r})"
        )
    classifier = wrapper.model  # type: ignore[attr-defined]
    if classifier is None:
        raise SegmentationError(
            "SaTSentenceSegmenter: the wtpsplit PyTorchWrapper has no "
            "`model` attribute; the installed wrapper shape is unsupported"
        )
    if not _is_torch_module(classifier):
        raise SegmentationError(
            "SaTSentenceSegmenter: the complete classifier at "
            "`wrapper.model` is not a torch.nn.Module; the installed "
            "wrapper shape is unsupported "
            f"(got {type(classifier).__name__!r})"
        )
    return classifier


def _classifier_observed_device(classifier: object) -> str:
    """Read every parameter/buffer device on the classifier.

    Returns the device type (``"cpu"`` / ``"cuda"`` / ``"mps"``) on
    which *all* parameters and buffers live. Raises
    :class:`SegmentationError` if any parameter or buffer is missing
    a device attribute, or if any tensor reports a different device
    than the others — both signals of a partial or broken placement.
    """
    devices: set[str] = set()
    for tensor in classifier.parameters():  # type: ignore[attr-defined]
        dev = getattr(tensor, "device", None)
        dev_type = getattr(dev, "type", None)
        if not isinstance(dev_type, str):
            raise SegmentationError(
                "SaTSentenceSegmenter: classifier parameter reports no "
                "device; placement cannot be verified"
            )
        devices.add(dev_type)
    for tensor in classifier.buffers():  # type: ignore[attr-defined]
        dev = getattr(tensor, "device", None)
        dev_type = getattr(dev, "type", None)
        if not isinstance(dev_type, str):
            raise SegmentationError(
                "SaTSentenceSegmenter: classifier buffer reports no device; "
                "placement cannot be verified"
            )
        devices.add(dev_type)
    if not devices:
        # No parameters and no buffers — the classifier is degenerate.
        # Treat as unknown rather than silently call it CPU.
        raise SegmentationError(
            "SaTSentenceSegmenter: classifier has no parameters or "
            "buffers; placement cannot be verified"
        )
    if len(devices) > 1:
        raise SegmentationError(
            "SaTSentenceSegmenter: classifier parameters/buffers span "
            f"multiple devices {sorted(devices)!r}; placement is partial"
        )
    return next(iter(devices))


def place_classifier(model: object, device: str) -> object:
    """Place the *complete* wtpsplit classifier on ``device`` and verify.

    The helper descends exactly one level into the ``PyTorchWrapper``
    (selecting the complete classifier) and refuses to recurse
    further. It then calls ``classifier.to(device)`` once and verifies
    that every parameter and buffer of the classifier now reports
    ``device.type == device``.

    Raises :class:`SegmentationError` if the wrapper shape is not
    recognised or if placement does not actually take effect. The
    function does not know about ``"auto"``; the caller is responsible
    for resolving the device first.

    The one exception is the CPU-only legacy test-double path: when
    ``device == "cpu"`` and ``model`` is not a recognised wtpsplit
    shape, the function is a no-op. This matches the contract: a
    resolved CPU device need not move a test double that has no real
    Torch state. Resolving to ``"cuda"`` / ``"mps"`` and getting a
    non-torch model — that *does* raise.
    """
    if not isinstance(device, str) or not device.strip():
        raise SegmentationError("SaTSentenceSegmenter: device cannot be blank")
    if not has_supported_shape(model):
        if device == "cpu":
            # Legacy CPU-only path: a test double without a real
            # torch classifier is not moved. This is the only path
            # where the segmenter silently tolerates an unrecognised
            # wrapper shape; any resolved accelerator raises.
            return model
        raise SegmentationError(
            "SaTSentenceSegmenter: the model does not expose a "
            "wtpsplit.extract.PyTorchWrapper with a torch.nn.Module "
            f"classifier; cannot honour device {device!r}"
        )
    classifier = _extract_classifier(model)
    classifier.to(device)  # type: ignore[attr-defined]
    observed = _classifier_observed_device(classifier)
    if observed != device:
        raise SegmentationError(
            "SaTSentenceSegmenter: placement of classifier on device "
            f"{device!r} did not take effect; classifier reports device "
            f"{observed!r}"
        )
    return classifier


def has_supported_shape(model: object) -> bool:
    """Return ``True`` iff ``model`` matches the supported wtpsplit
    shape (façade → PyTorchWrapper → torch classifier).

    Used by the segmenter to decide whether a CPU-only model factory
    (e.g. an injected test double with no ``.to``) can be tolerated.
    For real wtpsplit models the answer is ``True`` and the resolved
    accelerator is honoured.
    """
    try:
        PyTorchWrapper = _load_wtpsplit_pytorch_wrapper_class()
    except SegmentationError:
        return False
    wrapper = getattr(model, "model", None)
    if not isinstance(wrapper, PyTorchWrapper):
        return False
    return _is_torch_module(getattr(wrapper, "model", None))


def lazy_import_sat() -> object:
    """Import ``wtpsplit.SaT`` lazily, returning the class.

    Raises :class:`SegmentationError` if wtpsplit is missing or its
    version is unsupported.
    """
    try:
        import wtpsplit
    except ImportError as exc:
        raise SegmentationError(
            "SaTSentenceSegmenter requires the optional 'wtpsplit' dependency. "
            "Install it with: uv sync --extra segmentation"
        ) from exc
    _ensure_supported_wtpsplit(wtpsplit)
    return wtpsplit.SaT


__all__ = [
    "supported_version",
    "lazy_import_sat",
    "place_classifier",
    "has_supported_shape",
]
