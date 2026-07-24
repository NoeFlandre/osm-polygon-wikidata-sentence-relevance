"""Optional multilingual SaT (Sentence-and-Tagging) model adapter.

This module provides :class:`SaTSentenceSegmenter`, a concrete
:class:`~osm_polygon_sentence_relevance.segmentation.SentenceSegmenter` backed
by the ``wtpsplit`` SaT model. The ``segmentation`` extra installs both
``wtpsplit`` and its required PyTorch runtime:

    uv sync --extra segmentation   # installs wtpsplit 2.2.1 + torch; SaT weights still download lazily

The class performs a lazy import of ``wtpsplit`` and constructs the model on
the first non-empty call, so importing this module or running with plain
``uv sync`` never triggers a network/model load. Model weights are downloaded
and cached by the underlying library at first use; none are stored in this
repository.

Device handling: the segmenter accepts a ``device`` argument
(``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``) and a ``caps`` capability
object. The device is resolved exactly once, immediately before the model
is constructed; the resolved value is then passed unchanged into the
placement adapter in :mod:`osm_polygon_sentence_relevance.sentences._wtpsplit_device`,
which moves the *complete classifier* (not its backbone) onto the
resolved device. Device resolution and model-shape handling are two
distinct concerns: this module never silently downgrades the resolved
device to ``"cpu"`` based on the model's shape.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast

from osm_polygon_sentence_relevance.contracts.errors import SegmentationError
from osm_polygon_sentence_relevance.sentences import _wtpsplit_device
from osm_polygon_sentence_relevance.sentences.device import (
    PUBLIC_DEVICE_VALUES,
    TorchCapabilities,
    default_caps,
    resolve_device,
)


class _SaTModel(Protocol):
    """Minimal structural type for the wtpsplit SaT model surface we use."""

    def split(self, texts: list[str], **kwargs: object) -> Sequence[Sequence[str]]: ...


# Re-export the runtime-supported version so callers can introspect
# it without importing the private module.
WTPSPLIT_SUPPORTED_VERSION: str = _wtpsplit_device.supported_version()


def _lazy_import_sat() -> Callable[..., object]:
    """Backward-compat thin wrapper around the placement adapter's
    lazy importer. Kept here so existing call sites continue to work.
    """
    return cast(Callable[..., object], _wtpsplit_device.lazy_import_sat())


def _is_torch_module(obj: object) -> bool:
    """Forwarder for the placement adapter's torch-module check.

    The implementation lives in :mod:`_wtpsplit_device` so all
    wtpsplit-shaped concerns stay in one place.
    """
    return _wtpsplit_device._is_torch_module(obj)


class SaTSentenceSegmenter:
    """A :class:`SentenceSegmenter` backed by a wtpsplit SaT model.

    The model is constructed lazily on the first non-empty :meth:`split_batch`
    call and reused for all subsequent batches.

    Parameters
    ----------
    model_name:
        SaT model identifier passed to the factory.
    model_factory:
        Optional callable ``(model_name, **model_kwargs) -> model``. Defaults
        to importing ``wtpsplit.SaT``. Injectable for tests.
    model_kwargs:
        Extra keyword arguments forwarded only to model construction.
    split_kwargs:
        Extra keyword arguments forwarded to ``model.split`` on every batch.
    device:
        Requested inference device: one of ``"auto"``, ``"cpu"``, ``"cuda"``,
        ``"mps"``. ``"auto"`` (default) resolves to ``cuda`` when available,
        otherwise ``mps``, otherwise ``cpu``. Explicit ``cuda`` / ``mps``
        requests fail clearly when the backend is unavailable. The value is
        resolved exactly once, immediately before the first model
        construction; the resolved device is then passed unchanged into the
        placement adapter. The segmenter never silently falls back to
        ``"cpu"`` after an accelerator is selected.
    caps:
        Optional capability snapshot implementing
        :class:`~osm_polygon_sentence_relevance.sentences.device.TorchCapabilities`.
        Defaults to a lazy Torch-backed snapshot. Injectable for tests.
    """

    def __init__(
        self,
        model_name: str = "sat-12l-sm",
        *,
        model_factory: Callable[[str], object] | None = None,
        model_kwargs: Mapping[str, object] | None = None,
        split_kwargs: Mapping[str, object] | None = None,
        device: str = "auto",
        caps: TorchCapabilities | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_factory = model_factory
        # Copy caller-supplied mappings so later mutation cannot change
        # this segmenter's behavior.
        self._model_kwargs: dict[str, Any] = dict(model_kwargs or {})
        self._split_kwargs: dict[str, Any] = dict(split_kwargs or {})
        self._model: _SaTModel | None = None
        # ``requested_device`` is validated at construction time so that
        # the public attribute reflects what the user asked for,
        # regardless of when the model is actually built.
        self._caps: TorchCapabilities = caps if caps is not None else default_caps()
        # Validate the syntactic form eagerly so an unknown device fails
        # at the segmenter level, not after model construction. Hardware
        # resolution (which requires a built model) is deferred to the
        # first ``split_batch`` call to keep this constructor light.
        if not isinstance(device, str) or not device.strip():
            raise SegmentationError("device cannot be blank")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise SegmentationError(
                f"device must be one of {sorted(PUBLIC_DEVICE_VALUES)}; got {device!r}"
            )
        self.requested_device: str = device
        # The resolved device is computed exactly once, immediately
        # before the first model construction; ``None`` until then.
        # Device resolution and model-shape support are *separate*
        # concerns: this segmenter never inspects the model's shape
        # before calling the resolver, and never mutates the resolved
        # value after it is computed.
        self.resolved_device: str | None = None

    def _resolve_device_once(self) -> str:
        """Resolve the requested device exactly once per segmenter.

        Cached *only after* a successful placement: if placement
        fails, the resolved value is not committed, so the next
        ``split_batch`` can re-attempt placement. This is what
        prevents a Grid'5000 run that selected CUDA from silently
        retrying on CPU: a failed placement raises before any
        inference is attempted, and ``resolved_device`` stays
        ``None``.
        """
        if self.resolved_device is not None:
            return self.resolved_device
        resolved = resolve_device(self.requested_device, caps=self._caps)
        # NB: we deliberately do NOT cache ``resolved`` here; the cache
        # is only written after ``_ensure_placed`` succeeds. That keeps
        # the cached value in lock-step with the model's actually-placed
        # device.
        return resolved

    def _ensure_placed(self, model: _SaTModel) -> None:
        """Place ``model`` on the previously-resolved device.

        Two responsibilities, kept separate:

        - **Device resolution** happens in :meth:`_resolve_device_once`,
          exactly once, before any model inspection. The resolved
          value is never rewritten based on the model's shape.
        - **Placement** is delegated to
          :func:`_wtpsplit_device.place_classifier`, which selects
          the complete classifier and verifies placement. Placement
          failures propagate; the segmenter never silently downgrades
          the device.

        On success the resolved device is cached on ``self.resolved_device``;
        on failure ``resolved_device`` stays ``None`` so a subsequent
        ``split_batch`` can re-attempt placement rather than reusing a
        broken state.
        """
        if self.resolved_device is not None:
            return
        resolved = self._resolve_device_once()
        try:
            _wtpsplit_device.place_classifier(model, resolved)
        except SegmentationError as exc:
            raise SegmentationError(
                f"SaTSentenceSegmenter: model {self._model_name!r}: {exc}"
            ) from exc
        except Exception as exc:  # intentionally broad: wrap placement errors
            raise SegmentationError(
                f"SaTSentenceSegmenter: failed to place model "
                f"{self._model_name!r} on device {resolved!r}"
            ) from exc
        # Commit the resolved device only after successful placement.
        self.resolved_device = resolved

    def _get_model(self) -> _SaTModel:
        if self._model is not None:
            return self._model
        if self._model_factory is None:
            try:
                model_factory_callable: Callable[..., object] = _lazy_import_sat()
            except SegmentationError:
                raise
            except Exception as exc:
                raise SegmentationError(
                    f"SaTSentenceSegmenter: failed to construct model "
                    f"{self._model_name!r}"
                ) from exc
        else:
            model_factory_callable = self._model_factory
        try:
            model = model_factory_callable(self._model_name, **self._model_kwargs)
        except SegmentationError:
            raise
        except Exception as exc:  # intentionally broad: wrap any construction error
            raise SegmentationError(
                f"SaTSentenceSegmenter: failed to construct model {self._model_name!r}"
            ) from exc
        # Place the model on the resolved device before returning it.
        self._ensure_placed(cast("_SaTModel", model))
        self._model = cast("_SaTModel", model)
        return self._model

    def split_batch(
        self,
        texts: Sequence[str],
        languages: Sequence[str],
    ) -> Sequence[Sequence[str]]:
        """Segment each text into a sequence of sentence strings.

        The base SaT model is language-agnostic, so ``languages`` are accepted
        for interface compatibility but not forwarded. An empty batch returns
        ``()`` without constructing the model.
        """
        if not texts:
            return ()
        model = self._get_model()
        try:
            raw = model.split(list(texts), **self._split_kwargs)
        except Exception as exc:  # intentionally broad: wrap any inference error
            raise SegmentationError(
                "SaTSentenceSegmenter: model inference failed"
            ) from exc
        return tuple(tuple(group) for group in raw)
