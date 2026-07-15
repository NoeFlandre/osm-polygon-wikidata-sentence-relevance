"""Optional multilingual SaT (Sentence-and-Tagging) model adapter (Phase 3I).

This module provides :class:`SaTSentenceSegmenter`, a concrete
:class:`~osm_polygon_sentence_relevance.segmentation.SentenceSegmenter` backed
by the ``wtpsplit`` SaT model. The heavy dependency is optional and is only
required when the ``segmentation`` extra is installed:

    uv sync --extra segmentation

The class performs a lazy import of ``wtpsplit`` and constructs the model on
the first non-empty call, so importing this module or running with plain
``uv sync`` never triggers a network/model load. Model weights are downloaded
and cached by the underlying library at first use; none are stored in this
repository.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from osm_polygon_sentence_relevance.errors import SegmentationError
from osm_polygon_sentence_relevance.segmentation import SentenceSegmenter


def _lazy_import_sat():
    """Import ``wtpsplit.SaT`` lazily, raising a helpful error if missing."""
    try:
        from wtpsplit import SaT
    except ImportError as exc:  # noqa: BLE001 - surface a guided message
        raise SegmentationError(
            "SaTSentenceSegmenter requires the optional 'wtpsplit' dependency. "
            "Install it with: uv sync --extra segmentation"
        ) from exc
    return SaT


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
    """

    def __init__(
        self,
        model_name: str = "sat-3l-sm",
        *,
        model_factory: Callable[[str], object] | None = None,
        model_kwargs: Mapping[str, object] | None = None,
        split_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_factory = model_factory
        # Copy caller-supplied mappings so later mutation cannot change
        # this segmenter's behavior.
        self._model_kwargs: dict[str, Any] = dict(model_kwargs or {})
        self._split_kwargs: dict[str, Any] = dict(split_kwargs or {})
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        factory = self._model_factory
        if factory is None:
            factory = _lazy_import_sat()
        try:
            model = factory(self._model_name, **self._model_kwargs)
        except SegmentationError:
            raise
        except Exception as exc:  # noqa: BLE001 - wrap any construction error
            raise SegmentationError(
                f"SaTSentenceSegmenter: failed to construct model "
                f"{self._model_name!r}"
            ) from exc
        self._model = model
        return model

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
        except Exception as exc:  # noqa: BLE001 - wrap any inference error
            raise SegmentationError(
                "SaTSentenceSegmenter: model inference failed"
            ) from exc

        return tuple(tuple(group) for group in raw)
