"""Tests for the optional multilingual SaT model adapter (the implementation)."""

from __future__ import annotations

import pytest

from osm_polygon_sentence_relevance.errors import SegmentationError
from osm_polygon_sentence_relevance.sat_adapter import SaTSentenceSegmenter
from osm_polygon_sentence_relevance.segmentation import (
    SentenceSegmenter,
    split_validated_batch,
)
from osm_polygon_sentence_relevance.sentence_table import segment_joined_sections


class FakeSaTModel:
    """A fake SaT model that records construction and split arguments."""

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        FakeSaTModel.constructed.append(self)

    def split(self, texts, **split_kwargs):
        FakeSaTModel.split_calls += 1
        FakeSaTModel.last_split_kwargs = dict(split_kwargs)
        return [text.split("|") for text in texts]


def make_factory():
    def factory(model_name, **kwargs):
        return FakeSaTModel(model_name, **kwargs)

    return factory


@pytest.fixture(autouse=True)
def reset_fake():
    FakeSaTModel.constructed = []
    FakeSaTModel.split_calls = 0
    FakeSaTModel.last_split_kwargs = None
    return


class _CpuOnlyCaps:
    """A no-accelerator capability snapshot for hardware-independent tests."""

    cuda_available = False
    mps_available = False


@pytest.fixture
def caps_cpu_only():
    return _CpuOnlyCaps()


class TestSaTLazyConstruction:
    def test_lazy_construction(self):
        seg = SaTSentenceSegmenter(model_factory=make_factory(), caps=caps_cpu_only)
        assert FakeSaTModel.constructed == []
        seg.split_batch(["A.", "B."], ["en", "en"])
        assert len(FakeSaTModel.constructed) == 1

    def test_constructed_once_across_calls(self):
        seg = SaTSentenceSegmenter(model_factory=make_factory(), caps=caps_cpu_only)
        seg.split_batch(["A."], ["en"])
        seg.split_batch(["B."], ["en"])
        assert len(FakeSaTModel.constructed) == 1

    def test_correct_default_model_name(self):
        seg = SaTSentenceSegmenter(model_factory=make_factory(), caps=caps_cpu_only)
        seg.split_batch(["A."], ["en"])
        assert FakeSaTModel.constructed[0].model_name == "sat-3l-sm"

    def test_custom_model_name_and_constructor_kwargs(self):
        seg = SaTSentenceSegmenter(
            "sat-12l",
            model_factory=make_factory(),
            model_kwargs={"flash_attention": True},
            caps=caps_cpu_only,
        )
        seg.split_batch(["A."], ["en"])
        model = FakeSaTModel.constructed[0]
        assert model.model_name == "sat-12l"
        assert model.kwargs == {"flash_attention": True}

    def test_empty_batch_does_not_construct(self):
        seg = SaTSentenceSegmenter(model_factory=make_factory(), caps=caps_cpu_only)
        assert seg.split_batch([], []) == ()
        assert FakeSaTModel.constructed == []


class TestSaTInference:
    def test_split_kwargs_forwarded(self):
        seg = SaTSentenceSegmenter(
            model_factory=make_factory(),
            split_kwargs={"do_flush": True},
            caps=caps_cpu_only,
        )
        seg.split_batch(["A.", "B."], ["en", "en"])
        assert FakeSaTModel.split_calls == 1
        assert FakeSaTModel.last_split_kwargs == {"do_flush": True}

    def test_exactly_one_inference_call_per_batch(self):
        seg = SaTSentenceSegmenter(model_factory=make_factory(), caps=caps_cpu_only)
        seg.split_batch(["A.", "B.", "C."], ["en", "en", "en"])
        assert FakeSaTModel.split_calls == 1

    def test_input_and_output_order_preserved(self):
        seg = SaTSentenceSegmenter(model_factory=make_factory(), caps=caps_cpu_only)
        out = seg.split_batch(["a|b", "c|d"], ["en", "en"])
        assert out == (("a", "b"), ("c", "d"))

    def test_generator_output_materialized(self):
        class GenModel(FakeSaTModel):
            def split(self, texts, **kwargs):
                FakeSaTModel.split_calls += 1

                def gen():
                    for t in texts:
                        yield t.split("|")

                return gen()

        def gen_factory(model_name, **kwargs):
            return GenModel(model_name, **kwargs)

        seg = SaTSentenceSegmenter(model_factory=gen_factory, caps=caps_cpu_only)
        out = seg.split_batch(["a|b"], ["en"])
        assert out == (("a", "b"),)
        # Result must be immutable nested tuples, not a generator.
        assert isinstance(out, tuple)
        assert all(isinstance(group, tuple) for group in out)

    def test_caller_mutation_of_kwargs_has_no_effect(self):
        model_kwargs = {"x": 1}
        split_kwargs = {"y": 2}
        seg = SaTSentenceSegmenter(
            model_factory=make_factory(),
            model_kwargs=model_kwargs,
            split_kwargs=split_kwargs,
            caps=caps_cpu_only,
        )
        seg.split_batch(["A."], ["en"])
        model_kwargs["x"] = 999
        split_kwargs["y"] = 888
        # Second call still uses the originally provided kwargs.
        seg.split_batch(["B."], ["en"])
        assert FakeSaTModel.constructed[0].kwargs == {"x": 1}
        assert FakeSaTModel.last_split_kwargs == {"y": 2}


class TestSaTErrors:
    def test_construction_error_causality(self):
        def failing_factory(model_name, **kwargs):
            raise RuntimeError("boom-load")

        seg = SaTSentenceSegmenter(model_factory=failing_factory, caps=caps_cpu_only)
        with pytest.raises(SegmentationError) as exc:
            seg.split_batch(["A."], ["en"])
        assert exc.value.__cause__ is not None
        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_inference_error_causality(self):
        class BoomModel(FakeSaTModel):
            def split(self, texts, **kwargs):
                raise RuntimeError("boom-infer")

        def boom_factory(model_name, **kwargs):
            return BoomModel(model_name, **kwargs)

        seg = SaTSentenceSegmenter(model_factory=boom_factory, caps=caps_cpu_only)
        with pytest.raises(SegmentationError) as exc:
            seg.split_batch(["A."], ["en"])
        assert exc.value.__cause__ is not None
        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_missing_dependency_message(self):
        # Make ``wtpsplit`` appear absent by removing it from ``sys.modules``
        # and providing a meta-path finder hook that raises ImportError for it.
        import importlib.abc
        import importlib.machinery
        import sys

        class _BlockWtpsplit(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "wtpsplit" or fullname.startswith("wtpsplit."):
                    raise ImportError(f"No module named {fullname!r}")
                return None

        blocked = _BlockWtpsplit()
        sys.modules.pop("wtpsplit", None)
        sys.meta_path.insert(0, blocked)
        try:
            seg = SaTSentenceSegmenter(
                caps=caps_cpu_only
            )  # no factory -> uses real importer
            with pytest.raises(SegmentationError) as exc:
                seg.split_batch(["A."], ["en"])
            assert "uv sync --extra segmentation" in str(exc.value)
        finally:
            sys.meta_path.remove(blocked)
            sys.modules.pop("wtpsplit", None)


class TestSaTProtocolAndIntegration:
    def test_satisfies_sentence_segmenter(self):
        seg = SaTSentenceSegmenter(model_factory=make_factory(), caps=caps_cpu_only)
        assert isinstance(seg, SentenceSegmenter)

    def test_integrates_through_split_validated_batch(self):
        seg = SaTSentenceSegmenter(model_factory=make_factory(), caps=caps_cpu_only)
        out = split_validated_batch(seg, ["a|b", "c|d"], ["en", "en"])
        assert out == (("a", "b"), ("c", "d"))

    def test_end_to_end_through_segment_joined_sections(self):
        # Build one joined section row with a fake segmenter via factory.

        from tests.unit.sentences.test_sentence_table import _one_row

        table = _one_row(
            section_text_raw="First sentence.|Second sentence.",
            section_path_raw=['["Introduction"]'],
        )
        seg = SaTSentenceSegmenter(model_factory=make_factory(), caps=caps_cpu_only)
        result = segment_joined_sections(table, seg)
        rows = result.table.to_pylist()
        assert [r["sentence_text_raw"] for r in rows] == [
            "First sentence.",
            "Second sentence.",
        ]
