"""Coverage-targeted tests for process_single_shard validation paths."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from osm_polygon_sentence_relevance.application.pipeline import (
    process_single_shard,
)
from osm_polygon_sentence_relevance.ingestion.discovery import RegionShardSet


class _FakeSegmenter:
    """Satisfies the SentenceSegmenter runtime-checkable protocol."""

    def split_batch(
        self,
        texts: Sequence[str],
        languages: Sequence[str],
    ) -> Sequence[Sequence[str]]:
        return [[t] for t in texts]


def _fake_segmenter() -> _FakeSegmenter:
    return _FakeSegmenter()


def _fake_shard(tmp_path: Path) -> RegionShardSet:
    return RegionShardSet(
        shard_key="italy-latest",
        polygons=tmp_path / "polygons" / "italy-latest.parquet",
        polygon_articles=tmp_path / "polygon_articles" / "italy-latest.parquet",
        wikipedia_documents=tmp_path
        / "wikipedia"
        / "documents"
        / "italy-latest.parquet",
        wikipedia_sections=tmp_path / "wikipedia" / "sections" / "italy-latest.parquet",
        wikivoyage_documents=None,
        wikivoyage_sections=None,
    )


def test_rejects_non_region_shard(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="RegionShardSet"):
        process_single_shard(
            shard="not-a-shard",  # type: ignore[arg-type]
            input_root=tmp_path,
            segmenter=_fake_segmenter(),
            work_dir=tmp_path,
            source_commit="a" * 40,
            input_dataset_revision="r",
            pipeline_version="v1",
            model_name="m",
            batch_size=1,
        )


def test_rejects_non_segmenter(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="SentenceSegmenter"):
        process_single_shard(
            shard=_fake_shard(tmp_path),
            input_root=tmp_path,
            segmenter=object(),  # type: ignore[arg-type]
            work_dir=tmp_path,
            source_commit="a" * 40,
            input_dataset_revision="r",
            pipeline_version="v1",
            model_name="m",
            batch_size=1,
        )


def test_rejects_blank_input_dataset_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input_dataset_revision"):
        process_single_shard(
            shard=_fake_shard(tmp_path),
            input_root=tmp_path,
            segmenter=_fake_segmenter(),
            work_dir=tmp_path,
            source_commit="a" * 40,
            input_dataset_revision="",
            pipeline_version="v1",
            model_name="m",
            batch_size=1,
        )


def test_rejects_blank_pipeline_version(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pipeline_version"):
        process_single_shard(
            shard=_fake_shard(tmp_path),
            input_root=tmp_path,
            segmenter=_fake_segmenter(),
            work_dir=tmp_path,
            source_commit="a" * 40,
            input_dataset_revision="r",
            pipeline_version="",
            model_name="m",
            batch_size=1,
        )


def test_rejects_blank_model_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_name"):
        process_single_shard(
            shard=_fake_shard(tmp_path),
            input_root=tmp_path,
            segmenter=_fake_segmenter(),
            work_dir=tmp_path,
            source_commit="a" * 40,
            input_dataset_revision="r",
            pipeline_version="v1",
            model_name="",
            batch_size=1,
        )


def test_rejects_zero_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        process_single_shard(
            shard=_fake_shard(tmp_path),
            input_root=tmp_path,
            segmenter=_fake_segmenter(),
            work_dir=tmp_path,
            source_commit="a" * 40,
            input_dataset_revision="r",
            pipeline_version="v1",
            model_name="m",
            batch_size=0,
        )


def test_rejects_bool_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        process_single_shard(
            shard=_fake_shard(tmp_path),
            input_root=tmp_path,
            segmenter=_fake_segmenter(),
            work_dir=tmp_path,
            source_commit="a" * 40,
            input_dataset_revision="r",
            pipeline_version="v1",
            model_name="m",
            batch_size=True,
        )
