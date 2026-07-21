"""Characterization tests for the supported package surface."""

from inspect import signature

from osm_polygon_sentence_relevance.application.checkpoint import (
    load_shard_checkpoint,
    publish_shard_checkpoint,
    validate_source_commit,
    validate_work_dir,
)
from osm_polygon_sentence_relevance.application.pipeline import run_pipeline
from osm_polygon_sentence_relevance.output.dataset_card import (
    DatasetStatistics,
    compute_parquet_statistics,
    render_dataset_card,
    render_dataset_card_from_profile,
)


def test_supported_callable_signatures_are_stable() -> None:
    """Structural cleanup must preserve documented entry points."""
    assert "work_dir" in signature(run_pipeline).parameters
    assert list(signature(validate_source_commit).parameters) == ["value"]
    assert list(signature(validate_work_dir).parameters) == ["work_dir"]
    assert "verified_manifest" in signature(publish_shard_checkpoint).parameters
    assert "shard_key" in signature(load_shard_checkpoint).parameters
    assert DatasetStatistics.__name__ == "DatasetStatistics"
    assert callable(compute_parquet_statistics)
    assert callable(render_dataset_card)
    assert callable(render_dataset_card_from_profile)
