"""Application layer: CLI and pipeline orchestration."""

from osm_polygon_sentence_relevance.application.cli import main
from osm_polygon_sentence_relevance.application.pipeline import (
    PipelineResult,
    run_pipeline,
)

__all__ = ["main", "PipelineResult", "run_pipeline"]
