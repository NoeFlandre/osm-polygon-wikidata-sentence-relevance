"""Quality-first production defaults for Afghanistan sentence segmentation."""

from __future__ import annotations

from pathlib import Path

from osm_polygon_sentence_relevance.application.cli import _build_parser
from osm_polygon_sentence_relevance.sentences.sat import SaTSentenceSegmenter


def test_public_segmenter_defaults_to_best_supervised_mixture_model() -> None:
    segmenter = SaTSentenceSegmenter()
    assert segmenter._model_name == "sat-12l-sm"


def test_cli_defaults_to_best_supervised_mixture_model() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--input-root",
            "in",
            "--output-dir",
            "out",
            "--input-dataset-revision",
            "a" * 40,
            "--pipeline-version",
            "0.1.0",
        ]
    )
    assert args.sat_model == "sat-12l-sm"


def test_streaming_production_paths_use_quality_model() -> None:
    driver = Path("scripts/streaming/driver.py").read_text()
    finalization = Path("scripts/streaming/finalization.py").read_text()
    launcher = Path("scripts/grid5000/run_streaming_build.sh").read_text()
    assert '"sat-12l-sm"' in driver
    assert '"sat-12l-sm"' in finalization
    assert '--model-name "sat-12l-sm"' in launcher
    assert "sat-3l-sm" not in launcher
