"""Targeted tests for split-timing validation paths in finalization."""

from __future__ import annotations

import pytest

from osm_polygon_sentence_relevance.labeling.finalization import (
    LabelFinalizationError,
    _validate_split_timing,
)


def test_validate_split_timing_rejects_missing_initial() -> None:
    with pytest.raises(LabelFinalizationError, match="initial_inference_seconds"):
        _validate_split_timing(
            {
                "repair_inference_seconds": 1.0,
                "inference_seconds": 1.0,
            }
        )


def test_validate_split_timing_rejects_missing_repair() -> None:
    with pytest.raises(LabelFinalizationError, match="repair_inference_seconds"):
        _validate_split_timing(
            {
                "initial_inference_seconds": 1.0,
                "inference_seconds": 1.0,
            }
        )


def test_validate_split_timing_rejects_negative_component() -> None:
    with pytest.raises(LabelFinalizationError, match="non-negative"):
        _validate_split_timing(
            {
                "initial_inference_seconds": -1.0,
                "repair_inference_seconds": 1.0,
                "inference_seconds": 0.0,
            }
        )


def test_validate_split_timing_rejects_inconsistent_sum() -> None:
    with pytest.raises(LabelFinalizationError, match="inference_seconds"):
        _validate_split_timing(
            {
                "initial_inference_seconds": 1.0,
                "repair_inference_seconds": 2.0,
                "inference_seconds": 4.0,
            }
        )


def test_validate_split_timing_accepts_valid_split() -> None:
    _validate_split_timing(
        {
            "initial_inference_seconds": 1.0,
            "repair_inference_seconds": 2.0,
            "inference_seconds": 3.0,
        }
    )


def test_validate_split_timing_rejects_missing_inference() -> None:
    with pytest.raises(LabelFinalizationError, match="inference_seconds"):
        _validate_split_timing(
            {
                "initial_inference_seconds": 1.0,
                "repair_inference_seconds": 1.0,
            }
        )
