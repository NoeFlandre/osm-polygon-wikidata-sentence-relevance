"""Tests for the shared preferred label submission seam."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, cast
from zoneinfo import ZoneInfo

from osm_polygon_sentence_relevance.operator.config import Stage
from osm_polygon_sentence_relevance.operator.label_lanes import LabelLanePlan
from osm_polygon_sentence_relevance.operator.label_submission import (
    submit_preferred_label,
)


class _RecordingController:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def submit(self, **kwargs: Any) -> int:
        self.calls.append(kwargs)
        return 123


def test_submit_preferred_label_preserves_long_window_and_lane_arguments() -> None:
    controller = _RecordingController()
    label_plan = cast(LabelLanePlan, object())

    result = submit_preferred_label(
        controller,
        input_parquet=PurePosixPath("/remote/input.parquet"),
        model_file=PurePosixPath("/remote/model.gguf"),
        tokenizer_dir=PurePosixPath("/remote/tokenizer"),
        gpu_memory_mb=42_000,
        label_plan=label_plan,
        now=datetime(2026, 8, 18, 10, tzinfo=ZoneInfo("Europe/Paris")),
    )

    assert result == 123
    assert controller.calls == [
        {
            "component": Stage.LABEL,
            "input_parquet": PurePosixPath("/remote/input.parquet"),
            "model_file": PurePosixPath("/remote/model.gguf"),
            "tokenizer_dir": PurePosixPath("/remote/tokenizer"),
            "walltime_seconds": 3_300,
            "policy_type": "day",
            "gpu_memory_mb": 42_000,
            "label_plan": label_plan,
        }
    ]
