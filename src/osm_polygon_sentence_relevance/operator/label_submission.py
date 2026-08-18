"""Shared submission policy for preferred labeling allocations."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from osm_polygon_sentence_relevance.operator.config import Stage
from osm_polygon_sentence_relevance.operator.earliest_start import policy_type_for
from osm_polygon_sentence_relevance.operator.label_lanes import LabelLanePlan
from osm_polygon_sentence_relevance.operator.oar import GRID5000_TZ
from osm_polygon_sentence_relevance.operator.workflows import (
    PREFERRED_LABEL_WALLTIME_SECONDS,
)


def submit_preferred_label(
    controller: Any,
    *,
    input_parquet: PurePosixPath,
    model_file: PurePosixPath,
    tokenizer_dir: PurePosixPath,
    gpu_memory_mb: int = 40_000,
    label_plan: LabelLanePlan | None = None,
    now: datetime | None = None,
) -> int:
    """Submit a label job using the shared preferred allocation policy.

    ``now`` is injectable only to make the policy choice deterministic in
    tests; production callers use the current Europe/Paris time.
    """

    walltime_seconds = PREFERRED_LABEL_WALLTIME_SECONDS
    kwargs: dict[str, Any] = {
        "component": Stage.LABEL,
        "input_parquet": input_parquet,
        "model_file": model_file,
        "tokenizer_dir": tokenizer_dir,
        "walltime_seconds": walltime_seconds,
        "policy_type": policy_type_for(
            now if now is not None else datetime.now(tz=GRID5000_TZ),
            walltime_seconds=walltime_seconds,
        ),
        "gpu_memory_mb": gpu_memory_mb,
    }
    if label_plan is not None:
        kwargs["label_plan"] = label_plan
    return controller.submit(**kwargs)
