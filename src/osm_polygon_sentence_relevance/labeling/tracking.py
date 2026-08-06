"""Optional Trackio export for one validated, static labeling run."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .analytics import LabelAnalytics, build_label_analytics
from .finalization import ValidatedLabeledPublication, validate_labeled_publication
from .releases import release_lane, trackio_space_id


class TrackioError(RuntimeError):
    """Raised when a static Trackio run cannot be created safely."""


@dataclass(frozen=True, slots=True)
class TrackioResult:
    """Facts returned after the single static run has been logged."""

    project: str
    run_name: str
    row_count: int
    kpis: dict[str, int | float]
    space_id: str | None


def _trackio() -> Any:
    try:
        return importlib.import_module("trackio")
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise TrackioError("install the tracking extra to log Trackio metrics") from exc


def _manifest(directory: Path) -> dict[str, Any]:
    try:
        value = json.loads((directory / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        raise TrackioError("cannot read the validated labeling manifest") from exc
    if not isinstance(value, dict):
        raise TrackioError("labeling manifest must be a JSON object")
    return value


def _table_rows(analytics: LabelAnalytics) -> list[list[Any]]:
    return [
        [
            key.split("|", 1)[0],
            key.split("|", 1)[1],
            value,
            analytics.joint_percentages[key],
        ]
        for key, value in analytics.joint_counts.items()
    ]


def _coverage_rows(analytics: LabelAnalytics) -> list[list[Any]]:
    return [[key, value] for key, value in analytics.coverage_funnel.items()]


def _reason_rows(analytics: LabelAnalytics) -> list[list[Any]]:
    return [
        ["landuse_reason", key, value, analytics.landuse_reason_percentages[key]]
        for key, value in analytics.landuse_reason_counts.items()
    ] + [
        ["polygon_reason", key, value, analytics.polygon_reason_percentages[key]]
        for key, value in analytics.polygon_reason_counts.items()
    ]


def _slice_rows(analytics: LabelAnalytics) -> list[list[Any]]:
    return [
        [
            item.dimension,
            item.value,
            item.both_yes_rate,
            item.uncertain_rate,
            item.sample_size,
        ]
        for item in analytics.slices
    ]


def _table(trackio: Any, columns: list[str], data: list[list[Any]]) -> Any:
    return trackio.Table(columns=columns, data=data, log_mode="IMMUTABLE")


def _payload(
    trackio: Any,
    directory: Path,
    analytics: LabelAnalytics,
) -> dict[str, Any]:
    assets = directory / "assets"
    payload = {
        "total_labeled_sentences": analytics.total_labeled_sentences,
        "unique_polygons": analytics.unique_polygons,
        "unique_languages": analytics.unique_languages,
        "strong_positive_yield": analytics.strong_positive_yield,
        "joint_label_heatmap": trackio.Image(
            assets / "joint_label_heatmap.png", caption="Joint label heatmap"
        ),
        "polygon_coverage_funnel": trackio.Image(
            assets / "polygon_coverage_funnel.png", caption="Polygon coverage funnel"
        ),
        "reason_code_distribution": trackio.Image(
            assets / "reason_code_distribution.png",
            caption="Normalized reason-code distributions",
        ),
        "joint_label_table": _table(
            trackio,
            ["landuse_relevance", "polygon_relevance", "count", "percentage"],
            _table_rows(analytics),
        ),
        "polygon_coverage_funnel_table": _table(
            trackio, ["stage", "unique_polygons"], _coverage_rows(analytics)
        ),
        "reason_code_distribution_table": _table(
            trackio,
            ["question", "reason", "count", "percentage"],
            _reason_rows(analytics),
        ),
        "slice_yield_table": _table(
            trackio,
            ["dimension", "value", "both_yes_rate", "uncertain_rate", "sample_size"],
            _slice_rows(analytics),
        ),
    }
    h3_map = assets / "h3_sentence_distribution.png"
    if h3_map.is_file():
        payload["h3_sentence_distribution"] = trackio.Image(
            h3_map, caption="Labeled sentences by H3 cell"
        )
    return payload


def log_static_labeling_run(
    directory: Path,
    *,
    project: str,
    run_name: str | None = None,
    space_id: str | None = None,
) -> TrackioResult:
    """Validate and log one final Trackio step at ``step=0``.

    The manifest's analytics are compared with a fresh Parquet computation
    before Trackio is initialized. This keeps the run static and prevents a
    stale card or manifest from becoming the source of reported metrics.
    """

    project = project.strip()
    if not project:
        raise TrackioError("Trackio project must be non-blank")
    directory = Path(directory)
    try:
        validated: ValidatedLabeledPublication = validate_labeled_publication(directory)
        manifest = _manifest(directory)
        analytics = build_label_analytics(
            pq.read_table(directory / "sentences.parquet")
        )
    except TrackioError:
        raise
    except Exception as exc:
        raise TrackioError("cannot validate final labeling publication") from exc
    if manifest.get("statistics", {}).get("analytics") != analytics.to_dict():
        raise TrackioError("manifest analytics drift from final Parquet")
    trackio = _trackio()
    actual_run_name = run_name or f"final-{validated.parquet_sha256[:12]}"
    if not actual_run_name.strip():
        raise TrackioError("Trackio run name must be non-blank")
    identity = manifest.get("run_identity", {})
    lane = release_lane(identity)
    resolved_space_id = space_id or trackio_space_id(lane)
    config = {
        "row_count": validated.row_count,
        "parquet_sha256": validated.parquet_sha256,
        "source_commit": identity.get("source_commit"),
        "input_dataset_revision": identity.get("input_dataset_revision"),
        "model_repo_id": identity.get("model_repo_id"),
        "model_revision": identity.get("model_revision"),
        "prompt_version": identity.get("prompt_version"),
    }
    init_kwargs: dict[str, Any] = {
        "project": project,
        "name": actual_run_name,
        "config": {**config, "release_lane": lane.value},
        "embed": False,
        "auto_log_gpu": False,
    }
    init_kwargs["space_id"] = resolved_space_id
    started = False
    try:
        trackio.init(**init_kwargs)
        started = True
        trackio.log(_payload(trackio, directory, analytics), step=0)
    except Exception as exc:
        raise TrackioError("Trackio static run failed") from exc
    finally:
        if started:
            try:
                trackio.finish()
            except Exception as exc:
                raise TrackioError("Trackio static run could not finish") from exc
    return TrackioResult(
        project=project,
        run_name=actual_run_name,
        row_count=validated.row_count,
        kpis={
            "total_labeled_sentences": analytics.total_labeled_sentences,
            "unique_polygons": analytics.unique_polygons,
            "unique_languages": analytics.unique_languages,
            "strong_positive_yield": analytics.strong_positive_yield,
        },
        space_id=resolved_space_id,
    )


__all__ = ["TrackioError", "TrackioResult", "log_static_labeling_run"]
