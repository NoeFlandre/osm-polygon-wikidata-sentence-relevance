from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import osm_polygon_sentence_relevance.labeling.tracking as tracking_module
from osm_polygon_sentence_relevance.labeling.analytics import (
    LabelAnalytics,
    SliceYield,
    build_label_analytics,
)
from osm_polygon_sentence_relevance.labeling.tracking import (
    TrackioError,
    log_static_labeling_run,
)
from osm_polygon_sentence_relevance.labeling.v2_analytics import V2Analytics


def _output(tmp_path: Path) -> Path:
    output = tmp_path / "publication"
    output.mkdir()
    rows = [
        {
            "polygon_id": f"p{i % 2}",
            "language": "en",
            "source": "wikipedia",
            "osm_primary_tag": "landuse=farmland",
            "landuse_relevance": "yes" if i % 2 else "no",
            "polygon_relevance": "yes",
            "landuse_reason": "explicit_land_use",
            "polygon_reason": "direct_polygon_reference",
        }
        for i in range(100)
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output / "sentences.parquet")
    analytics = build_label_analytics(table)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "statistics": {"analytics": analytics.to_dict()},
                "run_identity": {
                    "source_commit": "a" * 40,
                    "input_dataset_revision": "b" * 40,
                    "model_repo_id": "model/repo",
                    "model_revision": "c" * 40,
                    "prompt_version": "test",
                },
            }
        )
    )
    assets = output / "assets"
    assets.mkdir()
    for name in (
        "joint_label_heatmap.png",
        "polygon_coverage_funnel.png",
        "reason_code_distribution.png",
    ):
        (assets / name).write_bytes(b"asset")
    return output


class _FakeTrackio(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("trackio")
        self.init_calls: list[dict] = []
        self.log_calls: list[tuple[dict, int]] = []
        self.finish_calls = 0

    def init(self, **kwargs):
        self.init_calls.append(kwargs)

    def log(self, metrics, *, step):
        self.log_calls.append((metrics, step))

    def finish(self):
        self.finish_calls += 1

    @staticmethod
    def Image(path, *, caption):
        return {"image": str(path), "caption": caption}

    @staticmethod
    def Table(*, columns, data, log_mode):
        return {"columns": columns, "data": data, "log_mode": log_mode}


def test_static_run_uses_installed_trackio_init_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StrictFake(_FakeTrackio):
        def init(
            self,
            *,
            project,
            name,
            config,
            embed,
            auto_log_gpu,
            space_id=None,
        ):
            super().init(
                project=project,
                name=name,
                config=config,
                embed=embed,
                auto_log_gpu=auto_log_gpu,
                space_id=space_id,
            )

    fake = StrictFake()
    monkeypatch.setitem(sys.modules, "trackio", fake)
    monkeypatch.setattr(
        tracking_module,
        "validate_labeled_publication",
        lambda _: SimpleNamespace(row_count=100, parquet_sha256="a" * 64),
    )

    result = log_static_labeling_run(_output(tmp_path), project="project")

    assert result.row_count == 100
    assert fake.init_calls[0]["auto_log_gpu"] is False


def test_static_run_logs_one_step_with_kpis_tables_images_and_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeTrackio()
    monkeypatch.setitem(sys.modules, "trackio", fake)
    monkeypatch.setattr(
        tracking_module,
        "validate_labeled_publication",
        lambda _: SimpleNamespace(row_count=100, parquet_sha256="a" * 64),
    )
    output = _output(tmp_path)
    result = log_static_labeling_run(
        output,
        project="afghanistan-labeling",
        run_name="final",
        space_id="owner/space",
    )
    assert result.project == "afghanistan-labeling"
    assert result.row_count == 100
    assert fake.init_calls[0]["space_id"] == "owner/space"
    assert len(fake.init_calls) == 1
    assert len(fake.log_calls) == 1
    metrics, step = fake.log_calls[0]
    assert step == 0
    assert metrics["total_labeled_sentences"] == 100
    assert "unique_polygons" in metrics
    assert "unique_languages" in metrics
    assert "strong_positive_yield" in metrics
    assert {key for key in metrics if key.endswith("_table")} == {
        "joint_label_table",
        "polygon_coverage_funnel_table",
        "reason_code_distribution_table",
        "slice_yield_table",
    }
    assert "joint_label_heatmap" in metrics
    assert "polygon_coverage_funnel" in metrics
    assert "reason_code_distribution" in metrics
    assert "slice_yield" not in metrics
    assert fake.finish_calls == 1


def test_static_run_refuses_manifest_analytics_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeTrackio()
    monkeypatch.setitem(sys.modules, "trackio", fake)
    monkeypatch.setattr(
        tracking_module,
        "validate_labeled_publication",
        lambda _: SimpleNamespace(row_count=100, parquet_sha256="a" * 64),
    )
    output = _output(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["statistics"]["analytics"]["unique_polygons"] = 999
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(TrackioError, match="analytics drift"):
        log_static_labeling_run(output, project="project")
    assert fake.init_calls == []


def test_static_run_reports_missing_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "trackio", None)
    monkeypatch.setattr(
        tracking_module,
        "validate_labeled_publication",
        lambda _: SimpleNamespace(row_count=100, parquet_sha256="a" * 64),
    )
    with pytest.raises(TrackioError, match="tracking extra"):
        log_static_labeling_run(_output(tmp_path), project="project")


def test_static_run_rejects_blank_project(tmp_path: Path) -> None:
    with pytest.raises(TrackioError, match="project must be non-blank"):
        log_static_labeling_run(_output(tmp_path), project=" ")


@pytest.mark.parametrize("manifest_value", ["not-json", "[]"])
def test_static_run_rejects_malformed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_value: str,
) -> None:
    output = _output(tmp_path)
    (output / "manifest.json").write_text(manifest_value)
    monkeypatch.setattr(
        tracking_module,
        "validate_labeled_publication",
        lambda _: SimpleNamespace(row_count=100, parquet_sha256="a" * 64),
    )
    with pytest.raises(TrackioError, match="manifest"):
        log_static_labeling_run(output, project="project")


def test_static_run_rejects_blank_run_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeTrackio()
    monkeypatch.setitem(sys.modules, "trackio", fake)
    monkeypatch.setattr(
        tracking_module,
        "validate_labeled_publication",
        lambda _: SimpleNamespace(row_count=100, parquet_sha256="a" * 64),
    )
    with pytest.raises(TrackioError, match="run name"):
        log_static_labeling_run(_output(tmp_path), project="project", run_name=" ")


def test_static_run_wraps_init_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken(_FakeTrackio):
        def init(self, **kwargs):
            raise RuntimeError("init failed")

    monkeypatch.setitem(sys.modules, "trackio", Broken())
    monkeypatch.setattr(
        tracking_module,
        "validate_labeled_publication",
        lambda _: SimpleNamespace(row_count=100, parquet_sha256="a" * 64),
    )
    with pytest.raises(TrackioError, match="static run failed"):
        log_static_labeling_run(_output(tmp_path), project="project")


def test_static_run_wraps_finish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken(_FakeTrackio):
        def finish(self):
            raise RuntimeError("finish failed")

    monkeypatch.setitem(sys.modules, "trackio", Broken())
    monkeypatch.setattr(
        tracking_module,
        "validate_labeled_publication",
        lambda _: SimpleNamespace(row_count=100, parquet_sha256="a" * 64),
    )
    with pytest.raises(TrackioError, match="could not finish"):
        log_static_labeling_run(_output(tmp_path), project="project")


def test_v2_payload_contains_only_declared_static_metrics_and_assets(
    tmp_path: Path,
) -> None:
    analytics = V2Analytics(
        total_labeled_sentences=12,
        unique_polygons=7,
        unique_languages=3,
        place_counts={"no": 5, "yes": 7},
        place_percentages={"no": 5 / 12, "yes": 7 / 12},
        area_bucket_counts={"large": 2, "small": 10},
        h3_cell_count=4,
        missing_coordinate_count=1,
    )
    fake = _FakeTrackio()

    payload = tracking_module._v2_payload(fake, tmp_path / "publication", analytics)

    assert payload["total_labeled_sentences"] == 12
    assert payload["unique_polygons"] == 7
    assert payload["unique_languages"] == 3
    assert payload["place_description_yield"] == 7 / 12
    assert payload["area_bucket_counts"] == {
        "columns": ["area_bucket", "sentences"],
        "data": [["large", 2], ["small", 10]],
        "log_mode": "IMMUTABLE",
    }
    assert payload["label_distribution"] == {
        "image": str(tmp_path / "publication" / "assets" / "label_distribution.png"),
        "caption": "Place-description labels",
    }
    assert payload["h3_sentence_distribution"] == {
        "image": str(
            tmp_path / "publication" / "assets" / "h3_sentence_distribution.png"
        ),
        "caption": "Labeled sentences by H3 cell",
    }
    assert set(payload) == {
        "total_labeled_sentences",
        "unique_polygons",
        "unique_languages",
        "place_description_yield",
        "area_bucket_counts",
        "label_distribution",
        "h3_sentence_distribution",
    }


def test_v2_payload_uses_zero_yield_when_no_positive_labels_exist(
    tmp_path: Path,
) -> None:
    analytics = V2Analytics(
        total_labeled_sentences=4,
        unique_polygons=2,
        unique_languages=1,
        place_counts={"no": 4},
        place_percentages={"no": 1.0},
        area_bucket_counts={"small": 4},
        h3_cell_count=1,
        missing_coordinate_count=0,
    )

    payload = tracking_module._v2_payload(
        _FakeTrackio(), tmp_path / "publication", analytics
    )

    assert payload["place_description_yield"] == 0.0


def test_v1_payload_preserves_all_metrics_tables_and_optional_h3_asset(
    tmp_path: Path,
) -> None:
    """The legacy static payload keeps its published keys and table schemas."""

    analytics = LabelAnalytics(
        total_labeled_sentences=12,
        unique_polygons=4,
        unique_languages=2,
        strong_positive_count=3,
        strong_positive_yield=0.25,
        joint_counts={"yes|yes": 3, "no|yes": 2},
        joint_percentages={"yes|yes": 0.25, "no|yes": 2 / 12},
        coverage_funnel={"all_polygons": 4, "polygon_relevant_polygons": 3},
        landuse_reason_counts={"explicit": 5},
        landuse_reason_percentages={"explicit": 5 / 12},
        polygon_reason_counts={"direct": 6},
        polygon_reason_percentages={"direct": 0.5},
        slices=(
            SliceYield(
                dimension="language",
                value="en",
                sample_size=12,
                both_yes_rate=0.25,
                uncertain_rate=0.1,
            ),
        ),
    )
    directory = tmp_path / "publication"
    assets = directory / "assets"
    assets.mkdir(parents=True)
    h3_map = assets / "h3_sentence_distribution.png"
    h3_map.write_bytes(b"asset")

    payload = tracking_module._payload(_FakeTrackio(), directory, analytics)

    assert payload == {
        "total_labeled_sentences": 12,
        "unique_polygons": 4,
        "unique_languages": 2,
        "strong_positive_yield": 0.25,
        "joint_label_heatmap": {
            "image": str(assets / "joint_label_heatmap.png"),
            "caption": "Joint label heatmap",
        },
        "polygon_coverage_funnel": {
            "image": str(assets / "polygon_coverage_funnel.png"),
            "caption": "Polygon coverage funnel",
        },
        "reason_code_distribution": {
            "image": str(assets / "reason_code_distribution.png"),
            "caption": "Normalized reason-code distributions",
        },
        "joint_label_table": {
            "columns": [
                "landuse_relevance",
                "polygon_relevance",
                "count",
                "percentage",
            ],
            "data": [["yes", "yes", 3, 0.25], ["no", "yes", 2, 2 / 12]],
            "log_mode": "IMMUTABLE",
        },
        "polygon_coverage_funnel_table": {
            "columns": ["stage", "unique_polygons"],
            "data": [["all_polygons", 4], ["polygon_relevant_polygons", 3]],
            "log_mode": "IMMUTABLE",
        },
        "reason_code_distribution_table": {
            "columns": ["question", "reason", "count", "percentage"],
            "data": [
                ["landuse_reason", "explicit", 5, 5 / 12],
                ["polygon_reason", "direct", 6, 0.5],
            ],
            "log_mode": "IMMUTABLE",
        },
        "slice_yield_table": {
            "columns": [
                "dimension",
                "value",
                "both_yes_rate",
                "uncertain_rate",
                "sample_size",
            ],
            "data": [["language", "en", 0.25, 0.1, 12]],
            "log_mode": "IMMUTABLE",
        },
        "h3_sentence_distribution": {
            "image": str(h3_map),
            "caption": "Labeled sentences by H3 cell",
        },
    }


def test_log_static_trackio_preserves_config_payload_and_finish_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeTrackio()
    payload = {"marker": "v2"}
    payload_calls: list[tuple[object, ...]] = []

    def record_v2_payload(*args):
        payload_calls.append(args)
        return payload

    cast_calls: list[tuple[object, object]] = []

    def record_cast(type_hint, value):
        cast_calls.append((type_hint, value))
        return value

    monkeypatch.setattr(tracking_module, "_v2_payload", record_v2_payload)
    monkeypatch.setattr(tracking_module, "cast", record_cast)

    tracking_module._log_static_trackio(
        trackio=fake,
        directory=tmp_path,
        project="project",
        run_name="run",
        resolved_space_id="owner/space",
        identity={
            "release_lane": "v2-worldwide",
            "source_commit": "s" * 40,
            "input_dataset_revision": "i" * 40,
            "model_repo_id": "owner/model",
            "model_revision": "m" * 40,
            "prompt_version": "prompt-v2",
        },
        validated=SimpleNamespace(row_count=42, parquet_sha256="d" * 64),
        analytics=SimpleNamespace(),
        is_v2=True,
    )

    assert fake.init_calls == [
        {
            "project": "project",
            "name": "run",
            "config": {
                "row_count": 42,
                "parquet_sha256": "d" * 64,
                "source_commit": "s" * 40,
                "input_dataset_revision": "i" * 40,
                "model_repo_id": "owner/model",
                "model_revision": "m" * 40,
                "prompt_version": "prompt-v2",
                "release_lane": "v2-worldwide",
            },
            "embed": False,
            "auto_log_gpu": False,
            "space_id": "owner/space",
        }
    ]
    assert fake.log_calls == [(payload, 0)]
    assert fake.finish_calls == 1
    assert payload_calls == [(fake, tmp_path, SimpleNamespace())]
    assert cast_calls == [(tracking_module.V2Analytics, SimpleNamespace())]

    v1_fake = _FakeTrackio()
    v1_payload = {"marker": "v1"}
    v1_payload_calls: list[tuple[object, ...]] = []

    def record_v1_payload(*args):
        v1_payload_calls.append(args)
        return v1_payload

    monkeypatch.setattr(tracking_module, "_payload", record_v1_payload)
    v1_analytics = SimpleNamespace()
    tracking_module._log_static_trackio(
        trackio=v1_fake,
        directory=tmp_path,
        project="project",
        run_name="run-v1",
        resolved_space_id=None,
        identity={"release_lane": "v1-afghanistan"},
        validated=SimpleNamespace(row_count=1, parquet_sha256="e" * 64),
        analytics=v1_analytics,
        is_v2=False,
    )
    assert v1_fake.log_calls == [(v1_payload, 0)]
    assert v1_payload_calls == [(v1_fake, tmp_path, v1_analytics)]
    assert cast_calls[-1] == (tracking_module.LabelAnalytics, v1_analytics)


def test_log_static_trackio_wraps_init_and_finish_errors_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracking_module, "_v2_payload", lambda *args: {"marker": "v2"})

    class InitBroken(_FakeTrackio):
        def init(self, **kwargs):
            raise RuntimeError("init")

    broken = InitBroken()
    with pytest.raises(TrackioError, match="^Trackio static run failed$"):
        tracking_module._log_static_trackio(
            trackio=broken,
            directory=Path("publication"),
            project="project",
            run_name="run",
            resolved_space_id=None,
            identity={"release_lane": "v1-afghanistan"},
            validated=SimpleNamespace(row_count=1, parquet_sha256="d" * 64),
            analytics=SimpleNamespace(),
            is_v2=True,
        )
    assert broken.finish_calls == 0

    class FinishBroken(_FakeTrackio):
        def finish(self):
            super().finish()
            raise RuntimeError("finish")

    with pytest.raises(TrackioError, match="^Trackio static run could not finish$"):
        tracking_module._log_static_trackio(
            trackio=FinishBroken(),
            directory=Path("publication"),
            project="project",
            run_name="run",
            resolved_space_id=None,
            identity={"release_lane": "v1-afghanistan"},
            validated=SimpleNamespace(row_count=1, parquet_sha256="d" * 64),
            analytics=SimpleNamespace(),
            is_v2=True,
        )
