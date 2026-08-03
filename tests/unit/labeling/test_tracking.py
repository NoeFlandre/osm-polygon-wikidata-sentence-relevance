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
from osm_polygon_sentence_relevance.labeling.analytics import build_label_analytics
from osm_polygon_sentence_relevance.labeling.tracking import (
    TrackioError,
    log_static_labeling_run,
)


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
        "slice_yield.html",
    ):
        (assets / name).write_bytes(b"asset")
    return output


class _FakeTrackio(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("trackio")
        self.init_calls: list[dict] = []
        self.log_calls: list[tuple[dict, int]] = []
        self.save_calls: list[Path] = []
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
    def Markdown(text):
        return {"markdown": text}

    def save(self, path):
        self.save_calls.append(Path(path))
        return "files/slice_yield.html"

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
    assert "slice_yield" in metrics
    assert fake.save_calls == [output / "assets" / "slice_yield.html"]
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
