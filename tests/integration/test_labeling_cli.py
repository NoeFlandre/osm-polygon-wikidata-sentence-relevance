from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.labeling.cli import main


class Engine:
    def generate(self, messages: list[list[dict[str, str]]]) -> list[str]:
        return [
            json.dumps(
                {
                    "landuse_relevance": "no",
                    "polygon_relevance": "yes",
                    "landuse_reason": "no_landuse_or_cover",
                    "polygon_reason": "direct_polygon_reference",
                    "evidence": "text",
                }
            )
            for _ in messages
        ]


def _input(path: Path) -> str:
    pq.write_table(
        pa.table(
            {
                "sentence_id": ["s1"],
                "sentence_text_raw": ["text"],
                "previous_sentence": [None],
                "next_sentence": [None],
                "polygon_name": ["Place"],
                "region": ["afghanistan"],
                "osm_primary_tag": ["place=city"],
                "osm_tags": [[{"key": "place", "value": "city"}]],
                "language": ["en"],
                "page_title": ["Place"],
                "section_path": [["History"]],
            }
        ),
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publish_command_does_not_require_label_runtime_arguments(
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, str]] = []

    def publish(output_dir: Path, dataset_id: str) -> object:
        calls.append((output_dir, dataset_id))
        return type("Publication", (), {"commit_id": "a" * 40, "commit_url": "url"})()

    assert (
        main(
            [
                "publish",
                "--output-dir",
                str(tmp_path / "output"),
                "--dataset-id",
                "owner/dataset",
            ],
            publish_fn=publish,
        )
        == 0
    )
    assert calls == [(tmp_path / "output", "owner/dataset")]


def test_track_command_logs_one_static_run_without_runtime_arguments() -> None:
    calls: list[dict[str, object]] = []

    def track(output_dir: Path, **kwargs: object) -> object:
        calls.append({"output_dir": output_dir, **kwargs})
        return SimpleNamespace(
            project="project",
            run_name="run",
            row_count=10,
            kpis={"strong_positive_yield": 0.5},
            space_id=None,
        )

    result = main(
        [
            "track",
            "--output-dir",
            "/data/output",
            "--project",
            "project",
            "--run-name",
            "run",
        ],
        track_fn=track,
    )

    assert result == 0
    assert calls == [
        {
            "output_dir": Path("/data/output"),
            "project": "project",
            "run_name": "run",
            "space_id": None,
        }
    ]


def test_label_command_runs_and_reports_resumable_result(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "sentences.parquet"
    digest = _input(source)
    rc = main(
        [
            "label",
            "--input-parquet",
            str(source),
            "--work-dir",
            str(tmp_path / "work"),
            "--input-dataset-revision",
            "a" * 40,
            "--model-revision",
            "b" * 40,
            "--model-file-sha256",
            "c" * 64,
            "--source-commit",
            "d" * 40,
            "--engine",
            "llama.cpp",
            "--engine-version",
            "0.21.0",
            "--batch-size",
            "1",
        ],
        engine_factory=lambda args: Engine(),
    )
    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["completed"] == 1
    assert output["interrupted"] is False
    assert output["input_sha256"] == digest


def test_probe_command_validates_real_prompt_response_without_checkpoint(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "sentences.parquet"
    _input(source)

    rc = main(
        [
            "probe",
            "--input-parquet",
            str(source),
            "--engine",
            "llama.cpp",
            "--sample-size",
            "1",
        ],
        engine_factory=lambda args: Engine(),
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {
        "engine": "llama.cpp",
        "validated_responses": 1,
    }
    assert not (tmp_path / "work").exists()


def test_label_row_limit_selects_a_partial_representative_run(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "sentences.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sentence_id": f"s{index}",
                    "sentence_text_raw": "text",
                    "previous_sentence": None,
                    "next_sentence": None,
                    "polygon_name": "Place",
                    "region": "afghanistan",
                    "osm_primary_tag": "place=city",
                    "osm_tags": [{"key": "place", "value": "city"}],
                    "language": language,
                    "page_title": "Place",
                    "section_path": ["History"],
                    "source": source_name,
                }
                for index, (language, source_name) in enumerate(
                    [
                        ("en", "wikipedia"),
                        ("fa", "wikipedia"),
                        ("ps", "wikipedia"),
                        ("fr", "wikivoyage"),
                    ]
                )
            ]
        ),
        source,
    )
    rc = main(
        [
            "label",
            "--input-parquet",
            str(source),
            "--work-dir",
            str(tmp_path / "work"),
            "--input-dataset-revision",
            "a" * 40,
            "--model-revision",
            "b" * 40,
            "--model-file-sha256",
            "c" * 64,
            "--source-commit",
            "d" * 40,
            "--engine",
            "llama.cpp",
            "--engine-version",
            "0.21.0",
            "--batch-size",
            "2",
            "--row-limit",
            "2",
        ],
        engine_factory=lambda args: Engine(),
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["total"] == 2


def test_label_command_rejects_mutable_or_malformed_revisions(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "sentences.parquet"
    _input(source)
    rc = main(
        [
            "label",
            "--input-parquet",
            str(source),
            "--work-dir",
            str(tmp_path / "work"),
            "--input-dataset-revision",
            "main",
            "--model-revision",
            "bad",
            "--model-file-sha256",
            "bad",
            "--source-commit",
            "bad",
            "--engine",
            "llama.cpp",
            "--engine-version",
            "x",
            "--batch-size",
            "1",
        ],
        engine_factory=lambda args: Engine(),
    )
    assert rc == 2
    assert "40-character" in capsys.readouterr().err
