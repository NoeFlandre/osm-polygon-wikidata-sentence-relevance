from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import osm_polygon_sentence_relevance.labeling.cli as labeling_cli
from osm_polygon_sentence_relevance.labeling.cli import main
from osm_polygon_sentence_relevance.labeling.finalization import TRACKIO_SPACE_ID
from osm_polygon_sentence_relevance.labeling.releases import V2_TRACKIO_SPACE_ID


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


def _label_args(source: Path, work: Path, *extra: str) -> list[str]:
    return [
        "label",
        "--input-parquet",
        str(source),
        "--work-dir",
        str(work),
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
        *extra,
    ]


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
            "space_id": TRACKIO_SPACE_ID,
        }
    ]


def test_track_command_infers_worldwide_space_from_manifest(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "run_identity": {
                    "release_lane": "v2-worldwide",
                }
            }
        )
    )
    calls: list[dict[str, object]] = []

    def track(output_dir: Path, **kwargs: object) -> object:
        calls.append({"output_dir": output_dir, **kwargs})
        return SimpleNamespace(
            project="project",
            run_name="run",
            row_count=1,
            kpis={},
            space_id=V2_TRACKIO_SPACE_ID,
        )

    assert (
        main(
            [
                "track",
                "--output-dir",
                str(output),
                "--project",
                "project",
            ],
            track_fn=track,
        )
        == 0
    )
    assert calls[0]["space_id"] == V2_TRACKIO_SPACE_ID


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


def test_label_command_starts_and_closes_optional_checkpoint_mirror(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "sentences.parquet"
    _input(source)
    events: list[object] = []

    class FakeMirror:
        def __init__(self, **kwargs: object) -> None:
            events.append(("init", kwargs["dataset_id"], kwargs["branch"]))

        def start(self) -> None:
            events.append("start")

        def enqueue(self, index: int) -> None:
            events.append(("enqueue", index))

        def close(self, *, wait: bool, timeout: float) -> None:
            events.append(("close", wait, timeout))

    monkeypatch.setattr(labeling_cli, "CheckpointMirror", FakeMirror)
    assert (
        main(
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
                "--checkpoint-dataset-id",
                "owner/dataset",
                "--checkpoint-branch",
                "checkpoints/aaaaaaaaaaaaaaaaaaaa",
                "--checkpoint-drain-seconds",
                "0",
            ],
            engine_factory=lambda args: Engine(),
        )
        == 0
    )
    assert events == [
        ("init", "owner/dataset", "checkpoints/aaaaaaaaaaaaaaaaaaaa"),
        "start",
        ("enqueue", 0),
        ("close", True, 0.0),
    ]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--llama-total-context", "1"], "total context"),
        (["--request-concurrency", "0"], "request concurrency"),
        (["--row-limit", "-1"], "row limit"),
        (["--checkpoint-dataset-id", "owner/dataset"], "checkpoint dataset"),
        (
            [
                "--checkpoint-dataset-id",
                "owner/dataset",
                "--checkpoint-branch",
                "checkpoints/aaaaaaaaaaaaaaaaaaaa",
                "--checkpoint-drain-seconds",
                "-1",
            ],
            "drain seconds",
        ),
    ],
)
def test_label_rejects_invalid_runtime_contracts(
    tmp_path: Path, capsys, extra: list[str], message: str
) -> None:
    source = tmp_path / "sentences.parquet"
    _input(source)
    assert (
        main(
            _label_args(source, tmp_path / "work", *extra),
            engine_factory=lambda args: Engine(),
        )
        == 2
    )
    assert message in capsys.readouterr().err


def test_label_rejects_input_without_prompt_columns(tmp_path: Path, capsys) -> None:
    source = tmp_path / "sentences.parquet"
    pq.write_table(pa.table({"sentence_id": ["s1"]}), source)
    assert (
        main(
            _label_args(source, tmp_path / "work"),
            engine_factory=lambda args: Engine(),
        )
        == 2
    )
    assert "missing required labeling columns" in capsys.readouterr().err


def test_probe_rejects_invalid_size_and_response_count(tmp_path: Path, capsys) -> None:
    source = tmp_path / "sentences.parquet"
    _input(source)
    assert (
        main(
            [
                "probe",
                "--input-parquet",
                str(source),
                "--engine",
                "llama.cpp",
                "--sample-size",
                "2",
            ],
            engine_factory=lambda args: Engine(),
        )
        == 2
    )
    assert "sample size" in capsys.readouterr().err

    class EmptyEngine:
        def generate(self, messages: list[list[dict[str, str]]]) -> list[str]:
            return []

    assert (
        main(
            [
                "probe",
                "--input-parquet",
                str(source),
                "--engine",
                "llama.cpp",
                "--sample-size",
                "1",
            ],
            engine_factory=lambda args: EmptyEngine(),
        )
        == 2
    )
    assert "response count" in capsys.readouterr().err


def test_finalize_command_delegates_and_reports_result(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "sentences.parquet"
    _input(source)

    def finalize(**kwargs: object) -> object:
        assert kwargs["input_path"] == source
        assert kwargs["dataset_repo_id"] == "owner/dataset"
        return SimpleNamespace(row_count=1, parquet_sha256="a" * 64)

    monkeypatch.setattr(labeling_cli, "finalize_labeled_dataset", finalize)
    assert (
        main(
            [
                "finalize",
                "--input-parquet",
                str(source),
                "--work-dir",
                str(tmp_path / "work"),
                "--output-dir",
                str(tmp_path / "output"),
                "--dataset-id",
                "owner/dataset",
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
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "rows": 1,
        "sha256": "a" * 64,
    }


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
