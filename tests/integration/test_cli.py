from __future__ import annotations

import json
from pathlib import Path

from osm_polygon_sentence_relevance.cli import main
from osm_polygon_sentence_relevance.errors import ExportError
from tests.helpers import make_fake_pipeline_result

# ===================================================================
# Test Suite for Local Build CLI (Phase 6A)
# ===================================================================


class TestCLI:
    def test_cli_help(self, capsys):
        # Running with --help should print help and return 0
        code = main(["--help"])
        assert code == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "usage:" in captured.err.lower()
        assert "--input-root" in captured.out

    def test_cli_missing_arguments(self, capsys):
        # Missing required arguments returns non-zero (2)
        code = main([])
        assert code == 2
        captured = capsys.readouterr()
        assert "required" in captured.err.lower()

    def test_cli_invalid_arguments_before_model(self, capsys):
        calls = []

        def mock_factory(*args, **kwargs):
            calls.append(args)
            return None

        # Invalid batch_size
        code = main(
            [
                "--input-root",
                "/tmp/in",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
                "--batch-size",
                "0",
            ],
            model_factory=mock_factory,
        )
        assert code == 1
        assert len(calls) == 0
        captured = capsys.readouterr()
        assert "positive integer" in captured.err

        # Blank revision
        code = main(
            [
                "--input-root",
                "/tmp/in",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "   ",
                "--pipeline-version",
                "ver",
            ],
            model_factory=mock_factory,
        )
        assert code == 1
        assert len(calls) == 0
        captured = capsys.readouterr()
        assert "revision cannot be blank" in captured.err

    def test_cli_path_configuration_safety(self, capsys):
        calls = []

        def mock_factory(*args, **kwargs):
            calls.append(args)
            return None

        # 1. Blank input-root
        code = main(
            [
                "--input-root",
                "   ",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
            ],
            model_factory=mock_factory,
        )
        assert code == 1
        assert len(calls) == 0
        captured = capsys.readouterr()
        assert "input_root cannot be blank" in captured.err

        # 2. Blank output-dir
        code = main(
            [
                "--input-root",
                "/tmp/in",
                "--output-dir",
                "   ",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
            ],
            model_factory=mock_factory,
        )
        assert code == 1
        assert len(calls) == 0
        captured = capsys.readouterr()
        assert "output_dir cannot be blank" in captured.err

        # 3. Blank sat-model
        code = main(
            [
                "--input-root",
                "/tmp/in",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
                "--sat-model",
                "   ",
            ],
            model_factory=mock_factory,
        )
        assert code == 1
        assert len(calls) == 0
        captured = capsys.readouterr()
        assert "sat_model cannot be blank" in captured.err

        # 4. Same path
        code = main(
            [
                "--input-root",
                "/tmp/same",
                "--output-dir",
                "/tmp/same/.",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
            ],
            model_factory=mock_factory,
        )
        assert code == 1
        assert len(calls) == 0
        captured = capsys.readouterr()
        assert "same path" in captured.err.lower()

        # 5. Overlapping: input is ancestor of output
        code = main(
            [
                "--input-root",
                "/tmp/ancestor",
                "--output-dir",
                "/tmp/ancestor/child/sub",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
            ],
            model_factory=mock_factory,
        )
        assert code == 1
        assert len(calls) == 0
        captured = capsys.readouterr()
        assert "ancestor" in captured.err.lower() or "overlap" in captured.err.lower()

        # 6. Overlapping: output is ancestor of input
        code = main(
            [
                "--input-root",
                "/tmp/ancestor/child/sub",
                "--output-dir",
                "/tmp/ancestor",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
            ],
            model_factory=mock_factory,
        )
        assert code == 1
        assert len(calls) == 0
        captured = capsys.readouterr()
        assert "ancestor" in captured.err.lower() or "overlap" in captured.err.lower()

    def test_cli_pipeline_success(self, capsys, monkeypatch):
        # Successful run prints stable JSON summary and returns 0
        pipeline_calls = []

        fake_result = make_fake_pipeline_result()

        def mock_run_pipeline(*args, **kwargs):
            pipeline_calls.append((args, kwargs))
            return fake_result

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            mock_run_pipeline,
        )

        calls = []

        def mock_factory(model_name, **kwargs):
            calls.append(model_name)
            return "fake-model"

        code = main(
            [
                "--input-root",
                "/tmp/in-root",
                "--output-dir",
                "/tmp/out-dir",
                "--input-dataset-revision",
                "rev-123",
                "--pipeline-version",
                "ver-456",
                "--batch-size",
                "64",
                "--sat-model",
                "my-sat-model",
                "--overwrite",
            ],
            model_factory=mock_factory,
        )

        assert code == 0
        assert len(pipeline_calls) == 1

        args, kwargs = pipeline_calls[0]
        assert kwargs["input_root"] == Path("/tmp/in-root")
        assert kwargs["output_dir"] == Path("/tmp/out-dir")
        assert kwargs["input_dataset_revision"] == "rev-123"
        assert kwargs["pipeline_version"] == "ver-456"
        assert kwargs["batch_size"] == 64
        assert kwargs["overwrite"] is True

        from osm_polygon_sentence_relevance.sat_adapter import SaTSentenceSegmenter

        segmenter = kwargs["segmenter"]
        assert isinstance(segmenter, SaTSentenceSegmenter)
        assert segmenter._model_name == "my-sat-model"

        # Assert stdout has the correct stable JSON format
        captured = capsys.readouterr()
        summary = json.loads(captured.out.strip())
        assert summary["parquet_path"] == "/tmp/out/sentences.parquet"
        assert summary["manifest_path"] == "/tmp/out/manifest.json"
        assert summary["processed_regions_count"] == 2
        assert summary["total_joined_section_occurrences"] == 15
        assert (
            summary["segmentation_report"]["wikipedia_sentence_occurrence_count"] == 5
        )
        assert summary["finalization_report"]["output_sentence_count"] == 6
        # Card path is part of the deterministic summary (Phase 8C
        # source-provenance completion).
        assert summary["card_path"] == "/tmp/out/README.md"

    def test_cli_local_pipeline_passes_no_dataset_id(self, capsys, monkeypatch):
        """Local mode (``--input-root``) must not invent a Hub dataset id;
        the value threaded to ``run_pipeline`` is ``None``.
        """
        pipeline_calls = []
        fake_result = make_fake_pipeline_result()

        def mock_run_pipeline(*args, **kwargs):
            pipeline_calls.append((args, kwargs))
            return fake_result

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            mock_run_pipeline,
        )

        def mock_factory(model_name, **kwargs):
            return "fake-model"

        code = main(
            [
                "--input-root",
                "/tmp/in-root",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
            ],
            model_factory=mock_factory,
        )
        assert code == 0
        args, kwargs = pipeline_calls[0]
        # ``input_dataset_id`` is not supplied at all in local mode.
        assert kwargs.get("input_dataset_id") is None

    def test_cli_hub_pipeline_threads_dataset_id_to_run_pipeline(
        self, capsys, monkeypatch
    ):
        """Hub mode threads the exact CLI ``--input-dataset-id`` value
        into ``run_pipeline(..., input_dataset_id=...)``. A stub
        acquisition supplies the snapshot.
        """
        from osm_polygon_sentence_relevance.ingestion.acquisition import (
            AcquisitionResult,
        )

        pipeline_calls = []
        fake_result = make_fake_pipeline_result()

        def mock_run_pipeline(*args, **kwargs):
            pipeline_calls.append((args, kwargs))
            return fake_result

        def mock_acquisition(dataset_id, requested_revision, **kwargs):
            return AcquisitionResult(
                dataset_id=dataset_id,
                requested_revision=requested_revision,
                resolved_sha="a" * 40,
                snapshot_path=Path("/tmp/snap"),
                discovered_region_count=0,
            )

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            mock_run_pipeline,
        )
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.acquisition.acquire_dataset_snapshot",
            mock_acquisition,
        )

        def mock_factory(model_name, **kwargs):
            return "fake-model"

        code = main(
            [
                "--input-dataset-id",
                "NoeFlandre/osm-polygon-wikidata-only",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "main",
                "--pipeline-version",
                "ver",
            ],
            model_factory=mock_factory,
        )
        assert code == 0
        args, kwargs = pipeline_calls[0]
        assert kwargs["input_dataset_id"] == ("NoeFlandre/osm-polygon-wikidata-only")
        captured = capsys.readouterr()
        summary = json.loads(captured.out.strip())
        assert summary["card_path"] == "/tmp/out/README.md"

    def test_cli_pipeline_failure(self, capsys, monkeypatch):
        # When pipeline fails, main returns non-zero, logs to stderr, and does not print JSON summary
        def mock_run_pipeline(*args, **kwargs):
            raise ExportError("Inconsistent pipeline_version values within rows")

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            mock_run_pipeline,
        )

        def mock_factory(model_name, **kwargs):
            return "fake-model"

        code = main(
            [
                "--input-root",
                "/tmp/in",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
            ],
            model_factory=mock_factory,
        )

        assert code == 1
        captured = capsys.readouterr()
        assert not captured.out.strip()
        assert "Inconsistent pipeline_version" in captured.err
