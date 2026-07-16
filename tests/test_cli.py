from __future__ import annotations

import json
from pathlib import Path

from osm_polygon_sentence_relevance.acquisition import AcquisitionResult
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
            "osm_polygon_sentence_relevance.cli.run_pipeline", mock_run_pipeline
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

    def test_cli_pipeline_failure(self, capsys, monkeypatch):
        # When pipeline fails, main returns non-zero, logs to stderr, and does not print JSON summary
        def mock_run_pipeline(*args, **kwargs):
            raise ExportError("Inconsistent pipeline_version values within rows")

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.cli.run_pipeline", mock_run_pipeline
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


# ===================================================================
# Test Suite for CLI Hugging Face input integration (Phase 6C)
# ===================================================================

SHA1 = "a" * 40


class TestCLIHubInput:
    def _fake_result(self):
        return make_fake_pipeline_result()

    def test_local_mode_unchanged(self, capsys, monkeypatch):
        # Existing local mode still passes the supplied revision unchanged.
        pipeline_calls = []

        def mock_run_pipeline(*args, **kwargs):
            pipeline_calls.append(kwargs)
            return self._fake_result()

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.cli.run_pipeline", mock_run_pipeline
        )

        acquisition_calls = []

        def mock_acquire(dataset_id, requested_revision, **kwargs):
            acquisition_calls.append((dataset_id, requested_revision))
            raise AssertionError("acquisition must not be called in local mode")

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
            ],
            acquisition_fn=mock_acquire,
        )

        assert code == 0
        assert len(pipeline_calls) == 1
        assert pipeline_calls[0]["input_dataset_revision"] == "rev-123"
        assert pipeline_calls[0]["input_root"] == Path("/tmp/in-root")

        captured = capsys.readouterr()
        summary = json.loads(captured.out.strip())
        assert summary["input"]["mode"] == "local"
        assert summary["input"]["dataset_id"] is None
        assert summary["input"]["requested_revision"] == "rev-123"
        assert summary["input"]["resolved_revision"] == "rev-123"
        assert summary["input"]["snapshot_path"] == "/tmp/in-root"

    def test_hub_main_revision_resolves_to_sha(self, capsys, monkeypatch):
        # Hub mode: acquire_dataset_snapshot is called, its snapshot_path is
        # used as input_root, and its resolved SHA is forwarded (not "main").
        pipeline_calls = []

        def mock_run_pipeline(*args, **kwargs):
            pipeline_calls.append(kwargs)
            return self._fake_result()

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.cli.run_pipeline", mock_run_pipeline
        )

        acquisition_calls = []

        def mock_acquire(dataset_id, requested_revision, **kwargs):
            acquisition_calls.append((dataset_id, requested_revision))
            return AcquisitionResult(
                dataset_id=dataset_id,
                requested_revision=requested_revision,
                resolved_sha=SHA1,
                snapshot_path=Path("/snapshot/main"),
                discovered_region_count=1,
            )

        code = main(
            [
                "--input-dataset-id",
                "NoeFlandre/osm-polygon-wikidata-only",
                "--output-dir",
                "/tmp/out-dir",
                "--input-dataset-revision",
                "main",
                "--pipeline-version",
                "ver-456",
            ],
            acquisition_fn=mock_acquire,
        )

        assert code == 0
        assert acquisition_calls == [("NoeFlandre/osm-polygon-wikidata-only", "main")]
        assert len(pipeline_calls) == 1
        assert pipeline_calls[0]["input_root"] == Path("/snapshot/main")
        assert pipeline_calls[0]["input_dataset_revision"] == SHA1

        captured = capsys.readouterr()
        summary = json.loads(captured.out.strip())
        assert summary["input"]["mode"] == "huggingface"
        assert summary["input"]["dataset_id"] == "NoeFlandre/osm-polygon-wikidata-only"
        assert summary["input"]["requested_revision"] == "main"
        assert summary["input"]["resolved_revision"] == SHA1
        assert summary["input"]["snapshot_path"] == "/snapshot/main"

    def test_hub_explicit_sha_mode(self, capsys, monkeypatch):
        # Explicit SHA requested revision is preserved and forwarded unchanged.
        pipeline_calls = []

        def mock_run_pipeline(*args, **kwargs):
            pipeline_calls.append(kwargs)
            return self._fake_result()

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.cli.run_pipeline", mock_run_pipeline
        )

        def mock_acquire(dataset_id, requested_revision, **kwargs):
            return AcquisitionResult(
                dataset_id=dataset_id,
                requested_revision=requested_revision,
                resolved_sha=SHA1,
                snapshot_path=Path("/snapshot/sha"),
                discovered_region_count=1,
            )

        code = main(
            [
                "--input-dataset-id",
                "NoeFlandre/osm-polygon-wikidata-only",
                "--output-dir",
                "/tmp/out-dir",
                "--input-dataset-revision",
                SHA1,
                "--pipeline-version",
                "ver-456",
            ],
            acquisition_fn=mock_acquire,
        )

        assert code == 0
        assert pipeline_calls[0]["input_dataset_revision"] == SHA1

        captured = capsys.readouterr()
        summary = json.loads(captured.out.strip())
        assert summary["input"]["requested_revision"] == SHA1
        assert summary["input"]["resolved_revision"] == SHA1

    def test_both_input_modes_rejected(self, capsys):
        # Supplying both --input-root and --input-dataset-id is rejected.
        code = main(
            [
                "--input-root",
                "/tmp/in",
                "--input-dataset-id",
                "NoeFlandre/osm-polygon-wikidata-only",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "main",
                "--pipeline-version",
                "ver",
            ]
        )
        assert code == 2
        captured = capsys.readouterr()
        assert (
            "mutually exclusive" in captured.err.lower()
            or "not allowed" in captured.err.lower()
        )

    def test_missing_both_input_modes_rejected(self, capsys):
        # Supplying neither input mode is rejected at argument validation.
        code = main(
            [
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "main",
                "--pipeline-version",
                "ver",
            ]
        )
        assert code == 2
        captured = capsys.readouterr()
        assert (
            "input-root" in captured.err.lower()
            or "input-dataset-id" in captured.err.lower()
        )

    def test_acquisition_failure_nonzero_no_json_no_segmenter(
        self, capsys, monkeypatch
    ):
        # Acquisition failure: non-zero, concise stderr, no success JSON,
        # and the segmenter is never constructed.
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.cli.run_pipeline",
            lambda *a, **k: self._fake_result(),
        )

        segmenter_calls = []

        def mock_factory(model_name, **kwargs):
            segmenter_calls.append(model_name)
            return "fake-model"

        def mock_acquire(dataset_id, requested_revision, **kwargs):
            from osm_polygon_sentence_relevance.errors import AcquisitionError

            raise AcquisitionError("Failed to resolve revision")

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
            acquisition_fn=mock_acquire,
            model_factory=mock_factory,
        )

        assert code == 1
        assert len(segmenter_calls) == 0
        captured = capsys.readouterr()
        assert not captured.out.strip()
        assert "Failed to resolve revision" in captured.err

    def test_invalid_arguments_no_acquisition_no_model(self, capsys, monkeypatch):
        # Invalid arguments invoke neither acquisition nor model construction.
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.cli.run_pipeline",
            lambda *a, **k: self._fake_result(),
        )

        acquisition_calls = []

        def mock_acquire(dataset_id, requested_revision, **kwargs):
            acquisition_calls.append((dataset_id, requested_revision))
            raise AssertionError("acquisition must not run on invalid args")

        segmenter_calls = []

        def mock_factory(model_name, **kwargs):
            segmenter_calls.append(model_name)
            return "fake-model"

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
            acquisition_fn=mock_acquire,
            model_factory=mock_factory,
        )

        assert code == 1
        assert len(acquisition_calls) == 0
        assert len(segmenter_calls) == 0
        captured = capsys.readouterr()
        assert "revision cannot be blank" in captured.err
