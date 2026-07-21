"""Hub-mode input tests for the local build CLI (the implementation)."""

import json
from pathlib import Path

from osm_polygon_sentence_relevance.acquisition import AcquisitionResult
from osm_polygon_sentence_relevance.cli import main
from tests.helpers import make_fake_pipeline_result

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
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            mock_run_pipeline,
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
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            mock_run_pipeline,
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
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            mock_run_pipeline,
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
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
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
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
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

    def test_cli_hub_rejects_surrounding_whitespace_dataset_id(
        self, capsys, monkeypatch
    ):
        """Surrounding whitespace on ``--input-dataset-id`` is rejected
        before any of the side-effecting hooks (acquisition_fn,
        model_factory, run_pipeline, publishing_fn) is invoked.
        """
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            lambda *a, **k: self._fake_result(),
        )

        acquisition_calls = []

        def mock_acquire(dataset_id, requested_revision, **kwargs):
            acquisition_calls.append((dataset_id, requested_revision))
            raise AssertionError(
                "acquisition must not run on a surrounding-whitespace dataset id"
            )

        segmenter_calls = []

        def mock_factory(model_name, **kwargs):
            segmenter_calls.append(model_name)
            return "fake-model"

        publishing_calls = []

        def mock_publishing(*args, **kwargs):
            publishing_calls.append((args, kwargs))
            raise AssertionError(
                "publishing must not run on a surrounding-whitespace dataset id"
            )

        code = main(
            [
                "--input-dataset-id",
                "  NoeFlandre/osm-polygon-wikidata-only  ",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "main",
                "--pipeline-version",
                "ver",
            ],
            acquisition_fn=mock_acquire,
            model_factory=mock_factory,
            publishing_fn=mock_publishing,
        )
        assert code == 1
        assert len(acquisition_calls) == 0
        assert len(segmenter_calls) == 0
        assert len(publishing_calls) == 0
        captured = capsys.readouterr()
        assert "input_dataset_id" in captured.err
        assert (
            "surrounding whitespace" in captured.err.lower()
            or "must be" in captured.err.lower()
        )
