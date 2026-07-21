"""CLI optional publishing tests (Phase 7D).

These exercise the optional ``--publish-*`` flags on the build CLI using
injected pipeline and publishing functions. No network calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osm_polygon_sentence_relevance.application.cli import (
    _ResolvedInput,
    _serialize_summary,
)
from osm_polygon_sentence_relevance.cli import main
from osm_polygon_sentence_relevance.errors import PublicationError
from tests.helpers import make_fake_pipeline_result


def _success_args(tmp_root: str) -> list[str]:
    return [
        "--input-root",
        tmp_root,
        "--output-dir",
        "/tmp/out",
        "--input-dataset-revision",
        "rev",
        "--pipeline-version",
        "ver",
    ]


class _RecordingPublisher:
    """Injected publisher that records its single invocation."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(
        self, export_dir, dataset_id, *, target_revision="main", commit_message=None
    ):
        self.calls.append(
            {
                "export_dir": export_dir,
                "dataset_id": dataset_id,
                "target_revision": target_revision,
                "commit_message": commit_message,
            }
        )
        return _make_publication_result(dataset_id, target_revision)


def _make_publication_result(dataset_id: str, revision: str) -> object:
    from osm_polygon_sentence_relevance.publishing import PublicationResult

    return PublicationResult(
        dataset_id=dataset_id,
        target_revision=revision,
        commit_id="oid-abc123" * 4,
        commit_url="https://huggingface.co/datasets/owner/dataset/commit/abcdef",
        row_count=6,
        sha256="a" * 64,
    )


# ---------------------------------------------------------------------------
# Backward compatibility: no publishing => no publisher calls
# ---------------------------------------------------------------------------


class TestBackwardCompatibilityNoPublishing:
    def test_local_build_without_publishing_never_calls_publisher(
        self, monkeypatch, capsys
    ):
        publisher = _RecordingPublisher()
        pipeline_calls: list[tuple] = []

        fake = make_fake_pipeline_result(
            parquet_path=Path("/tmp/out/sentences.parquet"),
            manifest_path=Path("/tmp/out/manifest.json"),
        )

        def mock_run_pipeline(*args, **kwargs):
            pipeline_calls.append((args, kwargs))
            return fake

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            mock_run_pipeline,
        )

        code = main(
            _success_args("/tmp/in"),
            model_factory=lambda model_name, **kw: "fake",
            publishing_fn=publisher,
        )
        assert code == 0
        assert len(publisher.calls) == 0
        assert len(pipeline_calls) == 1
        captured = capsys.readouterr()
        summary = json.loads(captured.out.strip())
        assert "publication" not in summary

    def test_hub_build_without_publishing_never_calls_publisher(
        self, monkeypatch, capsys
    ):
        publisher = _RecordingPublisher()
        pipeline_calls: list[tuple] = []

        fake = make_fake_pipeline_result(
            parquet_path=Path("/tmp/out/sentences.parquet"),
            manifest_path=Path("/tmp/out/manifest.json"),
        )

        class _Snap:
            resolved_sha = "deadbeef"
            snapshot_path = Path("/tmp/in")

        def mock_acquire(*args, **kwargs):
            return _Snap()

        def mock_run_pipeline(*args, **kwargs):
            pipeline_calls.append((args, kwargs))
            return fake

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            mock_run_pipeline,
        )

        code = main(
            [
                "--input-dataset-id",
                "owner/source",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "main",
                "--pipeline-version",
                "ver",
            ],
            model_factory=lambda model_name, **kw: "fake",
            acquisition_fn=mock_acquire,
            publishing_fn=publisher,
        )
        assert code == 0
        assert len(publisher.calls) == 0
        assert len(pipeline_calls) == 1
        captured = capsys.readouterr()
        summary = json.loads(captured.out.strip())
        assert "publication" not in summary

    def test_non_publishing_stdout_is_byte_for_byte_unchanged(
        self, monkeypatch, capsys
    ):
        """The success stdout for a non-publishing invocation must be
        byte-for-byte identical to the pre-Phase-7D characterized value
        (no ``publication`` key, no other drift)."""
        pipeline_calls: list[tuple] = []

        fake = make_fake_pipeline_result(
            parquet_path=Path("/tmp/out/sentences.parquet"),
            manifest_path=Path("/tmp/out/manifest.json"),
        )

        def mock_run_pipeline(*args, **kwargs):
            pipeline_calls.append((args, kwargs))
            return fake

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            mock_run_pipeline,
        )

        code = main(
            _success_args("/tmp/in"),
            model_factory=lambda model_name, **kw: "fake",
        )
        assert code == 0
        captured = capsys.readouterr()
        resolved_input = _ResolvedInput(
            mode="local",
            dataset_id=None,
            requested_revision="rev",
            resolved_revision="rev",
            snapshot_path=str(Path("/tmp/in")),
        )
        expected = _serialize_summary(fake, resolved_input) + "\n"
        assert captured.out == expected
        # Re-confirm via json equivalence as a secondary sanity check.
        assert json.loads(captured.out.strip()) == json.loads(expected.strip())


# ---------------------------------------------------------------------------
# Successful build + publishing
# ---------------------------------------------------------------------------


class TestPublishingSuccess:
    def test_publish_called_once_after_pipeline(self, monkeypatch, capsys):
        publisher = _RecordingPublisher()
        order: list[str] = []

        fake = make_fake_pipeline_result(
            parquet_path=Path("/tmp/out/sentences.parquet"),
            manifest_path=Path("/tmp/out/manifest.json"),
        )

        def mock_run_pipeline(*args, **kwargs):
            order.append("pipeline")
            return fake

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            mock_run_pipeline,
        )

        def recording_publisher(*args, **kwargs):
            order.append("publish")
            return publisher(*args, **kwargs)

        code = main(
            [*_success_args("/tmp/in"), "--publish-dataset-id", "owner/dataset"],
            model_factory=lambda model_name, **kw: "fake",
            publishing_fn=recording_publisher,
        )
        assert code == 0
        assert order == ["pipeline", "publish"]
        assert len(publisher.calls) == 1

    def test_publisher_receives_actual_export_dir_and_default_revision(
        self, monkeypatch, capsys
    ):
        publisher = _RecordingPublisher()
        fake = make_fake_pipeline_result(
            parquet_path=Path("/tmp/out/sentences.parquet"),
            manifest_path=Path("/tmp/out/manifest.json"),
        )

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            lambda *a, **k: fake,
        )

        code = main(
            [*_success_args("/tmp/in"), "--publish-dataset-id", "owner/dataset"],
            model_factory=lambda model_name, **kw: "fake",
            publishing_fn=publisher,
        )
        assert code == 0
        call = publisher.calls[0]
        # export dir derived from PipelineResult ExportResult, not raw CLI arg
        assert str(call["export_dir"]) == str(Path("/tmp/out"))
        assert call["dataset_id"] == "owner/dataset"
        assert call["target_revision"] == "main"
        assert call["commit_message"] is None

    def test_explicit_revision_and_message_forwarded_verbatim(
        self, monkeypatch, capsys
    ):
        publisher = _RecordingPublisher()
        fake = make_fake_pipeline_result(
            parquet_path=Path("/tmp/out/sentences.parquet"),
            manifest_path=Path("/tmp/out/manifest.json"),
        )
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            lambda *a, **k: fake,
        )

        code = main(
            [
                *_success_args("/tmp/in"),
                "--publish-dataset-id",
                "owner/dataset",
                "--publish-revision",
                "my-branch",
                "--publish-commit-message",
                "ship it",
            ],
            model_factory=lambda model_name, **kw: "fake",
            publishing_fn=publisher,
        )
        assert code == 0
        call = publisher.calls[0]
        assert call["target_revision"] == "my-branch"
        assert call["commit_message"] == "ship it"

    def test_success_summary_contains_publication_object(self, monkeypatch, capsys):
        publisher = _RecordingPublisher()
        fake = make_fake_pipeline_result(
            parquet_path=Path("/tmp/out/sentences.parquet"),
            manifest_path=Path("/tmp/out/manifest.json"),
        )
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            lambda *a, **k: fake,
        )

        code = main(
            [*_success_args("/tmp/in"), "--publish-dataset-id", "owner/dataset"],
            model_factory=lambda model_name, **kw: "fake",
            publishing_fn=publisher,
        )
        assert code == 0
        captured = capsys.readouterr()
        summary = json.loads(captured.out.strip())
        publication = summary["publication"]
        assert publication["dataset_id"] == "owner/dataset"
        assert publication["target_revision"] == "main"
        assert publication["commit_id"] == "oid-abc123" * 4
        assert publication["commit_url"].endswith("/commit/abcdef")
        assert publication["row_count"] == 6
        assert publication["sha256"] == "a" * 64


# ---------------------------------------------------------------------------
# Invalid / blank publishing arguments
# ---------------------------------------------------------------------------


class TestInvalidPublishingArguments:
    def _assert_rejected_before_pipeline(self, args, monkeypatch, capsys):
        publisher = _RecordingPublisher()
        pipeline_calls: list[tuple] = []

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            lambda *a, **k: (
                pipeline_calls.append((a, k)) or make_fake_pipeline_result()
            ),
        )

        code = main(
            args,
            model_factory=lambda model_name, **kw: "fake",
            publishing_fn=publisher,
        )
        assert code == 1
        assert len(publisher.calls) == 0
        assert len(pipeline_calls) == 0
        captured = capsys.readouterr()
        assert not captured.out.strip()

    def test_blank_publish_dataset_id_rejected(self, monkeypatch, capsys):
        self._assert_rejected_before_pipeline(
            [*_success_args("/tmp/in"), "--publish-dataset-id", "   "],
            monkeypatch,
            capsys,
        )

    def test_blank_publish_revision_rejected(self, monkeypatch, capsys):
        self._assert_rejected_before_pipeline(
            [
                *_success_args("/tmp/in"),
                "--publish-dataset-id",
                "owner/dataset",
                "--publish-revision",
                "   ",
            ],
            monkeypatch,
            capsys,
        )

    def test_blank_publish_commit_message_rejected(self, monkeypatch, capsys):
        self._assert_rejected_before_pipeline(
            [
                *_success_args("/tmp/in"),
                "--publish-dataset-id",
                "owner/dataset",
                "--publish-commit-message",
                "   ",
            ],
            monkeypatch,
            capsys,
        )

    def test_revision_without_dataset_id_rejected(self, monkeypatch, capsys):
        self._assert_rejected_before_pipeline(
            [*_success_args("/tmp/in"), "--publish-revision", "my-branch"],
            monkeypatch,
            capsys,
        )

    def test_commit_message_without_dataset_id_rejected(self, monkeypatch, capsys):
        self._assert_rejected_before_pipeline(
            [*_success_args("/tmp/in"), "--publish-commit-message", "ship"],
            monkeypatch,
            capsys,
        )


# ---------------------------------------------------------------------------
# Invalid publishing args fail before every expensive boundary (Hub mode)
# ---------------------------------------------------------------------------


class TestInvalidPublishingBoundariesHubMode:
    """All four expensive boundaries (acquisition, model construction,
    pipeline, publisher) must be skipped for invalid publishing args,
    even in Hub input mode."""

    @pytest.mark.parametrize(
        "bad_args",
        [
            ["--publish-dataset-id", "   "],
            [
                "--publish-dataset-id",
                "owner/dataset",
                "--publish-revision",
                "   ",
            ],
            [
                "--publish-dataset-id",
                "owner/dataset",
                "--publish-commit-message",
                "   ",
            ],
            ["--publish-revision", "my-branch"],
            ["--publish-commit-message", "ship"],
        ],
        ids=[
            "blank-dataset-id",
            "blank-revision",
            "blank-commit-message",
            "revision-without-dataset-id",
            "commit-message-without-dataset-id",
        ],
    )
    def test_invalid_publishing_args_skip_all_boundaries(
        self, bad_args, monkeypatch, capsys
    ):
        from osm_polygon_sentence_relevance.errors import ExportError

        acquisition_calls: list[object] = []
        model_calls: list[object] = []
        pipeline_calls: list[object] = []
        publisher = _RecordingPublisher()

        def mock_acquire(*args, **kwargs):
            acquisition_calls.append((args, kwargs))
            raise AssertionError("acquisition must not run for invalid args")

        def mock_factory(model_name, **kwargs):
            model_calls.append((model_name, kwargs))
            raise AssertionError("model construction must not run for invalid args")

        def mock_run_pipeline(*args, **kwargs):
            pipeline_calls.append((args, kwargs))
            raise ExportError("pipeline must not run for invalid args")

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.ingestion.acquisition.acquire_dataset_snapshot",
            mock_acquire,
        )
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            mock_run_pipeline,
        )

        code = main(
            [
                "--input-dataset-id",
                "owner/source",
                "--output-dir",
                "/tmp/out",
                "--input-dataset-revision",
                "main",
                "--pipeline-version",
                "ver",
                *bad_args,
            ],
            model_factory=mock_factory,
            acquisition_fn=mock_acquire,
            publishing_fn=publisher,
        )
        assert code == 1
        assert acquisition_calls == []
        assert model_calls == []
        assert pipeline_calls == []
        assert publisher.calls == []
        captured = capsys.readouterr()
        assert not captured.out.strip()


# ---------------------------------------------------------------------------
# Pipeline failure => zero publishing calls
# ---------------------------------------------------------------------------


class TestPipelineFailureNoPublish:
    def test_pipeline_error_causes_no_publishing(self, monkeypatch, capsys):
        from osm_polygon_sentence_relevance.errors import ExportError

        publisher = _RecordingPublisher()

        def mock_run_pipeline(*args, **kwargs):
            raise ExportError("boom")

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            mock_run_pipeline,
        )

        code = main(
            [*_success_args("/tmp/in"), "--publish-dataset-id", "owner/dataset"],
            model_factory=lambda model_name, **kw: "fake",
            publishing_fn=publisher,
        )
        assert code == 1
        assert len(publisher.calls) == 0
        captured = capsys.readouterr()
        assert not captured.out.strip()
        assert "boom" in captured.err


# ---------------------------------------------------------------------------
# Publishing failure => exit 1, stderr, export preserved
# ---------------------------------------------------------------------------


class TestPublishingFailure:
    def test_publishing_error_exits_1_with_stderr(self, monkeypatch, capsys):
        def failing_publisher(
            export_dir, dataset_id, *, target_revision="main", commit_message=None
        ):
            raise PublicationError("publish failed")

        fake = make_fake_pipeline_result(
            parquet_path=Path("/tmp/out/sentences.parquet"),
            manifest_path=Path("/tmp/out/manifest.json"),
        )
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            lambda *a, **k: fake,
        )

        code = main(
            [*_success_args("/tmp/in"), "--publish-dataset-id", "owner/dataset"],
            model_factory=lambda model_name, **kw: "fake",
            publishing_fn=failing_publisher,
        )
        assert code == 1
        captured = capsys.readouterr()
        assert not captured.out.strip()
        assert "publish failed" in captured.err

    def test_publishing_failure_preserves_export_dir(
        self, monkeypatch, capsys, tmp_path
    ):
        def failing_publisher(
            export_dir, dataset_id, *, target_revision="main", commit_message=None
        ):
            # Simulate that the local export still exists.
            assert Path(export_dir).exists()
            raise PublicationError("publish failed")

        export_dir = tmp_path / "out"
        export_dir.mkdir()
        parquet = export_dir / "sentences.parquet"
        parquet.write_text("data")
        manifest = export_dir / "manifest.json"
        manifest.write_text("{}")

        fake = make_fake_pipeline_result(
            parquet_path=export_dir / "sentences.parquet",
            manifest_path=export_dir / "manifest.json",
        )
        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.cli.run_pipeline",
            lambda *a, **k: fake,
        )

        code = main(
            [
                "--input-root",
                str(tmp_path / "in"),
                "--output-dir",
                str(export_dir),
                "--input-dataset-revision",
                "rev",
                "--pipeline-version",
                "ver",
                "--publish-dataset-id",
                "owner/dataset",
            ],
            model_factory=lambda model_name, **kw: "fake",
            publishing_fn=failing_publisher,
        )
        assert code == 1
        # The completed local export must remain byte-for-byte intact.
        assert parquet.exists()
        assert manifest.exists()
        assert parquet.read_text() == "data"
        assert manifest.read_text() == "{}"
        assert parquet.read_bytes() == b"data"
        assert manifest.read_bytes() == b"{}"


# ---------------------------------------------------------------------------
# Help + absence of token flag
# ---------------------------------------------------------------------------


class TestHelpAndNoTokenFlag:
    def test_help_documents_new_flags(self, capsys):
        code = main(["--help"])
        assert code == 0
        captured = capsys.readouterr()
        text = captured.out
        assert "--publish-dataset-id" in text
        assert "--publish-revision" in text
        assert "--publish-commit-message" in text

    def test_help_has_no_token_flag(self, capsys):
        code = main(["--help"])
        assert code == 0
        captured = capsys.readouterr()
        assert "--token" not in captured.out
        assert "--hf-token" not in captured.out
        assert "token" not in captured.out.lower()
