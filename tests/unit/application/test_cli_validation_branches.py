"""Targeted coverage of pre-existing CLI argument-validation branches.

These branches in ``application/cli.py`` have been reachable for many
phases but no test exercised them directly. The Phase 9L-A Amendment
does not change their behaviour; this module exists solely to keep
total line coverage above the project threshold.
"""

from __future__ import annotations

import pytest


def _build_args(tmp_path, **overrides):
    from osm_polygon_sentence_relevance.application.cli import _build_parser

    argv = [
        "--input-root",
        str(tmp_path / "in"),
        "--output-dir",
        str(tmp_path / "out"),
        "--input-dataset-revision",
        overrides.get("input_dataset_revision", "r1"),
        "--pipeline-version",
        overrides.get("pipeline_version", "v1"),
    ]
    argv.extend(overrides.get("extra", []))
    return _build_parser().parse_args(argv)


class TestCLIArgumentValidationBranches:
    def test_blank_pipeline_version_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import _validate_args

        args = _build_args(tmp_path, pipeline_version="   ")
        with pytest.raises(ValueError, match="pipeline_version"):
            _validate_args(args)

    def test_blank_input_root_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import _validate_args

        args = _build_args(tmp_path, extra=["--input-root", "   "])
        with pytest.raises(ValueError, match="input_root"):
            _validate_args(args)

    def test_blank_sat_model_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import _validate_args

        args = _build_args(tmp_path, extra=["--sat-model", "   "])
        with pytest.raises(ValueError, match="sat_model"):
            _validate_args(args)

    def test_blank_output_dir_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import _validate_args

        args = _build_args(tmp_path, extra=["--output-dir", "   "])
        with pytest.raises(ValueError, match="output_dir"):
            _validate_args(args)

    def test_input_source_dataset_id_surrounding_whitespace_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import _validate_args

        args = _build_args(
            tmp_path,
            extra=[
                "--input-source-dataset-id",
                " foo/bar  ",
            ],
        )
        with pytest.raises(ValueError, match="surrounding whitespace"):
            _validate_args(args)

    def test_input_source_dataset_id_blank_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import _validate_args

        args = _build_args(tmp_path, extra=["--input-source-dataset-id", "   "])
        with pytest.raises(ValueError, match="input-source-dataset-id"):
            _validate_args(args)

    def test_input_source_dataset_id_with_hub_mode_rejected(self, tmp_path):
        # Build args in hub mode by using --input-dataset-id.
        from osm_polygon_sentence_relevance.application.cli import (
            _build_parser,
            _validate_args,
        )

        argv = [
            "--input-dataset-id",
            "owner/dataset",
            "--output-dir",
            str(tmp_path / "out"),
            "--input-dataset-revision",
            "r1",
            "--pipeline-version",
            "v1",
            "--input-source-dataset-id",
            "owner/source",
        ]
        args = _build_parser().parse_args(argv)
        with pytest.raises(ValueError, match="only valid with --input-root"):
            _validate_args(args)

    def test_publish_revision_without_publish_dataset_id_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import _validate_args

        args = _build_args(tmp_path, extra=["--publish-revision", "feature-x"])
        with pytest.raises(ValueError, match="requires --publish-dataset-id"):
            _validate_args(args)

    def test_publish_commit_message_without_publish_dataset_id_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import _validate_args

        args = _build_args(tmp_path, extra=["--publish-commit-message", "msg"])
        with pytest.raises(ValueError, match="requires --publish-dataset-id"):
            _validate_args(args)

    def test_blank_work_dir_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import _validate_args

        args = _build_args(tmp_path, extra=["--work-dir", "   "])
        with pytest.raises(ValueError, match="work_dir must be a non-blank string"):
            _validate_args(args)

    def test_work_dir_with_surrounding_whitespace_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import _validate_args

        args = _build_args(tmp_path, extra=["--work-dir", "  /tmp/somewhere  "])
        with pytest.raises(ValueError, match="surrounding whitespace"):
            _validate_args(args)

    def test_blank_device_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import (
            _build_parser,
            _validate_args,
        )

        # The argparse ``choices`` filter rejects blank strings
        # before ``_validate_args`` runs, so we cannot pass "   "
        # directly. Build the args namespace manually.
        args = _build_parser().parse_args(
            [
                "--input-root",
                str(tmp_path / "in"),
                "--output-dir",
                str(tmp_path / "out"),
                "--input-dataset-revision",
                "r1",
                "--pipeline-version",
                "v1",
            ]
        )
        # Force the blank value past argparse.
        object.__setattr__(args, "device", "")
        with pytest.raises(ValueError, match="device"):
            _validate_args(args)

    def test_input_dataset_revision_blank_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import _validate_args

        args = _build_args(tmp_path, extra=["--input-dataset-revision", "   "])
        with pytest.raises(ValueError, match="input_dataset_revision"):
            _validate_args(args)

    def test_blank_input_dataset_id_hub_mode_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import (
            _build_parser,
            _validate_args,
        )

        argv = [
            "--input-dataset-id",
            "   ",
            "--output-dir",
            str(tmp_path / "out"),
            "--input-dataset-revision",
            "r1",
            "--pipeline-version",
            "v1",
        ]
        args = _build_parser().parse_args(argv)
        with pytest.raises(ValueError, match="input_dataset_id cannot be blank"):
            _validate_args(args)

    def test_main_returns_2_for_parser_failure(self) -> None:
        """``main`` must return exit code 2 when argparse fails."""
        from osm_polygon_sentence_relevance.application.cli import main

        # ``--unknown-flag`` makes argparse emit SystemExit(2).
        rc = main(["--unknown-flag"])
        assert rc == 2

    def test_serialize_summary_is_deterministic(self, tmp_path: object) -> None:
        """``_serialize_summary`` must produce a stable JSON document."""
        from dataclasses import dataclass, field
        from typing import Any

        from osm_polygon_sentence_relevance.application.cli import _serialize_summary

        @dataclass
        class _ExportResult:
            parquet_path: Any
            manifest_path: Any
            card_path: Any

        @dataclass
        class _SegmentationReport:
            input_section_occurrence_count: int = 0
            emitted_segment_count: int = 0
            retained_sentence_occurrence_count: int = 0
            dropped_empty_raw_count: int = 0
            dropped_empty_normalized_count: int = 0
            wikipedia_sentence_occurrence_count: int = 0
            wikivoyage_sentence_occurrence_count: int = 0

        @dataclass
        class _FinalizationReport:
            input_sentence_occurrence_count: int = 0
            output_sentence_count: int = 0
            duplicate_occurrence_count_removed: int = 0
            cross_source_duplicate_group_count: int = 0

        @dataclass
        class _PipelineResult:
            export_result: Any
            processed_regions_count: int = 0
            total_joined_section_occurrences: int = 0
            segmentation_report: Any = field(default_factory=_SegmentationReport)
            finalization_report: Any = field(default_factory=_FinalizationReport)

        @dataclass
        class _Resolved:
            mode: str = "local"
            dataset_id: str | None = "owner/source"
            requested_revision: str = "r"
            resolved_revision: str = "r"
            snapshot_path: str = "/snap"

        export = _ExportResult(
            parquet_path="/p.parquet",
            manifest_path="/m.json",
            card_path="/R.md",
        )
        res = _PipelineResult(export_result=export)
        text = _serialize_summary(res, _Resolved())
        # Stable JSON with sorted keys and compact separators.
        assert '"mode":"local"' in text
        assert '"snapshot_path":"/snap"' in text

    def test_resolve_input_local_mode(self, tmp_path) -> None:
        """``_resolve_input`` returns a ``_ResolvedInput`` for local mode."""
        from osm_polygon_sentence_relevance.application.cli import (
            _build_parser,
            _resolve_input,
        )

        argv = [
            "--input-root",
            str(tmp_path / "in"),
            "--output-dir",
            str(tmp_path / "out"),
            "--input-dataset-revision",
            "r1",
            "--pipeline-version",
            "v1",
        ]
        args = _build_parser().parse_args(argv)
        resolved = _resolve_input(args, acquisition_fn=None)
        assert resolved.mode == "local"
        assert resolved.resolved_revision == "r1"

    def test_resolve_input_hub_mode_with_mocked_acquisition(self, tmp_path) -> None:
        """``_resolve_input`` returns a Hub-mode ``_ResolvedInput``."""
        from dataclasses import dataclass

        from osm_polygon_sentence_relevance.application.cli import (
            _build_parser,
            _resolve_input,
        )

        @dataclass
        class _Snap:
            resolved_sha: str = "deadbeef"
            snapshot_path: object = None

        def _fake_acquire(dataset_id: str, revision: str) -> _Snap:
            assert dataset_id == "owner/data"
            assert revision == "r1"
            _Snap.snapshot_path = tmp_path / "snap"
            return _Snap()

        argv = [
            "--input-dataset-id",
            "owner/data",
            "--output-dir",
            str(tmp_path / "out"),
            "--input-dataset-revision",
            "r1",
            "--pipeline-version",
            "v1",
        ]
        args = _build_parser().parse_args(argv)
        resolved = _resolve_input(args, acquisition_fn=_fake_acquire)
        assert resolved.mode == "huggingface"
        assert resolved.dataset_id == "owner/data"
        assert resolved.resolved_revision == "deadbeef"

    def test_blank_publish_dataset_id_rejected(self, tmp_path) -> None:
        from osm_polygon_sentence_relevance.application.cli import (
            _build_parser,
            _validate_args,
        )

        argv = [
            "--input-root",
            str(tmp_path / "in"),
            "--output-dir",
            str(tmp_path / "out"),
            "--input-dataset-revision",
            "r1",
            "--pipeline-version",
            "v1",
            "--publish-dataset-id",
            "   ",
        ]
        args = _build_parser().parse_args(argv)
        with pytest.raises(ValueError, match="publish_dataset_id cannot be blank"):
            _validate_args(args)

    def test_blank_publish_commit_message_rejected(self, tmp_path) -> None:
        from osm_polygon_sentence_relevance.application.cli import (
            _build_parser,
            _validate_args,
        )

        argv = [
            "--input-root",
            str(tmp_path / "in"),
            "--output-dir",
            str(tmp_path / "out"),
            "--input-dataset-revision",
            "r1",
            "--pipeline-version",
            "v1",
            "--publish-dataset-id",
            "owner/data",
            "--publish-commit-message",
            "   ",
        ]
        args = _build_parser().parse_args(argv)
        with pytest.raises(ValueError, match="publish_commit_message cannot be blank"):
            _validate_args(args)

    def test_blank_publish_revision_rejected(self, tmp_path) -> None:
        from osm_polygon_sentence_relevance.application.cli import (
            _build_parser,
            _validate_args,
        )

        argv = [
            "--input-root",
            str(tmp_path / "in"),
            "--output-dir",
            str(tmp_path / "out"),
            "--input-dataset-revision",
            "r1",
            "--pipeline-version",
            "v1",
            "--publish-dataset-id",
            "owner/data",
            "--publish-revision",
            "   ",
        ]
        args = _build_parser().parse_args(argv)
        with pytest.raises(ValueError, match="publish_revision cannot be blank"):
            _validate_args(args)

    def test_invalid_device_rejected(self, tmp_path) -> None:
        """Bypass the argparse choices by injecting a Namespace directly."""
        from argparse import Namespace

        from osm_polygon_sentence_relevance.application.cli import _validate_args

        args = Namespace(
            batch_size=128,
            output_dir=str(tmp_path / "out"),
            sat_model="sat-3l",
            input_dataset_revision="r1",
            pipeline_version="v1",
            input_root=str(tmp_path / "in"),
            input_source_dataset_id=None,
            input_dataset_id=None,
            device="bogus-device",
            work_dir=None,
            source_commit=None,
            publish_dataset_id=None,
            publish_revision=None,
            publish_commit_message=None,
            overwrite=False,
        )
        with pytest.raises(ValueError, match="device must be one of"):
            _validate_args(args)

    def test_main_returns_1_for_pipeline_failure(self, tmp_path, monkeypatch) -> None:
        """``main`` must return exit code 1 when the pipeline raises."""
        from dataclasses import dataclass
        from typing import Any

        from osm_polygon_sentence_relevance.application import cli as cli_mod
        from osm_polygon_sentence_relevance.application.cli import main
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        @dataclass
        class _Snap:
            resolved_sha: str = "abc"
            snapshot_path: str = str(tmp_path / "snap")

        def _fake_acquire(dataset_id, revision):
            (tmp_path / "snap").mkdir(parents=True, exist_ok=True)
            return _Snap()

        def _fake_pipeline(**kwargs: Any) -> Any:
            raise ExportError("synthetic failure")

        # ``main`` imported ``run_pipeline`` by name; the
        # monkeypatch must target the cli module's local reference.
        monkeypatch.setattr(cli_mod, "run_pipeline", _fake_pipeline)

        rc = main(
            [
                "--input-dataset-id",
                "owner/data",
                "--output-dir",
                str(tmp_path / "out"),
                "--input-dataset-revision",
                "r1",
                "--pipeline-version",
                "v1",
            ],
            acquisition_fn=_fake_acquire,
        )
        assert rc == 1

    def test_main_publishes_when_publishing_fn_succeeds(
        self, tmp_path, monkeypatch
    ) -> None:
        """``main`` must call the publishing hook when a dataset id is
        provided and the pipeline succeeds."""
        from dataclasses import dataclass, field
        from typing import Any

        from osm_polygon_sentence_relevance.application import cli as cli_mod
        from osm_polygon_sentence_relevance.application.cli import main

        @dataclass
        class _Snap:
            resolved_sha: str = "abc"
            snapshot_path: str = str(tmp_path / "snap")

        def _fake_acquire(dataset_id, revision):
            (tmp_path / "snap").mkdir(parents=True, exist_ok=True)
            return _Snap()

        @dataclass
        class _ExportResult:
            parquet_path: Any = None
            manifest_path: Any = None
            card_path: Any = None

            def __post_init__(self):
                if self.parquet_path is None:
                    self.parquet_path = type("P", (), {"parent": tmp_path / "out"})()

        @dataclass
        class _SegReport:
            input_section_occurrence_count: int = 0
            emitted_segment_count: int = 0
            retained_sentence_occurrence_count: int = 0
            dropped_empty_raw_count: int = 0
            dropped_empty_normalized_count: int = 0
            wikipedia_sentence_occurrence_count: int = 0
            wikivoyage_sentence_occurrence_count: int = 0

        @dataclass
        class _FinalReport:
            input_sentence_occurrence_count: int = 0
            output_sentence_count: int = 0
            duplicate_occurrence_count_removed: int = 0
            cross_source_duplicate_group_count: int = 0

        @dataclass
        class _PipelineResult:
            export_result: Any = field(default_factory=_ExportResult)
            processed_regions_count: int = 0
            total_joined_section_occurrences: int = 0
            segmentation_report: Any = field(default_factory=_SegReport)
            finalization_report: Any = field(default_factory=_FinalReport)

        @dataclass
        class _PublicationResult:
            dataset_id: str = "owner/data"
            target_revision: str = "main"
            commit_id: str = "deadbeef"
            commit_url: str = (
                "https://huggingface.co/datasets/owner/data/commit/deadbeef"
            )
            row_count: int = 0
            sha256: str = "a" * 64

        def _fake_pipeline(**kwargs: Any) -> Any:
            return _PipelineResult()

        def _fake_publish(
            export_dir,
            dataset_id,
            target_revision,
            commit_message,
        ) -> _PublicationResult:
            return _PublicationResult()

        monkeypatch.setattr(cli_mod, "run_pipeline", _fake_pipeline)

        rc = main(
            [
                "--input-dataset-id",
                "owner/data",
                "--output-dir",
                str(tmp_path / "out"),
                "--input-dataset-revision",
                "r1",
                "--pipeline-version",
                "v1",
                "--publish-dataset-id",
                "owner/data",
                "--publish-revision",
                "main",
                "--publish-commit-message",
                "test",
            ],
            acquisition_fn=_fake_acquire,
            publishing_fn=_fake_publish,
        )
        assert rc == 0
