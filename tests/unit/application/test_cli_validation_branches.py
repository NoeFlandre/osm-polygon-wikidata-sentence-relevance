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
