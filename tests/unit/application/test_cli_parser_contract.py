"""Exact contract tests for the dataset orchestrator CLI parser."""

from __future__ import annotations

import argparse

import osm_polygon_sentence_relevance.application.cli as cli


def _action(parser: argparse.ArgumentParser, option: str) -> argparse.Action:
    return next(action for action in parser._actions if option in action.option_strings)


def test_parser_preserves_public_arguments_defaults_and_help() -> None:
    parser = cli._build_parser()

    assert (
        parser.description
        == "Deterministic OSM Polygon Sentence Relevance Dataset Orchestrator"
    )
    assert parser._mutually_exclusive_groups[0].required is True

    expected = {
        "--input-root": {
            "help": "Existing local input snapshot root directory",
            "default": None,
            "required": False,
            "type": None,
        },
        "--input-dataset-id": {
            "help": "Upstream Hugging Face dataset ID to acquire read-only snapshot from",
            "default": None,
            "required": False,
            "type": None,
        },
        "--output-dir": {
            "help": "Output directory",
            "default": None,
            "required": True,
            "type": None,
        },
        "--input-dataset-revision": {
            "help": "Input dataset revision",
            "default": None,
            "required": True,
            "type": None,
        },
        "--pipeline-version": {
            "help": "Pipeline version",
            "default": None,
            "required": True,
            "type": None,
        },
        "--batch-size": {
            "help": "Batch size for segmenter",
            "default": 128,
            "required": False,
            "type": int,
        },
        "--sat-model": {
            "help": "wtpsplit SaT model name",
            "default": "sat-12l-sm",
            "required": False,
            "type": None,
        },
        "--device": {
            "help": (
                "Accelerator for SaT inference. ``auto`` (default) prefers "
                "CUDA, then MPS, then CPU. Explicit ``cuda``/``mps`` fail "
                "when the backend is unavailable."
            ),
            "default": "auto",
            "required": False,
            "type": None,
        },
        "--input-source-dataset-id": {
            "help": (
                "Optional Hugging Face dataset ID of the upstream source for "
                "a local input snapshot. Only valid with --input-root; it "
                "populates the source provenance recorded in the manifest and "
                "dataset card without triggering any network request."
            ),
            "default": None,
            "required": False,
            "type": None,
        },
        "--overwrite": {
            "help": "Overwrite existing output directory",
            "default": False,
            "required": False,
            "type": None,
        },
        "--work-dir": {
            "help": (
                "Optional persistent work directory for shard-level "
                "checkpoints and a factual progress heartbeat. When "
                "supplied, the pipeline publishes one checkpoint per "
                "shard after segmentation, written under "
                "${work_dir}/shards/active/<shard_key>/, and a "
                "heartbeat.json updated at shard boundaries. A "
                "subsequent invocation with the same work_dir resumes "
                "from the last valid checkpoint; invalid or mismatched "
                "checkpoints are moved into "
                "${work_dir}/shards/quarantine/ with a UUID-suffixed "
                "unique name and their original bytes are preserved. "
                "Cannot overlap with --input-root or --output-dir. "
                "Ignored when omitted (legacy no-work-directory mode)."
            ),
            "default": None,
            "required": False,
            "type": None,
        },
        "--source-commit": {
            "help": (
                "Source commit SHA (40 lowercase hex characters) to bind "
                "each checkpoint and the heartbeat to a specific code "
                "revision. Required when --work-dir is supplied; "
                "ignored otherwise. The value is validated as a 40-char "
                "lowercase hex string and is recorded verbatim into "
                "every shard checkpoint."
            ),
            "default": None,
            "required": False,
            "type": None,
        },
        "--publish-dataset-id": {
            "help": (
                "Optional Hugging Face dataset ID to publish the export to "
                "(after a successful build). The target repository must already "
                "exist. No repository is created."
            ),
            "default": None,
            "required": False,
            "type": None,
        },
        "--publish-revision": {
            "help": "Target Hugging Face dataset revision for publishing (default: main). Only used with --publish-dataset-id.",
            "default": None,
            "required": False,
            "type": None,
        },
        "--publish-commit-message": {
            "help": "Optional commit message for the publishing commit. Only used with --publish-dataset-id.",
            "default": None,
            "required": False,
            "type": None,
        },
    }

    for option, contract in expected.items():
        action = _action(parser, option)
        assert action.help == contract["help"]
        assert action.default == contract["default"]
        assert action.required is contract["required"]
        assert action.type is contract["type"]

    assert _action(parser, "--device").choices == sorted(cli.PUBLIC_DEVICE_VALUES)
    assert isinstance(_action(parser, "--overwrite"), argparse._StoreTrueAction)
