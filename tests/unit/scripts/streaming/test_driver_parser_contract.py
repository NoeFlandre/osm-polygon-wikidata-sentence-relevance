"""Exact command-line contract tests for the streaming driver."""

from __future__ import annotations

import argparse

from scripts.streaming.driver import _build_driver_parser


def test_driver_parser_preserves_command_and_option_contract() -> None:
    parser = _build_driver_parser()

    assert parser.prog == "scripts.streaming.driver"
    command = next(action for action in parser._actions if action.dest == "command")
    assert isinstance(command, argparse._StoreAction)
    assert command.option_strings == []
    assert command.required is True
    assert command.choices == ["process-shard", "stream-build"]

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest != "help" and action.dest != "command"
    }
    expected = {
        "confirm_offload": {
            "option_strings": ["--confirm-offload"],
            "action": argparse._StoreTrueAction,
            "required": True,
            "default": False,
        },
        "shard": {
            "option_strings": ["--shard"],
            "action": argparse._StoreAction,
            "required": False,
            "default": None,
        },
        "max_shards": {
            "option_strings": ["--max-shards"],
            "action": argparse._StoreAction,
            "required": False,
            "default": None,
            "type": int,
        },
        "run_id": {
            "option_strings": ["--run-id"],
            "required": True,
        },
        "staging_revision": {
            "option_strings": ["--staging-revision"],
            "required": True,
        },
        "repo_id": {"option_strings": ["--repo-id"], "required": True},
        "upstream_repo_id": {
            "option_strings": ["--upstream-repo-id"],
            "required": True,
        },
        "resolved_revision": {
            "option_strings": ["--resolved-revision"],
            "required": True,
        },
        "source_commit": {
            "option_strings": ["--source-commit"],
            "required": True,
        },
        "work_dir": {
            "option_strings": ["--work-dir"],
            "required": True,
        },
        "input_root": {
            "option_strings": ["--input-root"],
            "required": False,
            "default": None,
        },
        "max_disk_bytes": {
            "option_strings": ["--max-disk-bytes"],
            "required": False,
            "default": 1 << 30,
            "type": int,
        },
        "pipeline_version": {
            "option_strings": ["--pipeline-version"],
            "required": False,
            "default": "v1",
        },
        "model_name": {
            "option_strings": ["--model-name"],
            "required": False,
            "default": "sat-12l-sm",
        },
        "batch_size": {
            "option_strings": ["--batch-size"],
            "required": False,
            "default": 128,
            "type": int,
        },
        "device": {
            "option_strings": ["--device"],
            "required": False,
            "default": "cuda",
            "choices": ["cuda"],
        },
        "resume_bundle": {
            "option_strings": ["--resume-bundle"],
            "required": False,
            "default": None,
        },
    }

    assert set(actions) == set(expected)
    for dest, assertions in expected.items():
        action = actions[dest]
        assert action.option_strings == assertions.pop("option_strings")
        expected_action = assertions.pop("action", None)
        if expected_action is not None:
            assert isinstance(action, expected_action)
        for name, value in assertions.items():
            assert getattr(action, name) == value
