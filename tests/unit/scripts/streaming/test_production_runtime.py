"""Production-contract tests for the runnable streaming loop."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from scripts.streaming.driver import list_remote_shard_keys, main


class _RepoFile:
    def __init__(self, path: str) -> None:
        self.path = path


def test_list_remote_shard_keys_uses_pinned_polygons_tree() -> None:
    api = mock.Mock()
    api.list_repo_tree.return_value = [
        _RepoFile("polygons/zambia-latest.parquet"),
        _RepoFile("polygons/albania-latest.parquet"),
        _RepoFile("polygons/README.md"),
    ]

    assert list_remote_shard_keys(
        hub_api=api,
        repo_id="owner/input",
        revision="a" * 40,
    ) == ["albania-latest", "zambia-latest"]
    api.list_repo_tree.assert_called_once_with(
        repo_id="owner/input",
        repo_type="dataset",
        revision="a" * 40,
        path_in_repo="polygons",
        recursive=False,
    )


def test_stream_build_processes_sorted_remote_shards_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    segmenter = mock.Mock()
    driver = mock.Mock()
    hub_api = mock.Mock()
    hub_api.list_repo_tree.return_value = [
        _RepoFile("polygons/b-latest.parquet"),
        _RepoFile("polygons/a-latest.parquet"),
    ]

    with (
        mock.patch(
            "osm_polygon_sentence_relevance.sentences.sat.SaTSentenceSegmenter",
            return_value=segmenter,
        ),
        mock.patch("huggingface_hub.HfApi", return_value=hub_api),
        mock.patch("scripts.streaming.driver.StreamDriver", return_value=driver),
    ):
        rc = main(
            [
                "stream-build",
                "--confirm-offload",
                "--run-id",
                "run-1",
                "--staging-revision",
                "checkpoints/run-1",
                "--repo-id",
                "owner/output",
                "--upstream-repo-id",
                "owner/input",
                "--resolved-revision",
                "a" * 40,
                "--source-commit",
                "b" * 40,
                "--work-dir",
                str(tmp_path),
                "--pipeline-version",
                "0.1.0",
                "--batch-size",
                "128",
                "--device",
                "cuda",
            ]
        )

    assert rc == 0
    assert driver.process_shard.call_args_list == [
        mock.call("a-latest", segmenter=segmenter),
        mock.call("b-latest", segmenter=segmenter),
    ]
    assert '"completed":2' in capsys.readouterr().out


def test_stream_build_max_shards_bounds_canary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    segmenter = mock.Mock()
    driver = mock.Mock()
    hub_api = mock.Mock()
    hub_api.list_repo_tree.return_value = [
        _RepoFile("polygons/a-latest.parquet"),
        _RepoFile("polygons/b-latest.parquet"),
    ]
    with (
        mock.patch(
            "osm_polygon_sentence_relevance.sentences.sat.SaTSentenceSegmenter",
            return_value=segmenter,
        ),
        mock.patch("huggingface_hub.HfApi", return_value=hub_api),
        mock.patch("scripts.streaming.driver.StreamDriver", return_value=driver),
    ):
        rc = main(
            [
                "stream-build",
                "--confirm-offload",
                "--max-shards",
                "1",
                "--run-id",
                "run-1",
                "--staging-revision",
                "checkpoints/run-1",
                "--repo-id",
                "owner/output",
                "--upstream-repo-id",
                "owner/input",
                "--resolved-revision",
                "a" * 40,
                "--source-commit",
                "b" * 40,
                "--work-dir",
                str(tmp_path),
                "--pipeline-version",
                "0.1.0",
            ]
        )

    assert rc == 0
    driver.process_shard.assert_called_once_with("a-latest", segmenter=segmenter)
