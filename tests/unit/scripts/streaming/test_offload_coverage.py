"""Focused defensive branches for verified remote checkpoints."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest import mock

import pytest
from scripts.streaming.offload import (
    CheckpointOffloader,
    CheckpointOffloadError,
    OffloadHandle,
    _list_files,
)


def _offloader(tmp_path: Path, hub: object | None = None) -> CheckpointOffloader:
    return CheckpointOffloader(
        hub_api=hub or mock.MagicMock(),
        repo_id="owner/repo",
        staging_revision="checkpoints/run-1",
        run_id="run-1",
        local_cache_dir=tmp_path,
    )


def test_constructor_rejects_invalid_repo(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repo_id"):
        CheckpointOffloader(
            hub_api=mock.Mock(),
            repo_id="invalid",
            staging_revision="branch",
            run_id="run",
            local_cache_dir=tmp_path,
        )


def test_constructor_rejects_blank_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="staging_revision"):
        CheckpointOffloader(
            hub_api=mock.Mock(),
            repo_id="owner/repo",
            staging_revision=" ",
            run_id="run",
            local_cache_dir=tmp_path,
        )


def test_ensure_branch_tolerates_duplicate(tmp_path: Path) -> None:
    hub = mock.MagicMock()
    hub.create_branch.side_effect = RuntimeError("409: branch already exists")
    _offloader(tmp_path, hub)._ensure_branch()


def test_ensure_branch_wraps_network_failure(tmp_path: Path) -> None:
    hub = mock.MagicMock()
    hub.create_branch.side_effect = OSError("network unavailable")
    with pytest.raises(CheckpointOffloadError, match="ensure staging branch") as err:
        _offloader(tmp_path, hub)._ensure_branch()
    assert isinstance(err.value.__cause__, OSError)


def test_list_files_treats_missing_branch_as_empty() -> None:
    hub = mock.MagicMock()
    hub.list_repo_tree.side_effect = RuntimeError("404 revision not found")
    assert (
        _list_files(
            hub_api=hub,
            repo_id="owner/repo",
            revision="missing",
            folder_path="checkpoints/run",
        )
        == {}
    )


def test_list_files_wraps_unexpected_failure() -> None:
    hub = mock.MagicMock()
    hub.list_repo_tree.side_effect = OSError("transport")
    with pytest.raises(CheckpointOffloadError, match="inspect staging") as err:
        _list_files(
            hub_api=hub,
            repo_id="owner/repo",
            revision="branch",
            folder_path="checkpoints/run",
        )
    assert isinstance(err.value.__cause__, OSError)


def test_handle_is_frozen() -> None:
    handle = OffloadHandle(
        repo_id="owner/repo",
        run_id="run",
        shard_key="shard",
        staging_revision="branch",
        folder_path="checkpoints/run/shard",
        expected_table_sha256="0" * 64,
        computed_table_sha256="0" * 64,
        table_bytes=1,
        metadata={},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        handle.shard_key = "changed"  # type: ignore[misc]
