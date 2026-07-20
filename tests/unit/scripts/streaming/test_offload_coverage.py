"""Coverage-targeted tests for the offload module uncovered branches."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest import mock

import pytest
from scripts.streaming.offload import (
    CheckpointOffloader,
    CheckpointOffloadError,
    OffloadHandle,
    _lazy_hf_hub_download,
    _phys,
    _sha256_file,
    _validate_run_id,
    _validate_shard_key,
    discover_run,
)


def _write_parquet(path: Path, content: bytes = b"hello world") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    import hashlib

    return hashlib.sha256(content).hexdigest().lower()


def _make_active(tmp_path: Path, shard: str, content: bytes = b"hello world"):
    active = tmp_path / "active" / shard
    sha = _write_parquet(active / "segmented.parquet", content)
    (active / "metadata.json").write_text(
        json.dumps(
            {
                "segmented_table_sha256": sha,
                "segmented_table_bytes": len(content),
                "shard_key": shard,
                "source_commit": "0" * 40,
                "input_dataset_revision": "a" * 40,
                "pipeline_version": "v1",
                "model_name": "sat-3l-sm",
                "batch_size": 128,
            }
        )
    )
    return active, sha


def _fake_hub(uploaded_paths: dict[str, list[str]] | None = None):
    hub = mock.MagicMock()
    hub.create_branch = mock.Mock(return_value=None)

    def _upload_folder(**kw):
        # Capture the path so tests can assert.
        fp = Path(kw["folder_path"])
        key = fp.name
        uploaded_paths.setdefault(key, [])
        uploaded_paths[key].append(str(fp))
        return mock.Mock(oid="oid")

    hub.upload_folder = mock.Mock(side_effect=_upload_folder)

    def _tree(repo_id, repo_type, revision, path_in_repo, recursive=False):
        # Pretend the empty bucket when nothing matches.
        if uploaded_paths is None:
            return iter([])
        # If path_in_repo targets a specific shard folder, emit entries.
        out = []
        for shard_key in uploaded_paths:
            # We do not actually need to filter here; tests use the
            # real flow. Return an empty iterator and let the per-shard
            # group dict drive the test.
            out.append(mock.Mock(path=f"checkpoints/RUN/{shard_key}/segmented.parquet"))
            out.append(mock.Mock(path=f"checkpoints/RUN/{shard_key}/metadata.json"))
        return iter(out)

    hub.list_repo_tree = mock.Mock(side_effect=_tree)
    return hub


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------


def test_validate_shard_key_rejects_blank() -> None:
    with pytest.raises(ValueError, match="shard_key"):
        _validate_shard_key("")


def test_validate_shard_key_rejects_uppercase() -> None:
    with pytest.raises(ValueError, match="shard_key"):
        _validate_shard_key("BadKey")


def test_validate_run_id_rejects_blank() -> None:
    with pytest.raises(ValueError, match="run_id"):
        _validate_run_id("")


def test_validate_run_id_rejects_whitespace() -> None:
    with pytest.raises(ValueError, match="run_id"):
        _validate_run_id("has space")


def test_validate_run_id_rejects_slash() -> None:
    with pytest.raises(ValueError, match="run_id"):
        _validate_run_id("a/b")


# ---------------------------------------------------------------------------
# Constructor validation.
# ---------------------------------------------------------------------------


def test_ctor_rejects_blank_repo_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repo_id"):
        CheckpointOffloader(
            hub_api=mock.Mock(),
            repo_id="",
            staging_revision="rev",
            run_id="run-1",
            local_cache_dir=tmp_path,
        )


def test_ctor_rejects_repo_id_no_slash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repo_id"):
        CheckpointOffloader(
            hub_api=mock.Mock(),
            repo_id="noslash",
            staging_revision="rev",
            run_id="run-1",
            local_cache_dir=tmp_path,
        )


def test_ctor_rejects_blank_staging_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="staging_revision"):
        CheckpointOffloader(
            hub_api=mock.Mock(),
            repo_id="owner/repo",
            staging_revision="",
            run_id="run-1",
            local_cache_dir=tmp_path,
        )


def test_ctor_rejects_blank_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        CheckpointOffloader(
            hub_api=mock.Mock(),
            repo_id="owner/repo",
            staging_revision="rev",
            run_id="",
            local_cache_dir=tmp_path,
        )


def test_ctor_rejects_slash_in_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        CheckpointOffloader(
            hub_api=mock.Mock(),
            repo_id="owner/repo",
            staging_revision="rev",
            run_id="bad/run",
            local_cache_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# _ensure_branch: duplicate-branch handling.
# ---------------------------------------------------------------------------


def test_ensure_branch_swallows_duplicate(tmp_path: Path) -> None:
    hub = mock.MagicMock()
    hub.create_branch = mock.Mock(
        side_effect=RuntimeError("409: branch already exists")
    )
    co = CheckpointOffloader(
        hub_api=hub,
        repo_id="owner/repo",
        staging_revision="rev",
        run_id="run-1",
        local_cache_dir=tmp_path,
    )
    # Should not raise.
    co._ensure_branch()


def test_ensure_branch_propagates_unexpected_error(tmp_path: Path) -> None:
    hub = mock.MagicMock()
    hub.create_branch = mock.Mock(side_effect=RuntimeError("network"))
    co = CheckpointOffloader(
        hub_api=hub,
        repo_id="owner/repo",
        staging_revision="rev",
        run_id="run-1",
        local_cache_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="network"):
        co._ensure_branch()


# ---------------------------------------------------------------------------
# _remote_already_uploaded.
# ---------------------------------------------------------------------------


def test_remote_already_uploaded_true(tmp_path: Path) -> None:
    hub = mock.MagicMock()
    hub.list_repo_tree = mock.Mock(
        return_value=iter(
            [
                mock.Mock(path="checkpoints/r1/s1/segmented.parquet"),
                mock.Mock(path="checkpoints/r1/s1/metadata.json"),
            ]
        )
    )
    co = CheckpointOffloader(
        hub_api=hub,
        repo_id="owner/repo",
        staging_revision="rev",
        run_id="r1",
        local_cache_dir=tmp_path,
    )
    assert co._remote_already_uploaded(shard_key="italy-latest") is True


def test_remote_already_uploaded_false(tmp_path: Path) -> None:
    hub = mock.MagicMock()
    hub.list_repo_tree = mock.Mock(
        return_value=iter([mock.Mock(path="checkpoints/r1/s1/metadata.json")])
    )
    co = CheckpointOffloader(
        hub_api=hub,
        repo_id="owner/repo",
        staging_revision="rev",
        run_id="r1",
        local_cache_dir=tmp_path,
    )
    assert co._remote_already_uploaded(shard_key="italy-latest") is False


def test_remote_already_uploaded_list_raises_returns_false(tmp_path: Path) -> None:
    hub = mock.MagicMock()
    hub.list_repo_tree = mock.Mock(side_effect=RuntimeError("oops"))
    co = CheckpointOffloader(
        hub_api=hub,
        repo_id="owner/repo",
        staging_revision="rev",
        run_id="r1",
        local_cache_dir=tmp_path,
    )
    assert co._remote_already_uploaded(shard_key="italy-latest") is False


# ---------------------------------------------------------------------------
# upload_and_verify: failure paths.
# ---------------------------------------------------------------------------


def test_upload_and_verify_rejects_missing_active_dir(tmp_path: Path) -> None:
    co = CheckpointOffloader(
        hub_api=mock.Mock(),
        repo_id="owner/repo",
        staging_revision="rev",
        run_id="run-1",
        local_cache_dir=tmp_path,
    )
    with pytest.raises(CheckpointOffloadError, match="active_dir"):
        co.upload_and_verify(
            shard_key="italy-latest",
            active_dir=tmp_path / "nope",
            metadata={},
        )


def test_upload_and_verify_rejects_missing_parquet(tmp_path: Path) -> None:
    co = CheckpointOffloader(
        hub_api=mock.Mock(),
        repo_id="owner/repo",
        staging_revision="rev",
        run_id="run-1",
        local_cache_dir=tmp_path,
    )
    bad = tmp_path / "active-no-parquet"
    bad.mkdir()
    with pytest.raises(CheckpointOffloadError, match="segmented.parquet"):
        co.upload_and_verify(
            shard_key="italy-latest",
            active_dir=bad,
            metadata={},
        )


def test_upload_and_verify_rejects_sha_metadata_mismatch(tmp_path: Path) -> None:
    """If metadata SHA does not match local file SHA, raise."""
    active, _ = _make_active(tmp_path, "italy-latest")
    co = CheckpointOffloader(
        hub_api=mock.Mock(),
        repo_id="owner/repo",
        staging_revision="rev",
        run_id="run-1",
        local_cache_dir=tmp_path,
    )
    with pytest.raises(CheckpointOffloadError, match="local SHA-256 mismatch"):
        co.upload_and_verify(
            shard_key="italy-latest",
            active_dir=active,
            metadata={"segmented_table_sha256": "0" * 64},
        )


def test_upload_and_verify_upload_folder_failure(tmp_path: Path) -> None:
    active, sha = _make_active(tmp_path, "italy-latest")
    hub = mock.MagicMock()
    hub.create_branch = mock.Mock(return_value=None)
    hub.list_repo_tree = mock.Mock(return_value=iter([]))
    hub.upload_folder = mock.Mock(side_effect=RuntimeError("upload fail"))
    co = CheckpointOffloader(
        hub_api=hub,
        repo_id="owner/repo",
        staging_revision="rev",
        run_id="run-1",
        local_cache_dir=tmp_path,
    )
    with pytest.raises(CheckpointOffloadError, match="upload_folder"):
        co.upload_and_verify(
            shard_key="italy-latest",
            active_dir=active,
            metadata={"segmented_table_sha256": sha},
        )


def test_upload_and_verify_readback_download_failure(tmp_path: Path) -> None:
    active, sha = _make_active(tmp_path, "italy-latest")
    hub = mock.MagicMock()
    hub.create_branch = mock.Mock(return_value=None)
    hub.list_repo_tree = mock.Mock(return_value=iter([]))
    hub.upload_folder = mock.Mock(return_value=mock.Mock(oid="oid"))
    co = CheckpointOffloader(
        hub_api=hub,
        repo_id="owner/repo",
        staging_revision="rev",
        run_id="run-1",
        local_cache_dir=tmp_path,
    )
    monkeypatch_dl = mock.Mock(side_effect=RuntimeError("network"))
    with (
        mock.patch(
            "scripts.streaming.offload._lazy_hf_hub_download",
            return_value=monkeypatch_dl,
        ),
        pytest.raises(CheckpointOffloadError, match="readback download"),
    ):
        co.upload_and_verify(
            shard_key="italy-latest",
            active_dir=active,
            metadata={"segmented_table_sha256": sha},
        )


def test_upload_and_verify_readback_sha_mismatch(tmp_path: Path) -> None:
    active, sha = _make_active(tmp_path, "italy-latest")
    hub = mock.MagicMock()
    hub.create_branch = mock.Mock(return_value=None)
    hub.list_repo_tree = mock.Mock(return_value=iter([]))
    hub.upload_folder = mock.Mock(return_value=mock.Mock(oid="oid"))
    co = CheckpointOffloader(
        hub_api=hub,
        repo_id="owner/repo",
        staging_revision="rev",
        run_id="run-1",
        local_cache_dir=tmp_path,
    )

    # Return bytes that do NOT match the local sha.
    def _bad_dl(**kw):
        out = Path(kw["local_dir"]) / Path(kw["filename"]).name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"corrupted")
        return str(out)

    with (
        mock.patch(
            "scripts.streaming.offload._lazy_hf_hub_download",
            return_value=_bad_dl,
        ),
        pytest.raises(CheckpointOffloadError, match="readback SHA-256"),
    ):
        co.upload_and_verify(
            shard_key="italy-latest",
            active_dir=active,
            metadata={"segmented_table_sha256": sha},
        )


def test_upload_and_verify_skips_upload_when_remote_present(
    monkeypatch, tmp_path: Path
) -> None:
    """Idempotency: if remote has segmented.parquet, no upload_folder call."""
    active, sha = _make_active(tmp_path, "italy-latest")
    hub = mock.MagicMock()
    hub.create_branch = mock.Mock(return_value=None)
    hub.list_repo_tree = mock.Mock(
        return_value=iter([mock.Mock(path="checkpoints/run-1/s1/segmented.parquet")])
    )
    hub.upload_folder = mock.Mock(return_value=mock.Mock(oid="oid"))

    def _good_dl(**kw):
        # Return the SAME bytes as the local upload so SHA matches.
        out = Path(kw["local_dir"]) / Path(kw["filename"]).name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"hello world")
        return str(out)

    co = CheckpointOffloader(
        hub_api=hub,
        repo_id="owner/repo",
        staging_revision="rev",
        run_id="run-1",
        local_cache_dir=tmp_path,
    )
    with mock.patch(
        "scripts.streaming.offload._lazy_hf_hub_download",
        return_value=_good_dl,
    ):
        handle = co.upload_and_verify(
            shard_key="italy-latest",
            active_dir=active,
            metadata={"segmented_table_sha256": sha},
        )
    assert hub.upload_folder.call_count == 0
    assert handle.shard_key == "italy-latest"


# ---------------------------------------------------------------------------
# discover_run: failure paths.
# ---------------------------------------------------------------------------


def test_discover_run_list_repo_tree_failure(tmp_path: Path) -> None:
    hub = mock.MagicMock()
    hub.list_repo_tree = mock.Mock(side_effect=RuntimeError("tree fail"))
    with pytest.raises(CheckpointOffloadError, match="list_repo_tree"):
        discover_run(
            hub_api=hub,
            repo_id="owner/repo",
            run_id="run-1",
            local_cache_dir=tmp_path,
        )


def test_discover_run_missing_segmented_raises(tmp_path: Path) -> None:
    hub = mock.MagicMock()
    hub.list_repo_tree = mock.Mock(
        return_value=iter([mock.Mock(path="checkpoints/run-1/s1/metadata.json")])
    )
    with pytest.raises(CheckpointOffloadError, match="missing segmented"):
        discover_run(
            hub_api=hub,
            repo_id="owner/repo",
            run_id="run-1",
            local_cache_dir=tmp_path,
        )


def test_discover_run_missing_metadata_raises(tmp_path: Path) -> None:
    hub = mock.MagicMock()
    hub.list_repo_tree = mock.Mock(
        return_value=iter([mock.Mock(path="checkpoints/run-1/s1/segmented.parquet")])
    )
    with pytest.raises(CheckpointOffloadError, match="missing metadata"):
        discover_run(
            hub_api=hub,
            repo_id="owner/repo",
            run_id="run-1",
            local_cache_dir=tmp_path,
        )


def test_discover_run_readback_failure(tmp_path: Path) -> None:
    hub = mock.MagicMock()
    hub.list_repo_tree = mock.Mock(
        return_value=iter(
            [
                mock.Mock(path="checkpoints/run-1/s1/segmented.parquet"),
                mock.Mock(path="checkpoints/run-1/s1/metadata.json"),
            ]
        )
    )
    bad_dl = mock.Mock(side_effect=RuntimeError("network"))
    with (
        mock.patch(
            "scripts.streaming.offload._lazy_hf_hub_download",
            return_value=bad_dl,
        ),
        pytest.raises(CheckpointOffloadError, match="readback failed"),
    ):
        discover_run(
            hub_api=hub,
            repo_id="owner/repo",
            run_id="run-1",
            local_cache_dir=tmp_path,
        )


def test_discover_run_sha_mismatch_raises(tmp_path: Path) -> None:
    hub = mock.MagicMock()
    hub.list_repo_tree = mock.Mock(
        return_value=iter(
            [
                mock.Mock(path="checkpoints/run-1/s1/segmented.parquet"),
                mock.Mock(path="checkpoints/run-1/s1/metadata.json"),
            ]
        )
    )

    cache = tmp_path / "cache"
    cache.mkdir()

    def _dl(**kw):
        out = Path(kw["local_dir"]) / Path(kw["filename"]).name
        out.parent.mkdir(parents=True, exist_ok=True)
        if "metadata" in kw["filename"]:
            out.write_text(
                json.dumps(
                    {
                        "shard_key": "s1",
                        "segmented_table_sha256": "0" * 64,
                        "segmented_table_bytes": 3,
                        "source_commit": "0" * 40,
                        "input_dataset_revision": "a" * 40,
                        "pipeline_version": "v1",
                        "model_name": "sat-3l-sm",
                        "batch_size": 128,
                    }
                ),
            )
        else:
            out.write_bytes(b"foo")
        return str(out)

    with (
        mock.patch(
            "scripts.streaming.offload._lazy_hf_hub_download",
            return_value=_dl,
        ),
        pytest.raises(CheckpointOffloadError, match="readback SHA-256"),
    ):
        discover_run(
            hub_api=hub,
            repo_id="owner/repo",
            run_id="run-1",
            local_cache_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    import hashlib

    assert _sha256_file(p) == hashlib.sha256(b"abc").hexdigest().lower()


def test_phys(tmp_path: Path) -> None:
    p = tmp_path / "x"
    p.mkdir()
    assert _phys(p) == p.resolve()


def test_offload_handle_is_frozen() -> None:
    h = OffloadHandle(
        repo_id="o/r",
        run_id="r",
        shard_key="s",
        staging_revision="HEAD",
        folder_path="checkpoints/r/s",
        expected_table_sha256="0" * 64,
        computed_table_sha256="0" * 64,
    )
    assert dataclasses.is_dataclass(h)
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.shard_key = "y"


def test_lazy_hf_hub_download_raises_when_missing(monkeypatch) -> None:
    """When huggingface_hub is not importable, raises CheckpointOffloadError."""
    import sys

    saved = sys.modules.pop("huggingface_hub", None)

    class _Blocker:
        def __getattr__(self, name):
            raise ImportError("no hub")

    sys.modules["huggingface_hub"] = _Blocker()  # type: ignore[assignment]
    try:
        with pytest.raises(CheckpointOffloadError, match="huggingface_hub is required"):
            _lazy_hf_hub_download()
    finally:
        sys.modules.pop("huggingface_hub", None)
        if saved is not None:
            sys.modules["huggingface_hub"] = saved
