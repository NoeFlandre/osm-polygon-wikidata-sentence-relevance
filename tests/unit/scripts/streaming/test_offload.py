"""RED tests for the checkpoint offload driver.

Contract:

* Operates against the EXISTING output Hub repository (no new repo).
* Uses a dedicated staging branch under
  ``checkpoints/<run-id>/<shard>/`` discovered via
  ``list_repo_tree``.
* Performs an independent readback (download) after upload and
  verifies byte equality (SHA-256 of the segmented parquet must
  match). On mismatch: abort, do NOT evict local checkpoint.
* Idempotent upload: a second call with the same
  ``(run_id, shard_key)`` returns the existing handle without
  re-uploading.
* Remote-first resume: ``discover_run(...)`` reconstructs the handle
  list from the staging branch; local ``state.json`` is a cache,
  the staging revision is the source of truth.
* ``discover_run`` rejects a missing or corrupt entry.
* The driver never publishes partial checkpoints as the final
  public dataset. The caller MUST run a separate
  ``publish_final_dataset`` step that does not import or reuse
  the offload helpers.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.contracts.schemas import (
    SEGMENTED_SENTENCES_SCHEMA,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "streaming"
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from scripts.streaming.offload import (  # noqa: E402
    CheckpointOffloader,
    CheckpointOffloadError,
    OffloadHandle,
    discover_run,
)

from osm_polygon_sentence_relevance.application.checkpoint import (  # noqa: E402
    segmented_schema_sha256,
)


def test_huggingface_hub_api_signature_compatibility() -> None:
    """Verify parameters passed in offload.py match installed huggingface_hub signatures."""
    import inspect

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        pytest.skip("huggingface_hub not installed in this environment")

    # Inspect HfApi methods used in offload.py
    create_branch_sig = inspect.signature(HfApi.create_branch)
    assert "repo_id" in create_branch_sig.parameters
    assert "branch" in create_branch_sig.parameters
    assert "repo_type" in create_branch_sig.parameters

    upload_folder_sig = inspect.signature(HfApi.upload_folder)
    assert "repo_id" in upload_folder_sig.parameters
    assert "folder_path" in upload_folder_sig.parameters
    assert "revision" in upload_folder_sig.parameters
    assert "repo_type" in upload_folder_sig.parameters

    list_repo_tree_sig = inspect.signature(HfApi.list_repo_tree)
    assert "repo_id" in list_repo_tree_sig.parameters
    assert "repo_type" in list_repo_tree_sig.parameters

    download_sig = inspect.signature(hf_hub_download)
    assert "repo_id" in download_sig.parameters
    assert "filename" in download_sig.parameters
    assert "revision" in download_sig.parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_checkpoint(active_dir: Path) -> dict:
    """Write a minimal valid checkpoint (segmented.parquet + metadata.json).

    Returns the metadata dict with ``segmented_table_sha256`` set to
    the actual SHA-256 of the on-disk parquet so callers can use it
    directly with the offload driver.
    """
    active_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "polygon_id": ["a", "b"],
            "wikidata": ["Q1", "Q1"],
            "document_id": ["d1", "d2"],
            "article_id": ["a1", "a2"],
            "source": ["wikipedia", "wikipedia"],
            "language": ["en", "en"],
            "site": ["en.wikipedia.org", "en.wikipedia.org"],
            "page_title": ["P1", "P2"],
            "section_id": ["s1", "s2"],
            "section_index": [0, 0],
            "section_path": [[], []],
            "sentence_index": [0, 0],
            "sentence_text_raw": ["hello", "world"],
            "sentence_text_normalized": ["hello", "world"],
            "previous_sentence": [None, None],
            "next_sentence": [None, None],
            "url": ["u1", "u2"],
            "page_id": [1, 2],
            "revision_id": [1, 2],
            "revision_timestamp": ["t1", "t2"],
            "document_content_hash": ["h1", "h2"],
            "section_content_hash": ["h3", "h4"],
            "segment_index": [0, 0],
            "segment_offset": [0, 0],
            "section_byte_offset": [0, 0],
            "sentence_byte_offset": [0, 0],
            "segment_length": [5, 5],
            "model_revision": ["v1", "v1"],
            "polygon_name": ["name1", "name2"],
            "osm_primary_tag": ["boundary", "boundary"],
            "osm_tags": pa.array(
                [{"key": "name", "value": "x1"}, {"key": "name", "value": "x2"}],
                type=pa.map_(pa.string(), pa.string()),
            ),
            "region": ["reg1", "reg2"],
            "lat": [12.34, 56.78],
            "lon": [12.34, 56.78],
        }
    )
    parquet_file = active_dir / "segmented.parquet"
    pq.write_table(
        table.select(SEGMENTED_SENTENCES_SCHEMA.names).cast(SEGMENTED_SENTENCES_SCHEMA),
        parquet_file,
    )
    sha = hashlib.sha256(parquet_file.read_bytes()).hexdigest()
    meta = {
        "schema_version": 2,
        "shard_key": "italy-latest",
        "input_dataset_revision": "abcdefabcdefabcdefabcdefabcdefabcdefabcd",
        "pipeline_version": "v1",
        "source_commit": "0123456789abcdef0123456789abcdef01234567",
        "model_name": "sat-3l-sm",
        "batch_size": 128,
        "input_root": "/tmp/streaming/run",
        "input_dataset_id": None,
        "segmented_table_sha256": sha,
        "segmented_table_bytes": parquet_file.stat().st_size,
        "segmented_schema_sha256": segmented_schema_sha256(),
        "completed_at_unix": 0,
        "source_files": [],
        "segmentation_report": {
            "input_section_occurrence_count": 0,
            "emitted_segment_count": 0,
            "retained_sentence_occurrence_count": 0,
            "dropped_empty_raw_count": 0,
            "dropped_empty_normalized_count": 0,
            "wikipedia_sentence_occurrence_count": 0,
            "wikivoyage_sentence_occurrence_count": 0,
        },
    }
    (active_dir / "metadata.json").write_text(json.dumps(meta))
    return meta


def _fake_uploaded_parquet(
    *,
    repo_id: str,
    folder_path: str,
    revision: str,
    commit_message: str | None = None,
    commit_description: str | None = None,
    path_in_repo: str,
    **_: object,
) -> str:
    """Stand-in for ``HfApi.upload_folder``.

    The driver's ``folder_path`` parameter carries the local active
    directory (an absolute path). The readback helper looks the bytes
    up under the OFFLOAD module's own per-shard relative path
    (``checkpoints/<run>/<shard>/...``) inside the fake remote root.
    The simplest deterministic mapping for tests: copy the local
    contents to the fake root at a path that also encodes the
    ``run_id`` and ``shard_key`` extracted from the active_dir's
    basename. Since tests always use ``active/<shard_key>`` as the
    layout, we use the basename.
    """
    import shutil

    src = Path(folder_path)
    if not src.is_dir():
        raise RuntimeError(f"folder_path does not exist: {folder_path}")
    dst = Path("/tmp/streaming-offload-fake") / path_in_repo
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return "fake-commit-sha"


# Mutable mapping used to configure the fake upload_folder fake's
# relative-path encoding. Tests set this before calling
# ``upload_and_verify``.
_FAKE_REMOTE_RUN_ID: dict[str, str] = {}


class _FakeLfs:
    def __init__(self, sha256: str) -> None:
        self.sha256 = sha256


class _FakeRepoFile:
    def __init__(self, path: str, source: Path) -> None:
        self.path = path
        self.size = source.stat().st_size
        self.lfs = (
            _FakeLfs(hashlib.sha256(source.read_bytes()).hexdigest())
            if source.name == "segmented.parquet"
            else None
        )


def _fake_list_repo_tree(**kwargs: object) -> list[_FakeRepoFile]:
    """List the fake remote with the same relative paths as HfApi."""
    root = Path("/tmp/streaming-offload-fake")
    path_in_repo = str(kwargs.get("path_in_repo", ""))
    folder = root / path_in_repo
    if not folder.is_dir():
        return []
    return [
        _FakeRepoFile(path.relative_to(root).as_posix(), path)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    ]


def _fake_create_branch(**kw: object) -> None:
    """Stand-in for ``HfApi.create_branch`` (idempotent)."""
    return None


# A globally-configurable readback failure flag. When True, the
# readback helper writes corrupt bytes so the offloader must abort.
_CORRUPT_READBACK = False


def _fake_hf_hub_download(
    *,
    repo_id: str,
    filename: str,
    local_dir: str | None = None,
    **_: object,
) -> str:
    """Stand-in for ``huggingface_hub.hf_hub_download`` used by both
    the offload readback and the discover-run probes."""
    import shutil

    fname = Path(filename)
    src = Path("/tmp/streaming-offload-fake") / fname
    if not src.is_file():
        raise RuntimeError(f"missing fake remote file: {src}")
    target = Path(local_dir or ".") / fname
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if _CORRUPT_READBACK and fname.name == "segmented.parquet":
        target.write_bytes(b"corrupted-readback")
    else:
        shutil.copy2(src, target)
    return str(target)


# ---------------------------------------------------------------------------
# GREEN: upload and verify happy path.
# ---------------------------------------------------------------------------


def test_offload_upload_and_verify_happy_path(tmp_path: Path, monkeypatch) -> None:
    active_dir = tmp_path / "active" / "italy-latest"
    meta = _write_checkpoint(active_dir)

    fake_state = Path("/tmp/streaming-offload-fake")
    if fake_state.exists():
        import shutil

        shutil.rmtree(fake_state)
    fake_state.mkdir(parents=True)

    global _CORRUPT_READBACK, _FAKE_REMOTE_RUN_ID
    _CORRUPT_READBACK = False
    _FAKE_REMOTE_RUN_ID = {"italy-latest": "run-test-001"}

    hub_api = mock.MagicMock()
    hub_api.create_branch.side_effect = _fake_create_branch
    hub_api.upload_folder.side_effect = _fake_uploaded_parquet
    hub_api.list_repo_tree.side_effect = _fake_list_repo_tree

    monkeypatch.setattr(
        "scripts.streaming.offload._lazy_hf_hub_download",
        lambda: _fake_hf_hub_download,
    )

    offloader = CheckpointOffloader(
        hub_api=hub_api,
        repo_id="NoeFlandre/osm-polygon-wikidata-sentence-relevance",
        staging_revision="run-test-001",
        run_id="run-test-001",
        local_cache_dir=tmp_path / "cache",
    )

    handle = offloader.upload_and_verify(
        shard_key="italy-latest",
        active_dir=active_dir,
        metadata=meta,
    )
    assert isinstance(handle, OffloadHandle)
    assert handle.shard_key == "italy-latest"
    assert handle.staging_revision == "run-test-001"
    assert handle.computed_table_sha256 == meta["segmented_table_sha256"]


# ---------------------------------------------------------------------------
# GREEN: upload + readback mismatch aborts, no eviction signal.
# ---------------------------------------------------------------------------


def test_offload_readback_mismatch_aborts(tmp_path: Path, monkeypatch) -> None:
    active_dir = tmp_path / "active" / "italy-latest"
    meta = _write_checkpoint(active_dir)
    fake_state = Path("/tmp/streaming-offload-fake")
    if fake_state.exists():
        import shutil

        shutil.rmtree(fake_state)
    fake_state.mkdir(parents=True)

    global _CORRUPT_READBACK, _FAKE_REMOTE_RUN_ID
    _CORRUPT_READBACK = True
    _FAKE_REMOTE_RUN_ID = {"italy-latest": "run-mismatch"}

    hub_api = mock.MagicMock()
    hub_api.create_branch.side_effect = _fake_create_branch
    hub_api.upload_folder.side_effect = _fake_uploaded_parquet
    hub_api.list_repo_tree.side_effect = _fake_list_repo_tree

    monkeypatch.setattr(
        "scripts.streaming.offload._lazy_hf_hub_download",
        lambda: _fake_hf_hub_download,
    )

    offloader = CheckpointOffloader(
        hub_api=hub_api,
        repo_id="NoeFlandre/osm-polygon-wikidata-sentence-relevance",
        staging_revision="run-mismatch",
        run_id="run-mismatch",
        local_cache_dir=tmp_path / "cache",
    )
    with pytest.raises(CheckpointOffloadError):
        offloader.upload_and_verify(
            shard_key="italy-latest",
            active_dir=active_dir,
            metadata=meta,
        )
    # Local directory is still intact (we must not be tricked into
    # deleting it on a remote failure).
    assert (active_dir / "segmented.parquet").exists()
    assert (active_dir / "metadata.json").exists()


# ---------------------------------------------------------------------------
# GREEN: idempotent upload.
# ---------------------------------------------------------------------------


def test_offload_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    active_dir = tmp_path / "active" / "italy-latest"
    meta = _write_checkpoint(active_dir)
    fake_state = Path("/tmp/streaming-offload-fake")
    if fake_state.exists():
        import shutil

        shutil.rmtree(fake_state)
    fake_state.mkdir(parents=True)

    global _CORRUPT_READBACK, _FAKE_REMOTE_RUN_ID
    _CORRUPT_READBACK = False
    _FAKE_REMOTE_RUN_ID = {"italy-latest": "run-idem"}

    hub_api = mock.MagicMock()
    hub_api.create_branch.side_effect = _fake_create_branch
    hub_api.upload_folder.side_effect = _fake_uploaded_parquet
    hub_api.list_repo_tree.side_effect = _fake_list_repo_tree

    monkeypatch.setattr(
        "scripts.streaming.offload._lazy_hf_hub_download",
        lambda: _fake_hf_hub_download,
    )

    offloader = CheckpointOffloader(
        hub_api=hub_api,
        repo_id="NoeFlandre/osm-polygon-wikidata-sentence-relevance",
        staging_revision="run-idem",
        run_id="run-idem",
        local_cache_dir=tmp_path / "cache",
    )

    h1 = offloader.upload_and_verify(
        shard_key="italy-latest", active_dir=active_dir, metadata=meta
    )
    n_calls_first = hub_api.upload_folder.call_count

    h2 = offloader.upload_and_verify(
        shard_key="italy-latest", active_dir=active_dir, metadata=meta
    )
    assert hub_api.upload_folder.call_count == n_calls_first
    assert h1.shard_key == h2.shard_key
    assert h1.computed_table_sha256 == h2.computed_table_sha256


def test_idempotent_reuse_rejects_corrupt_remote_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    """Existence alone must never make a remote checkpoint reusable."""
    active_dir = tmp_path / "active" / "italy-latest"
    meta = _write_checkpoint(active_dir)
    fake_state = Path("/tmp/streaming-offload-fake")
    if fake_state.exists():
        import shutil

        shutil.rmtree(fake_state)
    fake_state.mkdir(parents=True)

    global _CORRUPT_READBACK, _FAKE_REMOTE_RUN_ID
    _CORRUPT_READBACK = False
    _FAKE_REMOTE_RUN_ID = {"italy-latest": "run-corrupt-meta"}

    hub_api = mock.MagicMock()
    hub_api.create_branch.side_effect = _fake_create_branch
    hub_api.upload_folder.side_effect = _fake_uploaded_parquet
    hub_api.list_repo_tree.side_effect = _fake_list_repo_tree
    monkeypatch.setattr(
        "scripts.streaming.offload._lazy_hf_hub_download",
        lambda: _fake_hf_hub_download,
    )
    offloader = CheckpointOffloader(
        hub_api=hub_api,
        repo_id="owner/output",
        staging_revision="checkpoints/run-corrupt-meta",
        run_id="run-corrupt-meta",
        local_cache_dir=tmp_path / "cache",
    )
    offloader.upload_and_verify(
        shard_key="italy-latest", active_dir=active_dir, metadata=meta
    )
    remote_meta = (
        fake_state
        / "checkpoints"
        / "run-corrupt-meta"
        / "italy-latest"
        / "metadata.json"
    )
    payload = json.loads(remote_meta.read_text())
    payload["source_commit"] = "f" * 40
    remote_meta.write_text(json.dumps(payload))

    with pytest.raises(CheckpointOffloadError, match="source_commit"):
        offloader.upload_and_verify(
            shard_key="italy-latest", active_dir=active_dir, metadata=meta
        )
    assert hub_api.upload_folder.call_count == 1


# ---------------------------------------------------------------------------
# RED: discover_run reconstructs from the staging branch.
# ---------------------------------------------------------------------------


def test_discover_run_reconstructs_handles_from_staging_branch(
    tmp_path: Path, monkeypatch
) -> None:
    fake = Path("/tmp/streaming-offload-discover")
    if fake.exists():
        import shutil

        shutil.rmtree(fake)
    fake.mkdir(parents=True)
    for stem in ("italy-latest", "bavaria-latest"):
        d = fake / "checkpoints" / "run-discover" / stem
        meta = _write_checkpoint(d)
        meta["shard_key"] = stem
        (d / "metadata.json").write_text(json.dumps(meta))

    fake_tree = [
        _FakeRepoFile(
            "checkpoints/run-discover/italy-latest/segmented.parquet",
            fake / "checkpoints/run-discover/italy-latest/segmented.parquet",
        ),
        _FakeRepoFile(
            "checkpoints/run-discover/italy-latest/metadata.json",
            fake / "checkpoints/run-discover/italy-latest/metadata.json",
        ),
        _FakeRepoFile(
            "checkpoints/run-discover/bavaria-latest/segmented.parquet",
            fake / "checkpoints/run-discover/bavaria-latest/segmented.parquet",
        ),
        _FakeRepoFile(
            "checkpoints/run-discover/bavaria-latest/metadata.json",
            fake / "checkpoints/run-discover/bavaria-latest/metadata.json",
        ),
    ]

    hub_api = mock.MagicMock()
    hub_api.list_repo_tree.side_effect = lambda **kw: (
        list(fake_tree) if kw.get("recursive", True) else []
    )

    # discover_run uses hf_hub_download pointed at the fake root.
    downloaded: list[str] = []

    def fake_dl(**kw: object) -> str:
        local_dir = Path(kw["local_dir"])
        filename = kw["filename"]
        downloaded.append(str(filename))
        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        src = Path("/tmp/streaming-offload-discover") / filename
        if not src.is_file():
            raise RuntimeError(f"missing fake remote file: {src}")
        import shutil

        shutil.copy2(src, target)
        return str(target)

    monkeypatch.setattr(
        "scripts.streaming.offload._lazy_hf_hub_download",
        lambda: fake_dl,
    )

    handles = discover_run(
        hub_api=hub_api,
        repo_id="owner/repo",
        run_id="run-discover",
        local_cache_dir=tmp_path / "cache",
    )
    keys = sorted(h.shard_key for h in handles)
    assert keys == ["bavaria-latest", "italy-latest"]
    assert all(name.endswith("metadata.json") for name in downloaded)


# ---------------------------------------------------------------------------
# RED: discover_run rejects a corrupt entry.
# ---------------------------------------------------------------------------


def test_discover_run_rejects_corrupt_entry(tmp_path: Path) -> None:
    """A checkpoint dir missing metadata.json must abort the discovery."""

    fake = Path("/tmp/streaming-offload-corrupt")
    if fake.exists():
        import shutil

        shutil.rmtree(fake)
    fake.mkdir(parents=True)
    (fake / "checkpoints" / "run-corrupt" / "italy-latest").mkdir(
        parents=True, exist_ok=True
    )

    class _RepoFile:
        def __init__(self, path: str) -> None:
            self.path = path

    fake_tree = [_RepoFile("checkpoints/run-corrupt/italy-latest/segmented.parquet")]
    hub_api = mock.MagicMock()
    hub_api.list_repo_tree.side_effect = lambda **kw: (
        list(fake_tree) if kw.get("recursive", True) else []
    )

    with pytest.raises(CheckpointOffloadError):
        discover_run(
            hub_api=hub_api,
            repo_id="owner/repo",
            run_id="run-corrupt",
            local_cache_dir=tmp_path / "cache",
        )
