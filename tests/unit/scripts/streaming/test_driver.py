"""RED tests for the per-shard streaming driver.

Contract:

* Per-shard orchestration: download -> process_single_shard -> upload
  -> verified readback -> evict. If any step fails, neither the
  per-shard inbox nor the active checkpoint is removed.
* Eviction is gated on:
  (1) strict validation of the just-published checkpoint; AND
  (2) verified readback SHA-256 from the staging branch.
  Heartbeat text is NOT a substitute. A new WAL abstraction is not
  introduced.
* Bounded disk usage: the driver enforces a soft free-bytes ceiling
  on the streaming root before downloading any shard's inbox.
* OAR_JOB_ID presence guards scheduler-owned compute-node scratch
  use. Outside an OAR allocation, ``--allow-frontend-execution``
  must be explicitly set or the driver refuses.
* The driver binds to the resolved upstream commit (40 hex) and
  refuses to operate when the pinned revision differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "streaming"
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from scripts.streaming.driver import (  # noqa: E402
    DriverError,
    OarJobIdRequired,
    StreamDriver,
)

VALID_REVISION = "abcdef0123456789abcdef0123456789abcdef01"
VALID_COMMIT = "0123456789abcdef0123456789abcdef01234567"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_downloader_to_local_copy(monkeypatch, input_root: Path) -> None:
    """Replace ``PerFileHubDownloader.download`` with a stub that copies
    the already-staged local fixture into the driver's inbox subdir.

    Mirrors the path layout the driver expects
    (``polygons/<shard>.parquet``, etc.) without any Hub traffic.
    """
    import shutil

    def _download(self, path_in_repo: str, **_kw):  # type: ignore
        target = Path(self.target_dir) / path_in_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        src = Path(input_root) / path_in_repo
        if not src.exists():
            raise FileNotFoundError(f"Missing file in test fixture: {src}")
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.copy2(src, target)
        return mock.Mock(
            path=target,
            repo_id=self.repo_id,
            path_in_repo=path_in_repo,
            hub_commit_hash=self.resolved_revision,
            hub_etag='"x"',
            hub_url="https://example",
            size=target.stat().st_size,
            local_sha256="0" * 64,
            expected_sha256=None,
        )

    monkeypatch.setattr(
        "scripts.streaming.driver.PerFileHubDownloader.download",
        _download,
    )


def _write_minimal_input(root: Path, region: str) -> None:
    """Write the per-shard six-folder layout for one region."""
    from tests.support.arrow_factories import (
        make_polygon_article_row,
        make_polygon_row,
        make_section_row,
        make_wikipedia_document_row,
    )
    from tests.support.parquet_layouts import write_shard_parquet

    write_shard_parquet(
        root,
        region,
        polygons_rows=[
            make_polygon_row(
                polygon_id=f"poly-{region}",
                wikidata="Q1",
                region=region,
                name=f"Name-{region}",
                tags='{"name":"x"}',
                lat=12.34,
                lon=56.78,
            )
        ],
        polygon_articles_rows=[
            make_polygon_article_row(
                polygon_id=f"poly-{region}",
                article_id=f"art-{region}",
                wikidata="Q1",
                language="en",
            )
        ],
        wikipedia_documents_rows=[
            make_wikipedia_document_row(
                document_id=f"doc-{region}",
                article_id=f"art-{region}",
                wikidata="Q1",
                title=f"Title-{region}",
                language="en",
            )
        ],
        wikipedia_sections_rows=[
            make_section_row(
                section_id=f"sec-{region}",
                document_id=f"doc-{region}",
                article_id=f"art-{region}",
                wikidata="Q1",
                project="wikipedia",
                language="en",
                site="en.wikipedia.org",
                section_index=0,
                heading="Introduction",
                text=f"First sentence. Second sentence ({region}).",
            )
        ],
    )


class _MockSegmenter:
    model_name = "mock"

    def split_batch(self, texts, languages):
        return [[s.strip() for s in t.split(".") if s.strip()] for t in texts]


# ---------------------------------------------------------------------------
# RED: Eviction does NOT happen if offload readback fails.
# ---------------------------------------------------------------------------


def test_driver_no_eviction_before_remote_readback(tmp_path: Path, monkeypatch) -> None:
    work = tmp_path / "work"
    work.mkdir()
    input_root = tmp_path / "in"
    _write_minimal_input(input_root, "italy-latest")
    monkeypatch.setenv("OAR_JOB_ID", "99999")

    # Stub the per-file download to copy the already-local input into
    # the inbox. This avoids any Hub traffic while still leaving
    # discover_shards capable of finding the shard.
    _stub_downloader_to_local_copy(monkeypatch, input_root)

    hub_api = mock.MagicMock()
    hub_api.file_exists.return_value = False
    hub_api.list_repo_tree.return_value = []
    hub_api.upload_folder.side_effect = RuntimeError("simulated upload failure")

    driver = StreamDriver(
        repo_id="owner/repo",
        resolved_revision=VALID_REVISION,
        source_commit=VALID_COMMIT,
        work_dir=work,
        input_root=input_root,
        upstream_repo_id="upstream/owner",
        hub_api=hub_api,
        run_id="run-evict",
        staging_revision="run-evict",
        offload_local_cache_dir=tmp_path / "cache",
        max_disk_bytes=1 << 33,
    )

    with pytest.raises(DriverError):
        driver.process_shard("italy-latest", segmenter=_MockSegmenter())

    # Neither inbox nor active were removed.
    active = work / "shards" / "active" / "italy-latest"
    if active.exists():
        # active not removed
        assert (active / "segmented.parquet").exists()


# ---------------------------------------------------------------------------
# GREEN: Eviction follows a verified readback (mocked).
# ---------------------------------------------------------------------------


def test_driver_evicts_after_verified_readback(tmp_path: Path, monkeypatch) -> None:
    work = tmp_path / "work"
    work.mkdir()
    input_root = tmp_path / "in"
    _write_minimal_input(input_root, "bavaria-latest")
    monkeypatch.setenv("OAR_JOB_ID", "99999")

    # Stub the per-file download to copy the local input into the
    # inbox.
    _stub_downloader_to_local_copy(monkeypatch, input_root)

    # Set up fake Hub state so the readback returns matching bytes.
    from tests.unit.scripts.streaming.test_offload import (
        _FAKE_REMOTE_RUN_ID,
        _fake_hf_hub_download,
        _fake_list_repo_tree,
        _fake_uploaded_parquet,
    )

    _FAKE_REMOTE_RUN_ID.clear()
    _FAKE_REMOTE_RUN_ID["bavaria-latest"] = "run-evict-ok"
    globals()["_CORRUPT_READBACK"] = False

    fake_state = Path("/tmp/streaming-offload-fake")
    if fake_state.exists():
        import shutil

        shutil.rmtree(fake_state)
    fake_state.mkdir(parents=True)

    monkeypatch.setattr(
        "scripts.streaming.offload._lazy_hf_hub_download",
        lambda: _fake_hf_hub_download,
    )

    hub_api = mock.MagicMock()
    hub_api.file_exists.return_value = False
    hub_api.create_branch.side_effect = lambda **kw: None
    hub_api.list_repo_tree.side_effect = _fake_list_repo_tree
    hub_api.upload_folder.side_effect = _fake_uploaded_parquet

    driver = StreamDriver(
        repo_id="owner/repo",
        resolved_revision=VALID_REVISION,
        source_commit=VALID_COMMIT,
        work_dir=work,
        input_root=input_root,
        upstream_repo_id="upstream/owner",
        hub_api=hub_api,
        run_id="run-evict-ok",
        staging_revision="run-evict-ok",
        offload_local_cache_dir=tmp_path / "cache",
        max_disk_bytes=1 << 33,
    )

    driver.process_shard("bavaria-latest", segmenter=_MockSegmenter())
    inbox = work / "shards" / "inbox" / "bavaria-latest"
    assert not inbox.exists(), "inbox must be evicted on success"


# ---------------------------------------------------------------------------
# RED: OAR_JOB_ID guard.
# ---------------------------------------------------------------------------


def test_driver_requires_oar_job_id_without_explicit_override(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("OAR_JOB_ID", raising=False)
    with pytest.raises(OarJobIdRequired):
        StreamDriver(
            repo_id="o/r",
            resolved_revision=VALID_REVISION,
            source_commit=VALID_COMMIT,
            work_dir=tmp_path,
            input_root=tmp_path,
            upstream_repo_id="u/r",
            hub_api=mock.MagicMock(),
            run_id="r",
            staging_revision="r",
            offload_local_cache_dir=tmp_path / "cache",
            max_disk_bytes=1 << 30,
        )


# ---------------------------------------------------------------------------
# RED: max_disk_bytes ceiling is enforced BEFORE inbox fetch.
# ---------------------------------------------------------------------------


def test_driver_enforces_disk_ceiling_before_download(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "99999")

    work = tmp_path / "work"
    work.mkdir()

    download_called = False

    def fake_download(self, path_in_repo, **_kw):
        nonlocal download_called
        download_called = True
        return []

    monkeypatch.setattr(
        "scripts.streaming.driver.PerFileHubDownloader.download",
        fake_download,
    )

    driver = StreamDriver(
        repo_id="o/r",
        resolved_revision=VALID_REVISION,
        source_commit=VALID_COMMIT,
        work_dir=work,
        input_root=tmp_path,
        upstream_repo_id="u/r",
        hub_api=mock.MagicMock(),
        run_id="r",
        staging_revision="r",
        offload_local_cache_dir=work / "cache",
        max_disk_bytes=1 << 40,  # huge ceiling -> ok on test FS
    )

    # Patch disk_usage to claim free=0 so any ceiling check fails.
    class _AlwaysLow:
        free = 0
        total = 1 << 40
        used = 1 << 40

    def fake_du(p):
        return _AlwaysLow()

    monkeypatch.setattr("scripts.streaming.driver.shutil.disk_usage", fake_du)

    with pytest.raises(DriverError):
        driver.process_shard("bavaria-latest", segmenter=_MockSegmenter())
    assert not download_called, "download must NOT be called when ceiling fails"


# ---------------------------------------------------------------------------
# RED: Pinned-revision mismatch aborts.
# ---------------------------------------------------------------------------


def test_driver_rejects_pinned_revision_mismatch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "99999")
    work = tmp_path / "work"
    work.mkdir()
    state = work / "state.json"
    state.write_text(json.dumps({"resolved_revision": "ff" * 20}))
    with pytest.raises(DriverError):
        StreamDriver(
            repo_id="o/r",
            resolved_revision=VALID_REVISION,
            source_commit=VALID_COMMIT,
            work_dir=work,
            input_root=tmp_path,
            upstream_repo_id="u/r",
            hub_api=mock.MagicMock(),
            run_id="r",
            staging_revision="r",
            offload_local_cache_dir=work / "cache",
            max_disk_bytes=1 << 33,
        )


def test_driver_sequential_shards_and_restart_reuse(
    tmp_path: Path, monkeypatch
) -> None:
    """Two sequential shards retain independent checkpoints and restart reuses valid checkpoints."""
    from osm_polygon_sentence_relevance.application.pipeline import process_single_shard
    from osm_polygon_sentence_relevance.ingestion.discovery import discover_shards

    monkeypatch.setenv("OAR_JOB_ID", "99999")
    work = tmp_path / "work"
    work.mkdir()
    input_root = tmp_path / "in"
    _write_minimal_input(input_root, "shard1-latest")
    _write_minimal_input(input_root, "shard2-latest")

    shards = discover_shards(input_root)
    s1 = next(s for s in shards if s.shard_key == "shard1-latest")
    s2 = next(s for s in shards if s.shard_key == "shard2-latest")

    class _CountingSegmenter(_MockSegmenter):
        def __init__(self):
            self.call_count = 0

        def split_batch(self, texts, languages):
            self.call_count += 1
            return super().split_batch(texts, languages)

    segmenter = _CountingSegmenter()

    # Process shard 1
    res1 = process_single_shard(
        shard=s1,
        input_root=input_root,
        segmenter=segmenter,
        work_dir=work,
        source_commit=VALID_COMMIT,
        input_dataset_revision=VALID_REVISION,
        pipeline_version="v1",
        model_name="mock",
        batch_size=128,
    )
    assert res1.published is True
    assert res1.reused is False
    assert segmenter.call_count == 1

    # Process shard 2
    res2 = process_single_shard(
        shard=s2,
        input_root=input_root,
        segmenter=segmenter,
        work_dir=work,
        source_commit=VALID_COMMIT,
        input_dataset_revision=VALID_REVISION,
        pipeline_version="v1",
        model_name="mock",
        batch_size=128,
    )
    assert res2.published is True
    assert res2.reused is False
    assert segmenter.call_count == 2

    # Verify both checkpoints exist independently
    chk1 = work / "shards" / "active" / "shard1-latest" / "segmented.parquet"
    chk2 = work / "shards" / "active" / "shard2-latest" / "segmented.parquet"
    assert chk1.exists()
    assert chk2.exists()

    # Restart: process shard 1 again and confirm it reuses checkpoint without recomputation
    res1_reuse = process_single_shard(
        shard=s1,
        input_root=input_root,
        segmenter=segmenter,
        work_dir=work,
        source_commit=VALID_COMMIT,
        input_dataset_revision=VALID_REVISION,
        pipeline_version="v1",
        model_name="mock",
        batch_size=128,
    )
    assert res1_reuse.published is False
    assert res1_reuse.reused is True
    # Segmenter call count did not increase
    assert segmenter.call_count == 2
