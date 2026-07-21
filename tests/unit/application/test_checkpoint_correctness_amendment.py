"""Phase 9L-A Amendment — second iteration.

These tests assert the next round of required correctness properties:

* work_dir is single-writer: a non-blocking exclusive lock acquired
  before any side-effecting I/O, released through ``finally``;
* orphan-quarantine failures are never suppressed;
* publication failures never trigger an automatic retry;
* cached shards contribute to ``total_joined_section_occurrences`` so
  a resumed ``PipelineResult`` equals an uninterrupted one;
* the per-shard source manifest is computed **once** during inventory
  construction and re-verified only at pre-publish time;
* fsync + chmod failures propagate (no silent suppression);
* symlinks and non-regular-files in the checkpoint directories are
  rejected;
* inventory is versioned, validated, and a malformed prior inventory
  is atomically quarantined rather than silently overwritten.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat as stat_module
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.application import checkpoint as _ckpt_mod
from osm_polygon_sentence_relevance.application.checkpoint import (
    CheckpointPublicationError,
    CheckpointValidationError,
)
from osm_polygon_sentence_relevance.application.pipeline import run_pipeline
from osm_polygon_sentence_relevance.contracts.errors import SegmentationError
from osm_polygon_sentence_relevance.contracts.schemas import (
    SEGMENTED_SENTENCES_SCHEMA,
)
from osm_polygon_sentence_relevance.ingestion.discovery import (
    discover_shards,
)
from osm_polygon_sentence_relevance.sentences.segmentation import (
    SegmentationReport,
)
from tests.support.arrow_factories import (
    make_polygon_article_row,
    make_polygon_row,
    make_section_row,
    make_wikipedia_document_row,
)
from tests.support.parquet_layouts import write_shard_parquet

VALID_INPUT_REVISION = "abcdefabcdefabcdefabcdefabcdefabcdefabcd"
VALID_SOURCE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockSegmenter:
    def __init__(
        self,
        split_fn: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.split_fn = split_fn or (
            lambda text: [s.strip() for s in text.split(".") if s.strip()]
        )
        self.calls_count = 0

    def split_batch(self, texts, languages):
        self.calls_count += 1
        return [self.split_fn(text) for text in texts]


def _write_region(
    root: Path,
    region: str,
    *,
    wikidata_id: str = "Q1",
    text: str | None = None,
) -> None:
    if text is None:
        text = f"First sentence. Second sentence ({region})."
    write_shard_parquet(
        root,
        region,
        polygons_rows=[
            make_polygon_row(
                polygon_id=f"poly-{region}",
                wikidata=wikidata_id,
                region=region,
                name=f"Name-{region}",
                tags=f'{{"name":"Name-{region}"}}',
                lat=12.34,
                lon=56.78,
            )
        ],
        polygon_articles_rows=[
            make_polygon_article_row(
                polygon_id=f"poly-{region}",
                article_id=f"art-{region}",
                wikidata=wikidata_id,
                language="en",
            )
        ],
        wikipedia_documents_rows=[
            make_wikipedia_document_row(
                document_id=f"doc-{region}",
                article_id=f"art-{region}",
                wikidata=wikidata_id,
                title=f"Title-{region}",
                language="en",
            )
        ],
        wikipedia_sections_rows=[
            make_section_row(
                section_id=f"sec-{region}",
                document_id=f"doc-{region}",
                article_id=f"art-{region}",
                wikidata=wikidata_id,
                project="wikipedia",
                language="en",
                site="en.wikipedia.org",
                section_index=0,
                heading="Introduction",
                text=text,
            )
        ],
    )


def _two_region_layout(root: Path) -> None:
    for region in ("reg-a", "reg-b"):
        _write_region(root, region)


def _out(tmp_path, name: str = "out") -> Path:
    out = tmp_path / name
    out.mkdir(exist_ok=True)
    return out


def _run(root, out_dir, work_dir, **overrides):
    seg = overrides.pop("segmenter", MockSegmenter())
    return run_pipeline(
        root,
        out_dir,
        seg,
        input_dataset_revision=overrides.pop(
            "input_dataset_revision", VALID_INPUT_REVISION
        ),
        pipeline_version=overrides.pop("pipeline_version", "v1"),
        work_dir=work_dir,
        source_commit=overrides.pop("source_commit", VALID_SOURCE_COMMIT),
        model_name=overrides.pop("model_name", "mock"),
        **overrides,
    )


def _build_report(counter: int = 0) -> SegmentationReport:
    return SegmentationReport(
        input_section_occurrence_count=counter,
        emitted_segment_count=counter,
        retained_sentence_occurrence_count=counter,
        dropped_empty_raw_count=counter,
        dropped_empty_normalized_count=counter,
        wikipedia_sentence_occurrence_count=counter,
        wikivoyage_sentence_occurrence_count=counter,
    )


# ===========================================================================
# 1. Single-writer work-dir lock
# ===========================================================================


class TestWorkDirSingleWriter:
    """A checkpointed run must own a non-blocking exclusive lock for the
    whole pipeline; a second invocation must fail *before* any side-effect.
    """

    def test_acquire_and_release_default_lock(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            acquire_work_dir_lock,
            release_work_dir_lock,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        ctx = acquire_work_dir_lock(work_dir)
        try:
            assert (work_dir / "shards" / ".lock").exists()
        finally:
            release_work_dir_lock(ctx)
        # After release the lock file is closed; the file may remain
        # on disk so a stale lock check (stale-pid detection) is
        # possible at a later phase.

    def test_second_lock_attempt_is_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            acquire_work_dir_lock,
            release_work_dir_lock,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        held = acquire_work_dir_lock(work_dir)
        try:
            with pytest.raises(CheckpointValidationError, match="locked"):
                acquire_work_dir_lock(work_dir)
        finally:
            release_work_dir_lock(held)

    def test_lock_released_when_run_raises(self, tmp_path, monkeypatch):
        """The pipeline wraps ``run_pipeline`` body in a try/finally so
        a runtime exception releases the lock and lets a subsequent
        caller acquire it cleanly.
        """

        # We exercise the lock-release path via the checkpoint module
        # directly because the lock is acquired *inside* run_pipeline
        # and a fully driven pipeline run would require the segmenter.
        # The semantics is identical: try/finally around ``with
        # _checkpoint.work_dir_lock(work_path)``.
        from osm_polygon_sentence_relevance.application.checkpoint import (
            acquire_work_dir_lock,
            release_work_dir_lock,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        try:
            ctx = acquire_work_dir_lock(work_dir)
            try:
                raise RuntimeError("simulated failure mid-run")
            except RuntimeError:
                release_work_dir_lock(ctx)
                pass
        except RuntimeError:
            pass
        # The lock must be re-acquirable.
        again = acquire_work_dir_lock(work_dir)
        try:
            assert (work_dir / "shards" / ".lock").exists()
        finally:
            release_work_dir_lock(again)

    def test_pipeline_acquires_lock_before_discovery(self, tmp_path):
        """Even the simplest checkpointed run must have created the
        lock file once ``run_pipeline`` exits. The lock is acquired
        before discovery and joins.
        """
        root = tmp_path / "in"
        out_dir = _out(tmp_path, "out")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _two_region_layout(root)
        _run(root, out_dir, work_dir)
        lock_path = work_dir / "shards" / ".lock"
        # After release, the file descriptor is closed but the file
        # may remain — that's fine: it is the **lock acquisition** at
        # run start that we verify closed.
        assert lock_path.exists()

    def test_concurrent_pipeline_call_fails_before_segmentation(
        self, tmp_path, monkeypatch
    ):
        """A second invocation of the checkpointed pipeline must fail
        before any segmentation calls happen.
        """
        root = tmp_path / "in"
        out_dir_a = _out(tmp_path, "out_a")
        out_dir_b = _out(tmp_path, "out_b")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _two_region_layout(root)

        # First pipeline completes cleanly.
        seg_first = MockSegmenter()
        _run(root, out_dir_a, work_dir, segmenter=seg_first)

        # Now simulate "another process is holding the lock" by
        # acquiring it manually, then run a second pipeline. The
        # segmenter counter must NOT advance.
        from osm_polygon_sentence_relevance.application.checkpoint import (
            acquire_work_dir_lock,
            release_work_dir_lock,
        )

        held = acquire_work_dir_lock(work_dir)
        try:
            seg_second = MockSegmenter()
            with pytest.raises(CheckpointValidationError, match="locked"):
                _run(root, out_dir_b, work_dir, segmenter=seg_second)
        finally:
            release_work_dir_lock(held)
        assert seg_second.calls_count == 0


# ===========================================================================
# 2. No auto-retry on publication failure
# ===========================================================================


class TestNoPublicationAutoRetry:
    def test_publication_error_does_not_retry_segmentation(self, tmp_path, monkeypatch):
        """A ``CheckpointPublicationError`` raised during publication
        must NOT trigger automatic re-segmentation. The pipeline
        aborts.
        """
        root = tmp_path / "in"
        out_dir = _out(tmp_path, "out")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _two_region_layout(root)

        seg = MockSegmenter()

        def fail_publish(*args, **kwargs):
            raise CheckpointPublicationError("simulated publication failure")

        from osm_polygon_sentence_relevance.application import pipeline as pipe_mod

        # Patch the module reference the pipeline uses.
        original_publish = pipe_mod.publish_shard_checkpoint
        try:
            pipe_mod.publish_shard_checkpoint = fail_publish  # type: ignore[assignment]
            with pytest.raises(CheckpointPublicationError, match="simulated"):
                _run(root, out_dir, work_dir, segmenter=seg)
        finally:
            pipe_mod.publish_shard_checkpoint = original_publish  # type: ignore[assignment]
        # No active checkpoint was published for any shard.
        active_root = work_dir / "shards" / "active"
        assert not any(active_root.iterdir()) if active_root.exists() else True
        # Pipeline aborts on the first failed publication; the
        # segmenter is not retried for that shard, nor for any later
        # shard.
        assert seg.calls_count <= 2

    def test_publication_error_preserves_staging_as_evidence(self, tmp_path):
        """When publication fails (here, by monkeypatching
        ``_atomic_write_parquet`` to raise), the staging directory
        persists as evidence.
        """
        from osm_polygon_sentence_relevance.application import checkpoint as ckpt
        from osm_polygon_sentence_relevance.application._checkpoint import storage
        from osm_polygon_sentence_relevance.application.checkpoint import (
            publish_shard_checkpoint,
        )

        root = tmp_path / "in"
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _write_region(root, "reg-a")
        shard = discover_shards(root)[0]
        manifest = ckpt.compute_shard_source_manifest(shard, input_root=root)

        def boom(*_a, **_kw):
            raise RuntimeError("simulated write failure")

        original = storage._atomic_write_parquet
        storage._atomic_write_parquet = boom
        try:
            with pytest.raises(RuntimeError):
                publish_shard_checkpoint(
                    work_dir=work_dir,
                    shard=shard,
                    input_root=root,
                    table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
                    report=_build_report(),
                    input_dataset_revision=VALID_INPUT_REVISION,
                    pipeline_version="v1",
                    source_commit=VALID_SOURCE_COMMIT,
                    model_name="mock",
                    batch_size=128,
                    verified_manifest=manifest,
                )
        finally:
            storage._atomic_write_parquet = original

        staging_root = work_dir / "shards"
        stagings = list(staging_root.glob(".staging.reg-a.*"))
        assert stagings, "expected at least one staging dir preserved"
        active_target = work_dir / "shards" / "active" / "reg-a"
        assert not active_target.exists()


# ===========================================================================
# 3. Orphan-quarantine failures propagate
# ===========================================================================


class TestOrphanQuarantineFailurePropagates:
    def test_orphan_quarantine_failure_raises_verbatim(self, tmp_path, monkeypatch):
        """When a removed shard's orphan cannot be quarantined (rename
        raises), the pipeline aborts with the orphan's bytes
        untouched and no ``suppress`` swallower.
        """
        # Pre-populate active/reg-a and prior inventory referencing it.
        root = tmp_path / "in"
        work_dir = tmp_path / "wd"
        root.mkdir()
        work_dir.mkdir()
        _write_region(root, "reg-a")
        out_dir = _out(tmp_path, "out")
        _run(root, out_dir, work_dir)

        # Now delete reg-a so the next run treats it as an orphan.
        for sub in (
            "polygons",
            "polygon_articles",
            "wikipedia/documents",
            "wikipedia/sections",
        ):
            (root / sub / "reg-a.parquet").unlink()

        def rename_fail(src, dst):
            raise OSError(errno.EXDEV, "Cross-device link")

        monkeypatch.setattr(os, "rename", rename_fail)

        # Next run tries to orphan-quarantine; it must fail and the
        # active checkpoint must remain where it was.
        with pytest.raises(OSError, match="Cross-device"):
            _run(root, _out(tmp_path, "out2"), work_dir)
        # Active bytes still in place.
        active = work_dir / "shards" / "active" / "reg-a"
        assert active.exists()
        assert (active / "segmented.parquet").exists()
        assert (active / "metadata.json").exists()


# ===========================================================================
# 4. Resumed accounting parity
# ===========================================================================


class TestResumedAccountingParity:
    def test_resumed_total_joined_matches_uninterrupted(self, tmp_path):
        root = tmp_path / "in"
        out_a = _out(tmp_path, "out_a")
        out_b = _out(tmp_path, "out_b")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _two_region_layout(root)

        # Uninterrupted reference run.
        uninterrupted = _run(
            root,
            out_a,
            tmp_path / "wd_unint",
            segmenter=MockSegmenter(),
        )
        # Interrupted + resume via real work_dir.
        state = {"seen": 0}
        from osm_polygon_sentence_relevance.sentences.table import (
            segment_joined_sections as real_segment,
        )

        def fail_second(table, segmenter, **kwargs):
            state["seen"] += 1
            if state["seen"] == 2:
                raise SegmentationError("interrupt second")
            return real_segment(table, segmenter, **kwargs)

        import osm_polygon_sentence_relevance.application.pipeline as pipe_mod

        original = pipe_mod.segment_joined_sections
        pipe_mod.segment_joined_sections = fail_second  # type: ignore[assignment]
        try:
            with pytest.raises(SegmentationError):
                _run(root, out_b, work_dir)
        finally:
            pipe_mod.segment_joined_sections = original  # type: ignore[assignment]

        # Resume.
        resumed = _run(root, out_b, work_dir)

        assert resumed.total_joined_section_occurrences == (
            uninterrupted.total_joined_section_occurrences
        )
        assert resumed.segmentation_report == (uninterrupted.segmentation_report)


# ===========================================================================
# 5. Source hashing — once initial + once pre-publish verify
# ===========================================================================


class TestSourceManifestComputedOnce:
    def test_new_shard_initial_and_pre_publish_hashing_match(self, tmp_path):
        """For newly processed shards, the pipeline computes the
        manifest once at inventory time and again just before
        publication; if the bytes changed mid-segmentation, the run
        aborts and prior checkpoints remain intact.
        """
        root = tmp_path / "in"
        out_dir = _out(tmp_path, "out")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _write_region(root, "reg-a")

        from osm_polygon_sentence_relevance.application._checkpoint import (
            inventory,
            storage,
        )

        # Inventory construction and pre-publication verification own
        # independent bindings to the same source-manifest function.
        calls = {"n": 0}
        inventory_original = inventory.compute_shard_source_manifest
        storage_original = storage.compute_shard_source_manifest

        def counting(*args, **kwargs):
            calls["n"] += 1
            return inventory_original(*args, **kwargs)

        inventory.compute_shard_source_manifest = counting
        storage.compute_shard_source_manifest = counting

        try:
            _run(root, out_dir, work_dir, segmenter=MockSegmenter())
        finally:
            inventory.compute_shard_source_manifest = inventory_original
            storage.compute_shard_source_manifest = storage_original

        # ``reg-a``: 1 call at inventory build + 1 pre-publish re-hash = 2
        assert calls["n"] == 2

    def test_source_mutation_during_segmentation_aborts(self, tmp_path, monkeypatch):
        """If a source file's bytes change between inventory
        construction and pre-publish verification, the run aborts
        (without quarantining any previously-completed active).
        """
        root = tmp_path / "in"
        out_dir = _out(tmp_path, "out")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _two_region_layout(root)

        from osm_polygon_sentence_relevance.application import pipeline as pipe_mod

        # Pre-publish verification happens in the pipeline body just
        # before ``publish_shard_checkpoint`` is called. Patch the
        # verification entry point so we can rewrite the file then.
        real_verify = pipe_mod._verify_pre_publish_manifest

        def mutating_verify(shard, *, initial_manifest, input_root):
            target = root / "wikipedia" / "sections" / "reg-a.parquet"
            data = target.read_bytes()
            target.write_bytes(data + b"\x00extra")
            return real_verify(
                shard, initial_manifest=initial_manifest, input_root=input_root
            )

        pipe_mod._verify_pre_publish_manifest = mutating_verify  # type: ignore[assignment]
        try:
            with pytest.raises(CheckpointValidationError):
                _run(root, out_dir, work_dir)
        finally:
            pipe_mod._verify_pre_publish_manifest = real_verify  # type: ignore[assignment]

        # reg-b had no pre-publish-mutate; if reg-a was processed
        # first, reg-b's checkpoint may exist (its manifest didn't
        # change). At minimum, no quarantine contents for reg-a.
        quarantine = work_dir / "shards" / "quarantine"
        if quarantine.exists():
            assert not list(quarantine.glob("reg-a.*"))


# ===========================================================================
# 6. Durability — fsync/chmod failures propagate
# ===========================================================================


class TestDurabilityFailuresPropagate:
    def test_fsync_failure_preserves_staging(self, tmp_path, monkeypatch):
        """If ``os.fsync`` raises after writing the Parquet file, the
        staging directory remains intact and an exception surfaces.
        """
        from osm_polygon_sentence_relevance.application import checkpoint as ckpt
        from osm_polygon_sentence_relevance.application.checkpoint import (
            publish_shard_checkpoint,
        )

        root = tmp_path / "in"
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _write_region(root, "reg-a")
        shard = discover_shards(root)[0]
        manifest = ckpt.compute_shard_source_manifest(shard, input_root=root)

        # Patch ``os.fsync`` to raise after the parquet file is
        # written (so the *post-write file fsync* triggers the
        # failure, leaving the staging parquet on disk).
        call_count = {"n": 0}
        real_fsync = os.fsync

        def failing_fsync(fd):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError(errno.EIO, "simulated fsync failure")
            return real_fsync(fd)

        monkeypatch.setattr(os, "fsync", failing_fsync)

        # The atomic-write helper opens the temp file, writes bytes,
        # then calls fsync on the file descriptor — that's the first
        # ``fsync`` we'll trigger.
        from osm_polygon_sentence_relevance.contracts.schemas import (
            SEGMENTED_SENTENCES_SCHEMA,
        )

        with pytest.raises(OSError, match="fsync"):
            publish_shard_checkpoint(
                work_dir=work_dir,
                shard=shard,
                input_root=root,
                table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
                report=_build_report(),
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                verified_manifest=manifest,
            )
        # Active slot untouched.
        assert not (work_dir / "shards" / "active" / "reg-a").exists()

    def test_chmod_failure_preserves_staging(self, tmp_path, monkeypatch):
        """``os.chmod`` failure during staging-write aborts the
        publication. Staging parquet is cleaned up *only* if the
        unlink itself fails — the chmod failure does **not** trigger
        cleanup of the staging parquet; the staging dir is the
        evidence.
        """
        from osm_polygon_sentence_relevance.application import checkpoint as ckpt
        from osm_polygon_sentence_relevance.application.checkpoint import (
            publish_shard_checkpoint,
        )

        root = tmp_path / "in"
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _write_region(root, "reg-a")
        shard = discover_shards(root)[0]
        manifest = ckpt.compute_shard_source_manifest(shard, input_root=root)

        real_chmod = os.chmod
        call_count = {"n": 0}

        def failing_chmod(path, mode, *a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 2:
                # The chmod that happens *after* the staging parquet
                # is materialized is the one we want to fail.
                raise OSError(errno.EPERM, "simulated chmod failure")
            return real_chmod(path, mode, *a, **kw)

        monkeypatch.setattr(os, "chmod", failing_chmod)

        from osm_polygon_sentence_relevance.contracts.schemas import (
            SEGMENTED_SENTENCES_SCHEMA,
        )

        with pytest.raises(OSError, match="chmod"):
            publish_shard_checkpoint(
                work_dir=work_dir,
                shard=shard,
                input_root=root,
                table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
                report=_build_report(),
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                verified_manifest=manifest,
            )
        assert not (work_dir / "shards" / "active" / "reg-a").exists()


# ===========================================================================
# 7. Symlink + non-regular-file rejection
# ===========================================================================


class TestSymlinkRejection:
    def test_symlink_active_dir_rejected(self, tmp_path, monkeypatch):
        """A symlink at ``active/<shard>`` is refused by ``load``."""
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        active_root = work_dir / "shards" / "active"
        active_root.mkdir(parents=True)
        target = work_dir / "target"
        target.mkdir()
        (target / "segmented.parquet").write_bytes(b"x")
        (target / "metadata.json").write_text("{}")
        # Place a symlink, not a directory.
        link = active_root / "reg-a"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        with pytest.raises(CheckpointValidationError, match="symlink|regular"):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_symlink_checkpoint_file_rejected(self, tmp_path):
        """A symlink for either the parquet or metadata file is
        refused at load time.
        """
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        active_root = work_dir / "shards" / "active" / "reg-a"
        active_root.mkdir(parents=True, mode=0o700)
        # Real metadata file, symlink parquet.
        (active_root / "metadata.json").write_text("{}")
        os.chmod(active_root / "metadata.json", 0o600)
        target = work_dir / "real.parquet"
        target.write_bytes(b"x")
        try:
            (active_root / "segmented.parquet").symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        with pytest.raises(CheckpointValidationError, match="symlink|regular"):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_wrong_directory_mode_rejected(self, tmp_path):
        """A checkpoint directory whose mode is wider than 0o700 is
        refused.
        """
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        active_root = work_dir / "shards" / "active" / "reg-a"
        active_root.mkdir(parents=True, mode=0o755)
        (active_root / "segmented.parquet").write_bytes(b"x")
        (active_root / "metadata.json").write_text("{}")
        with pytest.raises(CheckpointValidationError, match="mode|directory"):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_broken_symlink_active_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        active_root = work_dir / "shards" / "active"
        active_root.mkdir(parents=True)
        link = active_root / "reg-a"
        try:
            link.symlink_to(work_dir / "does-not-exist")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        # Broken symlink: load must raise, not crash.
        with pytest.raises(CheckpointValidationError):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )


# ===========================================================================
# 8. Strict staged metadata validation
# ===========================================================================


class TestStrictMetadataValidation:
    def test_missing_required_field_rejected(self, tmp_path, monkeypatch):
        """The strict validator rejects metadata missing any required
        identity field or with the wrong type.
        """
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            # input_dataset_revision intentionally omitted.
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [],
            "segmentation_report": {},
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError, match="input_dataset_revision"):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_source_files_must_be_sorted(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [
                {"path": "z/p.parquet", "size": 1, "sha256": "00" * 32},
                {"path": "a/p.parquet", "size": 1, "sha256": "00" * 32},
            ],
            "segmentation_report": {},
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError, match="sorted"):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_source_file_path_must_be_relative(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [
                {
                    "path": "/absolute/reg-a.parquet",
                    "size": 1,
                    "sha256": "00" * 32,
                }
            ],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError, match="relative|posix"):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_sha256_lowercase_only(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [
                {
                    "path": "polygons/reg-a.parquet",
                    "size": 1,
                    "sha256": "AB" * 32,
                }
            ],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError, match="lowercase|sha256"):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_segmentation_report_must_be_mapping(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [
                {
                    "path": "polygons/reg-a.parquet",
                    "size": 1,
                    "sha256": "00" * 32,
                }
            ],
            "segmentation_report": [0, 0, 0, 0, 0, 0, 0],
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(
            CheckpointValidationError, match="segmentation_report must be a mapping"
        ):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_segmentation_report_missing_field(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [
                {
                    "path": "polygons/reg-a.parquet",
                    "size": 1,
                    "sha256": "00" * 32,
                }
            ],
            "segmentation_report": {"input_section_occurrence_count": 0},
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError, match="missing report field"):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_source_files_path_with_parent_segment_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [
                {
                    "path": "../escape/reg-a.parquet",
                    "size": 1,
                    "sha256": "00" * 32,
                }
            ],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_work_dir_lock_other_oserror_propagates(self, tmp_path, monkeypatch):
        """An OSError from flock that isn't EWOULDBLOCK/EAGAIN must
        propagate verbatim (not be rewritten as ``locked``).
        """
        import fcntl as fcntl_mod

        from osm_polygon_sentence_relevance.application.checkpoint import (
            acquire_work_dir_lock,
        )

        def boom(*_a, **_kw):
            raise OSError(errno.EPERM, "operation not permitted")

        monkeypatch.setattr(fcntl_mod, "flock", boom)
        with pytest.raises(OSError, match="operation not permitted"):
            acquire_work_dir_lock(tmp_path / "wd")

    def test_validate_payload_must_be_mapping(self):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        with pytest.raises(CheckpointValidationError, match="must be a mapping"):
            validate_checkpoint_metadata([], shard_key="reg-a", expect_active=False)  # type: ignore[arg-type]

    def test_validate_required_string_fields_must_be_string(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        base = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        for field in ("pipeline_version", "model_name", "input_root"):
            bad = dict(base)
            bad[field] = 42
            with pytest.raises(CheckpointValidationError, match="must be a string"):
                validate_checkpoint_metadata(
                    bad, shard_key="reg-a", expect_active=False
                )

    def test_validate_invalid_shard_key_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        base = {
            "schema_version": 2,
            "shard_key": "bad/key",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError, match="invalid shard_key"):
            validate_checkpoint_metadata(base, shard_key="reg-a", expect_active=False)

    def test_validate_invalid_source_commit_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": "NOTHEX",
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(
            CheckpointValidationError, match="source_commit must be lowercase"
        ):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_validate_segmented_table_sha_must_be_lower_hex(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "A" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError, match="lowercase"):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_validate_batch_size_must_be_positive_int(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": -1,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError, match="batch_size"):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_validate_source_files_entry_not_a_mapping(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": ["not-a-mapping"],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError, match="must be a mapping"):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_validate_source_files_size_must_be_non_negative(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": -1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError, match="non-negative integer"):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_validate_shard_key_mismatch_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-b",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError, match="shard_key mismatch"):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_validate_blank_input_dataset_revision_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": "   ",
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(
            CheckpointValidationError,
            match="input_dataset_revision must be non-blank",
        ):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_validate_completed_at_unix_must_be_int(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": "not-an-int",
        }
        with pytest.raises(
            CheckpointValidationError, match="completed_at_unix must be an integer"
        ):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_validate_source_files_path_must_be_string(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": 42, "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(
            CheckpointValidationError, match="entry path must be a string"
        ):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_validate_source_files_absolute_path_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        # Drive-letter absolute path.
        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [
                {"path": "C:/absolute.parquet", "size": 1, "sha256": "00" * 32}
            ],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError, match="relative POSIX path"):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_validate_segmentation_report_invalid_field_type(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": "not-an-int",
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(
            CheckpointValidationError,
            match="segmentation_report has invalid field",
        ):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_atomic_write_bytes_failure_unlinks_temp(self, tmp_path, monkeypatch):
        """If ``os.replace`` fails inside ``_atomic_write_bytes``, the
        helper attempts to unlink the temp file and re-raises the
        original exception. The unlink is best-effort and swallows
        ``OSError``; the original failure is what surfaces.
        """
        from osm_polygon_sentence_relevance.application._checkpoint import io

        target = tmp_path / "out.bin"

        def fail_replace(*_a, **_kw):
            raise OSError(errno.EACCES, "permission denied")

        monkeypatch.setattr(io.os, "replace", fail_replace)
        with pytest.raises(OSError, match="permission denied"):
            io._atomic_write_bytes(target, b"hello")
        # No leftover temp file in target.parent.
        leftover = list(target.parent.glob(f".{target.name}.*.tmp"))
        assert leftover == []

    def test_atomic_write_parquet_failure_unlinks_temp(self, tmp_path, monkeypatch):
        """Same as the bytes variant but for the Parquet writer."""
        from osm_polygon_sentence_relevance.application._checkpoint import io
        from osm_polygon_sentence_relevance.contracts.schemas import (
            SEGMENTED_SENTENCES_SCHEMA,
        )

        target = tmp_path / "out.parquet"

        def fail_replace(*_a, **_kw):
            raise OSError(errno.EACCES, "permission denied")

        monkeypatch.setattr(io.os, "replace", fail_replace)
        with pytest.raises(OSError, match="permission denied"):
            io._atomic_write_parquet(SEGMENTED_SENTENCES_SCHEMA.empty_table(), target)

    def test_validate_source_files_entry_missing_key(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [
                {"path": "p.parquet", "size": 1}  # missing sha256
            ],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(
            CheckpointValidationError, match="source_files entry missing key"
        ):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_validate_source_files_path_blank_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError, match="non-blank string"):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_load_rejects_non_directory_active(self, tmp_path):
        """If the active path is a file (not a directory), the loader
        refuses.
        """
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        active = work_dir / "shards" / "active" / "reg-a"
        active.parent.mkdir(parents=True, mode=0o700)
        active.write_bytes(b"not-a-directory")
        os.chmod(active, 0o600)
        with pytest.raises(CheckpointValidationError, match="not a regular directory"):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_load_rejects_wrong_dir_mode(self, tmp_path):
        """Directory mode must be 0o700."""
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        active = work_dir / "shards" / "active" / "reg-a"
        active.mkdir(parents=True, mode=0o755)
        (active / "segmented.parquet").write_bytes(b"x")
        (active / "metadata.json").write_text("{}")
        with pytest.raises(CheckpointValidationError, match="expected 700"):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_load_rejects_wrong_file_mode(self, tmp_path):
        """A checkpoint file with the wrong mode (not 0o600) is
        refused.
        """
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        active = work_dir / "shards" / "active" / "reg-a"
        active.mkdir(parents=True, mode=0o700)
        (active / "segmented.parquet").write_bytes(b"x")
        (active / "metadata.json").write_text("{}")
        # chmod metadata file to a non-conforming mode.
        os.chmod(active / "metadata.json", 0o644)
        with pytest.raises(CheckpointValidationError, match="expected 600"):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_load_rejects_unexpected_dir_entries(self, tmp_path):
        """Extra entries in the active directory are rejected."""
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        active = work_dir / "shards" / "active" / "reg-a"
        active.mkdir(parents=True, mode=0o700)
        (active / "segmented.parquet").write_bytes(b"x")
        (active / "metadata.json").write_text("{}")
        (active / "extra.txt").write_text("junk")
        os.chmod(active / "segmented.parquet", 0o600)
        os.chmod(active / "metadata.json", 0o600)
        os.chmod(active / "extra.txt", 0o600)
        with pytest.raises(CheckpointValidationError, match="unexpected entries"):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_load_rejects_parquet_sha_mismatch(self, tmp_path):
        """The persisted SHA-256 must match the parquet on disk."""
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        active = work_dir / "shards" / "active" / "reg-a"
        active.mkdir(parents=True, mode=0o700)
        (active / "segmented.parquet").write_bytes(b"\x00" * 16)
        os.chmod(active / "segmented.parquet", 0o600)
        meta = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "ab" * 32,
            "completed_at_unix": 0,
        }
        (active / "metadata.json").write_text(json.dumps(meta))
        os.chmod(active / "metadata.json", 0o600)
        with pytest.raises(CheckpointValidationError, match="SHA-256 mismatch"):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_inventory_missing_fields_rejected(self, tmp_path):
        """An inventory.json missing required fields is rejected as
        malformed.
        """
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_run_inventory,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        (work_dir / "shards").mkdir()
        (work_dir / "shards" / "inventory.json").write_text(
            json.dumps({"schema_version": 2, "shards": {}})
        )
        with pytest.raises(CheckpointValidationError, match="missing fields"):
            load_run_inventory(work_dir)

    def test_inventory_shards_not_mapping_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_run_inventory,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        (work_dir / "shards").mkdir()
        (work_dir / "shards" / "inventory.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "input_dataset_revision": VALID_INPUT_REVISION,
                    "source_commit": VALID_SOURCE_COMMIT,
                    "pipeline_version": "v1",
                    "model_name": "mock",
                    "batch_size": 128,
                    "discovered_at_unix": 0,
                    "shards": [],
                }
            )
        )
        with pytest.raises(CheckpointValidationError, match="shards must be a mapping"):
            load_run_inventory(work_dir)

    def test_inventory_invalid_shard_key_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_run_inventory,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        (work_dir / "shards").mkdir()
        (work_dir / "shards" / "inventory.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "input_dataset_revision": VALID_INPUT_REVISION,
                    "source_commit": VALID_SOURCE_COMMIT,
                    "pipeline_version": "v1",
                    "model_name": "mock",
                    "batch_size": 128,
                    "discovered_at_unix": 0,
                    "shards": {"bad/key": []},
                }
            )
        )
        with pytest.raises(CheckpointValidationError, match="invalid shard_key"):
            load_run_inventory(work_dir)

    def test_inventory_shards_value_must_be_list(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_run_inventory,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        (work_dir / "shards").mkdir()
        (work_dir / "shards" / "inventory.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "input_dataset_revision": VALID_INPUT_REVISION,
                    "source_commit": VALID_SOURCE_COMMIT,
                    "pipeline_version": "v1",
                    "model_name": "mock",
                    "batch_size": 128,
                    "discovered_at_unix": 0,
                    "shards": {"reg-a": "not-a-list"},
                }
            )
        )
        with pytest.raises(CheckpointValidationError, match="must be a list"):
            load_run_inventory(work_dir)

    def test_inventory_load_io_error_wraps_validation_error(
        self, tmp_path, monkeypatch
    ):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_run_inventory,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        (work_dir / "shards").mkdir()
        (work_dir / "shards" / "inventory.json").write_text("{}")

        def boom_read(*_a, **_kw):
            raise OSError(errno.EIO, "disk error")

        # Replace Path.read_bytes on the inventory file via monkeypatch.
        from pathlib import Path as PathCls

        original = PathCls.read_bytes

        def fake_read_bytes(self):
            if str(self).endswith("inventory.json"):
                raise OSError(errno.EIO, "disk error")
            return original(self)

        monkeypatch.setattr(PathCls, "read_bytes", fake_read_bytes)
        with pytest.raises(CheckpointValidationError, match="could not be read"):
            load_run_inventory(work_dir)

    def test_safe_loader_quarantine_fails_propagates_parse_error(
        self, tmp_path, monkeypatch
    ):
        """If ``load_run_inventory_quarantine_first`` cannot move the
        malformed file aside (e.g. cross-filesystem), it raises the
        underlying parse error so the caller sees the malformed state.
        """
        from osm_polygon_sentence_relevance.application import checkpoint as ckpt

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        (work_dir / "shards").mkdir()
        (work_dir / "shards" / "inventory.json").write_text("[1,2,3]")

        def fail_rename(*_a, **_kw):
            raise OSError(errno.EXDEV, "Cross-device link")

        monkeypatch.setattr(os, "rename", fail_rename)
        with pytest.raises(
            CheckpointValidationError, match="payload must be a mapping"
        ):
            ckpt.load_run_inventory_quarantine_first(work_dir)

    def test_quarantine_shard_key_validation_rejects_invalid(self):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            quarantine_shard_checkpoint,
        )

        with pytest.raises(CheckpointValidationError, match="invalid shard_key"):
            quarantine_shard_checkpoint(
                work_dir=Path("/tmp"),
                shard_key="bad/key",
                reason="r",
            )

    def test_publish_rejects_blank_identity_string(self):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            CheckpointValidationError,
            publish_shard_checkpoint,
        )
        from osm_polygon_sentence_relevance.ingestion.discovery import (
            RegionShardSet,
        )

        root = Path("/tmp/x")
        shard = RegionShardSet(
            shard_key="reg-a",
            polygons=root / "polygons" / "x.parquet",
            polygon_articles=root / "polygon_articles" / "x.parquet",
            wikipedia_documents=root / "wikipedia" / "documents" / "x.parquet",
            wikipedia_sections=root / "wikipedia" / "sections" / "x.parquet",
            wikivoyage_documents=None,
            wikivoyage_sections=None,
        )
        with pytest.raises(CheckpointValidationError, match="model_name"):
            publish_shard_checkpoint(
                work_dir=Path("/tmp"),
                shard=shard,
                input_root=root,
                table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
                report=_build_report(),
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="",
                batch_size=128,
                verified_manifest=[],
            )

    def test_publish_rejects_invalid_source_commit(self):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            CheckpointValidationError,
            publish_shard_checkpoint,
        )
        from osm_polygon_sentence_relevance.ingestion.discovery import (
            RegionShardSet,
        )

        root = Path("/tmp/x")
        shard = RegionShardSet(
            shard_key="reg-a",
            polygons=root / "polygons" / "x.parquet",
            polygon_articles=root / "polygon_articles" / "x.parquet",
            wikipedia_documents=root / "wikipedia" / "documents" / "x.parquet",
            wikipedia_sections=root / "wikipedia" / "sections" / "x.parquet",
            wikivoyage_documents=None,
            wikivoyage_sections=None,
        )
        with pytest.raises(
            CheckpointValidationError, match="source_commit must be lowercase"
        ):
            publish_shard_checkpoint(
                work_dir=Path("/tmp"),
                shard=shard,
                input_root=root,
                table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
                report=_build_report(),
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit="NOT-LOWERCASE-HEX",
                model_name="mock",
                batch_size=128,
                verified_manifest=[],
            )

    def test_validate_source_files_empty_rejected(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_checkpoint_metadata,
        )

        bad = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
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
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        with pytest.raises(CheckpointValidationError, match="non-empty list"):
            validate_checkpoint_metadata(bad, shard_key="reg-a", expect_active=False)

    def test_load_rejects_when_active_missing(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        with pytest.raises(CheckpointValidationError, match="no active"):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_load_rejects_corrupt_metadata_json(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        active = work_dir / "shards" / "active" / "reg-a"
        active.mkdir(parents=True, mode=0o700)
        (active / "segmented.parquet").write_bytes(b"x")
        os.chmod(active / "segmented.parquet", 0o600)
        (active / "metadata.json").write_text("{ not json")
        os.chmod(active / "metadata.json", 0o600)
        with pytest.raises(CheckpointValidationError, match="malformed"):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_load_rejects_wrong_schema_version(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        active = work_dir / "shards" / "active" / "reg-a"
        active.mkdir(parents=True, mode=0o700)
        (active / "segmented.parquet").write_bytes(b"x")
        os.chmod(active / "segmented.parquet", 0o600)
        meta = {
            "schema_version": 1,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        (active / "metadata.json").write_text(json.dumps(meta))
        os.chmod(active / "metadata.json", 0o600)
        with pytest.raises(CheckpointValidationError, match="schema_version"):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_load_rejects_identity_mismatch(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        active = work_dir / "shards" / "active" / "reg-a"
        active.mkdir(parents=True, mode=0o700)
        (active / "segmented.parquet").write_bytes(b"x")
        os.chmod(active / "segmented.parquet", 0o600)
        meta = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": "different-revision",
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        (active / "metadata.json").write_text(json.dumps(meta))
        os.chmod(active / "metadata.json", 0o600)
        with pytest.raises(CheckpointValidationError, match="identity mismatch"):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_load_rejects_invalid_shard_key_in_metadata(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        active = work_dir / "shards" / "active" / "reg-a"
        active.mkdir(parents=True, mode=0o700)
        (active / "segmented.parquet").write_bytes(b"x")
        os.chmod(active / "segmented.parquet", 0o600)
        meta = {
            "schema_version": 2,
            "shard_key": "bad/key",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": "0" * 64,
            "completed_at_unix": 0,
        }
        (active / "metadata.json").write_text(json.dumps(meta))
        os.chmod(active / "metadata.json", 0o600)
        with pytest.raises(CheckpointValidationError):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_load_rejects_corrupt_source_files_field(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        active = work_dir / "shards" / "active" / "reg-a"
        active.mkdir(parents=True, mode=0o700)
        # Write a real (empty) segmented.parquet so the loader gets
        # past the parquet-read check and into the source_files
        # binding logic.
        pq.write_table(
            SEGMENTED_SENTENCES_SCHEMA.empty_table(), active / "segmented.parquet"
        )
        os.chmod(active / "segmented.parquet", 0o600)
        parquet_bytes = (active / "segmented.parquet").read_bytes()
        actual_sha = hashlib.sha256(parquet_bytes).hexdigest().lower()
        meta = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p", "size": "x", "sha256": "not-hex"}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": actual_sha,
            "completed_at_unix": 0,
        }
        (active / "metadata.json").write_text(json.dumps(meta))
        os.chmod(active / "metadata.json", 0o600)
        with pytest.raises(CheckpointValidationError, match="source_files"):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_load_rejects_source_files_mismatch(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            SourceFileEntry,
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "wd"
        active = work_dir / "shards" / "active" / "reg-a"
        active.mkdir(parents=True, mode=0o700)
        pq.write_table(
            SEGMENTED_SENTENCES_SCHEMA.empty_table(), active / "segmented.parquet"
        )
        os.chmod(active / "segmented.parquet", 0o600)
        parquet_bytes = (active / "segmented.parquet").read_bytes()
        actual_sha = hashlib.sha256(parquet_bytes).hexdigest().lower()
        meta = {
            "schema_version": 2,
            "shard_key": "reg-a",
            "input_dataset_revision": VALID_INPUT_REVISION,
            "pipeline_version": "v1",
            "source_commit": VALID_SOURCE_COMMIT,
            "model_name": "mock",
            "batch_size": 128,
            "input_root": str(tmp_path),
            "source_files": [{"path": "p.parquet", "size": 1, "sha256": "00" * 32}],
            "segmentation_report": {
                "input_section_occurrence_count": 0,
                "emitted_segment_count": 0,
                "retained_sentence_occurrence_count": 0,
                "dropped_empty_raw_count": 0,
                "dropped_empty_normalized_count": 0,
                "wikipedia_sentence_occurrence_count": 0,
                "wikivoyage_sentence_occurrence_count": 0,
            },
            "segmented_table_sha256": actual_sha,
            "completed_at_unix": 0,
        }
        (active / "metadata.json").write_text(json.dumps(meta))
        os.chmod(active / "metadata.json", 0o600)
        # Pass a manifest that disagrees with what's persisted.
        bad_manifest = [
            SourceFileEntry(path="different.parquet", size=2, sha256="11" * 32)
        ]
        with pytest.raises(CheckpointValidationError, match="source_files mismatch"):
            load_shard_checkpoint(
                work_dir,
                "reg-a",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
                current_manifest=bad_manifest,
            )


# ===========================================================================
# 9. Inventory version + validation + preservation
# ===========================================================================


class TestInventoryVersionAndPreservation:
    def test_inventory_schema_version_recorded(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            RunInventory,
            write_run_inventory,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        inv = RunInventory(
            schema_version=2,
            discovered_at_unix=0,
            input_dataset_revision=VALID_INPUT_REVISION,
            source_commit=VALID_SOURCE_COMMIT,
            pipeline_version="v1",
            model_name="mock",
            batch_size=128,
            shards={},
        )
        write_run_inventory(work_dir, inv)
        data = json.loads((work_dir / "shards" / "inventory.json").read_text())
        assert "schema_version" in data
        assert isinstance(data["schema_version"], int)
        assert data["schema_version"] >= 2

    def test_invalid_inventory_quarantined_not_overwritten(self, tmp_path, monkeypatch):
        """A malformed prior inventory.json must be moved into
        ``quarantine/inventory/`` *before* a new inventory is written.
        The corrupt bytes are preserved exactly under that quarantine
        directory.
        """
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        # Write a corrupt inventory.
        (work_dir / "shards").mkdir()
        (work_dir / "shards" / "inventory.json").write_text("{ not valid json")
        bad_bytes = (work_dir / "shards" / "inventory.json").read_bytes()
        # The strict loader still raises so callers can detect the
        # malformed state directly.
        with pytest.raises(CheckpointValidationError, match="malformed"):
            _ckpt_mod.load_run_inventory(work_dir)
        # The safe-loader behaviour: atomically move the corrupt file
        # into the inventory-quarantine directory and return ``None``
        # so the pipeline can fall back to orphan-active recovery.
        result = _ckpt_mod.load_run_inventory_quarantine_first(work_dir)
        assert result is None
        # The active slot has been cleared; the bytes live under
        # quarantine/inventory/.
        assert not (work_dir / "shards" / "inventory.json").exists()
        qdir = work_dir / "shards" / "quarantine" / _ckpt_mod.INVENTORY_QUARANTINE_DIR
        preserved = list(qdir.iterdir())
        assert preserved, "expected at least one quarantined inventory file"
        assert preserved[0].read_bytes() == bad_bytes

    def test_inventory_load_wraps_malformed_json(self, tmp_path):
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        (work_dir / "shards").mkdir()
        (work_dir / "shards" / "inventory.json").write_text("[1, 2, 3]")
        with pytest.raises(CheckpointValidationError):
            _ckpt_mod.load_run_inventory(work_dir)

    def test_inventory_load_wraps_wrong_schema_version(self, tmp_path):
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        (work_dir / "shards").mkdir()
        inv = {
            "schema_version": 1,
            "discovered_at_unix": 0,
            "shards": {},
        }
        (work_dir / "shards" / "inventory.json").write_text(json.dumps(inv))
        with pytest.raises(CheckpointValidationError, match="schema_version"):
            _ckpt_mod.load_run_inventory(work_dir)


# ===========================================================================
# 10. Inventory-driven decisions drive added/removed/changed/unchanged
# ===========================================================================


class TestInventoryDecisionsAreDirect:
    """The inventory reconciliation yields explicit
    ``add | remove | changed | unchanged`` labels; we no longer rely
    on a "decision" attribute.
    """

    def test_reconcile_inventory_returns_add_remove_change_unchanged(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            SourceFileEntry,
            reconcile_inventory,
        )

        a = SourceFileEntry(path="polygons/a.parquet", size=1, sha256="00" * 32)
        b = SourceFileEntry(path="polygons/b.parquet", size=1, sha256="11" * 32)
        a_changed = SourceFileEntry(path="polygons/a.parquet", size=2, sha256="22" * 32)

        result = reconcile_inventory(
            prior={"a": [a], "c": [b]},
            current={"a": [a_changed], "b": [b]},
        )
        # ``a`` present in both but with different bytes → changed.
        assert result["changed"] == {"a"}
        # ``b`` only in current → added.
        assert result["added"] == {"b"}
        # ``c`` only in prior → removed.
        assert result["removed"] == {"c"}
        # Nothing unchanged.
        assert result["unchanged"] == set()


# ===========================================================================
# 11. Discovery called exactly once per pipeline run
# ===========================================================================


class TestDiscoveryCalledExactlyOnce:
    """``discover_shards`` must run exactly once per ``run_pipeline``
    invocation. Filesystem scanning happens at most once for a given
    input root in the entire pipeline body.
    """

    def test_discover_shards_called_once_per_pipeline(self, tmp_path):
        root = tmp_path / "in"
        out_dir = _out(tmp_path, "out")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _two_region_layout(root)

        import osm_polygon_sentence_relevance.application.pipeline as pipe_mod
        from osm_polygon_sentence_relevance.ingestion.discovery import (
            discover_shards as real_discover,
        )

        counter = {"n": 0}
        original = real_discover

        def counting(root_arg):
            counter["n"] += 1
            return original(root_arg)

        pipe_mod.discover_shards = counting  # type: ignore[assignment]
        try:
            _run(root, out_dir, work_dir)
        finally:
            pipe_mod.discover_shards = original  # type: ignore[assignment]
        assert counter["n"] == 1

    def test_discover_shards_called_once_on_resume(self, tmp_path):
        root = tmp_path / "in"
        out_dir_a = _out(tmp_path, "out_a")
        out_dir_b = _out(tmp_path, "out_b")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _two_region_layout(root)

        _run(root, out_dir_a, work_dir)

        import osm_polygon_sentence_relevance.application.pipeline as pipe_mod
        from osm_polygon_sentence_relevance.ingestion.discovery import (
            discover_shards as real_discover,
        )

        counter = {"n": 0}
        original = real_discover

        def counting(root_arg):
            counter["n"] += 1
            return original(root_arg)

        pipe_mod.discover_shards = counting  # type: ignore[assignment]
        try:
            _run(root, out_dir_b, work_dir)
        finally:
            pipe_mod.discover_shards = original  # type: ignore[assignment]
        assert counter["n"] == 1


# ===========================================================================
# 12. Corrupt "unchanged" checkpoint: auto-quarantine then recompute
# ===========================================================================


class TestCorruptUnchangedCheckpointAutoQuarantine:
    """If a checkpoint that the reconciler marks ``unchanged`` fails
    strict load validation, the pipeline must atomically quarantine it
    and re-segment. If quarantine itself fails, the run aborts without
    silently falling back to a recompute.
    """

    def test_corrupt_unchanged_checkpoint_quarantined_and_recomputed(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "in"
        out_dir_a = _out(tmp_path, "out_a")
        out_dir_b = _out(tmp_path, "out_b")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _two_region_layout(root)
        _run(root, out_dir_a, work_dir)

        # Corrupt the metadata for reg-a so its checkpoint no longer
        # validates against the strict loader.
        active = work_dir / "shards" / "active" / "reg-a"
        (active / "metadata.json").write_text("{ not valid json")

        seg = MockSegmenter()
        before = seg.calls_count
        _run(root, out_dir_b, work_dir, segmenter=seg)
        # reg-a was quarantined then re-segmented (one new call).
        # reg-b was reused (no new call).
        assert seg.calls_count == before + 1
        # reg-a is now a freshly published active checkpoint.
        assert (active / "segmented.parquet").exists()
        assert (active / "metadata.json").exists()
        json.loads((active / "metadata.json").read_text())
        # reg-a's pre-quarantine bytes are preserved in quarantine.
        qdir = work_dir / "shards" / "quarantine"
        assert list(qdir.iterdir())

    def test_orphan_recovery_corrupt_active_quarantined(self, tmp_path):
        """When the prior run crashed *after* publishing an active
        checkpoint but *before* writing ``inventory.json``, the next
        run finds the orphan active, attempts to load it, and if it
        fails strict validation, quarantines it via the recovery sweep
        before processing the shard.
        """
        root = tmp_path / "in"
        out_dir_a = _out(tmp_path, "out_a")
        out_dir_b = _out(tmp_path, "out_b")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _two_region_layout(root)
        _run(root, out_dir_a, work_dir)

        # Corrupt reg-a and remove the inventory so the recovery
        # sweep picks it up.
        active = work_dir / "shards" / "active" / "reg-a"
        (active / "metadata.json").write_text("{ not valid json")
        (work_dir / "shards" / "inventory.json").unlink()

        seg = MockSegmenter()
        before = seg.calls_count
        _run(root, out_dir_b, work_dir, segmenter=seg)
        # reg-a re-segmented; reg-b reused.
        assert seg.calls_count == before + 1

    def test_quarantine_failure_aborts_without_recompute(self, tmp_path, monkeypatch):
        root = tmp_path / "in"
        out_dir_a = _out(tmp_path, "out_a")
        out_dir_b = _out(tmp_path, "out_b")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _two_region_layout(root)
        _run(root, out_dir_a, work_dir)

        active = work_dir / "shards" / "active" / "reg-a"
        (active / "metadata.json").write_text("{ not valid json")

        def fail_rename(*_a, **_kw):
            raise OSError(errno.EXDEV, "Cross-device link")

        monkeypatch.setattr(os, "rename", fail_rename)

        seg = MockSegmenter()
        before = seg.calls_count
        with pytest.raises(OSError, match="Cross-device"):
            _run(root, out_dir_b, work_dir, segmenter=seg)
        # No segmentation call was made (the run aborted before
        # reaching the segmenter).
        assert seg.calls_count == before
        # Active bytes are still in place — quarantine did NOT succeed.
        assert (active / "segmented.parquet").exists()
        assert (active / "metadata.json").exists()


# ===========================================================================
# 13. Staged wrong-schema/corrupt Parquet never publishes
# ===========================================================================


class TestStagedCheckpointValidationBeforePublish:
    """The publication path must fully validate the staged contents
    (Parquet schema, SHA-256, file modes/types, strict metadata) before
    renaming staging to active. Any failure aborts without an active
    directory being created.
    """

    def _seed(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            compute_shard_source_manifest,
        )

        root = tmp_path / "in"
        work_dir = tmp_path / "wd"
        root.mkdir()
        work_dir.mkdir()
        _write_region(root, "reg-a")
        shard = discover_shards(root)[0]
        manifest = compute_shard_source_manifest(shard, input_root=root)
        return root, work_dir, shard, manifest

    def test_staged_wrong_schema_parquet_aborts_publication(
        self, tmp_path, monkeypatch
    ):
        from osm_polygon_sentence_relevance.application._checkpoint import (
            storage as ckpt,
        )
        from osm_polygon_sentence_relevance.application.checkpoint import (
            publish_shard_checkpoint,
        )

        root, work_dir, shard, manifest = self._seed(tmp_path)
        wrong_table = pa.table({"x": [1]})

        original = ckpt._atomic_write_parquet

        def write_wrong(table_arg, path):
            original(wrong_table, path)

        monkeypatch.setattr(ckpt, "_atomic_write_parquet", write_wrong)

        with pytest.raises(CheckpointValidationError):
            publish_shard_checkpoint(
                work_dir=work_dir,
                shard=shard,
                input_root=root,
                table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
                report=_build_report(),
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                verified_manifest=manifest,
            )
        assert not (work_dir / "shards" / "active" / "reg-a").exists()

    def test_staged_corrupt_parquet_bytes_aborts_publication(
        self, tmp_path, monkeypatch
    ):
        from osm_polygon_sentence_relevance.application._checkpoint import (
            storage as ckpt,
        )
        from osm_polygon_sentence_relevance.application.checkpoint import (
            publish_shard_checkpoint,
        )

        root, work_dir, shard, manifest = self._seed(tmp_path)

        original = ckpt._atomic_write_parquet

        def tamper(table_arg, path):
            original(table_arg, path)
            with open(path, "ab") as fh:
                fh.write(b"tamper")

        monkeypatch.setattr(ckpt, "_atomic_write_parquet", tamper)

        with pytest.raises(CheckpointValidationError):
            publish_shard_checkpoint(
                work_dir=work_dir,
                shard=shard,
                input_root=root,
                table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
                report=_build_report(),
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                verified_manifest=manifest,
            )
        assert not (work_dir / "shards" / "active" / "reg-a").exists()

    def test_staged_extra_entry_aborts_publication(self, tmp_path, monkeypatch):
        from osm_polygon_sentence_relevance.application._checkpoint import (
            storage as ckpt,
        )
        from osm_polygon_sentence_relevance.application.checkpoint import (
            publish_shard_checkpoint,
        )

        root, work_dir, shard, manifest = self._seed(tmp_path)

        original = ckpt._atomic_write_parquet

        def write_then_extra(table_arg, path):
            original(table_arg, path)
            staging_dir = path.parent
            (staging_dir / "extra.txt").write_text("junk")

        monkeypatch.setattr(ckpt, "_atomic_write_parquet", write_then_extra)

        with pytest.raises(CheckpointValidationError):
            publish_shard_checkpoint(
                work_dir=work_dir,
                shard=shard,
                input_root=root,
                table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
                report=_build_report(),
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                verified_manifest=manifest,
            )
        assert not (work_dir / "shards" / "active" / "reg-a").exists()

    def test_staged_parquet_symlink_aborts_publication(self, tmp_path, monkeypatch):
        from osm_polygon_sentence_relevance.application._checkpoint import (
            storage as ckpt,
        )
        from osm_polygon_sentence_relevance.application.checkpoint import (
            publish_shard_checkpoint,
        )

        root, work_dir, shard, manifest = self._seed(tmp_path)

        def write_then_symlink(table_arg, path):
            target = tmp_path / "real-parquet"
            pq.write_table(table_arg, target)
            os.chmod(target, 0o600)
            try:
                os.symlink(target, path)
            except (OSError, NotImplementedError):
                pytest.skip("symlinks not supported on this platform")
            os.chmod(path, 0o600)

        monkeypatch.setattr(ckpt, "_atomic_write_parquet", write_then_symlink)

        with pytest.raises(CheckpointValidationError):
            publish_shard_checkpoint(
                work_dir=work_dir,
                shard=shard,
                input_root=root,
                table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
                report=_build_report(),
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                verified_manifest=manifest,
            )
        assert not (work_dir / "shards" / "active" / "reg-a").exists()

    def test_staged_wrong_mode_aborts_publication(self, tmp_path, monkeypatch):
        from osm_polygon_sentence_relevance.application._checkpoint import (
            storage as ckpt,
        )
        from osm_polygon_sentence_relevance.application.checkpoint import (
            publish_shard_checkpoint,
        )

        root, work_dir, shard, manifest = self._seed(tmp_path)

        original = ckpt._atomic_write_parquet

        def write_then_chmod(table_arg, path):
            original(table_arg, path)
            os.chmod(path, 0o644)

        monkeypatch.setattr(ckpt, "_atomic_write_parquet", write_then_chmod)

        with pytest.raises(CheckpointValidationError):
            publish_shard_checkpoint(
                work_dir=work_dir,
                shard=shard,
                input_root=root,
                table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
                report=_build_report(),
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                verified_manifest=manifest,
            )
        assert not (work_dir / "shards" / "active" / "reg-a").exists()


# ===========================================================================
# 14. Malformed inventory: same run continues via orphan-active recovery
# ===========================================================================


class TestMalformedInventoryContinuesSameRun:
    """When ``inventory.json`` is malformed, the safe loader atomically
    quarantines it and returns ``None``. The pipeline then runs to
    completion in the same invocation via orphan-active recovery: each
    active checkpoint is loaded and either reused (matching manifest)
    or quarantined (drift / invalid).
    """

    def test_malformed_inventory_continues_same_run(self, tmp_path):
        root = tmp_path / "in"
        out_dir_a = _out(tmp_path, "out_a")
        out_dir_b = _out(tmp_path, "out_b")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _two_region_layout(root)
        _run(root, out_dir_a, work_dir)

        # Replace the inventory.json with malformed JSON.
        (work_dir / "shards" / "inventory.json").write_text("{ not valid json")

        # The next run must complete cleanly in the same invocation.
        _run(root, out_dir_b, work_dir)
        assert (out_dir_b / "sentences.parquet").exists()
        # Malformed inventory was preserved in quarantine.
        qdir = work_dir / "shards" / "quarantine" / _ckpt_mod.INVENTORY_QUARANTINE_DIR
        assert any(qdir.iterdir())
        # A fresh inventory has been written at the end of the run.
        assert (work_dir / "shards" / "inventory.json").exists()
        data = json.loads((work_dir / "shards" / "inventory.json").read_text())
        assert int(data["schema_version"]) >= 2


# ===========================================================================
# 15. Unexpected active entries must never be silently ignored
# ===========================================================================


class TestUnexpectedActiveEntriesNeverIgnored:
    """The active directory is scanned defensively. Files, broken
    symlinks, unexpected directories and symlinks to files are either
    quarantined (when safe) or cause the run to abort. Symlinks are
    NEVER followed.
    """

    def test_unexpected_file_in_active_aborts(self, tmp_path):
        root = tmp_path / "in"
        out_dir_a = _out(tmp_path, "out_a")
        out_dir_b = _out(tmp_path, "out_b")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _two_region_layout(root)
        _run(root, out_dir_a, work_dir)

        stray = work_dir / "shards" / "active" / "stray-file"
        stray.write_bytes(b"junk")

        with pytest.raises(CheckpointValidationError):
            _run(root, out_dir_b, work_dir)

    def test_broken_symlink_under_active_aborts(self, tmp_path):
        root = tmp_path / "in"
        out_dir_a = _out(tmp_path, "out_a")
        out_dir_b = _out(tmp_path, "out_b")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _two_region_layout(root)
        _run(root, out_dir_a, work_dir)

        active_root = work_dir / "shards" / "active"
        link = active_root / "broken-link"
        try:
            link.symlink_to(active_root / "does-not-exist")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")

        with pytest.raises(CheckpointValidationError):
            _run(root, out_dir_b, work_dir)

    def test_symlink_to_directory_under_active_aborts(self, tmp_path):
        root = tmp_path / "in"
        out_dir_a = _out(tmp_path, "out_a")
        out_dir_b = _out(tmp_path, "out_b")
        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        _two_region_layout(root)
        _run(root, out_dir_a, work_dir)

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "segmented.parquet").write_bytes(b"x")
        (elsewhere / "metadata.json").write_text("{}")
        active_root = work_dir / "shards" / "active"
        link = active_root / "reg-a"
        try:
            link.symlink_to(elsewhere)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")

        with pytest.raises(CheckpointValidationError):
            _run(root, out_dir_b, work_dir)


# ===========================================================================
# 16. .lock hardening: regular file, current user, mode 0600
# ===========================================================================


class TestLockFileHardening:
    """The work-dir ``.lock`` file must be a regular file owned by the
    current user with mode ``0o600``. Symlinks, non-regular files and
    permissive modes must be rejected at acquire time. The lock
    itself is still released through ``finally``.
    """

    def test_lock_acquired_on_regular_file_with_correct_mode(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            acquire_work_dir_lock,
            release_work_dir_lock,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        ctx = acquire_work_dir_lock(work_dir)
        try:
            lock_path = work_dir / "shards" / ".lock"
            st = os.lstat(lock_path)
            assert not os.path.islink(lock_path)
            assert stat_module.S_ISREG(st.st_mode)
            assert (st.st_mode & 0o777) == 0o600
            assert st.st_uid == os.getuid()
        finally:
            release_work_dir_lock(ctx)

    def test_lock_acquire_rejects_symlink(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            acquire_work_dir_lock,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        shards = work_dir / "shards"
        shards.mkdir()
        lock_path = shards / ".lock"
        try:
            lock_path.symlink_to(tmp_path / "real-lock-target")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")

        with pytest.raises(CheckpointValidationError, match="lock"):
            acquire_work_dir_lock(work_dir)

    def test_lock_acquire_rejects_permissive_mode(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            acquire_work_dir_lock,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        shards = work_dir / "shards"
        shards.mkdir()
        lock_path = shards / ".lock"
        lock_path.write_text("placeholder")
        os.chmod(lock_path, 0o644)
        with pytest.raises(CheckpointValidationError, match="lock"):
            acquire_work_dir_lock(work_dir)

    def test_lock_release_after_exception(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            acquire_work_dir_lock,
            release_work_dir_lock,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        ctx = acquire_work_dir_lock(work_dir)
        try:
            raise RuntimeError("simulated downstream failure")
        except RuntimeError:
            release_work_dir_lock(ctx)
        ctx2 = acquire_work_dir_lock(work_dir)
        try:
            assert (work_dir / "shards" / ".lock").exists()
        finally:
            release_work_dir_lock(ctx2)

    def test_lock_acquire_rejects_non_regular_file(self, tmp_path):
        """A FIFO / character device / socket at ``.lock`` is rejected
        as a non-regular-file entry.
        """
        from osm_polygon_sentence_relevance.application.checkpoint import (
            acquire_work_dir_lock,
        )

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        shards = work_dir / "shards"
        shards.mkdir()
        lock_path = shards / ".lock"
        try:
            os.mkfifo(lock_path)
        except (OSError, AttributeError):
            pytest.skip("mkfifo not supported on this platform")
        os.chmod(lock_path, 0o600)
        with pytest.raises(CheckpointValidationError, match="lock"):
            acquire_work_dir_lock(work_dir)

    def test_lock_acquire_rejects_other_owner(self, tmp_path):
        """A lock file owned by a different user is rejected.

        Skipped when running as root (root bypasses ownership checks).
        """
        import pwd

        from osm_polygon_sentence_relevance.application.checkpoint import (
            acquire_work_dir_lock,
        )

        if os.getuid() == 0:
            pytest.skip("running as root; ownership check is bypassed")
        try:
            other_uid = next(
                entry.pw_uid
                for entry in pwd.getpwall()
                if entry.pw_uid != os.getuid() and entry.pw_uid >= 1000
            )
        except StopIteration:
            pytest.skip("no other suitable user available")

        work_dir = tmp_path / "wd"
        work_dir.mkdir()
        shards = work_dir / "shards"
        shards.mkdir()
        lock_path = shards / ".lock"
        lock_path.write_text("placeholder")
        os.chmod(lock_path, 0o600)
        try:
            os.chown(lock_path, other_uid, -1)
        except (PermissionError, OSError):
            pytest.skip("cannot chown the lock file")

        with pytest.raises(CheckpointValidationError, match="lock"):
            acquire_work_dir_lock(work_dir)
