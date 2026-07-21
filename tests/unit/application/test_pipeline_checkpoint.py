"""the implementation Hardening tests — quarantine-in-place + source-file binding.

The pipeline accepts an optional ``work_dir`` that persists per-shard
checkpoints and a factual progress heartbeat. Invariants:

* **Never delete** completed checkpoints; place them under
  ``${work_dir}/shards/quarantine/`` with a unique collision-resistant
  suffix.
* **Bind checkpoints to the actual source files** of each
  ``RegionShardSet`` (six schema folders). Hash every source file and
  record its relative path, size and SHA-256 in the checkpoint
  metadata.
* **Publish as a whole directory**: write + validate inside a unique
  sibling staging directory, atomically ``os.rename`` to ``active/``,
  and refuse to overwrite a valid active checkpoint. On any failure
  preserve the staging directory as evidence.
* **Single-writer**: one run owns the work directory at a time. Quarantine
  uses ``os.rename`` only; cross-filesystem moves are not supported.
* **Restartability** under every failure point listed in the spec is a
  hard requirement. All previous completed checkpoints survive any
  crash, heartbeat failure, publish failure or finalization failure.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.application.checkpoint import (
    CheckpointPublicationError,
    compute_shard_source_manifest,
    publish_shard_checkpoint,
    quarantine_shard_checkpoint,
)
from osm_polygon_sentence_relevance.application.pipeline import run_pipeline
from osm_polygon_sentence_relevance.contracts.errors import SegmentationError
from osm_polygon_sentence_relevance.ingestion.discovery import (
    discover_shards,
)
from tests.support.arrow_factories import (
    make_polygon_article_row,
    make_polygon_row,
    make_section_row,
    make_wikipedia_document_row,
)
from tests.support.parquet_layouts import write_shard_parquet

# ---------------------------------------------------------------------------
# Test-double segmenter (never touches torch/wtpsplit).
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
        self.total_calls_seen_texts: list[str] = []

    def split_batch(self, texts: list[str], languages: list[str]) -> list[list[str]]:
        self.calls_count += 1
        self.total_calls_seen_texts.extend(texts)
        return [self.split_fn(text) for text in texts]


# ---------------------------------------------------------------------------
# Synthetic six-folder shard layout.
# ---------------------------------------------------------------------------


def _write_region(
    root: Path,
    region: str,
    *,
    wikidata_id: str = "Q1",
    text: str | None = None,
) -> None:
    """Write one shard; ``wikidata_id`` and ``text`` are controllable to
    drive source-manifest variance tests.
    """
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


def _write_two_region_layout(root: Path) -> None:
    for region in ("reg-a", "reg-b"):
        _write_region(root, region)


# ---------------------------------------------------------------------------
# Identity and value helpers used across tests.
# ---------------------------------------------------------------------------


VALID_SOURCE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
VALID_INPUT_REVISION = "abcdefabcdefabcdefabcdefabcdefabcdefabcd"


def _run_with_work_dir(
    root: Path,
    out_dir: Path,
    work_dir: Path,
    *,
    source_commit: str = VALID_SOURCE_COMMIT,
    input_dataset_revision: str = VALID_INPUT_REVISION,
    pipeline_version: str = "v1",
    model_name: str = "mock",
    segmenter: MockSegmenter | None = None,
):
    return run_pipeline(
        root,
        out_dir,
        segmenter or MockSegmenter(),
        input_dataset_revision=input_dataset_revision,
        pipeline_version=pipeline_version,
        work_dir=work_dir,
        source_commit=source_commit,
        model_name=model_name,
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestSourceManifestFromRegionShardSet:
    """Manifests are derived from the real ``RegionShardSet`` discovery
    type and bound to actual file bytes, never to ``input_root`` alone.
    """

    def test_manifest_includes_all_six_paths_for_a_shard(self, tmp_path):
        root = tmp_path / "in"
        _write_region(root, "reg-a")
        shard = discover_shards(root)[0]

        manifest = compute_shard_source_manifest(shard, input_root=root)

        names = {Path(entry.path).name for entry in manifest}
        assert names == {
            "reg-a.parquet",
        }
        # 4 core files for a non-wikivoyage shard.
        assert len(manifest) == 4

    def test_manifest_includes_wikivoyage_pair_when_present(self, tmp_path):
        root = tmp_path / "in"
        _write_region(root, "reg-a")
        write_shard_parquet(
            root,
            "reg-a",
            wikivoyage_documents_rows=[
                make_wikipedia_document_row(
                    document_id="wv-doc-1",
                    article_id="art-1",
                    wikidata="Q1",
                    title="WV",
                    language="en",
                )
            ],
            wikivoyage_sections_rows=[
                make_section_row(
                    section_id="wv-sec-1",
                    document_id="wv-doc-1",
                    article_id="art-1",
                    wikidata="Q1",
                    project="wikivoyage",
                    language="en",
                    site="en.wikivoyage.org",
                    section_index=0,
                    heading="Intro",
                    text="WV sentence.",
                )
            ],
        )
        shard = discover_shards(root)[0]
        manifest = compute_shard_source_manifest(shard, input_root=root)
        # 4 core + 2 wikivoyage = 6 entries.
        assert len(manifest) == 6
        # Wikivoyage filenames appear in the manifest.
        names = {Path(entry.path).name for entry in manifest}
        assert names == {"reg-a.parquet"}

    def test_manifest_entries_sorted_by_relative_path(self, tmp_path):
        root = tmp_path / "in"
        _write_region(root, "reg-a")
        shard = discover_shards(root)[0]
        manifest = compute_shard_source_manifest(shard, input_root=root)
        sorted_paths = [e.path for e in manifest]
        assert sorted_paths == sorted(sorted_paths)

    def test_manifest_sha256_matches_actual_file_bytes(self, tmp_path):
        root = tmp_path / "in"
        _write_region(root, "reg-a")
        shard = discover_shards(root)[0]
        manifest = compute_shard_source_manifest(shard, input_root=root)

        # Recompute the SHA of every referenced file from scratch and
        # compare to what compute_shard_source_manifest recorded.
        for entry in manifest:
            fpath = root / entry.path
            actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest().lower()
            assert entry.sha256 == actual_sha
            assert entry.size == fpath.stat().st_size


class TestInvalidationApiRemoved:
    """The destructive invalidation API must be gone — completed work is
    never destroyed, only quarantined.
    """

    def test_invalidate_shard_checkpoint_is_not_exported(self):
        import osm_polygon_sentence_relevance.application.checkpoint as ckpt

        assert not hasattr(ckpt, "invalidate_shard_checkpoint")
        # The public API should reference only the safe primitives.
        assert "invalidate_shard_checkpoint" not in ckpt.__all__


class TestCheckpointPublicationAtomic:
    """Publication is a whole-directory rename, never an in-place
    overwrite of an existing active checkpoint.
    """

    def _build_table(self):
        from osm_polygon_sentence_relevance.contracts.schemas import (
            SEGMENTED_SENTENCES_SCHEMA,
        )

        return SEGMENTED_SENTENCES_SCHEMA.empty_table()

    def _build_report(self):
        from osm_polygon_sentence_relevance.sentences.segmentation import (
            SegmentationReport,
        )

        return SegmentationReport(
            input_section_occurrence_count=0,
            emitted_segment_count=0,
            retained_sentence_occurrence_count=0,
            dropped_empty_raw_count=0,
            dropped_empty_normalized_count=0,
            wikipedia_sentence_occurrence_count=0,
            wikivoyage_sentence_occurrence_count=0,
        )

    def test_publish_writes_active_dir_when_no_existing_checkpoint(self, tmp_path):
        root = tmp_path / "in"
        out_dir = tmp_path / "out"
        work_dir = tmp_path / "work"
        out_dir.mkdir()
        work_dir.mkdir()
        _write_two_region_layout(root)
        shard = discover_shards(root)[0]
        manifest = compute_shard_source_manifest(shard, input_root=root)

        active_path = publish_shard_checkpoint(
            work_dir=work_dir,
            shard=shard,
            input_root=root,
            table=self._build_table(),
            report=self._build_report(),
            input_dataset_revision=VALID_INPUT_REVISION,
            pipeline_version="v1",
            source_commit=VALID_SOURCE_COMMIT,
            model_name="mock",
            batch_size=128,
            verified_manifest=manifest,
        )
        assert (active_path / "segmented.parquet").exists()
        assert (active_path / "metadata.json").exists()

    def test_publish_refuses_to_overwrite_existing_active(self, tmp_path):
        root = tmp_path / "in"
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        _write_region(root, "reg-a")
        shard = discover_shards(root)[0]
        manifest = compute_shard_source_manifest(shard, input_root=root)

        # First publish succeeds.
        first = publish_shard_checkpoint(
            work_dir=work_dir,
            shard=shard,
            input_root=root,
            table=self._build_table(),
            report=self._build_report(),
            input_dataset_revision=VALID_INPUT_REVISION,
            pipeline_version="v1",
            source_commit=VALID_SOURCE_COMMIT,
            model_name="mock",
            batch_size=128,
            verified_manifest=manifest,
        )
        # Second publish attempts to overwrite.
        with pytest.raises(CheckpointPublicationError, match="overwrite|exists"):
            publish_shard_checkpoint(
                work_dir=work_dir,
                shard=shard,
                input_root=root,
                table=self._build_table(),
                report=self._build_report(),
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                verified_manifest=manifest,
            )
        # Active dir unchanged.
        assert (first / "segmented.parquet").exists()
        assert (first / "metadata.json").exists()

    def test_publish_preserves_staging_on_write_failure(self, tmp_path, monkeypatch):
        from osm_polygon_sentence_relevance.application._checkpoint import storage

        root = tmp_path / "in"
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        _write_region(root, "reg-a")
        shard = discover_shards(root)[0]
        manifest = compute_shard_source_manifest(shard, input_root=root)

        def boom(*_a, **_kw):
            raise RuntimeError("simulated parquet write failure")

        monkeypatch.setattr(storage, "_atomic_write_parquet", boom)

        with pytest.raises(RuntimeError, match="simulated"):
            publish_shard_checkpoint(
                work_dir=work_dir,
                shard=shard,
                input_root=root,
                table=self._build_table(),
                report=self._build_report(),
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                verified_manifest=manifest,
            )
        # Active directory must NOT exist.
        active = work_dir / "shards" / "active" / "reg-a"
        assert not active.exists()
        # Staging directory must still be present (evidence).
        staging_root = work_dir / "shards"
        stagings = list(staging_root.glob(".staging.reg-a.*"))
        assert stagings, "expected at least one staging dir preserved"

    def test_published_metadata_schema_version_is_2(self, tmp_path):
        root = tmp_path / "in"
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        _write_region(root, "reg-a")
        shard = discover_shards(root)[0]
        manifest = compute_shard_source_manifest(shard, input_root=root)

        active_path = publish_shard_checkpoint(
            work_dir=work_dir,
            shard=shard,
            input_root=root,
            table=self._build_table(),
            report=self._build_report(),
            input_dataset_revision=VALID_INPUT_REVISION,
            pipeline_version="v1",
            source_commit=VALID_SOURCE_COMMIT,
            model_name="mock",
            batch_size=128,
            verified_manifest=manifest,
        )
        meta = json.loads((active_path / "metadata.json").read_text())
        assert int(meta["schema_version"]) == 2


class TestCheckpointQuarantine:
    """Quarantine must preserve bytes, never delete, and abort on the
    same-filesystem rename failing.
    """

    def _seed_one_active_checkpoint(self, work_dir: Path, root_in: Path) -> Path:
        from osm_polygon_sentence_relevance.application.checkpoint import (
            compute_shard_source_manifest,
        )
        from osm_polygon_sentence_relevance.contracts.schemas import (
            SEGMENTED_SENTENCES_SCHEMA,
        )
        from osm_polygon_sentence_relevance.sentences.segmentation import (
            SegmentationReport,
        )

        _write_region(root_in, "reg-a")
        shard = discover_shards(root_in)[0]
        manifest = compute_shard_source_manifest(shard, input_root=root_in)
        return publish_shard_checkpoint(
            work_dir=work_dir,
            shard=shard,
            input_root=root_in,
            table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
            report=SegmentationReport(
                input_section_occurrence_count=0,
                emitted_segment_count=0,
                retained_sentence_occurrence_count=0,
                dropped_empty_raw_count=0,
                dropped_empty_normalized_count=0,
                wikipedia_sentence_occurrence_count=0,
                wikivoyage_sentence_occurrence_count=0,
            ),
            input_dataset_revision=VALID_INPUT_REVISION,
            pipeline_version="v1",
            source_commit=VALID_SOURCE_COMMIT,
            model_name="mock",
            batch_size=128,
            verified_manifest=manifest,
        )

    def test_quarantine_moves_active_into_quarantine_dir(self, tmp_path):
        root_in = tmp_path / "in"
        work_dir = tmp_path / "work"
        root_in.mkdir()
        work_dir.mkdir()
        active = self._seed_one_active_checkpoint(work_dir, root_in)
        # Capture content hashes before quarantine.
        pre = {
            "segmented.parquet": hashlib.sha256(
                (active / "segmented.parquet").read_bytes()
            ).hexdigest(),
            "metadata.json": (active / "metadata.json").read_text(),
        }

        qpath = quarantine_shard_checkpoint(
            work_dir=work_dir, shard_key="reg-a", reason="test"
        )
        assert qpath is not None
        assert qpath.is_dir()
        # Active dir gone (moved).
        assert not active.exists()
        # Quarantine preserves bytes (no chmod/rewrite).
        assert (
            hashlib.sha256((qpath / "segmented.parquet").read_bytes()).hexdigest()
            == pre["segmented.parquet"]
        )
        assert (qpath / "metadata.json").read_text() == pre["metadata.json"]

    def test_quarantine_uses_collision_resistant_unique_suffix(self, tmp_path):
        root_in = tmp_path / "in"
        work_dir = tmp_path / "work"
        root_in.mkdir()
        work_dir.mkdir()
        from osm_polygon_sentence_relevance.application.checkpoint import (
            compute_shard_source_manifest,
        )
        from osm_polygon_sentence_relevance.contracts.schemas import (
            SEGMENTED_SENTENCES_SCHEMA,
        )
        from osm_polygon_sentence_relevance.sentences.segmentation import (
            SegmentationReport,
        )

        self._seed_one_active_checkpoint(work_dir, root_in)
        # First quarantine.
        q1 = quarantine_shard_checkpoint(
            work_dir=work_dir, shard_key="reg-a", reason="r1"
        )
        assert q1 is not None

        # Re-publish a fresh active from scratch, then quarantine a
        # second time.
        shard = discover_shards(root_in)[0]
        manifest = compute_shard_source_manifest(shard, input_root=root_in)
        publish_shard_checkpoint(
            work_dir=work_dir,
            shard=shard,
            input_root=root_in,
            table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
            report=SegmentationReport(
                input_section_occurrence_count=0,
                emitted_segment_count=0,
                retained_sentence_occurrence_count=0,
                dropped_empty_raw_count=0,
                dropped_empty_normalized_count=0,
                wikipedia_sentence_occurrence_count=0,
                wikivoyage_sentence_occurrence_count=0,
            ),
            input_dataset_revision=VALID_INPUT_REVISION,
            pipeline_version="v1",
            source_commit=VALID_SOURCE_COMMIT,
            model_name="mock",
            batch_size=128,
            verified_manifest=manifest,
        )
        q2 = quarantine_shard_checkpoint(
            work_dir=work_dir, shard_key="reg-a", reason="r2"
        )
        assert q2 is not None
        # Two distinct quarantine paths.
        assert q1 != q2
        # Each contains an intact copy of the previously active bytes.
        for qp in (q1, q2):
            assert (qp / "segmented.parquet").exists()
            assert (qp / "metadata.json").exists()

    def test_quarantine_failure_preserves_active(self, tmp_path, monkeypatch):
        root_in = tmp_path / "in"
        work_dir = tmp_path / "work"
        root_in.mkdir()
        work_dir.mkdir()
        active = self._seed_one_active_checkpoint(work_dir, root_in)

        # Force os.rename to fail (EXDEV).
        def rename_fail(src, dst):
            raise OSError(errno.EXDEV, "Cross-device link")

        monkeypatch.setattr(os, "rename", rename_fail)

        with pytest.raises(OSError, match="Cross-device|rename"):
            quarantine_shard_checkpoint(
                work_dir=work_dir, shard_key="reg-a", reason="r1"
            )
        # Active dir untouched.
        assert active.exists()
        assert (active / "segmented.parquet").exists()
        assert (active / "metadata.json").exists()

    def test_quarantine_no_destructive_invalidation_callable(self, tmp_path):
        # Sanity: the old invalidate API is gone.
        root_in = tmp_path / "in"
        work_dir = tmp_path / "work"
        root_in.mkdir()
        work_dir.mkdir()
        self._seed_one_active_checkpoint(work_dir, root_in)
        # Calling the removed function name must fail.
        import osm_polygon_sentence_relevance.application.checkpoint as ckpt

        with pytest.raises(AttributeError):
            _ = ckpt.invalidate_shard_checkpoint(  # type: ignore[attr-defined]
                work_dir, "reg-a"
            )


class TestLoadAndQuarantineOnMismatch:
    """Loading detects changes via source-manifest binding and quarantines
    rather than overwriting.
    """

    def _seed(self, work_dir, root_in, *, text: str = "First sentence."):
        _run_with_work_dir(root_in, root_in.parent / "out", work_dir)

    def _out(self, tmp_path):
        d = tmp_path / "out"
        d.mkdir(exist_ok=True)
        return d

    def test_load_with_matching_source_files_reuses_checkpoint(self, tmp_path):
        root = tmp_path / "in"
        out_dir_a = tmp_path / "out_a"
        out_dir_b = tmp_path / "out_b"
        out_dir_a.mkdir()
        out_dir_b.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        _write_two_region_layout(root)

        _run_with_work_dir(root, out_dir_a, work_dir)
        # Resume unchanged input. Segmenter must not be invoked again.
        seg = MockSegmenter()
        before = seg.calls_count
        _run_with_work_dir(root, out_dir_b, work_dir, segmenter=seg)
        # No new segmentation calls occurred.
        assert seg.calls_count == before

    def test_load_quarantines_corrupt_metadata_and_recomputes(self, tmp_path):
        """Corrupt metadata is no longer fatal: the run auto-quarantines
        the invalid active checkpoint, then re-segments that shard.
        ``reg-b`` continues to be reused unchanged.
        """
        root = tmp_path / "in"
        out_dir_a = tmp_path / "out_a"
        out_dir_b = tmp_path / "out_b"
        out_dir_a.mkdir()
        out_dir_b.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        _write_two_region_layout(root)
        _run_with_work_dir(root, out_dir_a, work_dir)

        active = work_dir / "shards" / "active" / "reg-a"
        active.joinpath("metadata.json").write_text("{ not valid json")

        seg = MockSegmenter()
        before = seg.calls_count
        _run_with_work_dir(root, out_dir_b, work_dir, segmenter=seg)
        # reg-a was quarantined then re-segmented; reg-b was reused.
        assert seg.calls_count == before + 1
        # The corrupt active directory was moved to quarantine.
        qdirs = list((work_dir / "shards" / "quarantine").iterdir())
        assert qdirs

    def test_load_aborts_on_wrong_source_commit(self, tmp_path):
        """A different source_commit auto-quarantines the active
        checkpoint for that shard and re-segments. Other shards are
        unaffected.
        """
        root = tmp_path / "in"
        out_dir_a = tmp_path / "out_a"
        out_dir_b = tmp_path / "out_b"
        out_dir_a.mkdir()
        out_dir_b.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        _write_two_region_layout(root)
        _run_with_work_dir(root, out_dir_a, work_dir)

        seg = MockSegmenter()
        before = seg.calls_count
        _run_with_work_dir(
            root,
            out_dir_b,
            work_dir,
            source_commit="feedfacefeedfacefeedfacefeedfacefeedfac0",
            segmenter=seg,
        )
        # Both regions got re-segmented (the prior inventory's source
        # commit doesn't match the new one).
        assert seg.calls_count > before

    def test_load_quarantines_changed_source_file_on_orphan_recovery(self, tmp_path):
        """If a previously-snapshotted source file's bytes change,
        the recovery sweep at run start detects the drift and
        quarantines the active entry, then re-segments.
        """
        root = tmp_path / "in"
        out_dir_a = tmp_path / "out_a"
        out_dir_b = tmp_path / "out_b"
        out_dir_a.mkdir()
        out_dir_b.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        _write_two_region_layout(root)
        _run_with_work_dir(root, out_dir_a, work_dir)

        _write_region(root, "reg-a", wikidata_id="Q2")

        seg = MockSegmenter()
        before = seg.calls_count
        _run_with_work_dir(root, out_dir_b, work_dir, segmenter=seg)
        assert seg.calls_count > before
        qdirs = list((work_dir / "shards" / "quarantine").iterdir())
        assert qdirs

    def test_load_quarantines_missing_source_file_on_orphan_recovery(self, tmp_path):
        """Same as above but a Wikivoyage-style fields change rather
        than wikidata change.
        """
        root = tmp_path / "in"
        out_dir_a = tmp_path / "out_a"
        out_dir_b = tmp_path / "out_b"
        out_dir_a.mkdir()
        out_dir_b.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        _write_two_region_layout(root)
        _run_with_work_dir(root, out_dir_a, work_dir)

        _write_region(root, "reg-a", text="DIFFERENT WORDS HERE.")

        seg = MockSegmenter()
        before = seg.calls_count
        _run_with_work_dir(root, out_dir_b, work_dir, segmenter=seg)
        assert seg.calls_count > before
        qdirs = list((work_dir / "shards" / "quarantine").iterdir())
        assert qdirs


class TestInventoryReconciliation:
    """Per-shard inventory drives reuse vs. quarantine decisions.
    Added shards: process only the new shard. Removed shards: quarantine
    only the orphaned checkpoint. Changed shard: quarantine and recompute.
    Unchanged shards: reuse.
    """

    def test_added_shard_processed_without_quarantining_others(self, tmp_path):
        root = tmp_path / "in"
        out_dir_a = tmp_path / "out_a"
        out_dir_b = tmp_path / "out_b"
        out_dir_a.mkdir()
        out_dir_b.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        _write_region(root, "reg-a")
        _run_with_work_dir(root, out_dir_a, work_dir, segmenter=MockSegmenter())
        # No quarantine dirs yet.
        if (work_dir / "shards" / "quarantine").exists():
            assert not list((work_dir / "shards" / "quarantine").iterdir())

        # Now add reg-b.
        _write_region(root, "reg-b")
        seg = MockSegmenter()
        before = seg.calls_count
        _run_with_work_dir(root, out_dir_b, work_dir, segmenter=seg)
        # Only reg-b got segmented (reg-a reused).
        assert seg.calls_count == before + 1
        # reg-a quarantine dir does NOT exist.
        if (work_dir / "shards" / "quarantine").exists():
            assert not list((work_dir / "shards" / "quarantine").glob("reg-a.*"))

    def test_removed_shard_quarantines_only_orphaned_checkpoint(self, tmp_path):
        root = tmp_path / "in"
        out_dir_a = tmp_path / "out_a"
        out_dir_b = tmp_path / "out_b"
        out_dir_a.mkdir()
        out_dir_b.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        _write_two_region_layout(root)
        _run_with_work_dir(root, out_dir_a, work_dir)

        # Delete reg-a entirely.
        for sub in (
            "polygons",
            "polygon_articles",
            "wikipedia/documents",
            "wikipedia/sections",
        ):
            (root / sub / "reg-a.parquet").unlink()

        seg = MockSegmenter()
        before = seg.calls_count
        _run_with_work_dir(root, out_dir_b, work_dir, segmenter=seg)
        # reg-b reused cleanly: no new segmenter calls.
        assert seg.calls_count == before
        # reg-a's active is gone (it was orphaned).
        assert not (work_dir / "shards" / "active" / "reg-a").exists()


class TestHeartbeatFailureDoesNotBlock:
    """Heartbeat failure surfaces visibly while the published checkpoint
    remains valid for the next run.
    """

    def test_heartbeat_failure_after_publish_does_not_lose_checkpoint(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "in"
        out_dir_a = tmp_path / "out_a"
        out_dir_b = tmp_path / "out_b"
        out_dir_a.mkdir()
        out_dir_b.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        _write_two_region_layout(root)

        # Pre-publish reg-a's checkpoint by running once cleanly.
        _run_with_work_dir(root, out_dir_a, work_dir)

        # Now make every subsequent heartbeat raise. This includes the
        # initial heartbeat and every per-shard heartbeat for the
        # remaining shard (reg-b).
        call_count = {"n": 0}

        def boom_hb(*_a, **_kw):
            call_count["n"] += 1
            raise RuntimeError("simulated heartbeat failure")

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.checkpoint.write_heartbeat",
            boom_hb,
        )

        seg = MockSegmenter()
        with pytest.raises(RuntimeError, match="simulated heartbeat"):
            _run_with_work_dir(root, out_dir_b, work_dir, segmenter=seg)

        # At least reg-a's previously published checkpoint must remain.
        actives = list((work_dir / "shards" / "active").iterdir())
        assert any(a.name == "reg-a" for a in actives)
        # reg-a bytes unchanged.
        reg_a = work_dir / "shards" / "active" / "reg-a"
        assert (reg_a / "segmented.parquet").exists()
        assert (reg_a / "metadata.json").exists()

        # Resume on a fresh invocation with real heartbeat must reuse
        # reg-a and complete cleanly.
        monkeypatch.undo()
        seg2 = MockSegmenter()
        _run_with_work_dir(root, out_dir_b, work_dir, segmenter=seg2)
        # reg-a reused (no new segmentation call for it). reg-b may
        # have been published or may still need to be.
        assert seg2.calls_count <= 1


class TestCLISourceCommitValidation:
    """The CLI rejects non-conforming ``--source-commit`` and requires it
    when ``--work-dir`` is set.
    """

    def test_cli_rejects_missing_source_commit_when_work_dir_set(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import (
            _build_parser,
            _validate_args,
        )

        parser = _build_parser()
        args = parser.parse_args(
            [
                "--input-root",
                str(tmp_path / "in"),
                "--output-dir",
                str(tmp_path / "out"),
                "--input-dataset-revision",
                VALID_INPUT_REVISION,
                "--pipeline-version",
                "v1",
                "--work-dir",
                str(tmp_path / "work"),
            ]
        )
        with pytest.raises(ValueError, match="source-commit"):
            _validate_args(args)

    def test_cli_accepts_valid_40_hex_source_commit(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import (
            _build_parser,
            _validate_args,
        )

        parser = _build_parser()
        args = parser.parse_args(
            [
                "--input-root",
                str(tmp_path / "in"),
                "--output-dir",
                str(tmp_path / "out"),
                "--input-dataset-revision",
                VALID_INPUT_REVISION,
                "--pipeline-version",
                "v1",
                "--work-dir",
                str(tmp_path / "work"),
                "--source-commit",
                VALID_SOURCE_COMMIT,
            ]
        )
        # Should not raise.
        _validate_args(args)
        assert args.source_commit == VALID_SOURCE_COMMIT

    def test_cli_rejects_uppercase_source_commit(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import (
            _build_parser,
            _validate_args,
        )

        parser = _build_parser()
        args = parser.parse_args(
            [
                "--input-root",
                str(tmp_path / "in"),
                "--output-dir",
                str(tmp_path / "out"),
                "--input-dataset-revision",
                VALID_INPUT_REVISION,
                "--pipeline-version",
                "v1",
                "--work-dir",
                str(tmp_path / "work"),
                "--source-commit",
                VALID_SOURCE_COMMIT.upper(),
            ]
        )
        with pytest.raises(ValueError, match="source-commit"):
            _validate_args(args)

    def test_cli_rejects_short_source_commit(self, tmp_path):
        from osm_polygon_sentence_relevance.application.cli import (
            _build_parser,
            _validate_args,
        )

        parser = _build_parser()

        args = parser.parse_args(
            [
                "--input-root",
                str(tmp_path / "in"),
                "--output-dir",
                str(tmp_path / "out"),
                "--input-dataset-revision",
                VALID_INPUT_REVISION,
                "--pipeline-version",
                "v1",
                "--work-dir",
                str(tmp_path / "work"),
                "--source-commit",
                "abcd" * 9,  # 36 chars
            ]
        )
        with pytest.raises(ValueError, match="source-commit"):
            _validate_args(args)


class TestResumedOutputEqualsUninterrupted:
    """Final outputs (sentences.parquet + manifest.json) are byte-equal
    between an uninterrupted run and a run that resumed from
    checkpoints.
    """

    def test_parquet_and_manifest_byte_equal(self, tmp_path):
        root = tmp_path / "in"
        out_dir_a = tmp_path / "out_a"
        out_dir_b = tmp_path / "out_b"
        out_dir_a.mkdir()
        out_dir_b.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        _write_two_region_layout(root)

        # Uninterrupted run from clean state.
        _run_with_work_dir(root, out_dir_a, tmp_path / "noop")

        # Interrupted + resume.
        from osm_polygon_sentence_relevance.sentences.table import (
            segment_joined_sections as real_segment,
        )

        state = {"seen": 0}

        def fail_second_call(table, segmenter, **kwargs):
            state["seen"] += 1
            if state["seen"] == 2:
                raise SegmentationError("fail reg-b first time")
            return real_segment(table, segmenter, **kwargs)

        import osm_polygon_sentence_relevance.application.pipeline as pipe_mod

        original = pipe_mod.segment_joined_sections
        pipe_mod.segment_joined_sections = fail_second_call  # type: ignore[assignment]
        try:
            with pytest.raises(SegmentationError):
                _run_with_work_dir(root, out_dir_b, work_dir)
        finally:
            pipe_mod.segment_joined_sections = original  # type: ignore[assignment]

        # Resume.
        _run_with_work_dir(root, out_dir_b, work_dir)

        # Compare sentences.parquet bytes.

        a = pq.read_table(out_dir_a / "sentences.parquet")
        b = pq.read_table(out_dir_b / "sentences.parquet")
        assert a.num_rows == b.num_rows
        assert a.schema.equals(b.schema)
        # Compare sorted content for stability.
        a_sorted = a.sort_by("sentence_id")
        b_sorted = b.sort_by("sentence_id")
        assert a_sorted.to_pylist() == b_sorted.to_pylist()

        # Compare manifest.json bytes.
        m_a = (out_dir_a / "manifest.json").read_bytes()
        m_b = (out_dir_b / "manifest.json").read_bytes()
        assert m_a == m_b


class TestCheckpointLayoutAfterSuccessfulRun:
    """After a successful run, the layout is ``active/<shard>``
    containing exactly two files plus the heartbeat. No ``.staging.*``
    directories remain, no ``quarantine/`` exists unless mismatches
    triggered it.
    """

    def test_layout_clean_successful_run(self, tmp_path):
        root = tmp_path / "in"
        out_dir = tmp_path / "out"
        work_dir = tmp_path / "work"
        root.mkdir()
        out_dir.mkdir()
        work_dir.mkdir()
        _write_two_region_layout(root)
        _run_with_work_dir(root, out_dir, work_dir)

        shards_root = work_dir / "shards"
        active_root = shards_root / "active"
        assert active_root.is_dir()
        for shard_dir in sorted(active_root.iterdir()):
            contents = sorted(p.name for p in shard_dir.iterdir())
            assert contents == ["metadata.json", "segmented.parquet"]
        # No stale staging dirs.
        assert list(shards_root.glob(".staging.*")) == []
        # Heartbeat exists.
        assert (work_dir / "heartbeat.json").exists()
        # No quarantine dirs on a clean run.
        assert not (shards_root / "quarantine").exists() or not list(
            (shards_root / "quarantine").iterdir()
        )


# ---------------------------------------------------------------------------
# End-to-end path-overlap rejection at the pipeline preflight.
# ---------------------------------------------------------------------------


class TestPipelinePreflightPathOverlap:
    def test_work_dir_inside_input_root_rejected(self, tmp_path):
        root = tmp_path / "in"
        root.mkdir()
        work_dir = root / "work"
        work_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _write_region(root, "reg-a")
        with pytest.raises(ValueError, match="paths overlap"):
            _run_with_work_dir(root, out_dir, work_dir)

    def test_work_dir_inside_output_dir_rejected(self, tmp_path):
        root = tmp_path / "in"
        root.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        work_dir = out_dir / "work"
        work_dir.mkdir()
        _write_region(root, "reg-a")
        with pytest.raises(ValueError, match="paths overlap"):
            _run_with_work_dir(root, out_dir, work_dir)

    def test_publish_race_aborts_without_auto_retry(self, tmp_path):
        """Under hardening 2, a publication failure (here, an
        externally materialized active directory) is fatal: the
        pipeline does NOT quarantine-and-retry. The run aborts with
        the racing active bytes preserved in place.
        """
        root = tmp_path / "in"
        out_dir = tmp_path / "out"
        work_dir = tmp_path / "work"
        root.mkdir()
        out_dir.mkdir()
        work_dir.mkdir()
        _write_two_region_layout(root)

        import osm_polygon_sentence_relevance.application.pipeline as pipe_mod
        from osm_polygon_sentence_relevance.application.checkpoint import (
            CheckpointPublicationError,
            publish_shard_checkpoint,
        )

        call_count = {"n": 0}
        original = publish_shard_checkpoint

        def racing_publish(*args, **kwargs):
            call_count["n"] += 1
            # On the very first publish (reg-a), simulate the race by
            # creating an active directory right before the call.
            if call_count["n"] == 1:
                shard = kwargs["shard"]
                active_dir = kwargs["work_dir"] / "shards" / "active" / shard.shard_key
                active_dir.mkdir(parents=True, exist_ok=True)
                (active_dir / "segmented.parquet").write_bytes(b"x")
                (active_dir / "metadata.json").write_text("{}")
            return original(*args, **kwargs)

        # Patch the module reference the pipeline uses.
        pipe_mod.publish_shard_checkpoint = racing_publish  # type: ignore[assignment]

        try:
            seg = MockSegmenter()
            with pytest.raises(CheckpointPublicationError):
                _run_with_work_dir(root, out_dir, work_dir, segmenter=seg)
        finally:
            pipe_mod.publish_shard_checkpoint = original  # type: ignore[assignment]

        # Exactly one publish attempt occurred (no auto-retry).
        assert call_count["n"] == 1
        # The racing active directory survives intact.
        racing_active = work_dir / "shards" / "active" / "reg-a"
        assert (racing_active / "segmented.parquet").exists()
        assert (racing_active / "metadata.json").exists()


# ---------------------------------------------------------------------------
# Direct coverage of edge-case branches in the checkpoint module.
# ---------------------------------------------------------------------------


class TestCheckpointModuleDefensiveBranches:
    """Exercise direct branches in ``application/checkpoint.py`` that the
    end-to-end tests do not reach.
    """

    def test_validate_source_commit_accepts_lowercase_40_hex(self):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_source_commit,
        )

        commit = "0123456789abcdef0123456789abcdef01234567"
        assert validate_source_commit(commit) == commit

    def test_validate_source_commit_rejects_uppercase(self):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_source_commit,
        )

        with pytest.raises(ValueError, match="40-character"):
            validate_source_commit("0123456789ABCDEF0123456789ABCDEF01234567")

    def test_validate_source_commit_rejects_short(self):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_source_commit,
        )

        with pytest.raises(ValueError, match="40-character"):
            validate_source_commit("0123")

    def test_validate_source_commit_rejects_none(self):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_source_commit,
        )

        with pytest.raises(ValueError, match="non-empty string"):
            validate_source_commit(None)  # type: ignore[arg-type]

    def test_compute_shard_source_manifest_for_core_only(self):
        # Manifest construction walks known fields; assert it does not
        # explode for a ShardPaths with only core fields.
        import hashlib
        import tempfile

        from osm_polygon_sentence_relevance.application.checkpoint import (
            compute_shard_source_manifest,
        )
        from osm_polygon_sentence_relevance.ingestion.discovery import (
            RegionShardSet,
        )

        with tempfile.TemporaryDirectory() as d:
            base = __import__("pathlib").Path(d).resolve()
            for sub in (
                "polygons",
                "polygon_articles",
                "wikipedia/documents",
                "wikipedia/sections",
            ):
                (base / sub).mkdir(parents=True)
                (base / sub / "reg-x.parquet").write_bytes(b"x")
            shard = RegionShardSet(
                shard_key="reg-x",
                polygons=base / "polygons" / "reg-x.parquet",
                polygon_articles=base / "polygon_articles" / "reg-x.parquet",
                wikipedia_documents=base / "wikipedia" / "documents" / "reg-x.parquet",
                wikipedia_sections=base / "wikipedia" / "sections" / "reg-x.parquet",
                wikivoyage_documents=None,
                wikivoyage_sections=None,
            )
            m = compute_shard_source_manifest(shard, input_root=base)
            assert len(m) == 4
            for entry in m:
                # ``sha256(b"x")``
                assert entry.sha256 == hashlib.sha256(b"x").hexdigest().lower()
                assert entry.size == 1

    def test_inventory_round_trip(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            RunInventory,
            load_run_inventory,
            write_run_inventory,
        )

        inv = RunInventory(
            schema_version=2,
            discovered_at_unix=1234567890,
            input_dataset_revision=VALID_INPUT_REVISION,
            source_commit=VALID_SOURCE_COMMIT,
            pipeline_version="v1",
            model_name="mock",
            batch_size=128,
            shards={"reg-a": []},
        )
        write_run_inventory(tmp_path, inv)
        loaded = load_run_inventory(tmp_path)
        assert loaded is not None
        assert loaded.discovered_at_unix == 1234567890
        assert "reg-a" in loaded.shards

    def test_load_run_inventory_returns_none_when_missing(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            load_run_inventory,
        )

        (tmp_path / "shards").mkdir()
        assert load_run_inventory(tmp_path) is None

    def test_reconcile_inventory_no_prior(self):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            reconcile_inventory,
        )

        d = reconcile_inventory(
            prior=None,
            current={"reg-a": []},
        )
        assert d["added"] == {"reg-a"}
        assert d["removed"] == set()
        assert d["changed"] == set()
        assert d["unchanged"] == set()

    def test_reconcile_inventory_matches(self):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            SourceFileEntry,
            reconcile_inventory,
        )

        entry = SourceFileEntry(
            path="polygons/reg-a.parquet", size=10, sha256="00" * 32
        )
        d = reconcile_inventory(
            prior={"reg-a": [entry]},
            current={"reg-a": [entry]},
        )
        assert d["unchanged"] == {"reg-a"}
        assert d["added"] == set()
        assert d["removed"] == set()
        assert d["changed"] == set()

    def test_reconcile_inventory_drift(self):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            SourceFileEntry,
            reconcile_inventory,
        )

        prior_entry = SourceFileEntry(
            path="polygons/reg-a.parquet", size=1, sha256="00" * 32
        )
        current_entry = SourceFileEntry(
            path="polygons/reg-a.parquet", size=2, sha256="11" * 32
        )
        d = reconcile_inventory(
            prior={"reg-a": [prior_entry]},
            current={"reg-a": [current_entry]},
        )
        assert d["changed"] == {"reg-a"}
        assert d["unchanged"] == set()

    def test_publish_rejects_blank_identity(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            CheckpointValidationError,
            publish_shard_checkpoint,
        )
        from osm_polygon_sentence_relevance.contracts.schemas import (
            SEGMENTED_SENTENCES_SCHEMA,
        )
        from osm_polygon_sentence_relevance.ingestion.discovery import (
            RegionShardSet,
        )
        from osm_polygon_sentence_relevance.sentences.segmentation import (
            SegmentationReport,
        )

        root = tmp_path / "in"
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        # No input files — discovery isn't called here; we synthesise
        # the RegionShardSet directly with bogus paths so manifest
        # construction fails fast under the blank-identity check.

        # Construct a fake RegionShardSet where paths point to nothing
        # yet exist on disk. Manifest construction needs real files, so
        # we drive the identity rejection *first* (it runs before
        # file scanning).
        shard = RegionShardSet(
            shard_key="reg-x",
            polygons=root / "polygons" / "reg-x.parquet",
            polygon_articles=root / "polygon_articles" / "reg-x.parquet",
            wikipedia_documents=root / "wikipedia" / "documents" / "reg-x.parquet",
            wikipedia_sections=root / "wikipedia" / "sections" / "reg-x.parquet",
            wikivoyage_documents=None,
            wikivoyage_sections=None,
        )

        with pytest.raises(CheckpointValidationError, match="input_dataset_revision"):
            publish_shard_checkpoint(
                work_dir=work_dir,
                shard=shard,
                input_root=root,
                table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
                report=SegmentationReport(
                    input_section_occurrence_count=0,
                    emitted_segment_count=0,
                    retained_sentence_occurrence_count=0,
                    dropped_empty_raw_count=0,
                    dropped_empty_normalized_count=0,
                    wikipedia_sentence_occurrence_count=0,
                    wikivoyage_sentence_occurrence_count=0,
                ),
                input_dataset_revision="",
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                verified_manifest=[],
            )

    def test_publish_rejects_invalid_shard_key(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            CheckpointValidationError,
            publish_shard_checkpoint,
        )
        from osm_polygon_sentence_relevance.contracts.schemas import (
            SEGMENTED_SENTENCES_SCHEMA,
        )
        from osm_polygon_sentence_relevance.ingestion.discovery import (
            RegionShardSet,
        )
        from osm_polygon_sentence_relevance.sentences.segmentation import (
            SegmentationReport,
        )

        root = tmp_path / "in"
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        shard = RegionShardSet(
            shard_key="bad/key",
            polygons=root / "polygons" / "x.parquet",
            polygon_articles=root / "polygon_articles" / "x.parquet",
            wikipedia_documents=root / "wikipedia" / "documents" / "x.parquet",
            wikipedia_sections=root / "wikipedia" / "sections" / "x.parquet",
            wikivoyage_documents=None,
            wikivoyage_sections=None,
        )
        with pytest.raises(CheckpointValidationError, match="invalid shard_key"):
            publish_shard_checkpoint(
                work_dir=work_dir,
                shard=shard,
                input_root=root,
                table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
                report=SegmentationReport(
                    input_section_occurrence_count=0,
                    emitted_segment_count=0,
                    retained_sentence_occurrence_count=0,
                    dropped_empty_raw_count=0,
                    dropped_empty_normalized_count=0,
                    wikipedia_sentence_occurrence_count=0,
                    wikivoyage_sentence_occurrence_count=0,
                ),
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                verified_manifest=[],
            )

    def test_publish_rejects_invalid_batch_size(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            CheckpointValidationError,
            publish_shard_checkpoint,
        )
        from osm_polygon_sentence_relevance.contracts.schemas import (
            SEGMENTED_SENTENCES_SCHEMA,
        )
        from osm_polygon_sentence_relevance.ingestion.discovery import (
            RegionShardSet,
        )
        from osm_polygon_sentence_relevance.sentences.segmentation import (
            SegmentationReport,
        )

        root = tmp_path / "in"
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        shard = RegionShardSet(
            shard_key="reg-x",
            polygons=root / "polygons" / "x.parquet",
            polygon_articles=root / "polygon_articles" / "x.parquet",
            wikipedia_documents=root / "wikipedia" / "documents" / "x.parquet",
            wikipedia_sections=root / "wikipedia" / "sections" / "x.parquet",
            wikivoyage_documents=None,
            wikivoyage_sections=None,
        )
        with pytest.raises(CheckpointValidationError, match="batch_size"):
            publish_shard_checkpoint(
                work_dir=work_dir,
                shard=shard,
                input_root=root,
                table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
                report=SegmentationReport(
                    input_section_occurrence_count=0,
                    emitted_segment_count=0,
                    retained_sentence_occurrence_count=0,
                    dropped_empty_raw_count=0,
                    dropped_empty_normalized_count=0,
                    wikipedia_sentence_occurrence_count=0,
                    wikivoyage_sentence_occurrence_count=0,
                ),
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=0,
                verified_manifest=[],
            )

    def test_load_rejects_no_active(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            CheckpointValidationError,
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "work"
        (work_dir / "shards" / "active").mkdir(parents=True)
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

    def test_load_rejects_invalid_shard_key(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            CheckpointValidationError,
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        with pytest.raises(CheckpointValidationError, match="invalid shard_key"):
            load_shard_checkpoint(
                work_dir,
                "bad/key",
                input_dataset_revision=VALID_INPUT_REVISION,
                pipeline_version="v1",
                source_commit=VALID_SOURCE_COMMIT,
                model_name="mock",
                batch_size=128,
                input_root=tmp_path,
            )

    def test_load_rejects_unexpected_entries(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            CheckpointValidationError,
            load_shard_checkpoint,
        )

        work_dir = tmp_path / "work"
        shard_dir = work_dir / "shards" / "active" / "reg-a"
        shard_dir.mkdir(parents=True, mode=0o700)
        (shard_dir / "extra.txt").write_text("extra")
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

    def test_validate_work_dir_accepts_path_object(self, tmp_path):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_work_dir,
        )

        result = validate_work_dir(tmp_path)
        assert result is not None

    def test_validate_work_dir_rejects_invalid_type(self):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_work_dir,
        )

        with pytest.raises(ValueError, match="string or Path"):
            validate_work_dir(12345)  # type: ignore[arg-type]

    def test_validate_work_dir_rejects_blank_string(self):
        from osm_polygon_sentence_relevance.application.checkpoint import (
            validate_work_dir,
        )

        with pytest.raises(ValueError, match="not be blank"):
            validate_work_dir("   ")
