"""the implementation — RED characterisation of `run_pipeline` across sequential
shard staging. Documents the existing inventory reconciliation
behaviour when only one shard of multiple is staged at a time.

This test pins the existing behaviour (RED) and asserts the
contract that the new ``process_single_shard`` production facade
must satisfy (GREEN).
"""

from __future__ import annotations

from pathlib import Path

from osm_polygon_sentence_relevance.application.checkpoint import (
    load_shard_checkpoint,
)
from osm_polygon_sentence_relevance.application.pipeline import (
    process_single_shard,
    run_pipeline,
)
from osm_polygon_sentence_relevance.ingestion.discovery import discover_shards
from tests.support.arrow_factories import (
    make_polygon_article_row,
    make_polygon_row,
    make_section_row,
    make_wikipedia_document_row,
)
from tests.support.parquet_layouts import write_shard_parquet

VALID_SOURCE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
VALID_INPUT_REVISION = "abcdefabcdefabcdefabcdefabcdefabcdefabcd"


class _MockSegmenter:
    def split_batch(self, texts: list[str], languages: list[str]) -> list[list[str]]:
        return [[s.strip() for s in t.split(".") if s.strip()] for t in texts]


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


def _fixture_two_shards(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "in"
    out = tmp_path / "out"
    out.mkdir()
    _write_region(root, "reg-a")
    _write_region(root, "reg-b")
    return root, out


# ---------------------------------------------------------------------------
# RED characterisation: run_pipeline + 1-shard staged at a time.
#
# Documents the existing reconciliation behaviour. Other shards
# (not staged in this call) are classified as "removed" and their
# prior checkpoints get quarantined, defeating partial-snapshot runs.
# ---------------------------------------------------------------------------


def test_run_pipeline_quarantines_prior_shards_not_present_in_input(
    tmp_path: Path,
) -> None:
    root, out = _fixture_two_shards(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    segmenter = _MockSegmenter()

    # First run processes both shards.
    run_pipeline(
        root,
        out,
        segmenter,
        input_dataset_revision=VALID_INPUT_REVISION,
        pipeline_version="v1",
        work_dir=work,
        source_commit=VALID_SOURCE_COMMIT,
        model_name="mock",
    )
    active_after_first = {p.name for p in (work / "shards" / "active").iterdir()}
    assert {"reg-a", "reg-b"} <= active_after_first

    # Remove reg-b from the input root, simulating per-shard staging.
    for sub in (
        "polygons",
        "polygon_articles",
        "wikipedia/documents",
        "wikipedia/sections",
    ):
        (root / sub / "reg-b.parquet").unlink()

    out2 = tmp_path / "out2"
    out2.mkdir()
    segmenter2 = _MockSegmenter()

    # RED: this call will quarantine reg-b even though reg-b's
    # checkpoint is still valid. This documents the unsafe behaviour
    # with shared work-dir across sequential partial staging.
    run_pipeline(
        root,
        out2,
        segmenter2,
        input_dataset_revision=VALID_INPUT_REVISION,
        pipeline_version="v1",
        work_dir=work,
        source_commit=VALID_SOURCE_COMMIT,
        model_name="mock",
    )
    quarantine = work / "shards" / "quarantine"
    quarantined_keys = {p.name for p in quarantine.iterdir()}
    assert any("reg-b" in name for name in quarantined_keys), (
        "Existing run_pipeline behaviour pins reg-b quarantine."
    )


# ---------------------------------------------------------------------------
# GREEN contract: process_single_shard(RegionShardSet) preserves
# the prior work-dir invariants for all other shards.
# ---------------------------------------------------------------------------


def test_process_single_shard_writes_one_shard_checkpoint(tmp_path: Path) -> None:
    root, out = _fixture_two_shards(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    segmenter = _MockSegmenter()

    shards = sorted(discover_shards(root), key=lambda s: s.shard_key)
    assert len(shards) == 2

    # Process only the first shard.
    result = process_single_shard(
        shard=shards[0],
        input_root=root,
        segmenter=segmenter,
        work_dir=work,
        source_commit=VALID_SOURCE_COMMIT,
        input_dataset_revision=VALID_INPUT_REVISION,
        pipeline_version="v1",
        model_name="mock",
        batch_size=128,
    )

    from osm_polygon_sentence_relevance.application.pipeline import (
        ShardCheckpointResult,
    )

    assert isinstance(result, ShardCheckpointResult)
    assert (work / "shards" / "active" / "reg-a").exists()
    assert (work / "shards" / "active" / "reg-b").exists() is False

    # Strict validation passes when the file is read back.
    res_table, res_report, meta = load_shard_checkpoint(
        work,
        "reg-a",
        input_dataset_revision=VALID_INPUT_REVISION,
        pipeline_version="v1",
        source_commit=VALID_SOURCE_COMMIT,
        model_name="mock",
        batch_size=128,
        input_root=root,
    )
    assert res_table.num_rows >= 0
    assert meta["shard_key"] == "reg-a"
    assert (
        meta["segmented_table_bytes"]
        == (work / "shards" / "active" / "reg-a" / "segmented.parquet").stat().st_size
    )
    assert len(meta["segmented_schema_sha256"]) == 64


# ---------------------------------------------------------------------------
# GREEN contract: process_single_shard is idempotent when called twice
# against the same shard.
# ---------------------------------------------------------------------------


def test_process_single_shard_idempotent_for_same_shard(tmp_path: Path) -> None:
    root, out = _fixture_two_shards(tmp_path)
    work = tmp_path / "work"
    work.mkdir()

    shards = sorted(discover_shards(root), key=lambda s: s.shard_key)
    seg1 = _MockSegmenter()
    res1 = process_single_shard(
        shard=shards[0],
        input_root=root,
        segmenter=seg1,
        work_dir=work,
        source_commit=VALID_SOURCE_COMMIT,
        input_dataset_revision=VALID_INPUT_REVISION,
        pipeline_version="v1",
        model_name="mock",
        batch_size=128,
    )
    seg2 = _MockSegmenter()
    res2 = process_single_shard(
        shard=shards[0],
        input_root=root,
        segmenter=seg2,
        work_dir=work,
        source_commit=VALID_SOURCE_COMMIT,
        input_dataset_revision=VALID_INPUT_REVISION,
        pipeline_version="v1",
        model_name="mock",
        batch_size=128,
    )

    # Both calls succeed; the second call reuses the cached checkpoint.
    assert res1.published is True
    assert res2.published is False
    assert res2.reused is True


# ---------------------------------------------------------------------------
# GREEN contract: process_single_shard does not touch sibling shards.
# ---------------------------------------------------------------------------


def test_process_single_shard_does_not_touch_other_shards(tmp_path: Path) -> None:
    root, out = _fixture_two_shards(tmp_path)
    work = tmp_path / "work"
    work.mkdir()

    shards = sorted(discover_shards(root), key=lambda s: s.shard_key)
    seg = _MockSegmenter()
    process_single_shard(
        shard=shards[0],
        input_root=root,
        segmenter=seg,
        work_dir=work,
        source_commit=VALID_SOURCE_COMMIT,
        input_dataset_revision=VALID_INPUT_REVISION,
        pipeline_version="v1",
        model_name="mock",
        batch_size=128,
    )
    # Manually publish a checkpoint for reg-b so we can verify it is
    # not torn down by another process_single_shard call.
    process_single_shard(
        shard=shards[1],
        input_root=root,
        segmenter=_MockSegmenter(),
        work_dir=work,
        source_commit=VALID_SOURCE_COMMIT,
        input_dataset_revision=VALID_INPUT_REVISION,
        pipeline_version="v1",
        model_name="mock",
        batch_size=128,
    )

    reg_b_checkpoint_path = work / "shards" / "active" / "reg-b"
    assert reg_b_checkpoint_path.exists()
    # Now reprocess reg-a; reg-b's checkpoint must NOT be quarantined.
    process_single_shard(
        shard=shards[0],
        input_root=root,
        segmenter=seg,
        work_dir=work,
        source_commit=VALID_SOURCE_COMMIT,
        input_dataset_revision=VALID_INPUT_REVISION,
        pipeline_version="v1",
        model_name="mock",
        batch_size=128,
    )
    assert (work / "shards" / "active" / "reg-b").exists(), (
        "process_single_shard must NOT quarantine siblings"
    )
