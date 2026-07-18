from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.errors import SegmentationError
from osm_polygon_sentence_relevance.pipeline import PipelineResult, run_pipeline
from tests.helpers import get_checksum

# ===================================================================
# Helpers to build physical schema conforming tables
# ===================================================================


def make_table_from_rows(schema: pa.Schema, rows: list[dict]) -> pa.Table:
    data = {}
    for field in schema:
        col_values = []
        for r in rows:
            if field.name in r:
                col_values.append(r[field.name])
            else:
                if pa.types.is_string(field.type):
                    col_values.append("")
                elif pa.types.is_integer(field.type):
                    col_values.append(0)
                elif pa.types.is_floating(field.type):
                    col_values.append(0.0)
                elif pa.types.is_boolean(field.type):
                    col_values.append(False)
                else:
                    col_values.append(None)
        data[field.name] = pa.array(col_values, type=field.type)
    return pa.table(data, schema=schema)


def write_dummy_region(
    root: Path,
    shard_key: str,
    *,
    polygon_id="poly-1",
    wikidata="Q1",
    document_id="doc-1",
    article_id="art-1",
    source="wikipedia",
    language="en",
    text="First sentence. Second sentence.",
) -> None:
    # 1. polygons
    poly_row = {
        "polygon_id": polygon_id,
        "wikidata": wikidata,
        "region": shard_key,
        "name": "Poly Name",
        "tags": '{"name":"Poly Name"}',
        "osm_primary_tag": "primary",
        "lat": 12.34,
        "lon": 56.78,
    }
    # 2. polygon_articles
    pa_row = {
        "polygon_id": polygon_id,
        "article_id": article_id,
        "wikidata": wikidata,
        "language": language,
        "page_id": 1,
        "revision_id": 1,
    }

    # Write polygons
    polygons_dir = root / "polygons"
    polygons_dir.mkdir(parents=True, exist_ok=True)
    from osm_polygon_sentence_relevance.schemas import POLYGONS_SCHEMA

    pq.write_table(
        make_table_from_rows(POLYGONS_SCHEMA, [poly_row]),
        polygons_dir / f"{shard_key}.parquet",
    )

    # Write polygon_articles
    pa_dir = root / "polygon_articles"
    pa_dir.mkdir(parents=True, exist_ok=True)
    from osm_polygon_sentence_relevance.schemas import POLYGON_ARTICLES_SCHEMA

    pq.write_table(
        make_table_from_rows(POLYGON_ARTICLES_SCHEMA, [pa_row]),
        pa_dir / f"{shard_key}.parquet",
    )

    # Write Wikipedia (core table)
    wp_doc_row = {
        "document_id": document_id,
        "article_id": article_id,
        "wikidata": wikidata,
        "language": language,
        "site": f"{language}.wikipedia.org",
        "title": "Document Title",
        "url": "https://wikipedia.org",
        "page_id": 1,
        "revision_id": 1,
        "revision_timestamp": "2026-07-15T00:00:00Z",
        "content_hash": "doc-hash-1",
    }
    wp_sec_row = {
        "section_id": "sec-1",
        "document_id": document_id,
        "article_id": article_id,
        "wikidata": wikidata,
        "language": language,
        "site": f"{language}.wikipedia.org",
        "page_id": 1,
        "revision_id": 1,
        "section_index": 0,
        "section_path": '["Intro"]',
        "text": text if source in ("wikipedia", "both") else "",
        "content_hash": "sec-hash-1",
    }
    wp_doc_dir = root / "wikipedia/documents"
    wp_doc_dir.mkdir(parents=True, exist_ok=True)
    from osm_polygon_sentence_relevance.schemas import WIKIPEDIA_DOCUMENTS_SCHEMA

    pq.write_table(
        make_table_from_rows(
            WIKIPEDIA_DOCUMENTS_SCHEMA,
            [wp_doc_row] if source in ("wikipedia", "both") else [],
        ),
        wp_doc_dir / f"{shard_key}.parquet",
    )

    wp_sec_dir = root / "wikipedia/sections"
    wp_sec_dir.mkdir(parents=True, exist_ok=True)
    from osm_polygon_sentence_relevance.schemas import SECTIONS_SCHEMA

    pq.write_table(
        make_table_from_rows(
            SECTIONS_SCHEMA, [wp_sec_row] if source in ("wikipedia", "both") else []
        ),
        wp_sec_dir / f"{shard_key}.parquet",
    )

    # Write Wikivoyage (optional tables)
    if source in ("wikivoyage", "both"):
        wv_doc_row = {
            "document_id": document_id + "-wv",
            "article_id": article_id,
            "wikidata": wikidata,
            "language": language,
            "site": f"{language}.wikivoyage.org",
            "title": "Voyage Title",
            "url": "https://wikivoyage.org",
            "page_id": 2,
            "revision_id": 2,
            "revision_timestamp": "2026-07-15T00:00:00Z",
            "content_hash": "doc-hash-2",
        }
        wv_sec_row = {
            "section_id": "sec-wv-1",
            "document_id": document_id + "-wv",
            "article_id": article_id,
            "wikidata": wikidata,
            "language": language,
            "site": f"{language}.wikivoyage.org",
            "page_id": 2,
            "revision_id": 2,
            "section_index": 0,
            "section_path": '["Intro"]',
            "text": text,
            "content_hash": "sec-hash-2",
        }
        wv_doc_dir = root / "wikivoyage/documents"
        wv_doc_dir.mkdir(parents=True, exist_ok=True)
        from osm_polygon_sentence_relevance.schemas import WIKIVOYAGE_DOCUMENTS_SCHEMA

        pq.write_table(
            make_table_from_rows(WIKIVOYAGE_DOCUMENTS_SCHEMA, [wv_doc_row]),
            wv_doc_dir / f"{shard_key}.parquet",
        )

        wv_sec_dir = root / "wikivoyage/sections"
        wv_sec_dir.mkdir(parents=True, exist_ok=True)
        from osm_polygon_sentence_relevance.schemas import SECTIONS_SCHEMA

        pq.write_table(
            make_table_from_rows(SECTIONS_SCHEMA, [wv_sec_row]),
            wv_sec_dir / f"{shard_key}.parquet",
        )


# ===================================================================
# Mock Sentence Segmenter
# ===================================================================


class MockSegmenter:
    def __init__(self, split_fn=None):
        self.split_fn = split_fn or (
            lambda text: [s.strip() for s in text.split(".") if s.strip()]
        )
        self.calls_count = 0

    def split_batch(self, texts: list[str], languages: list[str]) -> list[list[str]]:
        self.calls_count += 1
        return [self.split_fn(text) for text in texts]


# ===================================================================
# Test Suite for Pipeline Orchestration (Phase 5B)
# ===================================================================


class TestPipeline:
    def test_run_pipeline_success_simple(self):
        segmenter = MockSegmenter()
        with (
            tempfile.TemporaryDirectory() as tmp_root,
            tempfile.TemporaryDirectory() as tmp_out,
        ):
            root = Path(tmp_root)
            out_dir = Path(tmp_out) / "output"
            write_dummy_region(root, "reg-1", text="Sentence.")

            res = run_pipeline(
                root,
                out_dir,
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
            )
            assert isinstance(res, PipelineResult)
            assert res.processed_regions_count == 1
            assert res.total_joined_section_occurrences == 1

    def test_invalid_configuration(self):
        segmenter = MockSegmenter()
        # Invalid batch_size
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            run_pipeline(
                "/tmp/input",
                "/tmp/output",
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
                batch_size=0,
            )
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            run_pipeline(
                "/tmp/input",
                "/tmp/output",
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
                batch_size="10",
            )
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            run_pipeline(
                "/tmp/input",
                "/tmp/output",
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
                batch_size=True,
            )

        # Blank/invalid config
        with pytest.raises(
            ValueError, match="input_dataset_revision must be a non-blank string"
        ):
            run_pipeline(
                "/tmp/input",
                "/tmp/output",
                segmenter,
                input_dataset_revision="  ",
                pipeline_version="v1",
            )
        with pytest.raises(
            ValueError, match="pipeline_version must be a non-blank string"
        ):
            run_pipeline(
                "/tmp/input",
                "/tmp/output",
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="",
            )

        # Invalid segmenter
        with pytest.raises(TypeError):
            run_pipeline(
                "/tmp/input",
                "/tmp/output",
                "not-a-segmenter",
                input_dataset_revision="r1",
                pipeline_version="v1",
            )

        # Same path
        with pytest.raises(ValueError, match="same path"):
            run_pipeline(
                "/tmp/same",
                "/tmp/same",
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
            )

        # Overlapping: input is ancestor of output
        with pytest.raises(ValueError, match="ancestor|overlap"):
            run_pipeline(
                "/tmp/ancestor",
                "/tmp/ancestor/child",
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
            )

        # Overlapping: output is ancestor of input
        with pytest.raises(ValueError, match="ancestor|overlap"):
            run_pipeline(
                "/tmp/ancestor/child",
                "/tmp/ancestor",
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
            )

        # Sibling paths allowed (does not raise ValueError for path overlap)
        with tempfile.TemporaryDirectory() as tmpdir:
            sib1 = Path(tmpdir) / "sib1"
            sib2 = Path(tmpdir) / "sib2"
            sib1.mkdir()
            res = run_pipeline(
                sib1,
                sib2,
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
            )
            assert isinstance(res, PipelineResult)
            assert res.processed_regions_count == 0

        # Ensure segmenter is never called
        assert segmenter.calls_count == 0

    def test_shuffled_regions_identical_output(self, monkeypatch):
        # Two regions processed in different discovery order produce identical tables/reports
        segmenter = MockSegmenter()
        with (
            tempfile.TemporaryDirectory() as tmp_root,
            tempfile.TemporaryDirectory() as tmp_out,
        ):
            root = Path(tmp_root)
            out_a = Path(tmp_out) / "out_a"
            out_b = Path(tmp_out) / "out_b"

            # Write two regions
            write_dummy_region(root, "reg-a", text="Sentence one. Sentence two.")
            write_dummy_region(root, "reg-b", text="Sentence three. Sentence four.")

            # Run A (normal discovery)
            res_a = run_pipeline(
                root,
                out_a,
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
                overwrite=True,
            )

            # Mock discover_shards to return in reverse order for Run B
            from osm_polygon_sentence_relevance.discovery import discover_shards

            orig_discover = discover_shards

            def mock_discover(r):
                shards = orig_discover(r)
                return tuple(reversed(shards))

            monkeypatch.setattr(
                "osm_polygon_sentence_relevance.application.pipeline.discover_shards",
                mock_discover,
            )

            # Run B (reversed discovery order)
            res_b = run_pipeline(
                root,
                out_b,
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
                overwrite=True,
            )

            # Assert identical results
            assert res_a.processed_regions_count == res_b.processed_regions_count
            assert (
                res_a.total_joined_section_occurrences
                == res_b.total_joined_section_occurrences
            )
            assert res_a.segmentation_report == res_b.segmentation_report
            assert res_a.finalization_report == res_b.finalization_report

            # Parquet contents are identical
            table_a = pq.read_table(out_a / "sentences.parquet")
            table_b = pq.read_table(out_b / "sentences.parquet")
            assert table_a.equals(table_b)

            # Manifest contents are identical
            with (
                open(out_a / "manifest.json") as fa,
                open(out_b / "manifest.json") as fb,
            ):
                assert fa.read() == fb.read()

    def test_wikipedia_and_wikivoyage_end_to_end(self):
        segmenter = MockSegmenter()
        with (
            tempfile.TemporaryDirectory() as tmp_root,
            tempfile.TemporaryDirectory() as tmp_out,
        ):
            root = Path(tmp_root)
            out_dir = Path(tmp_out) / "output"

            write_dummy_region(root, "reg-1", source="both", text="One sentence.")

            res = run_pipeline(
                root,
                out_dir,
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
            )

            assert res.processed_regions_count == 1
            assert (
                res.total_joined_section_occurrences == 2
            )  # 1 WP section occurrence + 1 WV section occurrence
            # In report, WP and WV counts should remain observable
            assert res.segmentation_report.wikipedia_sentence_occurrence_count == 1
            assert res.segmentation_report.wikivoyage_sentence_occurrence_count == 1
            assert res.export_result.manifest_data["counts_by_source"] == {
                "wikipedia": 1
            }  # deduplicated to wikipedia canonical

    def test_cross_region_duplicate_sentences_dedup_globally(self):
        segmenter = MockSegmenter()
        with (
            tempfile.TemporaryDirectory() as tmp_root,
            tempfile.TemporaryDirectory() as tmp_out,
        ):
            root = Path(tmp_root)
            out_dir = Path(tmp_out) / "output"

            # Both regions have the exact same sentence
            write_dummy_region(root, "reg-1", text="Duplicate sentence.")
            write_dummy_region(root, "reg-2", text="Duplicate sentence.")

            res = run_pipeline(
                root,
                out_dir,
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
            )

            # We have 2 input occurrences, but global finalization deduplicates to 1 output row
            assert res.finalization_report.input_sentence_occurrence_count == 2
            assert res.finalization_report.output_sentence_count == 1
            assert res.finalization_report.duplicate_occurrence_count_removed == 1

            table = pq.read_table(out_dir / "sentences.parquet")
            assert table.num_rows == 1

    def test_empty_sentence_results(self):
        # A segmenter that returns empty segments (effectively dropping everything)
        segmenter = MockSegmenter(split_fn=lambda text: [])
        with (
            tempfile.TemporaryDirectory() as tmp_root,
            tempfile.TemporaryDirectory() as tmp_out,
        ):
            root = Path(tmp_root)
            out_dir = Path(tmp_out) / "output"

            write_dummy_region(root, "reg-1", text="Some sentence.")

            res = run_pipeline(
                root,
                out_dir,
                segmenter,
                input_dataset_revision="rev-empty",
                pipeline_version="ver-empty",
            )

            assert res.processed_regions_count == 1
            assert res.finalization_report.output_sentence_count == 0

            # The empty result retains schema and revision/version metadata
            table = pq.read_table(out_dir / "sentences.parquet")
            assert table.num_rows == 0
            meta = table.schema.metadata
            assert meta.get(b"input_dataset_revision") == b"rev-empty"
            assert meta.get(b"pipeline_version") == b"ver-empty"

    def test_processing_failure_leaves_output_unchanged(self, monkeypatch):
        # Setup pre-existing valid output
        segmenter = MockSegmenter()
        with (
            tempfile.TemporaryDirectory() as tmp_root,
            tempfile.TemporaryDirectory() as tmp_out,
        ):
            root = Path(tmp_root)
            out_dir = Path(tmp_out) / "output"

            write_dummy_region(root, "reg-1", text="Sentence.")
            _export_final = run_pipeline(
                root,
                out_dir,
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
            )
            prev_pq = out_dir / "sentences.parquet"
            prev_checksum = get_checksum(prev_pq)

            # Mock segmenter to throw error during next run
            def mock_fail(*args, **kwargs):
                raise RuntimeError("Segmentation engine crashed")

            monkeypatch.setattr(segmenter, "split_batch", mock_fail)

            with pytest.raises(SegmentationError):
                run_pipeline(
                    root,
                    out_dir,
                    segmenter,
                    input_dataset_revision="r1",
                    pipeline_version="v1",
                    overwrite=True,
                )

            # Verify target remains unchanged
            assert out_dir.exists()
            assert get_checksum(prev_pq) == prev_checksum

    def test_shards_explicitly_sorted(self, monkeypatch):
        processed_order = []

        from osm_polygon_sentence_relevance.pipeline import (
            build_region_section_occurrences,
        )

        orig_build = build_region_section_occurrences

        def mock_build(shard):
            processed_order.append(shard.shard_key)
            return orig_build(shard)

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.pipeline.build_region_section_occurrences",
            mock_build,
        )

        segmenter = MockSegmenter()
        with (
            tempfile.TemporaryDirectory() as tmp_root,
            tempfile.TemporaryDirectory() as tmp_out,
        ):
            root = Path(tmp_root)
            out_dir = Path(tmp_out) / "output"

            write_dummy_region(root, "reg-b", text="Sentence.")
            write_dummy_region(root, "reg-a", text="Sentence.")

            from osm_polygon_sentence_relevance.discovery import discover_shards

            orig_discover = discover_shards

            def mock_discover(r):
                shards = orig_discover(r)
                return tuple(sorted(shards, key=lambda s: s.shard_key, reverse=True))

            monkeypatch.setattr(
                "osm_polygon_sentence_relevance.application.pipeline.discover_shards",
                mock_discover,
            )

            run_pipeline(
                root,
                out_dir,
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
            )

            assert processed_order == ["reg-a", "reg-b"]

    def test_report_aggregation_failure_safety(self, monkeypatch):
        segmenter = MockSegmenter()
        with (
            tempfile.TemporaryDirectory() as tmp_root,
            tempfile.TemporaryDirectory() as tmp_out,
        ):
            root = Path(tmp_root)
            out_dir = Path(tmp_out) / "output"

            write_dummy_region(root, "reg-1", text="Sentence.")
            run_pipeline(
                root,
                out_dir,
                segmenter,
                input_dataset_revision="r1",
                pipeline_version="v1",
            )
            prev_pq = out_dir / "sentences.parquet"
            prev_checksum = get_checksum(prev_pq)

            # Change the data so if it runs successfully, it writes a different checksum
            write_dummy_region(root, "reg-1", text="Different sentence.")

            import inspect

            from osm_polygon_sentence_relevance.pipeline import SegmentationReport

            orig_init = SegmentationReport.__init__

            def mock_init(self, *args, **kwargs):
                frame = inspect.currentframe().f_back
                if "pipeline.py" in frame.f_code.co_filename:
                    raise ValueError("Simulated aggregation validation failure")
                return orig_init(self, *args, **kwargs)

            monkeypatch.setattr(SegmentationReport, "__init__", mock_init)

            with pytest.raises(
                ValueError, match="Simulated aggregation validation failure"
            ):
                run_pipeline(
                    root,
                    out_dir,
                    segmenter,
                    input_dataset_revision="r1",
                    pipeline_version="v1",
                    overwrite=True,
                )

            assert out_dir.exists()
            assert get_checksum(prev_pq) == prev_checksum

    def test_blank_input_dataset_id_is_rejected_before_output(self):
        """Blank/non-string ``input_dataset_id`` is rejected before any
        output mutation, alongside the existing revision/version checks.
        """
        segmenter = MockSegmenter()
        with (
            tempfile.TemporaryDirectory() as tmp_root,
            tempfile.TemporaryDirectory() as tmp_out,
        ):
            root = Path(tmp_root) / "in"
            out_dir = Path(tmp_out) / "out"
            with pytest.raises(
                ValueError,
                match="input_dataset_id must be a non-blank string",
            ):
                run_pipeline(
                    root,
                    out_dir,
                    segmenter,
                    input_dataset_revision="r1",
                    pipeline_version="v1",
                    input_dataset_id="   ",
                )
            # Output directory must not exist (no partial state).
            assert not out_dir.exists()
        assert segmenter.calls_count == 0

    def test_run_pipeline_rejects_surrounding_whitespace_dataset_id_before_discovery(
        self, monkeypatch
    ):
        """A non-blank ``input_dataset_id`` with leading/trailing
        whitespace is rejected before discovery or filesystem reads.
        """

        # ``discover_shards`` must never be called for an invalid id.
        def mock_discover_shards(*args, **kwargs):
            raise AssertionError(
                "discover_shards must not be called for an invalid dataset_id"
            )

        monkeypatch.setattr(
            "osm_polygon_sentence_relevance.application.pipeline.discover_shards",
            mock_discover_shards,
        )
        segmenter = MockSegmenter()
        with (
            tempfile.TemporaryDirectory() as tmp_root,
            tempfile.TemporaryDirectory() as tmp_out,
        ):
            root = Path(tmp_root) / "in"
            out_dir = Path(tmp_out) / "out"
            with pytest.raises(
                ValueError,
                match="input_dataset_id.*surrounding whitespace",
            ):
                run_pipeline(
                    root,
                    out_dir,
                    segmenter,
                    input_dataset_revision="r1",
                    pipeline_version="v1",
                    input_dataset_id="  NoeFlandre/wikidata-only  ",
                )
            # No partial output directory may exist.
            assert not out_dir.exists()
        # Segmentation must never be invoked.
        assert segmenter.calls_count == 0
