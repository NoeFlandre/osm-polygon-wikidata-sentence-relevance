from __future__ import annotations

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.errors import FinalizationError
from osm_polygon_sentence_relevance.finalization import (
    deterministic_sentence_id,
    finalize_sentence_dataset,
    sentence_content_hash,
)
from osm_polygon_sentence_relevance.schemas import (
    OUTPUT_SENTENCE_SCHEMA,
    SEGMENTED_SENTENCES_SCHEMA,
)
from tests.helpers import make_segmented_row

# ===================================================================
# Helpers to construct SEGMENTED_SENTENCES_SCHEMA tables
# ===================================================================


def rows_to_table(rows: list[dict]) -> pa.Table:
    if not rows:
        return SEGMENTED_SENTENCES_SCHEMA.empty_table()
    data = {}
    for field in SEGMENTED_SENTENCES_SCHEMA:
        col_values = [r[field.name] for r in rows]
        data[field.name] = pa.array(col_values, type=field.type)
    return pa.table(data, schema=SEGMENTED_SENTENCES_SCHEMA)


# ===================================================================
# Test suite for Finalization
# ===================================================================


class TestFinalization:
    def test_known_sha256_vectors(self):
        # Known SHA-256 of "hello world" UTF-8
        # b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
        h = sentence_content_hash("hello world")
        assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

        # Known ID hash of:
        # {"language":"en","polygon_id":"poly-1","sentence_content_hash":"content-hash-1","version":1}
        # -> 9ea8b47c9a8014abeb23d42ba4438524df55fb29b316c36e4e820ae39314a1f3
        sent_id = deterministic_sentence_id("poly-1", "en", "content-hash-1")
        assert (
            sent_id
            == "9ea8b47c9a8014abeb23d42ba4438524df55fb29b316c36e4e820ae39314a1f3"
        )

    def test_deterministic_id_and_delimiter_collision_safety(self):
        # If we used simple delimiter joining (e.g. "poly,en" + "en" vs "poly" + "en,en"), they might collide.
        # But JSON-based key sorting prevents this.
        id1 = deterministic_sentence_id("poly,en", "en", "content-hash")
        id2 = deterministic_sentence_id("poly", "en,en", "content-hash")
        assert id1 != id2

    def test_previous_next_within_one_section(self):
        # Three sentences within the same section and document.
        # The input is not sorted, to verify finalization sorts by sentence_index before determining context.
        rows = [
            make_segmented_row(
                sentence_index=2, sentence_text_normalized="sentence three"
            ),
            make_segmented_row(
                sentence_index=0, sentence_text_normalized="sentence one"
            ),
            make_segmented_row(
                sentence_index=1, sentence_text_normalized="sentence two"
            ),
        ]
        table = rows_to_table(rows)
        res = finalize_sentence_dataset(
            table, input_dataset_revision="r1", pipeline_version="v1"
        )
        res_rows = res.table.to_pylist()

        out_by_text = {r["sentence_text_normalized"]: r for r in res_rows}
        assert len(out_by_text) == 3

        assert out_by_text["sentence one"]["previous_sentence"] is None
        assert out_by_text["sentence one"]["next_sentence"] == "sentence two"

        assert out_by_text["sentence two"]["previous_sentence"] == "sentence one"
        assert out_by_text["sentence two"]["next_sentence"] == "sentence three"

        assert out_by_text["sentence three"]["previous_sentence"] == "sentence two"
        assert out_by_text["sentence three"]["next_sentence"] is None

    def test_context_boundary_isolation(self):
        # Boundaries: polygon_id, source, document_id, section_id.
        # We construct rows that have different boundaries and verify context is NOT shared.
        rows = [
            # Group 1: poly-1, wikipedia, doc-1, sec-1
            make_segmented_row(
                polygon_id="poly-1",
                source="wikipedia",
                document_id="doc-1",
                section_id="sec-1",
                sentence_index=0,
                sentence_text_normalized="s1",
            ),
            # Group 2: poly-2 (different polygon)
            make_segmented_row(
                polygon_id="poly-2",
                source="wikipedia",
                document_id="doc-1",
                section_id="sec-1",
                sentence_index=1,
                sentence_text_normalized="s2",
            ),
            # Group 3: wikivoyage (different source)
            make_segmented_row(
                polygon_id="poly-1",
                source="wikivoyage",
                document_id="doc-1",
                section_id="sec-1",
                sentence_index=1,
                sentence_text_normalized="s3",
            ),
            # Group 4: doc-2 (different document)
            make_segmented_row(
                polygon_id="poly-1",
                source="wikipedia",
                document_id="doc-2",
                section_id="sec-1",
                sentence_index=1,
                sentence_text_normalized="s4",
            ),
            # Group 5: sec-2 (different section)
            make_segmented_row(
                polygon_id="poly-1",
                source="wikipedia",
                document_id="doc-1",
                section_id="sec-2",
                sentence_index=1,
                sentence_text_normalized="s5",
            ),
        ]
        table = rows_to_table(rows)
        res = finalize_sentence_dataset(
            table, input_dataset_revision="r1", pipeline_version="v1"
        )
        res_rows = res.table.to_pylist()

        for r in res_rows:
            assert r["previous_sentence"] is None
            assert r["next_sentence"] is None

    def test_same_source_duplicate_collapse(self):
        # Two identical occurrences in the same source but different document/section.
        rows = [
            make_segmented_row(
                document_id="doc-1",
                section_id="sec-1",
                sentence_index=0,
                sentence_text_normalized="dup",
            ),
            make_segmented_row(
                document_id="doc-2",
                section_id="sec-2",
                sentence_index=1,
                sentence_text_normalized="dup",
            ),
        ]
        table = rows_to_table(rows)
        res = finalize_sentence_dataset(
            table, input_dataset_revision="r1", pipeline_version="v1"
        )
        assert res.table.num_rows == 1
        assert res.report.output_sentence_count == 1
        assert res.report.duplicate_occurrence_count_removed == 1

    def test_cross_source_duplicate_collapse(self):
        # Duplicate across wikipedia and wikivoyage.
        rows = [
            make_segmented_row(
                source="wikipedia",
                document_id="doc-wp",
                sentence_text_normalized="shared",
            ),
            make_segmented_row(
                source="wikivoyage",
                document_id="doc-wv",
                sentence_text_normalized="shared",
            ),
        ]
        table = rows_to_table(rows)
        res = finalize_sentence_dataset(
            table, input_dataset_revision="r1", pipeline_version="v1"
        )
        assert res.table.num_rows == 1
        assert res.report.output_sentence_count == 1
        assert res.report.cross_source_duplicate_group_count == 1

    def test_wikipedia_canonical_preference(self):
        # Wikipedia before Wikivoyage.
        rows = [
            make_segmented_row(
                source="wikivoyage", document_id="doc-a", sentence_text_normalized="dup"
            ),
            make_segmented_row(
                source="wikipedia", document_id="doc-b", sentence_text_normalized="dup"
            ),
        ]
        table = rows_to_table(rows)
        res = finalize_sentence_dataset(
            table, input_dataset_revision="r1", pipeline_version="v1"
        )
        row = res.table.to_pylist()[0]
        assert row["source"] == "wikipedia"
        assert row["document_id"] == "doc-b"

    def test_deterministic_canonical_tie_breaking(self):
        # Same source: tie-break on document_id, section_index, section_id, sentence_index.
        # We verify that changing each one changes the selected canonical.

        # 1. document_id tie break (ascending)
        rows = [
            make_segmented_row(document_id="doc-b", sentence_text_normalized="dup"),
            make_segmented_row(document_id="doc-a", sentence_text_normalized="dup"),
        ]
        res = finalize_sentence_dataset(
            rows_to_table(rows), input_dataset_revision="r1", pipeline_version="v1"
        )
        assert res.table.to_pylist()[0]["document_id"] == "doc-a"

        # 2. section_index tie break (ascending)
        rows = [
            make_segmented_row(
                document_id="doc-a", section_index=10, sentence_text_normalized="dup"
            ),
            make_segmented_row(
                document_id="doc-a", section_index=5, sentence_text_normalized="dup"
            ),
        ]
        res = finalize_sentence_dataset(
            rows_to_table(rows), input_dataset_revision="r1", pipeline_version="v1"
        )
        assert res.table.to_pylist()[0]["section_index"] == 5

        # 3. section_id tie break (ascending)
        rows = [
            make_segmented_row(
                document_id="doc-a",
                section_index=5,
                section_id="sec-y",
                sentence_text_normalized="dup",
            ),
            make_segmented_row(
                document_id="doc-a",
                section_index=5,
                section_id="sec-x",
                sentence_text_normalized="dup",
            ),
        ]
        res = finalize_sentence_dataset(
            rows_to_table(rows), input_dataset_revision="r1", pipeline_version="v1"
        )
        assert res.table.to_pylist()[0]["section_id"] == "sec-x"

        # 4. sentence_index tie break (ascending)
        rows = [
            make_segmented_row(
                document_id="doc-a",
                section_index=5,
                section_id="sec-x",
                sentence_index=3,
                sentence_text_normalized="dup",
            ),
            make_segmented_row(
                document_id="doc-a",
                section_index=5,
                section_id="sec-x",
                sentence_index=1,
                sentence_text_normalized="dup",
            ),
        ]
        res = finalize_sentence_dataset(
            rows_to_table(rows), input_dataset_revision="r1", pipeline_version="v1"
        )
        assert res.table.to_pylist()[0]["sentence_index"] == 1

    def test_duplicate_count_and_sorted_sources(self):
        # 3 occurrences: two wikivoyage, one wikipedia.
        rows = [
            make_segmented_row(source="wikivoyage", sentence_text_normalized="dup"),
            make_segmented_row(source="wikivoyage", sentence_text_normalized="dup"),
            make_segmented_row(source="wikipedia", sentence_text_normalized="dup"),
        ]
        res = finalize_sentence_dataset(
            rows_to_table(rows), input_dataset_revision="r1", pipeline_version="v1"
        )
        row = res.table.to_pylist()[0]
        assert row["duplicate_occurrence_count"] == 3
        # sorted: ["wikipedia", "wikivoyage"]
        assert row["duplicate_sources"] == ["wikipedia", "wikivoyage"]

    def test_same_text_across_polygons_remains_separate(self):
        # Different polygons.
        rows = [
            make_segmented_row(polygon_id="poly-1", sentence_text_normalized="same"),
            make_segmented_row(polygon_id="poly-2", sentence_text_normalized="same"),
        ]
        res = finalize_sentence_dataset(
            rows_to_table(rows), input_dataset_revision="r1", pipeline_version="v1"
        )
        assert res.table.num_rows == 2

    def test_same_text_across_languages_remains_separate(self):
        # Different languages.
        rows = [
            make_segmented_row(language="en", sentence_text_normalized="same"),
            make_segmented_row(language="fr", sentence_text_normalized="same"),
        ]
        res = finalize_sentence_dataset(
            rows_to_table(rows), input_dataset_revision="r1", pipeline_version="v1"
        )
        assert res.table.num_rows == 2

    def test_raw_text_differences_with_identical_normalized_text_collapse(self):
        # Different raw text, same normalized text.
        rows = [
            make_segmented_row(
                sentence_text_raw="Raw A", sentence_text_normalized="normalized"
            ),
            make_segmented_row(
                sentence_text_raw="Raw B", sentence_text_normalized="normalized"
            ),
        ]
        res = finalize_sentence_dataset(
            rows_to_table(rows), input_dataset_revision="r1", pipeline_version="v1"
        )
        assert res.table.num_rows == 1

    def test_shuffle_invariant_table_and_report(self):
        # Sorting order is deterministic (polygon_id, language, sentence_id).
        # We test that shuffling input rows produces identical output table and report.
        row1 = make_segmented_row(
            polygon_id="poly-1", language="en", sentence_text_normalized="one"
        )
        row2 = make_segmented_row(
            polygon_id="poly-1", language="fr", sentence_text_normalized="two"
        )
        row3 = make_segmented_row(
            polygon_id="poly-2", language="en", sentence_text_normalized="three"
        )

        table_a = rows_to_table([row1, row2, row3])
        table_b = rows_to_table([row3, row1, row2])

        res_a = finalize_sentence_dataset(
            table_a, input_dataset_revision="r1", pipeline_version="v1"
        )
        res_b = finalize_sentence_dataset(
            table_b, input_dataset_revision="r1", pipeline_version="v1"
        )

        assert res_a.table.equals(res_b.table)
        assert res_a.report == res_b.report

    def test_exact_output_schema(self):
        # The schema should exactly match OUTPUT_SENTENCE_SCHEMA (including order, type, nullability).
        rows = [make_segmented_row()]
        res = finalize_sentence_dataset(
            rows_to_table(rows), input_dataset_revision="r1", pipeline_version="v1"
        )

        # Verify schema field names and types exactly match
        assert res.table.schema.equals(OUTPUT_SENTENCE_SCHEMA)

        # Verify column order matches exactly
        assert res.table.column_names == list(OUTPUT_SENTENCE_SCHEMA.names)

        # Check nullability validation is correct
        for name in OUTPUT_SENTENCE_SCHEMA.names:
            expected_field = OUTPUT_SENTENCE_SCHEMA.field(name)
            actual_field = res.table.schema.field(name)
            assert expected_field.nullable == actual_field.nullable, (
                f"{name} nullability"
            )

    def test_revision_version_fields(self):
        rows = [make_segmented_row()]
        res = finalize_sentence_dataset(
            rows_to_table(rows),
            input_dataset_revision="test-rev",
            pipeline_version="test-ver",
        )
        row = res.table.to_pylist()[0]
        assert row["input_dataset_revision"] == "test-rev"
        assert row["pipeline_version"] == "test-ver"

    def test_blank_configuration_rejection(self):
        table = rows_to_table([make_segmented_row()])
        with pytest.raises(FinalizationError):
            finalize_sentence_dataset(
                table, input_dataset_revision="", pipeline_version="v1"
            )
        with pytest.raises(FinalizationError):
            finalize_sentence_dataset(
                table, input_dataset_revision="  ", pipeline_version="v1"
            )
        with pytest.raises(FinalizationError):
            finalize_sentence_dataset(
                table, input_dataset_revision="r1", pipeline_version=""
            )
        with pytest.raises(FinalizationError):
            finalize_sentence_dataset(
                table, input_dataset_revision="r1", pipeline_version="   "
            )

    def test_revision_version_type_rejection(self):
        table = rows_to_table([make_segmented_row()])
        # non-string revision
        with pytest.raises(FinalizationError):
            finalize_sentence_dataset(
                table, input_dataset_revision=123, pipeline_version="v1"
            )
        # non-string version
        with pytest.raises(FinalizationError):
            finalize_sentence_dataset(
                table, input_dataset_revision="r1", pipeline_version=True
            )
        # None version/revision
        with pytest.raises(FinalizationError):
            finalize_sentence_dataset(
                table, input_dataset_revision=None, pipeline_version="v1"
            )
        with pytest.raises(FinalizationError):
            finalize_sentence_dataset(
                table, input_dataset_revision="r1", pipeline_version=None
            )

    def test_invalid_sources_rejected(self):
        # source must be "wikipedia" or "wikivoyage"
        rows = [make_segmented_row(source="invalid_source")]
        table = rows_to_table(rows)
        with pytest.raises(FinalizationError):
            finalize_sentence_dataset(
                table, input_dataset_revision="r1", pipeline_version="v1"
            )

    def test_surrounding_whitespace_input_dataset_id_rejected(self, monkeypatch):
        """A non-blank ``input_dataset_id`` with leading/trailing
        whitespace is rejected with ``FinalizationError`` BEFORE any
        table processing. Surrounding whitespace is never silently
        normalized away.
        """
        # Verify the rejection happens BEFORE table processing by
        # passing an obviously-invalid table value (not a ``pa.Table``)
        # and asserting the raised error is the surrounding-whitespace
        # FinalizationError rather than the ``TypeError`` that would
        # fire from the ``isinstance(table, pa.Table)`` check.
        sentinel = object()
        with pytest.raises(
            FinalizationError,
            match="input_dataset_id.*surrounding whitespace",
        ):
            finalize_sentence_dataset(
                sentinel,  # type: ignore[arg-type]
                input_dataset_revision="r1",
                pipeline_version="v1",
                input_dataset_id="  NoeFlandre/wikidata-only  ",
            )

    def test_deterministic_tie_breaking_order_independence(self):
        # Two rows with identical tie-breaking canonical keys (document_id, section_index, etc.)
        # but different other fields (e.g., lat)
        row_a = make_segmented_row(lat=10.0, sentence_text_normalized="dup")
        row_b = make_segmented_row(lat=20.0, sentence_text_normalized="dup")

        table_1 = rows_to_table([row_a, row_b])
        table_2 = rows_to_table([row_b, row_a])

        res_1 = finalize_sentence_dataset(
            table_1, input_dataset_revision="r1", pipeline_version="v1"
        )
        res_2 = finalize_sentence_dataset(
            table_2, input_dataset_revision="r1", pipeline_version="v1"
        )

        # The finalized tables must be identical, including picked lat value!
        assert res_1.table.equals(res_2.table)
        assert res_1.report == res_2.report

    def test_deterministic_tie_breaking_nullable_fields(self):
        # 1. article_id (None vs string)
        row_art_none = make_segmented_row(
            article_id=None, sentence_text_normalized="dup"
        )
        row_art_str = make_segmented_row(
            article_id="art-xyz", sentence_text_normalized="dup"
        )
        table_1a = rows_to_table([row_art_none, row_art_str])
        table_1b = rows_to_table([row_art_str, row_art_none])
        res_1a = finalize_sentence_dataset(
            table_1a, input_dataset_revision="r1", pipeline_version="v1"
        )
        res_1b = finalize_sentence_dataset(
            table_1b, input_dataset_revision="r1", pipeline_version="v1"
        )
        assert res_1a.table.equals(res_1b.table)

        # 2. polygon_name (None vs string)
        row_poly_none = make_segmented_row(
            polygon_name=None, sentence_text_normalized="dup"
        )
        row_poly_str = make_segmented_row(
            polygon_name="Poly ABC", sentence_text_normalized="dup"
        )
        table_2a = rows_to_table([row_poly_none, row_poly_str])
        table_2b = rows_to_table([row_poly_str, row_poly_none])
        res_2a = finalize_sentence_dataset(
            table_2a, input_dataset_revision="r1", pipeline_version="v1"
        )
        res_2b = finalize_sentence_dataset(
            table_2b, input_dataset_revision="r1", pipeline_version="v1"
        )
        assert res_2a.table.equals(res_2b.table)

        # 3. lat/lon (None vs float)
        row_lat_none = make_segmented_row(
            lat=None, lon=None, sentence_text_normalized="dup"
        )
        row_lat_float = make_segmented_row(
            lat=34.5, lon=69.1, sentence_text_normalized="dup"
        )
        table_3a = rows_to_table([row_lat_none, row_lat_float])
        table_3b = rows_to_table([row_lat_float, row_lat_none])
        res_3a = finalize_sentence_dataset(
            table_3a, input_dataset_revision="r1", pipeline_version="v1"
        )
        res_3b = finalize_sentence_dataset(
            table_3b, input_dataset_revision="r1", pipeline_version="v1"
        )
        assert res_3a.table.equals(res_3b.table)

    def test_empty_input(self):
        empty_table = rows_to_table([])
        res = finalize_sentence_dataset(
            empty_table, input_dataset_revision="r1", pipeline_version="v1"
        )

        assert res.table.num_rows == 0
        assert res.table.schema.equals(OUTPUT_SENTENCE_SCHEMA)

        assert res.report.input_sentence_occurrence_count == 0
        assert res.report.output_sentence_count == 0
        assert res.report.duplicate_occurrence_count_removed == 0
        assert res.report.cross_source_duplicate_group_count == 0

    def test_input_unchanged(self):
        rows = [make_segmented_row()]
        table = rows_to_table(rows)
        original_names = table.column_names
        original_num_rows = table.num_rows

        # Run finalization
        finalize_sentence_dataset(
            table, input_dataset_revision="r1", pipeline_version="v1"
        )

        # Verify no mutations on table
        assert table.column_names == original_names
        assert table.num_rows == original_num_rows

    def test_report_arithmetic(self):
        # 4 input rows
        # - Group A (poly-1, en, normalized_a): Wikipedia, document 1, sec 1.
        # - Group A (poly-1, en, normalized_a): Wikivoyage, document 2, sec 2.
        # - Group B (poly-1, en, normalized_b): Wikipedia, document 1, sec 1.
        # - Group C (poly-1, fr, normalized_a): Wikipedia, document 1, sec 1.

        # Group A collapses (cross-source!). So input_count = 4, output_count = 3.
        # removed = 4 - 3 = 1.
        # cross_source_duplicate_group_count = 1.
        rows = [
            make_segmented_row(
                polygon_id="poly-1",
                language="en",
                sentence_text_normalized="normalized_a",
                source="wikipedia",
                document_id="doc-1",
                section_id="sec-1",
            ),
            make_segmented_row(
                polygon_id="poly-1",
                language="en",
                sentence_text_normalized="normalized_a",
                source="wikivoyage",
                document_id="doc-2",
                section_id="sec-2",
            ),
            make_segmented_row(
                polygon_id="poly-1",
                language="en",
                sentence_text_normalized="normalized_b",
                source="wikipedia",
                document_id="doc-1",
                section_id="sec-1",
            ),
            make_segmented_row(
                polygon_id="poly-1",
                language="fr",
                sentence_text_normalized="normalized_a",
                source="wikipedia",
                document_id="doc-1",
                section_id="sec-1",
            ),
        ]
        table = rows_to_table(rows)
        res = finalize_sentence_dataset(
            table, input_dataset_revision="r1", pipeline_version="v1"
        )

        assert res.report.input_sentence_occurrence_count == 4
        assert res.report.output_sentence_count == 3
        assert res.report.duplicate_occurrence_count_removed == 1
        assert res.report.cross_source_duplicate_group_count == 1

    def test_synthetic_end_to_end_flow(self):
        from osm_polygon_sentence_relevance.sentence_table import (
            segment_joined_sections,
        )

        # 1. Phase 2 joined table: two rows
        row1 = make_joined_row(
            polygon_id="poly-1",
            source="wikipedia",
            section_text_raw="Hello world. Welcome here.",
        )
        row2 = make_joined_row(
            polygon_id="poly-1",
            source="wikivoyage",
            section_text_raw="Hello world. Travel guide.",
        )
        joined_table = joined_rows_to_table([row1, row2])

        # 2. Fake segmenter: splits texts into sentences
        class FakeSegmenter:
            def split_batch(self, texts, languages):
                res = []
                for t in texts:
                    if "Hello world. Welcome here." in t:
                        res.append(["Hello world.", "Welcome here."])
                    elif "Hello world. Travel guide." in t:
                        res.append(["Hello world.", "Travel guide."])
                    else:
                        res.append([t])
                return res

        segmenter = FakeSegmenter()

        # 3. Phase 3 segmented table
        segmented_res = segment_joined_sections(joined_table, segmenter, batch_size=1)
        segmented_table = segmented_res.table

        # Verify segmented table has 4 rows
        assert segmented_table.num_rows == 4

        # 4. Phase 4 final table
        finalized_res = finalize_sentence_dataset(
            segmented_table,
            input_dataset_revision="main_rev",
            pipeline_version="1.2.3",
        )
        final_table = finalized_res.table

        # Verify deduplication collapses "Hello world." across sources!
        assert final_table.num_rows == 3
        assert finalized_res.report.input_sentence_occurrence_count == 4
        assert finalized_res.report.output_sentence_count == 3
        assert finalized_res.report.duplicate_occurrence_count_removed == 1
        assert finalized_res.report.cross_source_duplicate_group_count == 1


# ===================================================================
# Helpers to construct JOINED_SECTIONS_SCHEMA tables for end-to-end
# ===================================================================


def make_joined_row(
    *,
    polygon_id="poly-1",
    wikidata="Q1",
    document_id="doc-1",
    article_id="art-1",
    source="wikipedia",
    language="en",
    site="en.wikipedia.org",
    page_title="Page Title",
    url="https://example.com",
    page_id=1,
    revision_id=1,
    revision_timestamp="2026-07-15T00:00:00Z",
    document_content_hash="doc-hash-1",
    section_id="sec-1",
    section_index=0,
    section_path_raw='["Intro"]',
    section_text_raw="Section text.",
    section_content_hash="sec-hash-1",
    polygon_name="Poly Name",
    osm_primary_tag="primary",
    osm_tags_raw='{"name": "Poly Name"}',
    region="reg-1",
    lat=12.34,
    lon=56.78,
) -> dict:
    return {
        "polygon_id": polygon_id,
        "wikidata": wikidata,
        "document_id": document_id,
        "article_id": article_id,
        "source": source,
        "language": language,
        "site": site,
        "page_title": page_title,
        "url": url,
        "page_id": page_id,
        "revision_id": revision_id,
        "revision_timestamp": revision_timestamp,
        "document_content_hash": document_content_hash,
        "section_id": section_id,
        "section_index": section_index,
        "section_path_raw": section_path_raw,
        "section_text_raw": section_text_raw,
        "section_content_hash": section_content_hash,
        "polygon_name": polygon_name,
        "osm_primary_tag": osm_primary_tag,
        "osm_tags_raw": osm_tags_raw,
        "region": region,
        "lat": lat,
        "lon": lon,
    }


def joined_rows_to_table(rows: list[dict]) -> pa.Table:
    from osm_polygon_sentence_relevance.schemas import JOINED_SECTIONS_SCHEMA

    data = {}
    for field in JOINED_SECTIONS_SCHEMA:
        col_values = [r[field.name] for r in rows]
        data[field.name] = pa.array(col_values, type=field.type)
    return pa.table(data, schema=JOINED_SECTIONS_SCHEMA)
