"""Tests for Phase 3G/3H joined-sections validation and transformation."""

from __future__ import annotations

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.errors import (
    PreprocessingError,
    SegmentationError,
)
from osm_polygon_sentence_relevance.schemas import (
    JOINED_SECTIONS_SCHEMA,
    SEGMENTED_SENTENCES_SCHEMA,
)
from osm_polygon_sentence_relevance.sentence_table import (
    segment_joined_sections,
    validate_joined_sections_table,
)


class FakeSegmenter:
    """Deterministic segmenter: maps each input text to a fixed segment list."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = 0

    def split_batch(self, texts, languages):
        self.calls += 1
        return [self.mapping[text] for text in texts]


def _row(**overrides):
    """One valid joined-section row as a pyarrow table."""
    values = {
        "polygon_id": ["poly-1"],
        "wikidata": ["Q1"],
        "document_id": ["doc-1"],
        "article_id": ["art-1"],
        "source": ["wikipedia"],
        "language": ["en"],
        "site": ["en.wikipedia.org"],
        "page_title": ["Title"],
        "url": ["https://example.com"],
        "page_id": [10],
        "revision_id": [20],
        "revision_timestamp": ["2024-01-01T00:00:00Z"],
        "document_content_hash": ["h1"],
        "section_id": ["sec-1"],
        "section_index": [0],
        "section_path_raw": ['["Introduction"]'],
        "section_text_raw": ["Some text."],
        "section_content_hash": ["h2"],
        "polygon_name": ["Name"],
        "osm_primary_tag": ["boundary=administrative"],
        "osm_tags_raw": ['{"name": "Name"}'],
        "region": ["region"],
        "lat": [34.5],
        "lon": [69.1],
    }
    values.update(overrides)
    return pa.table(values, schema=JOINED_SECTIONS_SCHEMA)


class TestValidateJoinedSectionsTable:
    def test_exact_schema_succeeds(self):
        assert validate_joined_sections_table(_row()) is None

    def test_reordered_columns_succeed(self):
        table = _row()
        reordered = table.select(list(reversed(table.column_names)))
        assert validate_joined_sections_table(reordered) is None

    def test_extra_column_succeeds(self):
        table = _row()
        extra = table.append_column("bonus", pa.array(["x"]))
        assert validate_joined_sections_table(extra) is None

    def test_missing_field_fails(self):
        table = _row()
        reduced = table.drop_columns(["polygon_id"])
        with pytest.raises(SegmentationError) as exc:
            validate_joined_sections_table(reduced)
        assert "polygon_id" in str(exc.value)

    def test_wrong_type_fails(self):
        table = _row()
        bad = table.set_column(
            table.schema.get_field_index("page_id"),
            "page_id",
            pa.array(["not-int"]),
        )
        with pytest.raises(SegmentationError) as exc:
            validate_joined_sections_table(bad)
        assert "page_id" in str(exc.value)

    def test_null_in_required_field_fails(self):
        table = _row()
        bad = table.set_column(
            table.schema.get_field_index("wikidata"),
            "wikidata",
            pa.array([None], type=pa.string()),
        )
        with pytest.raises(SegmentationError) as exc:
            validate_joined_sections_table(bad)
        assert "wikidata" in str(exc.value)

    def test_null_in_nullable_fields_succeeds(self):
        table = _row(
            polygon_name=[None],
            lat=[None],
            lon=[None],
        )
        assert validate_joined_sections_table(table) is None

    def test_input_table_unchanged(self):
        table = _row()
        before = table.column_names
        validate_joined_sections_table(table)
        assert table.column_names == before
        assert table.num_rows == 1


def _one_row(section_text_raw="Some text.", **overrides):
    return _row(section_text_raw=[section_text_raw], **overrides)


class TestSegmentJoinedSections:
    def test_one_section_multiple_sentences(self):
        seg = FakeSegmenter({"Some text.": ["First.", "Second."]})
        result = segment_joined_sections(_one_row(), seg)
        rows = result.table.to_pylist()
        assert [r["sentence_text_raw"] for r in rows] == ["First.", "Second."]
        assert [r["sentence_index"] for r in rows] == [0, 1]

    def test_both_sources(self):
        wp = _row(
            section_text_raw=["A."],
            document_id=["doc-wp"],
            source=["wikipedia"],
        )
        wv = _row(
            section_text_raw=["B."],
            document_id=["doc-wv"],
            source=["wikivoyage"],
        )
        table = pa.concat_tables([wp, wv])
        seg = FakeSegmenter({"A.": ["A."], "B.": ["B."]})
        result = segment_joined_sections(table, seg)
        sources = {r["source"] for r in result.table.to_pylist()}
        assert sources == {"wikipedia", "wikivoyage"}
        assert result.report.wikipedia_sentence_occurrence_count == 1
        assert result.report.wikivoyage_sentence_occurrence_count == 1

    def test_deterministic_from_shuffled_input(self):
        rows = [
            _row(
                polygon_id=[f"p{i}"],
                document_id=[f"d{i}"],
                section_text_raw=[f"T{i}."],
                section_index=[i],
            )
            for i in range(3)
        ]
        shuffled = pa.concat_tables([rows[2], rows[0], rows[1]])
        seg = FakeSegmenter({f"T{i}.": [f"T{i}."] for i in range(3)})
        result = segment_joined_sections(shuffled, seg)
        texts = [r["sentence_text_raw"] for r in result.table.to_pylist()]
        assert texts == ["T0.", "T1.", "T2."]

    def test_exact_output_schema_and_column_order(self):
        seg = FakeSegmenter({"Some text.": ["One."]})
        result = segment_joined_sections(_one_row(), seg)
        assert result.table.schema.equals(SEGMENTED_SENTENCES_SCHEMA)
        assert result.table.column_names == list(SEGMENTED_SENTENCES_SCHEMA.names)

    def test_provenance_copied(self):
        seg = FakeSegmenter({"Some text.": ["One."]})
        result = segment_joined_sections(
            _one_row(
                polygon_id=["PZ"],
                wikidata=["Q9"],
                document_id=["D1"],
                article_id=["A1"],
                source=["wikipedia"],
                language=["fr"],
                site=["fr.wikipedia.org"],
                page_title=["Titre"],
                url=["https://fr.example/x"],
                page_id=[42],
                revision_id=[43],
                revision_timestamp=["2024-01-01T00:00:00Z"],
                document_content_hash=["dh"],
                section_id=["s1"],
                section_index=[3],
                section_content_hash=["sh"],
                polygon_name=["Name"],
                osm_primary_tag=["boundary=admin"],
                osm_tags_raw=['{"k": "v"}'],
                region=["r"],
                lat=[1.5],
                lon=[2.5],
            ),
            seg,
        )
        row = result.table.to_pylist()[0]
        assert row["polygon_id"] == "PZ"
        assert row["wikidata"] == "Q9"
        assert row["document_id"] == "D1"
        assert row["article_id"] == "A1"
        assert row["source"] == "wikipedia"
        assert row["language"] == "fr"
        assert row["site"] == "fr.wikipedia.org"
        assert row["page_title"] == "Titre"
        assert row["url"] == "https://fr.example/x"
        assert row["page_id"] == 42
        assert row["revision_id"] == 43
        assert row["revision_timestamp"] == "2024-01-01T00:00:00Z"
        assert row["document_content_hash"] == "dh"
        assert row["section_id"] == "s1"
        assert row["section_index"] == 3
        assert row["section_content_hash"] == "sh"
        assert row["polygon_name"] == "Name"
        assert row["osm_primary_tag"] == "boundary=admin"
        assert row["region"] == "r"
        assert row["lat"] == 1.5
        assert row["lon"] == 2.5

    def test_parsed_section_path_and_osm_tags(self):
        seg = FakeSegmenter({"Some text.": ["One."]})
        result = segment_joined_sections(
            _one_row(
                section_path_raw=['["Introduction", "History"]'],
                osm_tags_raw=['{"name": "X", "a": "b"}'],
            ),
            seg,
        )
        row = result.table.to_pylist()[0]
        assert row["section_path"] == ["Introduction", "History"]
        assert dict(row["osm_tags"]) == {"a": "b", "name": "X"}

    def test_batching_call_count(self):
        rows = [
            _row(
                polygon_id=[f"p{i}"],
                section_text_raw=[f"T{i}."],
                section_index=[i],
            )
            for i in range(5)
        ]
        table = pa.concat_tables(rows)
        seg = FakeSegmenter({f"T{i}.": [f"T{i}."] for i in range(5)})
        segment_joined_sections(table, seg, batch_size=2)
        assert seg.calls == 3

    def test_sentence_indices_reset_per_section(self):
        seg = FakeSegmenter({"A text.": ["S1.", "S2."], "B text.": ["S3."]})
        rows = [
            _row(polygon_id=["pa"], section_text_raw=["A text."], section_index=[0]),
            _row(polygon_id=["pb"], section_text_raw=["B text."], section_index=[1]),
        ]
        table = pa.concat_tables(rows)
        result = segment_joined_sections(table, seg)
        indices = [r["sentence_index"] for r in result.table.to_pylist()]
        assert indices == [0, 1, 0]

    def test_empty_and_marker_only_reflected_in_report(self):
        seg = FakeSegmenter(
            {
                "drop raw": ["  ", "Kept."],
                "drop norm": ["\u200b", "Also."],
            }
        )
        rows = [
            _row(polygon_id=["pa"], section_text_raw=["drop raw"], section_index=[0]),
            _row(polygon_id=["pb"], section_text_raw=["drop norm"], section_index=[1]),
        ]
        table = pa.concat_tables(rows)
        result = segment_joined_sections(table, seg)
        assert result.report.dropped_empty_raw_count == 1
        assert result.report.dropped_empty_normalized_count == 1
        assert result.report.retained_sentence_occurrence_count == 2

    def test_zero_sentence_section(self):
        seg = FakeSegmenter({"Empty.": []})
        result = segment_joined_sections(_one_row(section_text_raw="Empty."), seg)
        assert result.table.num_rows == 0
        assert result.report.retained_sentence_occurrence_count == 0

    def test_repeated_sentences_preserved(self):
        seg = FakeSegmenter({"T.": ["Same.", "Same."]})
        result = segment_joined_sections(_one_row(section_text_raw="T."), seg)
        assert [r["sentence_text_raw"] for r in result.table.to_pylist()] == [
            "Same.",
            "Same.",
        ]

    def test_empty_input(self):
        empty = JOINED_SECTIONS_SCHEMA.empty_table()
        seg = FakeSegmenter({})
        result = segment_joined_sections(empty, seg)
        assert result.table.num_rows == 0
        assert result.table.schema.equals(SEGMENTED_SENTENCES_SCHEMA)
        assert seg.calls == 0
        assert result.report.input_section_occurrence_count == 0

    @pytest.mark.parametrize(
        "bad_size",
        [0, -1, 1.5, "2", True, False],
    )
    def test_invalid_batch_size(self, bad_size):
        seg = FakeSegmenter({})
        with pytest.raises(SegmentationError):
            segment_joined_sections(_one_row(), seg, batch_size=bad_size)

    def test_malformed_section_path(self):
        seg = FakeSegmenter({"Some text.": ["One."]})
        with pytest.raises(PreprocessingError) as exc:
            segment_joined_sections(_one_row(section_path_raw=["not json"]), seg)
        assert seg.calls == 0
        assert "section_path" in str(exc.value)

    def test_malformed_osm_tags(self):
        seg = FakeSegmenter({"Some text.": ["One."]})
        with pytest.raises(PreprocessingError) as exc:
            segment_joined_sections(_one_row(osm_tags_raw=["{not json"]), seg)
        assert seg.calls == 0
        assert "osm_tags" in str(exc.value)

    def test_malformed_section_path_later_batch_zero_calls(self):
        good = _one_row(section_text_raw="A.", polygon_id=["p0"])
        bad = _one_row(
            section_text_raw="B.",
            polygon_id=["p2"],
            section_path_raw=["not json"],
        )
        mid = _one_row(section_text_raw="C.", polygon_id=["p1"])
        table = pa.concat_tables([good, mid, bad])
        seg = FakeSegmenter({})
        with pytest.raises(PreprocessingError):
            segment_joined_sections(table, seg, batch_size=1)
        assert seg.calls == 0

    def test_malformed_osm_tags_later_batch_zero_calls(self):
        good = _one_row(section_text_raw="A.", polygon_id=["p0"])
        bad = _one_row(
            section_text_raw="B.",
            polygon_id=["p2"],
            osm_tags_raw=["{not json"],
        )
        mid = _one_row(section_text_raw="C.", polygon_id=["p1"])
        table = pa.concat_tables([good, mid, bad])
        seg = FakeSegmenter({})
        with pytest.raises(PreprocessingError):
            segment_joined_sections(table, seg, batch_size=1)
        assert seg.calls == 0

    def test_invalid_source_later_batch_zero_calls(self):
        good = _one_row(section_text_raw="A.", polygon_id=["p0"])
        bad = _one_row(
            section_text_raw="B.",
            polygon_id=["p2"],
            source=["web"],
        )
        mid = _one_row(section_text_raw="C.", polygon_id=["p1"])
        table = pa.concat_tables([good, mid, bad])
        seg = FakeSegmenter({})
        with pytest.raises(SegmentationError):
            segment_joined_sections(table, seg, batch_size=1)
        assert seg.calls == 0

    def test_input_table_unchanged_by_transform(self):
        table = _one_row(section_text_raw="A.")
        before_names = table.column_names
        before_rows = table.num_rows
        seg = FakeSegmenter({"A.": ["A."]})
        segment_joined_sections(table, seg)
        assert table.column_names == before_names
        assert table.num_rows == before_rows
