"""Tests for the deterministic ``osm_tags`` map → list-of-struct conversion.

The canonical Viewer-compatible output schema encodes ``osm_tags`` as
``list<struct<key:string, value:string>>`` because the Hugging Face
``datasets`` library cannot ingest ``map<string, string>``.  This
module exercises the conversion contract:

* empty map → empty list;
* one entry → one-element list;
* multiple entries → sorted-by-key list with no value loss;
* Unicode keys and values preserved;
* ordering deterministic for identical input;
* non-string keys/values rejected;
* the conversion never silently drops or merges entries.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.contracts.schemas.pipeline import (
    OUTPUT_SENTENCE_SCHEMA,
)


def _make_map(pairs: list[tuple[str, str]]) -> pa.Array:
    if not pairs:
        keys = pa.array([], type=pa.string())
        values = pa.array([], type=pa.string())
        return pa.MapArray.from_arrays([0], keys, values)
    keys = pa.array([k for k, _ in pairs], type=pa.string())
    values = pa.array([v for _, v in pairs], type=pa.string())
    return pa.MapArray.from_arrays(
        pa.array([0, len(pairs)], type=pa.int32()), keys, values
    )


class TestOsmTagsSchemaIsViewerCompatible:
    def test_schema_has_no_map_types_anywhere(self) -> None:
        def _walk(field_type: pa.DataType) -> None:
            assert not pa.types.is_map(field_type), (
                f"Output schema contains map type {field_type}; "
                "datasets cannot ingest this"
            )
            if pa.types.is_list(field_type):
                _walk(field_type.value_type)
            elif pa.types.is_struct(field_type):
                for child in field_type:
                    _walk(child.type)
            elif pa.types.is_map(field_type):
                _walk(field_type.key_type)
                _walk(field_type.item_type)

        for field in OUTPUT_SENTENCE_SCHEMA:
            _walk(field.type)

    def test_osm_tags_is_list_of_key_value_struct(self) -> None:
        field = OUTPUT_SENTENCE_SCHEMA.field("osm_tags")
        assert pa.types.is_list(field.type)
        inner = field.type.value_type
        assert pa.types.is_struct(inner)
        names = {child.name for child in inner}
        assert names == {"key", "value"}


class TestMapToListOfStructConversion:
    def test_empty_map_becomes_empty_list(self) -> None:
        from osm_polygon_sentence_relevance.sentences.finalization import (
            convert_osm_tags_to_list_of_struct,
        )

        result = convert_osm_tags_to_list_of_struct({})
        assert result == []

    def test_single_entry_becomes_single_element_list(self) -> None:
        from osm_polygon_sentence_relevance.sentences.finalization import (
            convert_osm_tags_to_list_of_struct,
        )

        result = convert_osm_tags_to_list_of_struct({"highway": "primary"})
        assert result == [{"key": "highway", "value": "primary"}]

    def test_multiple_entries_sorted_by_key(self) -> None:
        from osm_polygon_sentence_relevance.sentences.finalization import (
            convert_osm_tags_to_list_of_struct,
        )

        result = convert_osm_tags_to_list_of_struct(
            {"name": "Foo", "highway": "primary", "surface": "asphalt"}
        )
        assert result == [
            {"key": "highway", "value": "primary"},
            {"key": "name", "value": "Foo"},
            {"key": "surface", "value": "asphalt"},
        ]

    def test_unicode_keys_and_values_preserved(self) -> None:
        from osm_polygon_sentence_relevance.sentences.finalization import (
            convert_osm_tags_to_list_of_struct,
        )

        result = convert_osm_tags_to_list_of_struct(
            {"name:zh": "台北車站", "name:ar": "محطة"}
        )
        keys = [item["key"] for item in result]
        assert "name:ar" in keys
        assert "name:zh" in keys
        values = {item["key"]: item["value"] for item in result}
        assert values["name:zh"] == "台北車站"
        assert values["name:ar"] == "محطة"

    def test_conversion_is_idempotent_and_deterministic(self) -> None:
        from osm_polygon_sentence_relevance.sentences.finalization import (
            convert_osm_tags_to_list_of_struct,
        )

        source = {"a": "1", "b": "2", "c": "3"}
        first = convert_osm_tags_to_list_of_struct(source)
        second = convert_osm_tags_to_list_of_struct(source)
        assert first == second
        # Deterministic order
        third = convert_osm_tags_to_list_of_struct({"c": "3", "b": "2", "a": "1"})
        assert first == third

    def test_rejects_non_string_key(self) -> None:
        from osm_polygon_sentence_relevance.sentences.finalization import (
            convert_osm_tags_to_list_of_struct,
        )

        with pytest.raises((TypeError, ValueError)):
            convert_osm_tags_to_list_of_struct({1: "x"})  # type: ignore[arg-type]

    def test_rejects_non_string_value(self) -> None:
        from osm_polygon_sentence_relevance.sentences.finalization import (
            convert_osm_tags_to_list_of_struct,
        )

        with pytest.raises((TypeError, ValueError)):
            convert_osm_tags_to_list_of_struct({"k": 1})  # type: ignore[arg-type]

    def test_accepts_pa_map_array(self) -> None:
        from osm_polygon_sentence_relevance.sentences.finalization import (
            convert_osm_tags_to_list_of_struct,
        )

        arr = _make_map([("highway", "primary"), ("name", "Foo")])
        result = convert_osm_tags_to_list_of_struct(arr)
        assert result == [
            {"key": "highway", "value": "primary"},
            {"key": "name", "value": "Foo"},
        ]

    def test_empty_pa_map_array_becomes_empty_list(self) -> None:
        from osm_polygon_sentence_relevance.sentences.finalization import (
            convert_osm_tags_to_list_of_struct,
        )

        arr = _make_map([])
        result = convert_osm_tags_to_list_of_struct(arr)
        assert result == []

    def test_list_of_tuples_input(self) -> None:
        from osm_polygon_sentence_relevance.sentences.finalization import (
            convert_osm_tags_to_list_of_struct,
        )

        result = convert_osm_tags_to_list_of_struct(
            [("name", "Foo"), ("highway", "primary")]
        )
        assert result == [
            {"key": "highway", "value": "primary"},
            {"key": "name", "value": "Foo"},
        ]

    def test_empty_list_input(self) -> None:
        from osm_polygon_sentence_relevance.sentences.finalization import (
            convert_osm_tags_to_list_of_struct,
        )

        assert convert_osm_tags_to_list_of_struct([]) == []

    def test_list_with_malformed_entries_rejected(self) -> None:
        from osm_polygon_sentence_relevance.sentences.finalization import (
            convert_osm_tags_to_list_of_struct,
        )

        with pytest.raises((TypeError, ValueError)):
            convert_osm_tags_to_list_of_struct([("k", "v"), "not-a-tuple"])

    def test_unsupported_input_type_rejected(self) -> None:
        from osm_polygon_sentence_relevance.sentences.finalization import (
            convert_osm_tags_to_list_of_struct,
        )

        with pytest.raises((TypeError, ValueError)):
            convert_osm_tags_to_list_of_struct(42)  # type: ignore[arg-type]


class TestFinalizationConvertsOsmTags:
    """The finalization step must emit the Viewer-compatible list-of-struct form."""

    def _segmented_table(self) -> pa.Table:
        # Build a single-row segmented table using the legacy map type
        # (the intermediate/segmented schema keeps the Arrow map form).
        from osm_polygon_sentence_relevance.contracts.schemas import (
            SEGMENTED_SENTENCES_SCHEMA,
        )

        osm_tags_map = pa.array(
            [{"highway": "primary", "name": "Foo"}],
            type=pa.map_(pa.string(), pa.string()),
        )
        row = {
            "polygon_id": ["a-latest:way:1"],
            "wikidata": ["Q1"],
            "document_id": ["1"],
            "article_id": [None],
            "source": ["wikipedia"],
            "language": ["en"],
            "site": ["en.wikipedia.org"],
            "page_title": ["Foo"],
            "url": ["https://en.wikipedia.org/wiki/Foo"],
            "page_id": [1],
            "revision_id": [1],
            "revision_timestamp": ["2020-01-01T00:00:00Z"],
            "document_content_hash": ["0" * 64],
            "section_id": ["0"],
            "section_index": [0],
            "section_path": [["Foo"]],
            "sentence_index": [0],
            "sentence_text_raw": ["Foo."],
            "sentence_text_normalized": ["Foo."],
            "section_content_hash": ["1" * 64],
            "polygon_name": [None],
            "osm_primary_tag": [None],
            "osm_tags": osm_tags_map,
            "region": ["a"],
            "lat": [0.0],
            "lon": [0.0],
        }
        return pa.Table.from_pydict(
            {name: row[name] for name in SEGMENTED_SENTENCES_SCHEMA.names},
            schema=SEGMENTED_SENTENCES_SCHEMA,
        )

    def test_finalization_with_empty_map_works(self) -> None:
        """A polygon with no OSM tags must finalise to an empty list, not None."""
        from osm_polygon_sentence_relevance.contracts.schemas import (
            SEGMENTED_SENTENCES_SCHEMA,
        )
        from osm_polygon_sentence_relevance.sentences.finalization import (
            finalize_sentence_dataset,
        )

        osm_tags_empty = pa.array([{}], type=pa.map_(pa.string(), pa.string()))
        base = {
            "polygon_id": ["b-latest:way:1"],
            "wikidata": ["Q2"],
            "document_id": ["1"],
            "article_id": [None],
            "source": ["wikipedia"],
            "language": ["en"],
            "site": ["en.wikipedia.org"],
            "page_title": ["Bar"],
            "url": ["https://en.wikipedia.org/wiki/Bar"],
            "page_id": [1],
            "revision_id": [1],
            "revision_timestamp": ["2020-01-01T00:00:00Z"],
            "document_content_hash": ["0" * 64],
            "section_id": ["0"],
            "section_index": [0],
            "section_path": [["Bar"]],
            "sentence_index": [0],
            "sentence_text_raw": ["Bar."],
            "sentence_text_normalized": ["Bar."],
            "section_content_hash": ["1" * 64],
            "polygon_name": [None],
            "osm_primary_tag": [None],
            "osm_tags": osm_tags_empty,
            "region": ["b"],
            "lat": [0.0],
            "lon": [0.0],
        }
        table = pa.Table.from_pydict(base, schema=SEGMENTED_SENTENCES_SCHEMA)
        result = finalize_sentence_dataset(
            table, input_dataset_revision="r", pipeline_version="v"
        )
        col = result.table.column("osm_tags").to_pylist()
        assert col == [[]]

    def test_output_osm_tags_is_list_of_struct(self) -> None:
        from osm_polygon_sentence_relevance.sentences.finalization import (
            finalize_sentence_dataset,
        )

        table = self._segmented_table()
        result = finalize_sentence_dataset(
            table,
            input_dataset_revision="r",
            pipeline_version="v",
        )
        col = result.table.column("osm_tags").to_pylist()
        assert col == [
            [
                {"key": "highway", "value": "primary"},
                {"key": "name", "value": "Foo"},
            ]
        ]
