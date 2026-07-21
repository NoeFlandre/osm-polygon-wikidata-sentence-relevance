"""Tests for the profile-based dataset-card renderer and the
schema-compatibility check.

The new renderer is profile-driven and embeds the two PNG assets, the
example row, and the explicit schema field documentation.  Two
profiles built from identical inputs must produce byte-identical
renders.  The schema-compatibility check must reject any
``map<...>`` field type anywhere in ``OUTPUT_SENTENCE_SCHEMA``.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.contracts.schemas import (
    OUTPUT_SENTENCE_SCHEMA,
    SEGMENTED_SENTENCES_SCHEMA,
)
from osm_polygon_sentence_relevance.output.dataset_card import (
    render_dataset_card_from_profile,
    schema_field_documentation,
    schema_has_map_types,
)
from osm_polygon_sentence_relevance.output.profile import (
    AssetInfo,
    DatasetProfile,
    ExampleRow,
    build_dataset_profile,
)


def _build_minimal_profile(tmp_path: Path) -> DatasetProfile:
    """Build a small parquet + profile for renderer tests."""
    import hashlib
    import datetime as _dt

    rows = []
    for idx in range(6):
        rows.append(
            {
                "sentence_id": hashlib.sha256(
                    str(idx).encode()
                ).hexdigest(),
                "polygon_id": f"a:way:{idx // 3}",
                "wikidata": f"Q{idx + 1}",
                "document_id": f"doc{idx}",
                "article_id": None,
                "source": "wikipedia" if idx % 2 == 0 else "wikivoyage",
                "language": "en",
                "site": "en.wikipedia.org",
                "page_title": f"Page {idx}",
                "section_id": "0",
                "section_index": 0,
                "section_path": ["Lead"],
                "sentence_index": idx,
                "sentence_text_raw": f"Row {idx} text.",
                "sentence_text_normalized": f"Row {idx} text.",
                "previous_sentence": None,
                "next_sentence": None,
                "url": f"https://en.wikipedia.org/wiki/Page_{idx}",
                "page_id": idx + 1,
                "revision_id": idx + 1,
                "revision_timestamp": _dt.datetime(
                    2024, 1, 1, 0, 0, 0, tzinfo=_dt.timezone.utc
                ).isoformat(),
                "document_content_hash": hashlib.sha256(
                    f"doc{idx}".encode()
                ).hexdigest(),
                "section_content_hash": hashlib.sha256(
                    b"s0"
                ).hexdigest(),
                "sentence_content_hash": hashlib.sha256(
                    f"Row {idx} text.".encode()
                ).hexdigest(),
                "duplicate_occurrence_count": 1,
                "duplicate_sources": ["wikipedia"],
                "polygon_name": None,
                "osm_primary_tag": None,
                "osm_tags": [{"key": "highway", "value": "primary"}],
                "region": "a",
                "lat": 34.0 + idx * 0.1,
                "lon": 69.0 + idx * 0.1,
                "input_dataset_revision": "rev",
                "pipeline_version": "1.0.0",
            }
        )
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SENTENCE_SCHEMA)
    table = table.replace_schema_metadata(
        {
            b"input_dataset_revision": b"rev",
            b"pipeline_version": b"1.0.0",
        }
    )
    parquet_path = tmp_path / "sentences.parquet"
    import pyarrow.parquet as pq

    pq.write_table(table, parquet_path)
    sha = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    return build_dataset_profile(
        parquet_path=parquet_path,
        parquet_sha256=sha,
        segmentation_model="sat-3l",
        segmentation_revision="abc1234",
        source_commit="HEAD",
        scratch_dir=tmp_path / "scratch",
    )


def _with_assets(profile: DatasetProfile) -> DatasetProfile:
    """Attach deterministic asset infos to the profile."""
    from dataclasses import replace

    return replace(
        profile,
        assets={
            "geographic_coverage.png": AssetInfo(
                name="geographic_coverage.png",
                sha256="a" * 64,
                bytes_=512,
            ),
            "language_distribution.png": AssetInfo(
                name="language_distribution.png",
                sha256="b" * 64,
                bytes_=256,
            ),
        },
    )


class TestSchemaHasMapTypes:
    def test_output_schema_has_no_map_types(self) -> None:
        assert schema_has_map_types(OUTPUT_SENTENCE_SCHEMA) is False

    def test_segmented_schema_with_map_is_detected(self) -> None:
        assert schema_has_map_types(SEGMENTED_SENTENCES_SCHEMA) is True

    def test_handcrafted_map_field_is_detected(self) -> None:
        bogus = pa.schema(
            [pa.field("x", pa.map_(pa.string(), pa.string()))]
        )
        assert schema_has_map_types(bogus) is True

    def test_list_of_map_is_detected(self) -> None:
        bogus = pa.schema(
            [pa.field("x", pa.list_(pa.map_(pa.string(), pa.string())))]
        )
        assert schema_has_map_types(bogus) is True

    def test_struct_of_map_is_detected(self) -> None:
        bogus = pa.schema(
            [
                pa.field(
                    "x",
                    pa.struct([pa.field("y", pa.map_(pa.string(), pa.string()))]),
                )
            ]
        )
        assert schema_has_map_types(bogus) is True

    def test_list_of_struct_of_map_is_detected(self) -> None:
        bogus = pa.schema(
            [
                pa.field(
                    "x",
                    pa.list_(
                        pa.struct(
                            [pa.field("y", pa.map_(pa.string(), pa.string()))]
                        )
                    ),
                )
            ]
        )
        assert schema_has_map_types(bogus) is True

    def test_preview_section_empty_region(self) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            DatasetProfile,
            ExampleRow,
        )
        from osm_polygon_sentence_relevance.output.dataset_card import (
            _profile_preview_section,
        )

        profile = DatasetProfile(
            version=1,
            row_count=0,
            unique_sentence_ids=0,
            unique_polygons=0,
            unique_wikidata_entities=0,
            unique_documents=0,
            source_counts={},
            language_counts={},
            region_counts={},
            rows_with_coordinates=0,
            rows_without_coordinates=0,
            rows_with_polygon_name=0,
            input_dataset_revision="r",
            pipeline_version="v",
            input_dataset_id=None,
            parquet_sha256="a" * 64,
            segmentation_model="m",
            segmentation_revision="r",
            source_commit="c",
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
            sentence_length_min=0,
            sentence_length_mean=0.0,
            sentence_length_max=0,
            example_row=ExampleRow(fields={}),
        )
        assert _profile_preview_section(profile) == ""

    def test_preview_section_with_latest_suffix(self) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            DatasetProfile,
            ExampleRow,
        )
        from osm_polygon_sentence_relevance.output.dataset_card import (
            _profile_preview_section,
        )

        profile = DatasetProfile(
            version=1,
            row_count=3,
            unique_sentence_ids=3,
            unique_polygons=1,
            unique_wikidata_entities=1,
            unique_documents=1,
            source_counts={"wikipedia": 3},
            language_counts={"en": 3},
            region_counts={"latest": 3},
            rows_with_coordinates=3,
            rows_without_coordinates=0,
            rows_with_polygon_name=0,
            input_dataset_revision="r",
            pipeline_version="v",
            input_dataset_id=None,
            parquet_sha256="a" * 64,
            segmentation_model="m",
            segmentation_revision="r",
            source_commit="c",
            lat_min=0.0,
            lat_max=1.0,
            lon_min=0.0,
            lon_max=1.0,
            sentence_length_min=0,
            sentence_length_mean=0.0,
            sentence_length_max=0,
            example_row=ExampleRow(fields={}),
        )
        section = _profile_preview_section(profile)
        assert "## Dataset scope" in section
        # Region key after "-latest" strip and title-case becomes "Latest".
        assert "Latest-only preview" in section


class TestSchemaFieldDocumentation:
    def test_covers_all_output_columns(self) -> None:
        rows = schema_field_documentation()
        names = {row[0] for row in rows}
        expected = set(OUTPUT_SENTENCE_SCHEMA.names)
        assert names == expected

    def test_deterministic(self) -> None:
        a = schema_field_documentation()
        b = schema_field_documentation()
        assert a == b

    def test_osm_tags_is_list_of_struct(self) -> None:
        rows = schema_field_documentation()
        osm_tags = next(row for row in rows if row[0] == "osm_tags")
        assert "struct" in osm_tags[1]
        assert "list" in osm_tags[1]


class TestRenderDatasetCardFromProfile:
    def test_basic_render_succeeds(self, tmp_path: Path) -> None:
        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))
        assert "OSM Polygon Wikidata Sentence Relevance" in card
        assert "Geographic coverage" in card
        assert "Language coverage" in card

    def test_embeds_png_assets(self, tmp_path: Path) -> None:
        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))
        assert "assets/geographic_coverage.png" in card
        assert "assets/language_distribution.png" in card

    def test_embeds_example_row(self, tmp_path: Path) -> None:
        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))
        assert "sentence_id" in card
        assert profile.example_row["polygon_id"] in card

    def test_yaml_lists_splits(self, tmp_path: Path) -> None:
        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))
        assert "dataset_info:" in card
        assert "splits:" in card
        assert "- name: train" in card

    def test_deterministic_for_identical_input(self, tmp_path: Path) -> None:
        profile_a = _build_minimal_profile(tmp_path)
        profile_b = _build_minimal_profile(tmp_path)
        card_a = render_dataset_card_from_profile(_with_assets(profile_a))
        card_b = render_dataset_card_from_profile(_with_assets(profile_b))
        assert card_a == card_b

    def test_mutated_profile_changes_render(self, tmp_path: Path) -> None:
        from dataclasses import replace

        profile = _build_minimal_profile(tmp_path)
        baseline = render_dataset_card_from_profile(_with_assets(profile))
        mutated = replace(profile, row_count=12345)
        assert render_dataset_card_from_profile(_with_assets(mutated)) != baseline

    def test_uses_segmentation_revision(self, tmp_path: Path) -> None:
        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))
        assert profile.segmentation_revision in card
        assert profile.segmentation_model in card

    def test_no_map_type_field_described(self, tmp_path: Path) -> None:
        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))
        # Card's schema table must label osm_tags as list-of-struct, not
        # as map. The explanatory prose can still mention the legacy
        # map<string,string> form (we want the schema table to be correct).
        rows = schema_field_documentation()
        osm_tags = next(row for row in rows if row[0] == "osm_tags")
        assert "map" not in osm_tags[1]
        assert "list" in osm_tags[1]
        # Make sure the rendered card contains the actual list-of-struct
        # label for osm_tags as a Markdown table row.
        assert "| `osm_tags` | `list<struct<`key`: string, `value`: string>>` |" in card

    def test_single_region_preview_section(self, tmp_path: Path) -> None:
        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))
        # Profile fixture uses single "a" region.
        assert "## Dataset scope" in card
        assert "A-only preview" in card
