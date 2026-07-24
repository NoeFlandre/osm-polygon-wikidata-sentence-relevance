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
    import datetime as _dt
    import hashlib

    rows = []
    for idx in range(6):
        rows.append(
            {
                "sentence_id": hashlib.sha256(str(idx).encode()).hexdigest(),
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
                    2024, 1, 1, 0, 0, 0, tzinfo=_dt.UTC
                ).isoformat(),
                "document_content_hash": hashlib.sha256(
                    f"doc{idx}".encode()
                ).hexdigest(),
                "section_content_hash": hashlib.sha256(b"s0").hexdigest(),
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
        bogus = pa.schema([pa.field("x", pa.map_(pa.string(), pa.string()))])
        assert schema_has_map_types(bogus) is True

    def test_list_of_map_is_detected(self) -> None:
        bogus = pa.schema([pa.field("x", pa.list_(pa.map_(pa.string(), pa.string())))])
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
                        pa.struct([pa.field("y", pa.map_(pa.string(), pa.string()))])
                    ),
                )
            ]
        )
        assert schema_has_map_types(bogus) is True

    def test_preview_section_empty_region(self) -> None:
        from osm_polygon_sentence_relevance.output._card.rendering import (
            _profile_preview_section,
        )
        from osm_polygon_sentence_relevance.output.profile import (
            DatasetProfile,
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
        from osm_polygon_sentence_relevance.output._card.rendering import (
            _profile_preview_section,
        )
        from osm_polygon_sentence_relevance.output.profile import (
            DatasetProfile,
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
        assert "Current release: **Latest only**" in section


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

    def test_public_card_is_concise_factual_and_non_redundant(
        self, tmp_path: Path
    ) -> None:
        """The generated public card must read like documentation, not a
        pipeline status report, and must expose derived accounting once."""
        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))

        assert "Current release: **A only**" in card
        assert "canary" not in card.lower()
        assert "validation snapshot" not in card.lower()
        assert "published incrementally" not in card.lower()
        assert "production export pipeline" not in card.lower()
        assert "Input sentence occurrences" in card
        assert "Duplicates removed" in card
        assert card.count("## Dataset summary") == 1
        assert card.count("## Processing method") == 1
        assert "## Wikipedia and Wikivoyage coverage" not in card
        assert "## Provenance and revision tracking" not in card
        assert "## Reproducibility" not in card

    def test_embeds_png_assets(self, tmp_path: Path) -> None:
        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))
        assert "assets/geographic_coverage.png" in card
        assert "assets/language_distribution.png" in card

    def test_embeds_png_assets_with_absolute_url(self, tmp_path: Path) -> None:
        from dataclasses import replace

        profile = _build_minimal_profile(tmp_path)
        # When ``asset_base_url`` is given the markdown must use the
        # absolute URL so renderers that don't rewrite relative paths
        # can still load the PNG.
        profile = replace(profile, row_count=profile.row_count)
        card = render_dataset_card_from_profile(
            _with_assets(profile),
            asset_base_url="https://example.com/assets",
        )
        assert "https://example.com/assets/geographic_coverage.png" in card
        assert "https://example.com/assets/language_distribution.png" in card

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

    def test_yaml_declares_configs_with_sentences_parquet(self, tmp_path: Path) -> None:
        """The YAML front matter must declare the parquet file as
        the default config's data source so the Hugging Face Dataset
        Viewer never interprets ``assets/*.png`` as dataset rows
        (imagefolder inference regression)."""
        import yaml

        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))
        # Parse the YAML front matter. The card always opens with
        # '---' on its own line.
        assert card.startswith("---")
        end = card.index("\n---\n", 3)
        front_matter = card[3:end]
        parsed = yaml.safe_load(front_matter)
        assert parsed is not None
        # Exactly one default config exists.
        assert "configs" in parsed
        assert len(parsed["configs"]) == 1
        assert parsed["configs"][0]["config_name"] == "default"
        # Exactly one train split exists and its path is exactly
        # sentences.parquet.
        data_files = parsed["configs"][0]["data_files"]
        assert len(data_files) == 1
        assert data_files[0]["split"] == "train"
        assert data_files[0]["path"] == "sentences.parquet"

    def test_yaml_assets_paths_never_doubles_as_data_files(
        self, tmp_path: Path
    ) -> None:
        """The README's data_files section must NEVER reference any
        asset/*.png path; the imagefolder builder would otherwise
        classify the PNGs as dataset rows."""
        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))
        # The assets/* links appear only inside markdown image
        # references (lines starting with "![") and the README
        # markdown body; the YAML front matter must not.
        yaml_end = card.index("\n---\n", 3)
        yaml_block = card[:yaml_end]
        assert "assets/" not in yaml_block, (
            "YAML front matter must not reference assets/* paths; "
            "the Viewer would infer imagefolder semantics and treat "
            "the PNGs as dataset rows."
        )

    def test_yaml_split_count_agrees_with_parquet(self, tmp_path: Path) -> None:
        """The ``num_examples`` field of the YAML train split must
        equal the parquet's row count and the profile's row_count."""
        import pyarrow.parquet as pq
        import yaml

        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))
        front_matter = card[3 : card.index("\n---\n", 3)]
        parsed = yaml.safe_load(front_matter)
        # Train split num_examples is declared once in
        # dataset_info.splits.
        split_entry = next(
            s for s in parsed["dataset_info"]["splits"] if s["name"] == "train"
        )
        parquet_rows = pq.read_table(tmp_path / "sentences.parquet").num_rows
        assert split_entry["num_examples"] == parquet_rows
        assert split_entry["num_examples"] == profile.row_count

    def test_yaml_parses_with_strict_yaml_library(self, tmp_path: Path) -> None:
        """Strict YAML parsing must accept the front matter."""
        import yaml

        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))
        front_matter = card[3 : card.index("\n---\n", 3)]
        parsed = yaml.safe_load(front_matter)
        # Required top-level keys.
        for key in ("license", "pretty_name", "configs", "dataset_info"):
            assert key in parsed
        # configs must be a list with exactly one entry.
        assert isinstance(parsed["configs"], list)
        assert len(parsed["configs"]) == 1

    def test_yaml_empty_language_block(self, tmp_path: Path) -> None:
        """A profile with zero languages renders an empty YAML
        language block without crashing."""
        from dataclasses import replace

        import yaml

        profile = _build_minimal_profile(tmp_path)
        # Empty language_counts drives the else branch in
        # ``_profile_yaml``.
        profile = replace(
            profile,
            language_counts={},
        )
        card = render_dataset_card_from_profile(_with_assets(profile))
        front_matter = card[3 : card.index("\n---\n", 3)]
        parsed = yaml.safe_load(front_matter)
        assert parsed["language"] == []

    def test_preview_section_empty_display_name(self, tmp_path: Path) -> None:
        """A region key that strips to an empty string falls back
        to the raw key (defensive fallback)."""
        from dataclasses import replace

        from osm_polygon_sentence_relevance.output._card.rendering import (
            _profile_preview_section,
        )

        profile = _build_minimal_profile(tmp_path)
        # ``-latest`` strips to an empty string and ``.title()``
        # of an empty string is still empty, so the fallback
        # ``display_name = region_key`` branch is hit.
        profile = replace(
            profile,
            region_counts={"-latest": 3},
        )
        section = _profile_preview_section(profile)
        assert "## Dataset scope" in section
        # The fallback uses the raw region_key "-latest".
        assert "-latest" in section

    def test_dataset_statistics_eq_with_non_statistics(self, tmp_path: Path) -> None:
        """DatasetStatistics.__eq__ returns NotImplemented for non-statistics."""
        profile = _build_minimal_profile(tmp_path)
        from osm_polygon_sentence_relevance.output.dataset_card import (
            DatasetStatistics,
        )

        a = DatasetStatistics(
            version=1,
            row_count=1,
            unique_sentence_ids=1,
            unique_polygons=1,
            unique_wikidata_entities=1,
            unique_documents=1,
            source_counts={"wikipedia": 1},
            language_counts={"en": 1},
            region_counts={"a": 1},
            rows_with_coordinates=1,
            rows_without_coordinates=0,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="a" * 64,
            input_dataset_id=None,
        )
        # Comparing against a non-statistics returns NotImplemented,
        # which Python falls back to ``False``.
        assert (a == "not a stats") is False
        assert (a == object()) is False
        # And not compared against the profile either (also not stats).
        assert (a == profile) is False

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
        assert "**High-confidence residual boundary violations:** 0" in card

    def test_processing_method_is_public_and_reproducible(self, tmp_path: Path) -> None:
        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))

        assert "## Processing method" in card
        assert "conservative residual-boundary repair" in card
        assert "abbreviations" in card
        assert "lowercase continuations" in card
        assert "numeric values" in card
        assert "URL query strings" in card
        assert "Publication scans every normalized sentence" in card
        assert (
            "https://github.com/NoeFlandre/"
            "osm-polygon-wikidata-sentence-relevance/tree/HEAD"
        ) in card

    def test_processing_method_explains_the_pipeline_in_order(
        self, tmp_path: Path
    ) -> None:
        profile = _build_minimal_profile(tmp_path)
        card = render_dataset_card_from_profile(_with_assets(profile))

        stages = (
            "1. **Section input.**",
            "2. **Model segmentation.**",
            "3. **Boundary repair.**",
            "4. **Normalization.**",
            "5. **Context and identity.**",
            "6. **Polygon-scoped deduplication.**",
            "7. **Publication audit.**",
        )
        positions = [card.index(stage) for stage in stages]
        assert positions == sorted(positions)
        assert "Wikipedia or Wikivoyage section" in card
        assert "section order is retained" in card
        assert "terminal punctuation stays with the preceding sentence" in card
        assert "The model is not asked to rewrite text" in card
        assert card.count("Unicode NFC") == 1

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
        assert "Current release: **A only**" in card
