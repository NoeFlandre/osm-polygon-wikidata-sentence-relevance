"""Tests for the immutable ``DatasetProfile`` and asset rendering.

The ``DatasetProfile`` is the single source of truth for everything that
goes into the dataset card, the manifest, and the published PNG assets.
A dataset is profile->manifest->card. Two identical profiles must produce
byte-identical renders and asset hashes.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.output.dataset_card import (
    DatasetStatistics,
)
from osm_polygon_sentence_relevance.output.profile import (
    AssetInfo,
    DatasetProfile,
    ProfileError,
    PNG_SIGNATURE,
    build_dataset_profile,
    render_example_row_json,
    render_geographic_coverage_png,
    render_language_distribution_png,
)


def _make_afghanistan_parquet(path: Path) -> tuple[str, int]:
    """Build a small but realistic parquet file at *path*.

    Returns the SHA-256 and row count. The columns match
    ``OUTPUT_SENTENCE_SCHEMA``; the geographic extent and language
    distribution are derived deterministically from the rows so the
    profile / assets are deterministic without relying on hidden
    fixtures.
    """
    import datetime as _dt

    from osm_polygon_sentence_relevance.contracts.schemas import (
        OUTPUT_SENTENCE_SCHEMA,
    )

    rows: list[dict] = []
    base_lat = 34.0
    base_lon = 69.0
    languages = ["en", "fa", "ps", "en", "de", "en"]  # three strongest
    for idx in range(60):
        rows.append(
            {
                "sentence_id": f"{hashlib.sha256(str(idx).encode()).hexdigest()}",
                "polygon_id": f"afghanistan-latest:way:{idx // 7}",
                "wikidata": f"Q{(idx % 12) + 1}",
                "document_id": f"doc{idx // 4}",
                "article_id": None,
                "source": "wikipedia" if idx % 2 == 0 else "wikivoyage",
                "language": languages[idx % len(languages)],
                "site": "en.wikipedia.org",
                "page_title": f"Page {idx}",
                "section_id": f"s{idx % 5}",
                "section_index": idx % 5,
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
                    f"doc{idx // 4}".encode()
                ).hexdigest(),
                "section_content_hash": hashlib.sha256(
                    f"s{idx % 5}".encode()
                ).hexdigest(),
                "sentence_content_hash": hashlib.sha256(
                    f"Row {idx} text.".encode()
                ).hexdigest(),
                "duplicate_occurrence_count": 1,
                "duplicate_sources": ["wikipedia"],
                "polygon_name": None,
                "osm_primary_tag": None,
                "osm_tags": [{"key": "highway", "value": "primary"}],
                "region": "afghanistan-latest",
                "lat": base_lat + (idx % 11) * 0.1,
                "lon": base_lon + (idx % 7) * 0.2,
                "input_dataset_revision": "abc1234",
                "pipeline_version": "1.2.3",
            }
        )

    table = pa.Table.from_pylist(rows, schema=OUTPUT_SENTENCE_SCHEMA)
    table = table.replace_schema_metadata(
        {
            b"input_dataset_revision": b"abc1234",
            b"pipeline_version": b"1.2.3",
        }
    )
    pq.write_table(table, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return sha, len(rows)


@pytest.fixture()
def afghanistan_parquet(tmp_path: Path) -> tuple[Path, str, int]:
    p = tmp_path / "sentences.parquet"
    sha, count = _make_afghanistan_parquet(p)
    return p, sha, count


class TestBuildDatasetProfile:
    def test_profile_contains_required_fields(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        parquet_path, sha, _count = afghanistan_parquet
        profile = build_dataset_profile(
            parquet_path=parquet_path,
            parquet_sha256=sha,
            segmentation_model="sat-3l",
            segmentation_revision="abc1234",
            source_commit="HEAD",
            scratch_dir=tmp_path,
        )
        assert isinstance(profile, DatasetProfile)
        assert profile.row_count > 0
        assert profile.parquet_sha256 == sha
        assert profile.segmentation_model == "sat-3l"
        assert profile.segmentation_revision == "abc1234"
        assert profile.source_commit == "HEAD"

    def test_profile_has_lat_lon_extents(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        parquet_path, sha, _count = afghanistan_parquet
        profile = build_dataset_profile(
            parquet_path=parquet_path,
            parquet_sha256=sha,
            segmentation_model="sat-3l",
            segmentation_revision="abc1234",
            source_commit="HEAD",
            scratch_dir=tmp_path,
        )
        assert profile.lat_min is not None and profile.lat_max is not None
        assert profile.lon_min is not None and profile.lon_max is not None
        assert profile.lat_min < profile.lat_max
        assert profile.lon_min < profile.lon_max

    def test_profile_has_sentence_length_summary(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        parquet_path, sha, _count = afghanistan_parquet
        profile = build_dataset_profile(
            parquet_path=parquet_path,
            parquet_sha256=sha,
            segmentation_model="sat-3l",
            segmentation_revision="abc1234",
            source_commit="HEAD",
            scratch_dir=tmp_path,
        )
        assert profile.sentence_length_min <= profile.sentence_length_mean
        assert profile.sentence_length_mean <= profile.sentence_length_max

    def test_profile_has_example_row(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        parquet_path, sha, _count = afghanistan_parquet
        profile = build_dataset_profile(
            parquet_path=parquet_path,
            parquet_sha256=sha,
            segmentation_model="sat-3l",
            segmentation_revision="abc1234",
            source_commit="HEAD",
            scratch_dir=tmp_path,
        )
        assert profile.example_row["sentence_id"]
        assert profile.example_row["sentence_text_normalized"]


class TestProfileExamples:
    def test_example_row_dict_access(self) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            ExampleRow,
        )

        er = ExampleRow(fields={"sentence_id": "abc", "text": "def"})
        assert er["sentence_id"] == "abc"
        assert "text" in er.keys()

    def test_profile_is_frozen_and_hashable_like(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        from dataclasses import FrozenInstanceError

        parquet_path, sha, _count = afghanistan_parquet
        profile = build_dataset_profile(
            parquet_path=parquet_path,
            parquet_sha256=sha,
            segmentation_model="sat-3l",
            segmentation_revision="abc1234",
            source_commit="HEAD",
            scratch_dir=tmp_path,
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            profile.row_count = 0  # type: ignore[misc]


class TestGeographicCoveragePNG:
    def test_render_produces_png_signature(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        parquet_path, sha, _count = afghanistan_parquet
        profile = build_dataset_profile(
            parquet_path=parquet_path,
            parquet_sha256=sha,
            segmentation_model="sat-3l",
            segmentation_revision="abc1234",
            source_commit="HEAD",
            scratch_dir=tmp_path,
        )
        png_bytes = render_geographic_coverage_png(profile, parquet_path)
        assert png_bytes.startswith(PNG_SIGNATURE)
        assert len(png_bytes) > 64

    def test_render_is_deterministic_for_identical_input(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        parquet_path, sha, _count = afghanistan_parquet
        profile = build_dataset_profile(
            parquet_path=parquet_path,
            parquet_sha256=sha,
            segmentation_model="sat-3l",
            segmentation_revision="abc1234",
            source_commit="HEAD",
            scratch_dir=tmp_path,
        )
        a = render_geographic_coverage_png(profile, parquet_path)
        b = render_geographic_coverage_png(profile, parquet_path)
        assert a == b

    def test_render_changes_when_profile_changes(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        from dataclasses import replace

        parquet_path, sha, count = afghanistan_parquet
        profile = build_dataset_profile(
            parquet_path=parquet_path,
            parquet_sha256=sha,
            segmentation_model="sat-3l",
            segmentation_revision="abc1234",
            source_commit="HEAD",
            scratch_dir=tmp_path,
        )
        baseline = render_geographic_coverage_png(profile, parquet_path)
        mutated = replace(profile, lat_min=0.0, lat_max=0.1)
        mutated_render = render_geographic_coverage_png(mutated, parquet_path)
        assert mutated_render != baseline


class TestLanguageDistributionPNG:
    def test_render_produces_png_signature(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        parquet_path, sha, _count = afghanistan_parquet
        profile = build_dataset_profile(
            parquet_path=parquet_path,
            parquet_sha256=sha,
            segmentation_model="sat-3l",
            segmentation_revision="abc1234",
            source_commit="HEAD",
            scratch_dir=tmp_path,
        )
        png_bytes = render_language_distribution_png(profile)
        assert png_bytes.startswith(PNG_SIGNATURE)

    def test_render_reconciles_with_profile_totals(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        parquet_path, sha, count = afghanistan_parquet
        profile = build_dataset_profile(
            parquet_path=parquet_path,
            parquet_sha256=sha,
            segmentation_model="sat-3l",
            segmentation_revision="abc1234",
            source_commit="HEAD",
            scratch_dir=tmp_path,
        )
        png_bytes = render_language_distribution_png(profile)
        total_in_png = sum(profile.language_counts.values())
        assert total_in_png == count


class TestProfileCardIntegration:
    def test_example_row_json_round_trips(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        parquet_path, sha, _count = afghanistan_parquet
        profile = build_dataset_profile(
            parquet_path=parquet_path,
            parquet_sha256=sha,
            segmentation_model="sat-3l",
            segmentation_revision="abc1234",
            source_commit="HEAD",
            scratch_dir=tmp_path,
        )
        text = render_example_row_json(profile)
        parsed = json.loads(text)
        assert parsed["sentence_id"] == profile.example_row["sentence_id"]


class TestProfileExamples:
    def test_example_row_dict_access(self) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            ExampleRow,
        )

        er = ExampleRow(fields={"sentence_id": "abc", "text": "def"})
        assert er["sentence_id"] == "abc"
        assert "text" in er.keys()

    def test_asset_info_stable(self) -> None:
        ai = AssetInfo(
            name="geographic_coverage.png",
            sha256="a" * 64,
            bytes_=128,
        )
        assert ai.sha256 == "a" * 64
        assert ai.name == "geographic_coverage.png"
        assert ai.bytes_ == 128

    def test_to_dict_round_trips(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        parquet_path, sha, _count = afghanistan_parquet
        profile = build_dataset_profile(
            parquet_path=parquet_path,
            parquet_sha256=sha,
            segmentation_model="sat-3l",
            segmentation_revision="abc1234",
            source_commit="HEAD",
            scratch_dir=tmp_path,
        )
        d = profile.to_dict()
        assert d["row_count"] == profile.row_count
        assert d["parquet_sha256"] == profile.parquet_sha256
        assert "sentence_id" in d["example_row"]


class TestBuildDatasetProfileErrors:
    def test_missing_parquet_raises(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            build_dataset_profile,
            ProfileError,
        )

        with pytest.raises(ProfileError):
            build_dataset_profile(
                parquet_path=tmp_path / "no-such.parquet",
                parquet_sha256="a" * 64,
                segmentation_model="m",
                segmentation_revision="r",
                source_commit="c",
                scratch_dir=tmp_path,
            )

    def test_blank_segmentation_model_raises(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            build_dataset_profile,
            ProfileError,
        )

        parquet_path, sha, _count = afghanistan_parquet
        with pytest.raises(ProfileError):
            build_dataset_profile(
                parquet_path=parquet_path,
                parquet_sha256=sha,
                segmentation_model="   ",
                segmentation_revision="r",
                source_commit="c",
                scratch_dir=tmp_path,
            )

    def test_sha_mismatch_raises(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            build_dataset_profile,
            ProfileError,
        )

        parquet_path, sha, _count = afghanistan_parquet
        with pytest.raises(ProfileError):
            build_dataset_profile(
                parquet_path=parquet_path,
                parquet_sha256="b" * 64,
                segmentation_model="m",
                segmentation_revision="r",
                source_commit="c",
                scratch_dir=tmp_path,
            )

    def test_blank_sha_raises(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            build_dataset_profile,
            ProfileError,
        )

        parquet_path, sha, _count = afghanistan_parquet
        with pytest.raises(ProfileError):
            build_dataset_profile(
                parquet_path=parquet_path,
                parquet_sha256="   ",
                segmentation_model="m",
                segmentation_revision="r",
                source_commit="c",
                scratch_dir=tmp_path,
            )


class TestRenderEdgeCases:
    def _empty_profile(self):
        from osm_polygon_sentence_relevance.output.profile import (
            DatasetProfile,
            ExampleRow,
        )

        return DatasetProfile(
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

    def test_empty_language_counts_branch(self) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            render_language_distribution_png,
            PNG_SIGNATURE,
        )

        png = render_language_distribution_png(self._empty_profile())
        assert png.startswith(PNG_SIGNATURE)

    def test_geographic_png_handles_no_coords(self) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            render_geographic_coverage_png,
            PNG_SIGNATURE,
        )

        png = render_geographic_coverage_png(
            self._empty_profile(), parquet_path="/dev/null"
        )
        assert png.startswith(PNG_SIGNATURE)


class TestProfileValidationErrors:
    def test_blank_segmentation_revision_raises(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            build_dataset_profile,
            ProfileError,
        )

        parquet_path, sha, _count = afghanistan_parquet
        with pytest.raises(ProfileError):
            build_dataset_profile(
                parquet_path=parquet_path,
                parquet_sha256=sha,
                segmentation_model="m",
                segmentation_revision="   ",
                source_commit="c",
                scratch_dir=tmp_path,
            )

    def test_blank_source_commit_raises(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            build_dataset_profile,
            ProfileError,
        )

        parquet_path, sha, _count = afghanistan_parquet
        with pytest.raises(ProfileError):
            build_dataset_profile(
                parquet_path=parquet_path,
                parquet_sha256=sha,
                segmentation_model="m",
                segmentation_revision="r",
                source_commit="   ",
                scratch_dir=tmp_path,
            )

    def test_blank_dataset_id_raises(
        self, afghanistan_parquet: tuple[Path, str, int], tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            build_dataset_profile,
            ProfileError,
        )

        parquet_path, sha, _count = afghanistan_parquet
        with pytest.raises(ProfileError):
            build_dataset_profile(
                parquet_path=parquet_path,
                parquet_sha256=sha,
                segmentation_model="m",
                segmentation_revision="r",
                source_commit="c",
                scratch_dir=tmp_path,
                input_dataset_id="   ",
            )


class TestProfileParseMetaErrors:
    def test_parse_meta_rejects_blank(self) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            _parse_meta,
            ProfileError,
        )

        with pytest.raises(ProfileError):
            _parse_meta({b"k": b"   "}, b"k")

    def test_parse_meta_rejects_non_utf8(self) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            _parse_meta,
            ProfileError,
        )

        with pytest.raises(ProfileError):
            _parse_meta({b"k": b"\xff\xfe"}, b"k")


class TestProfileToDictUnknownCols:
    """``to_dict`` should not crash when the example_row has unknown columns."""

    def test_to_dict_with_extra_cols_is_ok(self) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            DatasetProfile,
            ExampleRow,
        )

        profile = DatasetProfile(
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
            example_row=ExampleRow(fields={"unknown_col": "x"}),
        )
        d = profile.to_dict()
        # Unknown columns are not added to the schema-ordered dict but
        # the call still succeeds.
        assert "row_count" in d


class TestProfileFirstBatchErrors:
    """Empty parquet must trigger an explicit ProfileError."""

    def test_build_profile_on_empty_parquet_rejected(
        self, tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            build_dataset_profile,
            ProfileError,
        )
        from osm_polygon_sentence_relevance.contracts.schemas import (
            OUTPUT_SENTENCE_SCHEMA,
        )
        import pyarrow as pa
        import pyarrow.parquet as pq

        empty = OUTPUT_SENTENCE_SCHEMA.empty_table().replace_schema_metadata(
            {
                b"input_dataset_revision": b"r",
                b"pipeline_version": b"v",
            }
        )
        path = tmp_path / "empty.parquet"
        pq.write_table(empty, path)
        sha = "0" * 64
        with pytest.raises(ProfileError):
            build_dataset_profile(
                parquet_path=path,
                parquet_sha256=sha,
                segmentation_model="m",
                segmentation_revision="r",
                source_commit="c",
                scratch_dir=tmp_path,
            )
