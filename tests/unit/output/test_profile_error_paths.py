"""Coverage-driven tests for ``profile.py`` error paths and helpers.

The corrective release introduces new error-handling code paths
(``_load_afghanistan_outline`` and the figure-rendering fallbacks).
These tests ensure the defensive branches are reachable and the
error messages stay actionable.

The file is intentionally small: it exists to bring the project
back to the 95 % coverage gate the corrective release must hold.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.contracts.schemas import OUTPUT_SENTENCE_SCHEMA
from osm_polygon_sentence_relevance.output import profile as profile_mod
from osm_polygon_sentence_relevance.output.profile import (
    ProfileError,
    _load_afghanistan_outline,
    build_dataset_profile,
    render_geographic_coverage_png,
    render_language_distribution_png,
)


def _write_minimal_parquet(
    path: Path,
    *,
    metadata: dict[bytes, bytes] | None = None,
) -> str:
    rows = [
        {
            "sentence_id": hashlib.sha256(b"s0").hexdigest(),
            "polygon_id": "afghanistan-latest:way:1",
            "wikidata": "Q1",
            "document_id": "doc1",
            "article_id": None,
            "source": "wikipedia",
            "language": "en",
            "site": "en.wikipedia.org",
            "page_title": "P",
            "section_id": "0",
            "section_index": 0,
            "section_path": ["Lead"],
            "sentence_index": 0,
            "sentence_text_raw": "x",
            "sentence_text_normalized": "x",
            "previous_sentence": None,
            "next_sentence": None,
            "url": "https://en.wikipedia.org/wiki/P",
            "page_id": 1,
            "revision_id": 1,
            "revision_timestamp": _dt.datetime(
                2024, 1, 1, 0, 0, 0, tzinfo=_dt.UTC
            ).isoformat(),
            "document_content_hash": "a" * 64,
            "section_content_hash": "b" * 64,
            "sentence_content_hash": "c" * 64,
            "duplicate_occurrence_count": 1,
            "duplicate_sources": ["wikipedia"],
            "polygon_name": None,
            "osm_primary_tag": None,
            "osm_tags": [{"key": "highway", "value": "primary"}],
            "region": "afghanistan-latest",
            "lat": 33.5,
            "lon": 65.0,
            "input_dataset_revision": "rev",
            "pipeline_version": "1.0.0",
        }
    ]
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SENTENCE_SCHEMA)
    md = dict(metadata) if metadata else {}
    md.setdefault(b"input_dataset_revision", b"rev")
    md.setdefault(b"pipeline_version", b"1.0.0")
    md.setdefault(b"input_dataset_id", b"afghanistan-test/source")
    table = table.replace_schema_metadata(md)
    pq.write_table(table, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_test_outline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    body: str,
    target: Path,
) -> None:
    """Write *body* to *target* and pin the SHA constant to match.

    Lets the post-SHA error branches of
    :func:`_load_afghanistan_outline` execute under test without
    needing to brute-force an arbitrary file SHA.
    """
    target.write_text(body)
    actual = hashlib.sha256(target.read_bytes()).hexdigest().lower()
    monkeypatch.setattr(profile_mod, "_NATURAL_EARTH_PATH", target)
    monkeypatch.setattr(
        profile_mod, "_NATURAL_EARTH_EXPECTED_SHA256", actual
    )


class TestLoadAfghanistanOutlineErrors:
    def test_wrong_sha_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Point the loader at a temporary path with a different SHA.
        bogus = tmp_path / "bogus.geojson"
        bogus.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
        monkeypatch.setattr(profile_mod, "_NATURAL_EARTH_PATH", bogus)
        with pytest.raises(ProfileError, match="does not match"):
            _load_afghanistan_outline()

    def test_missing_file_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            profile_mod, "_NATURAL_EARTH_PATH", tmp_path / "absent.geojson"
        )
        with pytest.raises(ProfileError, match="is missing"):
            _load_afghanistan_outline()

    def test_malformed_geojson_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "bad.geojson"
        _install_test_outline(monkeypatch, body="not-json", target=target)
        with pytest.raises(ProfileError, match="malformed"):
            _load_afghanistan_outline()

    def test_no_features_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "no-feat.geojson"
        _install_test_outline(
            monkeypatch,
            body=json.dumps(
                {"type": "FeatureCollection", "features": []}
            ),
            target=target,
        )
        with pytest.raises(ProfileError, match="no features"):
            _load_afghanistan_outline()

    def test_non_polygon_geometry_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "non-poly.geojson"
        _install_test_outline(
            monkeypatch,
            body=json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [0, 0],
                            },
                            "properties": {},
                        }
                    ],
                }
            ),
            target=target,
        )
        with pytest.raises(ProfileError, match="not a Polygon"):
            _load_afghanistan_outline()

    def test_empty_polygon_rings_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "empty-rings.geojson"
        _install_test_outline(
            monkeypatch,
            body=json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": []},
                            "properties": {},
                        }
                    ],
                }
            ),
            target=target,
        )
        with pytest.raises(ProfileError, match="no coordinates"):
            _load_afghanistan_outline()


class TestBuildProfileErrorBranches:
    def test_missing_required_metadata_keys(
        self, tmp_path: Path
    ) -> None:
        """When required provenance keys are absent, ProfileError must
        fire before any row is read."""
        path = tmp_path / "p.parquet"
        sha = _write_minimal_parquet(
            path,
            metadata={
                b"input_dataset_revision": b"rev",
                b"pipeline_version": b"1.0.0",
            },
        )
        # Remove the pipeline_version key entirely.
        table = pq.read_table(path)
        metadata = dict(table.schema.metadata or {})
        metadata.pop(b"pipeline_version", None)
        new_table = table.replace_schema_metadata(metadata)
        new_path = tmp_path / "no-version.parquet"
        pq.write_table(new_table, new_path)
        sha = hashlib.sha256(new_path.read_bytes()).hexdigest()
        with pytest.raises(ProfileError, match="missing required provenance keys"):
            build_dataset_profile(
                parquet_path=new_path,
                parquet_sha256=sha,
                segmentation_model="m",
                segmentation_revision="r",
                source_commit="c",
                scratch_dir=tmp_path,
            )

    def test_parse_meta_missing_returns_none(self) -> None:
        from osm_polygon_sentence_relevance.output.profile import _parse_meta

        assert _parse_meta(None, b"input_dataset_revision") is None
        assert _parse_meta({}, b"input_dataset_revision") is None

    def test_parse_meta_invalid_utf8_raises(self) -> None:
        from osm_polygon_sentence_relevance.output.profile import _parse_meta

        with pytest.raises(ProfileError, match="not valid UTF-8"):
            _parse_meta({b"k": b"\xff\xfe"}, b"k")

    def test_blank_pipeline_version_metadata(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "p.parquet"
        sha = _write_minimal_parquet(
            path,
            metadata={
                b"input_dataset_revision": b"rev",
                b"pipeline_version": b"   ",
                b"input_dataset_id": b"afghanistan-test/source",
            },
        )
        with pytest.raises(ProfileError, match="cannot be blank"):
            build_dataset_profile(
                parquet_path=path,
                parquet_sha256=sha,
                segmentation_model="m",
                segmentation_revision="r",
                source_commit="c",
                scratch_dir=tmp_path,
            )

    def test_explicit_dataset_id_mismatch(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "p.parquet"
        sha = _write_minimal_parquet(
            path,
            metadata={
                b"input_dataset_id": b"someone/else",
            },
        )
        with pytest.raises(ProfileError, match="does not match Parquet metadata"):
            build_dataset_profile(
                parquet_path=path,
                parquet_sha256=sha,
                segmentation_model="m",
                segmentation_revision="r",
                source_commit="c",
                scratch_dir=tmp_path,
                input_dataset_id="another/different",
            )

    def test_parquet_schema_mismatch(
        self, tmp_path: Path
    ) -> None:
        """A parquet whose schema does not match OUTPUT_SENTENCE_SCHEMA
        must be rejected at profile-construction time."""
        path = tmp_path / "p.parquet"
        bad = pa.Table.from_pylist([{"only": "one"}])
        pq.write_table(bad, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ProfileError, match="schema does not match"):
            build_dataset_profile(
                parquet_path=path,
                parquet_sha256=sha,
                segmentation_model="m",
                segmentation_revision="r",
                source_commit="c",
                scratch_dir=tmp_path,
            )

    def test_parquet_row_revision_mismatch(
        self, tmp_path: Path
    ) -> None:
        """A row whose revision disagrees with the metadata must be
        rejected (the iteration re-checks the schema metadata)."""
        path = tmp_path / "p.parquet"
        sha = _write_minimal_parquet(
            path,
            metadata={
                b"input_dataset_revision": b"rev-A",
            },
        )
        # Force a row whose revision disagrees with the metadata.
        table = pq.read_table(path)
        # Schema metadata is the source of truth. The per-row check
        # uses column 'input_dataset_revision'; we patch a value
        # after reading.
        new_table = table.set_column(
            table.column_names.index("input_dataset_revision"),
            "input_dataset_revision",
            pa.array(["rev-B"], type=pa.string()),
        )
        new_path = tmp_path / "patched.parquet"
        pq.write_table(new_table, new_path)
        sha = hashlib.sha256(new_path.read_bytes()).hexdigest()
        # The validator catches the row revision mismatch in the
        # iteration loop; the message can vary slightly so we just
        # check for ProfileError.
        with pytest.raises(ProfileError):
            build_dataset_profile(
                parquet_path=new_path,
                parquet_sha256=sha,
                segmentation_model="m",
                segmentation_revision="r",
                source_commit="c",
                scratch_dir=tmp_path,
            )

    def test_parquet_row_pipeline_version_mismatch(
        self, tmp_path: Path
    ) -> None:
        """A row whose pipeline_version disagrees with the metadata."""
        path = tmp_path / "p.parquet"
        sha = _write_minimal_parquet(
            path,
            metadata={
                b"pipeline_version": b"v1",
            },
        )
        table = pq.read_table(path)
        new_table = table.set_column(
            table.column_names.index("pipeline_version"),
            "pipeline_version",
            pa.array(["v2"], type=pa.string()),
        )
        new_path = tmp_path / "patched.parquet"
        pq.write_table(new_table, new_path)
        sha = hashlib.sha256(new_path.read_bytes()).hexdigest()
        with pytest.raises(ProfileError):
            build_dataset_profile(
                parquet_path=new_path,
                parquet_sha256=sha,
                segmentation_model="m",
                segmentation_revision="r",
                source_commit="c",
                scratch_dir=tmp_path,
            )

    def test_parquet_with_polygon_name_records_count(
        self, tmp_path: Path
    ) -> None:
        """The profile must record ``rows_with_polygon_name``."""
        path = tmp_path / "p.parquet"
        sha = _write_minimal_parquet(
            path,
            metadata={
                b"input_dataset_revision": b"rev",
                b"pipeline_version": b"1.0.0",
                b"input_dataset_id": b"afghanistan-test/source",
            },
        )
        table = pq.read_table(path)
        # Set polygon_name on the row to verify the count.
        new_table = table.set_column(
            table.column_names.index("polygon_name"),
            "polygon_name",
            pa.array(["Some Polygon"], type=pa.string()),
        )
        new_path = tmp_path / "with-name.parquet"
        pq.write_table(new_table, new_path)
        sha = hashlib.sha256(new_path.read_bytes()).hexdigest()
        profile = build_dataset_profile(
            parquet_path=new_path,
            parquet_sha256=sha,
            segmentation_model="m",
            segmentation_revision="r",
            source_commit="c",
            scratch_dir=tmp_path,
        )
        assert profile.rows_with_polygon_name == 1


class TestRenderGeoErrorBranches:
    def test_render_handles_unreadable_parquet(
        self, tmp_path: Path
    ) -> None:
        """A path that is not a real parquet must not crash the renderer."""
        path = tmp_path / "p.parquet"
        _write_minimal_parquet(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        profile = build_dataset_profile(
            parquet_path=path,
            parquet_sha256=sha,
            segmentation_model="m",
            segmentation_revision="r",
            source_commit="c",
            scratch_dir=tmp_path,
        )
        # Replace the parquet with unreadable bytes.
        bogus = tmp_path / "bogus.parquet"
        bogus.write_bytes(b"not-a-parquet")
        png = render_geographic_coverage_png(profile, bogus)
        # The renderer should still emit a valid PNG (no scatter dots).
        assert png.startswith(b"\x89PNG\r\n\x1a\n")

    def test_render_handles_missing_parquet_path(
        self, tmp_path: Path
    ) -> None:
        """A non-existent parquet path must not crash the renderer."""
        path = tmp_path / "p.parquet"
        _write_minimal_parquet(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        profile = build_dataset_profile(
            parquet_path=path,
            parquet_sha256=sha,
            segmentation_model="m",
            segmentation_revision="r",
            source_commit="c",
            scratch_dir=tmp_path,
        )
        missing = tmp_path / "absent.parquet"
        png = render_geographic_coverage_png(profile, missing)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")


class TestRenderLanguageEmptyBranches:
    def test_empty_language_counts(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            DatasetProfile,
            ExampleRow,
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
        png = render_language_distribution_png(profile)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")


__all__ = [
    "TestLoadAfghanistanOutlineErrors",
    "TestBuildProfileErrorBranches",
    "TestRenderGeoErrorBranches",
    "TestRenderLanguageEmptyBranches",
]
