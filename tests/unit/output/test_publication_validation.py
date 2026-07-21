"""Tests for the publication-level validator.

The publication validator runs after the existing
``validate_export_directory`` to enforce the the implementation extensions:

* the canonical schema contains no Arrow ``map<...>`` fields (HF
  Viewer compatibility);
* the on-disk PNG assets exist and match the manifest SHA-256;
* the README equals ``render_dataset_card_from_profile(profile)``;
* the manifest is at ``manifest_version == 2`` and contains the
  required the implementation fields;
* the manifest example-row field matches the actual first Parquet
  row;
* the ``statistics`` accounting identities hold for all breakdown
  mappings;
* the asset / parquet row count / language sum all reconcile.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.contracts.errors import ExportError
from osm_polygon_sentence_relevance.contracts.schemas import OUTPUT_SENTENCE_SCHEMA
from osm_polygon_sentence_relevance.output import validation_publication
from osm_polygon_sentence_relevance.output.dataset_card import (
    render_dataset_card_from_profile,
)
from osm_polygon_sentence_relevance.output.manifest import (
    MANIFEST_VERSION,
    merge_profile_into_manifest,
)
from osm_polygon_sentence_relevance.output.profile import (
    AssetInfo,
    DatasetProfile,
    build_dataset_profile,
)


def _build_minimal_export(
    tmp_path: Path,
) -> tuple[Path, DatasetProfile, bytes, bytes]:
    """Build sentences.parquet + manifest.json + assets + README.

    Returns the export directory, the profile, the raw geographic PNG
    bytes, and the raw language PNG bytes so individual tests can
    assert against the bytes.
    """
    rows = []
    for idx in range(6):
        rows.append(
            {
                "sentence_id": hashlib.sha256(f"s{idx}".encode()).hexdigest(),
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
                "section_content_hash": hashlib.sha256(b"0").hexdigest(),
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

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    parquet_path = export_dir / "sentences.parquet"
    pq.write_table(table, parquet_path)
    parquet_sha = hashlib.sha256(parquet_path.read_bytes()).hexdigest()

    # Render the two PNGs from the profile (which we are about to
    # build).  Build a preliminary profile without assets to drive
    # PNG rendering; we'll rebuild with assets afterwards.
    preliminary = build_dataset_profile(
        parquet_path=parquet_path,
        parquet_sha256=parquet_sha,
        segmentation_model="sat-3l",
        segmentation_revision="abc1234",
        source_commit="HEAD",
        scratch_dir=tmp_path / "scratch1",
    )
    from osm_polygon_sentence_relevance.output.profile import (
        render_geographic_coverage_png,
        render_language_distribution_png,
    )

    geo_bytes = render_geographic_coverage_png(preliminary, parquet_path)
    lang_bytes = render_language_distribution_png(preliminary)

    assets_dir = export_dir / "assets"
    assets_dir.mkdir()
    (assets_dir / "geographic_coverage.png").write_bytes(geo_bytes)
    (assets_dir / "language_distribution.png").write_bytes(lang_bytes)

    geo_sha = hashlib.sha256(geo_bytes).hexdigest()
    lang_sha = hashlib.sha256(lang_bytes).hexdigest()

    profile = preliminary
    from dataclasses import replace

    profile = replace(
        profile,
        assets={
            "geographic_coverage.png": AssetInfo(
                name="geographic_coverage.png",
                sha256=geo_sha,
                bytes_=len(geo_bytes),
            ),
            "language_distribution.png": AssetInfo(
                name="language_distribution.png",
                sha256=lang_sha,
                bytes_=len(lang_bytes),
            ),
        },
    )

    base_manifest = {
        "manifest_version": MANIFEST_VERSION,
        "row_count": profile.row_count,
        "input_occurrence_count": profile.row_count,
        "duplicates_removed": 0,
        "cross_source_duplicate_groups": 0,
        "counts_by_source": dict(profile.source_counts),
        "counts_by_language": dict(profile.language_counts),
        "counts_by_region": dict(profile.region_counts),
        "input_dataset_revision": profile.input_dataset_revision,
        "pipeline_version": profile.pipeline_version,
        "input_dataset_id": profile.input_dataset_id,
        "sha256": profile.parquet_sha256,
        "statistics": {
            "version": 1,
            "row_count": profile.row_count,
            "unique_sentence_ids": profile.unique_sentence_ids,
            "unique_polygons": profile.unique_polygons,
            "unique_wikidata_entities": profile.unique_wikidata_entities,
            "unique_documents": profile.unique_documents,
            "source_counts": dict(profile.source_counts),
            "language_counts": dict(profile.language_counts),
            "region_counts": dict(profile.region_counts),
            "rows_with_coordinates": profile.rows_with_coordinates,
            "rows_without_coordinates": profile.rows_without_coordinates,
            "input_dataset_revision": profile.input_dataset_revision,
            "pipeline_version": profile.pipeline_version,
            "parquet_sha256": profile.parquet_sha256,
            "input_dataset_id": profile.input_dataset_id,
        },
    }
    final_manifest = merge_profile_into_manifest(base_manifest, profile)
    (export_dir / "manifest.json").write_text(
        json.dumps(final_manifest, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (export_dir / "README.md").write_text(
        render_dataset_card_from_profile(profile), encoding="utf-8"
    )

    return export_dir, profile, geo_bytes, lang_bytes


class TestValidatePublicationHappyPath:
    def test_valid_export_passes(self, tmp_path: Path) -> None:
        export_dir, profile, _geo, _lang = _build_minimal_export(tmp_path)
        result = validation_publication.validate_publication_directory(export_dir)
        assert isinstance(result, validation_publication.ValidatedPublication)
        assert result.asset_count == 2
        assert result.profile_row_count == profile.row_count

    @pytest.mark.parametrize(
        "field",
        [
            "input_occurrence_count",
            "duplicates_removed",
            "cross_source_duplicate_groups",
        ],
    )
    def test_rejects_manifest_duplicate_accounting_drift(
        self, tmp_path: Path, field: str
    ) -> None:
        export_dir, _profile, _geo, _lang = _build_minimal_export(tmp_path)
        manifest_path = export_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest[field] += 1
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ExportError, match=field):
            validation_publication.validate_publication_directory(export_dir)


class TestValidatePublicationAssetFailures:
    def test_missing_asset_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        # Delete one asset file but keep its entry in the manifest.
        (export_dir / "assets" / "geographic_coverage.png").unlink()
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_tampered_asset_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        # Replace asset bytes with unrelated content; sha changes.
        (export_dir / "assets" / "geographic_coverage.png").write_bytes(
            b"not-a-real-png"
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_unknown_extra_asset_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        # Add an asset file that the manifest does not list; this is
        # an exporter mistake (the manifest is the source of truth).
        (export_dir / "assets" / "extra.png").write_bytes(b"x")
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)


class TestValidatePublicationCardFailures:
    def test_stale_readme_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        (export_dir / "README.md").write_text("stale", encoding="utf-8")
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_accounting_drift_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Inject an off-by-one drift in source_counts.
        manifest["counts_by_source"]["wikipedia"] = (
            manifest["counts_by_source"]["wikipedia"] + 1
        )
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)


class TestValidatePublicationSchemaFailures:
    def test_legacy_map_type_export_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If somehow the parquet on disk still has map<...>, reject."""
        from osm_polygon_sentence_relevance.contracts.errors import ExportError
        from osm_polygon_sentence_relevance.output import validation_publication

        # Force schema_has_map_types to return True (simulating a
        # legacy schema exported by mistake).
        monkeypatch.setattr(
            validation_publication, "schema_has_map_types", lambda *_: True
        )
        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)


class TestValidatePublicationExampleRow:
    def test_example_row_mismatch_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Tamper with one example-row field.
        row = dict(manifest["example_row"])
        row["sentence_id"] = "z" * 64
        manifest["example_row"] = row
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)


class TestValidatorsUtilities:
    def test_example_row_matches_first_row(self, tmp_path: Path) -> None:
        """The validator's helper extracts the row that must match the manifest."""
        from osm_polygon_sentence_relevance.output.validation_publication import (
            first_parquet_row,
        )

        export_dir, profile, _g, _l = _build_minimal_export(tmp_path)
        first = first_parquet_row(export_dir / "sentences.parquet")
        assert first["sentence_id"] == profile.example_row["sentence_id"]

    def test_asset_helpers(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.output.validation_publication import (
            compute_asset_sha,
            load_asset_inventory,
        )

        export_dir, profile, geo, lang = _build_minimal_export(tmp_path)
        inventory = load_asset_inventory(export_dir)
        assert "geographic_coverage.png" in inventory
        assert "language_distribution.png" in inventory
        for name, info in inventory.items():
            assert info.name == name
            assert info.sha256 == compute_asset_sha(export_dir / "assets" / name)


class TestValidatePublicationManifestFailures:
    def test_missing_manifest_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir = tmp_path / "empty"
        export_dir.mkdir()
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_path_is_not_a_directory(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(f)

    def test_old_manifest_version_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["manifest_version"] = 1
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_missing_required_keys_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        del manifest["segmentation_model"]
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_assets_list_must_be_list(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["assets"] = {"geographic_coverage.png": {}}  # wrong type
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_assets_entry_must_have_name_and_sha(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["assets"] = [{"name": "", "sha256": "a" * 64}]
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_assets_entry_invalid_bytes(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["assets"][0]["bytes"] = -1
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)


class TestValidatePublicationSymlinkGuard:
    def test_symlinked_asset_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError
        from osm_polygon_sentence_relevance.output.validation_publication import (
            load_asset_inventory,
        )

        d = tmp_path / "isolated"
        d.mkdir()
        (d / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        # Make one of them a symlink (still readable, so is_file returns
        # True via path follows).
        (d / "a.png").unlink()
        real = d / "real.png"
        real.write_bytes(b"x")
        link = d / "a.png"
        link.symlink_to(real)
        with pytest.raises(ExportError):
            load_asset_inventory(d, assets_relative="")


class TestValidatePublicationSchemaMismatch:
    def test_parquet_schema_mismatch_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        # Rewrite parquet with a single-column schema.
        import pyarrow as pa
        import pyarrow.parquet as pq

        small = pa.Table.from_pylist([{"a": 1}])
        pq.write_table(small, export_dir / "sentences.parquet")
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)


class TestValidatePublicationRegionAndSourceAccounting:
    def test_region_drift_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Push counts_by_region sum past row_count.
        manifest["counts_by_region"]["a"] = manifest["counts_by_region"]["a"] + 99
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_source_drift_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Preserve sum but swap counts: same sum, different keys.
        s = manifest["counts_by_source"]
        manifest["counts_by_source"] = {
            "wikipedia": s.get("wikipedia", 0),
            "wanderlust": s.get("wikivoyage", 0),
        }
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)


class TestValidatePublicationSHAEdgeCases:
    def test_computed_sha_uses_correct_function(self, tmp_path: Path) -> None:
        # Make sure the validator's compute_asset_sha path matches
        # what the manifest will store.
        import hashlib

        from osm_polygon_sentence_relevance.output.validation_publication import (
            compute_asset_sha,
        )

        payload = b"\x89PNG\r\n\x1a\n" + b"binary-data"
        f = tmp_path / "blob.bin"
        f.write_bytes(payload)
        assert compute_asset_sha(f) == hashlib.sha256(payload).hexdigest().lower()

    def test_first_parquet_row_on_empty_file(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError
        from osm_polygon_sentence_relevance.output.validation_publication import (
            first_parquet_row,
        )

        empty = tmp_path / "empty.parquet"
        # Write a 0-row table conforming to the schema.
        import pyarrow.parquet as pq

        from osm_polygon_sentence_relevance.contracts.schemas import (
            OUTPUT_SENTENCE_SCHEMA,
        )

        table = OUTPUT_SENTENCE_SCHEMA.empty_table()
        pq.write_table(table, empty)
        with pytest.raises(ExportError):
            first_parquet_row(empty)


class TestValidatePublicationExtendedCoverage:
    def test_unreadable_asset_file(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError
        from osm_polygon_sentence_relevance.output.validation_publication import (
            compute_asset_sha,
        )

        with pytest.raises(ExportError):
            compute_asset_sha(tmp_path / "missing.png")

    def test_inventory_assets_dir_missing(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError
        from osm_polygon_sentence_relevance.output.validation_publication import (
            load_asset_inventory,
        )

        with pytest.raises(ExportError):
            load_asset_inventory(tmp_path / "nope")

    def test_schema_walks_map_branch(self, tmp_path: Path) -> None:
        # The validator must reject a schema containing a map field, even
        # one accessed after a list-of-struct combination.
        import pyarrow as pa

        from osm_polygon_sentence_relevance.output.dataset_card import (
            schema_has_map_types,
        )

        # list-of-struct, struct child is map: nested case.
        s = pa.schema(
            [
                pa.field(
                    "x",
                    pa.list_(
                        pa.struct(
                            [
                                pa.field(
                                    "y",
                                    pa.map_(pa.string(), pa.string()),
                                )
                            ]
                        )
                    ),
                )
            ]
        )
        assert schema_has_map_types(s) is True

    def test_example_row_value_mismatch_specific(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Tweak a non-sentence_id column in example_row.
        row = dict(manifest["example_row"])
        row["page_title"] = "tampered"
        manifest["example_row"] = row
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_extra_asset_file_on_disk(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        # Add an asset file that is not listed in the manifest.
        (export_dir / "assets" / "extra.png").write_bytes(b"x")
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_asset_bytes_mismatch(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Pretend the on-disk asset has a different size.
        for asset in manifest["assets"]:
            if asset["name"] == "geographic_coverage.png":
                asset["bytes"] = asset["bytes"] + 7
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)


class TestValidatePublicationPathShapeBranches:
    def test_assets_dir_is_a_file(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        import shutil

        shutil.rmtree(export_dir / "assets")
        (export_dir / "assets").write_bytes(b"")
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_parquet_path_is_a_directory(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        (export_dir / "sentences.parquet").unlink()
        (export_dir / "sentences.parquet").mkdir()
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_manifest_path_is_a_directory(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        (export_dir / "manifest.json").unlink()
        (export_dir / "manifest.json").mkdir()
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_card_path_is_a_directory(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        (export_dir / "README.md").unlink()
        (export_dir / "README.md").mkdir()
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_assets_path_does_not_exist(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        import shutil

        shutil.rmtree(export_dir / "assets")
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_parquet_path_does_not_exist(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        (export_dir / "sentences.parquet").unlink()
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_card_path_does_not_exist(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        (export_dir / "README.md").unlink()
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_manifest_path_does_not_exist(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        (export_dir / "manifest.json").unlink()
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)


class TestValidatePublicationRegionDisagree:
    def test_region_drift_without_sum_change(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Rename the region key (preserves sum, but breaks the
        # region one-by-one comparison).
        s = manifest["counts_by_region"]
        manifest["counts_by_region"] = {"renamed": s["a"]}
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_parquet_unreadable(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        # Replace the parquet with garbage bytes so pq.ParquetFile
        # raises.
        (export_dir / "sentences.parquet").write_bytes(b"not-a-parquet")
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_example_row_specific_column_drift(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Tweak an example_row field whose value is the actual parity
        # test (use a column that is unlikely to be hit by a sum).
        row = dict(manifest["example_row"])
        row["polygon_id"] = "tampered"
        manifest["example_row"] = row
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_assets_value_not_dict(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Insert an integer into the assets list.
        manifest["assets"] = list(manifest["assets"]) + [42]
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_assets_entry_missing_sha(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Drop the sha256 field entirely.
        for asset in manifest["assets"]:
            asset.pop("sha256", None)
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)


class TestValidatePublicationAssetInventory:
    def test_load_asset_inventory_skips_directories(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.output.validation_publication import (
            load_asset_inventory,
        )

        d = tmp_path / "mixed"
        d.mkdir()
        (d / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        sub = d / "subdir"
        sub.mkdir()
        inventory = load_asset_inventory(d, assets_relative="")
        assert "a.png" in inventory
        assert "subdir" not in inventory


class TestValidatePublicationAssetHardening:
    def test_assets_sha_mismatch_raises(self, tmp_path: Path) -> None:
        """The validator must reject when the manifest's recorded SHA
        doesn't match the actual asset bytes."""
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Replace the recorded SHA with one that doesn't match.
        for asset in manifest["assets"]:
            if asset["name"] == "geographic_coverage.png":
                asset["sha256"] = "0" * 64
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_assets_name_mismatch_raises(self, tmp_path: Path) -> None:
        """Manifest name does not match on-disk file extension."""
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        # Move a manifest-listed file out and add an extra file with a
        # different name so the manifest still claims the missing name.
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        (export_dir / "assets" / "geographic_coverage.png").unlink()
        # Rename the manifest entry's name so the file is missing
        # under the recorded name (set diff).
        for asset in manifest["assets"]:
            if asset["name"] == "geographic_coverage.png":
                asset["name"] = "renamed-coverage.png"
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_accounting_drift_language_sum(self, tmp_path: Path) -> None:
        """Even if the per-language check passes, the row-difference
        check should reject."""
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Bump language total past row_count.
        manifest["counts_by_language"]["en"] = (
            manifest["counts_by_language"]["en"] + 100
        )
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)


class TestValidatePublicationHelpers:
    def test_asset_entry_non_string_name(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        for asset in manifest["assets"]:
            asset["name"] = 42
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_asset_entry_non_string_sha(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        for asset in manifest["assets"]:
            asset["sha256"] = 42
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_blank_segmentation_revision_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["segmentation_revision"] = "   "
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_passing_in_explicit_scratch_dir(self, tmp_path: Path) -> None:
        """When *scratch_dir* is provided, the validator must use it
        instead of creating a temporary one."""
        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        scratch = tmp_path / "my-scratch"
        result = validation_publication.validate_publication_directory(
            export_dir,
            scratch_dir=scratch,
        )
        assert result.profile_row_count == _p.row_count


class TestValidatePublicationTypeChecks:
    def test_path_type_validation(self) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        with pytest.raises((TypeError, ExportError)):
            validation_publication.validate_publication_directory(42)

    def test_assets_non_list_manifest(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["assets"] = "not-a-list"
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)


class TestValidatePublicationContractExtensions:
    """The corrective release adds strict five-file contract checks
    that reject an extra file at the publication root and reject a
    missing required artefact. These tests pin those branches."""

    def test_extra_root_file_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        (export_dir / "extra.txt").write_text("nope")
        with pytest.raises(ExportError, match="contract"):
            validation_publication.validate_publication_directory(export_dir)

    def test_missing_parquet_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        (export_dir / "sentences.parquet").unlink()
        with pytest.raises(ExportError, match="Missing required artefact"):
            validation_publication.validate_publication_directory(export_dir)

    def test_parquet_sha_mismatch_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["sha256"] = "0" * 64
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError, match="sha"):
            validation_publication.validate_publication_directory(export_dir)

    def test_manifest_json_decode_error_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        (export_dir / "manifest.json").write_text("not-json")
        with pytest.raises(ExportError, match="Manifest is not readable"):
            validation_publication.validate_publication_directory(export_dir)

    def test_manifest_not_object_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        (export_dir / "manifest.json").write_text("[1, 2, 3]")
        with pytest.raises(ExportError, match="Manifest must be a JSON object"):
            validation_publication.validate_publication_directory(export_dir)

    def test_language_counts_disagree_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Add a phantom language with row_count - sum(other langs).
        existing = sum(manifest["counts_by_language"].values())
        phantom = manifest["row_count"] - existing
        manifest["counts_by_language"]["xx"] = phantom
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError, match="counts_by_language disagrees"):
            validation_publication.validate_publication_directory(export_dir)

    def test_symlink_to_contract_file_rejected(self, tmp_path: Path) -> None:
        """A symlink at the root whose target is a contract file must
        be rejected as a symlink (rather than as a missing file)."""
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        # Replace sentences.parquet with a symlink so the contract
        # check sees the contract file present but symlinked.
        (export_dir / "sentences.parquet").unlink()
        link = export_dir / "sentences.parquet"
        # Point at a file outside the export directory so the
        # contract check sees the symlink, not an extra file.
        link.symlink_to(tmp_path.parent / "some_other_file.bin")
        with pytest.raises(ExportError, match="symlink"):
            validation_publication.validate_publication_directory(export_dir)

    def test_profile_build_failure_wrapped(self, tmp_path: Path, monkeypatch) -> None:
        """If ``build_dataset_profile`` raises, the validator must
        re-raise as ``ExportError`` with the original cause."""
        from osm_polygon_sentence_relevance.contracts.errors import ExportError
        from osm_polygon_sentence_relevance.output import validation_publication

        def _broken(*args, **kwargs):
            raise validation_publication.ProfileError("synthetic profile error")

        monkeypatch.setattr(validation_publication, "build_dataset_profile", _broken)
        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        with pytest.raises(ExportError, match="Could not rebuild profile"):
            validation_publication.validate_publication_directory(export_dir)

    def test_stale_card_text_rejected(self, tmp_path: Path) -> None:
        """When the on-disk README differs from the deterministic
        profile render, the validator must reject."""
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        (export_dir / "README.md").write_text("not the rendered card", encoding="utf-8")
        with pytest.raises(ExportError, match="stale|deterministic"):
            validation_publication.validate_publication_directory(export_dir)

    def test_unexpected_subdirectory_rejected(self, tmp_path: Path) -> None:
        """An extra directory at the root is rejected as 'unexpected'."""
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        (export_dir / "extra-dir").mkdir()
        with pytest.raises(ExportError, match="unexpected subdirectory"):
            validation_publication.validate_publication_directory(export_dir)

    def test_card_unreadable_oserror(self, tmp_path: Path) -> None:
        """Replacing the card with a directory triggers the OSError path."""
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        (export_dir / "README.md").unlink()
        (export_dir / "README.md").mkdir()
        # The contract check rejects a directory at the root path,
        # so the OSError branch is unreachable through the public
        # validator entry point. We rely on the contract-level
        # rejection to surface the problem.
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_assets_set_mismatch_rejected(self, tmp_path: Path) -> None:
        """The on-disk asset set differs from the manifest's."""
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir, _p, _g, _l = _build_minimal_export(tmp_path)
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Insert a phantom asset entry; the on-disk set won't match.
        manifest["assets"] = list(manifest["assets"]) + [
            {"name": "ghost.png", "sha256": "a" * 64, "bytes": 1}
        ]
        (export_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ExportError, match="Asset set"):
            validation_publication.validate_publication_directory(export_dir)

    def test_manifest_repo_id_not_string(self) -> None:
        """``_derive_asset_base_url`` returns None when repo_id is not a string."""
        from osm_polygon_sentence_relevance.output.validation_publication import (
            _derive_asset_base_url,
        )

        # Non-string repo_id.
        assert _derive_asset_base_url({"dataset_repo_id": 12345}) is None
        # Empty string repo_id.
        assert _derive_asset_base_url({"dataset_repo_id": ""}) is None
        # Missing key entirely.
        assert _derive_asset_base_url({}) is None
        # Valid repo_id yields the HF CDN URL.
        url = _derive_asset_base_url({"dataset_repo_id": "owner/name"})
        assert url == "https://huggingface.co/datasets/owner/name/resolve/main/assets"
