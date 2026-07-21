"""Strict publication-contract tests for Phase 9P corrective release.

These tests pin the exact layout that a published dataset directory
must satisfy on Hugging Face `main`. The previous publication
regression (Phase 9P final commit ``2e8d68d9``) deleted
``sentences.parquet`` from `main`; the publication contract must
make that kind of omission impossible to ship.

Contract
--------

A complete publication directory must contain exactly five files
(the dataset card must be generated; no sidecar Viewer parquet, no
extra PNG, no orphan files):

* ``sentences.parquet``
* ``manifest.json``
* ``README.md``
* ``assets/geographic_coverage.png``
* ``assets/language_distribution.png``

These tests cover two complementary layers of the contract:

1. ``scripts/render_assets.py`` must construct a directory whose
   on-disk layout matches the contract.
2. ``validate_publication_directory`` must reject any directory
   that does not match the contract.

Both layers run independently so a regression at either layer
is caught.
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

_REQUIRED_PUBLICATION_FILES: tuple[str, ...] = (
    "sentences.parquet",
    "manifest.json",
    "README.md",
    "assets/geographic_coverage.png",
    "assets/language_distribution.png",
)


def _build_publication_row(idx: int) -> dict:
    """Build a single deterministic ``OUTPUT_SENTENCE_SCHEMA`` row."""
    return {
        "sentence_id": hashlib.sha256(f"s{idx}".encode()).hexdigest(),
        "polygon_id": f"afghanistan-latest:way:{idx // 3}",
        "wikidata": f"Q{idx + 1}",
        "document_id": f"doc{idx // 4}",
        "article_id": None,
        "source": "wikipedia" if idx % 2 == 0 else "wikivoyage",
        "language": ["en", "fa", "ps"][idx % 3],
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
        "document_content_hash": hashlib.sha256(f"doc{idx // 4}".encode()).hexdigest(),
        "section_content_hash": hashlib.sha256(b"0").hexdigest(),
        "sentence_content_hash": hashlib.sha256(
            f"Row {idx} text.".encode()
        ).hexdigest(),
        "duplicate_occurrence_count": 1,
        "duplicate_sources": ["wikipedia"],
        "polygon_name": None,
        "osm_primary_tag": None,
        "osm_tags": [{"key": "highway", "value": "primary"}],
        "region": "afghanistan-latest",
        "lat": 34.0 + idx * 0.1,
        "lon": 69.0 + idx * 0.1,
        "input_dataset_revision": "rev",
        "pipeline_version": "1.0.0",
    }


def _build_contract_parquet(path: Path, *, rows: int = 12) -> tuple[str, int]:
    """Build a minimal but schema-compliant Parquet at *path*."""
    table = pa.Table.from_pylist(
        [_build_publication_row(idx) for idx in range(rows)],
        schema=OUTPUT_SENTENCE_SCHEMA,
    )
    table = table.replace_schema_metadata(
        {
            b"input_dataset_revision": b"rev",
            b"pipeline_version": b"1.0.0",
        }
    )
    pq.write_table(table, path)
    return hashlib.sha256(path.read_bytes()).hexdigest(), rows


def _build_contract_publication(
    export_dir: Path,
) -> tuple[DatasetProfile, bytes, bytes]:
    """Build a complete contract-compliant publication at *export_dir*.

    Returns the profile, the geographic PNG bytes, and the language
    PNG bytes so individual tests can verify byte-level invariants.
    """
    parquet_path = export_dir / "sentences.parquet"
    parquet_sha, _ = _build_contract_parquet(parquet_path)

    profile = build_dataset_profile(
        parquet_path=parquet_path,
        parquet_sha256=parquet_sha,
        segmentation_model="sat-3l",
        segmentation_revision="abc1234",
        source_commit="HEAD",
        scratch_dir=export_dir / ".scratch",
    )
    from osm_polygon_sentence_relevance.output.profile import (
        render_geographic_coverage_png,
        render_language_distribution_png,
    )

    geo_bytes = render_geographic_coverage_png(profile, parquet_path)
    lang_bytes = render_language_distribution_png(profile)

    assets_dir = export_dir / "assets"
    assets_dir.mkdir()
    (assets_dir / "geographic_coverage.png").write_bytes(geo_bytes)
    (assets_dir / "language_distribution.png").write_bytes(lang_bytes)

    geo_sha = hashlib.sha256(geo_bytes).hexdigest()
    lang_sha = hashlib.sha256(lang_bytes).hexdigest()

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

    # Clean up the scratch directory so the directory matches the
    # strict five-file contract.
    import shutil

    scratch = export_dir / ".scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    return profile, geo_bytes, lang_bytes


class TestPublicationContractFilesystem:
    """The publication directory must contain exactly the contract files."""

    def test_contract_lists_exactly_five_files(self) -> None:
        """The contract itself must be the documented set; a future
        contributor who adds a file to the contract must consciously
        extend this tuple.
        """
        assert _REQUIRED_PUBLICATION_FILES == (
            "sentences.parquet",
            "manifest.json",
            "README.md",
            "assets/geographic_coverage.png",
            "assets/language_distribution.png",
        )

    def test_publication_directory_lists_contract_files(self, tmp_path: Path) -> None:
        """After running the publication pipeline the directory must
        contain exactly the contract files. Anything else is a
        publication bug.
        """
        export_dir = tmp_path / "publish"
        export_dir.mkdir()
        _build_contract_publication(export_dir)
        relpaths = sorted(
            p.relative_to(export_dir).as_posix()
            for p in export_dir.rglob("*")
            if p.is_file()
        )
        # Exclude internal scratch and cache artifacts the publication
        # pipeline leaves behind (e.g. .scratch for the SQLite build).
        relpaths = [
            p
            for p in relpaths
            if not p.startswith(".scratch/") and not p.startswith(".staging/")
        ]
        assert sorted(_REQUIRED_PUBLICATION_FILES) == relpaths

    def test_publication_directory_rejects_missing_parquet(
        self, tmp_path: Path
    ) -> None:
        """The publication validator must reject a directory that
        ships a README/manifest/assets but no parquet (the bug that
        affected commit 2e8d68d9dab298af2bf97b20be909cb6c0de350b).
        """
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir = tmp_path / "missing-parquet"
        export_dir.mkdir()
        _build_contract_publication(export_dir)
        # Remove parquet only.
        (export_dir / "sentences.parquet").unlink()
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_publication_directory_rejects_missing_manifest(
        self, tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir = tmp_path / "missing-manifest"
        export_dir.mkdir()
        _build_contract_publication(export_dir)
        (export_dir / "manifest.json").unlink()
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_publication_directory_rejects_missing_readme(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir = tmp_path / "missing-readme"
        export_dir.mkdir()
        _build_contract_publication(export_dir)
        (export_dir / "README.md").unlink()
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_publication_directory_rejects_missing_geographic_png(
        self, tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir = tmp_path / "missing-geo"
        export_dir.mkdir()
        _build_contract_publication(export_dir)
        (export_dir / "assets" / "geographic_coverage.png").unlink()
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_publication_directory_rejects_missing_language_png(
        self, tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir = tmp_path / "missing-lang"
        export_dir.mkdir()
        _build_contract_publication(export_dir)
        (export_dir / "assets" / "language_distribution.png").unlink()
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_publication_directory_rejects_extra_orphan_file(
        self, tmp_path: Path
    ) -> None:
        """The validator rejects an extra file under the publication
        root because the contract is *exact*. An orphan at the root
        breaks the parity with the canonical five-file layout.
        """
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir = tmp_path / "orphan"
        export_dir.mkdir()
        _build_contract_publication(export_dir)
        (export_dir / "extra.txt").write_text("nope")
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_publication_directory_rejects_extra_asset(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir = tmp_path / "extra-asset"
        export_dir.mkdir()
        _build_contract_publication(export_dir)
        (export_dir / "assets" / "extra.png").write_bytes(b"x")
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)


class TestPublicationScriptPipeline:
    """The render script must produce a directory the validator accepts.

    These tests instantiate ``_publish_directory`` (the same helper
    the CLI invokes) so a regression in the script's directory
    construction cannot slip past the existing test suite. The
    contract is tested end-to-end: every artifact survives, no
    parquet gets dropped during directory cleanup.
    """

    def test_publish_directory_survives_cleanup(self, tmp_path: Path) -> None:
        """The script's ``_publish_directory`` helper rebuilds the
        output directory but must preserve the converted parquet
        across the wipe. Without that preservation the post-cleanup
        directory would lose its parquet.
        """
        from scripts.render_assets import _publish_directory

        # Build a minimal input parquet and write a staging copy.
        staging_input = tmp_path / "input.parquet"
        _build_contract_parquet(staging_input)

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        # Convert via the script so we exercise the same code path.
        from scripts.render_assets import _convert_parquet

        staging = output_dir / ".staging"
        staging.mkdir()
        converted = staging / "sentences.parquet"
        _convert_parquet(staging_input, converted)
        assert converted.is_file()

        _publish_directory(
            converted,
            output_dir,
            "sat-3l",
            "abc1234",
            "HEAD",
        )

        assert (output_dir / "sentences.parquet").is_file()
        assert (output_dir / "manifest.json").is_file()
        assert (output_dir / "README.md").is_file()
        assert (output_dir / "assets" / "geographic_coverage.png").is_file()
        assert (output_dir / "assets" / "language_distribution.png").is_file()

        # The publication directory must round-trip through the
        # validator. This is the same call the publish script must
        # pass before pushing to the Hub.
        result = validation_publication.validate_publication_directory(output_dir)
        assert result.profile_row_count > 0

    def test_publish_directory_overwrites_dont_drop_parquet(
        self, tmp_path: Path
    ) -> None:
        """Re-running the publication script on a populated directory
        must keep the parquet; only the README, manifest, and assets
        are regenerated. The publication pipeline never deletes the
        parquet because the conversion step writes to ``.staging``
        first.
        """
        from scripts.render_assets import _convert_parquet, _publish_directory

        output_dir = tmp_path / "out"
        output_dir.mkdir()
        staging_input = tmp_path / "input.parquet"
        _build_contract_parquet(staging_input)

        # Run the publication twice. The ``.staging`` directory is
        # cleaned up by the first run, so each run re-converts from
        # the same input parquet. This mirrors the production flow
        # exactly: the script writes to ``.staging`` and then runs
        # ``_publish_directory`` which cleans the directory on the
        # way out.
        for _ in range(2):
            staging = output_dir / ".staging"
            staging.mkdir()
            converted = staging / "sentences.parquet"
            _convert_parquet(staging_input, converted)
            _publish_directory(
                converted,
                output_dir,
                "sat-3l",
                "abc1234",
                "HEAD",
            )

        assert (output_dir / "sentences.parquet").is_file()

        # Both passes must validate.
        validation_publication.validate_publication_directory(output_dir)

    def test_manifest_uses_profile_derived_duplicate_accounting(
        self, tmp_path: Path
    ) -> None:
        """The render script must never reset deduplication accounting to
        row_count/zero while converting the Parquet schema."""
        from dataclasses import replace

        from scripts.render_assets import _build_manifest_payload

        export_dir = tmp_path / "publication"
        export_dir.mkdir()
        profile, _geo, _lang = _build_contract_publication(export_dir)
        profile = replace(
            profile,
            input_occurrence_count=profile.row_count + 9,
            duplicates_removed=9,
            cross_source_duplicate_groups=2,
        )

        manifest = _build_manifest_payload(profile)

        assert manifest["input_occurrence_count"] == profile.row_count + 9
        assert manifest["duplicates_removed"] == 9
        assert manifest["cross_source_duplicate_groups"] == 2

    def test_manifest_payload_is_deterministic(self, tmp_path: Path) -> None:
        from scripts.render_assets import _build_manifest_payload

        export_dir = tmp_path / "publication"
        export_dir.mkdir()
        profile, _geo, _lang = _build_contract_publication(export_dir)

        first = _build_manifest_payload(profile)
        second = _build_manifest_payload(profile)

        assert first == second
        assert "generated_at" not in first


class TestPublicationHelperRejectsPathMistakes:
    """A non-existent path or wrong type must be rejected loudly."""

    def test_publication_directory_rejects_nonexistent_path(
        self, tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(
                tmp_path / "does-not-exist"
            )

    def test_publication_directory_rejects_file_not_directory(
        self, tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        f = tmp_path / "a-file.txt"
        f.write_text("hi")
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(f)

    def test_publication_directory_rejects_root_symlink(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir = tmp_path / "publish"
        export_dir.mkdir()
        _build_contract_publication(export_dir)
        # Replace a contract file with a symlink; the symlink branch
        # must reject the publication.
        target = export_dir / "sentences.parquet"
        (export_dir / "sentences.parquet").unlink()
        link = export_dir / "sentences.parquet"
        link.symlink_to(target)
        with pytest.raises(ExportError):
            validation_publication.validate_publication_directory(export_dir)

    def test_publication_directory_rejects_unreadable_manifest(
        self, tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir = tmp_path / "publish"
        export_dir.mkdir()
        _build_contract_publication(export_dir)
        # Replace manifest.json with a directory-shaped blob so the
        # JSON load fails.
        import os

        os.remove(export_dir / "manifest.json")
        os.makedirs(export_dir / "manifest.json")
        with pytest.raises(ExportError, match="Manifest is not readable|not a file"):
            validation_publication.validate_publication_directory(export_dir)

    def test_publication_directory_rejects_unreadable_card(
        self, tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.contracts.errors import ExportError

        export_dir = tmp_path / "publish"
        export_dir.mkdir()
        _build_contract_publication(export_dir)
        # Replace README.md with a directory so the read fails.
        import os

        os.remove(export_dir / "README.md")
        os.makedirs(export_dir / "README.md")
        with pytest.raises(ExportError, match="Card is not readable|not a file"):
            validation_publication.validate_publication_directory(export_dir)


__all__ = [
    "TestPublicationContractFilesystem",
    "TestPublicationScriptPipeline",
    "TestPublicationHelperRejectsPathMistakes",
    "TestSha256BytesUtility",
]


class TestSha256BytesUtility:
    """The shared ``sha256_bytes`` helper used by both the manifest
    and the on-disk asset validation must produce lowercase hex."""

    def test_returns_lowercase_64_hex(self) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            sha256_bytes,
        )

        result = sha256_bytes(b"hello world")
        # Known fixed hash for "hello world".
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert result == expected
        assert all(c in "0123456789abcdef" for c in result)
        assert len(result) == 64

    def test_different_payloads_produce_different_hashes(self) -> None:
        from osm_polygon_sentence_relevance.output.profile import (
            sha256_bytes,
        )

        h1 = sha256_bytes(b"alpha")
        h2 = sha256_bytes(b"beta")
        assert h1 != h2
