"""Publication script for the Afghanistan phase-9P export.

Reads the existing /tmp/afghanistan-main/sentences.parquet (which
still uses the legacy map<string,string> form for ``osm_tags``),
converts ``osm_tags`` to the Viewer-compatible
``list<struct<{key, value}>>`` form, rebuilds the manifest and the
two PNG assets, and writes the final export directory ready for
publication.

Run from the project root::

    PYTHONPATH=src .venv/bin/python scripts/render_assets.py \\
        --input-parquet /tmp/afghanistan-main/sentences.parquet \\
        --output-dir /tmp/afghanistan-publication \\
        --segmentation-model wtpsplit \\
        --segmentation-revision sat-3l \\
        --source-commit "$(git rev-parse HEAD)"
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.contracts.schemas import OUTPUT_SENTENCE_SCHEMA
from osm_polygon_sentence_relevance.output.checksum import sha256_file
from osm_polygon_sentence_relevance.output.dataset_card import (
    render_dataset_card_from_profile,
)
from osm_polygon_sentence_relevance.output.manifest import (
    MANIFEST_VERSION,
    merge_profile_into_manifest,
    write_manifest,
)
from osm_polygon_sentence_relevance.output.profile import (
    AssetInfo,
    DatasetProfile,
    build_dataset_profile,
    render_geographic_coverage_png,
    render_language_distribution_png,
)
from osm_polygon_sentence_relevance.sentences.finalization import (
    convert_osm_tags_to_list_of_struct,
)


def _convert_parquet(input_path: Path, output_path: Path) -> None:
    """Convert *input_path* in place: rewrite *output_path* with
    ``osm_tags`` changed from ``map<string,string>`` to the
    Viewer-compatible ``list<struct<key, value>>`` form.

    All other fields pass through byte-identical.  The Parquet
    schema metadata is preserved.
    """
    table = pq.read_table(input_path)
    out_columns: dict[str, list] = {}
    for field in OUTPUT_SENTENCE_SCHEMA:
        col = table.column(field.name)
        if field.name == "osm_tags":
            new_col = pa.array(
                [
                    convert_osm_tags_to_list_of_struct(row)
                    for row in col.to_pylist()
                ],
                type=field.type,
            )
            out_columns["osm_tags"] = new_col
        else:
            out_columns[field.name] = col
    new_table = pa.table(out_columns, schema=OUTPUT_SENTENCE_SCHEMA)
    metadata = dict(table.schema.metadata or {})
    new_table = new_table.replace_schema_metadata(metadata)
    pq.write_table(new_table, output_path)


def _build_publication(
    parquet_path: Path,
    segmentation_model: str,
    segmentation_revision: str,
    source_commit: str,
    scratch_dir: Path,
) -> tuple[DatasetProfile, bytes, bytes]:
    """Build the profile and the two PNG assets from *parquet_path*."""
    parquet_sha = sha256_file(parquet_path)
    profile = build_dataset_profile(
        parquet_path=parquet_path,
        parquet_sha256=parquet_sha,
        segmentation_model=segmentation_model,
        segmentation_revision=segmentation_revision,
        source_commit=source_commit,
        scratch_dir=scratch_dir,
    )
    geo_bytes = render_geographic_coverage_png(profile, parquet_path)
    lang_bytes = render_language_distribution_png(profile)
    from dataclasses import replace

    geo_sha = hashlib.sha256(geo_bytes).hexdigest()
    lang_sha = hashlib.sha256(lang_bytes).hexdigest()
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
    return profile, geo_bytes, lang_bytes


def _build_manifest_payload(
    profile: DatasetProfile,
    *,
    dataset_repo_id: str | None = None,
) -> dict:
    """Build the manifest dict without merging another stats pass.

    The profile is the single source of truth for every quantitative
    field; ``merge_profile_into_manifest`` overlays the per-asset
    SHA-256s, the segmentation metadata, and the example row.

    Parameters
    ----------
    dataset_repo_id
        Optional ``org/name`` of the Hugging Face dataset repo the
        publication targets.  Recorded in the manifest so the
        validator can reproduce the on-disk README's ``huggingface.co``
        asset URLs.
    """
    base = {
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
    return merge_profile_into_manifest(
        base,
        profile,
        generated_at=_dt.datetime.now(_dt.UTC).isoformat(),
        dataset_repo_id=dataset_repo_id,
    )


def _publish_directory(
    parquet_path: Path,
    output_dir: Path,
    segmentation_model: str,
    segmentation_revision: str,
    source_commit: str,
    *,
    asset_base_url: str | None = None,
    dataset_repo_id: str | None = None,
) -> tuple[Path, str, str]:
    """Build the publication directory at *output_dir*.

    Returns the parquet file path and its SHA-256.  Idempotent:
    *output_dir* is wiped before reconstruction (the staging path
    that holds the converted parquet is rebuilt before this function
    is called so it survives the wipe).
    """
    if output_dir.exists():
        # Preserve a sibling .staging/ directory; only wipe the
        # published artefact area.  This way the converted parquet
        # produced by ``_convert_parquet`` survives.
        for entry in output_dir.iterdir():
            if entry.name == ".staging":
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    output_parquet = output_dir / "sentences.parquet"
    shutil.copy2(parquet_path, output_parquet)

    with tempfile.TemporaryDirectory(prefix="pub-build-") as scratch:
        scratch_dir = Path(scratch)
        profile, geo_bytes, lang_bytes = _build_publication(
            output_parquet,
            segmentation_model,
            segmentation_revision,
            source_commit,
            scratch_dir,
        )

    assets_dir = output_dir / "assets"
    assets_dir.mkdir()
    (assets_dir / "geographic_coverage.png").write_bytes(geo_bytes)
    (assets_dir / "language_distribution.png").write_bytes(lang_bytes)

    # Clean up the staging directory used for the parquet conversion.
    # The publication directory must contain *only* the canonical
    # five-file contract artefacts so the validator and the
    # Hugging Face Dataset Viewer can ingest the upload.
    staging_dir = output_dir / ".staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    # Default asset_base_url: the Hugging Face CDN URL relative to the
    # public repo.  This makes the README render the PNGs on any
    # dataset-card renderer (some viewers do not rewrite relative
    # ``assets/`` paths reliably).
    if asset_base_url is None and dataset_repo_id is not None:
        asset_base_url = (
            f"https://huggingface.co/datasets/{dataset_repo_id}"
            "/resolve/main/assets"
        )

    manifest = _build_manifest_payload(
        profile, dataset_repo_id=dataset_repo_id
    )
    write_manifest(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(
        render_dataset_card_from_profile(
            profile, asset_base_url=asset_base_url
        ),
        encoding="utf-8",
    )

    publication_sha = sha256_file(output_parquet)
    return output_parquet, publication_sha, profile.parquet_sha256


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-parquet",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--segmentation-model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--segmentation-revision",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--source-commit",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="NoeFlandre/osm-polygon-wikidata-sentence-relevance",
    )
    parser.add_argument(
        "--asset-base-url",
        type=str,
        default=None,
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    staging = args.output_dir / ".staging"
    staging.mkdir(exist_ok=True)
    converted = staging / "sentences.parquet"
    _convert_parquet(args.input_parquet, converted)

    parquet_path, publication_sha, profile_sha = _publish_directory(
        converted,
        args.output_dir,
        args.segmentation_model,
        args.segmentation_revision,
        args.source_commit,
        asset_base_url=args.asset_base_url,
        dataset_repo_id=args.dataset_repo_id,
    )
    print(
        json.dumps(
            {
                "parquet": str(parquet_path),
                "parquet_sha256": publication_sha,
                "profile_sha256": profile_sha,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = ["main", "_convert_parquet", "_build_publication", "_build_manifest_payload"]
