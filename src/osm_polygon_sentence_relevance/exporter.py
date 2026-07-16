"""Deterministic, atomically installed dataset export.

This module is the stable public facade. Implementation is split into focused
internal helpers: manifest construction, streaming checksums, and rollback-safe
directory installation.  The atomic-swap algorithm (build tmpdir, back up the
existing output, rename into place, only then remove the backup) is unchanged.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from osm_polygon_sentence_relevance._export.atomic import (
    cleanup_on_failure,
    install_atomic,
    remove_backup,
)
from osm_polygon_sentence_relevance._export.checksum import sha256_file
from osm_polygon_sentence_relevance._export.manifest import (
    build_manifest_data,
    write_manifest,
)
from osm_polygon_sentence_relevance.errors import ExportError
from osm_polygon_sentence_relevance.finalization import FinalizedDataset
from osm_polygon_sentence_relevance.schemas import OUTPUT_SENTENCE_SCHEMA


@dataclass(frozen=True, slots=True)
class ExportResult:
    """The result of exporting a finalized dataset."""

    parquet_path: Path
    manifest_path: Path
    manifest_data: dict


def export_finalized_dataset(
    dataset: FinalizedDataset,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> ExportResult:
    """Export the finalized dataset and its manifest atomically to output_dir.

    Parameters
    ----------
    dataset : FinalizedDataset
        The finalized dataset instance containing the Arrow table and report.
    output_dir : str | Path
        The directory where files should be exported.
    overwrite : bool, default False
        Whether to overwrite an existing non-empty output directory.

    Returns
    -------
    ExportResult
        An object containing the paths to sentences.parquet and manifest.json
        along with the manifest dictionary data.

    Raises
    ------
    TypeError
        If dataset is not a FinalizedDataset instance.
    ExportError
        If table schema mismatch, inconsistent revision/version values,
        or target directory exists and overwrite=False.
    """
    # 1. Reject non-FinalizedDataset input
    if not isinstance(dataset, FinalizedDataset):
        raise TypeError("dataset must be a FinalizedDataset instance")

    # 2. Reject table schema mismatch
    if not dataset.table.schema.equals(OUTPUT_SENTENCE_SCHEMA):
        raise ExportError("Table schema does not match expected OUTPUT_SENTENCE_SCHEMA")

    # 3. Reject inconsistent revision/version values within rows
    metadata = dataset.table.schema.metadata
    meta_rev = (
        metadata.get(b"input_dataset_revision")
        if metadata and b"input_dataset_revision" in metadata
        else None
    )
    meta_ver = (
        metadata.get(b"pipeline_version")
        if metadata and b"pipeline_version" in metadata
        else None
    )

    input_dataset_revision = meta_rev.decode("utf-8") if meta_rev else None
    pipeline_version = meta_ver.decode("utf-8") if meta_ver else None

    if dataset.table.num_rows > 0:
        revisions = dataset.table.column("input_dataset_revision").unique().to_pylist()
        versions = dataset.table.column("pipeline_version").unique().to_pylist()

        if len(revisions) != 1:
            raise ExportError("Inconsistent input_dataset_revision values within rows")
        if len(versions) != 1:
            raise ExportError("Inconsistent pipeline_version values within rows")

        col_rev = revisions[0]
        col_ver = versions[0]

        if input_dataset_revision is not None and col_rev != input_dataset_revision:
            raise ExportError(
                f"Row revision '{col_rev}' does not match metadata "
                f"'{input_dataset_revision}'"
            )
        if pipeline_version is not None and col_ver != pipeline_version:
            raise ExportError(
                f"Row version '{col_ver}' does not match metadata '{pipeline_version}'"
            )

        input_dataset_revision = col_rev
        pipeline_version = col_ver
    else:
        if input_dataset_revision is None or pipeline_version is None:
            raise ExportError(
                "Empty dataset must contain revision and version in schema metadata"
            )

    # 4. Reject existing non-empty output directory unless overwrite=True
    #    Reject non-directory target regardless of overwrite
    output_path = Path(output_dir).resolve()
    if output_path.exists():
        if not output_path.is_dir():
            raise ExportError(
                f"Target path exists and is not a directory: {output_path}"
            )
        if any(output_path.iterdir()) and not overwrite:
            raise ExportError(
                f"Output directory exists and is not empty: {output_path}"
            )

    # 5. Atomic write and rename
    parent_dir = output_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir: Path | None = None
    backup_dir: Path | None = None
    try:
        tmp_dir = Path(tempfile.mkdtemp(dir=parent_dir))
        pq_path = tmp_dir / "sentences.parquet"
        pq.write_table(dataset.table, pq_path)

        # Calculate SHA-256 checksum of Parquet
        sha256_hex = sha256_file(pq_path)

        manifest_data = build_manifest_data(
            dataset,
            input_dataset_revision,
            pipeline_version,
            sha256_hex,
        )

        manifest_path = tmp_dir / "manifest.json"
        write_manifest(manifest_path, manifest_data)

        # Rollback-safe directory swap
        backup_dir = install_atomic(tmp_dir, output_path)

        # 5. Remove the backup only after successful replacement.
        if backup_dir is not None and backup_dir.exists():
            remove_backup(backup_dir)
    except Exception:
        cleanup_on_failure(tmp_dir, backup_dir)
        raise

    return ExportResult(
        parquet_path=output_path / "sentences.parquet",
        manifest_path=output_path / "manifest.json",
        manifest_data=manifest_data,
    )
