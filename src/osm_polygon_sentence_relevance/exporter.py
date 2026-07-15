from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections import Counter
import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.finalization import FinalizedDataset
from osm_polygon_sentence_relevance.schemas import OUTPUT_SENTENCE_SCHEMA
from osm_polygon_sentence_relevance.errors import ExportError


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
        raise ExportError(
            "Table schema does not match expected OUTPUT_SENTENCE_SCHEMA"
        )

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
        revisions = (
            dataset.table.column("input_dataset_revision").unique().to_pylist()
        )
        versions = dataset.table.column("pipeline_version").unique().to_pylist()

        if len(revisions) != 1:
            raise ExportError(
                "Inconsistent input_dataset_revision values within rows"
            )
        if len(versions) != 1:
            raise ExportError("Inconsistent pipeline_version values within rows")

        col_rev = revisions[0]
        col_ver = versions[0]

        if (
            input_dataset_revision is not None
            and col_rev != input_dataset_revision
        ):
            raise ExportError(
                f"Row revision '{col_rev}' does not match metadata '{input_dataset_revision}'"
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
    # Reject non-directory target regardless of overwrite
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

    tmp_dir = None
    backup_dir = None
    try:
        tmp_dir = Path(tempfile.mkdtemp(dir=parent_dir))
        pq_path = tmp_dir / "sentences.parquet"
        pq.write_table(dataset.table, pq_path)

        # Calculate SHA-256 checksum of Parquet
        sha256_hash = hashlib.sha256()
        with open(pq_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256_hash.update(chunk)
        sha256_hex = sha256_hash.hexdigest().lower()

        # Compute row counts by source, language, and region
        if dataset.table.num_rows > 0:
            counts_by_source = dict(
                Counter(dataset.table.column("source").to_pylist())
            )
            counts_by_language = dict(
                Counter(dataset.table.column("language").to_pylist())
            )
            counts_by_region = dict(
                Counter(dataset.table.column("region").to_pylist())
            )
        else:
            counts_by_source = {}
            counts_by_language = {}
            counts_by_region = {}

        manifest_data = {
            "row_count": dataset.table.num_rows,
            "input_occurrence_count": (
                dataset.report.input_sentence_occurrence_count
                if dataset.report
                else 0
            ),
            "duplicates_removed": (
                dataset.report.duplicate_occurrence_count_removed
                if dataset.report
                else 0
            ),
            "cross_source_duplicate_groups": (
                dataset.report.cross_source_duplicate_group_count
                if dataset.report
                else 0
            ),
            "counts_by_source": counts_by_source,
            "counts_by_language": counts_by_language,
            "counts_by_region": counts_by_region,
            "input_dataset_revision": input_dataset_revision,
            "pipeline_version": pipeline_version,
            "sha256": sha256_hex,
        }

        manifest_path = tmp_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    manifest_data,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            )

        # Rollback-safe directory swap:
        # 1. Fully build and verify the temporary export (done above).
        # 2. Rename existing output to a temporary backup (if it exists).
        if output_path.exists():
            import uuid
            backup_dir = parent_dir / f".backup_{uuid.uuid4().hex}"
            os.rename(output_path, backup_dir)

        try:
            # 3. Rename the new directory into place.
            os.rename(tmp_dir, output_path)
        except Exception as rename_err:
            # 4. If step 3 fails, restore the backup.
            if backup_dir and backup_dir.exists():
                if output_path.exists():
                    if output_path.is_dir():
                        shutil.rmtree(output_path)
                    else:
                        os.remove(output_path)
                try:
                    os.rename(backup_dir, output_path)
                except Exception as restore_err:
                    saved_backup = backup_dir
                    backup_dir = None
                    raise ExportError(
                        f"Atomic replacement failed, and backup restoration also failed. "
                        f"Previous dataset is preserved at {saved_backup}"
                    ) from restore_err
            raise rename_err

        # 5. Remove the backup only after successful replacement.
        if backup_dir and backup_dir.exists():
            try:
                shutil.rmtree(backup_dir)
            except Exception as rmtree_err:
                raise ExportError(
                    f"New dataset successfully exported, but failed to delete backup directory: {backup_dir}"
                ) from rmtree_err
    except Exception:
        if tmp_dir is not None and tmp_dir.exists():
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
        if backup_dir is not None and backup_dir.exists():
            try:
                shutil.rmtree(backup_dir)
            except Exception:
                pass
        raise

    return ExportResult(
        parquet_path=output_path / "sentences.parquet",
        manifest_path=output_path / "manifest.json",
        manifest_data=manifest_data,
    )
