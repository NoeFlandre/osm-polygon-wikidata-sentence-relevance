"""Read-only validation of an exported dataset directory.

Confirms that an already-exported directory is internally consistent
(Parquet present, manifest present and well-formed, SHA-256 and row
count matching, schema equal to ``OUTPUT_SENTENCE_SCHEMA``, Parquet
schema metadata for ``input_dataset_revision`` / ``pipeline_version``
present, decodable, and cross-checked against the manifest, and all
quantitative top-level manifest fields equal those in the versioned
``statistics`` object) before any later publication/upload step is
allowed to touch it.

This module never mutates, repairs, rewrites, or deletes anything, and
performs no network access. To recompute the canonical statistics and
verify the checked-in dataset card, it reads the entire Parquet file
once via :func:`pyarrow.parquet.read_table`.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.contracts.errors import ExportError
from osm_polygon_sentence_relevance.contracts.schemas import OUTPUT_SENTENCE_SCHEMA
from osm_polygon_sentence_relevance.output.checksum import sha256_file
from osm_polygon_sentence_relevance.output.dataset_card import (
    compute_parquet_statistics,
    render_dataset_card,
    statistics_from_dict,
)

# File names established by the existing exporter contract.
_PARQUET_NAME = "sentences.parquet"
_MANIFEST_NAME = "manifest.json"
_CARD_NAME = "README.md"

# Required Parquet schema-metadata keys (UTF-8 values), also recorded in
# the manifest and cross-checked against it.
_REVISION_META = b"input_dataset_revision"
_VERSION_META = b"pipeline_version"
_DATASET_ID_META = b"input_dataset_id"


@dataclass(frozen=True, slots=True)
class ValidatedExport:
    """Verified facts about a validated export directory.

    All fields are confirmed by ``validate_export_directory`` before an
    instance is constructed; nothing here is taken on trust from the
    manifest alone except where it has been cross-checked against the
    Parquet file on disk.
    """

    export_dir: Path
    parquet_path: Path
    manifest_path: Path
    card_path: Path
    row_count: int
    sha256: str


def _decode_meta_value(raw: bytes | None, key: str) -> str:
    """Decode and validate a required Parquet schema-metadata value.

    The value must be present, decodable as UTF-8, and contain non-whitespace
    content. The original (untrimmed) value is returned so equality comparison
    is not silently normalized.
    """
    if raw is None:
        raise ExportError(f"Parquet schema metadata is missing {key!r}")
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as err:
        raise ExportError(
            f"Parquet schema metadata {key!r} is not valid UTF-8"
        ) from err
    if not value.strip():
        raise ExportError(f"Parquet schema metadata {key!r} is blank")
    return value


def _require_manifest_string(manifest: dict, key: str) -> str:
    """Require a string field with non-whitespace content from the manifest.

    The original (untrimmed) value is returned so equality comparison is not
    silently normalized.
    """
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExportError(f"Manifest field {key!r} must be a non-empty string")
    return value


def validate_export_directory(path: str | Path) -> ValidatedExport:
    """Validate an exported dataset directory without modifying it.

    Parameters
    ----------
    path : str | Path
        The exported dataset directory to validate.

    Returns
    -------
    ValidatedExport
        Verified facts (resolved paths, row count, Parquet SHA-256).

    Raises
    ------
    TypeError
        If *path* is not a string or :class:`~pathlib.Path`.
    ExportError
        If the path is not a directory, required files are missing, the
        manifest is malformed, or the checksum, row count, schema, or
        schema-metadata contracts are violated.
    """
    # 1. Argument-type validation before touching the filesystem.
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a str or pathlib.Path")

    export_dir = Path(path).resolve()

    # 2. Directory existence/type check (early rejection of non-directories).
    if not export_dir.is_dir():
        raise ExportError(f"Export path is not a directory: {export_dir}")

    parquet_path = export_dir / _PARQUET_NAME
    manifest_path = export_dir / _MANIFEST_NAME
    card_path = export_dir / _CARD_NAME

    # 3. Required-file presence.
    if not parquet_path.is_file():
        raise ExportError(f"Missing Parquet file {_PARQUET_NAME!r} in {export_dir}")
    if not manifest_path.is_file():
        raise ExportError(f"Missing manifest file {_MANIFEST_NAME!r} in {export_dir}")
    if not card_path.is_file():
        raise ExportError(f"Missing dataset card {_CARD_NAME!r} in {export_dir}")

    # 4. Defensive manifest parsing.
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
    except (OSError, UnicodeDecodeError) as err:
        raise ExportError(f"Manifest is not readable: {err}") from err
    except json.JSONDecodeError as err:
        raise ExportError(f"Manifest is not valid JSON: {err}") from err

    if not isinstance(manifest, dict):
        raise ExportError(
            f"Manifest must be a JSON object, got {type(manifest).__name__}"
        )

    # 5. Read Parquet schema/metadata for the schema-shape + cross-checks
    #    (no row groups yet); the full table is loaded later (step 11) to
    #    recompute statistics. Wrap the external-library boundary so corrupt
    #    files surface as actionable ExportError with cause preserved.
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        parquet_schema = parquet_file.schema_arrow
        num_rows = parquet_file.metadata.num_rows
        schema_metadata = parquet_schema.metadata
    except ExportError:
        raise
    except Exception as err:
        raise ExportError(
            f"Parquet file {_PARQUET_NAME!r} could not be read: {err}"
        ) from err

    # 6. Schema contract: exact physical field/type/nullability comparison
    #    against OUTPUT_SENTENCE_SCHEMA (the constant itself carries no
    #    metadata, so check_metadata is left at its default of False).
    if not parquet_schema.equals(OUTPUT_SENTENCE_SCHEMA):
        raise ExportError("Parquet schema does not match OUTPUT_SENTENCE_SCHEMA")

    # 7. Parquet schema-metadata contract: presence + decodability.
    if schema_metadata is None:
        raise ExportError("Parquet schema metadata is missing")
    parquet_revision = _decode_meta_value(
        schema_metadata.get(_REVISION_META), "input_dataset_revision"
    )
    parquet_version = _decode_meta_value(
        schema_metadata.get(_VERSION_META), "pipeline_version"
    )
    # ``input_dataset_id``: local mode is represented by an absent
    # metadata key. When the key IS present, the value must decode as
    # UTF-8 and be a non-blank string. Present-but-blank metadata is
    # rejected with ``ExportError``; the validator never silently
    # normalizes a blank value to ``None``.
    raw_dataset_id = schema_metadata.get(_DATASET_ID_META)
    parquet_dataset_id: str | None
    if raw_dataset_id is None:
        parquet_dataset_id = None
    else:
        try:
            decoded = raw_dataset_id.decode("utf-8")
        except UnicodeDecodeError as err:
            raise ExportError(
                "Parquet schema metadata 'input_dataset_id' is not valid UTF-8"
            ) from err
        if not decoded.strip():
            raise ExportError(
                "Parquet schema metadata 'input_dataset_id' cannot be blank"
            )
        if decoded != decoded.strip():
            raise ExportError(
                "Parquet schema metadata 'input_dataset_id' has "
                "surrounding whitespace; surrounding whitespace is "
                "rejected, not silently normalized"
            )
        parquet_dataset_id = decoded

    # 8. Checksum contract: manifest value vs computed file digest.
    manifest_sha = manifest.get("sha256")
    if not isinstance(manifest_sha, str):
        raise ExportError("Manifest is missing a string 'sha256' checksum field")
    try:
        actual_sha = sha256_file(parquet_path)
    except ExportError:
        raise
    except Exception as err:
        raise ExportError(f"Could not compute Parquet checksum: {err}") from err
    if manifest_sha.lower() != actual_sha:
        raise ExportError(
            f"Manifest checksum {manifest_sha!r} does not match "
            f"Parquet checksum {actual_sha!r}"
        )

    # 9. Row-count contract: manifest value vs Parquet metadata.
    manifest_rows = manifest.get("row_count")
    if not isinstance(manifest_rows, int) or isinstance(manifest_rows, bool):
        raise ExportError("Manifest is missing an integer 'row_count' field")
    if manifest_rows != num_rows:
        raise ExportError(
            f"Manifest row_count {manifest_rows} does not match "
            f"Parquet row count {num_rows}"
        )

    # 10. Manifest revision/version contract (non-empty strings) and
    #     cross-check against the Parquet schema metadata values.
    manifest_revision = _require_manifest_string(manifest, "input_dataset_revision")
    manifest_version = _require_manifest_string(manifest, "pipeline_version")
    manifest_dataset_id = manifest.get("input_dataset_id")
    if manifest_dataset_id is not None and (
        not isinstance(manifest_dataset_id, str) or not manifest_dataset_id.strip()
    ):
        raise ExportError(
            "Manifest 'input_dataset_id' must be a non-blank string or null"
        )
    if manifest_revision != parquet_revision:
        raise ExportError(
            f"Manifest input_dataset_revision {manifest_revision!r} does not match "
            f"Parquet metadata {parquet_revision!r}"
        )
    if manifest_version != parquet_version:
        raise ExportError(
            f"Manifest pipeline_version {manifest_version!r} does not match "
            f"Parquet metadata {parquet_version!r}"
        )
    if manifest_dataset_id != parquet_dataset_id:
        raise ExportError(
            f"Manifest input_dataset_id {manifest_dataset_id!r} does not match "
            f"Parquet metadata {parquet_dataset_id!r}"
        )

    # 11. Statistics contract: recompute from the Parquet table on disk,
    #     never trusting the manifest, and require byte-identical equality
    #     with the manifest's recorded statistics. This catches stale,
    #     hand-edited, or numerically inconsistent cards/statistics.
    raw_stats = manifest.get("statistics")
    if not isinstance(raw_stats, dict):
        raise ExportError("Manifest is missing a 'statistics' object")

    try:
        manifest_statistics = statistics_from_dict(raw_stats)
    except (ValueError, TypeError) as err:
        raise ExportError(f"Manifest statistics object is invalid: {err}") from err

    try:
        with tempfile.TemporaryDirectory(prefix="validate-export-") as scratch:
            recomputed = compute_parquet_statistics(
                parquet_path,
                input_dataset_revision=manifest_revision,
                pipeline_version=manifest_version,
                parquet_sha256=actual_sha,
                input_dataset_id=parquet_dataset_id,
                scratch_dir=scratch,
            )
    except Exception as err:
        raise ExportError(f"Could not validate Parquet rows: {err}") from err

    if recomputed != manifest_statistics:
        raise ExportError(
            "Recomputed statistics from Parquet do not match the manifest "
            "statistics; the export is stale or manually altered"
        )

    # 11b. Top-level-vs-statistics drift. The manifest exposes
    #      ``counts_by_*``, ``row_count``, ``sha256``, ``input_dataset_revision``,
    #      and ``pipeline_version`` at the top level for compatibility
    #      with existing consumers; the source of truth is the versioned
    #      ``statistics`` object. Both must agree, so a manifest where
    #      only the top-level field was altered is rejected here even if
    #      the statistics object is internally correct.
    drift_pairs = (
        ("counts_by_source", manifest_statistics.source_counts),
        ("counts_by_language", manifest_statistics.language_counts),
        ("counts_by_region", manifest_statistics.region_counts),
        ("row_count", manifest_statistics.row_count),
        ("sha256", manifest_statistics.parquet_sha256),
        (
            "input_dataset_revision",
            manifest_statistics.input_dataset_revision,
        ),
        ("pipeline_version", manifest_statistics.pipeline_version),
        ("input_dataset_id", manifest_statistics.input_dataset_id),
    )
    for field_name, expected in drift_pairs:
        if manifest.get(field_name) != expected:
            raise ExportError(
                f"Manifest top-level field {field_name!r} disagrees with "
                f"the statistics object; the export is stale or the "
                f"manifest was manually altered"
            )

    # 12. Dataset-card contract: the checked-in README must equal the
    #     deterministic rendering of the validated statistics.
    try:
        card_text = card_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        raise ExportError(f"Dataset card is not readable: {err}") from err

    expected_card = render_dataset_card(manifest_statistics)
    if card_text != expected_card:
        raise ExportError(
            "Dataset card does not match the deterministic rendering of the "
            "validated statistics; it is stale or manually edited"
        )

    return ValidatedExport(
        export_dir=export_dir,
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        card_path=card_path,
        row_count=num_rows,
        sha256=actual_sha,
    )


__all__ = ["ValidatedExport", "validate_export_directory"]
