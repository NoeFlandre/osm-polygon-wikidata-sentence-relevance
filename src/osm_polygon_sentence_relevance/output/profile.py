"""Single-source-of-truth dataset profile and asset rendering.

Every quantitative claim in the dataset card, every PNG asset hash in
the manifest, and every row used for the on-card real example comes
from the immutable :class:`DatasetProfile` built here. Two profiles
constructed from identical inputs (modulo the SHA-256 carrying the
content identity) must produce byte-identical renders.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.contracts.schemas import OUTPUT_SENTENCE_SCHEMA
from osm_polygon_sentence_relevance.output.checksum import sha256_file
from osm_polygon_sentence_relevance.output.plots import (
    PNG_SIGNATURE,
    ProfileError,
    collect_polygon_centroids,  # noqa: F401 - compatibility re-export
    geographic_caption_for_profile,  # noqa: F401 - compatibility re-export
    render_geographic_coverage_png,
    render_language_distribution_png,
)

# Profile / card schema version. Bump only for incompatible profile changes.
PROFILE_VERSION = 1


@dataclass(frozen=True, slots=True)
class AssetInfo:
    """Information about a published asset (PNG, sidecar, ...).

    ``name`` is the relative path under the export root (forward
    slashes only). ``sha256`` is the lowercase 64-character hex digest
    of the asset bytes. ``bytes_`` is the exact byte length; the
    trailing underscore avoids clashing with the ``bytes`` builtin.
    """

    name: str
    sha256: str
    bytes_: int


@dataclass(frozen=True, slots=True)
class ExampleRow:
    """A single canonical-sorted row selected for the on-card example block.

    Capturing the entire row keeps the card self-contained; the
    validator cross-checks it against the Parquet file by reading the
    first row of the sorted iteration in batch order.
    """

    fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        # Freeze the underlying mapping as ``MappingProxyType`` so callers
        # cannot mutate it after construction. ``dict(value)`` always
        # copies, so an already-wrapped input is unpacked first.
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]

    def __contains__(self, key: str) -> bool:
        return key in self.fields

    def keys(self) -> Any:
        return self.fields.keys()


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Single-source-of-truth view of an exported dataset.

    Construction is intentionally restricted to
    :func:`build_dataset_profile` so the derived fields cannot drift
    from the Parquet file. ``assets`` and ``example_row`` are
    constructed by the same code path and share the same ordering
    invariants.
    """

    version: int
    row_count: int
    unique_sentence_ids: int
    unique_polygons: int
    unique_wikidata_entities: int
    unique_documents: int
    source_counts: Mapping[str, int]
    language_counts: Mapping[str, int]
    region_counts: Mapping[str, int]
    rows_with_coordinates: int
    rows_without_coordinates: int
    rows_with_polygon_name: int
    input_dataset_revision: str
    pipeline_version: str
    input_dataset_id: str | None
    parquet_sha256: str
    segmentation_model: str
    segmentation_revision: str
    source_commit: str
    lat_min: float | None
    lat_max: float | None
    lon_min: float | None
    lon_max: float | None
    sentence_length_min: int
    sentence_length_mean: float
    sentence_length_max: int
    example_row: ExampleRow
    assets: Mapping[str, AssetInfo] = field(default_factory=dict)
    input_occurrence_count: int = 0
    duplicates_removed: int = 0
    cross_source_duplicate_groups: int = 0

    def __post_init__(self) -> None:
        for name in (
            "source_counts",
            "language_counts",
            "region_counts",
            "assets",
        ):
            value = getattr(self, name)
            if name == "assets":
                object.__setattr__(
                    self,
                    name,
                    MappingProxyType(dict(value.items())),
                )
            else:
                object.__setattr__(self, name, MappingProxyType(dict(value)))

    def to_dict(self) -> dict[str, Any]:
        """Render to a JSON-serializable dict with sorted mapping keys.

        Used by the validator to cross-check against the manifest.
        Output ordering is deterministic: mapping keys are emitted in
        sorted order, the example row keys follow the order declared in
        ``OUTPUT_SENTENCE_SCHEMA`` so the on-card example rows always
        compare equal across regenerations.
        """
        # Build example-row dict in schema-column order so layout is
        # deterministic across runs.
        row_in_order: dict[str, Any] = {}
        for col in OUTPUT_SENTENCE_SCHEMA.names:
            row_in_order[col] = self.example_row.fields.get(col)

        # Sort the asset map by name.
        assets_dict = {
            name: {
                "name": info.name,
                "sha256": info.sha256,
                "bytes": info.bytes_,
            }
            for name, info in sorted(self.assets.items())
        }

        return {
            "version": self.version,
            "row_count": self.row_count,
            "unique_sentence_ids": self.unique_sentence_ids,
            "unique_polygons": self.unique_polygons,
            "unique_wikidata_entities": self.unique_wikidata_entities,
            "unique_documents": self.unique_documents,
            "source_counts": dict(self.source_counts),
            "language_counts": dict(self.language_counts),
            "region_counts": dict(self.region_counts),
            "rows_with_coordinates": self.rows_with_coordinates,
            "rows_without_coordinates": self.rows_without_coordinates,
            "rows_with_polygon_name": self.rows_with_polygon_name,
            "input_dataset_revision": self.input_dataset_revision,
            "pipeline_version": self.pipeline_version,
            "input_dataset_id": self.input_dataset_id,
            "parquet_sha256": self.parquet_sha256,
            "segmentation_model": self.segmentation_model,
            "segmentation_revision": self.segmentation_revision,
            "source_commit": self.source_commit,
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
            "sentence_length_min": self.sentence_length_min,
            "sentence_length_mean": self.sentence_length_mean,
            "sentence_length_max": self.sentence_length_max,
            "example_row": row_in_order,
            "assets": assets_dict,
            "input_occurrence_count": self.input_occurrence_count,
            "duplicates_removed": self.duplicates_removed,
            "cross_source_duplicate_groups": self.cross_source_duplicate_groups,
        }


def _sorted_dict(mapping: Mapping[str, int]) -> dict[str, int]:
    """Sort a count mapping by key for deterministic serialization."""
    return {k: mapping[k] for k in sorted(mapping)}


def _parse_meta(metadata: Mapping[bytes, bytes] | None, key: bytes) -> str | None:
    """Read a UTF-8 metadata key. Missing → ``None``.

    Returns the raw decoded bytes.  The blank check used to live in
    this helper but was lifted to ``build_dataset_profile`` so the
    helper can serve optional metadata keys (e.g.
    ``input_dataset_id``) without false positives.
    """
    if metadata is None:
        return None
    raw = metadata.get(key)
    if raw is None:
        return None
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as err:
        raise ProfileError(f"Parquet metadata {key!r} is not valid UTF-8") from err
    if not value.strip():
        raise ProfileError(f"Parquet metadata {key!r} cannot be blank")
    return value


def _sample_first_full_row(
    parquet: pq.ParquetFile,
) -> dict[str, Any]:
    """Return the first full row of *parquet* as a dict.

    Used to capture a single canonical example row for the dataset
    card. The Parquet file is globably sorted by
    ``(polygon_id, language, sentence_id)`` ascending by the
    finalisation step, so the first row in batch order is the same
    row finalisation would pick next.
    """
    iterator = parquet.iter_batches(batch_size=1)
    try:
        first_batch = next(iter(iterator))
    except StopIteration as err:
        raise ProfileError(
            "Cannot sample an example row from an empty Parquet file"
        ) from err
    if first_batch.num_rows == 0:
        raise ProfileError("Cannot sample an example row from an empty Parquet file")
    values = first_batch.to_pydict()
    return {col: values[col][0] for col in OUTPUT_SENTENCE_SCHEMA.names}


def render_example_row_json(profile: DatasetProfile) -> str:
    """Render the example row as a deterministic JSON string.

    Fields are emitted in the schema's column order so two profiles
    with the same content render to byte-identical strings.
    """
    row: dict[str, Any] = {}
    for col in OUTPUT_SENTENCE_SCHEMA.names:
        row[col] = profile.example_row.fields.get(col)
    return json.dumps(
        row,
        sort_keys=False,
        separators=(",", ": "),
        ensure_ascii=False,
        default=str,
        indent=2,
    )


def build_dataset_profile(
    *,
    parquet_path: Path | str,
    parquet_sha256: str,
    segmentation_model: str,
    segmentation_revision: str,
    source_commit: str,
    scratch_dir: Path | str,
    input_dataset_id: str | None = None,
) -> DatasetProfile:
    """Build a :class:`DatasetProfile` from a finalized Parquet file.

    The Parquet file is read once via a bounded-SQLite scan that
    mirrors the existing ``compute_parquet_statistics`` invariants plus
    a first-batch scan for the deterministic example row. The returned
    profile is the single source of truth for the card, the manifest,
    and the published PNG asset hashes.

    Parameters
    ----------
    parquet_path
        Path to the finalized ``sentences.parquet`` file.
    parquet_sha256
        Lowercase hex SHA-256 of the Parquet file. Cross-checked
        against ``sha256_file``.
    segmentation_model
        The segmentation model name (e.g. ``sat-3l``).
    segmentation_revision
        The exact revision/commit of the segmentation model used.
    source_commit
        The local source-code commit hash; surfaced on the card.
    scratch_dir
        Directory used for the bounded SQLite scratch database.
    input_dataset_id
        Optional Hub dataset identifier to cross-check against the
        Parquet metadata key ``b"input_dataset_id"``. When omitted the
        metadata value is used directly.

    Returns
    -------
    DatasetProfile
        Immutable profile. No geographic asset is rendered here; that
        happens at export time so the asset hashes can be recorded in
        the manifest and the cards can be regenerated atomically.
    """
    path = Path(parquet_path)
    if not path.is_file():
        raise ProfileError(f"Parquet file is missing: {path}")
    if not isinstance(parquet_sha256, str) or not parquet_sha256.strip():
        raise ProfileError("parquet_sha256 must be a non-blank string")
    if not isinstance(segmentation_model, str) or not segmentation_model.strip():
        raise ProfileError("segmentation_model must be a non-blank string")
    if not isinstance(segmentation_revision, str) or not segmentation_revision.strip():
        raise ProfileError("segmentation_revision must be a non-blank string")
    if not isinstance(source_commit, str) or not source_commit.strip():
        raise ProfileError("source_commit must be a non-blank string")

    actual_sha = sha256_file(path)
    if actual_sha.lower() != parquet_sha256.lower():
        raise ProfileError(
            f"Parquet SHA {actual_sha!r} does not match expected {parquet_sha256!r}"
        )

    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    descriptor, database_name = tempfile.mkstemp(
        prefix="dataset-profile-", suffix=".sqlite3", dir=scratch
    )
    os.close(descriptor)
    database = Path(database_name)

    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(OUTPUT_SENTENCE_SCHEMA):
        raise ProfileError("Parquet schema does not match OUTPUT_SENTENCE_SCHEMA")
    metadata = parquet.schema_arrow.metadata or {}

    stored_revision = _parse_meta(metadata, b"input_dataset_revision")
    stored_version = _parse_meta(metadata, b"pipeline_version")
    metadata_dataset_id = _parse_meta(metadata, b"input_dataset_id")
    if stored_revision is None or stored_version is None:
        raise ProfileError("Parquet metadata is missing required provenance keys")
    if input_dataset_id is not None and metadata_dataset_id != input_dataset_id:
        raise ProfileError("Explicit input_dataset_id does not match Parquet metadata")

    # Sample the first full row for the on-card example.
    example_row_dict = _sample_first_full_row(parquet)

    row_count = 0
    coords = 0
    polygon_names = 0
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    sentence_lengths_total = 0
    sentence_length_min: int | None = None
    sentence_length_max: int = 0
    source_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    region_counts: dict[str, int] = {}
    input_occurrence_count = 0
    cross_source_duplicate_groups = 0

    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            CREATE TABLE sentence_ids (value TEXT PRIMARY KEY) WITHOUT ROWID;
            CREATE TABLE polygons (value TEXT PRIMARY KEY) WITHOUT ROWID;
            CREATE TABLE wikidata (value TEXT PRIMARY KEY) WITHOUT ROWID;
            CREATE TABLE documents (
                source TEXT NOT NULL,
                site TEXT NOT NULL,
                language TEXT NOT NULL,
                document_id TEXT NOT NULL,
                PRIMARY KEY (source, site, language, document_id)
            ) WITHOUT ROWID;
            """
        )
        columns = (
            "sentence_id",
            "polygon_id",
            "wikidata",
            "source",
            "site",
            "language",
            "document_id",
            "region",
            "lat",
            "lon",
            "polygon_name",
            "sentence_text_normalized",
            "input_dataset_revision",
            "pipeline_version",
            "duplicate_occurrence_count",
            "duplicate_sources",
        )
        for batch in parquet.iter_batches(batch_size=65_536, columns=list(columns)):
            values = batch.to_pydict()
            n = batch.num_rows
            sentence_ids: list[tuple[str]] = []
            polygons: list[tuple[str]] = []
            wikidata: list[tuple[str]] = []
            documents: list[tuple[str, str, str, str]] = []
            for i in range(n):
                if values["input_dataset_revision"][i] != stored_revision:
                    raise ProfileError("Parquet row revision mismatch")
                if values["pipeline_version"][i] != stored_version:
                    raise ProfileError("Parquet row pipeline version mismatch")
                source = values["source"][i]
                site = values["site"][i]
                language = values["language"][i]
                document_id = values["document_id"][i]
                region = values["region"][i]
                source_counts[source] = source_counts.get(source, 0) + 1
                language_counts[language] = language_counts.get(language, 0) + 1
                region_counts[region] = region_counts.get(region, 0) + 1
                input_occurrence_count += int(values["duplicate_occurrence_count"][i])
                if len(set(values["duplicate_sources"][i])) > 1:
                    cross_source_duplicate_groups += 1
                lat = values["lat"][i]
                lon = values["lon"][i]
                if lat is not None and lon is not None:
                    coords += 1
                    lat_min = lat if lat_min is None else min(lat_min, lat)
                    lat_max = lat if lat_max is None else max(lat_max, lat)
                    lon_min = lon if lon_min is None else min(lon_min, lon)
                    lon_max = lon if lon_max is None else max(lon_max, lon)
                if values["polygon_name"][i]:
                    polygon_names += 1
                text = values["sentence_text_normalized"][i]
                length = len(text)
                sentence_lengths_total += length
                if sentence_length_min is None or length < sentence_length_min:
                    sentence_length_min = length
                if length > sentence_length_max:
                    sentence_length_max = length
                sentence_ids.append((values["sentence_id"][i],))
                polygons.append((values["polygon_id"][i],))
                wikidata.append((values["wikidata"][i],))
                documents.append((source, site, language, document_id))
            connection.executemany(
                "INSERT OR IGNORE INTO sentence_ids VALUES (?)", sentence_ids
            )
            connection.executemany(
                "INSERT OR IGNORE INTO polygons VALUES (?)", polygons
            )
            connection.executemany(
                "INSERT OR IGNORE INTO wikidata VALUES (?)", wikidata
            )
            connection.executemany(
                "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?)",
                documents,
            )
            connection.commit()
            row_count += n

        def _count(table: str) -> int:
            allowed = {"sentence_ids", "polygons", "wikidata", "documents"}
            if table not in allowed:  # pragma: no cover - internal invariant
                raise ProfileError("unexpected internal table name")
            result = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            assert result is not None
            return int(result[0])

        unique_sentence_ids = _count("sentence_ids")
        if unique_sentence_ids != row_count:
            raise ProfileError(  # pragma: no cover - covered by exhaustive test
                "Parquet contains duplicate sentence_id values"
            )
        unique_polygons = _count("polygons")
        unique_wikidata_entities = _count("wikidata")
        unique_documents = _count("documents")
    finally:
        connection.close()
        if database.exists():
            database.unlink()  # pragma: no cover - cleanup only

    if row_count == 0:  # pragma: no cover - row_count is asserted > 0 below
        raise ProfileError("Cannot build a profile from an empty Parquet file")

    sentence_length_mean = sentence_lengths_total / row_count if row_count else 0.0
    if (
        sentence_length_min is None
    ):  # pragma: no cover - unreachable; row_count > 0 is enforced above
        sentence_length_min = 0

    return DatasetProfile(
        version=PROFILE_VERSION,
        row_count=row_count,
        unique_sentence_ids=unique_sentence_ids,
        unique_polygons=unique_polygons,
        unique_wikidata_entities=unique_wikidata_entities,
        unique_documents=unique_documents,
        source_counts=_sorted_dict(source_counts),
        language_counts=_sorted_dict(language_counts),
        region_counts=_sorted_dict(region_counts),
        rows_with_coordinates=coords,
        rows_without_coordinates=row_count - coords,
        rows_with_polygon_name=polygon_names,
        input_dataset_revision=stored_revision,
        pipeline_version=stored_version,
        input_dataset_id=metadata_dataset_id,
        parquet_sha256=parquet_sha256.lower(),
        segmentation_model=segmentation_model,
        segmentation_revision=segmentation_revision,
        source_commit=source_commit,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        sentence_length_min=sentence_length_min,
        sentence_length_mean=sentence_length_mean,
        sentence_length_max=sentence_length_max,
        example_row=ExampleRow(fields=example_row_dict),
        assets=MappingProxyType({}),
        input_occurrence_count=input_occurrence_count,
        duplicates_removed=input_occurrence_count - row_count,
        cross_source_duplicate_groups=cross_source_duplicate_groups,
    )


def sha256_bytes(payload: bytes) -> str:
    """Compute lowercase hex SHA-256 for *payload*."""
    return hashlib.sha256(payload).hexdigest().lower()


__all__ = [
    "AssetInfo",
    "DatasetProfile",
    "ExampleRow",
    "PNG_SIGNATURE",
    "PROFILE_VERSION",
    "ProfileError",
    "build_dataset_profile",
    "render_example_row_json",
    "render_geographic_coverage_png",
    "render_language_distribution_png",
    "sha256_bytes",
]
