"""Deterministic dataset-card statistics.

This module turns a finalized output table into a small, fully-derived set
of immutable statistics. It contains no network calls and never mutates the
filesystem; the exporter owns all writes.

Design
------

- Statistics are computed **only** from the exact finalized Arrow table
  being exported (plus the already-computed Parquet SHA-256), never from
  free text.
- The same ``DatasetStatistics`` object is serialized into the manifest's
  versioned ``statistics`` field and rendered into the card, so the card
  and manifest can never disagree. The manifest's top-level count fields
  derive from the same instance (no duplicated statistics).
- Breakdowns are stored as sorted mappings so serialization and rendering
  are order-independent of input row order.
- Rendering is deterministic: identical statistics produce byte-identical
  cards. The validator recomputes statistics from Parquet and rejects any
  card or manifest that does not equal the deterministic render/values.

Null-handling rules (see ``compute_statistics``):

- ``wikidata`` is non-nullable in ``OUTPUT_SENTENCE_SCHEMA`` but counted
  defensively: only non-null distinct values contribute.
- A row "has coordinates" iff **both** ``lat`` and ``lon`` are non-null.
- ``document_id`` is **not** guaranteed globally unique across
  ``(source, site, language)`` by the input contract; ``unique_documents``
  counts distinct ``(source, site, language, document_id)`` tuples.

Strict deserialization
----------------------

``statistics_from_dict`` rejects coerced values, unknown keys, negative
counts, breaking-accounting identities, malformed SHA-256, and blank
revision/version strings. ``ValueError`` instances describe each failure
class so the validator can wrap them with the original cause preserved.
"""

from __future__ import annotations

import os
import re
import sqlite3
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.contracts.schemas import OUTPUT_SENTENCE_SCHEMA

# Schema version of the versioned statistics object stored in the
# manifest. This initial schema version starts at 1.
# Bump only when the set of required fields or their invariants change
# incompatibly; the validator pins against this constant.
STATISTICS_VERSION = 1

# Required-keys mapping for the statistics object. Strict: unknown keys
# are rejected, exact types are enforced, and accounting invariants
# must hold. Every key on this tuple is required for the v1 schema.
# ``input_dataset_id`` is always present in serializations; its value
# may be ``None`` to represent a local build that did not record a
# Hub identity.
_STATISTICS_KEYS: tuple[str, ...] = (
    "version",
    "row_count",
    "unique_sentence_ids",
    "unique_polygons",
    "unique_wikidata_entities",
    "unique_documents",
    "source_counts",
    "language_counts",
    "region_counts",
    "rows_with_coordinates",
    "rows_without_coordinates",
    "input_dataset_revision",
    "pipeline_version",
    "parquet_sha256",
    "input_dataset_id",
)

# File name established by the exporter contract.
_CARD_NAME = "README.md"

# Provenance / coordinate-related column names (kept local to avoid
# importing the whole schema module here unless needed).
_COL_SENTENCE_ID = "sentence_id"
_COL_POLYGON_ID = "polygon_id"
_COL_WIKIDATA = "wikidata"
_COL_DOCUMENT_ID = "document_id"
_COL_SOURCE = "source"
_COL_LANGUAGE = "language"
_COL_REGION = "region"
_COL_LAT = "lat"
_COL_LON = "lon"
_COL_SITE = "site"

# Lowercase 64-character hexadecimal SHA-256.
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    """Immutable, fully-derived statistics for one exported dataset.

    Every field is computed from the finalized table (plus the Parquet
    SHA-256). ``version`` lets the validator reject manifests that were
    produced by an incompatible statistics schema. The three breakdown
    mappings (``source_counts``, ``language_counts``, ``region_counts``)
    are stored as :class:`types.MappingProxyType` views so callers cannot
    mutate them after construction; this is enforced alongside the
    frozen dataclass so both attribute assignment and dictionary mutation
    are forbidden.
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
    input_dataset_revision: str
    pipeline_version: str
    parquet_sha256: str
    input_dataset_id: str | None = None

    def __post_init__(self) -> None:
        # Defensively copy and wrap the three breakdown mappings in
        # ``MappingProxyType`` so the public interface is read-only.
        # ``dict(value)`` always copies, so an already-wrapped input is
        # unpacked from its backing dict before re-wrapping. Without
        # that copy, a caller that retains the backing dict can mutate
        # the stored mapping indirectly via the original proxy.
        for field_name in ("source_counts", "language_counts", "region_counts"):
            value = getattr(self, field_name)
            object.__setattr__(self, field_name, MappingProxyType(dict(value)))

    def __eq__(self, other: object) -> bool:
        # Compare the three breakdown mappings by content (rather than by
        # proxy identity) so two statistics built from equal dicts are
        # equal regardless of whether one was constructed with a raw
        # dict and the other via deserialization.
        if not isinstance(other, DatasetStatistics):
            return NotImplemented
        return (
            self.version == other.version
            and self.row_count == other.row_count
            and self.unique_sentence_ids == other.unique_sentence_ids
            and self.unique_polygons == other.unique_polygons
            and self.unique_wikidata_entities == other.unique_wikidata_entities
            and self.unique_documents == other.unique_documents
            and dict(self.source_counts) == dict(other.source_counts)
            and dict(self.language_counts) == dict(other.language_counts)
            and dict(self.region_counts) == dict(other.region_counts)
            and self.rows_with_coordinates == other.rows_with_coordinates
            and self.rows_without_coordinates == other.rows_without_coordinates
            and self.input_dataset_revision == other.input_dataset_revision
            and self.pipeline_version == other.pipeline_version
            and self.parquet_sha256 == other.parquet_sha256
            and self.input_dataset_id == other.input_dataset_id
        )

    def __hash__(self) -> int:
        # Disable hashing: the proxy mappings are unhashable.
        raise TypeError("DatasetStatistics is not hashable")


def _sorted_dict(mapping: dict[str, int]) -> dict[str, int]:
    """Return a new dict sorted by key for deterministic serialization."""
    return {k: mapping[k] for k in sorted(mapping)}


_INPUT_DATASET_ID_KEY = b"input_dataset_id"


def _resolve_input_dataset_id(
    table: pa.Table,
    explicit: str | None,
) -> str | None:
    """Resolve the upstream dataset identifier for ``compute_statistics``.

    The single source of truth is the Parquet schema metadata key
    ``b"input_dataset_id"`` written by the finalizer. Contract:

    - Metadata key absent → ``None`` (local mode).
    - Metadata key present → must decode as UTF-8; the decoded string
      must be non-blank after ``str.strip()`` (so a blank or
      whitespace-only value is rejected) and must NOT carry
      surrounding whitespace (the resolver rejects, never
      normalizes).
    - The stored value is preserved byte-for-byte; the only
      normalization step is the non-blank / no-surrounding-whitespace
      validation, both of which raise ``ValueError`` on violation.
    - When the caller passes an explicit non-``None`` value, it must
      agree byte-for-byte with the metadata (``ValueError``
      otherwise); equal-but-mutated forms are rejected so an upstream
      caller cannot smuggle a different identifier through the same
      metadata key.

    ``UnicodeDecodeError`` is wrapped with the originating exception
    preserved as ``__cause__`` for malformed metadata.
    """
    meta = table.schema.metadata or {}
    raw = meta.get(_INPUT_DATASET_ID_KEY)
    meta_value: str | None
    if raw is None:
        meta_value = None
    else:
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as err:
            raise ValueError(
                "Parquet metadata input_dataset_id is not valid UTF-8"
            ) from err
        if not decoded.strip():
            raise ValueError("Parquet metadata input_dataset_id cannot be blank")
        if decoded != decoded.strip():
            raise ValueError(
                "Parquet metadata input_dataset_id has surrounding "
                "whitespace; surrounding whitespace is rejected, not "
                "silently normalized"
            )
        meta_value = decoded
    if explicit is None:
        return meta_value
    if explicit != meta_value:
        raise ValueError(
            "input_dataset_id passed to compute_statistics "
            f"({explicit!r}) disagrees with Parquet metadata "
            f"({meta_value!r})"
        )
    return explicit


def compute_statistics(
    table: pa.Table,
    *,
    input_dataset_revision: str,
    pipeline_version: str,
    parquet_sha256: str,
    input_dataset_id: str | None = None,
) -> DatasetStatistics:
    """Compute immutable statistics directly from the finalized table.

    Parameters
    ----------
    table : pa.Table
        The finalized output table conforming to ``OUTPUT_SENTENCE_SCHEMA``.
    input_dataset_revision : str
        The resolved input dataset revision recorded in the export.
    pipeline_version : str
        The pipeline version recorded in the export.
    parquet_sha256 : str
        Lower-cased hex SHA-256 of the exported Parquet file (content
        identity for the card and manifest).
    input_dataset_id : str | None, default None
        The optional upstream dataset identifier recorded in the
        finalized Parquet schema metadata under the key
        ``b"input_dataset_id"``. When omitted, the value is read from
        the table's schema metadata directly so callers can pass
        ``table`` without re-stating the field. A blank metadata value
        is treated as "not recorded" (``None``). Pass ``None``
        explicitly only when the table genuinely has no metadata key.

    Returns
    -------
    DatasetStatistics
        All figures derived from ``table``; breakdowns sorted by key.
        ``unique_documents`` counts distinct
        ``(source, site, language, document_id)`` tuples.
    """
    row_count = table.num_rows

    def _distinct_non_null(column: str) -> set[str]:
        values = table.column(column).to_pylist()
        return {v for v in values if v is not None}

    def _counts(column: str) -> dict[str, int]:
        raw = table.column(column).to_pylist()
        aggregated: dict[str, int] = {}
        for value in raw:
            if value is None:
                continue
            aggregated[value] = aggregated.get(value, 0) + 1
        return _sorted_dict(aggregated)

    # Document identity: distinct (source, site, language, document_id)
    # tuples. ``document_id`` is not guaranteed globally unique across the
    # four provenance dimensions by the input contract; the tuple is.
    src = table.column(_COL_SOURCE).to_pylist()
    site = table.column(_COL_SITE).to_pylist()
    lang = table.column(_COL_LANGUAGE).to_pylist()
    doc = table.column(_COL_DOCUMENT_ID).to_pylist()
    document_identities: set[tuple[str, str, str, str]] = set()
    for s, st, lg, d in zip(src, site, lang, doc, strict=True):
        if d is None:
            continue
        document_identities.add((s, st, lg, d))

    # Coordinate presence: both lat and lon non-null.
    lat_col = table.column(_COL_LAT).to_pylist()
    lon_col = table.column(_COL_LON).to_pylist()
    coords_present = sum(
        1
        for lat, lon in zip(lat_col, lon_col, strict=True)
        if lat is not None and lon is not None
    )

    return DatasetStatistics(
        version=STATISTICS_VERSION,
        row_count=row_count,
        unique_sentence_ids=len(_distinct_non_null(_COL_SENTENCE_ID)),
        unique_polygons=len(_distinct_non_null(_COL_POLYGON_ID)),
        unique_wikidata_entities=len(_distinct_non_null(_COL_WIKIDATA)),
        unique_documents=len(document_identities),
        source_counts=_counts(_COL_SOURCE),
        language_counts=_counts(_COL_LANGUAGE),
        region_counts=_counts(_COL_REGION),
        rows_with_coordinates=coords_present,
        rows_without_coordinates=row_count - coords_present,
        input_dataset_revision=input_dataset_revision,
        pipeline_version=pipeline_version,
        input_dataset_id=_resolve_input_dataset_id(table, input_dataset_id),
        parquet_sha256=parquet_sha256,
    )


def compute_parquet_statistics(
    parquet_path: str | Path,
    *,
    input_dataset_revision: str,
    pipeline_version: str,
    parquet_sha256: str,
    input_dataset_id: str | None,
    scratch_dir: str | Path,
    batch_size: int = 65_536,
) -> DatasetStatistics:
    """Derive exact statistics from Parquet without loading it into memory.

    Record batches remain bounded. Distinct identities are held in a
    temporary on-disk SQLite database, and strict global ordering plus
    provenance equality are verified during the same scan.
    """

    path = Path(parquet_path)
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if (
        not isinstance(input_dataset_revision, str)
        or not input_dataset_revision.strip()
    ):
        raise ValueError("input_dataset_revision must be a non-blank string")
    if not isinstance(pipeline_version, str) or not pipeline_version.strip():
        raise ValueError("pipeline_version must be a non-blank string")

    descriptor, database_name = tempfile.mkstemp(
        prefix="dataset-statistics-", suffix=".sqlite3", dir=scratch
    )
    os.close(descriptor)
    database = Path(database_name)

    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(OUTPUT_SENTENCE_SCHEMA):
        raise ValueError("Parquet schema does not match OUTPUT_SENTENCE_SCHEMA")
    metadata = parquet.schema_arrow.metadata or {}
    try:
        stored_revision = metadata[b"input_dataset_revision"].decode("utf-8")
        stored_version = metadata[b"pipeline_version"].decode("utf-8")
    except KeyError as error:
        raise ValueError("Parquet schema metadata is incomplete") from error
    except UnicodeDecodeError as error:
        raise ValueError("Parquet provenance metadata is not valid UTF-8") from error
    if stored_revision != input_dataset_revision:
        raise ValueError("Parquet metadata input revision mismatch")
    if stored_version != pipeline_version:
        raise ValueError("Parquet metadata pipeline version mismatch")
    metadata_dataset_id = _resolve_input_dataset_id(
        pa.Table.from_batches([], schema=parquet.schema_arrow), input_dataset_id
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
        "input_dataset_revision",
        "pipeline_version",
    )
    source_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    region_counts: dict[str, int] = {}
    row_count = 0
    coordinates = 0
    previous_key: tuple[str, str, str] | None = None

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
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
            values = batch.to_pydict()
            sentence_ids: list[tuple[str]] = []
            polygons: list[tuple[str]] = []
            wikidata: list[tuple[str]] = []
            documents: list[tuple[str, str, str, str]] = []
            for index in range(batch.num_rows):
                sentence_id = values["sentence_id"][index]
                polygon_id = values["polygon_id"][index]
                language = values["language"][index]
                key = (polygon_id, language, sentence_id)
                if previous_key is not None and key <= previous_key:
                    raise ValueError("Parquet rows are not strictly globally sorted")
                previous_key = key
                if values["input_dataset_revision"][index] != input_dataset_revision:
                    raise ValueError("Parquet row input revision mismatch")
                if values["pipeline_version"][index] != pipeline_version:
                    raise ValueError("Parquet row pipeline version mismatch")

                source = values["source"][index]
                site = values["site"][index]
                document_id = values["document_id"][index]
                region = values["region"][index]
                source_counts[source] = source_counts.get(source, 0) + 1
                language_counts[language] = language_counts.get(language, 0) + 1
                region_counts[region] = region_counts.get(region, 0) + 1
                if (
                    values["lat"][index] is not None
                    and values["lon"][index] is not None
                ):
                    coordinates += 1
                sentence_ids.append((sentence_id,))
                polygons.append((polygon_id,))
                wikidata.append((values["wikidata"][index],))
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
                "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?)", documents
            )
            connection.commit()
            row_count += batch.num_rows

        def count(table: str) -> int:
            allowed = {"sentence_ids", "polygons", "wikidata", "documents"}
            if table not in allowed:  # pragma: no cover - internal invariant
                raise AssertionError("unexpected statistics table")
            result = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            assert result is not None
            return int(result[0])

        unique_sentence_ids = count("sentence_ids")
        if unique_sentence_ids != row_count:
            raise ValueError("Parquet contains duplicate sentence_id values")
        return DatasetStatistics(
            version=STATISTICS_VERSION,
            row_count=row_count,
            unique_sentence_ids=unique_sentence_ids,
            unique_polygons=count("polygons"),
            unique_wikidata_entities=count("wikidata"),
            unique_documents=count("documents"),
            source_counts=_sorted_dict(source_counts),
            language_counts=_sorted_dict(language_counts),
            region_counts=_sorted_dict(region_counts),
            rows_with_coordinates=coordinates,
            rows_without_coordinates=row_count - coordinates,
            input_dataset_revision=input_dataset_revision,
            pipeline_version=pipeline_version,
            parquet_sha256=parquet_sha256,
            input_dataset_id=metadata_dataset_id,
        )
    finally:
        connection.close()
        if database.exists():
            database.unlink()


def statistics_to_dict(stats: DatasetStatistics) -> dict[str, Any]:
    """Serialize a ``DatasetStatistics`` to an order-stable dict."""
    return {
        "version": stats.version,
        "row_count": stats.row_count,
        "unique_sentence_ids": stats.unique_sentence_ids,
        "unique_polygons": stats.unique_polygons,
        "unique_wikidata_entities": stats.unique_wikidata_entities,
        "unique_documents": stats.unique_documents,
        "source_counts": dict(stats.source_counts),
        "language_counts": dict(stats.language_counts),
        "region_counts": dict(stats.region_counts),
        "rows_with_coordinates": stats.rows_with_coordinates,
        "rows_without_coordinates": stats.rows_without_coordinates,
        "input_dataset_revision": stats.input_dataset_revision,
        "pipeline_version": stats.pipeline_version,
        "parquet_sha256": stats.parquet_sha256,
        # ``input_dataset_id`` is None for local builds; serialized as
        # ``null`` so the JSON shape is identical whether or not a Hub
        # identity was recorded.
        "input_dataset_id": stats.input_dataset_id,
    }


def statistics_from_dict(data: Any) -> DatasetStatistics:
    """Reconstruct a ``DatasetStatistics`` from a manifest ``statistics`` dict.

    Strict: unknown keys are rejected, no coercion is applied, all numeric
    fields must be plain (non-bool, non-numeric-string, non-negative)
    Python ``int`` at their declared precision, mappings must use string
    keys with non-negative Python ``int`` values, revision/version must be
    non-blank, ``parquet_sha256`` must be a lowercase 64-character hex
    string, and the breaking-accounting invariants
    ``rows_with_coordinates + rows_without_coordinates == row_count`` and
    ``sum(counts) == row_count`` (for each of ``source_counts`` /
    ``language_counts`` / ``region_counts``) must hold. Unique counts may
    not exceed the row count.

    Raises
    ------
    ValueError
        For each classification of bad input.
    """
    if not isinstance(data, dict):
        raise ValueError("statistics object must be a JSON object")

    unknown = set(data) - set(_STATISTICS_KEYS)
    if unknown:
        raise ValueError(
            "statistics object has unknown keys: " + ", ".join(sorted(unknown))
        )
    missing = [k for k in _STATISTICS_KEYS if k not in data]
    if missing:
        raise ValueError(f"statistics object missing keys: {missing}")

    def _require_input_dataset_id(raw: object) -> str | None:
        """Validate the v1 ``input_dataset_id`` field.

        The key is always present in v1 statistics. Its value may be
        ``None`` (local build) or a non-blank string (Hub build).
        Other types, blank strings, or values with surrounding whitespace
        are rejected. The value is preserved exactly as supplied.
        """
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise ValueError("statistics.input_dataset_id must be a string or null")
        if not raw.strip():
            raise ValueError("statistics.input_dataset_id cannot be blank")
        if raw != raw.strip():
            raise ValueError(
                "statistics.input_dataset_id has surrounding whitespace; "
                "surrounding whitespace is rejected, not silently normalized"
            )
        return raw

    def _strict_int(key: str) -> int:
        value = data[key]
        # ``type(value) is int`` (not ``isinstance``) rejects bool exactly.
        if type(value) is not int:
            raise ValueError(
                f"statistics.{key} must be an integer (got {type(value).__name__})"
            )
        if value < 0:
            raise ValueError(
                f"statistics.{key} must be a non-negative integer (got {value})"
            )
        return value

    def _nonblank_str(key: str) -> str:
        value = data[key]
        if not isinstance(value, str):
            raise ValueError(f"statistics.{key} must be a string")
        if not value.strip():
            raise ValueError(f"statistics.{key} must be a non-blank string")
        return value

    def _counts(key: str) -> dict[str, int]:
        value = data[key]
        if not isinstance(value, dict):
            raise ValueError(f"statistics.{key} must be a JSON object")
        out: dict[str, int] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"statistics.{key} keys must be strings (got {type(k).__name__})"
                )
            if type(v) is not int or v < 0:
                raise ValueError(
                    f"statistics.{key} values must be non-negative integers"
                )
            out[k] = v
        return _sorted_dict(out)

    version = _strict_int("version")
    if version != STATISTICS_VERSION:
        raise ValueError(
            f"statistics.version {version!r} does not match the supported "
            f"version {STATISTICS_VERSION}"
        )

    row_count = _strict_int("row_count")
    rows_with_coords = _strict_int("rows_with_coordinates")
    rows_without_coords = _strict_int("rows_without_coordinates")
    if rows_with_coords + rows_without_coords != row_count:
        raise ValueError(
            "statistics accounting identity violated: "
            "rows_with_coordinates + rows_without_coordinates "
            f"({rows_with_coords + rows_without_coords}) != "
            f"row_count ({row_count})"
        )

    source_counts = _counts("source_counts")
    language_counts = _counts("language_counts")
    region_counts = _counts("region_counts")
    if sum(source_counts.values()) != row_count:
        raise ValueError(
            f"statistics.source_counts sums to "
            f"{sum(source_counts.values())} but row_count is {row_count}"
        )
    if sum(language_counts.values()) != row_count:
        raise ValueError(
            f"statistics.language_counts sums to "
            f"{sum(language_counts.values())} but row_count is {row_count}"
        )
    if sum(region_counts.values()) != row_count:
        raise ValueError(
            f"statistics.region_counts sums to "
            f"{sum(region_counts.values())} but row_count is {row_count}"
        )

    unique_fields = (
        "unique_sentence_ids",
        "unique_polygons",
        "unique_wikidata_entities",
        "unique_documents",
    )
    unique_counts: dict[str, int] = {}
    for key in unique_fields:
        count = _strict_int(key)
        if count > row_count:
            raise ValueError(
                f"statistics.{key} ({count}) cannot exceed row_count ({row_count})"
            )
        unique_counts[key] = count

    parquet_sha256 = _nonblank_str("parquet_sha256")
    if not _SHA256_PATTERN.match(parquet_sha256):
        raise ValueError(
            "statistics.parquet_sha256 must be a lowercase 64-character hex string"
        )

    return DatasetStatistics(
        version=version,
        row_count=row_count,
        unique_sentence_ids=unique_counts["unique_sentence_ids"],
        unique_polygons=unique_counts["unique_polygons"],
        unique_wikidata_entities=unique_counts["unique_wikidata_entities"],
        unique_documents=unique_counts["unique_documents"],
        source_counts=source_counts,
        language_counts=language_counts,
        region_counts=region_counts,
        rows_with_coordinates=rows_with_coords,
        rows_without_coordinates=rows_without_coords,
        input_dataset_revision=_nonblank_str("input_dataset_revision"),
        pipeline_version=_nonblank_str("pipeline_version"),
        parquet_sha256=parquet_sha256,
        input_dataset_id=_require_input_dataset_id(data.get("input_dataset_id")),
    )
