"""Deterministic dataset-card generation (Phase 8C, amended).

This module turns a finalized output table into a small, fully-derived set
of immutable statistics and renders them into a Hugging Face-compatible
``README.md`` dataset card. It contains no network calls and never mutates
the filesystem; the exporter owns all writes.

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
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.contracts.schemas import OUTPUT_SENTENCE_SCHEMA

# Schema version of the versioned statistics object stored in the
# manifest. Phase 8C is the initial release; this constant starts at 1.
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


def _yaml_quote_scalar(value: str) -> str:
    """Quote a YAML scalar so special characters cannot break the sequence.

    Emits a double-quoted scalar with backslashes and double quotes
    escaped; non-printable characters are JSON-style escaped so the
    quoting is valid in standard YAML 1.2 libraries as well as in the
    Hugging Face dataset-card parser. Newlines are escaped as ``\\n``
    rather than emitted literally so the rendering stays single-line per
    YAML sequence entry.
    """
    safe = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\x00", "\\0")
        .replace("\x0b", "\\v")
        .replace("\x0c", "\\f")
    )
    return f'"{safe}"'


def _yaml_features_block(stats: DatasetStatistics) -> str:
    """Render the ``dataset_info.features`` block for the YAML front matter.

    The schema is declared in deterministic order so two ``DatasetStatistics``
    inputs with the same logical content produce byte-identical YAML.
    ``osm_tags`` is declared as a ``Sequence`` of ``{key, value}`` structs
    because the Hugging Face Dataset Viewer cannot process the parquet
    ``map<string, string>`` type; the parquet file is encoded to use the
    same logical representation so the declared features match the actual
    bytes.
    """
    feature_lines = [
        "  - name: sentence_id",
        "    dtype: string",
        "  - name: polygon_id",
        "    dtype: string",
        "  - name: wikidata",
        "    dtype: string",
        "  - name: document_id",
        "    dtype: string",
        "  - name: article_id",
        "    dtype: string",
        "  - name: source",
        "    dtype: string",
        "  - name: language",
        "    dtype: string",
        "  - name: site",
        "    dtype: string",
        "  - name: page_title",
        "    dtype: string",
        "  - name: section_id",
        "    dtype: string",
        "  - name: section_index",
        "    dtype: int64",
        "  - name: section_path",
        "    sequence: string",
        "  - name: sentence_index",
        "    dtype: int64",
        "  - name: sentence_text_raw",
        "    dtype: string",
        "  - name: sentence_text_normalized",
        "    dtype: string",
        "  - name: previous_sentence",
        "    dtype: string",
        "  - name: next_sentence",
        "    dtype: string",
        "  - name: url",
        "    dtype: string",
        "  - name: page_id",
        "    dtype: int64",
        "  - name: revision_id",
        "    dtype: int64",
        "  - name: revision_timestamp",
        "    dtype: string",
        "  - name: document_content_hash",
        "    dtype: string",
        "  - name: section_content_hash",
        "    dtype: string",
        "  - name: sentence_content_hash",
        "    dtype: string",
        "  - name: duplicate_occurrence_count",
        "    dtype: int64",
        "  - name: duplicate_sources",
        "    sequence: string",
        "  - name: polygon_name",
        "    dtype: string",
        "  - name: osm_primary_tag",
        "    dtype: string",
        "  - name: osm_tags",
        "    sequence:",
        "    - name: key",
        "      dtype: string",
        "    - name: value",
        "      dtype: string",
        "  - name: region",
        "    dtype: string",
        "  - name: lat",
        "    dtype: float64",
        "  - name: lon",
        "    dtype: float64",
        "  - name: input_dataset_revision",
        "    dtype: string",
        "  - name: pipeline_version",
        "    dtype: string",
    ]
    return "\n".join(feature_lines)


def _yaml_front_matter(stats: DatasetStatistics) -> str:
    """Render valid Hugging Face dataset-card YAML front matter.

    Emits YAML 1.2 block-sequence form for the language list so a
    standard parser produces the same ``list[str]`` in every case:
    ``language:`` followed by ``- "x"`` lines, or ``language: []`` when
    the list is empty. Strings that contain ``:``, ``|``, ``"``, ``\\``,
    or leading/trailing whitespace are emitted as double-quoted scalars to
    keep the rendering parseable and deterministic without a YAML
    dependency.  Includes a ``dataset_info`` block (with ``splits`` before
    ``features``) so the Hugging Face Dataset Viewer can interpret
    ``osm_tags`` as a ``Sequence`` of ``{key, value}`` structs (the
    parquet ``map`` type is not supported by ``datasets``).  ``splits``
    appears before ``features`` so the Hub's strict YAML parser does not
    reject a sibling-key dedent after a deeply-nested feature item.
    """
    if stats.language_counts:
        languages_block = "\n".join(
            f"- {_yaml_quote_scalar(lang)}" for lang in stats.language_counts
        )
        language_section = f"language:\n{languages_block}"
    else:
        language_section = "language: []"
    features_block = _yaml_features_block(stats)
    splits_block = (
        f"  splits:\n"
        f"  - name: train\n"
        f"    num_examples: {stats.row_count}\n"
        f"  features:\n"
        f"{features_block}"
    )
    lines = [
        "---",
        "license: other",
        "pretty_name: OSM Polygon Wikidata Sentence Relevance",
        language_section,
        "dataset_info:",
        splits_block,
        "---",
    ]
    return "\n".join(lines)


def _escape_md_cell(value: str) -> str:
    """Escape a value for safe inclusion in a Markdown table cell.

    Pipes, backslashes, carriage returns, and newlines are replaced with
    HTML character references so the cell cannot break table syntax even
    if the value contains ``|``, ``\\``, ``\\r``, or a literal newline.
    """
    return (
        value.replace("&", "&amp;")
        .replace("|", "&#124;")
        .replace("\\", "&#92;")
        .replace("\r", "&#13;")
        .replace("\n", "&#10;")
        .replace("\t", "&#9;")
    )


def _escape_md_inline(value: str) -> str:
    """Escape a value for safe inclusion inside Markdown inline code.

    Backticks, pipes, backslashes, and control characters are escaped so
    the surrounding `` ` `` `` ` `` pair cannot be broken or split by an
    adversarial revision/version string. The literal text is preserved by
    HTML-entity substitution rather than removal, so the rendered card
    still shows the raw value to a human reader.
    """
    return (
        value.replace("&", "&amp;")
        .replace("`", "&#96;")
        .replace("|", "&#124;")
        .replace("\\", "&#92;")
        .replace("\r", "&#13;")
        .replace("\n", "&#10;")
        .replace("\t", "&#9;")
    )


def _counts_table(title: str, counts: Mapping[str, int]) -> str:
    """Render a deterministic Markdown breakdown table (rows sorted).

    Cell content is HTML-entity-escaped so a key containing a pipe,
    backslash, or newline cannot break the Markdown table structure.
    """
    if not counts:
        return f"### {title}\n\n_No values._\n"
    header = f"### {title}\n\n| Key | Count |\n| --- | --- |"
    body = "\n".join(
        f"| {_escape_md_cell(k)} | {v} |" for k, v in sorted(counts.items())
    )
    return f"{header}\n{body}\n"


def _quote_url_component_dataset_id(value: str) -> str:
    """Percent-encode an upstream dataset identifier for the Hub URL.

    Uses :func:`urllib.parse.quote` with ``safe="/"`` so ``owner/repo``
    style identifiers pass through unchanged, but every other character
    including Unicode code points (which are converted to UTF-8 first)
    is encoded as ``%XX``. This makes the URL component immune to
    smuggled ``?``, ``#``, ``%``, backslashes, brackets, parens, spaces,
    accented characters, and CJK code points.

    The documented slash handling intentionally diverges from the
    ambiguous "Unicode alnum" behavior of an earlier implementation.
    """
    return quote(value, safe="/")


def _quote_url_component_revision(value: str) -> str:
    """Percent-encode a Hub revision path component.

    Uses :func:`urllib.parse.quote` with ``safe=""`` so every path
    separator (``/``), query marker (``?``), fragment marker (``#``),
    percent sign, backslash, bracket, and paren, along with every
    non-ASCII code point (UTF-8 encoded), is percent-encoded. The Hub
    renders the revision as a single segment after ``/tree/``; this
    function makes that contract explicit.
    """
    return quote(value, safe="")


def _escape_md_link_label(value: str) -> str:
    """Escape a value for use as the visible text of a Markdown link.

    Markdown link labels terminate on the first ``]``; the URL inside
    ``(...)`` terminates on the closing ``)``. Both must be
    entity-escaped when the value is untrusted, along with the markers
    that break surrounding inline code spans and the control characters
    that confuse some Markdown renderers.

    The exact entity substitution is deterministic and uses HTML numeric
    character references (``&#NN;``) so any Markdown parser recovers the
    original text on render. Currently escaped:

    - ``&`` → ``&amp;`` (must be first);
    - ``[`` → ``&#91;`` and ``]`` → ``&#93;`` (cannot be skipped);
    - `` `` ` `` → ``&#96;`` (cannot be skipped);
    - ``\\`` → ``&#92;``;
    - ``\r`` / ``\n`` / ``\t`` → ``&#13;`` / ``&#10;`` / ``&#9;``.
    """
    return (
        value.replace("&", "&amp;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("`", "&#96;")
        .replace("\\", "&#92;")
        .replace("\r", "&#13;")
        .replace("\n", "&#10;")
        .replace("\t", "&#9;")
    )


def _markdown_dataset_link(label: str, dataset_id: str) -> str:
    """Build a safely-encoded Markdown link to a Hub dataset page.

    The label is escaped via ``_escape_md_link_label`` so an adversarial
    dataset identifier cannot terminate the link (``[``/``]``) or break
    the surrounding Markdown (backticks/backslashes/CR/LF). The URL
    component is percent-encoded with ``safe="/"`` via
    :func:`urllib.parse.quote` so an adversarial identifier cannot inject
    a URL path or break the URL.
    """
    url = (
        f"https://huggingface.co/datasets/{_quote_url_component_dataset_id(dataset_id)}"
    )
    return f"[{_escape_md_link_label(label)}]({url})"


def _source_provenance_section(stats: DatasetStatistics) -> str:
    """Render the "Source dataset and recorded input revision" section.

    Branch on the recorded ``input_dataset_id``:

    - **Hub mode** (``stats.input_dataset_id is not None``): the
      dataset identifier is shown with a Markdown link to the Hugging
      Face dataset page. When the recorded revision looks like a
      40-character lowercase hex SHA-1, a second link to the
      commit/tree page is also shown. The wording describes the value
      factually ("recorded immutable commit identifier" /
      "recorded reference") and never infers which acquisition
      mechanism produced it.
    - **Local mode** (``stats.input_dataset_id is None``): the card
      states explicitly that the build used a local input snapshot
      with no recorded Hub dataset ID and that the recorded revision
      string is whatever the operator configured at export time.
    """
    revision_rendered = _escape_md_inline(stats.input_dataset_revision)
    dataset_id = stats.input_dataset_id
    if dataset_id:
        # Optional revision-tree link, only when the revision actually
        # looks like an immutable commit SHA-1 (40 lowercase hex chars).
        import re as _re

        if _re.fullmatch(r"[0-9a-f]{40}", stats.input_dataset_revision.strip()):
            dataset_link = _markdown_dataset_link(dataset_id, dataset_id)
            tree_url = (
                "https://huggingface.co/datasets/"
                f"{_quote_url_component_dataset_id(dataset_id)}/tree/"
                f"{_quote_url_component_revision(stats.input_dataset_revision)}"
            )
            tree_link = (
                f"[{_escape_md_link_label(stats.input_dataset_revision)}]({tree_url})"
            )
            return (
                "The sentences were extracted from the input dataset "
                f"{dataset_link} at the recorded input revision "
                f"{tree_link}. This value is a recorded immutable "
                "commit identifier; the card does not assume which "
                "acquisition mechanism produced it."
            )
        return (
            "The sentences were extracted from the input dataset "
            f"{_markdown_dataset_link(dataset_id, dataset_id)} at the "
            f"recorded input revision `{revision_rendered}`. This value "
            "is a recorded reference; it is reproduced exactly as it "
            "was supplied to the build."
        )
    # Local mode: no Hub dataset ID was recorded.
    return (
        "This build used a local input snapshot: there is no recorded "
        "Hugging Face dataset ID for the upstream source. The recorded "
        f"input revision `{revision_rendered}` is the value the "
        "operator configured at export time and is preserved exactly "
        "as recorded; no Hub commit SHA is implied."
    )


def _single_region_preview_section(stats: DatasetStatistics) -> str:
    """Render a prominent "Dataset scope" section when exactly one region
    is in the export.

    Returns ``""`` when ``region_counts`` has zero or multiple keys so
    multi-region and empty exports do not advertise a misleading
    preview scope.  All figures are derived from ``stats``; the region
    display name is the title-cased form of the single region key.
    """
    if len(stats.region_counts) != 1:
        return ""
    region_key, region_rows = next(iter(stats.region_counts.items()))
    display_name = region_key.title()
    sha = stats.parquet_sha256
    revision = _escape_md_inline(stats.input_dataset_revision)
    sha_inline = _escape_md_inline(sha)
    return f"""\
## Dataset scope

This published artifact is the **{display_name}-only preview** of the
OSM Polygon Wikidata Sentence Relevance dataset. It contains
{stats.row_count} deduplicated sentence rows extracted from
{stats.unique_polygons} unique OSM polygons in the
{_escape_md_inline(region_key)} shard, covering
{stats.unique_wikidata_entities} Wikidata entities and
{stats.unique_documents} unique documents. The full multi-region
dataset is published incrementally; the current artifact covers a
single region only and is intended as a canary/validation snapshot
of the production export pipeline.

- **Region:** {display_name}
- **Region key in this preview:** `{_escape_md_inline(region_key)}`
- **Region rows:** {region_rows}
- **Preview rows:** {stats.row_count}
- **Preview polygons:** {stats.unique_polygons}
- **Preview Wikidata entities:** {stats.unique_wikidata_entities}
- **Preview documents:** {stats.unique_documents}
- **Recorded input revision:** `{revision}`
- **Preview Parquet SHA-256:** `{sha_inline}`
"""


def render_dataset_card(stats: DatasetStatistics) -> str:
    """Render the deterministic Hugging Face dataset card.

    The card is derived entirely from ``stats`` (which itself comes from
    the finalized table). It includes a visible note that the quantitative
    sections are generated automatically and must not be edited by hand.
    """
    with_coords = stats.rows_with_coordinates
    without_coords = stats.rows_without_coordinates
    total_coords = with_coords + without_coords

    source_section = _counts_table("Source coverage", stats.source_counts)
    language_section = _counts_table("Language coverage", stats.language_counts)
    region_section = _counts_table("Region coverage", stats.region_counts)
    preview_section = _single_region_preview_section(stats)

    return f"""\
{_yaml_front_matter(stats)}

<!-- GENERATED AUTOMATICALLY DURING EXPORT. DO NOT EDIT MANUALLY. -->
<!-- The quantitative sections below are computed from the exported Parquet -->
<!-- data and must be regenerated whenever the dataset is rebuilt. -->

# OSM Polygon Wikidata Sentence Relevance

This dataset contains normalized sentences extracted from OpenStreetMap
(OSM) polygon articles and their linked Wikipedia / Wikivoyage pages.
Each row is a deduplicated sentence occurrence scoped to a polygon,
language, and content hash.

The dataset is a sentence-level snapshot with sentence/provenance fields
only. Land-use relevance labels, polygon-description classifications,
relevance scores, and similarity-pair annotations are future downstream
work and are absent from this dataset.

{preview_section}## Dataset summary

- **Total sentence rows:** {stats.row_count}
- **Unique sentence IDs:** {stats.unique_sentence_ids}
- **Unique polygons:** {stats.unique_polygons}
- **Unique Wikidata entities:** {stats.unique_wikidata_entities}
- **Unique document identities
  (source, site, language, document_id):** {stats.unique_documents}
- **Rows with coordinates:** {with_coords} / {total_coords}
- **Rows without coordinates:** {without_coords} / {total_coords}
- **Input dataset revision:** `{_escape_md_inline(stats.input_dataset_revision)}`
- **Pipeline version:** `{_escape_md_inline(stats.pipeline_version)}`
- **Exported Parquet SHA-256:** `{_escape_md_inline(stats.parquet_sha256)}`

## Source dataset and recorded input revision

{_source_provenance_section(stats)}

## Wikipedia and Wikivoyage coverage

{source_section}
This dataset combines Wikipedia and Wikivoyage provenance. Wikipedia and
Wikivoyage rows may describe the same Wikidata entity; cross-source
duplicates are collapsed to a single canonical occurrence during
finalization (see *Deterministic IDs, deduplication, and context policy*).
A "document identity" is defined as the tuple
`(source, site, language, document_id)`; a row in one language, site, or
source is not considered the same document as a row with the same raw
`document_id` but different tuple dimensions.

{language_section}
{region_section}
## Output schema (field descriptions)

The export is a single Parquet table (`sentences.parquet`) with the
following schema:

| Field | Type | Description |
| --- | --- | --- |
| `sentence_id` | string | Deterministic ID from polygon, language, and sentence content hash. |
| `polygon_id` | string | OSM polygon identifier. |
| `wikidata` | string | Wikidata entity QID for the polygon/page. |
| `document_id` | string | Document/page identifier within its source/site/language. |
| `article_id` | string (nullable) | Article identifier where available. |
| `source` | string | `wikipedia` or `wikivoyage`. |
| `language` | string | Language code of the sentence. |
| `site` | string | Source site (e.g. `en.wikipedia.org`). |
| `page_title` | string | Page title. |
| `section_id` | string | Section identifier. |
| `section_index` | int64 | Section ordinal within the document. |
| `section_path` | list<string> | Section breadcrumb path. |
| `sentence_index` | int64 | Sentence ordinal within the section. |
| `sentence_text_raw` | string | Segment text after surrounding-whitespace trimming (the segmenter may also trim internal whitespace). |
| `sentence_text_normalized` | string | Normalized sentence text used as the dedup key. |
| `previous_sentence` | string (nullable) | Prior sentence in the section (context). |
| `next_sentence` | string (nullable) | Next sentence in the section (context). |
| `url` | string | Source URL of the document. |
| `page_id` | int64 | Source page ID. |
| `revision_id` | int64 | Source revision ID. |
| `revision_timestamp` | string | Source revision timestamp. |
| `document_content_hash` | string | Hash of the source document. |
| `section_content_hash` | string | Hash of the source section. |
| `sentence_content_hash` | string | Hash of the normalized sentence (dedup key component). |
| `duplicate_occurrence_count` | int64 | Number of source occurrences collapsed into this row. |
| `duplicate_sources` | list<string> | Distinct sources among the collapsed occurrences. |
| `polygon_name` | string (nullable) | Human-readable polygon name. |
| `osm_primary_tag` | string (nullable) | Primary OSM tag of the polygon. |
| `osm_tags` | map<string,string> | OSM tags of the polygon. |
| `region` | string | Input region/extract name. |
| `lat` | float64 (nullable) | Latitude of the polygon centroid, if known. |
| `lon` | float64 (nullable) | Longitude of the polygon centroid, if known. |
| `input_dataset_revision` | string | Exact input revision recorded for reproducibility. |
| `pipeline_version` | string | Pipeline version recorded for reproducibility. |

## Sentence preprocessing and normalization

Sentences are extracted per section. Each segment emitted by the segmenter
has its surrounding whitespace trimmed and is then passed through a
fixed-order normalization pipeline that performs Unicode NFC
normalization, removes the configured zero-width characters (U+200B,
U+2060, U+FEFF), replaces Unicode control characters with spaces,
collapses whitespace, and strips leading MediaWiki edit markers. The
pipeline **preserves case**, punctuation,
accents, and joiner characters; only `sentence_text_normalized` (the
post-pipeline text) is used as the content hash and dedup key.
`sentence_text_raw` is the segment text after surrounding-whitespace
trimming — the downstream normalizer above is applied to produce
`sentence_text_normalized`, so `sentence_text_raw` is **not** the
original mediawiki source text.

## Deterministic IDs, deduplication, and context policy

Each sentence occurrence is assigned a deterministic `sentence_id` derived
from `polygon_id`, `language`, and the SHA-256 of the normalized text.
Exact duplicates (same polygon, language, and normalized text) are
collapsed into a single canonical occurrence. When a Wikipedia and a
Wikivoyage occurrence collide, Wikipedia is chosen as the canonical
source; the full set of contributing sources is recorded in
`duplicate_sources`. Intra-section previous/next sentences are attached
as context and never alter the identity or content of the row itself.

## Provenance and revision tracking

Every export records `input_dataset_revision` and `pipeline_version` in
both the Parquet schema metadata and the `manifest.json`. The manifest
also stores the Parquet SHA-256, so the content identity of the exact
exported bytes is verifiable independently of this card. The versioned
`statistics` object inside the manifest and the top-level count fields
are derived from the same computation; the validator rejects manifests
where they disagree.

## Intended use

This dataset supports text-research tasks over OSM polygon articles:
contextual analysis of how places are described across Wikipedia and
Wikivoyage, sentence-level corpus studies, and downstream modeling that
needs sentence text plus article provenance. It is intended for research
and evaluation, not as a substitute for the upstream sources. The
dataset is not a labelled dataset: it does not contain relevance labels,
similarity pairs, or classification outputs, and should not be treated as
one.

## Limitations and known biases

- Extraction depends on upstream OSM / Wikimedia availability, coverage, and
  language balance; over-represented languages and regions will be
  reflected in the statistics above.
- Coordinates are only present when the source polygon carries centroid
  information; see the coordinate counts above.
- Sentence segmentation uses an automatic multilingual model and may
  mis-segment short or mixed-script text.
- Deduplication is exact-match on the normalized text within a polygon
  and language; this collapses identical sentences only and does not in
  any way imply semantic similarity or relevance. The exporter does not
  apply semantic deduplication or sentence-pair scoring.

## Licensing

This dataset combines content from OpenStreetMap and Wikimedia projects.
The repository `LICENSE` covers only the code and this dataset-card
generator; it does not grant rights to the dataset's underlying content.
The dataset's underlying content is governed by the upstream terms of
OpenStreetMap and Wikimedia. Provenance fields and source URLs/revision
identifiers are retained in every row to support attribution; satisfying
the upstream attribution and licence requirements (such as ODbL for
OpenStreetMap contributions and CC BY-SA or project-specific terms for
Wikimedia contributions) remains the responsibility of downstream users.
No single SPDX identifier covers the combined dataset, which is why
`license: other` is used in the front matter.

## Reproducibility

Builds are deterministic for identical inputs, identical code revision,
locked dependencies, and a compatible execution environment. Re-running
the pipeline on the same immutable input revision and pipeline version
reproduces the same Parquet bytes, the same `manifest.json`, and this
auto-generated card.

---

*This dataset card was generated automatically from the exported data. The
statistics above are derived during export and must not be edited manually;
rebuild the dataset to regenerate them.*
"""


__all__ = [
    "STATISTICS_VERSION",
    "DatasetStatistics",
    "compute_statistics",
    "compute_parquet_statistics",
    "statistics_to_dict",
    "statistics_from_dict",
    "render_dataset_card",
    "render_dataset_card_from_profile",
    "schema_has_map_types",
    "schema_field_documentation",
]


def schema_has_map_types(schema: Any) -> bool:
    """Return True if *schema* contains any ``map<...>`` field, recursively.

    The Hugging Face ``datasets`` library cannot ingest
    ``map<string, string>`` columns; any export with such a field will
    fail Viewer-side requests (``/info``, ``/parquet``).  Used by the
    validator to reject exports that have not yet been migrated.
    """
    for field in schema:
        t = field.type
        if pa.types.is_map(t):
            return True
        if pa.types.is_list(t):
            if pa.types.is_map(t.value_type):
                return True
            if pa.types.is_struct(t.value_type):
                for child in t.value_type:
                    if pa.types.is_map(child.type):
                        return True
        elif pa.types.is_struct(t):
            for child in t:
                if pa.types.is_map(child.type):
                    return True
    return False


_SCHEMA_FIELD_DESCRIPTIONS: dict[str, str] = {
    "sentence_id": (
        "Deterministic ID from polygon, language, and sentence "
        "content hash."
    ),
    "polygon_id": "OSM polygon identifier.",
    "wikidata": "Wikidata entity QID for the polygon/page.",
    "document_id": (
        "Document/page identifier within its source/site/language."
    ),
    "article_id": "Article identifier where available.",
    "source": "`wikipedia` or `wikivoyage`.",
    "language": "Language code of the sentence.",
    "site": "Source site (e.g. `en.wikipedia.org`).",
    "page_title": "Page title.",
    "section_id": "Section identifier.",
    "section_index": "Section ordinal within the document.",
    "section_path": "Section breadcrumb path.",
    "sentence_index": "Sentence ordinal within the section.",
    "sentence_text_raw": (
        "Segment text after surrounding-whitespace trimming."
    ),
    "sentence_text_normalized": (
        "Normalised sentence text used as the dedup key."
    ),
    "previous_sentence": (
        "Prior sentence in the section (context)."
    ),
    "next_sentence": "Next sentence in the section (context).",
    "url": "Source URL of the document.",
    "page_id": "Source page ID.",
    "revision_id": "Source revision ID.",
    "revision_timestamp": "Source revision timestamp.",
    "document_content_hash": "Hash of the source document.",
    "section_content_hash": "Hash of the source section.",
    "sentence_content_hash": (
        "Hash of the normalised sentence (dedup key component)."
    ),
    "duplicate_occurrence_count": (
        "Number of source occurrences collapsed into this row."
    ),
    "duplicate_sources": (
        "Distinct sources among the collapsed occurrences."
    ),
    "polygon_name": "Human-readable polygon name.",
    "osm_primary_tag": "Primary OSM tag of the polygon.",
    "osm_tags": (
        "OSM tags of the polygon, encoded as a list of "
        "`{key, value}` structs so the Hugging Face Viewer can "
        "ingest the export."
    ),
    "region": "Input region/extract name.",
    "lat": (
        "Latitude of the polygon centroid, if known."
    ),
    "lon": (
        "Longitude of the polygon centroid, if known."
    ),
    "input_dataset_revision": (
        "Exact input revision recorded for reproducibility."
    ),
    "pipeline_version": (
        "Pipeline version recorded for reproducibility."
    ),
}


def _profile_field_type_label(field_name: str) -> str:
    """Return the on-card type label for *field_name*."""
    f = OUTPUT_SENTENCE_SCHEMA.field(field_name)
    if pa.types.is_list(f.type):
        inner = f.type.value_type
        if pa.types.is_struct(inner):
            children = ", ".join(
                f"`{c.name}`: string"
                for c in inner
            )
            return f"list<struct<{children}>>"
        return f"list<{inner}>"
    return str(f.type)


def schema_field_documentation() -> list[tuple[str, str, str, str]]:
    """Return deterministic ``(name, type, nullable, description)`` rows.

    Order matches ``OUTPUT_SENTENCE_SCHEMA`` so two profiles with the
    same schema emit byte-identical documentation sections.
    """
    rows: list[tuple[str, str, str, str]] = []
    for f in OUTPUT_SENTENCE_SCHEMA:
        nullable = "yes" if f.nullable else "no"
        desc = _SCHEMA_FIELD_DESCRIPTIONS.get(
            f.name, "(no documentation)"
        )
        rows.append((
            f.name,
            _profile_field_type_label(f.name),
            nullable,
            desc,
        ))
    return rows


def _profile_schema_table() -> str:
    """Render the on-card schema field documentation table."""
    rows = schema_field_documentation()
    header = (
        "| Field | Type | Nullable | Description |\n"
        "| --- | --- | --- | --- |"
    )
    body = "\n".join(
        f"| `{name}` | `{type_label}` | {nullable} | {desc} |"
        for name, type_label, nullable, desc in rows
    )
    return f"{header}\n{body}\n"


def _profile_yaml(stats: DatasetStatistics, profile: Any) -> str:
    """Render the minimal valid YAML front matter for the on-card.

    Includes ``language`` block, a ``license: other`` declaration, and
    a ``dataset_info.splits`` entry so the Hugging Face Viewer can
    resolve the default config/split.  No ``dataset_info.features``
    block: the parquet is now Viewer-compatible directly, so we
    delegate feature discovery to the actual parquet bytes (which is
    what produces the structured JSON on the Viewer UI).
    """
    if stats.language_counts:
        languages_block = "\n".join(
            f"- {_yaml_quote_scalar(lang)}" for lang in stats.language_counts
        )
        language_section = f"language:\n{languages_block}"
    else:
        language_section = "language: []"
    lines = [
        "---",
        "license: other",
        "pretty_name: OSM Polygon Wikidata Sentence Relevance",
        language_section,
        "dataset_info:",
        "  features:",
        "  - name: osm_tags",
        "    sequence:",
        "    - name: key",
        "      dtype: string",
        "    - name: value",
        "      dtype: string",
        "  splits:",
        "  - name: train",
        f"    num_examples: {stats.row_count}",
        "---",
    ]
    return "\n".join(lines)


def _escape_md_inline_profile(value: str) -> str:
    """Escape inline code content for profile-based rendering."""
    return (
        value.replace("&", "&amp;")
        .replace("`", "&#96;")
        .replace("|", "&#124;")
        .replace("\\", "&#92;")
        .replace("\r", "&#13;")
        .replace("\n", "&#10;")
        .replace("\t", "&#9;")
    )


def _profile_preview_section(profile: Any) -> str:
    """Render the single-region preview section.

    Returns the empty string when the profile covers zero or multiple
    regions so multi-region and empty exports do not advertise a
    misleading preview scope.
    """
    if len(profile.region_counts) != 1:
        return ""
    region_key, region_rows = next(iter(profile.region_counts.items()))
    display_name = region_key.replace("-latest", "").replace("-", " ").title()
    if not display_name:
        display_name = region_key
    sha = profile.parquet_sha256
    revision = _escape_md_inline_profile(profile.input_dataset_revision)
    sha_inline = _escape_md_inline_profile(sha)
    return (
        f"## Dataset scope\n\n"
        f"This published artifact is the **{display_name}-only preview** "
        f"of the OSM Polygon Wikidata Sentence Relevance dataset. "
        f"It contains {profile.row_count} deduplicated sentence rows "
        f"extracted from {profile.unique_polygons} unique OSM polygons "
        f"in the `{region_key}` shard, covering "
        f"{profile.unique_wikidata_entities} Wikidata entities and "
        f"{profile.unique_documents} unique documents. The full "
        f"multi-region dataset is published incrementally; the current "
        f"artifact covers a single region only and is intended as a "
        f"canary/validation snapshot of the production export "
        f"pipeline.\n\n"
        f"- **Region:** {display_name}\n"
        f"- **Region key:** `{_escape_md_inline_profile(region_key)}`\n"
        f"- **Region rows:** {region_rows}\n"
        f"- **Preview rows:** {profile.row_count}\n"
        f"- **Preview polygons:** {profile.unique_polygons}\n"
        f"- **Preview Wikidata entities:** "
        f"{profile.unique_wikidata_entities}\n"
        f"- **Preview documents:** {profile.unique_documents}\n"
        f"- **Recorded input revision:** `{revision}`\n"
        f"- **Preview Parquet SHA-256:** `{sha_inline}`\n"
    )


def render_dataset_card_from_profile(
    profile: Any,
    *,
    asset_base_url: str | None = None,
) -> str:
    """Render the dataset card from an immutable ``DatasetProfile``.

    This renderer is the canonical post-Phase 9P format. It uses
    profile-derived fields (so the card and manifest cannot drift) and
    embeds two PNG assets, a full real example row, and the schema
    field documentation. All quantitative figures are derived from the
    profile; nothing is hand-typed.
    """
    from osm_polygon_sentence_relevance.output.profile import (
        render_example_row_json,
    )

    stats = DatasetStatistics(
        version=STATISTICS_VERSION,
        row_count=profile.row_count,
        unique_sentence_ids=profile.unique_sentence_ids,
        unique_polygons=profile.unique_polygons,
        unique_wikidata_entities=profile.unique_wikidata_entities,
        unique_documents=profile.unique_documents,
        source_counts=profile.source_counts,
        language_counts=profile.language_counts,
        region_counts=profile.region_counts,
        rows_with_coordinates=profile.rows_with_coordinates,
        rows_without_coordinates=profile.rows_without_coordinates,
        input_dataset_revision=profile.input_dataset_revision,
        pipeline_version=profile.pipeline_version,
        input_dataset_id=profile.input_dataset_id,
        parquet_sha256=profile.parquet_sha256,
    )
    coords = profile.rows_with_coordinates
    total = profile.rows_with_coordinates + profile.rows_without_coordinates

    # Asset embeds. The Hugging Face Hub resolves a relative
    # ``assets/foo.png`` link against the README's location and
    # serves the bytes through its CDN; some renderers, however, do
    # not rewrite that path. To keep the README self-contained and
    # renderable everywhere, the embedded Markdown uses an explicit
    # URL (relative by default; the caller may override with
    # ``asset_base_url``) so the image loads regardless of which
    # path resolver the viewer applies.
    geo = profile.assets.get("geographic_coverage.png")
    lang_png = profile.assets.get("language_distribution.png")

    def _img(alt: str, name: str) -> str:
        if asset_base_url is None:
            return f"![{alt}](assets/{name})"
        base = asset_base_url.rstrip("/")
        return f"![{alt}]({base}/{name})"

    geo_md = (
        _img("Geographic coverage", "geographic_coverage.png")
        if geo is not None
        else "_(Geographic coverage asset unavailable.)_"
    )
    lang_md = (
        _img("Language distribution", "language_distribution.png")
        if lang_png is not None
        else "_(Language distribution asset unavailable.)_"
    )

    example_json = render_example_row_json(profile)

    schema_table = _profile_schema_table()
    preview_section = _profile_preview_section(profile)

    yaml_block = _profile_yaml(stats, profile)

    coords_extent = (
        f"{profile.lat_min:.4f}, {profile.lon_min:.4f}"
        if profile.lat_min is not None
        and profile.lat_max is not None
        and profile.lon_min is not None
        and profile.lon_max is not None
        else "(no coordinates)"
    )
    coords_extent = (
        f"{profile.lat_min:.4f} → {profile.lat_max:.4f}, "
        f"{profile.lon_min:.4f} → {profile.lon_max:.4f}"
    )

    language_table_rows = "\n".join(
        f"| `{_escape_md_inline_profile(lang)}` | {count} |"
        for lang, count in sorted(
            profile.language_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )
    )
    full_language_table = (
        "| Language | Rows |\n| --- | --- |\n"
        f"{language_table_rows}\n"
    )

    return f"""\
{yaml_block}

<!-- GENERATED AUTOMATICALLY DURING EXPORT. DO NOT EDIT MANUALLY. -->
<!-- The quantitative sections below are computed from the exported Parquet -->
<!-- data via the immutable DatasetProfile; rebuild the dataset to regenerate. -->

# OSM Polygon Wikidata Sentence Relevance

This dataset contains normalised sentences extracted from OpenStreetMap
(OSM) polygon articles and their linked Wikipedia / Wikivoyage pages.
Each row is a deduplicated sentence occurrence scoped to a polygon,
language, and content hash.

The dataset is a sentence-level snapshot with sentence/provenance fields
only. Land-use relevance labels, polygon-description classifications,
relevance scores, and similarity-pair annotations are future downstream
work and are absent from this dataset.

{preview_section}## Dataset summary

- **Total sentence rows:** {profile.row_count}
- **Unique sentence IDs:** {profile.unique_sentence_ids}
- **Unique polygons:** {profile.unique_polygons}
- **Unique Wikidata entities:** {profile.unique_wikidata_entities}
- **Unique document identities
  (source, site, language, document_id):** {profile.unique_documents}
- **Rows with coordinates:** {coords} / {total}
- **Rows without coordinates:** {profile.rows_without_coordinates} / {total}
- **Rows with polygon_name:** {profile.rows_with_polygon_name}
- **Sentence length (chars):**
  min {profile.sentence_length_min},
  mean {profile.sentence_length_mean:.2f},
  max {profile.sentence_length_max}
- **Coordinate extent:** {coords_extent}
- **Input dataset revision:** `{_escape_md_inline_profile(profile.input_dataset_revision)}`
- **Pipeline version:** `{_escape_md_inline_profile(profile.pipeline_version)}`
- **Exported Parquet SHA-256:** `{_escape_md_inline_profile(profile.parquet_sha256)}`

## Source dataset and recorded input revision

{_source_provenance_section(stats)}

## Geographic coverage

{geo_md}

The scatter plot shows the centroid of every polygon with both
`lat` and `lon` populated (one dot per polygon, not one per row).
The visible extent is derived from the actual data; the
`(min, max)` pair reported under *Dataset summary* is the precise
bounding box used to auto-fit the asset.

## Language coverage

{lang_md}

<details>
<summary>Full language breakdown ({len(profile.language_counts)} languages)</summary>

{full_language_table}
</details>

A "document identity" is defined as the tuple
`(source, site, language, document_id)`. Rows with the same raw
`document_id` but different source / site / language are treated as
distinct documents.

## Wikipedia and Wikivoyage coverage

{_counts_table("Source coverage", profile.source_counts)}

Wikipedia and Wikivoyage rows may describe the same Wikidata entity;
cross-source duplicates are collapsed to a single canonical
occurrence during finalisation. The canonical selection rule is
documented below in *Deterministic IDs, deduplication, and context
policy*.

## Sentence segmentation and normalisation

Sentences were extracted using the segmentation model
`{_escape_md_inline_profile(profile.segmentation_model)}`
at the exact model revision
`{_escape_md_inline_profile(profile.segmentation_revision)}`,
recorded for reproducibility.

Each emitted segment has its surrounding whitespace trimmed and is
then passed through a fixed-order normalisation pipeline:

1. Unicode NFC normalisation.
2. Removal of configured zero-width characters (`U+200B`, `U+2060`,
   `U+FEFF`).
3. Replacement of Unicode control characters with spaces.
4. Whitespace collapse.
5. Stripping of leading MediaWiki edit markers.

The pipeline preserves case, punctuation, accents, and joiner
characters; only `sentence_text_normalized` (the post-pipeline text)
is used as the content hash and dedup key.

## Deterministic IDs, deduplication, and context policy

Each sentence occurrence is assigned a deterministic `sentence_id`
derived from `polygon_id`, `language`, and the SHA-256 of the
normalised text. Exact duplicates (same polygon, language, and
normalised text) are collapsed into a single canonical occurrence.
When a Wikipedia and a Wikivoyage occurrence collide, Wikipedia is
chosen as the canonical source; the full set of contributing sources
is recorded in `duplicate_sources`. Intra-section previous/next
sentences are attached as context and never alter the identity or
content of the row itself.

## Real example row (from the export)

<details>
<summary>Show one row from the canonical-sorted Parquet</summary>

```json
{example_json}
```

</details>

## Output schema (field descriptions)

The export is a single Parquet table (`sentences.parquet`) with the
following schema. Note the ``osm_tags`` column is now encoded as a
list of ``{{key, value}}`` structs (the legacy ``map<string,string>``
form is not ingestible by the Hugging Face ``datasets`` library).

{schema_table}

## Provenance and revision tracking

Every export records `input_dataset_revision`, `pipeline_version`,
and (when applicable) `input_dataset_id` in both the Parquet schema
metadata and the `manifest.json`. The manifest also stores the
Parquet SHA-256, the asset SHA-256s, the segmentation model name and
exact revision, and the source-code commit that produced this export,
so the content identity of every byte in this directory is
verifiable independently of this card. The versioned `statistics`
object inside the manifest and the top-level count fields are
derived from the same computation; the validator rejects manifests
where they disagree.

## Intended use

This dataset supports text-research tasks over OSM polygon articles:
contextual analysis of how places are described across Wikipedia and
Wikivoyage, sentence-level corpus studies, and downstream modeling
that needs sentence text plus article provenance. It is intended for
research and evaluation, not as a substitute for the upstream sources.
The dataset is not a labelled dataset: it does not contain relevance
labels, similarity pairs, or classification outputs, and should not
be treated as one.

## Limitations and known biases

- Extraction depends on upstream OSM / Wikimedia availability,
  coverage, and language balance; over-represented languages and
  regions will be reflected in the statistics above.
- Coordinates are only present when the source polygon carries
  centroid information; see the coordinate counts above.
- Sentence segmentation uses an automatic multilingual model and may
  mis-segment short or mixed-script text.
- Deduplication is exact-match on the normalised text within a
  polygon and language; this collapses identical sentences only and
  does not in any way imply semantic similarity or relevance. The
  exporter does not apply semantic deduplication or
  sentence-pair scoring.

## Licensing

This dataset combines content from OpenStreetMap and Wikimedia
projects. The repository `LICENSE` covers only the code and this
dataset-card generator; it does not grant rights to the dataset's
underlying content, which is governed by the upstream terms of
OpenStreetMap and Wikimedia. Provenance fields and source
URL/revision identifiers are retained in every row to support
attribution; satisfying the upstream attribution and licence
requirements (such as ODbL for OpenStreetMap contributions and
CC BY-SA or project-specific terms for Wikimedia contributions)
remains the responsibility of downstream users. No single SPDX
identifier covers the combined dataset, which is why `license: other`
is used in the front matter.

## Reproducibility

Builds are deterministic for identical inputs, identical code
revision, locked dependencies, and a compatible execution
environment. Re-running the pipeline on the same immutable input
revision and pipeline version reproduces the same Parquet bytes, the
same `manifest.json`, the same asset PNGs, and this auto-generated
card.

---

*This dataset card was generated automatically from the exported
data via the immutable ``DatasetProfile`` and must not be edited
manually; rebuild the dataset to regenerate.*
"""
