"""Single-source-of-truth dataset profile and asset rendering.

Every quantitative claim in the dataset card, every PNG asset hash in
the manifest, and every row used for the on-card real example comes
from the immutable :class:`DatasetProfile` built here. Two profiles
constructed from identical inputs (modulo the SHA-256 carrying the
content identity) must produce byte-identical renders.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import tempfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.contracts.errors import ExportError
from osm_polygon_sentence_relevance.contracts.schemas import OUTPUT_SENTENCE_SCHEMA
from osm_polygon_sentence_relevance.output.checksum import sha256_file

# PNG file signature: 8 bytes 89 50 4E 47 0D 0A 1A 0A.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Profile / card schema version. Bumped when the shape of the on-card
# versioned object changes incompatibly. Validators pin against this.
PROFILE_VERSION = 1


class ProfileError(ExportError):
    """Raised when profile construction fails."""


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
                    MappingProxyType({k: v for k, v in value.items()}),
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
        assets_dict = {name: {
            "name": info.name,
            "sha256": info.sha256,
            "bytes": info.bytes_,
        } for name, info in sorted(self.assets.items())}

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
    first_batch = next(iter(parquet.iter_batches(batch_size=1)))
    if first_batch.num_rows == 0:
        raise ProfileError("Cannot sample an example row from an empty Parquet file")
    values = first_batch.to_pydict()
    return {col: values[col][0] for col in OUTPUT_SENTENCE_SCHEMA.names}


def _build_signature_png(
    width: int,
    height: int,
    *,
    pixel_callback: Any,
) -> bytes:
    """Build a deterministic RGBA PNG of *width* × *height*.

    ``pixel_callback`` receives ``(x, y)`` and must return an RGBA
    ``(r, g, b, a)`` tuple of integers in ``[0, 255]``. We use the
    stdlib ``zlib`` so the rendering is fully deterministic and has
    no third-party visual-prng or font-render dependency.
    """
    # PNG header
    out = io.BytesIO()
    out.write(PNG_SIGNATURE)
    # IHDR
    ihdr = bytearray()
    ihdr.extend(b"IHDR")
    ihdr.extend(width.to_bytes(4, "big"))
    ihdr.extend(height.to_bytes(4, "big"))
    ihdr.append(8)  # bit depth
    ihdr.append(6)  # color type: RGBA
    ihdr.extend(b"\x00\x00\x00")  # compression / filter / interlace
    out.write(_chunk(b"IHDR", bytes(ihdr)))
    # IDAT
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter: None
        for x in range(width):
            r, g, b, a = pixel_callback(x, y)
            raw.extend([r & 0xFF, g & 0xFF, b & 0xFF, a & 0xFF])
    compressed = zlib.compress(bytes(raw), level=9)
    out.write(_chunk(b"IDAT", compressed))
    # IEND
    out.write(_chunk(b"IEND", b""))
    return out.getvalue()


def _chunk(tag: bytes, data: bytes) -> bytes:
    """Build a PNG chunk with the standard CRC-32 over tag+data."""
    body = tag + data
    return (
        len(data).to_bytes(4, "big")
        + body
        + zlib.crc32(body).to_bytes(4, "big")
    )


def _draw_geographic_pixels(
    lats: list[float],
    lons: list[float],
    *,
    width: int,
    height: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> bytes:
    """Render an RGBA PNG with one pixel per (lat, lon) row.

    The background is white, the scatter dots are a fixed opaque blue.
    The visualisation is byte-stable for the same input set because the
    Bresenham-style whole-pixel placement does not depend on any
    external RNG.
    """
    # Pad the data bounding box so dots near the edges are not clipped.
    pad_lat = max((lat_max - lat_min) * 0.05, 0.01)
    pad_lon = max((lon_max - lon_min) * 0.05, 0.01)
    span_lat = (lat_max - lat_min) or 1.0
    span_lon = (lon_max - lon_min) or 1.0

    pixel_set: set[tuple[int, int]] = set()
    for lat, lon in zip(lats, lons, strict=False):
        xf = (lon - (lon_min - pad_lon)) / (span_lon + 2 * pad_lon) * (width - 1)
        yf = (1.0 - (lat - (lat_min - pad_lat)) / (span_lat + 2 * pad_lat)) * (
            height - 1
        )
        xi = max(0, min(width - 1, int(round(xf))))
        yi = max(0, min(height - 1, int(round(yf))))
        pixel_set.add((xi, yi))

    def _pixel(x: int, y: int) -> tuple[int, int, int, int]:
        if (x, y) in pixel_set:
            return 31, 119, 180, 255
        return 255, 255, 255, 255

    return _build_signature_png(width, height, pixel_callback=_pixel)


def _draw_language_bar(
    language_counts: Mapping[str, int],
    *,
    width: int,
    height: int,
) -> bytes:
    """Render a horizontal bar chart for the top languages.

    Renders white background with bars drawn in a fixed opaque blue.
    The image height is fixed; the bar width is proportional to the
    count for that language. Languages are listed top-down in order of
    count (descending). The image is fully deterministic.
    """
    if not language_counts:
        return _build_signature_png(
            width, height, pixel_callback=lambda x, y: (255, 255, 255, 255)
        )

    sorted_langs = sorted(
        language_counts.items(), key=lambda kv: (-kv[1], kv[0])
    )
    max_count = sorted_langs[0][1]

    n = len(sorted_langs)
    row_height = max(1, height // n)
    usable_height = row_height * n

    # Cache bar pixels row-by-row.
    bar_pixels: list[set[int]] = [set() for _ in range(usable_height)]
    for idx, (lang, count) in enumerate(sorted_langs):
        y_base = idx * row_height
        bar_width = max(1, int(round(width * count / max_count)))
        for offset in range(row_height):
            for x in range(bar_width):
                bar_pixels[y_base + offset].add(x)

    def _pixel(x: int, y: int) -> tuple[int, int, int, int]:
        if y < usable_height and x in bar_pixels[y]:
            return 31, 119, 180, 255
        return 255, 255, 255, 255

    return _build_signature_png(width, height, pixel_callback=_pixel)


def render_geographic_coverage_png(
    profile: DatasetProfile, parquet_path: Path | str
) -> bytes:
    """Render a deterministic PNG of the dataset's geographic coverage.

    The PNG is built pixel-by-pixel without any external rendering
    library so two identical profiles produce byte-identical bytes and
    so the PNG depends on no system fonts / tile servers / clock.
    Reads the row-level lat / lon from *parquet_path* (the data the
    profile was built from) to plot every polygon centroid.
    """
    if (
        profile.lat_min is None
        or profile.lat_max is None
        or profile.lon_min is None
        or profile.lon_max is None
    ):
        return _build_signature_png(
            320, 240,
            pixel_callback=lambda x, y: (255, 255, 255, 255),
        )
    lats: list[float] = []
    lons: list[float] = []
    path = Path(parquet_path)
    if path.is_file():
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=65_536, columns=["lat", "lon"]
        ):
            values = batch.to_pydict()
            for lat, lon in zip(values["lat"], values["lon"], strict=False):
                if lat is not None and lon is not None:
                    lats.append(lat)
                    lons.append(lon)
    return _draw_geographic_pixels(
        lats,
        lons,
        width=480,
        height=320,
        lat_min=profile.lat_min,
        lat_max=profile.lat_max,
        lon_min=profile.lon_min,
        lon_max=profile.lon_max,
    )


def render_language_distribution_png(profile: DatasetProfile) -> bytes:
    """Render a deterministic horizontal bar chart for the languages."""
    width = 640
    height = max(120, 12 * len(profile.language_counts))
    return _draw_language_bar(
        profile.language_counts, width=width, height=height
    )


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
    if (
        not isinstance(segmentation_model, str)
        or not segmentation_model.strip()
    ):
        raise ProfileError("segmentation_model must be a non-blank string")
    if (
        not isinstance(segmentation_revision, str)
        or not segmentation_revision.strip()
    ):
        raise ProfileError(
            "segmentation_revision must be a non-blank string"
        )
    if not isinstance(source_commit, str) or not source_commit.strip():
        raise ProfileError("source_commit must be a non-blank string")

    actual_sha = sha256_file(path)
    if actual_sha.lower() != parquet_sha256.lower():
        raise ProfileError(
            f"Parquet SHA {actual_sha!r} does not match expected "
            f"{parquet_sha256!r}"
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
    if input_dataset_id is not None:
        if metadata_dataset_id != input_dataset_id:
            raise ProfileError(
                "Explicit input_dataset_id does not match Parquet metadata"
            )

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
        )
        for batch in parquet.iter_batches(
            batch_size=65_536, columns=list(columns)
        ):
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
                language_counts[language] = (
                    language_counts.get(language, 0) + 1
                )
                region_counts[region] = region_counts.get(region, 0) + 1
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
            if table not in allowed:
                raise ProfileError("unexpected internal table name")
            result = connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()
            assert result is not None
            return int(result[0])

        unique_sentence_ids = _count("sentence_ids")
        if unique_sentence_ids != row_count:
            raise ProfileError(
                "Parquet contains duplicate sentence_id values"
            )
        unique_polygons = _count("polygons")
        unique_wikidata_entities = _count("wikidata")
        unique_documents = _count("documents")
    finally:
        connection.close()
        if database.exists():
            database.unlink()

    if row_count == 0:
        raise ProfileError(
            "Cannot build a profile from an empty Parquet file"
        )

    sentence_length_mean = (
        sentence_lengths_total / row_count if row_count else 0.0
    )
    if sentence_length_min is None:
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
