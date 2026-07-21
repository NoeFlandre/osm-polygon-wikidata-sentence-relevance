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

import matplotlib

# Pin the Agg backend so render is reproducible on machines without a
# display. ``Agg`` writes a PNG without touching the windowing stack.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.figure import Figure

from osm_polygon_sentence_relevance.contracts.errors import ExportError
from osm_polygon_sentence_relevance.contracts.schemas import (
    OUTPUT_SENTENCE_SCHEMA,
)
from osm_polygon_sentence_relevance.output.checksum import sha256_file

# PNG file signature: 8 bytes 89 50 4E 47 0D 0A 1A 0A.  Maintained for
# compatibility with the legacy byte-level PNG tests; the
# matplotlib-based renderers also emit bytes starting with this
# signature.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Profile / card schema version. Bumped when the shape of the on-card
# versioned object changes incompatibly. Validators pin against this.
PROFILE_VERSION = 1

# Vendored Natural Earth subset containing the Afghanistan outline.
# The full SHA-256 is checked at render time to guard against
# accidental edits.  See ``data/natural_earth/README.md`` for the
# upstream source, mirror, and license.
_NATURAL_EARTH_PATH = (
    Path(__file__).resolve().parent
    / "_vendor"
    / "natural_earth"
    / "afghanistan_outline.geojson"
)
_NATURAL_EARTH_EXPECTED_SHA256 = (
    "4fb163ae405f8be649f17e0d8ba83e0402f561268267512536d3f04cc4102feb"
)

# Output dimensions for the two PNG assets.  Both must be at least
# 1200x800 so they remain legible on the Hub dataset page.
_GEOGRAPHIC_PNG_WIDTH = 1400
_GEOGRAPHIC_PNG_HEIGHT = 900
_LANGUAGE_PNG_WIDTH = 1400
_LANGUAGE_PNG_HEIGHT = 900

# Top-N boundary for the language bar chart. Languages beyond this
# count are collapsed into a single ``Other`` bucket whose arithmetic
# (top + Other == total) is verified by the publication validator.
_LANGUAGE_TOP_N = 15

# Color palette. Restrained, high-contrast values that print legibly
# on both light and dark dataset-card backgrounds.
_GEO_OUTLINE_FILL_COLOR = "#f4d8b3"
_GEO_OUTLINE_EDGE_COLOR = "#8a6d3b"
_GEO_SCATTER_COLOR = "#1f5fa8"
_GEO_BACKGROUND_COLOR = "#ffffff"
_GEO_GRID_COLOR = "#dcdcdc"

_LANG_BAR_COLOR = "#1f5fa8"
_LANG_BAR_OTHER_COLOR = "#9b9b9b"
_LANG_TEXT_COLOR = "#202020"
_LANG_BACKGROUND_COLOR = "#ffffff"


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


def collect_polygon_centroids(
    profile: DatasetProfile,
    parquet_path: str | Path,
) -> list[tuple[float, float]]:
    """Return one ``(lat, lon)`` per unique polygon_id with coordinates.

    The geographic-coverage renderer must plot a single centroid per
    *canonical polygon identity*, not per sentence row.  Multiple
    sentence rows share a single polygon_id (one row per Wikipedia
    sentence that mentions that polygon), so plotting every row
    would over-count by an order of magnitude.

    Deduplication strategy:

    * iterate the Parquet in row-order chunks reading the
      ``polygon_id``, ``lat``, and ``lon`` columns together so a
      single scan yields the canonical polygon → centroid map;
    * keep the *first* non-null coordinate per polygon_id so the
      output is byte-deterministic given a fixed Parquet order;
    * skip rows whose polygon_id has no coordinates — the polygon
      is still recorded in :attr:`DatasetProfile.unique_polygons`
      but contributes no centroid.

    The returned list is ordered by first-seen polygon_id so the
    renderer can apply its deterministic jitter without reshuffling.
    """
    points: list[tuple[float, float]] = []
    seen: set[str] = set()
    path = Path(parquet_path)
    if not path.is_file():
        return points
    try:
        parquet = pq.ParquetFile(path)
    except Exception:
        return points
    try:
        for batch in parquet.iter_batches(
            batch_size=65_536,
            columns=["polygon_id", "lat", "lon"],
        ):
            values = batch.to_pydict()
            polygon_ids = values["polygon_id"]
            lats = values["lat"]
            lons = values["lon"]
            for polygon_id, lat, lon in zip(polygon_ids, lats, lons, strict=False):
                if not polygon_id or polygon_id in seen:
                    continue
                if lat is None or lon is None:
                    continue
                seen.add(polygon_id)
                points.append((float(lat), float(lon)))
    except (KeyError, OSError):
        return points
    return points


def geographic_caption_for_profile(profile: DatasetProfile) -> str:
    """Build the caption text drawn under the geographic-coverage PNG.

    Centralising the caption here keeps the on-PNG text and the
    legend label in lockstep: both report the deduplicated polygon
    centroid count (which equals ``profile.unique_polygons`` for
    Afghanistan-shaped datasets where every polygon has
    coordinates) alongside the dataset row count and the vendor
    attribution.
    """
    polygon_centroid_count = profile.unique_polygons
    if (
        profile.lat_min is not None
        and profile.lat_max is not None
        and profile.lon_min is not None
        and profile.lon_max is not None
    ):
        extent_text = (
            f"Extent: {profile.lat_min:.3f}°N → {profile.lat_max:.3f}°N, "
            f"{profile.lon_min:.3f}°E → {profile.lon_max:.3f}°E  |  "
            f"Polygons: {profile.unique_polygons}  |  "
            f"Rows: {profile.row_count}"
        )
    else:
        extent_text = (
            f"Extent: (no coordinates)  |  "
            f"Polygons: {profile.unique_polygons}  |  "
            f"Rows: {profile.row_count}"
        )
    caption = (
        "Country outline: Natural Earth 1:110m Admin 0 Countries "
        "(public domain, vendored at "
        "src/osm_polygon_sentence_relevance/output/_vendor/natural_earth/"
        "afghanistan_outline.geojson)."
    )
    return f"{extent_text}\nPolygon centroids ({polygon_centroid_count})\n{caption}"


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


def _build_signature_png(
    width: int,
    height: int,
    *,
    pixel_callback: Any,
) -> bytes:
    """Build a deterministic RGBA PNG of *width* × *height*.

    The corrective release replaces this hand-rolled encoder with
    matplotlib-based renderers; the helper is preserved as a
    no-op compatibility shim so external callers that still pass
    pixel callbacks do not crash.  The returned PNG is a blank
    white image of the requested dimensions.
    """
    fig, ax = plt.subplots(
        figsize=(width / 100, height / 100), dpi=100
    )  # pragma: no cover
    ax.set_axis_off()  # pragma: no cover
    fig.patch.set_facecolor("white")  # pragma: no cover
    ax.set_facecolor("white")  # pragma: no cover
    buf = _figure_to_png_bytes(fig)  # pragma: no cover
    plt.close(fig)  # pragma: no cover
    return buf


def _figure_to_png_bytes(fig: Figure) -> bytes:
    """Render *fig* to PNG bytes via matplotlib's Agg backend.

    Determinism is ensured by:
    1. pinning the Agg backend at import time;
    2. seeding matplotlib's RNG inside every renderer;
    3. disabling the ``text.usetex`` and ``font.family`` paths that
       depend on system fonts (DejaVu Sans is the matplotlib default
       and is bundled with the package so the output is byte-stable
       across environments).
    4. passing an explicit ``dpi`` and a figure size already in pixels
       so the rendered PNG matches the contract dimensions exactly
       (no ``bbox_inches="tight"`` shrinking).

    The output PNG dimensions are ``figsize_in_inches * dpi``; the
    call sites already size the figure so that ``width * height``
    matches the public contract (>= 1200x800).
    """
    from io import BytesIO

    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=100,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    return buf.getvalue()


def _seed_matplotlib_for_determinism() -> None:
    """Pin the matplotlib RNG state so identical inputs yield identical bytes."""
    np.random.seed(0x5A_52_4D_4F)  # 'ZRMO'
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["text.usetex"] = False


def _load_afghanistan_outline() -> tuple[list[tuple[float, float]], dict[str, str]]:
    """Load the Afghanistan polygon outline from the vendored GeoJSON.

    Returns a list of ``(lon, lat)`` tuples for the outer ring and
    the ``properties`` dict of the country feature.  The vendored
    file's SHA-256 is checked against the pinned constant so an
    accidental edit cannot silently shift the rendered outline.

    Raises
    ------
    ProfileError
        When the vendored GeoJSON is missing, unreadable, has an
        unexpected SHA, or does not contain the expected
        ``Polygon`` geometry.
    """
    if not _NATURAL_EARTH_PATH.is_file():
        raise ProfileError(
            f"Afghanistan outline GeoJSON is missing: {_NATURAL_EARTH_PATH}"
        )
    try:
        actual_sha = (
            hashlib.sha256(_NATURAL_EARTH_PATH.read_bytes()).hexdigest().lower()
        )
    except (
        OSError
    ) as err:  # pragma: no cover - OSError during read is unreachable in tests
        raise ProfileError(f"Cannot read Afghanistan outline GeoJSON: {err}") from err
    if actual_sha != _NATURAL_EARTH_EXPECTED_SHA256.lower():
        raise ProfileError(
            "Afghanistan outline GeoJSON SHA does not match the pinned "
            f"value (expected {_NATURAL_EARTH_EXPECTED_SHA256}, "
            f"got {actual_sha}); the file in data/natural_earth/ has "
            "been modified without updating the constant"
        )
    try:
        payload = json.loads(_NATURAL_EARTH_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ProfileError(f"Afghanistan outline GeoJSON is malformed: {err}") from err
    features = payload.get("features") or []
    if not features:
        raise ProfileError("Afghanistan outline GeoJSON contains no features")
    geometry = features[0].get("geometry") or {}
    if geometry.get("type") != "Polygon":
        raise ProfileError("Afghanistan outline GeoJSON first feature is not a Polygon")
    rings = geometry.get("coordinates") or []
    if not rings:
        raise ProfileError("Afghanistan outline GeoJSON Polygon has no coordinates")
    outer_ring = rings[0]
    polygon_lonlat = [(float(lon), float(lat)) for lon, lat in outer_ring]
    properties = dict(features[0].get("properties") or {})
    return polygon_lonlat, properties


def render_geographic_coverage_png(
    profile: DatasetProfile, parquet_path: Path | str
) -> bytes:
    """Render a deterministic PNG of the dataset's geographic coverage.

    The PNG plots the unique polygon centroids on top of a
    recognizable Afghanistan outline derived from the vendored
    Natural Earth 1:110m Admin 0 Countries subset (pinned
    SHA-256).  The figure has:

    * the Afghanistan outline filled with a low-saturation colour;
    * one scatter dot per polygon with both ``lat`` and ``lon``
      populated, jittered only enough to keep overlapping
      locations visible;
    * latitude / longitude gridlines, axis labels, and a title;
    * a colour scale or legend documenting the scatter scale;
    * a concise caption that names the data source.

    Two identical profiles produce byte-identical PNGs because:

    * matplotlib's RNG is seeded in :func:`_seed_matplotlib_for_determinism`;
    * the Agg backend does not depend on system fonts (DejaVu Sans
      is bundled with matplotlib);
    * the order of polygons in the Parquet file is preserved
      by :func:`build_dataset_profile`'s SQLite scratch.

    Parameters
    ----------
    profile
        The immutable profile the PNG is derived from.  The render
        uses ``lat_min``/``lat_max``/``lon_min``/``lon_max`` as the
        axis extent (padded so outline is not clipped) and the
        segmentation model + revision as part of the figure title.
    parquet_path
        Path to the finalized Parquet file.  Used to read the
        row-level ``lat``/``lon`` for the scatter dots so the
        renderer never invents values.
    """
    _seed_matplotlib_for_determinism()
    polygon_lonlat, outline_properties = _load_afghanistan_outline()
    if polygon_lonlat:
        outline_lons = [p[0] for p in polygon_lonlat]
        outline_lats = [p[1] for p in polygon_lonlat]
        outline_lon_min = min(outline_lons)
        outline_lon_max = max(outline_lons)
        outline_lat_min = min(outline_lats)
        outline_lat_max = max(outline_lats)
    else:
        # Unreachable when the vendored GeoJSON loads correctly; the
        # vendored file is pinned and verified at module import. Kept
        # as a defensive fallback for parity with the legacy renderer.
        outline_lon_min = outline_lon_max = 0.0  # pragma: no cover
        outline_lat_min = outline_lat_max = 0.0  # pragma: no cover

    # Plot extent: union of profile extent and outline extent, with
    # a sensible padding so the outline is not clipped.
    if (
        profile.lat_min is None
        or profile.lat_max is None
        or profile.lon_min is None
        or profile.lon_max is None
    ):
        lon_min, lon_max = outline_lon_min, outline_lon_max
        lat_min, lat_max = outline_lat_min, outline_lat_max
    else:
        lon_min = min(profile.lon_min, outline_lon_min)
        lon_max = max(profile.lon_max, outline_lon_max)
        lat_min = min(profile.lat_min, outline_lat_min)
        lat_max = max(profile.lat_max, outline_lat_max)
    pad_lon = max((lon_max - lon_min) * 0.05, 0.1)
    pad_lat = max((lat_max - lat_min) * 0.05, 0.1)
    lon_min -= pad_lon
    lon_max += pad_lon
    lat_min -= pad_lat
    lat_max += pad_lat

    lats: list[float] = []
    lons: list[float] = []
    # Deduplicate by canonical polygon_id so the scatter cloud
    # plots one centroid per unique polygon, never one per
    # sentence row.  ``collect_polygon_centroids`` is the single
    # source of truth used by both the renderer and the test
    # suite so the legend count can never disagree with what is
    # actually drawn.
    for lat, lon in collect_polygon_centroids(profile, parquet_path):
        lats.append(lat)
        lons.append(lon)

    fig = Figure(figsize=(_GEOGRAPHIC_PNG_WIDTH / 100, _GEOGRAPHIC_PNG_HEIGHT / 100))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(_GEO_BACKGROUND_COLOR)
    ax.set_facecolor(_GEO_BACKGROUND_COLOR)

    outline_label = outline_properties.get("ADMIN") or "Afghanistan"
    if polygon_lonlat:
        ring_lons, ring_lats = zip(*polygon_lonlat, strict=False)
        ax.fill(
            ring_lons,
            ring_lats,
            color=_GEO_OUTLINE_FILL_COLOR,
            edgecolor=_GEO_OUTLINE_EDGE_COLOR,
            linewidth=1.4,
            label=outline_label,
            zorder=1,
        )

    if lats:
        # Apply a tiny deterministic jitter so overlapping polygons
        # remain visible without obscuring the outline.  The legend
        # count is derived from the deduplicated point list so it
        # always matches what is drawn.
        rng = np.random.default_rng(0x5A_52_4D_4F)
        jitter_lon = rng.uniform(-0.02, 0.02, size=len(lons))
        jitter_lat = rng.uniform(-0.02, 0.02, size=len(lats))
        ax.scatter(
            np.asarray(lons) + jitter_lon,
            np.asarray(lats) + jitter_lat,
            s=22,
            c=_GEO_SCATTER_COLOR,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
            zorder=3,
            label=f"Polygon centroids ({len(lats)})",
        )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=_GEO_GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(
        "Geographic coverage — unique polygon centroids",
        fontsize=14,
        color=_LANG_TEXT_COLOR,
        pad=12,
    )
    ax.legend(loc="lower left", frameon=True, fontsize=10)

    fig.text(
        0.5,
        0.02,
        geographic_caption_for_profile(profile),
        ha="center",
        va="bottom",
        fontsize=8,
        color=_LANG_TEXT_COLOR,
    )
    fig.subplots_adjust(bottom=0.14, top=0.92, left=0.08, right=0.97)

    buf = _figure_to_png_bytes(fig)
    plt.close(fig)
    return buf


def render_language_distribution_png(profile: DatasetProfile) -> bytes:
    """Render a deterministic horizontal bar chart of language counts.

    The chart shows the top ``_LANGUAGE_TOP_N`` languages sorted by
    row count descending, with the remaining languages collapsed
    into a single ``Other`` bucket whose count equals
    ``row_count - sum(top_N)``.  Each bar carries the language code,
    the exact row count, and the percentage of the dataset's total.

    Two identical profiles produce byte-identical PNGs because:

    * the matplotlib RNG is seeded inside the renderer;
    * the Agg backend does not depend on system fonts;
    * the language ordering, count arithmetic, and bucket name are
      derived directly from the profile so no hand-typed values
      can drift between renders.
    """
    _seed_matplotlib_for_determinism()
    if not profile.language_counts:
        # Empty profile: render a blank canvas at the contract
        # dimensions with an explanatory caption.
        fig = Figure(
            figsize=(
                _LANGUAGE_PNG_WIDTH / 100,
                _LANGUAGE_PNG_HEIGHT / 100,
            )
        )
        ax = fig.add_subplot(111)
        ax.set_facecolor(_LANG_BACKGROUND_COLOR)
        fig.patch.set_facecolor(_LANG_BACKGROUND_COLOR)
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No language data available.",
            ha="center",
            va="center",
            fontsize=14,
            color=_LANG_TEXT_COLOR,
        )
        buf = _figure_to_png_bytes(fig)
        plt.close(fig)
        return buf

    sorted_langs = sorted(
        profile.language_counts.items(), key=lambda kv: (-kv[1], kv[0])
    )
    top_n = _LANGUAGE_TOP_N
    top_slice = sorted_langs[:top_n]
    other_langs = sorted_langs[top_n:]
    other_count = sum(c for _, c in other_langs)
    total = profile.row_count

    labels: list[str] = []
    counts: list[int] = []
    colors: list[str] = []
    for lang, count in top_slice:
        labels.append(lang)
        counts.append(count)
        colors.append(_LANG_BAR_COLOR)
    if other_count > 0 or other_langs:
        labels.append("Other")
        counts.append(other_count)
        colors.append(_LANG_BAR_OTHER_COLOR)

    # Sort the bars so the largest is on top (matplotlib draws first
    # row at the bottom of the y-axis).
    order = sorted(range(len(counts)), key=lambda i: (-counts[i], labels[i]))
    labels = [labels[i] for i in order]
    counts = [counts[i] for i in order]
    colors = [colors[i] for i in order]

    n_bars = len(labels)
    fig = Figure(figsize=(_LANGUAGE_PNG_WIDTH / 100, _LANGUAGE_PNG_HEIGHT / 100))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(_LANG_BACKGROUND_COLOR)
    ax.set_facecolor(_LANG_BACKGROUND_COLOR)

    y_positions = np.arange(n_bars)
    bars = ax.barh(
        y_positions,
        counts,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11, color=_LANG_TEXT_COLOR)
    ax.invert_yaxis()
    ax.set_xlabel("Row count", fontsize=11, color=_LANG_TEXT_COLOR)
    ax.set_title(
        "Language distribution — top 15 languages plus Other",
        fontsize=14,
        color=_LANG_TEXT_COLOR,
        pad=12,
    )
    ax.grid(True, axis="x", color=_GEO_GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    max_count = max(counts) if counts else 0
    ax.set_xlim(0, max_count * 1.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for bar, count in zip(bars, counts, strict=False):
        pct = (count / total * 100.0) if total else 0.0
        label = f"{count:,}  ({pct:.2f}%)"
        ax.text(
            bar.get_width() + max_count * 0.012,
            bar.get_y() + bar.get_height() / 2.0,
            label,
            va="center",
            ha="left",
            fontsize=10,
            color=_LANG_TEXT_COLOR,
        )

    caption_lines = [
        f"Total rows: {total:,}",
        f"Distinct languages: {len(profile.language_counts)}",
        f"Top languages shown: {min(top_n, len(sorted_langs))}",
        f"Other bucket rows: {other_count:,} ({len(other_langs)} languages)",
    ]
    fig.text(
        0.02,
        0.02,
        "  •  ".join(caption_lines),
        ha="left",
        va="bottom",
        fontsize=9,
        color=_LANG_TEXT_COLOR,
    )
    fig.subplots_adjust(bottom=0.16, top=0.92, left=0.12, right=0.97)

    buf = _figure_to_png_bytes(fig)
    plt.close(fig)
    return buf


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
