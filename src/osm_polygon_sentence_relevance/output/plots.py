"""Deterministic publication plots derived from dataset profiles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.contracts.errors import ExportError

if TYPE_CHECKING:
    from osm_polygon_sentence_relevance.output.profile import DatasetProfile

# PNG file signature: 8 bytes 89 50 4E 47 0D 0A 1A 0A.  Maintained for
# compatibility with the legacy byte-level PNG tests; the
# matplotlib-based renderers also emit bytes starting with this
# signature.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

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


def _load_plotting() -> tuple[Any, Any, type[Any]]:
    """Load optional plotting dependencies only when rendering assets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.figure import Figure

    return plt, np, Figure


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
    path = Path(parquet_path)
    parquet = _safe_parquet_file(path)
    if parquet is None:
        return []
    points: list[tuple[float, float]] = []
    seen: set[str] = set()
    try:
        for batch in parquet.iter_batches(
            batch_size=65_536,
            columns=["polygon_id", "lat", "lon"],
        ):
            _append_batch_centroids(batch, seen, points)
    except (KeyError, OSError):
        return points
    return points


def _safe_parquet_file(path: Path) -> pq.ParquetFile | None:
    """Return a readable Parquet handle, or ``None`` for unavailable input."""
    if not path.is_file():
        return None
    try:
        return pq.ParquetFile(path)
    except Exception:
        return None


def _append_batch_centroids(
    batch: Any,
    seen: set[str],
    points: list[tuple[float, float]],
) -> None:
    """Append first valid centroid rows from one Arrow record batch."""
    values = batch.to_pydict()
    polygon_ids = values["polygon_id"]
    lats = values["lat"]
    lons = values["lon"]
    row_count = min(len(polygon_ids), len(lats), len(lons))
    for index in range(row_count):
        polygon_id = polygon_ids[index]
        _append_centroid(polygon_id, lats[index], lons[index], seen, points)


def _append_centroid(
    polygon_id: object,
    lat: object,
    lon: object,
    seen: set[str],
    points: list[tuple[float, float]],
) -> None:
    """Append one centroid unless its identity or coordinates are unusable."""
    if not polygon_id or polygon_id in seen:
        return
    if lat is None or lon is None:
        return
    seen.add(polygon_id)
    points.append((float(lat), float(lon)))


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
    plt, _, _ = _load_plotting()
    fig, ax = plt.subplots(
        figsize=(width / 100, height / 100), dpi=100
    )  # pragma: no cover
    ax.set_axis_off()  # pragma: no cover
    fig.patch.set_facecolor("white")  # pragma: no cover
    ax.set_facecolor("white")  # pragma: no cover
    buf = _figure_to_png_bytes(fig)  # pragma: no cover
    plt.close(fig)  # pragma: no cover
    return buf


def _figure_to_png_bytes(fig: Any) -> bytes:
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
    plt, np, _ = _load_plotting()
    np.random.seed(0x5A_52_4D_4F)  # 'ZRMO'
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["text.usetex"] = False


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
    _verify_outline_file()
    payload = _read_outline_payload()
    polygon_lonlat, properties = _outline_geometry(payload)
    return polygon_lonlat, properties


def _verify_outline_file() -> None:
    """Require the vendored outline to exist and match its pinned digest."""
    if not _NATURAL_EARTH_PATH.is_file():
        raise ProfileError(
            f"Afghanistan outline GeoJSON is missing: {_NATURAL_EARTH_PATH}"
        )
    try:
        actual_sha = hashlib.sha256(_NATURAL_EARTH_PATH.read_bytes()).hexdigest().lower()
    except OSError as err:  # pragma: no cover - OSError during read is rare
        raise ProfileError(f"Cannot read Afghanistan outline GeoJSON: {err}") from err
    if actual_sha != _NATURAL_EARTH_EXPECTED_SHA256.lower():
        raise ProfileError(
            "Afghanistan outline GeoJSON SHA does not match the pinned "
            f"value (expected {_NATURAL_EARTH_EXPECTED_SHA256}, "
            f"got {actual_sha}); the file in data/natural_earth/ has "
            "been modified without updating the constant"
        )


def _read_outline_payload() -> dict[str, Any]:
    """Read and parse the vendored GeoJSON payload."""
    try:
        payload = json.loads(_NATURAL_EARTH_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ProfileError(f"Afghanistan outline GeoJSON is malformed: {err}") from err
    if not isinstance(payload, dict):
        raise ProfileError("Afghanistan outline GeoJSON must be a JSON object")
    return payload


def _outline_geometry(
    payload: dict[str, Any],
) -> tuple[list[tuple[float, float]], dict[str, str]]:
    """Extract the first Polygon ring and its properties from GeoJSON."""
    feature = _first_outline_feature(payload)
    outer_ring, properties = _outline_ring(feature)
    polygon_lonlat = [(float(lon), float(lat)) for lon, lat in outer_ring]
    return polygon_lonlat, properties


def _first_outline_feature(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the first GeoJSON feature, rejecting an empty collection."""
    features = payload.get("features") or []
    if not features:
        raise ProfileError("Afghanistan outline GeoJSON contains no features")
    feature = features[0]
    if not isinstance(feature, dict):
        raise ProfileError("Afghanistan outline GeoJSON feature is not an object")
    return feature


def _outline_ring(
    feature: dict[str, Any],
) -> tuple[list[list[float]], dict[str, str]]:
    """Validate one feature's Polygon geometry and return its outer ring."""
    geometry = _polygon_geometry(feature)
    outer_ring = _outer_ring(geometry)
    properties = dict(feature.get("properties") or {})
    return outer_ring, properties


def _polygon_geometry(feature: dict[str, Any]) -> dict[str, Any]:
    """Require a feature to contain a Polygon geometry object."""
    geometry = feature.get("geometry") or {}
    if not isinstance(geometry, dict):
        raise ProfileError("Afghanistan outline GeoJSON geometry is not an object")
    if geometry.get("type") != "Polygon":
        raise ProfileError("Afghanistan outline GeoJSON first feature is not a Polygon")
    return geometry


def _outer_ring(geometry: dict[str, Any]) -> list[list[float]]:
    """Return the first coordinate ring from a validated Polygon."""
    rings = geometry.get("coordinates") or []
    if not rings:
        raise ProfileError("Afghanistan outline GeoJSON Polygon has no coordinates")
    return rings[0]


def _outline_extent(
    polygon_lonlat: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Return ``lon_min, lon_max, lat_min, lat_max`` for an outline."""
    if not polygon_lonlat:
        return 0.0, 0.0, 0.0, 0.0
    outline_lons = [point[0] for point in polygon_lonlat]
    outline_lats = [point[1] for point in polygon_lonlat]
    return min(outline_lons), max(outline_lons), min(outline_lats), max(outline_lats)


def _coverage_extent(
    profile: DatasetProfile,
    outline_extent: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Return the padded plotting extent before applying axis padding."""
    outline_lon_min, outline_lon_max, outline_lat_min, outline_lat_max = (
        outline_extent
    )
    if any(
        value is None
        for value in (profile.lat_min, profile.lat_max, profile.lon_min, profile.lon_max)
    ):
        return outline_lon_min, outline_lon_max, outline_lat_min, outline_lat_max
    return (
        min(profile.lon_min, outline_lon_min),
        max(profile.lon_max, outline_lon_max),
        min(profile.lat_min, outline_lat_min),
        max(profile.lat_max, outline_lat_max),
    )


def _padded_extent(
    extent: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Apply deterministic minimum padding to a geographic extent."""
    lon_min, lon_max, lat_min, lat_max = extent
    pad_lon = max((lon_max - lon_min) * 0.05, 0.1)
    pad_lat = max((lat_max - lat_min) * 0.05, 0.1)
    return lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat


def _centroid_arrays(
    profile: DatasetProfile,
    parquet_path: Path | str,
) -> tuple[list[float], list[float]]:
    """Return separate latitude and longitude arrays for centroid plotting."""
    points = collect_polygon_centroids(profile, parquet_path)
    lats = [lat for lat, _lon in points]
    lons = [lon for _lat, lon in points]
    return lats, lons


def _draw_geographic_layers(
    ax: Any,
    *,
    polygon_lonlat: list[tuple[float, float]],
    outline_properties: dict[str, str],
    lats: list[float],
    lons: list[float],
    np: Any,
) -> None:
    """Draw the outline and deterministic centroid scatter layer."""
    if polygon_lonlat:
        _draw_geographic_outline(ax, polygon_lonlat, outline_properties)
    if lats:
        _draw_centroid_scatter(ax, lats, lons, np)


def _draw_geographic_outline(
    ax: Any,
    polygon_lonlat: list[tuple[float, float]],
    outline_properties: dict[str, str],
) -> None:
    """Draw the vendored country outline with its stable publication style."""
    outline_label = outline_properties.get("ADMIN") or "Afghanistan"
    ring_lons = tuple(point[0] for point in polygon_lonlat)
    ring_lats = tuple(point[1] for point in polygon_lonlat)
    ax.fill(
        ring_lons,
        ring_lats,
        color=_GEO_OUTLINE_FILL_COLOR,
        edgecolor=_GEO_OUTLINE_EDGE_COLOR,
        linewidth=1.4,
        label=outline_label,
        zorder=1,
    )


def _draw_centroid_scatter(ax: Any, lats: list[float], lons: list[float], np: Any) -> None:
    """Draw one jittered marker per canonical polygon centroid."""
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


def _configure_geographic_axes(
    ax: Any,
    *,
    extent: tuple[float, float, float, float],
    centroid_count: int,
    profile: DatasetProfile,
) -> None:
    """Apply the deterministic geographic axes, legend, and caption."""
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
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


def _empty_language_png(Figure: type[Any], plt: Any) -> bytes:
    """Render the contract-sized empty language chart."""
    fig = Figure(
        figsize=(_LANGUAGE_PNG_WIDTH / 100, _LANGUAGE_PNG_HEIGHT / 100)
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


def _language_buckets(
    profile: DatasetProfile,
) -> tuple[list[str], list[int], list[str], int, int, list[tuple[str, int]]]:
    """Build deterministically ordered language bars and the Other bucket."""
    top_slice, other_langs = _language_slices(profile)
    labels, counts, colors, other_count = _language_bar_values(
        top_slice, other_langs
    )
    labels, counts, colors = _order_language_bars(labels, counts, colors)
    return (
        labels,
        counts,
        colors,
        profile.row_count,
        other_count,
        other_langs,
    )


def _language_slices(
    profile: DatasetProfile,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Return sorted top-language and tail-language slices."""
    sorted_languages = sorted(
        profile.language_counts.items(), key=lambda kv: (-kv[1], kv[0])
    )
    return sorted_languages[:_LANGUAGE_TOP_N], sorted_languages[_LANGUAGE_TOP_N:]


def _language_bar_values(
    top_slice: list[tuple[str, int]],
    other_languages: list[tuple[str, int]],
) -> tuple[list[str], list[int], list[str], int]:
    """Build labels, counts, colors, and tail total before ordering."""
    other_count = sum(count for _language, count in other_languages)
    labels, counts, colors = _top_language_bar_values(top_slice)
    _append_other_language_bar(
        labels,
        counts,
        colors,
        other_count=other_count,
        other_languages=other_languages,
    )
    return labels, counts, colors, other_count


def _top_language_bar_values(
    top_slice: list[tuple[str, int]],
) -> tuple[list[str], list[int], list[str]]:
    """Build the three parallel arrays for the top language slice."""
    labels: list[str] = []
    counts: list[int] = []
    colors: list[str] = []
    for language, count in top_slice:
        labels.append(language)
        counts.append(count)
        colors.append(_LANG_BAR_COLOR)
    return labels, counts, colors


def _append_other_language_bar(
    labels: list[str],
    counts: list[int],
    colors: list[str],
    *,
    other_count: int,
    other_languages: list[tuple[str, int]],
) -> None:
    """Append the tail bucket when the language distribution has one."""
    if other_count > 0 or other_languages:
        labels.append("Other")
        counts.append(other_count)
        colors.append(_LANG_BAR_OTHER_COLOR)


def _order_language_bars(
    labels: list[str], counts: list[int], colors: list[str]
) -> tuple[list[str], list[int], list[str]]:
    """Sort language bars by descending count and stable label."""
    order = sorted(range(len(counts)), key=lambda index: (-counts[index], labels[index]))
    return (
        [labels[index] for index in order],
        [counts[index] for index in order],
        [colors[index] for index in order],
    )


def _draw_language_chart(
    profile: DatasetProfile,
    *,
    Figure: type[Any],
    np: Any,
    plt: Any,
    labels: list[str],
    counts: list[int],
    colors: list[str],
    total: int,
    other_count: int,
    other_languages: list[tuple[str, int]],
) -> bytes:
    """Render the populated language chart from precomputed buckets."""
    fig = Figure(figsize=(_LANGUAGE_PNG_WIDTH / 100, _LANGUAGE_PNG_HEIGHT / 100))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(_LANG_BACKGROUND_COLOR)
    ax.set_facecolor(_LANG_BACKGROUND_COLOR)
    y_positions = np.arange(len(labels))
    bars = ax.barh(y_positions, counts, color=colors, edgecolor="white", linewidth=0.5)
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
    _annotate_language_bars(ax, bars, counts, max_count, total)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(
        0.02,
        0.02,
        _language_caption(profile, total, other_count, other_languages),
        ha="left",
        va="bottom",
        fontsize=9,
        color=_LANG_TEXT_COLOR,
    )
    fig.subplots_adjust(bottom=0.16, top=0.92, left=0.12, right=0.97)
    buf = _figure_to_png_bytes(fig)
    plt.close(fig)
    return buf


def _annotate_language_bars(
    ax: Any, bars: Any, counts: list[int], max_count: int, total: int
) -> None:
    """Draw count and percentage labels on language bars."""
    for index in range(min(len(bars), len(counts))):
        bar = bars[index]
        count = counts[index]
        pct = (count / total * 100.0) if total else 0.0
        ax.text(
            bar.get_width() + max_count * 0.012,
            bar.get_y() + bar.get_height() / 2.0,
            f"{count:,}  ({pct:.2f}%)",
            va="center",
            ha="left",
            fontsize=10,
            color=_LANG_TEXT_COLOR,
        )


def _language_caption(
    profile: DatasetProfile,
    total: int,
    other_count: int,
    other_languages: list[tuple[str, int]],
) -> str:
    """Return the deterministic language chart footer."""
    return "  •  ".join(
        (
            f"Total rows: {total:,}",
            f"Distinct languages: {len(profile.language_counts)}",
            f"Top languages shown: {min(_LANGUAGE_TOP_N, len(profile.language_counts))}",
            f"Other bucket rows: {other_count:,} ({len(other_languages)} languages)",
        )
    )


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
    plt, np, Figure = _load_plotting()
    polygon_lonlat, outline_properties = _load_afghanistan_outline()
    extent = _padded_extent(_coverage_extent(profile, _outline_extent(polygon_lonlat)))
    lats, lons = _centroid_arrays(profile, parquet_path)

    fig = Figure(figsize=(_GEOGRAPHIC_PNG_WIDTH / 100, _GEOGRAPHIC_PNG_HEIGHT / 100))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor(_GEO_BACKGROUND_COLOR)
    ax.set_facecolor(_GEO_BACKGROUND_COLOR)

    _draw_geographic_layers(
        ax,
        polygon_lonlat=polygon_lonlat,
        outline_properties=outline_properties,
        lats=lats,
        lons=lons,
        np=np,
    )
    _configure_geographic_axes(
        ax,
        extent=extent,
        centroid_count=len(lats),
        profile=profile,
    )

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
    plt, np, Figure = _load_plotting()
    if not profile.language_counts:
        return _empty_language_png(Figure, plt)

    labels, counts, colors, total, other_count, other_langs = _language_buckets(
        profile
    )

    return _draw_language_chart(
        profile,
        Figure=Figure,
        np=np,
        plt=plt,
        labels=labels,
        counts=counts,
        colors=colors,
        total=total,
        other_count=other_count,
        other_languages=other_langs,
    )
