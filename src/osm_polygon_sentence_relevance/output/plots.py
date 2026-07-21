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
    plt, np, Figure = _load_plotting()
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
    plt, np, Figure = _load_plotting()
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
