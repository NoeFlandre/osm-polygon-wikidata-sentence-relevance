"""Publication-level validation for the public export layout.

The publication validator runs alongside the existing
``validate_export_directory`` to enforce the additional publication
contracts:

* ``OUTPUT_SENTENCE_SCHEMA`` contains no Arrow ``map<...>`` field
  (Hugging Face Viewer compatibility);
* the on-disk PNG asset files exist and their SHA-256s match the
  manifest;
* every asset listed in the manifest exists on disk (no extras
  allowed);
* the README on disk equals the deterministic render of the
  ``DatasetProfile`` built from the export;
* the manifest's ``example_row`` matches the actual first row of
  the Parquet file (which is the canonical-sorted first row);
* the accounting identities (per ``compute_parquet_statistics``)
  hold for the versioned statistics object.

Use ``validate_publication_directory(path)`` to run all of the
above on an exported directory containing ``sentences.parquet``,
``manifest.json``, ``README.md``, and an ``assets/`` directory.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.contracts.errors import ExportError
from osm_polygon_sentence_relevance.contracts.schemas import OUTPUT_SENTENCE_SCHEMA
from osm_polygon_sentence_relevance.output.checksum import sha256_file
from osm_polygon_sentence_relevance.output.dataset_card import (
    render_dataset_card_from_profile,
    schema_has_map_types,
)
from osm_polygon_sentence_relevance.output.profile import (
    AssetInfo,
    DatasetProfile,
    ProfileError,
    build_dataset_profile,
)


def _derive_asset_base_url(manifest: Mapping[str, Any]) -> str | None:
    """Return the asset base URL the publication script would have used.

    The script derives the URL from the ``dataset_repo_id`` argument
    (defaulting to the canonical output repository) so the on-disk README
    uses absolute ``huggingface.co/.../assets`` image URLs.  The
    validator must reproduce the same choice to deterministically
    re-render the card.
    """
    repo_id = manifest.get("dataset_repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        return None
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/assets"


_PARQUET_NAME = "sentences.parquet"
_MANIFEST_NAME = "manifest.json"
_CARD_NAME = "README.md"
_ASSETS_DIR_NAME = "assets"

# Canonical contract for a published directory. The set of
# files that must appear in the directory at exactly these relative
# paths and no other files (apart from internal scratch/cache the
# publication pipeline strips before validation runs). This is the
# layout the corrective release enforces so the missing-parquet
# regression that affected commit ``2e8d68d9`` cannot recur.
_REQUIRED_PUBLICATION_FILES: tuple[str, ...] = (
    "sentences.parquet",
    "manifest.json",
    "README.md",
    "assets/geographic_coverage.png",
    "assets/language_distribution.png",
)
_REQUIRED_ASSET_NAMES: frozenset[str] = frozenset(
    {"geographic_coverage.png", "language_distribution.png"}
)

_REQUIRED_MANIFEST_KEYS: frozenset[str] = frozenset(
    {
        "manifest_version",
        "assets",
        "segmentation_model",
        "segmentation_revision",
        "source_commit",
        "row_count",
        "input_occurrence_count",
        "duplicates_removed",
        "counts_by_source",
        "counts_by_language",
        "counts_by_region",
        "input_dataset_revision",
        "pipeline_version",
        "input_dataset_id",
        "sha256",
        "statistics",
        "example_row",
        "rows_with_polygon_name",
        "lat_min",
        "lat_max",
        "lon_min",
        "lon_max",
        "sentence_length_min",
        "sentence_length_mean",
        "sentence_length_max",
    }
)


@dataclass(frozen=True, slots=True)
class ValidatedPublication:
    """Verified facts about a validated publication directory.

    All fields are derived from the disk-side artefacts after the
    validator has cross-checked them against the deterministic
    profile; nothing here is taken on trust from the manifest alone.
    """

    export_dir: Path
    parquet_path: Path
    manifest_path: Path
    card_path: Path
    assets_dir: Path
    asset_count: int
    profile_row_count: int
    profile: DatasetProfile


def compute_asset_sha(path: Path) -> str:
    """Return the lowercase hex SHA-256 of the file at *path*.

    Reads the file in one shot, which is acceptable because the
    current assets are bounded PNGs (a few hundred KB).  ``OSError``
    is re-raised as :class:`ExportError` so a missing or unreadable
    asset is an actionable validation failure.
    """
    try:
        with open(path, "rb") as fh:
            payload = fh.read()
    except OSError as err:
        raise ExportError(f"Cannot read asset {path!s}: {err}") from err
    return hashlib.sha256(payload).hexdigest().lower()


def first_parquet_row(parquet_path: Path) -> dict[str, object]:
    """Return the first full row of *parquet_path* as a dict.

    Used by the publication validator to cross-check the manifest's
    ``example_row``.  The Parquet file is globally sorted by
    ``(polygon_id, language, sentence_id)`` ascending by the
    finalisation step, so the first row is the canonical-sorted
    first occurrence.
    """
    table = pq.read_table(parquet_path, columns=None)
    if table.num_rows == 0:
        raise ExportError(f"Cannot read example row from empty Parquet {parquet_path}")
    values = table.slice(0, 1).to_pydict()
    return {col: values[col][0] for col in table.column_names}


def load_asset_inventory(
    export_dir: Path, *, assets_relative: str = _ASSETS_DIR_NAME
) -> dict[str, AssetInfo]:
    """Return the on-disk ``assets/`` directory as a name -> ``AssetInfo`` map.

    Files in the assets directory are matched against this map by
    filename; files that are not PNGs are recorded too (the manifest
    is the source of truth) but the validator will later require the
    manifest to declare every recorded file.
    """
    assets_dir = export_dir / assets_relative
    inventory: dict[str, AssetInfo] = {}
    if not assets_dir.is_dir():
        raise ExportError(f"Assets directory is missing: {assets_dir}")
    for entry in sorted(assets_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.is_symlink():
            raise ExportError(f"Asset must not be a symlink: {entry}")
        sha = compute_asset_sha(entry)
        inventory[entry.name] = AssetInfo(
            name=entry.name,
            sha256=sha,
            bytes_=entry.stat().st_size,
        )
    return inventory


def _require_manifest_string(manifest: Mapping[str, object], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExportError(f"Manifest field {key!r} must be a non-blank string")
    return value


def _check_accounting_identity(
    name: str, breakdown: Mapping[str, int], row_count: int
) -> None:
    total = sum(breakdown.values())
    if total != row_count:
        raise ExportError(
            f"Manifest {name} sums to {total} but row_count is {row_count}; "
            "the export is stale or the manifest was manually altered"
        )


def validate_publication_directory(
    path: str | Path,
    *,
    segmentation_model: str | None = None,
    segmentation_revision: str | None = None,
    source_commit: str | None = None,
    scratch_dir: str | Path | None = None,
) -> ValidatedPublication:
    """Validate a publication directory.

    Parameters
    ----------
    path
        The exported directory containing ``sentences.parquet``,
        ``manifest.json``, ``README.md``, and ``assets/``.
    segmentation_model, segmentation_revision, source_commit
        Optional overrides for the segmentation model name and exact
        revision, and the source-code commit hash to record on the
        profile.  When omitted, the manifest's recorded values are
        used (so the publication validator does not require external
        metadata).
    scratch_dir
        Optional path for the bounded-SQLite scratch used by
        :func:`build_dataset_profile`.  When ``None``, a fresh
        temporary directory is created by the underlying function.

    Returns
    -------
    ValidatedPublication
        Verified facts about the publication.

    Raises
    ------
    ExportError
        On every validation failure.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a str or pathlib.Path")
    export_dir = Path(path).resolve()
    if not export_dir.is_dir():
        raise ExportError(f"Export path is not a directory: {export_dir}")

    parquet_path = export_dir / _PARQUET_NAME
    manifest_path = export_dir / _MANIFEST_NAME
    card_path = export_dir / _CARD_NAME
    assets_dir = export_dir / _ASSETS_DIR_NAME

    for required in (parquet_path, manifest_path, card_path, assets_dir):
        if not required.exists():
            raise ExportError(f"Missing required artefact: {required}")

    # The existence checks above guarantee the path-shape branches
    # below; ``is_file()``/``is_dir()`` narrow the error to whichever
    # artefact is the wrong kind.
    if not parquet_path.is_file():
        raise ExportError(f"Parquet path is not a file: {parquet_path}")
    if not manifest_path.is_file():
        raise ExportError(f"Manifest path is not a file: {manifest_path}")
    if not card_path.is_file():
        raise ExportError(f"Card path is not a file: {card_path}")
    if not assets_dir.is_dir():
        raise ExportError(f"Assets path is not a directory: {assets_dir}")

    # Enforce the strict five-file publication contract. After this
    # gate every file in the directory must be one of the contract
    # artefacts; anything else (orphan files at the root, hidden
    # directories, sidecar parquets) is a publication regression.
    actual_relpaths: set[str] = set()
    for entry in sorted(export_dir.iterdir()):
        rel = entry.relative_to(export_dir).as_posix()
        if entry.is_dir():
            if rel == _ASSETS_DIR_NAME:
                for asset_entry in sorted(entry.iterdir()):
                    asset_rel = f"{rel}/{asset_entry.relative_to(entry).as_posix()}"
                    actual_relpaths.add(asset_rel)
            else:
                raise ExportError(
                    f"Publication directory contains unexpected subdirectory: {rel!r}"
                )
        elif entry.is_file():
            actual_relpaths.add(rel)
        elif entry.is_symlink():
            raise ExportError(f"Publication directory contains a symlink: {rel!r}")

    expected_relpaths = set(_REQUIRED_PUBLICATION_FILES)
    if actual_relpaths != expected_relpaths:
        missing = sorted(expected_relpaths - actual_relpaths)
        extra = sorted(actual_relpaths - expected_relpaths)
        problems: list[str] = []
        if missing:
            problems.append(f"missing={missing}")
        if extra:
            problems.append(f"extra={extra}")
        raise ExportError(
            "Publication directory does not match the canonical "
            "five-file contract: " + "; ".join(problems)
        )

    # Loader the manifest strictly.
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
    except (OSError, json.JSONDecodeError) as err:
        raise ExportError(f"Manifest is not readable: {err}") from err
    if not isinstance(manifest, dict):
        raise ExportError("Manifest must be a JSON object")

    version = manifest.get("manifest_version")
    if version != 2:
        raise ExportError(
            f"Manifest version {version!r} is not supported by the "
            "publication validator; expected 2"
        )

    missing_keys = sorted(set(_REQUIRED_MANIFEST_KEYS) - set(manifest.keys()))
    if missing_keys:
        raise ExportError(f"Manifest is missing required keys: {missing_keys}")

    # Parquet schema is identical to OUTPUT_SENTENCE_SCHEMA (we
    # already enforced this on the export side; the publication
    # validator re-checks here so a tampered parquet is rejected).
    try:
        parquet_file = pq.ParquetFile(parquet_path)
    except Exception as err:
        raise ExportError(f"Parquet file is unreadable: {err}") from err
    parquet_schema = parquet_file.schema_arrow
    if not parquet_schema.equals(OUTPUT_SENTENCE_SCHEMA):
        raise ExportError(
            "Parquet schema does not match OUTPUT_SENTENCE_SCHEMA; "
            "the publication cannot proceed"
        )

    # Cross-check no map types anywhere in the canonical schema. This
    # is a fixed property of OUTPUT_SENTENCE_SCHEMA but the check is
    # wired into the validator anyway so a future contributor who
    # adds a map type by mistake is blocked here.
    if schema_has_map_types(OUTPUT_SENTENCE_SCHEMA):
        raise ExportError(
            "OUTPUT_SENTENCE_SCHEMA contains a map<string, ...> field; "
            "the Hugging Face Viewer cannot ingest this"
        )

    # Determine segmentation metadata either from the manifest or
    # from the explicit overrides.
    seg_model = (
        segmentation_model
        if segmentation_model is not None
        else _require_manifest_string(manifest, "segmentation_model")
    )
    seg_rev = (
        segmentation_revision
        if segmentation_revision is not None
        else _require_manifest_string(manifest, "segmentation_revision")
    )
    src_commit = (
        source_commit
        if source_commit is not None
        else _require_manifest_string(manifest, "source_commit")
    )

    actual_sha = sha256_file(parquet_path)
    manifest_sha = _require_manifest_string(manifest, "sha256")
    if actual_sha.lower() != manifest_sha.lower():
        raise ExportError(
            f"Manifest sha {manifest_sha!r} does not match Parquet sha {actual_sha!r}"
        )

    # Build the profile and use it as the single source of truth.
    if scratch_dir is None:
        scratch_ctx = tempfile.TemporaryDirectory(prefix="pub-validate-")
        scratch = Path(scratch_ctx.name)
    else:
        scratch = Path(scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)
    try:
        try:
            profile = build_dataset_profile(
                parquet_path=parquet_path,
                parquet_sha256=manifest_sha,
                segmentation_model=seg_model,
                segmentation_revision=seg_rev,
                source_commit=src_commit,
                scratch_dir=scratch,
                input_dataset_id=manifest.get("input_dataset_id"),
            )
        except ProfileError as err:
            raise ExportError(f"Could not rebuild profile from Parquet: {err}") from err
    finally:
        if scratch_dir is None:
            scratch_ctx.cleanup()

    # Re-attach the assets to the profile from the manifest so the
    # card renderer has them (the manifest is the source of truth for
    # asset metadata, not the parquet).
    from dataclasses import replace

    asset_map: dict[str, AssetInfo] = {}
    for entry in manifest.get("assets", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        sha = entry.get("sha256")
        size = entry.get("bytes")
        if isinstance(name, str) and isinstance(sha, str) and isinstance(size, int):
            asset_map[name] = AssetInfo(name=name, sha256=sha, bytes_=size)
    if asset_map:
        profile = replace(profile, assets=asset_map)

    # Cross-check the manifest's quantitative fields against the
    # profile. The profile is derived from the parquet so the
    # manifest must agree byte-for-byte.
    _check_accounting_identity(
        "counts_by_source", manifest["counts_by_source"], profile.row_count
    )
    _check_accounting_identity(
        "counts_by_language", manifest["counts_by_language"], profile.row_count
    )
    _check_accounting_identity(
        "counts_by_region", manifest["counts_by_region"], profile.row_count
    )

    # Cross-check language breakdown vs profile one-by-one (so a
    # value-mutation that preserves the sum is still caught).
    if dict(manifest["counts_by_language"]) != dict(profile.language_counts):
        raise ExportError("Manifest counts_by_language disagrees with the profile")
    if dict(manifest["counts_by_source"]) != dict(profile.source_counts):
        raise ExportError("Manifest counts_by_source disagrees with the profile")
    if dict(manifest["counts_by_region"]) != dict(profile.region_counts):
        raise ExportError("Manifest counts_by_region disagrees with the profile")
    for key, expected in (
        ("input_occurrence_count", profile.input_occurrence_count),
        ("duplicates_removed", profile.duplicates_removed),
        (
            "cross_source_duplicate_groups",
            profile.cross_source_duplicate_groups,
        ),
    ):
        if manifest[key] != expected:
            raise ExportError(f"Manifest {key} disagrees with the profile")

    # Asset cross-check
    inventory = load_asset_inventory(export_dir)
    # Defensive manifest parsing: the v2 schema requires ``assets`` to
    # be a list of {"name", "sha256", "bytes"} objects.  The
    # surrounding schema guarantees this for a freshly-written
    # manifest, but a tampered manifest can violate the contract.
    manifest_assets = manifest["assets"]
    if not isinstance(manifest_assets, list):
        raise ExportError("Manifest 'assets' must be a JSON array")
    manifest_asset_map: dict[str, Mapping[str, object]] = {}
    for entry in manifest_assets:
        if not isinstance(entry, dict):
            raise ExportError("Manifest asset entries must be JSON objects")
        name = entry.get("name")
        sha = entry.get("sha256")
        if not isinstance(name, str) or not isinstance(sha, str):
            raise ExportError("Manifest asset entries must have a 'name' and 'sha256'")
        if not name or not sha:
            raise ExportError(
                "Manifest asset entries must have non-empty 'name' and 'sha256'"
            )
        manifest_asset_map[name] = entry

    if set(inventory) != set(manifest_asset_map):
        raise ExportError(
            "Asset set in manifest does not match the on-disk assets; "
            f"manifest={sorted(manifest_asset_map)}, "
            f"on-disk={sorted(inventory)}"
        )

    for name, info in inventory.items():
        manifest_entry = manifest_asset_map[name]
        if info.sha256 != str(manifest_entry.get("sha256")):
            raise ExportError(
                f"Asset {name!r} has manifest sha "
                f"{manifest_entry.get('sha256')!r} but on-disk sha is "
                f"{info.sha256!r}"
            )
        manifest_bytes = manifest_entry.get("bytes")
        if not isinstance(manifest_bytes, int) or manifest_bytes < 0:
            raise ExportError(f"Manifest asset {name!r} has invalid 'bytes' value")
        if manifest_bytes != info.bytes_:
            raise ExportError(
                f"Asset {name!r} has manifest bytes {manifest_bytes} "
                f"but on-disk bytes {info.bytes_}"
            )

    # Rebuild the dataset card from the profile and compare.
    try:
        card_text = card_path.read_text(encoding="utf-8")
    except OSError as err:
        raise ExportError(f"Card is not readable: {err}") from err
    expected_card = render_dataset_card_from_profile(
        profile, asset_base_url=_derive_asset_base_url(manifest)
    )
    if card_text != expected_card:
        raise ExportError(
            "Card on disk does not match the deterministic profile render; "
            "it is stale or manually edited"
        )

    # Cross-check the manifest's example row against the first row
    # of the parquet.  Use the same column-order normalisation the
    # renderer uses so byte-identical rows compare equal.
    manifest_example_row = manifest["example_row"]
    actual_example_row = first_parquet_row(parquet_path)
    for col in OUTPUT_SENTENCE_SCHEMA.names:
        manifest_value = manifest_example_row.get(col)
        actual_value = actual_example_row.get(col)
        if manifest_value != actual_value:
            raise ExportError(
                f"Manifest example row disagrees with Parquet for "
                f"column {col!r} (manifest={manifest_value!r}, "
                f"parquet={actual_value!r})"
            )

    return ValidatedPublication(
        export_dir=export_dir,
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        card_path=card_path,
        assets_dir=assets_dir,
        asset_count=len(inventory),
        profile_row_count=profile.row_count,
        profile=profile,
    )


__all__ = [
    "ValidatedPublication",
    "validate_publication_directory",
    "compute_asset_sha",
    "first_parquet_row",
    "load_asset_inventory",
]


# Re-export ExampleRow so external consumers can import it from this
# module without going through ``osm_polygon_sentence_relevance.output.profile``.
__all__.append("ExampleRow")
