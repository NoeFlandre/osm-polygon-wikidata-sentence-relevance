"""Local Parquet shard discovery.

Scans only the six allowlisted subdirectories for ``.parquet`` files,
groups them by shard key (filename stem), and validates that each shard
has the required core files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from osm_polygon_sentence_relevance.constants import ALLOWED_INPUT_PATHS
from osm_polygon_sentence_relevance.errors import ShardDiscoveryError

# Mapping from ALLOWED_INPUT_PATHS entries to logical table names.
_PATH_TO_TABLE: dict[str, str] = {
    "polygons": "polygons",
    "polygon_articles": "polygon_articles",
    "wikipedia/documents": "wikipedia_documents",
    "wikipedia/sections": "wikipedia_sections",
    "wikivoyage/documents": "wikivoyage_documents",
    "wikivoyage/sections": "wikivoyage_sections",
}

# Core tables that must be present for a processable shard.
_CORE_TABLES = frozenset(
    {
        "polygons",
        "polygon_articles",
        "wikipedia_documents",
        "wikipedia_sections",
    }
)

# Wikivoyage pair — must be both present or both absent.
_WIKIVOYAGE_TABLES = frozenset(
    {
        "wikivoyage_documents",
        "wikivoyage_sections",
    }
)


@dataclass(frozen=True)
class RegionShardSet:
    """Immutable set of Parquet file paths for one regional shard."""

    shard_key: str
    polygons: Path
    polygon_articles: Path
    wikipedia_documents: Path
    wikipedia_sections: Path
    wikivoyage_documents: Path | None
    wikivoyage_sections: Path | None


def discover_shards(root: Path) -> tuple[RegionShardSet, ...]:
    """Discover processable shard sets under *root*.

    Returns shard sets sorted lexicographically by shard key.
    An empty root returns an empty tuple.

    Raises
    ------
    ShardDiscoveryError
        If a shard is missing core files, or has only one of the
        Wikivoyage document/section pair.
    """
    # Scan each allowlisted subdirectory for .parquet files.
    # key → { logical_table_name → Path }
    shard_files: dict[str, dict[str, Path]] = {}

    for rel_path in ALLOWED_INPUT_PATHS:
        table_name = _PATH_TO_TABLE[rel_path]
        dirpath = root / rel_path
        if not dirpath.is_dir():
            continue
        for fpath in sorted(dirpath.iterdir()):
            if fpath.is_file() and fpath.suffix == ".parquet":
                shard_key = fpath.stem
                shard_files.setdefault(shard_key, {})[table_name] = fpath

    # Validate and build RegionShardSet for each shard key.
    result: list[RegionShardSet] = []
    for shard_key in sorted(shard_files):
        tables = shard_files[shard_key]

        # Check core tables.
        missing_core = _CORE_TABLES - tables.keys()
        if missing_core:
            raise ShardDiscoveryError(
                shard_key,
                f"missing core tables: {sorted(missing_core)}",
            )

        # Check Wikivoyage pair consistency.
        wv_present = _WIKIVOYAGE_TABLES & tables.keys()
        if len(wv_present) == 1:
            have = next(iter(wv_present))
            missing_wv = next(iter(_WIKIVOYAGE_TABLES - wv_present))
            raise ShardDiscoveryError(
                shard_key,
                f"incomplete wikivoyage pair: have {have!r} but missing {missing_wv!r}",
            )

        result.append(
            RegionShardSet(
                shard_key=shard_key,
                polygons=tables["polygons"],
                polygon_articles=tables["polygon_articles"],
                wikipedia_documents=tables["wikipedia_documents"],
                wikipedia_sections=tables["wikipedia_sections"],
                wikivoyage_documents=tables.get("wikivoyage_documents"),
                wikivoyage_sections=tables.get("wikivoyage_sections"),
            )
        )

    return tuple(result)
