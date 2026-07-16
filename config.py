"""
DEPRECATED — this module is superseded by the src-layout package.

Use instead:
    from osm_polygon_sentence_relevance.settings import PipelineSettings
    from osm_polygon_sentence_relevance.constants import INPUT_DATASET_ID, ...

This file is kept for backward-compatibility with any external references.

Original description:
Configuration module for the osm-polygon-wikidata-sentence-relevance project.
Manages the data directories, supporting local and external storage configurations.
"""

import os
import sys
from pathlib import Path

# The remote repositories for reference
GITHUB_REPO = (
    "https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance.git"
)
HUGGINGFACE_DATASET = (
    "https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance"
)

# Default external storage location requested by the user
DEFAULT_LOCAL_DATA_DIR = Path(
    "/Volumes/Seagate M3/projects/osm-polygon-wikidata-sentence-relevance"
)

# Local fallback directory inside the workspace (ignored by git)
FALLBACK_DATA_DIR = Path(__file__).resolve().parent / "data"


def get_data_dir() -> Path:
    """
    Resolves and returns the path to the data directory.

    1. Checks the `OSM_DATA_DIR` environment variable.
    2. Uses the external Seagate M3 volume path if it is available.
    3. Falls back to a local `data` directory in the project root.
    """
    # 1. Environment variable override
    env_dir = os.getenv("OSM_DATA_DIR")
    if env_dir:
        path = Path(env_dir)
        if not path.exists():
            print(
                f"Warning: OSM_DATA_DIR is set to '{path}' but it does not exist.",
                file=sys.stderr,
            )
        return path

    # 2. Check the primary Seagate M3 volume path
    if DEFAULT_LOCAL_DATA_DIR.exists():
        return DEFAULT_LOCAL_DATA_DIR

    # 3. Fallback to local workspace data folder
    print(
        f"Warning: Primary data directory '{DEFAULT_LOCAL_DATA_DIR}' is not accessible.\n"
        f"Make sure your external drive 'Seagate M3' is mounted.\n"
        f"Falling back to local directory: '{FALLBACK_DATA_DIR}'",
        file=sys.stderr,
    )
    return FALLBACK_DATA_DIR


if __name__ == "__main__":
    # Self-test/diagnostics
    print("Project Configuration Diagnostics:")
    print(f"  GitHub Repo: {GITHUB_REPO}")
    print(f"  HuggingFace Dataset: {HUGGINGFACE_DATASET}")
    print(f"  Primary Data Path: {DEFAULT_LOCAL_DATA_DIR}")
    print(f"  Fallback Data Path: {FALLBACK_DATA_DIR}")

    resolved_path = get_data_dir()
    print(f"\nCurrently resolved data directory: {resolved_path}")
    print(
        f"Data directory status: {'EXISTS' if resolved_path.exists() else 'NOT FOUND'}"
    )
