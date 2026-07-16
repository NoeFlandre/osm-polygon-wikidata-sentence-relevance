"""Immutable pipeline settings with path resolution.

``PipelineSettings`` is a frozen dataclass that resolves the local data
directory via a 3-level precedence chain without creating directories or
touching the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from osm_polygon_sentence_relevance.constants import (
    DEFAULT_INPUT_REVISION,
    INPUT_DATASET_ID,
    OUTPUT_DATASET_ID,
    PIPELINE_VERSION,
)
from osm_polygon_sentence_relevance.errors import ConfigurationError

# Path to the Seagate M3 external-drive location.
_SEAGATE_DATA_DIR = Path(
    "/Volumes/Seagate M3/projects/osm-polygon-wikidata-sentence-relevance"
)

# Repo-local fallback (relative to this file's grandparent: …/src/pkg/ → repo root).
_REPO_LOCAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class PipelineSettings:
    """Immutable pipeline configuration."""

    input_dataset: str
    input_revision: str
    output_dataset: str
    data_dir: Path
    pipeline_version: str

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        input_dataset: str = INPUT_DATASET_ID,
        input_revision: str = DEFAULT_INPUT_REVISION,
        output_dataset: str = OUTPUT_DATASET_ID,
        data_dir: Path | str | None = None,
        pipeline_version: str = PIPELINE_VERSION,
    ) -> PipelineSettings:
        """Build validated, immutable settings.

        Parameters
        ----------
        data_dir
            Explicit override.  When *None* the 3-level precedence applies:
            1. ``OSM_DATA_DIR`` environment variable
            2. Seagate M3 path (if it exists on the filesystem)
            3. Repo-local ``data/``

        Raises
        ------
        ConfigurationError
            If any dataset ID or revision is empty.
        """
        if not input_dataset or not input_dataset.strip():
            raise ConfigurationError("input_dataset must be non-empty")
        if not input_revision or not input_revision.strip():
            raise ConfigurationError("input_revision must be non-empty")
        if not output_dataset or not output_dataset.strip():
            raise ConfigurationError("output_dataset must be non-empty")

        resolved_dir = Path(data_dir) if data_dir is not None else _resolve_data_dir()

        return cls(
            input_dataset=input_dataset,
            input_revision=input_revision,
            output_dataset=output_dataset,
            data_dir=resolved_dir,
            pipeline_version=pipeline_version,
        )


def _resolve_data_dir() -> Path:
    """Apply the 3-level data-directory precedence.

    Does **not** create directories or access the network.
    """
    env_value = os.environ.get("OSM_DATA_DIR")
    if env_value:
        return Path(env_value)

    if _SEAGATE_DATA_DIR.exists():
        return _SEAGATE_DATA_DIR

    return _REPO_LOCAL_DATA_DIR
