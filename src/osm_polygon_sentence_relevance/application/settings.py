"""Immutable pipeline settings with portable path resolution.

``PipelineSettings`` is a frozen dataclass that resolves the local data
directory via a 3-level precedence chain **without** creating directories
or touching the network:

1. explicit ``data_dir`` argument;
2. nonblank ``OSM_DATA_DIR`` environment variable (whitespace-only is
   treated as unset);
3. ``Path.cwd() / "data"``.

The implementation deliberately avoids probing personal or platform-specific
mount points (no external-drive detection).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from osm_polygon_sentence_relevance.contracts.constants import (
    DEFAULT_INPUT_REVISION,
    INPUT_DATASET_ID,
    OUTPUT_DATASET_ID,
    PIPELINE_VERSION,
)
from osm_polygon_sentence_relevance.contracts.errors import ConfigurationError


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
            1. ``OSM_DATA_DIR`` environment variable (if nonblank)
            2. ``Path.cwd() / "data"``

        Raises
        ------
        ConfigurationError
            If any dataset ID or revision is empty (after stripping).
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
    """Apply the data-directory precedence.

    Returns the first of: shell-expanded ``OSM_DATA_DIR`` (if nonblank after
    stripping), then ``Path.cwd() / "data"``. Does **not** create directories
    or access the network.
    """
    # os.path.expanduser expands a leading ~ so ``OSM_DATA_DIR=~/data`` works.
    env_value = (os.environ.get("OSM_DATA_DIR") or "").strip()
    if env_value:
        return Path(os.path.expanduser(env_value))

    return Path.cwd() / "data"


__all__ = ["PipelineSettings"]
