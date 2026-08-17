"""Release lanes for the public sentence-relevance dataset.

Both releases live on the Hugging Face ``main`` revision, each below its own
explicit folder. Checkpoint files use a private, run-scoped prefix on the same
revision and are never mixed with release files.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum


class ReleaseLane(StrEnum):
    """Public dataset release represented by one immutable run."""

    V1_AFGHANISTAN = "v1-afghanistan"
    V2_WORLDWIDE = "v2-worldwide"


OUTPUT_DATASET_ID = "NoeFlandre/osm-polygon-wikidata-sentence-relevance"
V1_TRACKIO_SPACE_ID = "NoeFlandre/afghanistan-labeling-trackio"
V2_TRACKIO_SPACE_ID = "NoeFlandre/worldwide-stratified-labeling-trackio"
V1_REMOTE_PREFIX = "v1-afghanistan"
V2_REMOTE_PREFIX = "v2-worldwide"
CHECKPOINT_REMOTE_PREFIX = ".pipeline/checkpoints"
_RUN_ID = re.compile(r"[0-9a-f]{20}")


def release_lane(identity: Mapping[str, object]) -> ReleaseLane:
    """Infer a release lane from the immutable labeling identity."""

    explicit = identity.get("release_lane")
    if explicit is not None:
        try:
            return ReleaseLane(str(explicit))
        except ValueError as exc:
            raise ValueError("release lane is invalid") from exc
    if identity.get("sampling_version") is not None:
        return ReleaseLane.V2_WORLDWIDE
    return ReleaseLane.V1_AFGHANISTAN


def release_prefix(lane: ReleaseLane) -> str:
    """Return the remote directory prefix for one release lane."""

    return V2_REMOTE_PREFIX if lane is ReleaseLane.V2_WORLDWIDE else V1_REMOTE_PREFIX


def remote_release_path(lane: ReleaseLane, relative_path: str) -> str:
    """Map a validated local artifact path into the single HF ``main`` tree."""

    prefix = release_prefix(lane)
    return f"{prefix}/{relative_path}" if prefix else relative_path


def checkpoint_prefix(run_id: str) -> str:
    """Return the same-main remote prefix for a run's durable checkpoints."""

    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run ID must be 20 lowercase hexadecimal characters")
    return f"{CHECKPOINT_REMOTE_PREFIX}/{run_id}"


def trackio_space_id(lane: ReleaseLane) -> str:
    """Return the Trackio Space dedicated to a release lane."""

    return (
        V2_TRACKIO_SPACE_ID if lane is ReleaseLane.V2_WORLDWIDE else V1_TRACKIO_SPACE_ID
    )


def trackio_space_url(lane: ReleaseLane) -> str:
    """Return the public Trackio dashboard URL for a release lane."""

    return f"https://huggingface.co/spaces/{trackio_space_id(lane)}"


__all__ = [
    "CHECKPOINT_REMOTE_PREFIX",
    "OUTPUT_DATASET_ID",
    "ReleaseLane",
    "V1_TRACKIO_SPACE_ID",
    "V1_REMOTE_PREFIX",
    "V2_REMOTE_PREFIX",
    "V2_TRACKIO_SPACE_ID",
    "checkpoint_prefix",
    "release_lane",
    "release_prefix",
    "remote_release_path",
    "trackio_space_id",
    "trackio_space_url",
]
