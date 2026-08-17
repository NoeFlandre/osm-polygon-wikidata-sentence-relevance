from __future__ import annotations

import pytest

from osm_polygon_sentence_relevance.labeling.releases import (
    V1_TRACKIO_SPACE_ID,
    V2_TRACKIO_SPACE_ID,
    ReleaseLane,
    checkpoint_prefix,
    release_lane,
    remote_release_path,
    trackio_space_id,
)


def test_legacy_identity_is_v1_and_uses_explicit_release_folder() -> None:
    assert release_lane({}) is ReleaseLane.V1_AFGHANISTAN
    assert remote_release_path(ReleaseLane.V1_AFGHANISTAN, "README.md") == (
        "v1-afghanistan/README.md"
    )
    assert trackio_space_id(ReleaseLane.V1_AFGHANISTAN) == V1_TRACKIO_SPACE_ID


def test_sampling_identity_is_worldwide_v2_on_same_main_tree() -> None:
    assert release_lane({"sampling_version": "labeling-v2"}) is ReleaseLane.V2_WORLDWIDE
    assert remote_release_path(ReleaseLane.V2_WORLDWIDE, "README.md") == (
        "v2-worldwide/README.md"
    )
    assert trackio_space_id(ReleaseLane.V2_WORLDWIDE) == V2_TRACKIO_SPACE_ID


def test_release_lane_rejects_unknown_explicit_lane() -> None:
    with pytest.raises(ValueError, match="release lane"):
        release_lane({"release_lane": "other"})


def test_checkpoint_prefix_is_run_scoped_and_same_main() -> None:
    assert checkpoint_prefix("a" * 20) == ".pipeline/checkpoints/" + "a" * 20


@pytest.mark.parametrize("run_id", ["A" * 20, "short", "a" * 19, "a" * 21])
def test_checkpoint_prefix_rejects_invalid_run_ids(run_id: str) -> None:
    with pytest.raises(ValueError, match="run ID"):
        checkpoint_prefix(run_id)
