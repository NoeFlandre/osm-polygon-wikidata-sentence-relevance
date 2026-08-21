from __future__ import annotations

from osm_polygon_sentence_relevance.application._checkpoint.common import (
    _safe_shard_text,
)


def test_safe_shard_text_rejects_path_like_values() -> None:
    assert _safe_shard_text("region-a")
    assert not _safe_shard_text("../region-a")
    assert not _safe_shard_text("bad\0name")
    assert not _safe_shard_text("bad..name")
