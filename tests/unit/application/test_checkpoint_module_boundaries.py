"""Ownership contracts for the checkpoint implementation modules."""

from osm_polygon_sentence_relevance.application import checkpoint
from osm_polygon_sentence_relevance.application._checkpoint import (
    inventory,
    locking,
    storage,
    validation,
)


def test_checkpoint_facade_reexports_canonical_owners() -> None:
    assert checkpoint.acquire_work_dir_lock is locking.acquire_work_dir_lock
    assert checkpoint.RunInventory is inventory.RunInventory
    assert (
        checkpoint.validate_checkpoint_metadata
        is validation.validate_checkpoint_metadata
    )
    assert checkpoint.publish_shard_checkpoint is storage.publish_shard_checkpoint
    assert checkpoint.load_shard_checkpoint is storage.load_shard_checkpoint
