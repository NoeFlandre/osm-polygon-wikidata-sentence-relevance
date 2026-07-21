"""Shared checkpoint constants and public error types."""

from __future__ import annotations

import hashlib
import re

from osm_polygon_sentence_relevance.contracts.errors import CheckpointError
from osm_polygon_sentence_relevance.contracts.schemas import SEGMENTED_SENTENCES_SCHEMA

SHARD_PARQUET_NAME = "segmented.parquet"
SHARD_METADATA_NAME = "metadata.json"
SHARDS_ACTIVE_DIRNAME = "active"
SHARDS_QUARANTINE_DIRNAME = "quarantine"
SHARDS_STAGING_PREFIX = ".staging."
HEARTBEAT_NAME = "heartbeat.json"
WORK_DIR_LOCK_NAME = ".lock"
INVENTORY_QUARANTINE_DIR = "inventory"
_DIR_MODE = 0o700
_FILE_MODE = 0o600
_METADATA_SCHEMA_VERSION = 2
_LOWER_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_LOWER_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def _valid_shard_key(shard_key: str) -> bool:
    return (
        isinstance(shard_key, str)
        and bool(shard_key)
        and shard_key == shard_key.strip()
        and "/" not in shard_key
        and "\0" not in shard_key
        and ".." not in shard_key
    )


def segmented_schema_sha256() -> str:
    """Return the canonical SHA-256 fingerprint of the checkpoint schema."""

    return hashlib.sha256(
        SEGMENTED_SENTENCES_SCHEMA.serialize().to_pybytes()
    ).hexdigest()


class CheckpointValidationError(CheckpointError):
    """Raised when a checkpoint on disk is malformed, mismatched, or partial."""


class CheckpointPublicationError(CheckpointError):
    """Raised when a publish attempt fails. Active bytes are untouched."""
