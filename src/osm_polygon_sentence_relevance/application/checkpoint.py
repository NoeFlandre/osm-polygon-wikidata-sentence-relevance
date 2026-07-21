"""Stable checkpoint API backed by focused internal modules."""

from ._checkpoint.common import (
    HEARTBEAT_NAME,
    INVENTORY_QUARANTINE_DIR,
    SHARD_METADATA_NAME,
    SHARD_PARQUET_NAME,
    SHARDS_ACTIVE_DIRNAME,
    SHARDS_QUARANTINE_DIRNAME,
    SHARDS_STAGING_PREFIX,
    WORK_DIR_LOCK_NAME,
    CheckpointPublicationError,
    CheckpointValidationError,
    segmented_schema_sha256,
)
from ._checkpoint.inventory import (
    RunInventory,
    SourceFileEntry,
    compute_run_inventory,
    compute_shard_source_manifest,
    load_run_inventory,
    load_run_inventory_quarantine_first,
    reconcile_inventory,
    write_run_inventory,
)
from ._checkpoint.io import (  # noqa: F401 - private compatibility aliases
    _atomic_write_bytes,
    _atomic_write_parquet,
)
from ._checkpoint.locking import (
    WorkDirLock,
    acquire_work_dir_lock,
    release_work_dir_lock,
)
from ._checkpoint.storage import (
    _verify_pre_publish_manifest,  # noqa: F401 - used by pipeline compatibility
    load_shard_checkpoint,
    publish_shard_checkpoint,
    quarantine_shard_checkpoint,
    write_heartbeat,
)
from ._checkpoint.validation import (
    scan_active_directory,
    validate_checkpoint_metadata,
    validate_source_commit,
    validate_work_dir,
)

__all__ = [
    "CheckpointPublicationError",
    "CheckpointValidationError",
    "HEARTBEAT_NAME",
    "INVENTORY_QUARANTINE_DIR",
    "RunInventory",
    "SHARDS_ACTIVE_DIRNAME",
    "SHARDS_QUARANTINE_DIRNAME",
    "SHARDS_STAGING_PREFIX",
    "SHARD_METADATA_NAME",
    "segmented_schema_sha256",
    "SHARD_PARQUET_NAME",
    "SourceFileEntry",
    "WorkDirLock",
    "WORK_DIR_LOCK_NAME",
    "acquire_work_dir_lock",
    "compute_run_inventory",
    "compute_shard_source_manifest",
    "load_run_inventory",
    "load_run_inventory_quarantine_first",
    "load_shard_checkpoint",
    "publish_shard_checkpoint",
    "quarantine_shard_checkpoint",
    "reconcile_inventory",
    "release_work_dir_lock",
    "scan_active_directory",
    "validate_checkpoint_metadata",
    "validate_source_commit",
    "validate_work_dir",
    "write_heartbeat",
    "write_run_inventory",
]
