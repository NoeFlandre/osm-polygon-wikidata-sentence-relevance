"""Source-file manifests and persisted run inventory."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_sentence_relevance.ingestion.discovery import RegionShardSet

from .common import (
    _DIR_MODE,
    _METADATA_SCHEMA_VERSION,
    INVENTORY_QUARANTINE_DIR,
    SHARDS_QUARANTINE_DIRNAME,
    CheckpointValidationError,
    _valid_shard_key,
)
from .io import _atomic_write_bytes, _fsync_dir_strict


@dataclass(frozen=True, slots=True)
class SourceFileEntry:
    """A single source-file binding inside a checkpoint manifest."""

    path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


def _relative_to_input_root(path: Path, input_root: Path) -> str:
    """Return ``path`` as a forward-slash relative path to ``input_root``."""
    rel = path.relative_to(input_root)
    return rel.as_posix()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def compute_shard_source_manifest(
    shard: RegionShardSet, *, input_root: Path
) -> list[SourceFileEntry]:
    """Return a sorted list of source-file bindings for ``shard``.

    The manifest is derived from the explicit per-shard file references
    held by the :class:`RegionShardSet` (four core Parquet files plus
    the optional Wikivoyage pair). Entries are sorted by the canonical
    forward-slash relative path.
    """
    input_root = input_root.expanduser().resolve(strict=False)
    candidates: list[Path] = [
        shard.polygons,
        shard.polygon_articles,
        shard.wikipedia_documents,
        shard.wikipedia_sections,
    ]
    if shard.wikivoyage_documents is not None:
        candidates.append(shard.wikivoyage_documents)
    if shard.wikivoyage_sections is not None:
        candidates.append(shard.wikivoyage_sections)

    entries: list[SourceFileEntry] = []
    for fpath in candidates:
        rel = _relative_to_input_root(fpath, input_root)
        entries.append(
            SourceFileEntry(
                path=rel,
                size=fpath.stat().st_size,
                sha256=_hash_file(fpath),
            )
        )
    entries.sort(key=lambda e: e.path)
    return entries


@dataclass(frozen=True, slots=True)
class RunInventory:
    """A run-level snapshot of the discovered shard set and per-shard
    source manifests. Built once at the start of each ``run_pipeline``
    invocation and used to reconcile against any prior inventory.
    """

    schema_version: int
    discovered_at_unix: int
    input_dataset_revision: str
    source_commit: str
    pipeline_version: str
    model_name: str
    batch_size: int
    shards: dict[str, list[SourceFileEntry]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "discovered_at_unix": int(self.discovered_at_unix),
            "input_dataset_revision": self.input_dataset_revision,
            "source_commit": self.source_commit,
            "pipeline_version": self.pipeline_version,
            "model_name": self.model_name,
            "batch_size": int(self.batch_size),
            "shards": {
                shard_key: [e.to_dict() for e in entries]
                for shard_key, entries in sorted(self.shards.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunInventory:
        shards_data = data.get("shards", {})
        shards: dict[str, list[SourceFileEntry]] = {}
        for shard_key, entries in shards_data.items():
            shards[shard_key] = [
                SourceFileEntry(
                    path=str(e["path"]),
                    size=int(e["size"]),
                    sha256=str(e["sha256"]).lower(),
                )
                for e in entries
            ]
        return cls(
            schema_version=int(data.get("schema_version", -1)),
            discovered_at_unix=int(data.get("discovered_at_unix", 0)),
            input_dataset_revision=str(data.get("input_dataset_revision", "")),
            source_commit=str(data.get("source_commit", "")),
            pipeline_version=str(data.get("pipeline_version", "")),
            model_name=str(data.get("model_name", "")),
            batch_size=int(data.get("batch_size", 0)),
            shards=shards,
        )


def compute_run_inventory(
    shards: tuple[RegionShardSet, ...],
    *,
    input_root: Path,
    input_dataset_revision: str,
    pipeline_version: str,
    source_commit: str,
    model_name: str,
    batch_size: int,
) -> RunInventory:
    """Build a :class:`RunInventory` from the discovered shards.

    Each shard's source manifest is computed **once** here. Downstream
    code reuses this snapshot rather than hashing again.
    """
    inventory_shards: dict[str, list[SourceFileEntry]] = {}
    for shard in shards:
        inventory_shards[shard.shard_key] = compute_shard_source_manifest(
            shard, input_root=input_root
        )
    return RunInventory(
        schema_version=_METADATA_SCHEMA_VERSION,
        discovered_at_unix=int(time.time()),
        input_dataset_revision=input_dataset_revision,
        source_commit=source_commit,
        pipeline_version=pipeline_version,
        model_name=model_name,
        batch_size=int(batch_size),
        shards=inventory_shards,
    )


def _inventory_manifests_equal(
    prior: list[SourceFileEntry] | None, current: list[SourceFileEntry]
) -> bool:
    return prior == current


def reconcile_inventory(
    prior: dict[str, list[SourceFileEntry]] | None,
    current: dict[str, list[SourceFileEntry]],
) -> dict[str, set[str]]:
    """Return a ``{added, removed, changed, unchanged}`` decision for
    each shard key, comparing two per-shard manifests.

    * ``added`` — present only in ``current``.
    * ``removed`` — present only in ``prior``.
    * ``changed`` — present in both, but with a different manifest.
    * ``unchanged`` — present in both, with identical manifest.

    ``prior`` may be ``None`` for the first run; in that case every
    shard in ``current`` is reported as ``added``.
    """
    added: set[str] = set()
    removed: set[str] = set()
    changed: set[str] = set()
    unchanged: set[str] = set()

    prior_keys = set(prior.keys()) if prior else set()
    current_keys = set(current.keys())

    for key in sorted(current_keys - prior_keys):
        added.add(key)
    for key in sorted(prior_keys - current_keys):
        removed.add(key)
    for key in sorted(prior_keys & current_keys):
        prior_manifest = (prior or {}).get(key)
        current_manifest = current.get(key)
        if prior_manifest is None or current_manifest is None:
            # ``key`` is in both ``prior`` and ``current`` so neither
            # lookup can return ``None`` by construction. The runtime
            # assertion is here only to satisfy the type checker; on
            # the Python level this branch is dead.
            raise CheckpointValidationError(
                f"reconcile_inventory: missing manifest for shared key {key!r}"
            )
        if _inventory_manifests_equal(prior_manifest, current_manifest):
            unchanged.add(key)
        else:
            changed.add(key)
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": unchanged,
    }


def _inventory_path(work_dir: Path) -> Path:
    return work_dir / "shards" / "inventory.json"


def _inventory_quarantine_path(work_dir: Path) -> Path:
    base = work_dir / "shards" / SHARDS_QUARANTINE_DIRNAME / INVENTORY_QUARANTINE_DIR
    base.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(base, _DIR_MODE)
    return base


def write_run_inventory(work_dir: Path, inventory: RunInventory) -> Path:
    """Atomically write the run-level inventory under ``work_dir``."""
    shards_root = work_dir / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    os.chmod(shards_root, _DIR_MODE)
    data = json.dumps(
        inventory.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    target = _inventory_path(work_dir)
    _atomic_write_bytes(target, data)
    return target


def _parse_inventory_payload(raw: bytes, *, source_path: Path) -> RunInventory:
    """Parse ``raw`` bytes as a JSON inventory payload."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(
            f"inventory at {source_path} is malformed JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise CheckpointValidationError(
            f"inventory at {source_path} is malformed: payload must be a mapping"
        )
    schema_version = int(data.get("schema_version", -1))
    if schema_version != _METADATA_SCHEMA_VERSION:
        raise CheckpointValidationError(
            f"inventory at {source_path} has schema_version={schema_version!r}, "
            f"expected {_METADATA_SCHEMA_VERSION}"
        )
    required = (
        "input_dataset_revision",
        "source_commit",
        "pipeline_version",
        "model_name",
        "batch_size",
        "discovered_at_unix",
        "shards",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise CheckpointValidationError(
            f"inventory at {source_path} is malformed: missing fields {missing}"
        )
    if not isinstance(data["shards"], dict):
        raise CheckpointValidationError(
            f"inventory at {source_path} is malformed: shards must be a mapping"
        )
    for key, entries in data["shards"].items():
        if not isinstance(key, str) or not _valid_shard_key(key):
            raise CheckpointValidationError(
                f"inventory at {source_path} is malformed: invalid shard_key {key!r}"
            )
        if not isinstance(entries, list):
            raise CheckpointValidationError(
                f"inventory at {source_path} is malformed: shards[{key!r}] "
                f"must be a list"
            )
    return RunInventory.from_dict(data)


def load_run_inventory(work_dir: Path) -> RunInventory | None:
    """Load the persisted inventory at ``work_dir`` if present.

    Returns ``None`` when no inventory exists. Malformed JSON, wrong
    schema, missing identity fields, or wrong shard-key shape raise
    :class:`CheckpointValidationError`. The corrupt file is **not**
    modified by this call.
    """
    path = _inventory_path(work_dir)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CheckpointValidationError(
            f"inventory at {path} could not be read: {exc}"
        ) from exc
    return _parse_inventory_payload(raw, source_path=path)


def load_run_inventory_quarantine_first(work_dir: Path) -> RunInventory | None:
    """Load the persisted inventory at ``work_dir``.

    If the inventory exists and parses cleanly, return it. If it is
    malformed, atomically move the bytes into
    ``shards/quarantine/inventory/`` (preserving them as evidence)
    and return ``None``. This lets the calling pipeline fall back to
    orphan-active recovery and complete the run in the same
    invocation, without a manual second run.

    If the rename itself fails (e.g. cross-filesystem ``EXDEV``), the
    file is left untouched and the underlying parse error is
    re-raised so the caller can decide what to do.
    """
    path = _inventory_path(work_dir)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CheckpointValidationError(
            f"inventory at {path} could not be read: {exc}"
        ) from exc
    try:
        return _parse_inventory_payload(raw, source_path=path)
    except CheckpointValidationError as parse_exc:
        # Move the malformed bytes aside so a fresh
        # ``write_run_inventory`` doesn't silently overwrite evidence.
        qdir = _inventory_quarantine_path(work_dir)
        ts = int(time.time())
        target = qdir / f"inventory.{ts}.{uuid.uuid4().hex[:8]}"
        try:
            os.rename(path, target)
        except OSError:
            # Could not move it (e.g. EXDEV). Re-raise the original
            # parse error so the caller sees the malformed state.
            raise parse_exc from None
        _fsync_dir_strict(qdir)
        # Quarantined successfully; signal "no prior inventory" to
        # the caller so it can continue via orphan-active recovery.
        return None
