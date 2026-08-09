"""Per-shard streaming driver.

For one ``shard_key`` the driver:

1. Downloads the six required parquets into
   ``${work_dir}/shards/inbox/<shard>/`` via
   :class:`scripts.streaming.downloader.PerFileHubDownloader`.
2. Calls ``process_single_shard`` (production facade) to publish a
   durable checkpoint under ``${work_dir}/shards/active/<shard>/``.
3. Validates the just-published checkpoint with strict
   ``load_shard_checkpoint``.
4. Offloads the checkpoint to a dedicated staging branch in the
   EXISTING output Hub repository via
   :class:`scripts.streaming.offload.CheckpointOffloader`.
5. On verified readback success, evicts BOTH the inbox subdir AND
   the local active slot for the shard.

No step may skip the previous one. Failures abort with both the
inbox and the active slot preserved.

Operational invariants:

* ``OAR_JOB_ID`` must be a numeric scheduler-owned value.
* The driver is bound to the resolved upstream commit; mismatches
  against ``state.json`` abort.
* The ``max_disk_bytes`` ceiling is enforced before each download
  call.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from osm_polygon_sentence_relevance.application.checkpoint import (
    load_shard_checkpoint,
)
from osm_polygon_sentence_relevance.application.pipeline import (
    process_single_shard,
)
from osm_polygon_sentence_relevance.ingestion.discovery import (
    discover_shards,
)
from osm_polygon_sentence_relevance.sentences.segmentation import SentenceSegmenter
from scripts.streaming.data_root import (
    DataRootRejected,
    check_data_root,
    discover_oar_scratch_dir,
)
from scripts.streaming.downloader import PerFileHubDownloader
from scripts.streaming.offload import (
    CheckpointOffloader,
    CheckpointOffloadError,
    OffloadHandle,
    inspect_remote_checkpoint,
)

log = logging.getLogger(__name__)


class DriverError(RuntimeError):
    """Top-level error raised by the streaming driver."""


class OarJobIdRequired(DriverError):
    """Raised when the driver is invoked outside an OAR allocation."""


def sat_split_kwargs(batch_size: int) -> dict[str, int]:
    """Return the bounded SaT batching profile used by streaming runs.

    ``process_single_shard`` already bounds the number of sections presented
    to the segmenter.  Matching SaT's internal batch size to that bound avoids
    an unnecessary second layer of 32-item micro-batches while preserving the
    exact input order and segmentation model decisions.
    """
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    return {"batch_size": batch_size, "outer_batch_size": 1000}


def _emit_shard_progress(
    shard_key: str,
    completed_sections: int,
    total_sections: int,
) -> None:
    """Emit one machine-readable heartbeat after a durable section batch."""

    print(
        json.dumps(
            {
                "completed_sections": completed_sections,
                "event": "shard-progress",
                "shard_key": shard_key,
                "total_sections": total_sections,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


# Per-shard inbox layout mirrors the production layout under
# ``input_root``. The six required parquet dirs use ``<shard_key>.parquet``
# filenames so ``process_single_shard`` and ``discover_shards``
# accept the staging inbox transparently.
_DOWNLOAD_LAYOUT: tuple[tuple[str, str], ...] = (
    ("polygons", "<shard>.parquet"),
    ("polygon_articles", "<shard>.parquet"),
    ("wikipedia/documents", "<shard>.parquet"),
    ("wikipedia/sections", "<shard>.parquet"),
    ("wikivoyage/documents", "<shard>.parquet"),
    ("wikivoyage/sections", "<shard>.parquet"),
)

_STATE_IDENTITY_FIELDS = (
    "repo_id",
    "resolved_revision",
    "source_commit",
    "run_id",
    "staging_revision",
    "pipeline_version",
    "model_name",
    "batch_size",
)


def list_remote_shard_keys(
    *,
    hub_api: Any,
    repo_id: str,
    revision: str,
    directory: str = "polygons",
    allow_empty: bool = False,
) -> list[str]:
    """Return a deterministic shard-key inventory from one table directory.

    ``HfApi.list_repo_tree`` owns pagination.  Only direct Parquet files
    under the pinned revision are accepted; folders and unrelated files
    are ignored.  The polygons table is mandatory for every processable
    shard and therefore provides the authoritative streaming inventory.
    """

    if not directory or directory.startswith("/") or ".." in directory.split("/"):
        raise ValueError("directory must be a relative repository path")
    directory = directory.rstrip("/")
    entries = hub_api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        path_in_repo=directory,
        recursive=False,
    )
    keys: set[str] = set()
    for entry in entries:
        path = getattr(entry, "path", "") or ""
        prefix = f"{directory}/"
        if not path.startswith(prefix) or not path.endswith(".parquet"):
            continue
        filename = path[len(prefix) :]
        if "/" in filename:
            continue
        key = filename.removesuffix(".parquet")
        if key:
            keys.add(key)
    if not keys and allow_empty:
        return []
    if not keys:
        if directory == "polygons":
            raise DriverError(
                f"no polygon shard files found for {repo_id!r} at {revision!r}"
            )
        raise DriverError(
            f"no shard files found under {directory!r} for {repo_id!r} at {revision!r}"
        )
    return sorted(keys)


@dataclass(frozen=True, slots=True)
class DriverConfig:
    """Configuration for a single driver invocation."""

    repo_id: str
    resolved_revision: str
    source_commit: str
    work_dir: Path
    input_root: Path
    upstream_repo_id: str
    run_id: str
    staging_revision: str
    offload_local_cache_dir: Path
    max_disk_bytes: int
    pipeline_version: str = "v1"
    model_name: str = "sat-12l-sm"
    batch_size: int = 128


class StreamDriver:
    """Single-stream per-shard driver."""

    def __init__(
        self,
        *,
        repo_id: str,
        resolved_revision: str,
        source_commit: str,
        work_dir: Path,
        input_root: Path,
        upstream_repo_id: str,
        hub_api: Any,
        run_id: str,
        staging_revision: str,
        offload_local_cache_dir: Path,
        max_disk_bytes: int,
        pipeline_version: str = "v1",
        model_name: str = "sat-12l-sm",
        batch_size: int = 128,
        local_input_root: Path | None = None,
        optional_shard_keys: frozenset[str] | set[str] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> None:
        if not isinstance(repo_id, str) or "/" not in repo_id or not repo_id.strip():
            raise ValueError("repo_id must be a non-blank owner/name string")
        if (
            not isinstance(resolved_revision, str)
            or len(resolved_revision) != 40
            or not all(c in "0123456789abcdef" for c in resolved_revision.lower())
        ):
            raise ValueError(
                "resolved_revision must be a 40-character lowercase hex SHA"
            )
        if (
            not isinstance(source_commit, str)
            or len(source_commit) != 40
            or not all(c in "0123456789abcdef" for c in source_commit.lower())
        ):
            raise ValueError("source_commit must be a 40-character lowercase hex SHA")

        self.cfg = DriverConfig(
            repo_id=repo_id,
            resolved_revision=resolved_revision.lower(),
            source_commit=source_commit.lower(),
            work_dir=Path(work_dir),
            input_root=Path(input_root),
            upstream_repo_id=upstream_repo_id,
            run_id=run_id,
            staging_revision=staging_revision,
            offload_local_cache_dir=Path(offload_local_cache_dir),
            max_disk_bytes=int(max_disk_bytes),
            pipeline_version=pipeline_version,
            model_name=model_name,
            batch_size=batch_size,
        )
        if hub_api is None:
            raise ValueError("hub_api must be supplied")
        self.hub_api = hub_api
        self._progress_callback = progress_callback
        self._verified_checkpoints: dict[str, dict[str, str | int]] = {}
        if optional_shard_keys is not None:
            if any(
                not isinstance(key, str) or not key.strip()
                for key in optional_shard_keys
            ):
                raise ValueError("optional_shard_keys must contain non-blank strings")
            self._optional_shard_keys: frozenset[str] | None = frozenset(
                optional_shard_keys
            )
        else:
            self._optional_shard_keys = None

        # OAR_JOB_ID guard: numeric string required
        oar_job_id = os.environ.get("OAR_JOB_ID")
        if not oar_job_id or not oar_job_id.strip() or not oar_job_id.strip().isdigit():
            raise OarJobIdRequired(
                "OAR_JOB_ID must be a non-blank numeric string; execution requires an allocated OAR compute job."
            )

        try:
            resolved_work = check_data_root(
                self.cfg.work_dir, role="work", min_free_bytes=0
            )
        except DataRootRejected as exc:
            if exc.reason == "TMP_FORBIDDEN":
                allocation_scratch = discover_oar_scratch_dir(min_free_bytes=0)
                candidate = Path(os.path.realpath(self.cfg.work_dir))
                try:
                    candidate.relative_to(allocation_scratch)
                except ValueError as relation_error:
                    raise DriverError(
                        "work_dir resolves to temporary storage outside the "
                        "current OAR allocation scratch directory"
                    ) from relation_error
                if not candidate.is_dir():
                    raise DriverError(
                        "allocation-local work_dir must already be a directory"
                    ) from exc
                resolved_work = candidate
            else:
                raise
        self.work_dir = resolved_work

        state_path = self.work_dir / "state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
            except json.JSONDecodeError as exc:
                raise DriverError(f"state.json is malformed JSON: {exc}") from exc
            if not isinstance(state, dict):
                raise DriverError("state.json must contain a JSON object")
            self._verified_checkpoints = self._load_verified_checkpoints(state)

    def has_verified_checkpoint(self, shard_key: str) -> bool:
        """Return whether this persistent work root already verified a shard."""
        return shard_key in self._verified_checkpoints

    def process_shard(
        self,
        shard_key: str,
        *,
        segmenter: SentenceSegmenter,
    ) -> OffloadHandle:
        """Run the full per-shard pipeline: check remote -> download -> process -> offload -> evict."""
        # Remote state is authoritative.  Corruption or identity drift must
        # abort loudly; treating either as "absent" could overwrite evidence
        # or silently mix two runs.
        expected_identity = {
            "source_commit": self.cfg.source_commit,
            "input_dataset_revision": self.cfg.resolved_revision,
            "pipeline_version": self.cfg.pipeline_version,
            "model_name": self.cfg.model_name,
            "batch_size": self.cfg.batch_size,
        }
        try:
            existing = inspect_remote_checkpoint(
                hub_api=self.hub_api,
                repo_id=self.cfg.repo_id,
                staging_revision=self.cfg.staging_revision,
                run_id=self.cfg.run_id,
                shard_key=shard_key,
                local_cache_dir=self.cfg.offload_local_cache_dir,
                expected_identity=expected_identity,
            )
        except CheckpointOffloadError as exc:
            raise DriverError(
                f"remote checkpoint validation failed for {shard_key}: {exc}"
            ) from exc
        if existing is not None:
            log.info("reusing verified remote checkpoint for %s", shard_key)
            self._evict_local_shard_state(shard_key)
            self._write_state(updated=True, completed=existing)
            return existing

        inbox = self.work_dir / "shards" / "inbox" / shard_key
        self._assert_disk_ceiling_or_raise()

        self._download_shard(shard_key=shard_key, inbox=inbox)
        try:
            shards = discover_shards(inbox)
            if not shards:
                raise DriverError(f"no shards discovered in staged inbox {inbox!r}")
            shard = next((s for s in shards if s.shard_key == shard_key), None)
            if shard is None:
                raise DriverError(
                    f"shard {shard_key!r} not present in staged inbox {inbox!r}"
                )

            self._assert_disk_ceiling_or_raise()

            progress_callback = (
                partial(self._progress_callback, shard_key)
                if self._progress_callback is not None
                else None
            )
            res = process_single_shard(
                shard=shard,
                input_root=self.work_dir / "shards" / "inbox",
                segmenter=segmenter,
                work_dir=self.work_dir,
                source_commit=self.cfg.source_commit,
                input_dataset_revision=self.cfg.resolved_revision,
                pipeline_version=self.cfg.pipeline_version,
                model_name=self.cfg.model_name,
                batch_size=self.cfg.batch_size,
                progress_callback=progress_callback,
            )
            if not res.published and not res.reused:
                raise DriverError(
                    f"process_single_shard returned no checkpoint for {shard_key}"
                )

            table, report, meta = load_shard_checkpoint(
                self.work_dir,
                shard_key,
                input_dataset_revision=self.cfg.resolved_revision,
                pipeline_version=self.cfg.pipeline_version,
                source_commit=self.cfg.source_commit,
                model_name=self.cfg.model_name,
                batch_size=self.cfg.batch_size,
                input_root=self.work_dir / "shards" / "inbox",
            )
            del table, report

            offloader = CheckpointOffloader(
                hub_api=self.hub_api,
                repo_id=self.cfg.repo_id,
                staging_revision=self.cfg.staging_revision,
                run_id=self.cfg.run_id,
                local_cache_dir=self.cfg.offload_local_cache_dir,
            )
            try:
                active_dir = self.work_dir / "shards" / "active" / shard_key
                handle = offloader.upload_and_verify(
                    shard_key=shard_key,
                    active_dir=active_dir,
                    metadata=meta,
                )
            except CheckpointOffloadError as exc:
                raise DriverError(f"offload failed for {shard_key}: {exc}") from exc

            # Strict eviction of BOTH local inbox and active checkpoint after verified readback
            self._evict_local_shard_state(shard_key)

            self._write_state(updated=True, completed=handle)
            return handle

        except Exception:
            self._write_state(updated=False)
            raise

    def evict_active(self, shard_key: str) -> None:
        """Remove per-shard active directory."""
        active = self.work_dir / "shards" / "active" / shard_key
        if active.exists():
            from scripts.streaming.data_root import safe_cleanup_scratch

            safe_cleanup_scratch(active, prefix_requirement="shards")

    def _evict_local_shard_state(self, shard_key: str) -> None:
        from scripts.streaming.data_root import safe_cleanup_scratch

        for path in (
            self.work_dir / "shards" / "inbox" / shard_key,
            self.work_dir / "shards" / "active" / shard_key,
            self.work_dir / "shards" / "partial" / shard_key,
        ):
            if path.exists() or path.is_symlink():
                safe_cleanup_scratch(path, prefix_requirement="shards")

    def _assert_disk_ceiling_or_raise(self) -> None:
        try:
            usage = shutil.disk_usage(str(self.work_dir))
        except OSError as exc:
            raise DriverError(
                f"disk_usage probe failed for {self.work_dir!r}: {exc}"
            ) from exc
        if usage.free < self.cfg.max_disk_bytes:
            raise DriverError(
                f"disk ceiling violated: free={usage.free} < max_disk_bytes={self.cfg.max_disk_bytes}"
            )

    def _download_shard(self, *, shard_key: str, inbox: Path) -> None:
        if inbox.exists():
            try:
                existing = discover_shards(inbox)
                if any(s.shard_key == shard_key for s in existing):
                    log.info("reusing partial inbox for %s at %s", shard_key, inbox)
                    return
            except Exception:
                log.warning("partial inbox at %s is corrupt; cleaning up", inbox)
            from scripts.streaming.data_root import safe_cleanup_scratch

            safe_cleanup_scratch(inbox, prefix_requirement="shards")

        inbox.mkdir(parents=True, exist_ok=True)
        dl = PerFileHubDownloader(
            repo_id=self.cfg.upstream_repo_id,
            resolved_revision=self.cfg.resolved_revision,
            target_dir=inbox,
            hub_api=self.hub_api,
            # The upstream revision is immutable.  Let the Hub cache resume
            # interrupted transfers instead of forcing a fresh copy after a
            # short allocation ends.
            force_download=False,
        )
        if self._optional_shard_keys is None:
            optional: dict[str, bool] = {}
            for subdir in ("wikivoyage/documents", "wikivoyage/sections"):
                filename = f"{subdir}/{shard_key}.parquet"
                try:
                    optional[subdir] = bool(
                        self.hub_api.file_exists(
                            repo_id=self.cfg.upstream_repo_id,
                            filename=filename,
                            revision=self.cfg.resolved_revision,
                            repo_type="dataset",
                        )
                    )
                except Exception as exc:
                    raise DriverError(
                        f"could not determine whether optional file exists: {filename}"
                    ) from exc
            if len(set(optional.values())) != 1:
                raise DriverError(
                    "wikivoyage documents and sections must either both exist or both be absent"
                )
            has_optional = optional["wikivoyage/documents"]
        else:
            has_optional = shard_key in self._optional_shard_keys

        download_paths: list[str] = []
        for subdir, fname in _DOWNLOAD_LAYOUT:
            if subdir.startswith("wikivoyage") and not has_optional:
                continue
            upstream_rel = f"{subdir}/{fname.replace('<shard>', shard_key)}"
            download_paths.append(upstream_rel)
        try:
            # Metadata is not consumed by this driver: the strict source
            # manifest is computed after staging.  Skipping the extra Hub
            # metadata request and transferring independent parquets in
            # parallel removes up to six extra metadata requests per shard while
            # preserving the same byte verification and checkpoint rules.
            dl.download_many(
                download_paths,
                max_workers=min(6, len(download_paths)),
                fetch_metadata=False,
            )
        except Exception as exc:
            failed_path = download_paths[0] if download_paths else shard_key
            raise DriverError(
                f"failed to download required file {failed_path}: {exc}"
            ) from exc

    def _state_identity(self) -> dict[str, str | int]:
        return {
            "repo_id": self.cfg.repo_id,
            "resolved_revision": self.cfg.resolved_revision,
            "source_commit": self.cfg.source_commit,
            "run_id": self.cfg.run_id,
            "staging_revision": self.cfg.staging_revision,
            "pipeline_version": self.cfg.pipeline_version,
            "model_name": self.cfg.model_name,
            "batch_size": self.cfg.batch_size,
        }

    def _load_verified_checkpoints(
        self, state: dict[str, Any]
    ) -> dict[str, dict[str, str | int]]:
        identity = self._state_identity()
        for field, expected in identity.items():
            if field in state and state[field] != expected:
                label = "pinned revision" if field == "resolved_revision" else field
                raise DriverError(
                    f"{label} mismatch: state.json has {state[field]!r}, "
                    f"config has {expected!r}"
                )
        raw = state.get("verified_checkpoints")
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise DriverError("state.json verified_checkpoints must be an object")
        if raw and any(field not in state for field in _STATE_IDENTITY_FIELDS):
            raise DriverError(
                "state.json checkpoint ledger is missing its complete run identity"
            )
        result: dict[str, dict[str, str | int]] = {}
        for shard_key, descriptor in raw.items():
            if not _valid_streaming_shard_key(shard_key) or not isinstance(
                descriptor, dict
            ):
                raise DriverError("state.json checkpoint ledger is malformed")
            if set(descriptor) != {
                "segmented_table_sha256",
                "segmented_table_bytes",
            }:
                raise DriverError("state.json checkpoint descriptor is malformed")
            digest = descriptor["segmented_table_sha256"]
            size = descriptor["segmented_table_bytes"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
            ):
                raise DriverError("state.json checkpoint descriptor is malformed")
            result[shard_key] = {
                "segmented_table_sha256": digest,
                "segmented_table_bytes": size,
            }
        return result

    def _write_state(
        self, *, updated: bool, completed: OffloadHandle | None = None
    ) -> None:
        state_path = self.work_dir / "state.json"
        state: dict[str, Any] = {}
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
            except json.JSONDecodeError:
                state = {}
        checkpoints = dict(self._verified_checkpoints)
        if completed is not None:
            if (
                completed.repo_id != self.cfg.repo_id
                or completed.run_id != self.cfg.run_id
                or completed.staging_revision != self.cfg.staging_revision
                or not _valid_streaming_shard_key(completed.shard_key)
                or completed.expected_table_sha256 != completed.computed_table_sha256
                or len(completed.computed_table_sha256) != 64
                or any(
                    char not in "0123456789abcdef"
                    for char in completed.computed_table_sha256
                )
                or isinstance(completed.table_bytes, bool)
                or completed.table_bytes <= 0
            ):
                raise DriverError("verified checkpoint handle does not match this run")
            checkpoints[completed.shard_key] = {
                "segmented_table_sha256": completed.computed_table_sha256,
                "segmented_table_bytes": completed.table_bytes,
            }
        state.update(self._state_identity())
        state["schema_version"] = 1
        state["last_updated"] = updated
        state["verified_checkpoints"] = checkpoints
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = state_path.with_name(f".{state_path.name}.tmp-{os.getpid()}")
        payload = json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            descriptor = os.open(
                temp_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, state_path)
            directory_fd = os.open(state_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self._verified_checkpoints = checkpoints
        finally:
            if temp_path.exists():
                temp_path.unlink()


def _valid_streaming_shard_key(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and value
        and all(
            (char.isascii() and char.isalnum() and char == char.lower())
            or char in "-_."
            for char in value
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Console entry point for the per-shard streaming driver."""
    import argparse

    parser = argparse.ArgumentParser(prog="scripts.streaming.driver")
    parser.add_argument("command", choices=["process-shard", "stream-build"])
    parser.add_argument("--confirm-offload", action="store_true", required=True)
    parser.add_argument("--shard")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--staging-revision", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--upstream-repo-id", required=True)
    parser.add_argument("--resolved-revision", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--input-root")
    parser.add_argument("--max-disk-bytes", type=int, default=1 << 30)
    parser.add_argument("--pipeline-version", default="v1")
    parser.add_argument("--model-name", default="sat-12l-sm")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cuda"], default="cuda")

    args = parser.parse_args(argv)

    if not args.confirm_offload:
        print("OPERATION_REQUIRES_EXPLICIT_APPROVAL")
        return 2

    if args.max_shards is not None and args.max_shards <= 0:
        parser.error("--max-shards must be a positive integer")
    if args.command == "process-shard" and not args.shard:
        parser.error("process-shard requires --shard")

    # Construct the guarded driver before loading Torch/model code.  Its
    # constructor rejects non-OAR execution and unsafe work roots.
    import huggingface_hub

    hub_api = huggingface_hub.HfApi()
    optional_shard_keys: frozenset[str] | None = None
    if args.command == "stream-build":
        document_keys = frozenset(
            list_remote_shard_keys(
                hub_api=hub_api,
                repo_id=args.upstream_repo_id,
                revision=args.resolved_revision,
                directory="wikivoyage/documents",
                allow_empty=True,
            )
        )
        section_keys = frozenset(
            list_remote_shard_keys(
                hub_api=hub_api,
                repo_id=args.upstream_repo_id,
                revision=args.resolved_revision,
                directory="wikivoyage/sections",
                allow_empty=True,
            )
        )
        if document_keys != section_keys:
            raise DriverError(
                "wikivoyage documents and sections must have identical shard inventories"
            )
        optional_shard_keys = document_keys
    driver = StreamDriver(
        repo_id=args.repo_id,
        resolved_revision=args.resolved_revision,
        source_commit=args.source_commit,
        work_dir=Path(args.work_dir),
        input_root=Path(args.input_root or args.work_dir),
        upstream_repo_id=args.upstream_repo_id,
        hub_api=hub_api,
        run_id=args.run_id,
        staging_revision=args.staging_revision,
        offload_local_cache_dir=Path(args.work_dir) / "cache",
        max_disk_bytes=args.max_disk_bytes,
        pipeline_version=args.pipeline_version,
        model_name=args.model_name,
        batch_size=args.batch_size,
        optional_shard_keys=optional_shard_keys,
        progress_callback=_emit_shard_progress,
    )

    if args.command == "process-shard":
        from osm_polygon_sentence_relevance.sentences.sat import SaTSentenceSegmenter

        segmenter = SaTSentenceSegmenter(
            model_name=args.model_name,
            split_kwargs=sat_split_kwargs(args.batch_size),
            device=args.device,
        )
        driver.process_shard(args.shard, segmenter=segmenter)
        print(f"OK: processed and offloaded {args.shard}")
        return 0

    if args.command == "stream-build":
        shard_keys = list_remote_shard_keys(
            hub_api=hub_api,
            repo_id=args.upstream_repo_id,
            revision=args.resolved_revision,
        )
        if args.shard:
            if args.shard not in shard_keys:
                parser.error(f"unknown shard: {args.shard}")
            shard_keys = [args.shard]
        if args.max_shards is not None:
            shard_keys = shard_keys[: args.max_shards]
        total = len(shard_keys)
        pending_shards = [
            shard_key
            for shard_key in shard_keys
            if driver.has_verified_checkpoint(shard_key) is not True
        ]
        previously_completed = total - len(pending_shards)
        print(
            json.dumps(
                {
                    "completed": previously_completed,
                    "event": "resume",
                    "remaining": len(pending_shards),
                    "total": total,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        if not pending_shards:
            return 0

        from osm_polygon_sentence_relevance.sentences.sat import SaTSentenceSegmenter

        segmenter = SaTSentenceSegmenter(
            model_name=args.model_name,
            split_kwargs=sat_split_kwargs(args.batch_size),
            device=args.device,
        )
        for completed_count, shard_key in enumerate(
            pending_shards,
            start=previously_completed + 1,
        ):
            driver.process_shard(shard_key, segmenter=segmenter)
            print(
                json.dumps(
                    {
                        "completed": completed_count,
                        "shard_key": shard_key,
                        "total": total,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        return 0

    return 2


if __name__ == "__main__":
    import sys

    sys.exit(main())


__all__ = [
    "DriverConfig",
    "DriverError",
    "OarJobIdRequired",
    "StreamDriver",
    "list_remote_shard_keys",
    "main",
    "sat_split_kwargs",
]
