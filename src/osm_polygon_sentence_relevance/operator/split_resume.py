"""Read-only inspection of streamed sentence-splitting checkpoints.

The splitter stores its authoritative progress on the Hugging Face staging
revision, unlike the labeler which stores paired local checkpoint files.  This
module keeps that difference out of the operator control flow and validates
the same metadata and Hub file hashes used by the streaming worker.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_sentence_relevance.operator.oar import ExitClass, JobStatus


class SplitResumeError(RuntimeError):
    """The split checkpoint inventory cannot be trusted for resumption."""


@dataclass(frozen=True, slots=True)
class SplitResumeInspection:
    """Validated split checkpoint inventory and worker exit evidence."""

    exit_code: int | None
    checkpoint_count: int
    total_shards: int
    identity_matches: bool

    @property
    def strictly_partial(self) -> bool:
        """Return whether at least one but not every shard is checkpointed."""

        return 0 < self.checkpoint_count < self.total_shards


def _read_exit_code(ssh: Any, exit_file: str) -> int | None:
    quoted = shlex.quote(exit_file)
    result = ssh.run(f"if test -f {quoted}; then cat {quoted}; fi")
    text = str(getattr(result, "text", None) or getattr(result, "stdout", ""))
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError as exc:
        raise SplitResumeError("remote split exit file is not an integer") from exc


def inspect_split_resume(
    *,
    ssh: Any,
    repo_id: str,
    input_repo_id: str,
    input_revision: str,
    run_id: str,
    source_commit: str,
    pipeline_version: str,
    model_name: str,
    batch_size: int,
    staging_revision: str,
    exit_file: str,
    cache_dir: Path,
    hub_api: Any | None = None,
) -> SplitResumeInspection:
    """Validate the split worker exit and every authoritative Hub checkpoint.

    Metadata is downloaded, but Parquet bytes are not.  The streaming
    offload verifier checks each metadata identity, byte size and Hub LFS
    digest before a checkpoint is counted.  The input polygon inventory
    supplies the total shard count without downloading the dataset.
    """

    exit_code = _read_exit_code(ssh, exit_file)
    try:
        if hub_api is None:
            from huggingface_hub import HfApi

            hub_api = HfApi()
        from scripts.streaming.driver import list_remote_shard_keys
        from scripts.streaming.offload import discover_run

        expected_identity = {
            "source_commit": source_commit,
            "input_dataset_revision": input_revision,
            "pipeline_version": pipeline_version,
            "model_name": model_name,
            "batch_size": batch_size,
        }
        handles = discover_run(
            hub_api=hub_api,
            repo_id=repo_id,
            run_id=run_id,
            staging_revision=staging_revision,
            local_cache_dir=cache_dir,
            expected_identity=expected_identity,
        )
        total_shards = len(
            list_remote_shard_keys(
                hub_api=hub_api,
                repo_id=input_repo_id,
                revision=input_revision,
            )
        )
    except Exception as exc:
        if isinstance(exc, SplitResumeError):
            raise
        raise SplitResumeError("could not validate split staging checkpoints") from exc

    return SplitResumeInspection(
        exit_code=exit_code,
        checkpoint_count=len(handles),
        total_shards=total_shards,
        identity_matches=True,
    )


def classify_split_terminal(
    status: JobStatus,
    inspection: SplitResumeInspection,
    *,
    exit_code: int | None,
) -> ExitClass:
    """Classify one split allocation without confusing it with label output."""

    if not inspection.identity_matches:
        return ExitClass.FAILED
    if inspection.checkpoint_count >= inspection.total_shards > 0:
        return ExitClass.COMPLETE if exit_code in {None, 0, 130} else ExitClass.FAILED
    # A validated partial inventory is the durable source of truth even when
    # the worker was killed after writing its last checkpoint and therefore
    # reported a nonzero process exit.  Resume from the next missing shard;
    # never discard already verified checkpoints because of that exit code.
    if inspection.strictly_partial:
        return ExitClass.CONTINUE
    if exit_code is not None and exit_code not in {0, 130}:
        return ExitClass.FAILED
    if status.message.casefold().find("cancel") >= 0:
        return ExitClass.CANCELLED
    return ExitClass.FAILED


def split_failure_reason(
    inspection: SplitResumeInspection, *, exit_code: int | None
) -> str:
    """Return a stable operator-facing reason for a split failure."""

    if not inspection.identity_matches:
        return "identity-mismatch"
    if exit_code is not None and exit_code not in {0, 130}:
        return "nonzero-exit"
    if inspection.checkpoint_count == 0:
        return "no-durable-work"
    return "checkpoint-progress-invalid"


__all__ = [
    "SplitResumeError",
    "SplitResumeInspection",
    "classify_split_terminal",
    "inspect_split_resume",
    "split_failure_reason",
]
