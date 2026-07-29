"""Classify a previously-recorded Grid'5000 allocation from its durable artifacts.

A stored allocation may be live (queued/running/finishing) or terminal
(terminated/error/missing). Live jobs are reattached and monitored without
submitting anything new. Terminal jobs are classified using the actual
artifacts the labeling CLI produces:

* nonzero payload exit -> deterministic or operational failure (never resumed)
* exit 0 with a final manifest under ``label-output/`` -> complete
* exit 0 without a final manifest and with strictly validated
  ``progress.json`` plus complete paired ``checkpoints/batch-NNNNNN.*``
  records where ``0 < completed < total`` -> resumable
* anything else with insufficient evidence -> fail safely

Resumability is never inferred from an exit code alone; the checkpoints and
progress metadata must validate against the recorded ``RunIdentity``.

The remote layout used here is the authoritative production layout written by
:class:`osm_polygon_sentence_relevance.labeling.checkpoint.CheckpointStore`:

* ``${label_work_root}/progress.json``
* ``${label_work_root}/timing.json`` when present
* ``${label_work_root}/checkpoints/batch-NNNNNN.parquet``
* ``${label_work_root}/checkpoints/batch-NNNNNN.json``

There is no ``${label_work_root}/checkpoints/<run_id>`` directory and no
separate ``batch-NNNNNN.parquet.sha256`` files; SHA-256 is computed on demand
from the Parquet bytes and compared against ``parquet_sha256`` in metadata.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from osm_polygon_sentence_relevance.operator.oar import (
    ExitClass,
    JobState,
    JobStatus,
)
from osm_polygon_sentence_relevance.operator.relay_transport import (
    validate_safe_remote_path,
)
from osm_polygon_sentence_relevance.operator.ssh import SshClient


def _ssh_text(ssh: SshClient, command: str) -> str:
    """Run ``command`` and return the result text regardless of shape."""

    result = ssh.run(command)
    text_attr = getattr(result, "text", None)
    if text_attr:
        return text_attr
    return getattr(result, "stdout", "")


class ResumeError(RuntimeError):
    """A recorded job cannot be safely classified or resumed."""


@dataclass(frozen=True, slots=True)
class ProgressFacts:
    """Validated progress record returned by a finished allocation."""

    completed: int
    total: int
    identity_matches: bool

    @property
    def strictly_partial(self) -> bool:
        return 0 < self.completed < self.total


@dataclass(frozen=True, slots=True)
class ResumeInspection:
    """All evidence required to classify a terminal allocation."""

    exit_code: int | None
    manifest_present: bool
    progress: ProgressFacts
    checkpoint_pairs: int
    checkpoint_parquet_shas_match: bool
    identity_matches: bool
    checkpoint_indexes: tuple[int, ...] = ()


class LiveReattach(RuntimeError):
    """Signals the caller that the recorded job is still live."""


def is_terminal(state: JobState) -> bool:
    """Return True for any OAR state that is no longer live."""

    return state in {
        JobState.TERMINATED,
        JobState.ERROR,
        JobState.MISSING,
    }


def _read_remote_text(ssh: SshClient, path: str) -> str:
    """Read the full text of a remote file or raise ResumeError."""

    chunk = ssh.read_since(path, 0)
    return str(getattr(chunk, "text", None) or getattr(chunk, "stdout", ""))


def _read_remote_bytes_digest(ssh: SshClient, remote_path: str) -> str:
    """Read SHA-256 digest of a remote file safely.

    Uses ``sha256sum --`` with the literal path. The path is required to be
    safe (no whitespace/quotes/control chars). The digest is the first token
    of the output. Raises :class:`ResumeError` on any failure.
    """

    validate_safe_remote_path(remote_path)
    chunk = ssh.run(f"sha256sum -- {remote_path}")
    text_attr = getattr(chunk, "text", None)
    text = (
        text_attr.strip()
        if text_attr is not None
        else getattr(chunk, "stdout", "").strip()
    )
    if not text:
        raise ResumeError(f"remote SHA-256 read returned no output: {remote_path}")
    digest = text.split()[0]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ResumeError(f"remote SHA-256 digest is malformed: {digest!r}")
    return digest


def _parse_batch_index(name: str) -> int | None:
    """Parse ``batch-NNNNNN.{parquet,json}``; return ``None`` for other names."""

    if not name.startswith("batch-"):
        return None
    stem, dot, ext = name.partition(".")
    if dot != "." or ext not in {"parquet", "json"}:
        return None
    digits = stem[len("batch-") :]
    if len(digits) != 6 or not digits.isdigit():
        return None
    return int(digits)


def _read_progress_payload(
    ssh: SshClient, label_work_root: str
) -> dict[str, object] | None:
    """Read and validate ``${label_work_root}/progress.json``.

    A missing file is represented by ``None`` because a payload can fail
    before its first checkpoint. Malformed, non-mapping content still raises
    :class:`ResumeError`. The production ``CheckpointStore`` writes
    progress.json at the label-work root, *not* under ``checkpoints/``.
    """

    raw = _read_remote_text(ssh, f"{label_work_root.rstrip('/')}/progress.json")
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ResumeError("progress.json is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ResumeError("progress.json payload is not a mapping")
    return dict(payload)


def _enumerate_remote_checkpoint_files(
    ssh: SshClient, label_work_root: str
) -> tuple[tuple[int, str, str], ...]:
    """Return ``(index, basename, kind)`` for every expected entry.

    Refuses unexpected entries (subdirectories, symlinks, non-paired files).
    The remote ``find`` is invoked with a literal pattern and rejects paths
    that contain shell metacharacters.
    """

    ckpts_dir = f"{label_work_root.rstrip('/')}/checkpoints"
    validate_safe_remote_path(ckpts_dir)
    listing = ssh.run(
        f"find {ckpts_dir} -mindepth 1 -maxdepth 1 "
        "-printf '%y\\t%f\\n' 2>/dev/null | sort"
    )
    result: list[tuple[int, str, str]] = []
    listing_attr = getattr(listing, "text", None)
    listing_text = (
        listing_attr if listing_attr is not None else getattr(listing, "stdout", "")
    )
    for line in listing_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        kind, separator, name = stripped.partition("\t")
        if separator != "\t" or kind != "f":
            raise ResumeError("remote checkpoint directory contains a non-file entry")
        idx = _parse_batch_index(name)
        if idx is None:
            raise ResumeError(f"unexpected remote checkpoint entry: {name}")
        kind = name.rsplit(".", 1)[1]
        result.append((idx, name, kind))
    return tuple(result)


def inspect_remote_resume(
    ssh: SshClient,
    *,
    label_work_root: str,
    label_output_root: str,
    expected_identity: Mapping[str, object],
    exit_file: str,
) -> ResumeInspection:
    """Read the durable artifacts a finished labeling allocation produced.

    This runs against the site where the allocation ran. It never transfers
    files. All checks are read-only. A failed validation raises
    :class:`ResumeError` so callers never silently resume on insufficient
    evidence.

    The remote layout is the exact production layout written by
    :class:`CheckpointStore`: ``progress.json`` at ``label_work_root``, with
    paired ``checkpoints/batch-NNNNNN.{parquet,json}`` files underneath.
    """

    validate_safe_remote_path(label_work_root)
    validate_safe_remote_path(label_output_root)
    validate_safe_remote_path(exit_file)

    exit_text = _read_remote_text(ssh, exit_file)
    try:
        exit_code = int(exit_text.strip()) if exit_text.strip() else None
    except ValueError as exc:
        raise ResumeError("remote exit file is not an integer") from exc

    manifest_probe = ssh.run(
        f"if test -f {label_output_root.rstrip('/')}/manifest.json; "
        "then printf yes; else printf no; fi"
    )
    manifest_attr = getattr(manifest_probe, "text", None)
    manifest_text = (
        manifest_attr.strip()
        if manifest_attr is not None
        else getattr(manifest_probe, "stdout", "").strip()
    )
    manifest_present = manifest_text == "yes"

    progress_payload = _read_progress_payload(ssh, label_work_root)
    if progress_payload is None:
        progress = ProgressFacts(completed=0, total=0, identity_matches=False)
        progress_identity_matches = False
    else:
        progress_identity = progress_payload.get("identity")
        progress_identity_matches = isinstance(progress_identity, Mapping) and dict(
            progress_identity
        ) == dict(expected_identity)
        try:
            completed = int(cast(int | str, progress_payload.get("completed", 0)))
        except (TypeError, ValueError) as exc:
            raise ResumeError("progress.json completed is invalid") from exc
        if "total" in progress_payload:
            try:
                total = int(cast(int | str, progress_payload["total"]))
            except (TypeError, ValueError) as exc:
                raise ResumeError("progress.json total is invalid") from exc
        elif "remaining" in progress_payload:
            try:
                remaining = int(cast(int | str, progress_payload["remaining"]))
            except (TypeError, ValueError) as exc:
                raise ResumeError("progress.json remaining is invalid") from exc
            total = completed + remaining
        else:
            raise ResumeError("progress.json missing total/remaining")
        if completed < 0 or total <= 0 or completed > total:
            raise ResumeError("progress.json counters are inconsistent")
        progress = ProgressFacts(
            completed=completed,
            total=total,
            identity_matches=progress_identity_matches,
        )

    entries = _enumerate_remote_checkpoint_files(ssh, label_work_root)
    if entries:
        indexes: set[int] = set()
        parquets: set[str] = set()
        jsons: set[str] = set()
        for idx, name, kind in entries:
            indexes.add(idx)
            if kind == "parquet":
                parquets.add(name)
            else:
                jsons.add(name)
        # Reject any index that has only one of (parquet, json).
        # Pair check per index:
        for idx in sorted(indexes):
            has_pq = f"batch-{idx:06d}.parquet" in parquets
            has_js = f"batch-{idx:06d}.json" in jsons
            if not (has_pq and has_js):
                raise ResumeError(f"checkpoint pair incomplete for batch-{idx:06d}")
        pair_count = len(indexes)
        ordered = tuple(sorted(indexes))
    else:
        pair_count = 0
        ordered = ()

    shas_match = True
    identity_match = True
    for idx in ordered:
        stem = f"batch-{idx:06d}"
        meta_remote = f"{label_work_root.rstrip('/')}/checkpoints/{stem}.json"
        meta_text = _read_remote_text(ssh, meta_remote)
        try:
            meta_payload = json.loads(meta_text)
        except json.JSONDecodeError as exc:
            raise ResumeError(f"checkpoint metadata is not valid JSON: {stem}") from exc
        if not isinstance(meta_payload, Mapping):
            raise ResumeError(f"checkpoint metadata is not a mapping: {stem}")
        if meta_payload.get("identity") != dict(expected_identity):
            identity_match = False
        expected_sha = meta_payload.get("parquet_sha256")
        if not isinstance(expected_sha, str):
            shas_match = False
            continue
        actual_sha = _read_remote_bytes_digest(
            ssh,
            f"{label_work_root.rstrip('/')}/checkpoints/{stem}.parquet",
        )
        if actual_sha.lower() != expected_sha.strip().lower():
            shas_match = False

    return ResumeInspection(
        exit_code=exit_code,
        manifest_present=manifest_present,
        progress=progress,
        checkpoint_pairs=pair_count,
        checkpoint_parquet_shas_match=shas_match,
        identity_matches=identity_match and progress_identity_matches,
        checkpoint_indexes=ordered,
    )


def classify_terminal(
    status: JobStatus,
    inspection: ResumeInspection,
) -> ExitClass:
    """Return the authoritative classification of a terminal allocation."""

    if inspection.exit_code is None:
        return ExitClass.FAILED
    if inspection.exit_code != 0:
        return ExitClass.FAILED
    if not inspection.identity_matches:
        return ExitClass.FAILED
    if inspection.manifest_present:
        return ExitClass.COMPLETE
    if (
        inspection.progress.strictly_partial
        and inspection.checkpoint_parquet_shas_match
        and inspection.checkpoint_pairs > 0
    ):
        return ExitClass.CONTINUE
    if status.state is JobState.MISSING and inspection.checkpoint_pairs == 0:
        raise ResumeError(
            "missing durable evidence: no manifest, no checkpoints, no progress"
        )
    return ExitClass.FAILED


__all__ = [
    "LiveReattach",
    "ProgressFacts",
    "ResumeError",
    "ResumeInspection",
    "classify_terminal",
    "inspect_remote_resume",
    "is_terminal",
]
