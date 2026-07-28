"""Typed OAR lifecycle parsing and continuation classification."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from enum import StrEnum

from osm_polygon_sentence_relevance.operator.ssh import SshClient, SshRemoteError

_JOB_ID_RE = re.compile(r"(?:OAR_JOB_ID=|^)([1-9][0-9]*)$", re.MULTILINE)


class JobState(StrEnum):
    """Normalized OAR job state."""

    QUEUED = "queued"
    RUNNING = "running"
    FINISHING = "finishing"
    TERMINATED = "terminated"
    ERROR = "error"
    MISSING = "missing"


class ExitClass(StrEnum):
    """Controller action after a terminal allocation."""

    COMPLETE = "complete"
    CONTINUE = "continue"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SubmissionRequest:
    """One audited OAR command string."""

    command: tuple[str, ...]

    def shell_command(self) -> str:
        """Return POSIX-quoted argv for the Grid'5000 frontend."""

        if not self.command:
            raise ValueError("submission command cannot be empty")
        return " ".join(shlex.quote(value) for value in self.command)


@dataclass(frozen=True, slots=True)
class JobStatus:
    """Facts needed to monitor and classify an OAR job."""

    job_id: int
    state: JobState
    exit_code: int | None = None
    message: str = ""
    node: str | None = None


@dataclass(frozen=True, slots=True)
class CheckpointFacts:
    """Validated durable progress at allocation end."""

    completed: int
    total: int
    valid: bool
    interrupted: bool = False


class OarError(RuntimeError):
    """Invalid scheduler response or operation."""


def parse_job_id(output: str) -> int:
    """Extract one unambiguous positive OAR job ID."""

    matches = _JOB_ID_RE.findall(output.strip())
    if len(matches) != 1:
        raise OarError("submission did not return one job ID")
    return int(matches[0])


def classify_exit(status: JobStatus, checkpoint: CheckpointFacts) -> ExitClass:
    """Classify terminal state without retrying deterministic failures."""

    if checkpoint.valid and checkpoint.completed >= checkpoint.total > 0:
        return ExitClass.COMPLETE
    lowered = status.message.casefold()
    expected_walltime = "walltime" in lowered or "expected_walltime" in lowered
    if (
        checkpoint.valid
        and checkpoint.completed > 0
        and (checkpoint.interrupted or expected_walltime)
    ):
        return ExitClass.CONTINUE
    if "cancel" in lowered or "deleted" in lowered:
        return ExitClass.CANCELLED
    return ExitClass.FAILED


class OarClient:
    """OAR frontend adapter over the bounded SSH transport."""

    def __init__(self, ssh: SshClient) -> None:
        self._ssh = ssh

    def submit(self, request: SubmissionRequest) -> int:
        result = self._ssh.run(request.shell_command())
        return parse_job_id(result.stdout)

    def status(self, job_id: int) -> JobStatus:
        if job_id <= 0:
            raise ValueError("job_id must be positive")
        try:
            result = self._ssh.run(f"oarstat -fj {job_id} -J")
        except SshRemoteError as exc:
            if exc.returncode == 6:
                return JobStatus(job_id, JobState.MISSING)
            raise
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise OarError("invalid OAR status JSON") from exc
        record = payload.get(str(job_id), payload)
        raw_state = str(record.get("state", "")).casefold()
        states = {
            "waiting": JobState.QUEUED,
            "hold": JobState.QUEUED,
            "launching": JobState.QUEUED,
            "running": JobState.RUNNING,
            "finishing": JobState.FINISHING,
            "terminated": JobState.TERMINATED,
            "error": JobState.ERROR,
        }
        state = states.get(raw_state)
        if state is None:
            raise OarError("unsupported OAR job state")
        exit_code_raw = record.get("exit_code")
        exit_code = int(exit_code_raw) if exit_code_raw is not None else None
        return JobStatus(
            job_id=job_id,
            state=state,
            exit_code=exit_code,
            message=str(record.get("message", "")),
            node=record.get("assigned_network_address"),
        )

    def cancel(self, job_id: int) -> None:
        if job_id <= 0:
            raise ValueError("job_id must be positive")
        self._ssh.run(f"oardel {job_id}")


__all__ = [
    "CheckpointFacts",
    "ExitClass",
    "JobState",
    "JobStatus",
    "OarClient",
    "OarError",
    "SubmissionRequest",
    "classify_exit",
    "parse_job_id",
]
