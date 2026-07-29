"""Typed OAR lifecycle parsing and continuation classification."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from zoneinfo import ZoneInfo

from osm_polygon_sentence_relevance.operator.ssh import SshClient, SshRemoteError

_JOB_ID_RE = re.compile(r"(?:OAR_JOB_ID=|^)([1-9][0-9]*)$", re.MULTILINE)

#: Grid'5000 frontends report wall-clock times in the Europe/Paris zone.
GRID5000_TZ: ZoneInfo = ZoneInfo("Europe/Paris")
#: OAR emits ``scheduled_start`` either as an epoch integer or as a
#: ``YYYY-MM-DD HH:MM:SS`` wall-clock string already in frontend local time.
_TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}")


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
    scheduled_start: str | None = None
    walltime_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class CheckpointFacts:
    """Validated durable progress at allocation end."""

    completed: int
    total: int
    valid: bool
    interrupted: bool = False


class OarError(RuntimeError):
    """Invalid scheduler response or operation."""


#: OAR states that describe an allocation the operator must not duplicate.
LIVE_STATES: frozenset[JobState] = frozenset(
    {JobState.QUEUED, JobState.RUNNING, JobState.FINISHING}
)


def is_live_state(state: JobState) -> bool:
    """Return True for queued, running, or finishing allocations."""

    return state in LIVE_STATES


def _parse_scheduled_start(raw: object) -> str | None:
    """Render OAR's forecast start as an explicit Europe/Paris wall clock."""

    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        if raw <= 0:
            return None
        return datetime.fromtimestamp(raw, tz=GRID5000_TZ).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(raw, str) and _TIMESTAMP_RE.fullmatch(raw):
        return raw
    return None


def _parse_walltime(raw: object) -> int | None:
    """Parse OAR walltime expressed as seconds or an ``HH:MM:SS`` string."""

    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, str):
        matched = re.fullmatch(r"([0-9]+):([0-9]{2}):([0-9]{2})", raw)
        if matched is not None:
            hours, minutes, seconds = (int(group) for group in matched.groups())
            return hours * 3600 + minutes * 60 + seconds
        if raw.isdigit():
            return int(raw)
    return None


def format_walltime(seconds: int) -> str:
    """Format a strictly positive walltime in seconds as ``HH:MM:SS``.

    OAR reports the requested walltime of an allocation. A zero or negative
    walltime is never a valid request, so it is rejected here.
    """

    if seconds <= 0:
        raise ValueError("walltime must be strictly positive")
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if minutes >= 60 or secs >= 60:
        raise ValueError("walltime components must be within 00..59")
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_job_status(status: JobStatus) -> str:
    """Render one concise, factual scheduler status."""

    walltime = ""
    if status.walltime_seconds is not None and status.walltime_seconds > 0:
        walltime = f"; walltime {format_walltime(status.walltime_seconds)}"
    if status.state is JobState.QUEUED:
        if status.scheduled_start is not None:
            return (
                f"queued; scheduled start {status.scheduled_start} "
                f"Europe/Paris{walltime}"
            )
        return f"queued; scheduler has no start-time prediction{walltime}"
    if status.state is JobState.RUNNING:
        node = f" on {status.node}" if status.node else ""
        return f"running{node}{walltime}"
    if (
        status.state in {JobState.TERMINATED, JobState.ERROR, JobState.MISSING}
        and status.exit_code is not None
    ):
        return f"{status.state.value} (exit {status.exit_code})"
    return status.state.value


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

    def __init__(
        self,
        ssh: SshClient,
        *,
        preflight: Callable[[], None] | None = None,
    ) -> None:
        self._ssh = ssh
        self._preflight = preflight

    def submit(self, request: SubmissionRequest) -> int:
        if self._preflight is not None:
            self._preflight()
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
        scheduled_start = _parse_scheduled_start(record.get("scheduled_start"))
        return JobStatus(
            job_id=job_id,
            state=state,
            exit_code=exit_code,
            message=str(record.get("message", "")),
            node=record.get("assigned_network_address"),
            scheduled_start=scheduled_start,
            walltime_seconds=_parse_walltime(record.get("walltime")),
        )

    def cancel(self, job_id: int) -> None:
        if job_id <= 0:
            raise ValueError("job_id must be positive")
        self._ssh.run(f"oardel {job_id}")


__all__ = [
    "GRID5000_TZ",
    "LIVE_STATES",
    "CheckpointFacts",
    "ExitClass",
    "JobState",
    "JobStatus",
    "OarClient",
    "OarError",
    "SubmissionRequest",
    "classify_exit",
    "format_job_status",
    "format_walltime",
    "is_live_state",
    "parse_job_id",
]
