"""Stateless auxiliary OAR job monitoring.

This module owns non-durable auxiliary OAR monitoring concerns: polling single
jobs, deduplicating scheduler transition messages, and optional streaming of
remote log output.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from osm_polygon_sentence_relevance.operator.oar import (
    JobState,
    JobStatus,
    OarClient,
    format_job_status,
)
from osm_polygon_sentence_relevance.operator.ssh import SshClient
from osm_polygon_sentence_relevance.operator.staging import RemoteLayout

#: Comparison key for deduplicating scheduler status updates.
type StatusKey = tuple[
    JobState,
    str | None,
    str | None,
    int | None,
    int | None,
]


def _default_emit_line(line: str) -> None:
    print(line, flush=True)


def report_job_status(
    status: JobStatus,
    previous: StatusKey | None,
    emit_line: Callable[[str], None] = _default_emit_line,
) -> StatusKey:
    """Print a scheduler transition once and return its comparison key.

    The comparison key includes every field the operator surfaces to the human
    reader so a walltime update (``HH:MM:SS`` rolls over) or an exit-code
    change produces exactly one fresh emission.
    """

    current: StatusKey = (
        status.state,
        status.node,
        status.scheduled_start,
        status.walltime_seconds,
        status.exit_code,
    )
    if current == previous:
        return current
    emit_line(f"[job {status.job_id}] {format_job_status(status)}")
    return current


def monitor_job_with_log(
    ssh: SshClient,
    oar: OarClient,
    layout: RemoteLayout,
    job_id: int,
    log_name: str,
    poll_seconds: float,
    *,
    emit_line: Callable[[str], None] = _default_emit_line,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Poll an auxiliary job while streaming log output to emission target.

    ``TERMINATED`` returns successfully regardless of the scheduler exit code;
    the caller separately validates payload exit artifacts. ``ERROR`` and
    ``MISSING`` raise :class:`RuntimeError`.
    """

    offset = 0
    previous: StatusKey | None = None
    log_path = str(layout.logs / str(job_id) / log_name)

    while True:
        status = oar.status(job_id)
        previous = report_job_status(status, previous, emit_line=emit_line)
        chunk = ssh.read_since(log_path, offset)
        if chunk.reset:
            offset = 0
        elif chunk.text:
            offset = chunk.next_offset
            for line in chunk.text.splitlines():
                emit_line(f"[job {job_id}] {line}")
        if status.state is JobState.TERMINATED:
            return
        if status.state in {JobState.ERROR, JobState.MISSING}:
            raise RuntimeError("remote allocation failed")
        sleeper(poll_seconds)


__all__ = [
    "StatusKey",
    "monitor_job_with_log",
    "report_job_status",
]
