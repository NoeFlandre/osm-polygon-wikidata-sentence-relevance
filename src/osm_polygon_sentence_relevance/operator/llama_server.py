"""CUDA llama-server build and recovery lifecycle management."""

from __future__ import annotations

import time
from collections.abc import Callable

from osm_polygon_sentence_relevance.operator.job_monitor import monitor_job_with_log
from osm_polygon_sentence_relevance.operator.oar import JobState, OarClient
from osm_polygon_sentence_relevance.operator.result_text import result_text
from osm_polygon_sentence_relevance.operator.ssh import SshClient
from osm_polygon_sentence_relevance.operator.state import RunPhase, StateStore
from osm_polygon_sentence_relevance.operator.workflows import (
    RemoteLayout,
    llama_build_submission,
)


def llama_server_ready(ssh: SshClient, layout: RemoteLayout) -> bool:
    """Check whether the persistent CUDA llama-server binary exists and is executable."""

    return (
        result_text(
            ssh.run(
                "if test -x "
                f"{layout.root!s}/llama-server-bin/llama-server; "
                "then printf yes; else printf no; fi"
            )
        )
        == "yes"
    )


def ensure_llama_server(
    ssh: SshClient,
    oar: OarClient,
    state: StateStore,
    layout: RemoteLayout,
    poll_seconds: float,
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    """Submit or reattach to the durable auxiliary CUDA build job."""

    durable = state.load()
    raw_job_id = durable.facts.get("llama_build_job_id")
    job_id = raw_job_id if type(raw_job_id) is int and raw_job_id > 0 else None
    if job_id is not None:
        status = oar.status(job_id)
        if status.state in {
            JobState.QUEUED,
            JobState.RUNNING,
            JobState.FINISHING,
        }:
            print(f"Reattaching to CUDA llama-server build job {job_id}", flush=True)
        elif (
            status.state is JobState.TERMINATED
            and status.exit_code in {None, 0}
            and llama_server_ready(ssh, layout)
        ):
            print(f"Reusing completed CUDA llama-server build job {job_id}", flush=True)
            return job_id
        else:
            job_id = None

    if job_id is None:
        current = state.load()
        if current.phase is not RunPhase.REMOTE_PREPARED:
            raise RuntimeError("CUDA build submission has invalid durable phase")
        print(
            "[operator] CUDA llama-server binary is absent; submitting its build",
            flush=True,
        )
        job_id = oar.submit(llama_build_submission(layout))
        state.transition(
            expected=RunPhase.REMOTE_PREPARED,
            target=RunPhase.REMOTE_PREPARED,
            facts={"llama_build_job_id": job_id},
        )
        print(f"Submitted CUDA llama-server build job {job_id}", flush=True)

    monitor_job_with_log(
        ssh,
        oar,
        layout,
        job_id,
        "build.stdout.log",
        poll_seconds,
        sleeper=sleeper,
    )
    if not llama_server_ready(ssh, layout):
        raise RuntimeError("CUDA llama-server build did not produce a binary")
    return job_id


__all__ = [
    "ensure_llama_server",
    "llama_server_ready",
]
