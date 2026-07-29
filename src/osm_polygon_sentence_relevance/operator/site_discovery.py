"""Grid'5000 site discovery: default sites and factual remote probing.

The CLI is a thin orchestrator; this module owns the read-only concern of
turning one Grid'5000 frontend into a :class:`SiteProbe`. Probing reads
``oarnodes -J`` via :mod:`sites_availability` so ``idle_compatible`` is a
direct OAR observation rather than a queue-depth projection. Queue depth is
recorded for diagnostics only and never drives compatibility or an ETA.
"""

from __future__ import annotations

import re
from typing import Final

from osm_polygon_sentence_relevance.operator.quota import QuotaUsage, read_home_quota
from osm_polygon_sentence_relevance.operator.relay_transport import (
    validate_safe_remote_path,
)
from osm_polygon_sentence_relevance.operator.sites import SiteProbe, SiteRequirements
from osm_polygon_sentence_relevance.operator.sites_availability import (
    AvailabilityProbe,
    availability_command,
    parse_availability_stdout,
)
from osm_polygon_sentence_relevance.operator.ssh import SshClient, SshError

DEFAULT_SITES: Final[tuple[str, ...]] = (
    "bordeaux",
    "grenoble",
    "lille",
    "louvain",
    "luxembourg",
    "lyon",
    "nancy",
    "nantes",
    "rennes",
    "sophia",
    "strasbourg",
    "toulouse",
)

_RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{20}")


def _validate_target(target: str) -> None:
    """Reject any character that could subvert a shell interpolation."""

    validate_safe_remote_path(target)


def _site_name(target: str) -> str:
    """Derive the Grid'5000 site name from an SSH target."""

    return target.split("@")[-1].split(".")[0]


def _aggregate_peak_gpu(
    probe: AvailabilityProbe,
) -> tuple[int, tuple[int, int] | None]:
    """Return the (memory_mb, capability) of the largest compatible node."""

    if not probe.gpu_nodes:
        return 0, None
    best = max(probe.gpu_nodes, key=lambda node: node.gpu_memory_mb)
    return best.gpu_memory_mb, best.cuda_capability


def _queue_depth(ssh: SshClient) -> int:
    """Count the user's current Waiting/Hold jobs (diagnostic only)."""

    command = "oarstat -u 2>/dev/null | awk '$5 ~ /Waiting|Hold/ {n++} END {print n+0}'"
    try:
        return int(ssh.run(command).stdout.strip())
    except (SshError, ValueError):
        return 0


def _parse_managed_probe(lines: list[str]) -> tuple[str, str, str]:
    """Validate the three-line df/managed/runtime probe output.

    Returns ``(free_kb, managed_flag, runtime_ready_flag)`` where both flags
    are exactly ``"0"`` or ``"1"``.
    """

    if len(lines) != 3:
        raise ValueError("invalid site probe output")
    free_kb_raw, managed_raw, runtime_ready_raw = (line.strip() for line in lines)
    if managed_raw not in {"0", "1"} or runtime_ready_raw not in {"0", "1"}:
        raise ValueError("invalid managed-run probe output")
    return free_kb_raw, managed_raw, runtime_ready_raw


def probe_site(
    target: str,
    run_id: str | None = None,
    requirements: SiteRequirements | None = None,
) -> SiteProbe:
    """Probe one Grid'5000 frontend for factual capacity and availability."""

    if run_id is not None and _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run ID must be twenty lowercase hexadecimal characters")
    _validate_target(target)
    ssh = SshClient(target=target, attempts=1, command_timeout=30)
    managed_probe = (
        f'run_root="$HOME/osm-polygon-operator/{run_id}"; '
        'test -f "$run_root/.operator-managed.json" && managed=1; '
        'test -x "$run_root/llama-server-bin/llama-server" && runtime_ready=1'
        if run_id is not None
        else ":"
    )
    probe = r"""
set -euo pipefail
command -v oarnodes >/dev/null
command -v jq >/dev/null
free_kb=$(df -Pk "$HOME" | awk 'NR==2 {print $4}')
managed=0
runtime_ready=0
__MANAGED_PROBE__
printf '%s\n%s\n%s\n' "$free_kb" "$managed" "$runtime_ready"
""".replace("__MANAGED_PROBE__", managed_probe).strip()
    try:
        avail = parse_availability_stdout(ssh.run(availability_command()).stdout)
        free_kb_raw, managed_raw, runtime_ready_raw = _parse_managed_probe(
            ssh.run(probe).stdout.splitlines()
        )
        quota: QuotaUsage = read_home_quota(ssh)
        gpu_memory, gpu_capability = _aggregate_peak_gpu(avail)
        idle = avail.idle_compatible(requirements or SiteRequirements())
        waiting = _queue_depth(ssh)
    except (SshError, ValueError, IndexError):
        return SiteProbe(
            name=_site_name(target),
            target=target,
            reachable=False,
            gpu_memory_mb=0,
            cuda_capability=None,
            persistent_free_bytes=0,
            queued_jobs=0,
        )
    return SiteProbe(
        name=_site_name(target),
        target=target,
        reachable=True,
        gpu_memory_mb=gpu_memory,
        cuda_capability=gpu_capability,
        persistent_free_bytes=min(
            int(free_kb_raw) * 1024,
            quota.soft_headroom_bytes,
        ),
        queued_jobs=waiting,
        idle_compatible=idle,
        has_managed_run=managed_raw == "1",
        label_runtime_ready=runtime_ready_raw == "1",
    )


__all__ = ["DEFAULT_SITES", "probe_site"]
