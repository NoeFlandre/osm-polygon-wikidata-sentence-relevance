"""Factual Grid'5000 site availability parsing from OAR's inventory.

OAR exposes ``oarnodes -J`` as the authoritative resource inventory for a
frontend. This module parses that inventory to answer one factual question:
*is there at least one compatible GPU resource on this site that is currently
idle (Alive, no job assigned, sufficient memory and CUDA capability)?*

A site being ``queued_jobs == 0`` does not prove that claim: it only describes
the number of jobs this user observed ahead of a submission. Selecting a site
based on queue depth alone would invent a forecast OAR does not provide.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from osm_polygon_sentence_relevance.operator.sites import SiteRequirements

#: Command contract: emit one JSON array of OAR node records.
#: ``-J`` requests the JSON inventory; ``jq`` filters to GPU nodes that are
#: ``Alive`` and reports their busy state, memory, and CUDA capability.
_OARNODES_QUERY: Final[str] = (
    "oarnodes -J | jq -c '.[] | select(.gpu_count > 0 "
    'and .state == "Alive") | '
    "{state, jobs: (.jobs // [] | length), gpu_mem, "
    "gpu_compute_capability_major}'"
)


@dataclass(frozen=True, slots=True)
class GpuNode:
    """One Alive GPU resource parsed from ``oarnodes -J``."""

    gpu_memory_mb: int
    cuda_capability: tuple[int, int]
    jobs_assigned: int


@dataclass(frozen=True, slots=True)
class AvailabilityProbe:
    """Factual availability derived from the OAR inventory."""

    gpu_nodes: tuple[GpuNode, ...]

    def idle_compatible(self, requirements: SiteRequirements) -> bool:
        """True iff at least one GPU resource is currently idle and compatible.

        Idle means no OAR job is currently assigned to the resource (the
        inventory reports zero assigned jobs for it). This is a direct OAR
        observation, never a projection.
        """

        return any(
            node.jobs_assigned == 0
            and node.gpu_memory_mb >= requirements.gpu_memory_mb
            and node.cuda_capability >= requirements.cuda_capability
            for node in self.gpu_nodes
        )

    def meets(self, requirements: SiteRequirements) -> bool:
        """Return True iff any resource satisfies the hard compatibility checks."""

        return any(
            node.gpu_memory_mb >= requirements.gpu_memory_mb
            and node.cuda_capability >= requirements.cuda_capability
            for node in self.gpu_nodes
        )


def _coerce_int(value: object, *, minimum: int) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= minimum else None
    if isinstance(value, str) and value.isdigit():
        coerced = int(value)
        return coerced if coerced >= minimum else None
    return None


def parse_oarnodes_records(
    payload: object,
) -> tuple[GpuNode, ...]:
    """Parse the ``oarnodes -J | jq`` array into immutable GpuNode records."""

    raw_records: Sequence[object]
    if isinstance(payload, Mapping):
        # Some jq shapes emit one object per line rather than an array.
        raw_records = (payload,)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        raw_records = payload
    else:
        raise ValueError("oarnodes payload must be a mapping or sequence")
    nodes: list[GpuNode] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("oarnodes record must be a mapping")
        record = raw_record
        if str(record.get("state", "")) != "Alive":
            continue
        gpu_mem = _coerce_int(record.get("gpu_mem"), minimum=0)
        if gpu_mem is None:
            continue
        cuda_major = _coerce_int(record.get("gpu_compute_capability_major"), minimum=0)
        if cuda_major is None:
            continue
        jobs_assigned = _coerce_int(record.get("jobs"), minimum=0)
        if jobs_assigned is None:
            continue
        nodes.append(
            GpuNode(
                gpu_memory_mb=gpu_mem,
                cuda_capability=(cuda_major, 0),
                jobs_assigned=jobs_assigned,
            )
        )
    return tuple(nodes)


def parse_availability_stdout(stdout: str) -> AvailabilityProbe:
    """Decode the ``oarnodes -J | jq -c`` line-delimited JSON output."""

    nodes: list[GpuNode] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid oarnodes JSON") from exc
        nodes.extend(parse_oarnodes_records(payload))
    return AvailabilityProbe(tuple(nodes))


def availability_command() -> str:
    """Return the read-only command executed by the existing SSH transport."""

    return _OARNODES_QUERY


__all__ = [
    "AvailabilityProbe",
    "GpuNode",
    "availability_command",
    "parse_availability_stdout",
    "parse_oarnodes_records",
]
