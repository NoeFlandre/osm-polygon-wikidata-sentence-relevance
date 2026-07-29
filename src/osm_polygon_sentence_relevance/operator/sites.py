"""Deterministic Grid'5000 site compatibility and selection.

Selection is driven by factual hard constraints (reachability, GPU memory,
CUDA capability, persistent storage) and by the factual "an idle compatible
GPU resource is currently free on this frontend" observation. It never ranks
a site as immediately runnable from the user's queue count alone; it never
projects a start time. For queued batch jobs the operator submits exactly
once and reports OAR's forecast afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SiteProbe:
    """Observed capacity and factual availability for one Grid'5000 site."""

    name: str
    target: str
    reachable: bool
    gpu_memory_mb: int
    cuda_capability: tuple[int, int] | None
    persistent_free_bytes: int
    queued_jobs: int
    idle_compatible: bool = False
    has_managed_run: bool = False
    label_runtime_ready: bool = False


@dataclass(frozen=True, slots=True)
class SiteRequirements:
    """Hard constraints used before a site can be selected."""

    gpu_memory_mb: int = 40_000
    cuda_capability: tuple[int, int] = (7, 0)
    persistent_free_bytes: int = 8 * 1024**3
    resume_persistent_free_bytes: int = 512 * 1024**2


@dataclass(frozen=True, slots=True)
class SiteDecision:
    """Compatibility decision for a probed site."""

    probe: SiteProbe
    compatible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SiteSelection:
    """Selected site plus the complete deterministic decision record."""

    selected: SiteProbe
    decisions: tuple[SiteDecision, ...]


class NoCompatibleSiteError(RuntimeError):
    """No reachable site satisfies the requested hard constraints."""


def evaluate_site(probe: SiteProbe, requirements: SiteRequirements) -> SiteDecision:
    """Evaluate one probe without side effects."""

    reasons: list[str] = []
    if not probe.reachable:
        reasons.append("unreachable")
    if probe.gpu_memory_mb < requirements.gpu_memory_mb:
        reasons.append("insufficient_gpu_memory")
    if (
        probe.cuda_capability is None
        or probe.cuda_capability < requirements.cuda_capability
    ):
        reasons.append("insufficient_cuda_capability")
    required_storage = (
        requirements.resume_persistent_free_bytes
        if probe.has_managed_run
        else requirements.persistent_free_bytes
    )
    if probe.persistent_free_bytes < required_storage:
        reasons.append("insufficient_persistent_storage")
    if probe.queued_jobs < 0:
        reasons.append("invalid_queue_count")
    return SiteDecision(probe=probe, compatible=not reasons, reasons=tuple(reasons))


def select_site(
    probes: list[SiteProbe] | tuple[SiteProbe, ...],
    requirements: SiteRequirements | None = None,
) -> SiteSelection:
    """Choose the compatible site deterministically from factual observations.

    Compatibility is enforced first (reachable, GPU memory, CUDA capability,
    persistent storage). Among compatible sites, factual current idle
    availability from ``oarnodes`` wins, then deterministic name order. Queue
    depth is never used as a forecast. After submission, OAR's forecast is
    reported from ``JobStatus.scheduled_start`` -- it is a forecast, not a
    commitment.
    """

    if not probes:
        raise NoCompatibleSiteError("no Grid'5000 sites were probed")
    if requirements is None:
        requirements = SiteRequirements()
    decisions = tuple(
        evaluate_site(probe, requirements)
        for probe in sorted(probes, key=lambda item: item.name)
    )
    compatible = [decision.probe for decision in decisions if decision.compatible]
    if not compatible:
        raise NoCompatibleSiteError("no compatible Grid'5000 site is available")
    selected = min(
        compatible,
        key=lambda probe: (
            not probe.idle_compatible,
            not probe.has_managed_run,
            probe.name,
        ),
    )
    return SiteSelection(selected=selected, decisions=decisions)


__all__ = [
    "NoCompatibleSiteError",
    "SiteDecision",
    "SiteProbe",
    "SiteRequirements",
    "SiteSelection",
    "evaluate_site",
    "select_site",
]
