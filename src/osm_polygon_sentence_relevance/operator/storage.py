"""Production Grid'5000 remote-storage management.

Owns the staging-headroom reservation, storage-only compatibility diagnosis,
managed-run cleanup (preview and execute), and home-quota headroom enforcement
that the operator CLI delegates to. All remote actions go through the injected
SSH transport.
"""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final

from osm_polygon_sentence_relevance.operator.config import Stage
from osm_polygon_sentence_relevance.operator.quota import read_home_quota
from osm_polygon_sentence_relevance.operator.sites import SiteProbe, SiteRequirements

if TYPE_CHECKING:
    from osm_polygon_sentence_relevance.operator.ssh import SshClient

LABEL_STAGING_HEADROOM_BYTES: Final[int] = 22 * 1024**3


def required_staging_headroom(stage: str, requested_bytes: int) -> int:
    """Reserve measured label-environment space before persistent staging."""

    if requested_bytes < 0:
        raise ValueError("requested storage headroom must be non-negative")
    if stage in {Stage.LABEL.value, Stage.ALL.value}:
        return max(requested_bytes, LABEL_STAGING_HEADROOM_BYTES)
    return requested_bytes


def cleanup_can_restore_compatibility(
    probes: list[SiteProbe],
    requirements: SiteRequirements,
) -> bool:
    """Return whether storage is the only failed hard constraint anywhere."""

    return any(
        probe.reachable
        and probe.gpu_memory_mb >= requirements.gpu_memory_mb
        and probe.cuda_capability is not None
        and probe.cuda_capability >= requirements.cuda_capability
        and probe.persistent_free_bytes < requirements.persistent_free_bytes
        for probe in probes
    )


def cleanup_managed_runs(
    ssh: SshClient,
    *,
    execute: bool,
    protected_root: PurePosixPath | None = None,
) -> tuple[str, ...]:
    """Preview or delete terminal pipeline-managed runs on one frontend.

    Confined to direct children of ``$HOME/osm-polygon-operator``. Only
    directories with a real ``.operator-managed.json`` marker whose status is
    ``complete`` or ``failed`` are eligible. Symlink candidates and markers are
    ignored, the managed root and the protected run root are never removed.
    """

    action = "delete" if execute else "preview"
    protected = str(protected_root) if protected_root is not None else ""
    script = f"""
set -euo pipefail
root="$HOME/osm-polygon-operator"
protected={shlex.quote(protected)}
[ -d "$root" ] || exit 0
find "$root" -mindepth 1 -maxdepth 1 -type d -print0 |
while IFS= read -r -d '' candidate; do
  [ ! -L "$candidate" ] || continue
  [ -z "$protected" ] || [ "$candidate" != "$protected" ] || continue
  marker="$candidate/.operator-managed.json"
  [ -f "$marker" ] && [ ! -L "$marker" ] || continue
  status=$(sed -n 's/.*"status":"\\([^"]*\\)".*/\\1/p' "$marker")
  case "$status" in complete|failed) ;; *) continue ;; esac
  if [ "$status" = failed ]; then
    resumable=$(find -P "$candidate" -type f \\( -name progress.json -o -path '*/checkpoints/*' \\) -print -quit)
    [ -z "$resumable" ] || continue
  fi
  printf '%s\\n' "$candidate"
  if [ {shlex.quote(action)} = delete ]; then rm -rf -- "$candidate"; fi
done
""".strip()
    result = ssh.run(script)
    return tuple(line for line in result.stdout.splitlines() if line)


def ensure_home_headroom(
    ssh: SshClient,
    *,
    protected_root: PurePosixPath,
    minimum_headroom_bytes: int,
) -> None:
    """Reclaim terminal managed runs, then fail closed above the soft quota."""

    if minimum_headroom_bytes < 0:
        raise ValueError("minimum storage headroom must be non-negative")
    quota = read_home_quota(ssh)
    if quota.soft_headroom_bytes >= minimum_headroom_bytes:
        return
    cleanup_managed_runs(ssh, execute=True, protected_root=protected_root)
    quota = read_home_quota(ssh)
    if quota.soft_headroom_bytes < minimum_headroom_bytes:
        raise RuntimeError("Grid'5000 home soft quota has insufficient safe headroom")


__all__ = [
    "LABEL_STAGING_HEADROOM_BYTES",
    "cleanup_can_restore_compatibility",
    "cleanup_managed_runs",
    "ensure_home_headroom",
    "required_staging_headroom",
]
