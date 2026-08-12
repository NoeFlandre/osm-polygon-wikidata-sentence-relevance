"""Prepare a Grid'5000 site for a durable continuation.

The CLI binds SSH, quota, staging, and llama-server seams at invocation time.
Keeping the preparation policy here makes same-site and cross-site resume
behavior testable without importing the Typer command module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from osm_polygon_sentence_relevance.operator.config import OperatorConfig, Stage
from osm_polygon_sentence_relevance.operator.ssh import SshClient
from osm_polygon_sentence_relevance.operator.state import RunPhase, StateStore
from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout


@dataclass(frozen=True, slots=True)
class RemotePreparationServices:
    """Replaceable side-effect seams used during continuation preparation."""

    ssh_factory: Callable[..., SshClient]
    remote_home: Callable[[SshClient], PurePosixPath]
    usage_policy_preflight: Callable[..., None]
    ensure_home_headroom: Callable[..., None]
    stager_type: Any
    stage_hf_token: Callable[..., None]
    oar_type: Any
    ensure_llama_server: Callable[..., int]
    label_staging_headroom_bytes: int
    submission_headroom_bytes: int


def prepare_destination(
    *,
    store: StateStore,
    config: OperatorConfig,
    site: str,
    relay_root: Path | None,
    poll_seconds: float,
    services: RemotePreparationServices,
) -> None:
    """Prepare a continuation site before allowing a new submission.

    Same-site continuation reuses the already validated checkout and assets.
    Cross-site continuation performs the normal policy, quota, checkout, and
    immutable-asset preflights before the validated relay is installed.
    """

    current = store.load()
    is_label = current.facts.get("active_stage") == Stage.LABEL.value
    if current.phase is not RunPhase.REMOTE_PREPARED:
        store.transition(
            expected=current.phase,
            target=RunPhase.REMOTE_PREPARED,
            facts={"site": site, "job_id": current.facts.get("job_id")},
        )

    ssh = services.ssh_factory(target=site, command_timeout=1800)
    home = services.remote_home(ssh)
    layout = RemoteLayout(home / "osm-polygon-operator" / config.run_id)
    services.usage_policy_preflight(ssh, site)
    services.ensure_home_headroom(
        ssh,
        protected_root=layout.root,
        minimum_headroom_bytes=services.label_staging_headroom_bytes,
    )

    # Refresh the managed checkout even for same-site continuation. A resumed
    # run may carry a newer behavior-preserving execution commit.
    stager = services.stager_type(ssh)
    stager.prepare(config, layout)
    services.stage_hf_token(stager, layout)
    if relay_root is not None and is_label:
        assets = stager.prepare_label_assets(config, layout, download_input=True)
        if not assets.llama_server_ready:

            def submission_preflight() -> None:
                services.usage_policy_preflight(ssh, site)
                services.ensure_home_headroom(
                    ssh,
                    protected_root=layout.root,
                    minimum_headroom_bytes=services.submission_headroom_bytes,
                )

            oar = services.oar_type(ssh, preflight=submission_preflight)
            services.ensure_llama_server(
                ssh,
                oar,
                store,
                layout,
                poll_seconds,
            )
    if relay_root is not None:
        store.transition(
            expected=RunPhase.REMOTE_PREPARED,
            target=RunPhase.REMOTE_PREPARED,
            facts={"resume_relay_root": str(relay_root)},
        )


__all__ = ["RemotePreparationServices", "prepare_destination"]
