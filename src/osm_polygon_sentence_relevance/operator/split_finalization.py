"""Finalize validated split checkpoints and hand off to labeling."""

from __future__ import annotations

import time

from osm_polygon_sentence_relevance.labeling.v2_contracts import (
    V2_LOGIT_PROMPT_VERSION,
)
from osm_polygon_sentence_relevance.operator.config import OperatorConfig, Scope, Stage
from osm_polygon_sentence_relevance.operator.job_monitor import monitor_job_with_log
from osm_polygon_sentence_relevance.operator.label_lanes import label_lane_plan
from osm_polygon_sentence_relevance.operator.oar import OarClient
from osm_polygon_sentence_relevance.operator.remote_completion import (
    assert_remote_exit_zero,
    mark_remote_status,
    publish_split,
)
from osm_polygon_sentence_relevance.operator.ssh import SshClient
from osm_polygon_sentence_relevance.operator.state import RunPhase, StateStore
from osm_polygon_sentence_relevance.operator.workflows import (
    RemoteLayout,
    split_finalization_submission,
)


def finalize_split_checkpointed(
    *,
    store: StateStore,
    config: OperatorConfig,
    ssh: SshClient,
    layout: RemoteLayout,
    oar: OarClient,
    poll_seconds: float,
) -> int:
    """Finalize complete split checkpoints and continue the workflow.

    This is shared by fresh and resumed runs. For ``stage=split`` it publishes
    the validated split release. For ``stage=all`` it preserves the finalized
    split output and reopens durable state for label submission.
    """

    final_job = oar.submit(split_finalization_submission(config, layout))
    store.transition(
        expected=RunPhase.CHECKPOINTED,
        target=RunPhase.FINALIZING,
        facts={"finalization_job_id": final_job},
    )
    print(f"Submitted finalization job {final_job}", flush=True)
    monitor_job_with_log(
        ssh,
        oar,
        layout,
        final_job,
        "finalize.stdout.log",
        poll_seconds,
        sleeper=time.sleep,
    )
    assert_remote_exit_zero(ssh, layout, final_job, "finalize.exit_code")
    store.transition(
        expected=RunPhase.FINALIZING,
        target=RunPhase.VALIDATED,
        facts={"split_output_job_id": final_job},
    )
    if config.stage is Stage.SPLIT:
        output_dir = layout.logs / str(final_job) / "output"
        hub_commit = publish_split(
            ssh,
            layout,
            output_dir,
            config.output_dataset_id,
        )
        store.transition(
            expected=RunPhase.VALIDATED,
            target=RunPhase.COMPLETE,
            facts={"published": True, "hub_commit": hub_commit},
        )
        print(f"Sentence splitting complete: run {config.run_id}", flush=True)
        mark_remote_status(ssh, layout, "complete")
    else:
        store.transition(
            expected=RunPhase.VALIDATED,
            target=RunPhase.REMOTE_PREPARED,
            facts={
                "active_stage": Stage.LABEL.value,
                **(
                    {
                        "label_lane": label_lane_plan(
                            config,
                            layout.root,
                            {},
                        ).lane.value
                    }
                    if config.scope is Scope.ALL
                    and config.prompt_version == V2_LOGIT_PROMPT_VERSION
                    else {}
                ),
            },
        )
    return final_job


__all__ = ["finalize_split_checkpointed"]
