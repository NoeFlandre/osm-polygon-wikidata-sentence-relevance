"""Typed command descriptions for retained Grid'5000 compute payloads."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

from osm_polygon_sentence_relevance.labeling.v2_contracts import V2_LOGIT_PROMPT_VERSION
from osm_polygon_sentence_relevance.operator.config import OperatorConfig, Scope, Stage
from osm_polygon_sentence_relevance.operator.label_lanes import (
    LabelLanePlan,
    label_lane_plan,
)
from osm_polygon_sentence_relevance.operator.oar import (
    SubmissionRequest,
    format_walltime,
)

DEFAULT_LABEL_WALLTIME_SECONDS = 3_300
MICRO_LABEL_WALLTIME_SECONDS = 1_200
DEFAULT_SPLIT_WALLTIME_SECONDS = 1_800
_LABEL_GRACE_SECONDS = 300
_LABEL_SCHEDULER_MARGIN_SECONDS = 300


@dataclass(frozen=True, slots=True)
class RemoteLayout:
    """Persistent paths for one remote operator run."""

    root: PurePosixPath

    @property
    def repo(self) -> PurePosixPath:
        return self.root / "repo"

    @property
    def hf_home(self) -> PurePosixPath:
        return self.root / "hf_home"

    @property
    def hf_token(self) -> PurePosixPath:
        return self.root / ".hf-token"

    @property
    def logs(self) -> PurePosixPath:
        return self.root / "logs"

    @property
    def split_work(self) -> PurePosixPath:
        return self.root / "split-work"

    @property
    def label_work(self) -> PurePosixPath:
        return self.root / "label-work"

    @property
    def label_output(self) -> PurePosixPath:
        return self.root / "label-output"

    @property
    def input_dir(self) -> PurePosixPath:
        return self.root / "input"

    @property
    def v2_input(self) -> PurePosixPath:
        return self.input_dir / "v2-sentences.parquet"


def split_submission(
    config: OperatorConfig,
    layout: RemoteLayout,
    *,
    walltime_seconds: int = DEFAULT_SPLIT_WALLTIME_SECONDS,
) -> SubmissionRequest:
    """Build one resumable splitter submission."""

    revision = config.input_dataset_revision
    if revision is None:
        raise ValueError("immutable input revision is required")
    shard = (config.region or "") if config.scope is Scope.REGION else ""
    if not 900 <= walltime_seconds <= 3_600:
        raise ValueError("split walltime must be between 15 and 60 minutes")
    command: tuple[str, ...] = (
        str(layout.repo / "scripts/grid5000/submit_streaming_build.sh"),
        str(layout.repo),
        str(layout.hf_home),
        str(layout.logs),
        config.output_dataset_id,
        config.input_dataset_id,
        config.execution_commit or config.source_commit,
        revision,
        config.run_id,
        str(config.requirements.batch_size),
        "0",
        shard,
        config.source_commit,
    )
    if walltime_seconds != DEFAULT_SPLIT_WALLTIME_SECONDS:
        command += (str(walltime_seconds),)
    return SubmissionRequest(command)


def split_finalization_submission(
    config: OperatorConfig, layout: RemoteLayout
) -> SubmissionRequest:
    """Build finalization submission after all requested shard checkpoints."""

    revision = config.input_dataset_revision
    if revision is None:
        raise ValueError("immutable input revision is required")
    expected_shard = config.region or "all"
    worldwide_v2 = config.scope is Scope.ALL and config.stage is Stage.ALL
    sampling_target = config.requirements.sampling_target if worldwide_v2 else None
    command = (
        str(layout.repo / "scripts/grid5000/submit_streaming_finalization.sh"),
        str(layout.repo),
        str(layout.hf_home),
        str(layout.logs),
        config.output_dataset_id,
        config.input_dataset_id,
        config.source_commit,
        revision,
        config.run_id,
        f"checkpoints/{config.run_id}",
        expected_shard,
        str(sampling_target or 0),
        config.requirements.sampling_seed,
        "02:00:00",
        "cpu",
    )
    return SubmissionRequest(command)


def label_submission(
    config: OperatorConfig,
    layout: RemoteLayout,
    *,
    input_parquet: PurePosixPath,
    model_file: PurePosixPath,
    tokenizer_dir: PurePosixPath,
    walltime_seconds: int = DEFAULT_LABEL_WALLTIME_SECONDS,
    policy_type: str | None = None,
    gpu_memory_mb: int = 40_000,
    label_plan: LabelLanePlan | None = None,
) -> SubmissionRequest:
    """Build one resumable labeling allocation."""

    revision = config.input_dataset_revision
    if revision is None:
        raise ValueError("immutable input revision is required")
    v2 = config.prompt_version == V2_LOGIT_PROMPT_VERSION
    if v2:
        plan = label_plan or label_lane_plan(config, layout.root, {})
        if plan.parent_run_id != config.run_id:
            raise ValueError("label lane does not belong to this operator run")
        effective_config = plan.config
        work_dir = plan.work_dir
        output_dir = plan.output_dir
    else:
        if label_plan is not None:
            raise ValueError("label lanes are only available for worldwide V2 runs")
        plan = None
        effective_config = config
        work_dir = layout.label_work
        output_dir = layout.label_output
    requirements = effective_config.requirements
    wrapper_name = (
        "run_worldwide_labeling_job.sh" if v2 else "run_afghanistan_labeling_job.sh"
    )
    submit_name = (
        "submit_worldwide_labeling.sh" if v2 else "submit_afghanistan_labeling.sh"
    )
    wrapper_args = (
        str(layout.repo / "scripts/grid5000" / wrapper_name),
        str(layout.repo),
        str(layout.hf_home),
        str(layout.logs),
        str(input_parquet),
        str(work_dir),
        str(output_dir),
        str(model_file),
        str(tokenizer_dir),
        effective_config.label_model_revision,
        revision,
        effective_config.source_commit,
        effective_config.output_dataset_id,
        str(requirements.batch_size),
        str(requirements.row_limit),
        str(requirements.llama_parallel),
        str(requirements.llama_per_slot_context),
        str(requirements.request_concurrency),
        str(requirements.sampling_target or 0),
        requirements.sampling_seed,
        str(requirements.sampling_h3_resolution),
        *((plan.lane.value,) if plan is not None else ()),
        effective_config.execution_commit or effective_config.source_commit,
    )
    if walltime_seconds == DEFAULT_LABEL_WALLTIME_SECONDS and policy_type is None:
        return SubmissionRequest(
            (
                str(layout.repo / "scripts/grid5000" / submit_name),
                *wrapper_args[1:],
            )
        )
    if not 900 <= walltime_seconds <= 3_600:
        raise ValueError("label walltime must be between 15 and 60 minutes")
    if policy_type not in {"day", "night"}:
        raise ValueError("label policy type must be day or night")
    if gpu_memory_mb <= 0:
        raise ValueError("GPU memory must be positive")
    duration_seconds = (
        walltime_seconds - _LABEL_GRACE_SECONDS - _LABEL_SCHEDULER_MARGIN_SECONDS
    )
    payload = "exec " + shlex.join(
        (
            "env",
            f"LABEL_DEADLINE_DURATION={duration_seconds}s",
            f"LABEL_DEADLINE_GRACE={_LABEL_GRACE_SECONDS}s",
            *wrapper_args,
        )
    )
    command = (
        str(layout.repo / "scripts/grid5000/_submit_gpu_job.sh"),
        str(gpu_memory_mb),
        format_walltime(walltime_seconds),
        policy_type,
        payload,
    )
    return SubmissionRequest(command)


def llama_build_submission(layout: RemoteLayout) -> SubmissionRequest:
    """Build pinned CUDA llama-server inside an allocated GPU job."""

    revision = "555881ebc8b0fc0402b30e09258a32a7bfd13c52"
    submit_helper = layout.repo / "scripts/grid5000/_submit_gpu_job.sh"
    target = layout.root / "llama-server-bin"
    source = layout.root / "llama.cpp"
    logs = layout.logs
    payload = (
        "set -euo pipefail; umask 077; "
        f'job_log={logs}/"${{OAR_JOB_ID:?}}"; '
        'mkdir -p -m 0700 "$job_log"; '
        'exec > >(tee -a "$job_log/build.stdout.log") '
        '2> >(tee -a "$job_log/build.stderr.log" >&2); '
        f"if [ ! -d {source}/.git ]; then "
        f"git clone https://github.com/ggml-org/llama.cpp.git {source}; fi; "
        f"git -C {source} fetch --no-tags origin; "
        f"git -C {source} cat-file -e {revision}^{{commit}}; "
        f"git -C {source} checkout --detach {revision}; "
        f"cmake -S {source} -B {source}/build -DGGML_CUDA=ON "
        "-DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release; "
        f"cmake --build {source}/build --target llama-server -j 12; "
        f"mkdir -p -m 0700 {target}; "
        f"install -m 0700 {source}/build/bin/llama-server {target}/llama-server"
    )
    return SubmissionRequest(
        (
            str(submit_helper),
            "40000",
            "01:00:00",
            "night",
            payload,
        )
    )


__all__ = [
    "RemoteLayout",
    "DEFAULT_LABEL_WALLTIME_SECONDS",
    "DEFAULT_SPLIT_WALLTIME_SECONDS",
    "MICRO_LABEL_WALLTIME_SECONDS",
    "label_submission",
    "llama_build_submission",
    "split_finalization_submission",
    "split_submission",
]
