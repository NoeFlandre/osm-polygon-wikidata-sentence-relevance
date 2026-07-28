"""Typed command descriptions for retained Grid'5000 compute payloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from osm_polygon_sentence_relevance.operator.config import OperatorConfig, Scope
from osm_polygon_sentence_relevance.operator.oar import SubmissionRequest


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


def split_submission(config: OperatorConfig, layout: RemoteLayout) -> SubmissionRequest:
    """Build one resumable splitter submission."""

    revision = config.input_dataset_revision
    if revision is None:
        raise ValueError("immutable input revision is required")
    shard = (config.region or "") if config.scope is Scope.REGION else ""
    command = (
        str(layout.repo / "scripts/grid5000/submit_streaming_build.sh"),
        str(layout.repo),
        str(layout.hf_home),
        str(layout.logs),
        config.output_dataset_id,
        config.input_dataset_id,
        config.source_commit,
        revision,
        config.run_id,
        str(config.requirements.batch_size),
        "0",
        shard,
    )
    return SubmissionRequest(command)


def split_finalization_submission(
    config: OperatorConfig, layout: RemoteLayout
) -> SubmissionRequest:
    """Build finalization submission after all requested shard checkpoints."""

    revision = config.input_dataset_revision
    if revision is None:
        raise ValueError("immutable input revision is required")
    expected_shard = config.region or "all"
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
) -> SubmissionRequest:
    """Build one resumable labeling allocation."""

    revision = config.input_dataset_revision
    if revision is None:
        raise ValueError("immutable input revision is required")
    requirements = config.requirements
    command = (
        str(layout.repo / "scripts/grid5000/submit_afghanistan_labeling.sh"),
        str(layout.repo),
        str(layout.hf_home),
        str(layout.logs),
        str(input_parquet),
        str(layout.label_work),
        str(layout.label_output),
        str(model_file),
        str(tokenizer_dir),
        config.label_model_revision,
        revision,
        config.source_commit,
        config.output_dataset_id,
        str(requirements.batch_size),
        str(requirements.row_limit),
        str(requirements.llama_parallel),
        str(requirements.llama_per_slot_context),
        str(requirements.request_concurrency),
    )
    return SubmissionRequest(command)


def llama_build_submission(layout: RemoteLayout) -> SubmissionRequest:
    """Build pinned CUDA llama-server inside an allocated GPU job."""

    target = layout.root / "llama-server-bin"
    source = layout.root / "llama.cpp"
    payload = (
        "set -euo pipefail; umask 077; "
        f"if [ ! -d {source}/.git ]; then "
        f"git clone https://github.com/ggml-org/llama.cpp.git {source}; fi; "
        f"git -C {source} fetch --no-tags origin 555881e; "
        f"git -C {source} checkout --detach 555881e; "
        f"cmake -S {source} -B {source}/build -DGGML_CUDA=ON "
        "-DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release; "
        f"cmake --build {source}/build --target llama-server -j 12; "
        f"mkdir -p -m 0700 {target}; "
        f"install -m 0700 {source}/build/bin/llama-server {target}/llama-server"
    )
    return SubmissionRequest(
        (
            "oarsub",
            "-q",
            "default",
            "-t",
            "exotic",
            "-t",
            "night",
            "-p",
            "gpu_mem>=40000",
            "-l",
            "gpu=1,walltime=01:00:00",
            payload,
        )
    )


__all__ = [
    "RemoteLayout",
    "label_submission",
    "llama_build_submission",
    "split_finalization_submission",
    "split_submission",
]
