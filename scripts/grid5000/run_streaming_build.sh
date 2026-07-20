#!/usr/bin/env bash
# Compute-node payload for the resumable per-shard CUDA build.

set -euo pipefail
umask 077

if [ "$#" -ne 10 ]; then
    echo "run_streaming_build: exactly ten positional arguments are required" >&2
    exit 2
fi

REPO_ROOT="$1"; readonly REPO_ROOT
HF_HOME="$2"; readonly HF_HOME
WORK_DIR="$3"; readonly WORK_DIR
OUTPUT_REPO_ID="$4"; readonly OUTPUT_REPO_ID
INPUT_REPO_ID="$5"; readonly INPUT_REPO_ID
EXPECTED_SOURCE_COMMIT="$6"; readonly EXPECTED_SOURCE_COMMIT
INPUT_REVISION="$7"; readonly INPUT_REVISION
RUN_ID="$8"; readonly RUN_ID
BATCH_SIZE="$9"; readonly BATCH_SIZE
MAX_SHARDS="${10}"; readonly MAX_SHARDS

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "run_streaming_build: locked interpreter is missing" >&2
    exit 1
fi

export HF_HOME
# Input shards and checkpoint commits require Hub access.  Grid'5000
# sessions may inherit this variable from earlier offline smoke runs.
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

args=(
    -m scripts.streaming.driver stream-build
    --confirm-offload
    --run-id "${RUN_ID}"
    --staging-revision "checkpoints/${RUN_ID}"
    --repo-id "${OUTPUT_REPO_ID}"
    --upstream-repo-id "${INPUT_REPO_ID}"
    --resolved-revision "${INPUT_REVISION}"
    --source-commit "${EXPECTED_SOURCE_COMMIT}"
    --work-dir "${WORK_DIR}"
    --batch-size "${BATCH_SIZE}"
    --pipeline-version "0.1.0"
    --model-name "sat-3l-sm"
    --device cuda
    --max-disk-bytes 5368709120
)
if [ "${MAX_SHARDS}" -gt 0 ]; then
    args+=(--max-shards "${MAX_SHARDS}")
fi

exec "${PYTHON}" "${args[@]}"
