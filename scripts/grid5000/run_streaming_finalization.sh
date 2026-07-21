#!/usr/bin/env bash
# Compute-node payload for one bounded one-shard streaming finalization.

set -euo pipefail
umask 077

if [ "$#" -ne 10 ]; then
    echo "run_streaming_finalization: exactly ten positional arguments are required" >&2
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
STAGING_REVISION="$9"; readonly STAGING_REVISION
EXPECTED_SHARD="${10}"; readonly EXPECTED_SHARD

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "run_streaming_finalization: locked interpreter is missing" >&2
    exit 1
fi

export HF_HOME
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"

CACHE_DIR="${WORK_DIR}/cache"
SCRATCH_DIR="${WORK_DIR}/scratch"
OUTPUT_DIR="${WORK_DIR}/out"
mkdir -m 0700 -- "${CACHE_DIR}" "${SCRATCH_DIR}"

args=(
    -m scripts.streaming.finalization
    --repo-id "${OUTPUT_REPO_ID}"
    --upstream-repo-id "${INPUT_REPO_ID}"
    --run-id "${RUN_ID}"
    --staging-revision "${STAGING_REVISION}"
    --source-commit "${EXPECTED_SOURCE_COMMIT}"
    --input-revision "${INPUT_REVISION}"
    --cache-dir "${CACHE_DIR}"
    --scratch-dir "${SCRATCH_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --expected-shard "${EXPECTED_SHARD}"
)

exec "${PYTHON}" "${args[@]}"
