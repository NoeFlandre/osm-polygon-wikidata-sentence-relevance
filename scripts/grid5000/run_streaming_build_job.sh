#!/usr/bin/env bash
# Scheduler-owned wrapper for one resumable streaming CUDA allocation.

set -euo pipefail
umask 077

: "${OAR_JOB_ID:?OAR_JOB_ID is required}"
if ! [[ "${OAR_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "run_streaming_build_job: OAR_JOB_ID must be numeric" >&2
    exit 2
fi
if [ "$#" -ne 10 ]; then
    echo "run_streaming_build_job: exactly ten positional arguments are required" >&2
    exit 2
fi

REPO_ROOT="$1"; readonly REPO_ROOT
HF_HOME="$2"; readonly HF_HOME
LOG_ROOT="$3"; readonly LOG_ROOT
OUTPUT_REPO_ID="$4"; readonly OUTPUT_REPO_ID
INPUT_REPO_ID="$5"; readonly INPUT_REPO_ID
EXPECTED_SOURCE_COMMIT="$6"; readonly EXPECTED_SOURCE_COMMIT
INPUT_REVISION="$7"; readonly INPUT_REVISION
RUN_ID="$8"; readonly RUN_ID
BATCH_SIZE="$9"; readonly BATCH_SIZE
MAX_SHARDS="${10}"; readonly MAX_SHARDS

if [ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${EXPECTED_SOURCE_COMMIT}" ]; then
    echo "run_streaming_build_job: checkout commit mismatch" >&2
    exit 1
fi
if [ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]; then
    echo "run_streaming_build_job: checkout is dirty" >&2
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
PAYLOAD="${REPO_ROOT}/scripts/grid5000/run_streaming_build.sh"
if [ ! -x "${PYTHON}" ] || [ ! -x "${PAYLOAD}" ]; then
    echo "run_streaming_build_job: required executable is missing" >&2
    exit 1
fi

SCRATCH_BASE="${LOCALSCRATCH:-${OAR_JOB_SCRATCH_DIR:-/tmp/oar-${OAR_JOB_ID}}}"
case "${SCRATCH_BASE}" in
    *"${OAR_JOB_ID}"*) ;;
    *) echo "run_streaming_build_job: scratch path is not allocation-bound" >&2; exit 1 ;;
esac
mkdir -p -m 0700 -- "${SCRATCH_BASE}"
WORK_DIR="${SCRATCH_BASE}/osm_streaming_${RUN_ID}"
mkdir -m 0700 -- "${WORK_DIR}"

JOB_LOG_DIR="${LOG_ROOT}/${OAR_JOB_ID}"
mkdir -m 0700 -- "${JOB_LOG_DIR}"

"${PYTHON}" "${REPO_ROOT}/scripts/grid5000/gpu_preflight.py" \
    >"${JOB_LOG_DIR}/gpu_preflight.json" \
    2>"${JOB_LOG_DIR}/gpu_preflight.stderr.log"

set +e
"${PAYLOAD}" "${REPO_ROOT}" "${HF_HOME}" "${WORK_DIR}" \
    "${OUTPUT_REPO_ID}" "${INPUT_REPO_ID}" "${EXPECTED_SOURCE_COMMIT}" \
    "${INPUT_REVISION}" "${RUN_ID}" "${BATCH_SIZE}" "${MAX_SHARDS}" \
    >"${JOB_LOG_DIR}/build.stdout.log" \
    2>"${JOB_LOG_DIR}/build.stderr.log"
build_rc=$?
set -e
printf '%s\n' "${build_rc}" >"${JOB_LOG_DIR}/build.exit_code"
chmod 0600 "${JOB_LOG_DIR}"/*
exit "${build_rc}"
