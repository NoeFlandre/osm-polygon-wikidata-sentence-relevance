#!/usr/bin/env bash
# Scheduler-owned wrapper for one short, resumable streaming CUDA allocation.

set -euo pipefail
umask 077

: "${OAR_JOB_ID:?OAR_JOB_ID is required}"
if ! [[ "${OAR_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "run_streaming_build_job: OAR_JOB_ID must be numeric" >&2
    exit 2
fi
if [ "$#" -ne 12 ]; then
    echo "run_streaming_build_job: exactly twelve positional arguments are required" >&2
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
SHARD_KEY="${11}"; readonly SHARD_KEY
DATA_SOURCE_COMMIT="${12}"; readonly DATA_SOURCE_COMMIT

RUN_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"; readonly RUN_ROOT
case "${RUN_ROOT}" in
    "${HOME}/osm-polygon-operator/"*) ;;
    *)
        echo "run_streaming_build_job: managed run root is outside the operator directory" >&2
        exit 1
        ;;
esac
MARKER="${RUN_ROOT}/.operator-managed.json"; readonly MARKER
mark_failed_on_exit() {
    rc=$?
    if [ "${rc}" -ne 0 ] && [ -f "${MARKER}" ] && [ ! -L "${MARKER}" ]; then
        marker_tmp="${MARKER}.tmp.${OAR_JOB_ID}"
        if printf '%s\n' '{"schema_version":1,"status":"failed"}' >"${marker_tmp}"; then
            chmod 0600 "${marker_tmp}" 2>/dev/null || true
            mv -f -- "${marker_tmp}" "${MARKER}" 2>/dev/null || true
        fi
    fi
    exit "${rc}"
}
trap mark_failed_on_exit EXIT
HF_TOKEN_FILE="${RUN_ROOT}/.hf-token"; readonly HF_TOKEN_FILE
if [ ! -f "${HF_TOKEN_FILE}" ] || [ -L "${HF_TOKEN_FILE}" ]; then
    echo "run_streaming_build_job: Hugging Face credential file is missing or unsafe" >&2
    exit 1
fi
if [ "$(stat -c %a -- "${HF_TOKEN_FILE}")" != "600" ]; then
    echo "run_streaming_build_job: Hugging Face credential file must be mode 0600" >&2
    exit 1
fi
export HF_TOKEN="$(cat -- "${HF_TOKEN_FILE}")"
[ -n "${HF_TOKEN}" ] || { echo "run_streaming_build_job: Hugging Face credential is empty" >&2; exit 1; }

if [ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${EXPECTED_SOURCE_COMMIT}" ]; then
    echo "run_streaming_build_job: checkout commit mismatch" >&2
    exit 1
fi
if [ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]; then
    echo "run_streaming_build_job: checkout is dirty" >&2
    exit 1
fi

PAYLOAD="${REPO_ROOT}/scripts/grid5000/run_streaming_build.sh"
DEADLINE_HELPER="${REPO_ROOT}/scripts/grid5000/_deadline_helper.sh"
if [ ! -x "${PAYLOAD}" ] || [ ! -r "${DEADLINE_HELPER}" ]; then
    echo "run_streaming_build_job: required payload or deadline helper is missing" >&2
    exit 1
fi
# shellcheck source=_deadline_helper.sh
source "${DEADLINE_HELPER}"
# shellcheck source=_checkout_guard.sh
source "$(dirname "${BASH_SOURCE[0]}")/_checkout_guard.sh"

SCRATCH_BASE="${LOCALSCRATCH:-${OAR_JOB_SCRATCH_DIR:-/tmp/oar-${OAR_JOB_ID}}}"
case "${SCRATCH_BASE}" in
    *"${OAR_JOB_ID}"*) ;;
    *) echo "run_streaming_build_job: scratch path is not allocation-bound" >&2; exit 1 ;;
esac
mkdir -p -m 0700 -- "${SCRATCH_BASE}"
# Partial shard checkpoints must survive this allocation ending. Keep the
# work directory in the managed persistent run root; allocation scratch is
# reserved for disposable caches and scheduler-local temporary files.
WORK_DIR="${RUN_ROOT}/work"
if [ -L "${WORK_DIR}" ] || { [ -e "${WORK_DIR}" ] && [ ! -d "${WORK_DIR}" ]; }; then
    echo "run_streaming_build_job: persistent work directory is unsafe" >&2
    exit 1
fi
mkdir -p -m 0700 -- "${WORK_DIR}"
chmod 0700 -- "${WORK_DIR}"

JOB_LOG_DIR="${LOG_ROOT}/${OAR_JOB_ID}"
mkdir -m 0700 -- "${JOB_LOG_DIR}"

prepare_compute_environment "${REPO_ROOT}" "${SCRATCH_BASE}" "${JOB_LOG_DIR}" \
    "run_streaming_build_job"
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "run_streaming_build_job: compute-node Python is missing" >&2
    exit 1
fi

"${PYTHON}" "${REPO_ROOT}/scripts/grid5000/gpu_preflight.py" \
    >"${JOB_LOG_DIR}/gpu_preflight.json" \
    2>"${JOB_LOG_DIR}/gpu_preflight.stderr.log"

set +e
# Spend most of the 30-minute allocation on CUDA segmentation.  Four minutes
# remain for a graceful SIGINT checkpoint, with one minute of scheduler margin.
# Completed section batches are already durable, so a forced stop resumes from
# the last validated batch.
deadline_helper_run 25m 4m "${PAYLOAD}" \
    "${REPO_ROOT}" "${HF_HOME}" "${WORK_DIR}" \
    "${OUTPUT_REPO_ID}" "${INPUT_REPO_ID}" "${EXPECTED_SOURCE_COMMIT}" \
    "${INPUT_REVISION}" "${RUN_ID}" "${BATCH_SIZE}" "${MAX_SHARDS}" \
    "${SHARD_KEY}" "${DATA_SOURCE_COMMIT}" \
    >"${JOB_LOG_DIR}/build.stdout.log" \
    2>"${JOB_LOG_DIR}/build.stderr.log"
build_rc=$?
set -e
printf '%s\n' "${build_rc}" >"${JOB_LOG_DIR}/build.exit_code"
chmod 0600 "${JOB_LOG_DIR}"/*
exit "${build_rc}"
