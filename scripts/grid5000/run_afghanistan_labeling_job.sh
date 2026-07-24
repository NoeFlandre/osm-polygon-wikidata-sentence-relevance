#!/usr/bin/env bash
# Scheduler-owned wrapper for one resumable Afghanistan labeling allocation.

set -euo pipefail
umask 077

: "${OAR_JOB_ID:?OAR_JOB_ID is required}"
if ! [[ "${OAR_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "run_afghanistan_labeling_job: OAR_JOB_ID must be numeric" >&2
    exit 2
fi
if [ "$#" -ne 14 ]; then
    echo "run_afghanistan_labeling_job: exactly fourteen arguments are required" >&2
    exit 2
fi

REPO_ROOT="$1"; readonly REPO_ROOT
LOG_ROOT="$3"; readonly LOG_ROOT
EXPECTED_SOURCE_COMMIT="${11}"; readonly EXPECTED_SOURCE_COMMIT

if [ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${EXPECTED_SOURCE_COMMIT}" ]; then
    echo "run_afghanistan_labeling_job: checkout commit mismatch" >&2
    exit 1
fi
if [ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]; then
    echo "run_afghanistan_labeling_job: checkout is dirty" >&2
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
PAYLOAD="${REPO_ROOT}/scripts/grid5000/run_afghanistan_labeling.sh"
if [ ! -x "${PYTHON}" ] || [ ! -x "${PAYLOAD}" ]; then
    echo "run_afghanistan_labeling_job: required executable is missing" >&2
    exit 1
fi

JOB_LOG_DIR="${LOG_ROOT}/${OAR_JOB_ID}"
mkdir -m 0700 -- "${JOB_LOG_DIR}"

"${PYTHON}" "${REPO_ROOT}/scripts/grid5000/gpu_preflight.py" \
    >"${JOB_LOG_DIR}/gpu_preflight.json" \
    2>"${JOB_LOG_DIR}/gpu_preflight.stderr.log"

set +e
"${PAYLOAD}" "${REPO_ROOT}" "$4" "$5" "$6" "$7" "$8" "$9" \
    "${10}" "${11}" "${12}" "${13}" "${14}" \
    >"${JOB_LOG_DIR}/labeling.stdout.log" \
    2>"${JOB_LOG_DIR}/labeling.stderr.log"
labeling_rc=$?
set -e
printf '%s\n' "${labeling_rc}" >"${JOB_LOG_DIR}/labeling.exit_code"
chmod 0600 "${JOB_LOG_DIR}"/*
exit "${labeling_rc}"
