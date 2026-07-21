#!/usr/bin/env bash
# Scheduler-owned wrapper for one bounded one-shard OAR finalization.

set -euo pipefail
umask 077

: "${OAR_JOB_ID:?OAR_JOB_ID is required}"
if ! [[ "${OAR_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "run_streaming_finalization_job: OAR_JOB_ID must be numeric" >&2
    exit 2
fi
if [ "$#" -ne 12 ]; then
    echo "run_streaming_finalization_job: exactly twelve positional arguments are required" >&2
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
STAGING_REVISION="$9"; readonly STAGING_REVISION
EXPECTED_SHARD="${10}"; readonly EXPECTED_SHARD
WALLTIME="${11}"; readonly WALLTIME
NODE_TYPE="${12}"; readonly NODE_TYPE

# HEAD may legitimately differ from EXPECTED_SOURCE_COMMIT: the latter
# is the data identity (the source commit that produced the staged
# checkpoint) and the wrapper does not enforce that the checkout was
# built at that exact commit.  What the wrapper DOES enforce is that
# the production finalization payload exists as a tracked file at
# HEAD and that the working tree is clean, so the script the OAR
# scheduler is actually running is provably the script present in
# the checkout.
if ! git -C "${REPO_ROOT}" cat-file -e "HEAD:scripts/streaming/finalization.py" 2>/dev/null; then
    echo "run_streaming_finalization_job: HEAD does not contain scripts/streaming/finalization.py; refusing" >&2
    exit 1
fi
if [ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]; then
    echo "run_streaming_finalization_job: checkout is dirty" >&2
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
PAYLOAD="${REPO_ROOT}/scripts/grid5000/run_streaming_finalization.sh"
if [ ! -x "${PYTHON}" ] || [ ! -x "${PAYLOAD}" ]; then
    echo "run_streaming_finalization_job: required executable is missing" >&2
    exit 1
fi

SCRATCH_BASE="${LOCALSCRATCH:-${OAR_JOB_SCRATCH_DIR:-/tmp/oar-${OAR_JOB_ID}}}"
case "${SCRATCH_BASE}" in
    *"${OAR_JOB_ID}"*) ;;
    *) echo "run_streaming_finalization_job: scratch path is not allocation-bound" >&2; exit 1 ;;
esac
mkdir -p -m 0700 -- "${SCRATCH_BASE}"
WORK_DIR="${SCRATCH_BASE}/osm_finalize_${RUN_ID}"
mkdir -m 0700 -- "${WORK_DIR}"

JOB_LOG_DIR="${LOG_ROOT}/${OAR_JOB_ID}"
mkdir -m 0700 -- "${JOB_LOG_DIR}"

set +e
"${PAYLOAD}" "${REPO_ROOT}" "${HF_HOME}" "${WORK_DIR}" \
    "${OUTPUT_REPO_ID}" "${INPUT_REPO_ID}" "${EXPECTED_SOURCE_COMMIT}" \
    "${INPUT_REVISION}" "${RUN_ID}" "${STAGING_REVISION}" "${EXPECTED_SHARD}" \
    >"${JOB_LOG_DIR}/finalize.stdout.log" \
    2>"${JOB_LOG_DIR}/finalize.stderr.log"
finalize_rc=$?
set -e
printf '%s\n' "${finalize_rc}" >"${JOB_LOG_DIR}/finalize.exit_code"
chmod 0600 "${JOB_LOG_DIR}"/*
exit "${finalize_rc}"
