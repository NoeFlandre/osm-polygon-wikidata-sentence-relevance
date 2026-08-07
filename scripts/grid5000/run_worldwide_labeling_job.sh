#!/usr/bin/env bash
# Compute-node guard and resumable wrapper for the worldwide V2 label lane.

set -euo pipefail
umask 077
: "${OAR_JOB_ID:?OAR_JOB_ID is required}"
if ! [[ "${OAR_JOB_ID}" =~ ^[0-9]+$ ]]; then echo "run_worldwide_labeling_job: OAR_JOB_ID must be numeric" >&2; exit 2; fi
if [ "$#" -ne 20 ]; then echo "run_worldwide_labeling_job: exactly twenty arguments are required" >&2; exit 2; fi
REPO_ROOT="$(cd "$1" && pwd -P)"; readonly REPO_ROOT
LOG_ROOT="$3"; EXPECTED_SOURCE_COMMIT="${11}"; WORK_DIR="$5"
readonly LOG_ROOT EXPECTED_SOURCE_COMMIT WORK_DIR
RUN_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"; readonly RUN_ROOT
. "$(dirname "${BASH_SOURCE[0]}")/_checkout_guard.sh"
if [ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${EXPECTED_SOURCE_COMMIT}" ] || \
   ! validate_clean_checkout "${REPO_ROOT}" "${RUN_ROOT}"; then
    echo "run_worldwide_labeling_job: strict checkout guard failed" >&2
    exit 1
fi
command -v flock >/dev/null || { echo "run_worldwide_labeling_job: flock is required" >&2; exit 1; }
RUN_LOCK="${RUN_ROOT}/worldwide-labeling.run.lock"; readonly RUN_LOCK
[ ! -L "${RUN_LOCK}" ] || { echo "run_worldwide_labeling_job: lock must not be a symlink" >&2; exit 2; }
: >"${RUN_LOCK}"; chmod 0600 "${RUN_LOCK}"; exec 9<>"${RUN_LOCK}"
flock -n 9 || { echo "run_worldwide_labeling_job: another allocation owns this run" >&2; exit 75; }
PAYLOAD="${REPO_ROOT}/scripts/grid5000/run_worldwide_labeling.sh"; readonly PAYLOAD
[ -x "${PAYLOAD}" ] || { echo "run_worldwide_labeling_job: payload is missing" >&2; exit 1; }
PYTHON="${REPO_ROOT}/.venv/bin/python"; [ -x "${PYTHON}" ] || { echo "run_worldwide_labeling_job: Python is missing" >&2; exit 1; }
JOB_LOG_DIR="${LOG_ROOT}/${OAR_JOB_ID}"; mkdir -m 0700 -- "${JOB_LOG_DIR}"
set +e
. "$(dirname "${BASH_SOURCE[0]}")/_deadline_helper.sh"
DEADLINE_DURATION="${LABEL_DEADLINE_DURATION:-45m}"; DEADLINE_GRACE="${LABEL_DEADLINE_GRACE:-5m}"
deadline_helper_run "${DEADLINE_DURATION}" "${DEADLINE_GRACE}" "${PAYLOAD}" \
    "${REPO_ROOT}" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" "${17}" "${18}" "${19}" "${20}" \
    >"${JOB_LOG_DIR}/labeling.stdout.log" 2>"${JOB_LOG_DIR}/labeling.stderr.log"
rc=$?
set -e
printf '%s\n' "${rc}" >"${JOB_LOG_DIR}/labeling.exit_code"
chmod 0600 "${JOB_LOG_DIR}"/*
exit "${rc}"
