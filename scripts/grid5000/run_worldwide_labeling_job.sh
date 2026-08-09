#!/usr/bin/env bash
# Compute-node guard and resumable wrapper for the worldwide V2 label lane.

set -euo pipefail
umask 077
: "${OAR_JOB_ID:?OAR_JOB_ID is required}"
if ! [[ "${OAR_JOB_ID}" =~ ^[0-9]+$ ]]; then echo "run_worldwide_labeling_job: OAR_JOB_ID must be numeric" >&2; exit 2; fi
if [ "$#" -ne 22 ]; then echo "run_worldwide_labeling_job: exactly twenty-two arguments are required" >&2; exit 2; fi
REPO_ROOT="$(cd "$1" && pwd -P)"; readonly REPO_ROOT
LOG_ROOT="$3"; DATA_SOURCE_COMMIT="${11}"; WORK_DIR="$5"; LABEL_LANE="${21}"; EXECUTION_COMMIT="${22}"
readonly LOG_ROOT DATA_SOURCE_COMMIT WORK_DIR LABEL_LANE EXECUTION_COMMIT
case "${LABEL_LANE}" in smoke|production) ;; *) echo "run_worldwide_labeling_job: label lane is invalid" >&2; exit 2;; esac
RUN_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"; readonly RUN_ROOT
. "$(dirname "${BASH_SOURCE[0]}")/_checkout_guard.sh"
mark_failed_on_exit() {
    local rc=$?
    mark_managed_run_failed "${RUN_ROOT}" "${OAR_JOB_ID}" "${rc}"
    exit "${rc}"
}
trap mark_failed_on_exit EXIT
HF_TOKEN_FILE="${RUN_ROOT}/.hf-token"; readonly HF_TOKEN_FILE
if [ ! -f "${HF_TOKEN_FILE}" ] || [ -L "${HF_TOKEN_FILE}" ] || \
   [ "$(stat -c %a -- "${HF_TOKEN_FILE}")" != "600" ]; then
    echo "run_worldwide_labeling_job: Hugging Face credential file is missing or unsafe" >&2
    exit 1
fi
export HF_TOKEN="$(cat -- "${HF_TOKEN_FILE}")"
[ -n "${HF_TOKEN}" ] || { echo "run_worldwide_labeling_job: Hugging Face credential is empty" >&2; exit 1; }
if [ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${EXECUTION_COMMIT}" ] || \
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
JOB_LOG_DIR="${LOG_ROOT}/${OAR_JOB_ID}"; mkdir -m 0700 -- "${JOB_LOG_DIR}"
SCRATCH_BASE="${LOCALSCRATCH:-${OAR_JOB_SCRATCH_DIR:-/tmp/oar-${OAR_JOB_ID}}}"
case "${SCRATCH_BASE}" in
    *"${OAR_JOB_ID}"*) ;;
    *) echo "run_worldwide_labeling_job: scratch path is not allocation-bound" >&2; exit 1 ;;
esac
mkdir -p -m 0700 -- "${SCRATCH_BASE}"
prepare_compute_environment "${REPO_ROOT}" "${SCRATCH_BASE}" "${JOB_LOG_DIR}" \
    "run_worldwide_labeling_job"
PYTHON="${REPO_ROOT}/.venv/bin/python"
set +e
. "$(dirname "${BASH_SOURCE[0]}")/_deadline_helper.sh"
DEADLINE_DURATION="${LABEL_DEADLINE_DURATION:-45m}"; DEADLINE_GRACE="${LABEL_DEADLINE_GRACE:-5m}"
deadline_helper_run "${DEADLINE_DURATION}" "${DEADLINE_GRACE}" "${PAYLOAD}" \
    "${REPO_ROOT}" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" "${17}" "${18}" "${19}" "${20}" "${21}" \
    >"${JOB_LOG_DIR}/labeling.stdout.log" 2>"${JOB_LOG_DIR}/labeling.stderr.log"
rc=$?
set -e
printf '%s\n' "${rc}" >"${JOB_LOG_DIR}/labeling.exit_code"
chmod 0600 "${JOB_LOG_DIR}"/*
exit "${rc}"
