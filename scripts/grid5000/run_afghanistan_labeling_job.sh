#!/usr/bin/env bash
# Scheduler-owned wrapper for one resumable Afghanistan labeling allocation.
#
# The wrapper translates the front-end arguments into the bounded payload
# contract. The parallelism argument is propagated as the final positional
# argument so the payload can derive the total context and launch the real
# llama-server binary directly.

set -euo pipefail
umask 077

: "${OAR_JOB_ID:?OAR_JOB_ID is required}"
if ! [[ "${OAR_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "run_afghanistan_labeling_job: OAR_JOB_ID must be numeric" >&2
    exit 2
fi
if [ "$#" -ne 17 ]; then
    echo "run_afghanistan_labeling_job: exactly seventeen arguments are required" >&2
    exit 2
fi

REPO_ROOT="$(cd "$1" && pwd -P)"; readonly REPO_ROOT
LOG_ROOT="$3"; readonly LOG_ROOT
EXPECTED_SOURCE_COMMIT="${11}"; readonly EXPECTED_SOURCE_COMMIT
LLAMA_PARALLEL="${15}"; readonly LLAMA_PARALLEL
LLAMA_PER_SLOT_CONTEXT="${16}"; readonly LLAMA_PER_SLOT_CONTEXT
REQUEST_CONCURRENCY="${17}"; readonly REQUEST_CONCURRENCY
RUN_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"; readonly RUN_ROOT

# ``$2`` is the HF cache and ``$5`` is the persistent work directory; the
# approved run root is the repository parent directory on persistent storage.
# Sourcing the guard here means the validation logic can also be tested in
# isolation.
# shellcheck source=scripts/grid5000/_checkout_guard.sh
. "$(dirname "${BASH_SOURCE[0]}")/_checkout_guard.sh"
WORK_DIR="$5"

if [ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" != "${EXPECTED_SOURCE_COMMIT}" ]; then
    echo "run_afghanistan_labeling_job: checkout commit mismatch" >&2
    exit 1
fi
if ! validate_clean_checkout "${REPO_ROOT}" "${RUN_ROOT}"; then
    echo "run_afghanistan_labeling_job: checkout failed the strict clean-checkout guard" >&2
    exit 1
fi

# Final protection against two schedulers running the same immutable labeling
# identity at once. The Mac-side operator normally prevents this; the
# allocation-local nonblocking lock also covers scheduler/start-time races.
command -v flock >/dev/null || {
    echo "run_afghanistan_labeling_job: flock is required" >&2
    exit 1
}
RUN_LOCK="${RUN_ROOT}/labeling.run.lock"; readonly RUN_LOCK
if [ -L "${RUN_LOCK}" ]; then
    echo "run_afghanistan_labeling_job: run lock must not be a symlink" >&2
    exit 1
fi
: >"${RUN_LOCK}"
chmod 0600 "${RUN_LOCK}"
exec 9<>"${RUN_LOCK}"
if ! flock -n 9; then
    echo "run_afghanistan_labeling_job: another allocation owns this run" >&2
    exit 75
fi

case "${LLAMA_PARALLEL}" in
    1|2|4|8|16|32) ;;
    *) echo "run_afghanistan_labeling_job: LLAMA_PARALLEL must be one of 1, 2, 4, 8, 16, 32" >&2; exit 2;;
esac
if ! [[ "${LLAMA_PER_SLOT_CONTEXT}" =~ ^[1-9][0-9]*$ ]] || \
   [ "${LLAMA_PER_SLOT_CONTEXT}" -lt 4096 ] || \
   ! [[ "${REQUEST_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || \
   [ "${REQUEST_CONCURRENCY}" -gt "${LLAMA_PARALLEL}" ]; then
    echo "run_afghanistan_labeling_job: invalid context or concurrency" >&2
    exit 2
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
PAYLOAD="${REPO_ROOT}/scripts/grid5000/run_afghanistan_labeling.sh"
LLAMA_SERVER_DIR="${RUN_ROOT}/llama-server-bin"
if [ ! -x "${PYTHON}" ] || [ ! -x "${PAYLOAD}" ] || \
   [ ! -x "${LLAMA_SERVER_DIR}/llama-server" ]; then
    echo "run_afghanistan_labeling_job: required executable is missing" >&2
    exit 1
fi
export PATH="${LLAMA_SERVER_DIR}:${PATH}"

JOB_LOG_DIR="${LOG_ROOT}/${OAR_JOB_ID}"
mkdir -m 0700 -- "${JOB_LOG_DIR}"

set +e
# Wrap the payload so a 55-minute OAR allocation gives the labeling CLI
# 45 minutes, a 5-minute graceful checkpoint window, and scheduler margin.
# The labelling CLI handles SIGINT by writing ``interrupted=true`` and
# exiting 0 so the helper propagates 0 and the next allocation can
# resume from the same checkpoint directory.
#
# shellcheck source=scripts/grid5000/_deadline_helper.sh
. "$(dirname "${BASH_SOURCE[0]}")/_deadline_helper.sh"
deadline_helper_run 45m 5m "${PAYLOAD}" \
    "${REPO_ROOT}" "$4" "$5" "$6" "$7" "$8" "$9" \
    "${10}" "${11}" "${12}" "${13}" "${14}" "${LLAMA_PARALLEL}" \
    "${LLAMA_PER_SLOT_CONTEXT}" "${REQUEST_CONCURRENCY}" \
    >"${JOB_LOG_DIR}/labeling.stdout.log" \
    2>"${JOB_LOG_DIR}/labeling.stderr.log"
labeling_rc=$?
set -e
printf '%s\n' "${labeling_rc}" >"${JOB_LOG_DIR}/labeling.exit_code"
chmod 0600 "${JOB_LOG_DIR}"/*
exit "${labeling_rc}"
