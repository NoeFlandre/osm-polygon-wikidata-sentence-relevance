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
if [ "$#" -ne 15 ]; then
    echo "run_afghanistan_labeling_job: exactly fifteen arguments are required" >&2
    exit 2
fi

REPO_ROOT="$(cd "$1" && pwd -P)"; readonly REPO_ROOT
LOG_ROOT="$3"; readonly LOG_ROOT
EXPECTED_SOURCE_COMMIT="${11}"; readonly EXPECTED_SOURCE_COMMIT
LLAMA_PARALLEL="${15}"; readonly LLAMA_PARALLEL
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

case "${LLAMA_PARALLEL}" in
    1|2|4|8|16|32) ;;
    *) echo "run_afghanistan_labeling_job: LLAMA_PARALLEL must be one of 1, 2, 4, 8, 16, 32" >&2; exit 2;;
esac

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
# Wrap the payload in the deadline helper so a 12-hour OAR allocation
# gives the labelling CLI 700m (11h40m) and a 10-minute grace before SIGKILL.
# The labelling CLI handles SIGINT by writing ``interrupted=true`` and
# exiting 0 so the helper propagates 0 and the next allocation can
# resume from the same checkpoint directory.
#
# shellcheck source=scripts/grid5000/_deadline_helper.sh
. "$(dirname "${BASH_SOURCE[0]}")/_deadline_helper.sh"
deadline_helper_run 700m 10m "${PAYLOAD}" \
    "${REPO_ROOT}" "$4" "$5" "$6" "$7" "$8" "$9" \
    "${10}" "${11}" "${12}" "${13}" "${14}" "${LLAMA_PARALLEL}" \
    >"${JOB_LOG_DIR}/labeling.stdout.log" \
    2>"${JOB_LOG_DIR}/labeling.stderr.log"
labeling_rc=$?
set -e
printf '%s\n' "${labeling_rc}" >"${JOB_LOG_DIR}/labeling.exit_code"
chmod 0600 "${JOB_LOG_DIR}"/*
exit "${labeling_rc}"
