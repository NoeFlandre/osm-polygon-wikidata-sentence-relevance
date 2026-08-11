#!/usr/bin/env bash
# Frontend-only adapter. Submits exactly one non-interactive OAR job that
# runs the bounded one-shard finalization against a previously streamed
# staging branch.  GPU is NOT required: finalization is CPU-only.

set -euo pipefail
umask 077

if [ "$#" -ne 15 ]; then
    echo "submit_streaming_finalization: exactly fifteen positional arguments are required" >&2
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
SAMPLING_TARGET="${11}"; readonly SAMPLING_TARGET
SAMPLING_SEED="${12}"; readonly SAMPLING_SEED
WALLTIME="${13}"; readonly WALLTIME
NODE_TYPE="${14}"; readonly NODE_TYPE
PERSIST_DIR="${15}"; readonly PERSIST_DIR

for path in "${REPO_ROOT}" "${HF_HOME}" "${LOG_ROOT}"; do
    case "${path}" in /*) ;; *) echo "submit_streaming_finalization: persistent path must be absolute" >&2; exit 2 ;; esac
    if [ ! -d "${path}" ] || [ -L "${path}" ]; then
        echo "submit_streaming_finalization: persistent path must be a real directory" >&2
        exit 2
    fi
done
if ! [[ "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
   ! [[ "${INPUT_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "submit_streaming_finalization: revisions must be 40 lowercase hex characters" >&2
    exit 2
fi
if ! [[ "${OUTPUT_REPO_ID}" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]] || \
   ! [[ "${INPUT_REPO_ID}" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]]; then
    echo "submit_streaming_finalization: repository IDs must be owner/name" >&2
    exit 2
fi
if ! [[ "${RUN_ID}" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || \
   ! [[ "${EXPECTED_SHARD}" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || \
   ! [[ "${SAMPLING_TARGET}" =~ ^[0-9]+$ ]] || \
   [ -z "${SAMPLING_SEED}" ] || \
   ! [[ "${WALLTIME}" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || \
   ! [[ "${PERSIST_DIR}" = /* ]]; then
    echo "submit_streaming_finalization: invalid run-id/shard/sampling/walltime" >&2
    exit 2
fi
case "${NODE_TYPE}" in cpu|gpu) ;; *) echo "submit_streaming_finalization: node-type must be cpu|gpu" >&2; exit 2 ;; esac

WRAPPER="${REPO_ROOT}/scripts/grid5000/run_streaming_finalization_job.sh"
if [ ! -x "${WRAPPER}" ] || ! command -v oarsub >/dev/null 2>&1; then
    echo "submit_streaming_finalization: wrapper or oarsub is unavailable" >&2
    exit 1
fi

shell_quote() {
    printf "'%s'" "${1//\'/\'\\\'\'}"
}

command_string="exec $(shell_quote "${WRAPPER}")"
for value in "$@"; do
    command_string="${command_string} $(shell_quote "${value}")"
done

# Keep finalization allocations policy-compliant without forcing every job into
# the night window. A weekday CPU job may use the day window only when its full
# requested walltime fits before 19:00; otherwise it remains night-bound.
IFS=: read -r wall_hours wall_minutes wall_seconds <<<"${WALLTIME}"
WALLTIME_SECONDS=$((10#${wall_hours} * 3600 + 10#${wall_minutes} * 60 + 10#${wall_seconds}))
readonly WALLTIME_SECONDS
read -r policy_weekday policy_hour policy_minute < <(
    TZ=Europe/Paris date '+%u %H %M'
)
if ! [[ "${policy_weekday}" =~ ^[1-7]$ &&
    "${policy_hour}" =~ ^[0-9]{2}$ &&
    "${policy_minute}" =~ ^[0-9]{2}$ ]]; then
    echo "submit_streaming_finalization: invalid Europe/Paris scheduler clock" >&2
    exit 1
fi
policy_type=night
now_seconds=$((10#${policy_hour} * 3600 + 10#${policy_minute} * 60))
if [ "${policy_weekday}" -le 5 ] &&
   [ "${now_seconds}" -ge $((9 * 3600)) ] &&
   [ "$((now_seconds + WALLTIME_SECONDS))" -le $((19 * 3600)) ]; then
    policy_type=day
fi
readonly policy_type

if [ "${NODE_TYPE}" = "gpu" ]; then
    exec oarsub -q default -t exotic -t "${policy_type}" \
        -l gpu=1,walltime="${WALLTIME}" "${command_string}"
fi
exec oarsub -q default -t "${policy_type}" -l walltime="${WALLTIME}" "${command_string}"
