#!/usr/bin/env bash
# Frontend-only adapter. Submits exactly one short, resumable GPU job.

set -euo pipefail
umask 077

if [ "$#" -ne 11 ]; then
    echo "submit_streaming_build: exactly eleven positional arguments are required" >&2
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

for path in "${REPO_ROOT}" "${HF_HOME}" "${LOG_ROOT}"; do
    case "${path}" in /*) ;; *) echo "submit_streaming_build: persistent path must be absolute" >&2; exit 2 ;; esac
    if [ ! -d "${path}" ] || [ -L "${path}" ]; then
        echo "submit_streaming_build: persistent path must be a real directory" >&2
        exit 2
    fi
done
if ! [[ "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
   ! [[ "${INPUT_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "submit_streaming_build: revisions must be 40 lowercase hex characters" >&2
    exit 2
fi
if ! [[ "${OUTPUT_REPO_ID}" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]] || \
   ! [[ "${INPUT_REPO_ID}" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]]; then
    echo "submit_streaming_build: repository IDs must be owner/name" >&2
    exit 2
fi
if ! [[ "${RUN_ID}" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || \
   ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${MAX_SHARDS}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "submit_streaming_build: invalid run ID or numeric argument" >&2
    exit 2
fi
if [ -n "${SHARD_KEY}" ] && ! [[ "${SHARD_KEY}" =~ ^[a-z0-9]+([.-][a-z0-9]+)*-latest$ ]]; then
    echo "submit_streaming_build: invalid shard key" >&2
    exit 2
fi
if [ -n "${SHARD_KEY}" ] && [ "${MAX_SHARDS}" -ne 0 ]; then
    echo "submit_streaming_build: shard key conflicts with max shards" >&2
    exit 2
fi

WRAPPER="${REPO_ROOT}/scripts/grid5000/run_streaming_build_job.sh"
HELPER="${REPO_ROOT}/scripts/grid5000/_submit_gpu_job.sh"
if [ ! -x "${WRAPPER}" ] || [ ! -x "${HELPER}" ]; then
    echo "submit_streaming_build: wrapper or GPU submit helper is unavailable" >&2
    exit 1
fi

shell_quote() {
    printf "'%s'" "${1//\'/\'\\\'\'}"
}

command_string="exec $(shell_quote "${WRAPPER}")"
for value in "$@"; do
    command_string="${command_string} $(shell_quote "${value}")"
done

# Keep streaming allocations small so OAR can backfill them quickly.  The
# policy type is derived from the current Europe/Paris window and the full
# 30-minute walltime, so a weekday job never crosses 09:00 or 19:00.
read -r policy_weekday policy_hour policy_minute < <(
    TZ=Europe/Paris date '+%u %H %M'
)
if ! [[ "${policy_weekday}" =~ ^[1-7]$ &&
    "${policy_hour}" =~ ^[0-9]{2}$ &&
    "${policy_minute}" =~ ^[0-9]{2}$ ]]; then
    echo "submit_streaming_build: invalid Europe/Paris scheduler clock" >&2
    exit 1
fi
policy_type=night
now_seconds=$((10#${policy_hour} * 3600 + 10#${policy_minute} * 60))
if [ "${policy_weekday}" -le 5 ] &&
   [ "${now_seconds}" -ge $((9 * 3600)) ] &&
   [ "$((now_seconds + 30 * 60))" -le $((19 * 3600)) ]; then
    policy_type=day
fi

exec "${HELPER}" "40000" "00:30:00" "${policy_type}" "${command_string}"
