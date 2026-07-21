#!/usr/bin/env bash
# Frontend-only adapter. Submits exactly one non-interactive OAR job that
# runs the bounded one-shard finalization against a previously streamed
# staging branch.  GPU is NOT required: finalization is CPU-only.

set -euo pipefail
umask 077

if [ "$#" -ne 12 ]; then
    echo "submit_streaming_finalization: exactly twelve positional arguments are required" >&2
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
   ! [[ "${WALLTIME}" =~ ^[0-9]+:[0-9]+:[0-9]+$ ]]; then
    echo "submit_streaming_finalization: invalid run-id/shard/walltime" >&2
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

if [ "${NODE_TYPE}" = "gpu" ]; then
    exec oarsub -q production -l gpu=1,walltime="${WALLTIME}" "${command_string}"
fi
exec oarsub -q default -l walltime="${WALLTIME}" "${command_string}"
