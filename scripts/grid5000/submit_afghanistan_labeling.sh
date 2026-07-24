#!/usr/bin/env bash
# Submit one non-interactive Afghanistan labeling run from a Grid'5000 frontend.

set -euo pipefail
umask 077

if [ "$#" -ne 14 ]; then
    echo "submit_afghanistan_labeling: exactly fourteen arguments are required" >&2
    exit 2
fi

REPO_ROOT="$1"; readonly REPO_ROOT
HF_HOME="$2"; readonly HF_HOME
LOG_ROOT="$3"; readonly LOG_ROOT
INPUT_PARQUET="$4"; readonly INPUT_PARQUET
WORK_DIR="$5"; readonly WORK_DIR
OUTPUT_DIR="$6"; readonly OUTPUT_DIR
MODEL_FILE="$7"; readonly MODEL_FILE
TOKENIZER_DIR="$8"; readonly TOKENIZER_DIR
MODEL_REVISION="$9"; readonly MODEL_REVISION
INPUT_REVISION="${10}"; readonly INPUT_REVISION
SOURCE_COMMIT="${11}"; readonly SOURCE_COMMIT
DATASET_ID="${12}"; readonly DATASET_ID
BATCH_SIZE="${13}"; readonly BATCH_SIZE
ROW_LIMIT="${14}"; readonly ROW_LIMIT

for path in "${REPO_ROOT}" "${HF_HOME}" "${LOG_ROOT}" "${TOKENIZER_DIR}"; do
    case "${path}" in /*) ;; *) echo "submit_afghanistan_labeling: directory paths must be absolute" >&2; exit 2;; esac
    if [ ! -d "${path}" ] || [ -L "${path}" ]; then
        echo "submit_afghanistan_labeling: required directory is unavailable" >&2
        exit 2
    fi
done
for path in "${INPUT_PARQUET}" "${MODEL_FILE}"; do
    case "${path}" in /*) ;; *) echo "submit_afghanistan_labeling: file paths must be absolute" >&2; exit 2;; esac
    if [ ! -f "${path}" ] || [ -L "${path}" ]; then
        echo "submit_afghanistan_labeling: required file is unavailable" >&2
        exit 2
    fi
done
for path in "${WORK_DIR}" "${OUTPUT_DIR}"; do
    case "${path}" in /*) ;; *) echo "submit_afghanistan_labeling: result paths must be absolute" >&2; exit 2;; esac
    if [ -L "${path}" ]; then
        echo "submit_afghanistan_labeling: result paths must not be symlinks" >&2
        exit 2
    fi
done
if ! [[ "${MODEL_REVISION}" =~ ^[0-9a-f]{40}$ ]] || \
   ! [[ "${INPUT_REVISION}" =~ ^[0-9a-f]{40}$ ]] || \
   ! [[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "submit_afghanistan_labeling: revisions must be immutable lowercase commits" >&2
    exit 2
fi
if ! [[ "${DATASET_ID}" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]] || \
   ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${ROW_LIMIT}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "submit_afghanistan_labeling: dataset ID or numeric argument is invalid" >&2
    exit 2
fi

WRAPPER="${REPO_ROOT}/scripts/grid5000/run_afghanistan_labeling_job.sh"
if [ ! -x "${WRAPPER}" ] || ! command -v oarsub >/dev/null 2>&1; then
    echo "submit_afghanistan_labeling: wrapper or oarsub is unavailable" >&2
    exit 1
fi

shell_quote() {
    printf "'%s'" "${1//\'/\'\\\'\'}"
}

command_string="exec $(shell_quote "${WRAPPER}")"
for value in "$@"; do
    command_string="${command_string} $(shell_quote "${value}")"
done

exec oarsub -q default -t exotic -p "gpu_mem>=60000" \
    -l gpu=1,walltime=01:00:00 "${command_string}"
