#!/usr/bin/env bash
# Submit one non-interactive Afghanistan labeling run from a Grid'5000 frontend.
#
# The launcher's parallelism is a first-class positional argument. The
# supported set is the small validated set {1, 2, 4, 8, 16, 32}; the wrapper
# propagates it as the final argument so the payload can compute the total
# context (parallel * 4096) and launch the real llama-server binary.

set -euo pipefail
umask 077

if [ "$#" -ne 15 ]; then
    echo "submit_afghanistan_labeling: exactly fifteen arguments are required" >&2
    exit 2
fi

canonicalize_path() {
    local path="$1"
    if [ -L "${path}" ]; then
        path="$(readlink -f "${path}")"
    fi
    if [ ! -e "${path}" ] && [ ! -d "$(dirname "${path}")" ]; then
        return 1
    fi
    if command -v readlink >/dev/null 2>&1; then
        local resolved
        if resolved="$(readlink -f "${path}" 2>/dev/null)"; then
            printf '%s\n' "${resolved}"
            return 0
        fi
    fi
    local dir base resolved_dir
    dir="$(dirname "${path}")"
    base="$(basename "${path}")"
    resolved_dir="$(cd "${dir}" && pwd -P)"
    printf '%s/%s\n' "${resolved_dir}" "${base}"
}

is_under_root() {
    local candidate="$1"
    local root="$2"
    case "${candidate}" in
        "${root}"|"${root}/*") return 0;;
        *) return 1;;
    esac
}

validate_path() {
    local path="$1"
    local must_exist="$2"
    local allow_symlink="$3"
    local description="$4"
    local canonicalized

    if [ ! -e "${path}" ]; then
        if [ "${must_exist}" = "yes" ]; then
            echo "submit_afghanistan_labeling: required ${description} is unavailable" >&2
            exit 2
        fi
    elif [ "${allow_symlink}" != "yes" ] && [ -L "${path}" ]; then
        echo "submit_afghanistan_labeling: ${description} must not be a symlink" >&2
        exit 2
    fi

    if ! canonicalized="$(canonicalize_path "${path}")"; then
        echo "submit_afghanistan_labeling: required ${description} is not resolvable" >&2
        exit 2
    fi
    printf '%s\n' "${canonicalized}"
}

REPO_ROOT_CANON="$(canonicalize_path "$1")" || {
    echo "submit_afghanistan_labeling: repository root is not canonicalizable" >&2
    exit 2
}
HF_HOME_CANON="$(validate_path "$2" yes no "HF home directory")"
LOG_ROOT_CANON="$(validate_path "$3" yes no "log directory")"
INPUT_PARQUET_CANON="$(validate_path "$4" yes no "input Parquet file")"
WORK_DIR_CANON="$(validate_path "$5" no no "work directory")"
OUTPUT_DIR_CANON="$(validate_path "$6" no no "output directory")"
MODEL_FILE_CANON="$(validate_path "$7" yes no "model file")"
TOKENIZER_DIR_CANON="$(validate_path "$8" yes no "tokenizer directory")"

REPO_ROOT="${REPO_ROOT_CANON}"; readonly REPO_ROOT
HF_HOME="${HF_HOME_CANON}"; readonly HF_HOME
LOG_ROOT="${LOG_ROOT_CANON}"; readonly LOG_ROOT
INPUT_PARQUET="${INPUT_PARQUET_CANON}"; readonly INPUT_PARQUET
WORK_DIR="${WORK_DIR_CANON}"; readonly WORK_DIR
OUTPUT_DIR="${OUTPUT_DIR_CANON}"; readonly OUTPUT_DIR
MODEL_FILE="${MODEL_FILE_CANON}"; readonly MODEL_FILE
TOKENIZER_DIR="${TOKENIZER_DIR_CANON}"; readonly TOKENIZER_DIR

MODEL_REVISION="$9"; readonly MODEL_REVISION
INPUT_REVISION="${10}"; readonly INPUT_REVISION
SOURCE_COMMIT="${11}"; readonly SOURCE_COMMIT
DATASET_ID="${12}"; readonly DATASET_ID
BATCH_SIZE="${13}"; readonly BATCH_SIZE
ROW_LIMIT="${14}"; readonly ROW_LIMIT
LLAMA_PARALLEL="${15}"; readonly LLAMA_PARALLEL

RUN_ROOT="${REPO_ROOT_CANON%/*}"

for path in "${REPO_ROOT}" "${HF_HOME}" "${LOG_ROOT}" "${TOKENIZER_DIR}" "${INPUT_PARQUET}" "${WORK_DIR}" "${OUTPUT_DIR}" "${MODEL_FILE}" "${TOKENIZER_DIR}"; do
    case "${path}" in
        /*) ;;
        *) echo "submit_afghanistan_labeling: path must be absolute" >&2
           exit 2
        ;;
    esac
    if ! is_under_root "${path}" "${RUN_ROOT}"; then
        echo "submit_afghanistan_labeling: path is outside the approved run root: ${path}" >&2
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
case "${LLAMA_PARALLEL}" in
    1|2|4|8|16|32) ;;
    *) echo "submit_afghanistan_labeling: LLAMA_PARALLEL must be one of 1, 2, 4, 8, 16, 32" >&2; exit 2;;
esac

for path in "${HF_HOME}" "${LOG_ROOT}" "${TOKENIZER_DIR}"; do
    if [ ! -d "${path}" ] || [ -L "${path}" ]; then
        echo "submit_afghanistan_labeling: required directory is unavailable" >&2
        exit 2
    fi
done
for path in "${INPUT_PARQUET}" "${MODEL_FILE}"; do
    if [ ! -f "${path}" ] || [ -L "${path}" ]; then
        echo "submit_afghanistan_labeling: required file is unavailable" >&2
        exit 2
    fi
done

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

# Production / non-preemptible queue: the load is sustained and must not be
# interrupted by another user's best-effort workload. The nantes site uses
# the default queue (``-q default``) with the ``exotic`` type so the job
# is scheduled on the production / non-preemptible node class. The
# ``-t besteffort`` marker is intentionally absent.
exec oarsub -q default -t exotic -p "gpu_mem>=60000" \
    -l gpu=1,walltime=12:00:00 "${command_string}"
