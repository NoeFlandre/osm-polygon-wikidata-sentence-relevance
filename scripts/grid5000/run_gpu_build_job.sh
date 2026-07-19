#!/usr/bin/env bash
# Grid'5000 non-interactive batch entrypoint for the FULL
# resumable build (Phase 9L-B).
#
# This script is the *compute-node wrapper* for a non-interactive
# OAR batch job. It is the script that the scheduler invokes inside
# the allocation, after the scheduler has set ``OAR_JOB_ID``.
# It does NOT submit a job; it is the job payload.
#
# ``CUDA_VISIBLE_DEVICES`` is informational only. Grid'5000 scopes
# reserved GPUs through its resource isolation and does not
# guarantee ``CUDA_VISIBLE_DEVICES`` is set on a given allocation.
# The wrapper therefore never requires, reads, assigns, defaults,
# normalises, prints, or exports it. The authoritative runtime proof
# of GPU scoping is ``torch.cuda.device_count() == 1`` inside
# ``gpu_preflight.py``, which runs as the first phase of the
# committed build payload.
#
# Hard safety contract:
#
#   * requires OAR_JOB_ID in the scheduler-provided environment;
#     never overwrites it;
#   * never touches CUDA_VISIBLE_DEVICES (informational only);
#   * requires the eight positional arguments REPO_ROOT HF_HOME
#     LOG_ROOT INPUT_ROOT WORK_DIR OUTPUT_DIR EXPECTED_SOURCE_COMMIT
#     INPUT_REVISION;
#   * rejects /tmp, /var/tmp, /dev/shm, traversal, symlinks,
#     empty paths, and any non-absolute path;
#   * rejects overlap between INPUT_ROOT, WORK_DIR, and OUTPUT_DIR
#     (no path may be nested under another);
#   * requires INPUT_ROOT to exist (the snapshot must already be
#     staged);
#   * requires OUTPUT_DIR to be fresh (does not exist yet);
#   * accepts WORK_DIR that already exists and may contain valid
#     checkpoints for resume;
#   * refuses to reuse or overwrite an existing
#     ${LOG_ROOT}/${OAR_JOB_ID};
#   * requires the git HEAD inside ${REPO_ROOT} to equal
#     EXPECTED_SOURCE_COMMIT (verified before any model work);
#   * requires the working tree to be clean;
#   * invokes the locked venv interpreter, never a bare interpreter;
#   * invokes the committed build payload
#     ${REPO_ROOT}/scripts/grid5000/run_gpu_build.sh exactly
#     once;
#   * captures stdout, stderr, and the real exit code without
#     masking failure or retrying;
#   * never exports schedule-owned variables;
#   * never edits, commits, or pushes source;
#   * never sends dataset / publication material, never downloads
#     model weights, never falls back to CPU/MPS/auto;
#   * leaves WORK_DIR untouched on failure (resume contract).

set -euo pipefail

umask 077

# --- Scheduler-owned variables (refuse to overwrite) -----------------

: "${OAR_JOB_ID:?OAR_JOB_ID is required (set by the OAR scheduler; never modify)}"

if ! [[ "${OAR_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "run_gpu_build_job: OAR_JOB_ID is not numeric" >&2
    exit 1
fi

# --- Positional arguments (exactly eight required) --------------------

if [ "$#" -ne 8 ]; then
    echo "run_gpu_build_job: exactly eight positional arguments are required" >&2
    exit 1
fi

REPO_ROOT="${1}"; readonly REPO_ROOT                       # "$1"
HF_HOME="${2}"; readonly HF_HOME                           # "$2"
LOG_ROOT="${3}"; readonly LOG_ROOT                         # "$3"
INPUT_ROOT="${4}"; readonly INPUT_ROOT                     # "$4"
WORK_DIR="${5}"; readonly WORK_DIR                         # "$5"
OUTPUT_DIR="${6}"; readonly OUTPUT_DIR                     # "$6"
EXPECTED_SOURCE_COMMIT="${7}"; readonly EXPECTED_SOURCE_COMMIT  # "$7"
INPUT_REVISION="${8}"; readonly INPUT_REVISION             # "$8"

# --- Path validation: absolute, non-empty, traversal-free, no symlink -

_require_absolute_path() {
    local label="$1"
    local value="$2"
    if [ -z "${value}" ]; then
        echo "run_gpu_build_job: ${label} is empty" >&2
        exit 1
    fi
    case "${value}" in
        /*) ;;
        *) echo "run_gpu_build_job: ${label} is not an absolute path" >&2
           exit 1 ;;
    esac
    case "${value}" in
        *..*|*//*)
            echo "run_gpu_build_job: ${label} contains forbidden traversal or empty segment" >&2
            exit 1
            ;;
    esac
    case "${value}" in
        */.) echo "run_gpu_build_job: ${label} contains forbidden self-reference segment" >&2
             exit 1 ;;
    esac
}

_require_absolute_path "REPO_ROOT"  "${REPO_ROOT}"
_require_absolute_path "HF_HOME"    "${HF_HOME}"
_require_absolute_path "LOG_ROOT"   "${LOG_ROOT}"
_require_absolute_path "INPUT_ROOT" "${INPUT_ROOT}"
_require_absolute_path "WORK_DIR"   "${WORK_DIR}"
_require_absolute_path "OUTPUT_DIR" "${OUTPUT_DIR}"

_canonicalise_directory() {
    local value="$1"
    local normalised
    normalised="${value%/}"
    if [ ! -d "${normalised}" ]; then
        return 1
    fi
    local phys
    phys="$(cd -- "${normalised}" && pwd -P)" || return 1
    if [ "${phys}" != "${normalised}" ]; then
        return 1
    fi
    printf '%s' "${phys}"
}

# Canonicalise the directories that must already exist. OUTPUT_DIR
# is the fresh-build contract: it must NOT exist before invocation.
if ! REPO_ROOT_REAL="$(_canonicalise_directory "${REPO_ROOT}")"; then
    echo "run_gpu_build_job: REPO_ROOT fails strict canonicalisation" >&2
    exit 1
fi
if ! HF_HOME_REAL="$(_canonicalise_directory "${HF_HOME}")"; then
    echo "run_gpu_build_job: HF_HOME fails strict canonicalisation" >&2
    exit 1
fi
if ! LOG_ROOT_REAL="$(_canonicalise_directory "${LOG_ROOT}")"; then
    echo "run_gpu_build_job: LOG_ROOT fails strict canonicalisation" >&2
    exit 1
fi
if ! INPUT_ROOT_REAL="$(_canonicalise_directory "${INPUT_ROOT}")"; then
    echo "run_gpu_build_job: INPUT_ROOT fails strict canonicalisation (is the snapshot staged?)" >&2
    exit 1
fi
# WORK_DIR may exist (resume) or not (fresh). If it exists, it must
# canonicalise cleanly (no symlinks).
WORK_NORMALISED="${WORK_DIR%/}"
if [ -e "${WORK_NORMALISED}" ]; then
    if [ ! -d "${WORK_NORMALISED}" ]; then
        echo "run_gpu_build_job: WORK_DIR exists but is not a directory" >&2
        exit 1
    fi
    if ! WORK_DIR_REAL="$(_canonicalise_directory "${WORK_DIR}")"; then
        echo "run_gpu_build_job: WORK_DIR fails strict canonicalisation" >&2
        exit 1
    fi
else
    WORK_DIR_REAL="${WORK_NORMALISED}"
fi

OUTPUT_NORMALISED="${OUTPUT_DIR%/}"
if [ -e "${OUTPUT_NORMALISED}" ]; then
    echo "run_gpu_build_job: OUTPUT_DIR already exists (fresh build only)" >&2
    exit 1
fi

if [ ! -r "${HF_HOME_REAL}" ]; then
    echo "run_gpu_build_job: HF_HOME is not a readable directory" >&2
    exit 1
fi

# Refuse ephemeral node-local storage.
# The check is inlined for each variable to avoid any use of
# ``eval`` (the eval-based loop form is rejected by the
# forbidden-pattern audit).
case "${REPO_ROOT}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_build_job: REPO_ROOT points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${HF_HOME}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_build_job: HF_HOME points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${LOG_ROOT}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_build_job: LOG_ROOT points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${INPUT_ROOT}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_build_job: INPUT_ROOT points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${WORK_DIR}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_build_job: WORK_DIR points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${OUTPUT_DIR}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_build_job: OUTPUT_DIR points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac

# --- Path overlap checks --------------------------------------------
#
# INPUT_ROOT, WORK_DIR, and OUTPUT_DIR must be disjoint subtrees.
# No path may be a subdirectory of another. Use the canonicalised
# forms for the directories that exist (REPO_ROOT, HF_HOME, LOG_ROOT,
# INPUT_ROOT, WORK_DIR if it existed) and the normalised form for
# OUTPUT_DIR.
_check_overlap() {
    local label_a="$1"
    local path_a="$2"
    local label_b="$3"
    local path_b="$4"
    # Two paths overlap if one is a strict prefix of the other at a
    # path-segment boundary. Identical paths also overlap.
    if [ "${path_a}" = "${path_b}" ]; then
        echo "run_gpu_build_job: ${label_a} and ${label_b} are the same path" >&2
        exit 1
    fi
    case "${path_a}" in
        "${path_b}/"*)
            echo "run_gpu_build_job: ${label_a} overlaps ${label_b}" >&2
            exit 1
            ;;
    esac
    case "${path_b}" in
        "${path_a}/"*)
            echo "run_gpu_build_job: ${label_b} overlaps ${label_a}" >&2
            exit 1
            ;;
    esac
}

_check_overlap "INPUT_ROOT" "${INPUT_ROOT_REAL}" "WORK_DIR"   "${WORK_DIR_REAL}"
_check_overlap "INPUT_ROOT" "${INPUT_ROOT_REAL}" "OUTPUT_DIR" "${OUTPUT_NORMALISED}"
_check_overlap "WORK_DIR"   "${WORK_DIR_REAL}"   "OUTPUT_DIR" "${OUTPUT_NORMALISED}"
_check_overlap "INPUT_ROOT" "${INPUT_ROOT_REAL}" "REPO_ROOT"  "${REPO_ROOT_REAL}"
_check_overlap "WORK_DIR"   "${WORK_DIR_REAL}"   "REPO_ROOT"  "${REPO_ROOT_REAL}"
_check_overlap "OUTPUT_DIR" "${OUTPUT_NORMALISED}" "REPO_ROOT" "${REPO_ROOT_REAL}"

# --- Expected source commit: 40 lowercase hex chars ------------------

if [[ ! "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "run_gpu_build_job: EXPECTED_SOURCE_COMMIT is not exactly 40 lowercase hex characters" >&2
    exit 1
fi

# --- Input revision: 40 lowercase hex chars (never "main") -----------

if [[ ! "${INPUT_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "run_gpu_build_job: INPUT_REVISION must be 40 lowercase hex characters (main is rejected)" >&2
    exit 1
fi

# --- Locked interpreter and committed build payload -----------------

PROJECT_PYTHON="${REPO_ROOT_REAL}/.venv/bin/python"
BUILD_PAYLOAD="${REPO_ROOT_REAL}/scripts/grid5000/run_gpu_build.sh"

if [ ! -f "${PROJECT_PYTHON}" ]; then
    echo "run_gpu_build_job: locked project interpreter is missing" >&2
    exit 1
fi
if [ ! -x "${PROJECT_PYTHON}" ]; then
    echo "run_gpu_build_job: locked project interpreter is not executable" >&2
    exit 1
fi
if [ ! -f "${BUILD_PAYLOAD}" ]; then
    echo "run_gpu_build_job: committed build payload is missing" >&2
    exit 1
fi
if [ ! -r "${BUILD_PAYLOAD}" ]; then
    echo "run_gpu_build_job: committed build payload is not readable" >&2
    exit 1
fi

# --- Git HEAD and dirty-tree check ----------------------------------

if ! command -v git >/dev/null 2>&1; then
    echo "run_gpu_build_job: git binary is required (command -v git failed)" >&2
    exit 1
fi

SOURCE_COMMIT="$(
    set +e
    git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null
    git_rev_parse_rc=$?
    set -e
    if [ "${git_rev_parse_rc}" -ne 0 ]; then
        echo "run_gpu_build_job: git rev-parse HEAD failed" >&2
        exit 1
    fi
)"
if [ -z "${SOURCE_COMMIT}" ]; then
    echo "run_gpu_build_job: git rev-parse HEAD returned empty" >&2
    exit 1
fi

DIRTY_OUTPUT="$(
    set +e
    git -C "${REPO_ROOT}" status --porcelain 2>/dev/null
    git_status_rc=$?
    set -e
    if [ "${git_status_rc}" -ne 0 ]; then
        echo "run_gpu_build_job: git status --porcelain failed" >&2
        exit 1
    fi
)"
if [ -n "${DIRTY_OUTPUT}" ]; then
    echo "run_gpu_build_job: working tree is dirty (git status --porcelain is non-empty)" >&2
    exit 1
fi

if [ "${SOURCE_COMMIT}" != "${EXPECTED_SOURCE_COMMIT}" ]; then
    echo "run_gpu_build_job: source commit does not match EXPECTED_SOURCE_COMMIT" >&2
    exit 1
fi

# --- Per-OAR-job log directory (fresh, mode 0700) --------------------

JOB_LOG_DIR="${LOG_ROOT_REAL}/${OAR_JOB_ID}"
if [ -e "${JOB_LOG_DIR}" ]; then
    echo "run_gpu_build_job: refusing to reuse existing job log directory" >&2
    exit 1
fi

if ! mkdir -m 0700 "${JOB_LOG_DIR}"; then
    echo "run_gpu_build_job: failed to create job log directory" >&2
    exit 1
fi
chmod 0700 "${JOB_LOG_DIR}"

# --- Invoke the committed build payload exactly once -----------------

BUILD_STDOUT="${JOB_LOG_DIR}/build.stdout.log"
BUILD_STDERR="${JOB_LOG_DIR}/build.stderr.log"
BUILD_EXIT_CODE="${JOB_LOG_DIR}/build.exit_code"

# Propagate the canonicalised environment to the build payload.
# OAR_JOB_ID is scheduler-owned and must NEVER be reassigned.
# CUDA_VISIBLE_DEVICES is informational only; if the scheduler
# set it, the payload inherits it unchanged.
BUILD_LOG_DIR="${JOB_LOG_DIR}"
export REPO_ROOT HF_HOME BUILD_LOG_DIR INPUT_ROOT WORK_DIR OUTPUT_DIR
export EXPECTED_SOURCE_COMMIT INPUT_REVISION

set +e
bash "${BUILD_PAYLOAD}" \
    >"${BUILD_STDOUT}" \
    2>"${BUILD_STDERR}"
build_rc=$?
set -e

umask 077
printf '%s\n' "${build_rc}" > "${BUILD_EXIT_CODE}"
chmod 0600 "${BUILD_EXIT_CODE}"

# WORK_DIR is left untouched on both success and failure so the
# resume contract holds. The payload itself records exit code
# without masking, retry, or fallback.

exit "${build_rc}"
