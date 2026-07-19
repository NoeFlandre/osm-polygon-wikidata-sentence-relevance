#!/usr/bin/env bash
# Grid'5000 non-interactive OAR submission adapter for the FULL
# resumable build (Phase 9L-B).
#
# This is a FRONTEND-ONLY helper. It is NOT the compute-node
# payload. It builds exactly one command-string argument and hands
# it to `oarsub`, which Nancy's OAR build accepts as the single
# positional program. This OAR version rejects
# `oarsub [opts] script arg1 arg2 ... argN`, so all eight
# arguments must be embedded inside one shell-quoted command string.
#
# Hard safety contract (Phase 9L-B):
#
#   * requires exactly eight positional arguments;
#   * requires EXPECTED_SOURCE_COMMIT to be exactly 40 lowercase
#     hex characters;
#   * requires INPUT_REVISION to be exactly 40 lowercase hex
#     characters; "main" is explicitly rejected because a real
#     build must bind to an immutable snapshot revision;
#   * requires the compute-node wrapper to exist and be executable
#     at ${REPO_ROOT}/scripts/grid5000/run_gpu_build_job.sh;
#   * requires canonical absolute persistent directories
#     (REPO_ROOT, HF_HOME, LOG_ROOT, INPUT_ROOT, WORK_DIR,
#     OUTPUT_DIR) and rejects ephemeral node-local storage
#     (/tmp, /var/tmp, /dev/shm), traversal (..), empty segments
#     (//), and any symlinked operator-controlled path component;
#   * requires `command -v oarsub` (the scheduler CLI lives on the
#     frontend, not on the Mac);
#   * never imports Python / Torch / SaT;
#   * never performs inference, polling, cancellation, SSH, file
#     synchronization, git mutation, downloads, retries, or cleanup;
#   * forwards oarsub stdout/stderr unchanged and returns oarsub's
#     real exit code;
#   * submits exactly once (no retry, no fallback to CPU/MPS/auto);
#   * uses the production queue with one GPU and a proposed
#     benchmark-derived walltime (pending representative-shard
#     throughput measurement; not yet validated).
#
# Storage agnosticism: the adapter is fully storage-agnostic. It
# never hard-codes site-specific mount points (home, group
# storage, or any platform-specific path); only node-local
# ephemeral locations are explicitly rejected.
# path. The operator chooses persistent directories explicitly.

set -euo pipefail

umask 077

# --- Positional arguments (exactly eight required) --------------------

if [ "$#" -ne 8 ]; then
    echo "submit_gpu_build: exactly eight positional arguments are required" >&2
    exit 1
fi

# positional args:
#   $1 = REPO_ROOT
#   $2 = HF_HOME
#   $3 = LOG_ROOT
#   $4 = INPUT_ROOT
#   $5 = WORK_DIR
#   $6 = OUTPUT_DIR
#   $7 = EXPECTED_SOURCE_COMMIT  (40 lowercase hex)
#   $8 = INPUT_REVISION          (40 lowercase hex; never "main")
REPO_ROOT="${1}"; readonly REPO_ROOT
HF_HOME="${2}"; readonly HF_HOME
LOG_ROOT="${3}"; readonly LOG_ROOT
INPUT_ROOT="${4}"; readonly INPUT_ROOT
WORK_DIR="${5}"; readonly WORK_DIR
OUTPUT_DIR="${6}"; readonly OUTPUT_DIR
EXPECTED_SOURCE_COMMIT="${7}"; readonly EXPECTED_SOURCE_COMMIT
INPUT_REVISION="${8}"; readonly INPUT_REVISION

# --- Path validation: absolute, non-empty, traversal-free, no symlink -

_require_absolute_path() {
    local label="$1"
    local value="$2"
    if [ -z "${value}" ]; then
        echo "submit_gpu_build: ${label} is empty" >&2
        return 1
    fi
    case "${value}" in
        /*) ;;
        *) echo "submit_gpu_build: ${label} is not an absolute path" >&2
           return 1 ;;
    esac
    case "${value}" in
        *..*|*//*)
            echo "submit_gpu_build: ${label} contains forbidden traversal or empty segment" >&2
            return 1
            ;;
    esac
    case "${value}" in
        */.) echo "submit_gpu_build: ${label} contains forbidden self-reference segment" >&2
             return 1 ;;
    esac
}

_require_absolute_path "REPO_ROOT"  "${REPO_ROOT}"
_require_absolute_path "HF_HOME"    "${HF_HOME}"
_require_absolute_path "LOG_ROOT"   "${LOG_ROOT}"
_require_absolute_path "INPUT_ROOT" "${INPUT_ROOT}"
_require_absolute_path "WORK_DIR"   "${WORK_DIR}"
_require_absolute_path "OUTPUT_DIR" "${OUTPUT_DIR}"

_require_real_dir() {
    local label="$1"
    local value="$2"
    local normalised
    normalised="${value%/}"
    if [ ! -d "${normalised}" ]; then
        echo "submit_gpu_build: ${label} is not a directory" >&2
        return 1
    fi
    local phys
    phys="$(cd -- "${normalised}" && pwd -P)" || return 1
    if [ "${phys}" != "${normalised}" ]; then
        echo "submit_gpu_build: ${label} must not be a symlink" >&2
        return 1
    fi
}

# REPO_ROOT, HF_HOME, LOG_ROOT, INPUT_ROOT, WORK_DIR must exist and
# be real directories (WORK_DIR may already contain checkpoints for
# a resume). OUTPUT_DIR must NOT exist (fresh-build contract); it
# is created on the compute node by the payload.
_require_real_dir "REPO_ROOT"  "${REPO_ROOT}"
_require_real_dir "HF_HOME"    "${HF_HOME}"
_require_real_dir "LOG_ROOT"   "${LOG_ROOT}"
_require_real_dir "INPUT_ROOT" "${INPUT_ROOT}"
_require_real_dir "WORK_DIR"   "${WORK_DIR}"

OUTPUT_NORMALISED="${OUTPUT_DIR%/}"
if [ -e "${OUTPUT_NORMALISED}" ]; then
    echo "submit_gpu_build: OUTPUT_DIR already exists (fresh build only)" >&2
    exit 1
fi

# Refuse ephemeral node-local storage for every persistent path.
# The check is inlined for each variable to avoid any use of
# ``eval`` (the eval-based loop form is rejected by the
# forbidden-pattern audit).
case "${REPO_ROOT}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "submit_gpu_build: REPO_ROOT points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${HF_HOME}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "submit_gpu_build: HF_HOME points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${LOG_ROOT}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "submit_gpu_build: LOG_ROOT points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${INPUT_ROOT}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "submit_gpu_build: INPUT_ROOT points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${WORK_DIR}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "submit_gpu_build: WORK_DIR points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${OUTPUT_DIR}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "submit_gpu_build: OUTPUT_DIR points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac

# --- Expected source commit: 40 lowercase hex chars ------------------

if [[ ! "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "submit_gpu_build: EXPECTED_SOURCE_COMMIT is not exactly 40 lowercase hex characters" >&2
    exit 1
fi

# --- Input revision: 40 lowercase hex chars (never "main") -----------

if [[ ! "${INPUT_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "submit_gpu_build: INPUT_REVISION must be 40 lowercase hex characters (main is rejected)" >&2
    exit 1
fi

# --- Compute-node wrapper presence + executable ----------------------

WRAPPER="${REPO_ROOT}/scripts/grid5000/run_gpu_build_job.sh"
if [ ! -f "${WRAPPER}" ]; then
    echo "submit_gpu_build: compute-node wrapper is missing" >&2
    exit 1
fi
if [ ! -x "${WRAPPER}" ]; then
    echo "submit_gpu_build: compute-node wrapper is not executable" >&2
    exit 1
fi

# --- oarsub presence ------------------------------------------------

if ! command -v oarsub >/dev/null 2>&1; then
    echo "submit_gpu_build: oarsub not found on PATH (run on the Grid'5000 frontend)" >&2
    exit 1
fi

# --- Safe single-argument shell quoting ------------------------------

shell_quote() {
    local in="$1"
    local out=""
    local c
    while [ -n "${in}" ]; do
        c="${in%"${in#?}"}"
        in="${in#?}"
        if [ "${c}" = "'" ]; then
            out="${out}'\\''"
        else
            out="${out}${c}"
        fi
    done
    printf "'%s'" "${out}"
}

# --- Build the single command string --------------------------------

QUOTED_WRAPPER="$(shell_quote "${WRAPPER}")"
QUOTED_REPO="$(shell_quote "${REPO_ROOT}")"
QUOTED_HF="$(shell_quote "${HF_HOME}")"
QUOTED_LOG="$(shell_quote "${LOG_ROOT}")"
QUOTED_INPUT="$(shell_quote "${INPUT_ROOT}")"
QUOTED_WORK="$(shell_quote "${WORK_DIR}")"
QUOTED_OUTPUT="$(shell_quote "${OUTPUT_DIR}")"
QUOTED_COMMIT="$(shell_quote "${EXPECTED_SOURCE_COMMIT}")"
QUOTED_REVISION="$(shell_quote "${INPUT_REVISION}")"

COMMAND_STRING="exec ${QUOTED_WRAPPER} ${QUOTED_REPO} ${QUOTED_HF} ${QUOTED_LOG} ${QUOTED_INPUT} ${QUOTED_WORK} ${QUOTED_OUTPUT} ${QUOTED_COMMIT} ${QUOTED_REVISION}"

# --- Submit exactly once --------------------------------------------
#
# Production queue, exactly one GPU. Walltime is a *proposed*
# upper bound (pending benchmark): the operator is expected to
# measure a representative shard and adjust the walltime to fit
# the full 291-shard resumable build. The value is NOT yet
# validated as sufficient.
RESOURCE_SPEC="gpu=1,walltime=12:00:00"

oarsub -q production -l "${RESOURCE_SPEC}" "${COMMAND_STRING}"
exit $?
