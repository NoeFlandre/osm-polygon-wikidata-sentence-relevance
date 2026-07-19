#!/usr/bin/env bash
# Grid'5000 non-interactive OAR submission adapter (Phase 9F).
#
# This is a FRONTEND-ONLY helper. It is NOT the compute-node
# payload. It builds exactly one command-string argument and hands
# it to `oarsub`, which Nancy's OAR build accepts as the single
# positional program. This OAR version rejects
# `oarsub [opts] script arg1 arg2 arg3 arg4`, so all four arguments
# must be embedded inside one shell-quoted command string.
#
# Why a submission adapter instead of a direct oarsub line? The
# compute-node wrapper `run_gpu_smoke_job.sh` requires exactly four
# positional arguments (REPO_ROOT HF_HOME LOG_ROOT
# EXPECTED_SOURCE_COMMIT). The interactive form (oarsub with the
# single-dash I flag) is too fragile for automation: a dropped
# local SSH transport makes OAR treat the empty shell as a frag
# request. The batch form makes the scheduler the owner of the job
# lifetime, and this adapter serializes the four arguments into the
# one command string the scheduler understands.
#
# Hard safety contract:
#
#   * requires exactly four positional arguments;
#   * requires EXPECTED_SOURCE_COMMIT to be exactly 40 lowercase
#     hex characters (the same guard the compute-node wrapper uses);
#   * requires the compute-node wrapper to exist and be executable
#     at ${REPO_ROOT}/scripts/grid5000/run_gpu_smoke_job.sh;
#   * requires canonical absolute persistent directories
#     (REPO_ROOT, HF_HOME, LOG_ROOT) and rejects ephemeral node-local
#     storage (/tmp, /var/tmp, /dev/shm), traversal (..), empty
#     segments (//), and any symlinked operator-controlled path
#     component (consistency with the job wrapper);
#   * requires `command -v oarsub` (the scheduler CLI lives on the
#     frontend, not on the Mac);
#   * never imports Python / Torch / SaT;
#   * never performs inference, polling, cancellation, SSH, file
#     synchronization, git mutation, downloads, retries, or cleanup;
#   * forwards oarsub stdout/stderr unchanged and returns oarsub's
#     real exit code; it never parses, manufactures, or validates a
#     job ID.

set -euo pipefail

# Restrictive creation permissions for any artefacts this helper
# might create (none by design, but defence-in-depth).
umask 077

# --- Positional arguments (exactly four required) --------------------

if [ "$#" -ne 4 ]; then
    echo "submit_gpu_smoke: exactly four positional arguments are required" >&2
    exit 1
fi

# positional args: $1 = REPO_ROOT, $2 = HF_HOME, $3 = LOG_ROOT,
#                   $4 = EXPECTED_SOURCE_COMMIT
REPO_ROOT="${1}"; readonly REPO_ROOT
HF_HOME="${2}"; readonly HF_HOME
LOG_ROOT="${3}"; readonly LOG_ROOT
EXPECTED_SOURCE_COMMIT="${4}"; readonly EXPECTED_SOURCE_COMMIT

# --- Path validation: absolute, non-empty, traversal-free, no symlink -

# Helper: reject an empty / non-absolute / traversal path. Returns
# non-zero (with message to stderr) when the value is unusable as a
# canonical absolute operator path.
_require_absolute_path() {
    local label="$1"
    local value="$2"
    if [ -z "${value}" ]; then
        echo "submit_gpu_smoke: ${label} is empty" >&2
        return 1
    fi
    case "${value}" in
        /*) ;;
        *) echo "submit_gpu_smoke: ${label} is not an absolute path" >&2
           return 1 ;;
    esac
    case "${value}" in
        *..*|*//*)
            echo "submit_gpu_smoke: ${label} contains forbidden traversal or empty segment" >&2
            return 1
            ;;
    esac
    case "${value}" in
        */.) echo "submit_gpu_smoke: ${label} contains forbidden self-reference segment" >&2
             return 1 ;;
    esac
}

_require_absolute_path "REPO_ROOT" "${REPO_ROOT}"
_require_absolute_path "HF_HOME" "${HF_HOME}"
_require_absolute_path "LOG_ROOT" "${LOG_ROOT}"

# Helper: require a real directory at an operator-controlled path
# and reject any symlink at that component (consistency with the
# compute-node wrapper). The physical path must equal the supplied
# path character-by-character; a symlink would make them differ.
_require_real_dir() {
    local label="$1"
    local value="$2"
    local normalised
    normalised="${value%/}"
    if [ ! -d "${normalised}" ]; then
        echo "submit_gpu_smoke: ${label} is not a directory" >&2
        return 1
    fi
    local phys
    phys="$(cd -- "${normalised}" && pwd -P)" || return 1
    if [ "${phys}" != "${normalised}" ]; then
        echo "submit_gpu_smoke: ${label} must not be a symlink" >&2
        return 1
    fi
}

_require_real_dir "REPO_ROOT" "${REPO_ROOT}"
_require_real_dir "HF_HOME" "${HF_HOME}"
_require_real_dir "LOG_ROOT" "${LOG_ROOT}"

# Refuse ephemeral node-local storage. The OAR compute node's /tmp
# is ephemeral; the smoke and its cache MUST NOT live there.
case "${HF_HOME}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "submit_gpu_smoke: HF_HOME points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${LOG_ROOT}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "submit_gpu_smoke: LOG_ROOT points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac

# --- Expected source commit: 40 lowercase hex chars ------------------

if [[ ! "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "submit_gpu_smoke: EXPECTED_SOURCE_COMMIT is not exactly 40 lowercase hex characters" >&2
    exit 1
fi

# --- Compute-node wrapper presence + executable ----------------------
#
# The path is held in WRAPPER for the oarsub invocation below but
# is never echoed to the operator (defence against leaking the
# filesystem layout in error messages).

WRAPPER="${REPO_ROOT}/scripts/grid5000/run_gpu_smoke_job.sh"
if [ ! -f "${WRAPPER}" ]; then
    echo "submit_gpu_smoke: compute-node wrapper is missing" >&2
    exit 1
fi
if [ ! -x "${WRAPPER}" ]; then
    echo "submit_gpu_smoke: compute-node wrapper is not executable" >&2
    exit 1
fi

# --- oarsub presence ------------------------------------------------

if ! command -v oarsub >/dev/null 2>&1; then
    echo "submit_gpu_smoke: oarsub not found on PATH (run on the Grid'5000 frontend)" >&2
    exit 1
fi

# --- Safe single-argument shell quoting ------------------------------
#
# Portable POSIX quoting. Every value is wrapped in single quotes;
# any embedded single quote is closed, escaped with a backslash
# quote, and reopened using the `'\''` idiom. No `eval`, no
# unquoted interpolation, no Bash `%q`, no temporary scripts.
#
# After quoting, the value is safe to embed inside the single
# command string that `oarsub` receives: spaces, semicolons, `$()`,
# backticks, wildcards, and single quotes stay literal and cannot
# be executed by the shell that later runs the command string.
shell_quote() {
    local in="$1"
    local out=""
    local c
    while [ -n "${in}" ]; do
        c="${in%"${in#?}"}"   # first character of in
        in="${in#?}"          # remainder
        if [ "${c}" = "'" ]; then
            out="${out}'\\''"
        else
            out="${out}${c}"
        fi
    done
    printf "'%s'" "${out}"
}

# --- Build the single command string --------------------------------
#
# The payload starts with `exec` so the wrapper becomes the job
# process (no redundant subshell). All four arguments are quoted
# individually using shell_quote.
QUOTED_WRAPPER="$(shell_quote "${WRAPPER}")"
QUOTED_REPO="$(shell_quote "${REPO_ROOT}")"
QUOTED_HF="$(shell_quote "${HF_HOME}")"
QUOTED_LOG="$(shell_quote "${LOG_ROOT}")"
QUOTED_COMMIT="$(shell_quote "${EXPECTED_SOURCE_COMMIT}")"

COMMAND_STRING="exec ${QUOTED_WRAPPER} ${QUOTED_REPO} ${QUOTED_HF} ${QUOTED_LOG} ${QUOTED_COMMIT}"

# --- Submit exactly once --------------------------------------------
#
# Forward oarsub stdout/stderr to the operator unchanged, and exit
# with oarsub's real exit status. No job-ID parsing, no retry, no
# fallback. A non-zero oarsub exit is the failure signal the
# operator must triage.
oarsub -q production -l gpu=1,walltime=00:30:00 "${COMMAND_STRING}"
exit $?
