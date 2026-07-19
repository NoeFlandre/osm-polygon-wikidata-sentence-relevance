#!/usr/bin/env bash
# Grid'5000 non-interactive batch entrypoint (Phase 9D + Phase 9H).
#
# This script is the *compute-node wrapper* for a non-interactive
# OAR batch job. It is the script that ``oarsub`` invokes inside
# the allocation, after the scheduler has set ``OAR_JOB_ID``.
# It does NOT submit a job; it is the job payload.
#
# Phase 9H: ``CUDA_VISIBLE_DEVICES`` is informational only.
# Grid'5000 scopes reserved GPUs through its resource isolation
# and does not guarantee ``CUDA_VISIBLE_DEVICES`` is set on a
# given allocation. The wrapper therefore never requires, reads,
# assigns, defaults, normalises, prints, or exports it. The
# authoritative runtime proof of GPU scoping is
# ``torch.cuda.device_count() == 1`` inside ``gpu_preflight.py``,
# which runs as the first phase of the committed smoke harness.
#
# Why batch instead of the interactive form of oarsub? An
# interactive OAR job depends on a TTY-bridge shell that the
# operator holds open. The shell on a local Mac session can be
# killed by tool timeouts, network drops, or laptop sleeps; when
# the local SSH transport dies, OAR treats the empty interactive
# shell as a frag request and finalizes the allocation without
# running any payload. The batch form makes the scheduler, not
# SSH, the owner of the job lifetime.
#
# Hard safety contract:
#
#   * requires OAR_JOB_ID in the scheduler-provided environment;
#     never overwrites it;
#   * never touches CUDA_VISIBLE_DEVICES (informational only;
#     the compute-node preflight proves GPU scoping via Torch);
#   * requires the four positional arguments REPO_ROOT HF_HOME
#     LOG_ROOT EXPECTED_SOURCE_COMMIT -- they are not optional;
#   * rejects /tmp, /var/tmp, /dev/shm, traversal, symlinks,
#     empty paths, and any non-absolute path;
#   * refuses to reuse or overwrite an existing
#     ${LOG_ROOT}/${OAR_JOB_ID};
#   * invokes the locked venv launcher never a bare launcher;
#   * invokes the existing committed smoke harness
#     ${REPO_ROOT}/scripts/grid5000/run_gpu_smoke.sh exactly
#     once;
#   * captures stdout, stderr, and the real exit code without
#     masking failure or retrying;
#   * never exports schedule-owned variables;
#   * never edits, commits, or pushes source;
#   * never sends dataset / publication material, never builds
#     the dataset, never downloads model weights, never falls
#     back to CPU/MPS/auto.

set -euo pipefail

# Restrictive creation permissions for log artefacts.
umask 077

# --- Scheduler-owned variables (refuse to overwrite) -----------------

: "${OAR_JOB_ID:?OAR_JOB_ID is required (set by the OAR scheduler; never modify)}"

# OAR_JOB_ID is used below as a path component of the
# per-job log directory. Validate that it is exactly a
# sequence of decimal digits BEFORE any filesystem use, and
# never echo the supplied value back to the operator (the
# value is scheduler-owned and may carry injection-style
# content if a wrapper is reused outside the cluster).
if ! [[ "${OAR_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "run_gpu_smoke_job: OAR_JOB_ID is not numeric" >&2
    exit 1
fi

# --- Positional arguments (exactly four required) --------------------

if [ "$#" -ne 4 ]; then
    echo "run_gpu_smoke_job: exactly four positional arguments are required" >&2
    exit 1
fi

REPO_ROOT="${1}"; readonly REPO_ROOT                       # "$1" - persistent repo root
HF_HOME="${2}"; readonly HF_HOME                           # "$2" - persistent HF cache root
LOG_ROOT="${3}"; readonly LOG_ROOT                         # "$3" - persistent log root
EXPECTED_SOURCE_COMMIT="${4}"; readonly EXPECTED_SOURCE_COMMIT  # "$4" - exact commit

# --- Path validation: absolute, non-empty, traversal-free, no symlink -

# Helper: reject an empty / non-absolute / traversal path.
_require_absolute_path() {
    local label="$1"
    local value="$2"
    if [ -z "${value}" ]; then
        echo "run_gpu_smoke_job: ${label} is empty" >&2
        exit 1
    fi
    case "${value}" in
        /*) ;;
        *) echo "run_gpu_smoke_job: ${label} is not an absolute path" >&2
           exit 1 ;;
    esac
    case "${value}" in
        *..*|*//*)
            echo "run_gpu_smoke_job: ${label} contains forbidden traversal or empty segment" >&2
            exit 1
            ;;
    esac
    case "${value}" in
        */.) echo "run_gpu_smoke_job: ${label} contains forbidden self-reference segment" >&2
             exit 1 ;;
    esac
}

_require_absolute_path "REPO_ROOT" "${REPO_ROOT}"
_require_absolute_path "HF_HOME" "${HF_HOME}"
_require_absolute_path "LOG_ROOT" "${LOG_ROOT}"

# Strict directory-only canonicaliser.
#
# Goal: every input MUST be an existing absolute directory whose
# normalised textual form equals its physical resolved form. Any
# discrepancy indicates the operator-supplied path crosses a
# symlink somewhere along its components, which we refuse as a
# security-relevant ambiguity: the wrapper would otherwise
# silently resolve to a path the operator did not type.
#
# Steps:
#   1. Reject non-absolute, empty, traversal, and self-reference
#      inputs (already done by _require_absolute_path above).
#   2. Strip an optional single trailing slash so that
#      ``/home/x`` and ``/home/x/`` compare equal.
#   3. Refuse non-directories; the wrapper requires an existing
#      directory at each of REPO_ROOT, HF_HOME, LOG_ROOT.
#   4. Resolve via ``cd -- "${value}" && pwd -P``. ``cd --`` is
#      the portable POSIX form that treats the next argument as
#      a path even if it begins with ``-``; ``pwd -P`` prints
#      the canonical physical path with all symlinks resolved.
#   5. Require that the resolved physical path equals the
#      normalised supplied value, character-by-character. Any
#      inequality indicates an intermediate or final symlink
#      and is rejected.
#
# The function never echoes the supplied value back to the
# operator; the error messages below are deliberately path-free
# to avoid leaking filesystem layout if the wrapper is reused
# outside the cluster.
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

if ! REPO_ROOT_REAL="$(_canonicalise_directory "${REPO_ROOT}")"; then
    echo "run_gpu_smoke_job: REPO_ROOT fails strict canonicalisation" >&2
    exit 1
fi
if ! HF_HOME_REAL="$(_canonicalise_directory "${HF_HOME}")"; then
    echo "run_gpu_smoke_job: HF_HOME fails strict canonicalisation" >&2
    exit 1
fi
if ! LOG_ROOT_REAL="$(_canonicalise_directory "${LOG_ROOT}")"; then
    echo "run_gpu_smoke_job: LOG_ROOT fails strict canonicalisation" >&2
    exit 1
fi

# The canonicaliser above already verified directory existence.
# HF_HOME must additionally be readable (it is a populated
# cache that the smoke harness will read from).
if [ ! -r "${HF_HOME_REAL}" ]; then
    echo "run_gpu_smoke_job: HF_HOME is not a readable directory" >&2
    exit 1
fi

# Refuse ephemeral node-local storage for HF_HOME and the
# per-job log directory. The OAR compute node's /tmp is
# ephemeral; the smoke MUST NOT write to it.
case "${HF_HOME}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_smoke_job: HF_HOME points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${LOG_ROOT}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_smoke_job: LOG_ROOT points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac

# --- Expected source commit: 40 lowercase hex chars ------------------

if [[ ! "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "run_gpu_smoke_job: EXPECTED_SOURCE_COMMIT is not exactly 40 lowercase hex characters" >&2
    exit 1
fi

# --- Locked interpreter and committed smoke harness ------------------

PROJECT_PYTHON="${REPO_ROOT_REAL}/.venv/bin/python"
SMOKE_HARNESS="${REPO_ROOT_REAL}/scripts/grid5000/run_gpu_smoke.sh"

if [ ! -f "${PROJECT_PYTHON}" ]; then
    echo "run_gpu_smoke_job: locked project interpreter is missing" >&2
    exit 1
fi
if [ ! -x "${PROJECT_PYTHON}" ]; then
    echo "run_gpu_smoke_job: locked project interpreter is not executable" >&2
    exit 1
fi
if [ ! -f "${SMOKE_HARNESS}" ]; then
    echo "run_gpu_smoke_job: committed smoke harness is missing" >&2
    exit 1
fi
if [ ! -r "${SMOKE_HARNESS}" ]; then
    echo "run_gpu_smoke_job: committed smoke harness is not readable" >&2
    exit 1
fi

# --- Per-OAR-job log directory (fresh, mode 0700) --------------------
#
# The wrapper explicitly refuses to reuse an existing
# ${LOG_ROOT}/${OAR_JOB_ID} layout. The committed smoke harness
# would already refuse to overwrite any of the six artefacts,
# but a higher-level fail-fast is also required: the operator
# may re-submit the same OAR_JOB_ID by accident.

JOB_LOG_DIR="${LOG_ROOT_REAL}/${OAR_JOB_ID}"
if [ -e "${JOB_LOG_DIR}" ]; then
    echo "run_gpu_smoke_job: refusing to reuse existing job log directory" >&2
    exit 1
fi

# mkdir without -p: an existing parent directory failure will
# surface immediately. Mode 0700 is enforced by the literal
# argument; umask 077 above ensures supplementary bits stay
# restricted.
if ! mkdir -m 0700 "${JOB_LOG_DIR}"; then
    echo "run_gpu_smoke_job: failed to create job log directory" >&2
    exit 1
fi

# Defence in depth: enforce mode 0700 even if mkdir honours
# only the requested bits. This guards against a future change
# of umask that might allow group/world leakage.
chmod 0700 "${JOB_LOG_DIR}"

# --- Invoke the committed smoke harness exactly once -----------------
#
# Capture stdout, stderr, and the real exit code. The harness
# writes its own three JSON artefacts into ${JOB_LOG_DIR}; the
# wrapper only owns the two log files and the exit-code file.

SMOKE_STDOUT="${JOB_LOG_DIR}/smoke.stdout.log"
SMOKE_STDERR="${JOB_LOG_DIR}/smoke.stderr.log"
SMOKE_EXIT_CODE="${JOB_LOG_DIR}/smoke.exit_code"

# The committed smoke harness reads the environment variables that
# we propagate from the wrapper's positional arguments. OAR_JOB_ID
# is scheduler-owned and must NEVER be reassigned by the wrapper --
# the ``env`` of the bash invocation below inherits it directly.
# CUDA_VISIBLE_DEVICES is informational only (Phase 9H); if the
# scheduler set it, the harness inherits it unchanged. The wrapper
# never reads, assigns, defaults, normalises, or exports it.
SMOKE_LOG_DIR="${JOB_LOG_DIR}"
export REPO_ROOT HF_HOME SMOKE_LOG_DIR EXPECTED_SOURCE_COMMIT

# The harness validates paths, OAR_JOB_ID, and
# EXPECTED_SOURCE_COMMIT *again* before any model work; this is
# intentional defence-in-depth, not a duplicated contract.
set +e
bash "${SMOKE_HARNESS}" \
    >"${SMOKE_STDOUT}" \
    2>"${SMOKE_STDERR}"
smoke_rc=$?
set -e

# Capture the real exit status. No masking, no retry, no
# fallback. A non-zero exit code is the failure signal the
# operator must triage.
umask 077
printf '%s\n' "${smoke_rc}" > "${SMOKE_EXIT_CODE}"
# Restrictive permissions on the exit-code file (defence in
# depth; the parent job log dir is already 0700).
chmod 0600 "${SMOKE_EXIT_CODE}"

# --- Success artefact contract --------------------------------------
#
# When the smoke harness exits 0, the operator must be able to
# trust that the per-job log directory holds EXACTLY six direct
# entries, each of which is one of the expected regular files
# (mode 0600), inside a 0700 directory. We verify this
# postcondition without modifying anything; on violation we
# abort with a path-free error and leave all partial and any
# unexpected entries intact for forensic review. A failing
# harness (smoke_rc != 0) skips this check: partial artefacts
# are part of the failure evidence.
if [ "${smoke_rc}" -eq 0 ]; then
    _expected_files=(
        "gpu_preflight.json"
        "run_metadata.json"
        "smoke_result.json"
        "smoke.stdout.log"
        "smoke.stderr.log"
        "smoke.exit_code"
    )

    # Verify the directory mode is exactly 0700.
    _actual_mode=$(stat -c '%a' "${JOB_LOG_DIR}" 2>/dev/null \
        || stat -f '%Lp' "${JOB_LOG_DIR}")
    if [ "${_actual_mode}" != "700" ]; then
        echo "run_gpu_smoke_job: job log directory mode is not 0700" >&2
        exit 1
    fi

    # Verify the directory holds EXACTLY six direct child
    # entries of ANY type. Counting every direct entry (not
    # just regular files) guarantees the harness has neither
    # added an unexpected file NOR an unexpected directory,
    # symlink, FIFO, or socket. The per-name loop below then
    # independently requires that each of the six expected
    # names is a regular file with mode 0600. We never follow
    # symlinks and never delete or normalise unexpected
    # entries; they are preserved for forensic inspection.
    _entry_count=$(find "${JOB_LOG_DIR}" -mindepth 1 -maxdepth 1 \
        -print | wc -l | tr -d ' ')
    if [ "${_entry_count}" -ne 6 ]; then
        echo "run_gpu_smoke_job: expected exactly six direct entries in job log directory" >&2
        exit 1
    fi

    # Verify each expected file exists as a regular file with
    # mode 0600. Do not modify the files; if any check fails,
    # leave the directory intact and abort.
    for _expected in "${_expected_files[@]}"; do
        _path="${JOB_LOG_DIR}/${_expected}"
        if [ ! -f "${_path}" ]; then
            echo "run_gpu_smoke_job: success-contract artefact missing" >&2
            exit 1
        fi
        _file_mode=$(stat -c '%a' "${_path}" 2>/dev/null \
            || stat -f '%Lp' "${_path}")
        if [ "${_file_mode}" != "600" ]; then
            echo "run_gpu_smoke_job: success-contract artefact mode is not 0600" >&2
            exit 1
        fi
    done
fi

# Return the underlying smoke exit status. Preserve all
# artefacts unconditionally on failure.
exit "${smoke_rc}"
