#!/usr/bin/env bash
# Grid'5000 full resumable build payload (Phase 9L-B).
#
# EXPLICITLY INVOKED job payload. It runs *inside* an allocated OAR
# CUDA allocation (after the scheduler has set OAR_JOB_ID). It is
# NOT a submission tool.
#
# Phase 9H: ``CUDA_VISIBLE_DEVICES`` is informational only.
# Grid'5000 does not guarantee ``CUDA_VISIBLE_DEVICES`` on a given
# allocation; the payload must not require it. If the scheduler
# set it, the harness inherits it unchanged. The authoritative
# runtime proof of GPU scoping is
# ``torch.cuda.device_count() == 1`` inside ``gpu_preflight.py``,
# which runs as the first phase below.
#
# Hard safety contract (Phase 9L-B):
#   * refuses to run without OAR_JOB_ID
#   * never touches CUDA_VISIBLE_DEVICES (informational only;
#     not required, not assigned, not defaulted, not exported)
#   * requires caller-provided REPO_ROOT, HF_HOME, BUILD_LOG_DIR,
#     INPUT_ROOT, WORK_DIR, OUTPUT_DIR, EXPECTED_SOURCE_COMMIT,
#     INPUT_REVISION
#   * requires the LOCKED interpreter
#     ${REPO_ROOT}/.venv/bin/python
#   * invokes the LOCKED CLI
#     ${REPO_ROOT}/.venv/bin/osm-polygon-sentence-relevance exactly
#     once, with --device cuda, --input-root, --input-source-dataset-id,
#     --input-dataset-revision, --output-dir, --work-dir,
#     --source-commit, and the pinned local input snapshot
#   * never exports REPO_ROOT or REPO_ROOT_HINT to inner Python
#   * never mutates sys.path for production inference
#   * never uses bare python interpreters
#   * explicit device="cuda" ONLY (no auto, cpu, or mps)
#   * no Hub publishing flags -- the launcher is for build only
#   * no publishing or re-publishing flags -- OUTPUT_DIR is created
#     fresh by this payload; subsequent invocations use a new
#     OUTPUT_DIR and the previous OUTPUT_DIR is never touched
#   * umask 077 for restrictive log permissions
#   * mktemp inside BUILD_LOG_DIR for preflight/metadata temp files
#   * refuses to overwrite any pre-existing final report
#   * uses the same atomic no-clobber install helper as the smoke
#   * leaves WORK_DIR untouched on failure (resume contract: a
#     later invocation can reuse the same WORK_DIR and the
#     pipeline will resume from the last valid checkpoint)
#   * never prints supplied REPO_ROOT / HF_HOME / BUILD_LOG_DIR /
#     INPUT_ROOT / WORK_DIR / OUTPUT_DIR / PROJECT_PYTHON /
#     artifact paths in error or status output
#
# Resume semantics (Phase 9L-A + 9L-B):
#   The pipeline's --work-dir contract publishes one checkpoint per
#   shard. A failed or walltime-terminated job leaves WORK_DIR
#   untouched: every checkpoint published before the failure
#   remains valid. A subsequent invocation with the same WORK_DIR
#   resumes from the last valid checkpoint; the launcher never
#   deletes or quarantines valid checkpoints. The payload exits
#   with the CLI's real exit code (no masking, no retry).

set -euo pipefail

umask 077

# --- Required environment -------------------------------------------

: "${OAR_JOB_ID:?OAR_JOB_ID is required (run inside an OAR allocation)}"

# CUDA_VISIBLE_DEVICES is intentionally NOT required, NOT
# assigned, NOT defaulted, NOT exported.

REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required (caller-provided persistent path)}"
HF_HOME="${HF_HOME:?HF_HOME is required (caller-provided persistent path)}"
BUILD_LOG_DIR="${BUILD_LOG_DIR:?BUILD_LOG_DIR is required (caller-provided persistent path)}"
INPUT_ROOT="${INPUT_ROOT:?INPUT_ROOT is required (caller-provided staged snapshot path)}"
WORK_DIR="${WORK_DIR:?WORK_DIR is required (caller-provided persistent resume path)}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required (caller-provided fresh output path)}"

: "${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required (40 lowercase hex chars)}"
if [[ ! "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "run_gpu_build: EXPECTED_SOURCE_COMMIT is not exactly 40 lowercase hex characters" >&2
    exit 1
fi
: "${INPUT_REVISION:?INPUT_REVISION is required (40 lowercase hex chars; main is rejected)}"
if [[ ! "${INPUT_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "run_gpu_build: INPUT_REVISION must be 40 lowercase hex characters (main is rejected)" >&2
    exit 1
fi

# --- Locked interpreter --------------------------------------------

PROJECT_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [ ! -f "${PROJECT_PYTHON}" ]; then
    echo "run_gpu_build: locked project interpreter is missing" >&2
    exit 1
fi
if [ ! -x "${PROJECT_PYTHON}" ]; then
    echo "run_gpu_build: locked project interpreter is not executable" >&2
    exit 1
fi
if [ ! -f "${REPO_ROOT}/scripts/grid5000/gpu_preflight.py" ]; then
    echo "run_gpu_build: gpu_preflight.py is missing in scripts/grid5000/" >&2
    exit 1
fi
if [ ! -f "${REPO_ROOT}/scripts/grid5000/_validate_artifact.py" ]; then
    echo "run_gpu_build: _validate_artifact.py is missing in scripts/grid5000/" >&2
    exit 1
fi

ARTIFACT_VALIDATOR="${REPO_ROOT}/scripts/grid5000/_validate_artifact.py"
RUN_METADATA_HELPER="${REPO_ROOT}/scripts/grid5000/_run_metadata.py"

# --- Locked CLI binary ---------------------------------------------

PROJECT_CLI="${REPO_ROOT}/.venv/bin/osm-polygon-sentence-relevance"
if [ ! -f "${PROJECT_CLI}" ]; then
    echo "run_gpu_build: locked CLI entry point is missing" >&2
    exit 1
fi
if [ ! -x "${PROJECT_CLI}" ]; then
    echo "run_gpu_build: locked CLI entry point is not executable" >&2
    exit 1
fi

# --- Runtime path validation (fail before any model construction) ---

if [ ! -d "${REPO_ROOT}" ]; then
    echo "run_gpu_build: REPO_ROOT is not a directory" >&2
    exit 1
fi
if [ ! -r "${HF_HOME}" ] || [ ! -d "${HF_HOME}" ]; then
    echo "run_gpu_build: HF_HOME is not a readable directory" >&2
    exit 1
fi
if [ ! -d "${BUILD_LOG_DIR}" ] || [ ! -w "${BUILD_LOG_DIR}" ]; then
    echo "run_gpu_build: BUILD_LOG_DIR is not a writable directory" >&2
    exit 1
fi
if [ ! -d "${INPUT_ROOT}" ] || [ ! -r "${INPUT_ROOT}" ]; then
    echo "run_gpu_build: INPUT_ROOT is not a readable directory (stage the snapshot first)" >&2
    exit 1
fi

# Refuse ephemeral node-local storage.
case "${HF_HOME}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_build: HF_HOME points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${BUILD_LOG_DIR}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_build: BUILD_LOG_DIR points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${INPUT_ROOT}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_build: INPUT_ROOT points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${WORK_DIR}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_build: WORK_DIR points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${OUTPUT_DIR}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_build: OUTPUT_DIR points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac

# WORK_DIR may pre-exist for resume; create it if missing.
# OUTPUT_DIR must NOT pre-exist; the payload creates it fresh.
if [ ! -d "${WORK_DIR}" ]; then
    mkdir -p "${WORK_DIR}"
fi
if [ -e "${OUTPUT_DIR}" ]; then
    echo "run_gpu_build: OUTPUT_DIR already exists (fresh build only)" >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"

# Refuse to overwrite any pre-existing final report.
if [ -e "${BUILD_LOG_DIR}/gpu_preflight.json" ]; then
    echo "run_gpu_build: gpu_preflight.json already exists in BUILD_LOG_DIR" >&2
    exit 1
fi
if [ -e "${BUILD_LOG_DIR}/run_metadata.json" ]; then
    echo "run_gpu_build: run_metadata.json already exists in BUILD_LOG_DIR" >&2
    exit 1
fi
if [ -e "${BUILD_LOG_DIR}/build_result.json" ]; then
    echo "run_gpu_build: build_result.json already exists in BUILD_LOG_DIR" >&2
    exit 1
fi

# --- Offline / determinism -----------------------------------------

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PREFLIGHT_JSON="${BUILD_LOG_DIR}/gpu_preflight.json"
RUN_METADATA_PATH="${BUILD_LOG_DIR}/run_metadata.json"

# === Phase 1: preflight =============================================
PREFLIGHT_TMP="$(mktemp "${BUILD_LOG_DIR}/.gpu_preflight.XXXXXX.json")"

cleanup_preflight() {
    if [ -f "${PREFLIGHT_TMP}" ]; then
        rm -f "${PREFLIGHT_TMP}"
    fi
}
trap cleanup_preflight EXIT INT TERM

echo "[build] running gpu_preflight" >&2

set +e
"${PROJECT_PYTHON}" "${REPO_ROOT}/scripts/grid5000/gpu_preflight.py" 1>"${PREFLIGHT_TMP}"
preflight_rc=$?
set -e

if [ "${preflight_rc}" -ne 0 ]; then
    echo "run_gpu_build: gpu_preflight failed" >&2
    exit 1
fi

set +e
"${PROJECT_PYTHON}" "${ARTIFACT_VALIDATOR}" preflight "${PREFLIGHT_TMP}"
validator_rc=$?
set -e

if [ "${validator_rc}" -ne 0 ]; then
    echo "run_gpu_build: gpu_preflight.json contract validation failed" >&2
    exit 1
fi

set +e
"${PROJECT_PYTHON}" "${ARTIFACT_VALIDATOR}" install "${PREFLIGHT_TMP}" "${PREFLIGHT_JSON}"
install_rc=$?
set -e

if [ "${install_rc}" -ne 0 ]; then
    echo "run_gpu_build: gpu_preflight.json install failed" >&2
    exit 1
fi

trap - EXIT INT TERM

# === Phase 1.5: write deterministic run metadata ====================

if [ ! -r "${RUN_METADATA_HELPER}" ]; then
    echo "run_gpu_build: _run_metadata.py is missing" >&2
    exit 1
fi

HOSTNAME_SHORT="$(hostname -s 2>/dev/null || echo unknown)"

set +e
"${PROJECT_PYTHON}" "${RUN_METADATA_HELPER}" "${RUN_METADATA_PATH}" \
    "${EXPECTED_SOURCE_COMMIT}" \
    "sat-3l-sm" \
    "137da054051ad9f1eac42025f758db4ac9f22535" \
    "facebookAI/xlm-roberta-base" \
    "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089" \
    "${OAR_JOB_ID}" \
    "${HOSTNAME_SHORT}"
metadata_rc=$?
set -e

if [ "${metadata_rc}" -ne 0 ]; then
    echo "run_gpu_build: run_metadata.json write failed" >&2
    exit 1
fi

if [ ! -f "${RUN_METADATA_PATH}" ]; then
    echo "run_gpu_build: run_metadata.json was not created" >&2
    exit 1
fi

# === Phase 2: full resumable build via the existing public CLI =====
#
# The pipeline is invoked exactly once via the locked CLI entry
# point at ${REPO_ROOT}/.venv/bin/osm-polygon-sentence-relevance.
#
# Local-input mode: --input-root points at the staged immutable
# snapshot under INPUT_ROOT; --input-source-dataset-id records the
# upstream Hub dataset ID for provenance without triggering any
# network fetch; --input-dataset-revision pins the immutable
# 40-hex Hub revision the snapshot was resolved from.
#
# Required CLI flags:
#   --input-root "${INPUT_ROOT}"
#   --input-source-dataset-id NoeFlandre/osm-polygon-wikidata-only
#   --input-dataset-revision "${INPUT_REVISION}" (immutable 40-hex)
#   --output-dir "${OUTPUT_DIR}"                   (fresh)
#   --pipeline-version v0.1.0                      (build version label)
#   --device cuda                                  (explicit only;
#       no auto, cpu, or mps fallback)
#   --work-dir "${WORK_DIR}"                       (resumable
#       checkpoint contract from Phase 9L-A)
#   --source-commit "${EXPECTED_SOURCE_COMMIT}"    (40 lowercase hex;
#       bound to every shard checkpoint and heartbeat)
#
# No publishing flags. No CPU/MPS/auto fallback. No destructive
# re-rendering of existing files (the OUTPUT_DIR is created fresh
# by this payload; subsequent invocations use a new OUTPUT_DIR).
#
# WORK_DIR persistence: a non-zero exit (including walltime
# termination) leaves WORK_DIR untouched. The payload does not
# quarantine, delete, or modify existing checkpoints. A subsequent
# invocation with the same WORK_DIR resumes from the last valid
# checkpoint.

PIPELINE_VERSION="0.1.0"
INPUT_DATASET_ID="NoeFlandre/osm-polygon-wikidata-only"

set +e
"${PROJECT_CLI}" \
    --input-root "${INPUT_ROOT}" \
    --input-source-dataset-id "${INPUT_DATASET_ID}" \
    --input-dataset-revision "${INPUT_REVISION}" \
    --output-dir "${OUTPUT_DIR}" \
    --pipeline-version "${PIPELINE_VERSION}" \
    --device cuda \
    --batch-size 128 \
    --sat-model "sat-3l-sm" \
    --work-dir "${WORK_DIR}" \
    --source-commit "${EXPECTED_SOURCE_COMMIT}"
build_rc=$?
set -e

# Capture the real exit status. No masking, no retry, no fallback.
# WORK_DIR is preserved unconditionally so the resume contract
# holds.
echo "[build] done (rc=${build_rc})" >&2
exit "${build_rc}"
