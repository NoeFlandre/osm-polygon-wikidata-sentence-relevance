#!/usr/bin/env bash
# Grid'5000 GPU smoke payload (Phase 9B).
#
# EXPLICITLY INVOKED job payload. It runs *inside* an allocated
# OAR CUDA allocation (after the scheduler has set OAR_JOB_ID and
# CUDA_VISIBLE_DEVICES). It is NOT a submission tool.
#
# Hard safety contract (Phase 9B + safety amendment):
#   * refuses to run without OAR_JOB_ID
#   * refuses to run without CUDA_VISIBLE_DEVICES
#   * requires caller-provided REPO_ROOT, HF_HOME, SMOKE_LOG_DIR
#   * requires the LOCKED interpreter ${REPO_ROOT}/.venv/bin/python
#     (installed editable from exact commit B)
#   * never exports REPO_ROOT or REPO_ROOT_HINT to inner Python
#   * never mutates sys.path for production inference
#   * never uses bare `python` or `python3`
#   * explicit device="cuda" ONLY
#   * umask 077 for restrictive log permissions
#   * mktemp inside SMOKE_LOG_DIR for the preflight temp file
#   * refuses to overwrite any pre-existing final report
#   * uses an atomic no-clobber install helper that calls os.link
#   * each phase creates its temp file only after the previous
#     install has succeeded; every existing temp file is covered
#     by the active trap
#   * never prints supplied REPO_ROOT / HF_HOME / SMOKE_LOG_DIR
#     /PROJECT_PYTHON / artifact paths in error or status output

set -euo pipefail

# Restrictive creation permissions for log artifacts.
umask 077

# --- Required environment -------------------------------------------

: "${OAR_JOB_ID:?OAR_JOB_ID is required (run inside an OAR allocation)}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required (request one GPU)}"

REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required (caller-provided persistent path)}"
HF_HOME="${HF_HOME:?HF_HOME is required (caller-provided persistent path)}"
SMOKE_LOG_DIR="${SMOKE_LOG_DIR:?SMOKE_LOG_DIR is required (caller-provided persistent path)}"

# --- Preflight first --- (the smoke-result temp file is created later, after preflight)

# Preflight field contract -- enforced by
#   scripts/grid5000/_validate_artifact.py::validate_preflight:
#   * oar_job_id (non-blank string)
#   * hostname (non-blank string)
#   * torch_version (non-blank string)
#   * torch_cuda_runtime_version (non-blank string)
#   * device_0_name (non-blank string)
#   * visible_cuda_device_count == 1
#
# Smoke-result field contract -- enforced by
#   scripts/grid5000/_validate_artifact.py::validate_smoke_result:
#   * resolved_device == "cuda"
#   * model_name == "sat-3l-sm"
#   * input_count == 3
#   * sentence_counts: three positive integers
#   * elapsed_seconds: finite, non-negative number
#   * torch_version (non-blank string)
#   * torch_cuda_runtime_version (non-blank string)
#   * cuda_device_name (non-blank string)
#
# Atomic install -- enforced by the same module's CLI:
#   validate_preflight / validate_smoke_result / install_artifact
# (uses os.link; no mv, no path leakage, no path values in errors).

# --- Locked interpreter --------------------------------------------

PROJECT_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [ ! -f "${PROJECT_PYTHON}" ]; then
    echo "run_gpu_smoke: locked project interpreter is missing" >&2
    echo "  Create it beforehand with: uv sync --locked --extra segmentation" >&2
    exit 1
fi
if [ ! -x "${PROJECT_PYTHON}" ]; then
    echo "run_gpu_smoke: locked project interpreter is not executable" >&2
    exit 1
fi
if [ ! -f "${REPO_ROOT}/scripts/grid5000/_validate_artifact.py" ]; then
    echo "run_gpu_smoke: _validate_artifact.py is missing in scripts/grid5000/" >&2
    exit 1
fi
if [ ! -r "${REPO_ROOT}/scripts/grid5000/_validate_artifact.py" ]; then
    echo "run_gpu_smoke: _validate_artifact.py is not readable" >&2
    exit 1
fi
if [ ! -f "${REPO_ROOT}/scripts/grid5000/gpu_preflight.py" ]; then
    echo "run_gpu_smoke: gpu_preflight.py is missing in scripts/grid5000/" >&2
    exit 1
fi

# Absolute path to the private validator; the smoke payload never
# depends on the working directory matching the repository root.
ARTIFACT_VALIDATOR="${REPO_ROOT}/scripts/grid5000/_validate_artifact.py"

# --- Runtime path validation (fail before any model construction) ---

if [ ! -d "${REPO_ROOT}" ]; then
    echo "run_gpu_smoke: REPO_ROOT is not a directory" >&2
    exit 1
fi
if [ ! -r "${HF_HOME}" ] || [ ! -d "${HF_HOME}" ]; then
    echo "run_gpu_smoke: HF_HOME is not a readable directory" >&2
    exit 1
fi
if [ ! -d "${SMOKE_LOG_DIR}" ] || [ ! -w "${SMOKE_LOG_DIR}" ]; then
    echo "run_gpu_smoke: SMOKE_LOG_DIR is not writable" >&2
    exit 1
fi

# Reject node-local ephemeral locations.
case "${HF_HOME}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_smoke: HF_HOME points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac
case "${SMOKE_LOG_DIR}" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
        echo "run_gpu_smoke: SMOKE_LOG_DIR points to forbidden ephemeral storage" >&2
        exit 1
        ;;
esac

# Refuse to overwrite any pre-existing final report.
if [ -e "${SMOKE_LOG_DIR}/gpu_preflight.json" ]; then
    echo "run_gpu_smoke: gpu_preflight.json already exists in SMOKE_LOG_DIR" >&2
    exit 1
fi
if [ -e "${SMOKE_LOG_DIR}/smoke_result.json" ]; then
    echo "run_gpu_smoke: smoke_result.json already exists in SMOKE_LOG_DIR" >&2
    exit 1
fi
if [ -e "${SMOKE_LOG_DIR}/run_metadata.json" ]; then
    echo "run_gpu_smoke: run_metadata.json already exists in SMOKE_LOG_DIR" >&2
    exit 1
fi

# --- Offline / determinism (only HF_HOME is exported; no REPO_ROOT) -

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Final paths are local variables; never exported. The shell passes
# them only as explicit argv to the validator CLI.
PREFLIGHT_JSON="${SMOKE_LOG_DIR}/gpu_preflight.json"
SMOKE_RESULT_JSON="${SMOKE_LOG_DIR}/smoke_result.json"

# === Phase 1: preflight =============================================
# Created temp file lives only during this phase.
PREFLIGHT_TMP="$(mktemp "${SMOKE_LOG_DIR}/.gpu_preflight.XXXXXX.json")"

cleanup_preflight() {
    # Only the preflight temp file is in play here.
    if [ -f "${PREFLIGHT_TMP}" ]; then
        rm -f "${PREFLIGHT_TMP}"
    fi
}
trap cleanup_preflight EXIT INT TERM

echo "[smoke] running gpu_preflight" >&2

set +e
"${PROJECT_PYTHON}" "${REPO_ROOT}/scripts/grid5000/gpu_preflight.py" 1>"${PREFLIGHT_TMP}"
preflight_rc=$?
set -e

if [ "${preflight_rc}" -ne 0 ]; then
    echo "run_gpu_smoke: gpu_preflight failed" >&2
    exit 1
fi

# Validate the preflight JSON contract before any install.
set +e
"${PROJECT_PYTHON}" "${ARTIFACT_VALIDATOR}" preflight \
    "${PREFLIGHT_TMP}"
validator_rc=$?
set -e

if [ "${validator_rc}" -ne 0 ]; then
    echo "run_gpu_smoke: gpu_preflight.json contract validation failed" >&2
    exit 1
fi

# Atomic no-clobber install of the preflight result.
set +e
"${PROJECT_PYTHON}" "${ARTIFACT_VALIDATOR}" install \
    "${PREFLIGHT_TMP}" "${PREFLIGHT_JSON}"
install_rc=$?
set -e

if [ "${install_rc}" -ne 0 ]; then
    echo "run_gpu_smoke: gpu_preflight.json install failed" >&2
    exit 1
fi

# Disarm the preflight trap before declaring the smoke-result temp.
trap - EXIT INT TERM

# === Phase 1.5: write deterministic run metadata ====================
# The metadata records the exact source commit and the immutable
# model/tokenizer revisions so the smoke can be re-derived later.
RUN_METADATA_PATH="${SMOKE_LOG_DIR}/run_metadata.json"
RUN_METADATA_HELPER="${REPO_ROOT}/scripts/grid5000/_run_metadata.py"

# Refuse to overwrite any pre-existing run-metadata artifact.
if [ -e "${RUN_METADATA_PATH}" ]; then
    echo "run_gpu_smoke: run_metadata.json already exists in SMOKE_LOG_DIR" >&2
    exit 1
fi
if [ ! -r "${RUN_METADATA_HELPER}" ]; then
    echo "run_gpu_smoke: _run_metadata.py is missing" >&2
    exit 1
fi

# Discover the exact source commit and hostname from the
# scheduler-owned compute node. We invoke the VCS tool (read-only)
# so the contract is "same checkout, same content" regardless of
# worktree / packed-refs shape.
if ! command -v git >/dev/null 2>&1; then
    echo "run_gpu_smoke: git binary is required (command -v git failed)" >&2
    exit 1
fi

# EXPECTED_SOURCE_COMMIT must be passed by the operator; no
# default, ever. The payload is modified *after* Phase 9B was
# committed, so a hardcoded default would self-reference.
: "${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required (40 lowercase hex chars)}"
if [[ ! "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "run_gpu_smoke: EXPECTED_SOURCE_COMMIT is not exactly 40 lowercase hex characters" >&2
    exit 1
fi

SOURCE_COMMIT="$(
    set +e
    git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null
    git_rev_parse_rc=$?
    set -e
    if [ "${git_rev_parse_rc}" -ne 0 ]; then
        echo "run_gpu_smoke: git rev-parse HEAD failed" >&2
        exit 1
    fi
)"
if [ -z "${SOURCE_COMMIT}" ]; then
    echo "run_gpu_smoke: git rev-parse HEAD returned empty" >&2
    exit 1
fi

# Refuse to run from a dirty checkout: any uncommitted or staged
# change would mean the metadata SHA disagrees with the runtime
# files. The git stderr is silenced so REPO_ROOT (which git
# would otherwise include in error messages) never appears in
# the captured stdout/stderr.
DIRTY_OUTPUT="$(
    set +e
    git -C "${REPO_ROOT}" status --porcelain 2>/dev/null
    git_status_rc=$?
    set -e
    if [ "${git_status_rc}" -ne 0 ]; then
        echo "run_gpu_smoke: git status --porcelain failed" >&2
        exit 1
    fi
)"
if [ -n "${DIRTY_OUTPUT}" ]; then
    echo "run_gpu_smoke: working tree is dirty (git status --porcelain is non-empty)" >&2
    exit 1
fi

if [ "${SOURCE_COMMIT}" != "${EXPECTED_SOURCE_COMMIT}" ]; then
    echo "run_gpu_smoke: source commit ${SOURCE_COMMIT} does not match EXPECTED_SOURCE_COMMIT" >&2
    exit 1
fi

HOSTNAME_SHORT="$(hostname -s 2>/dev/null || echo unknown)"

set +e
# Run-metadata keys (exact schema enforced by the helper):
#   source_commit  model_name  model_revision
#   tokenizer_name  tokenizer_revision  oar_job_id  hostname
"${PROJECT_PYTHON}" "${RUN_METADATA_HELPER}" "${RUN_METADATA_PATH}" \
    "${SOURCE_COMMIT}" \
    "sat-3l-sm" \
    "137da054051ad9f1eac42025f758db4ac9f22535" \
    "facebookAI/xlm-roberta-base" \
    "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089" \
    "${OAR_JOB_ID}" \
    "${HOSTNAME_SHORT}"
metadata_rc=$?
set -e

if [ "${metadata_rc}" -ne 0 ]; then
    echo "run_gpu_smoke: run_metadata.json write failed" >&2
    exit 1
fi

# Refuse to proceed if the metadata file was not produced (defensive).
if [ ! -f "${RUN_METADATA_PATH}" ]; then
    echo "run_gpu_smoke: run_metadata.json was not created" >&2
    exit 1
fi

# === Phase 2: real SaT inference on CUDA ============================

# Preflight has been validated and installed. Run metadata is in
# place. No temp file is in play. Create the smoke-result temp file;
# the active trap now covers exactly this temp file.
SMOKE_RESULT_TMP="$(mktemp "${SMOKE_LOG_DIR}/.smoke_result.XXXXXX.json")"

cleanup_result() {
    # Only the smoke-result temp file is in play here.
    if [ -f "${SMOKE_RESULT_TMP}" ]; then
        rm -f "${SMOKE_RESULT_TMP}"
    fi
}
trap cleanup_result EXIT INT TERM

MODEL_NAME="sat-3l-sm"

echo "[smoke] constructing SaTSentenceSegmenter(device=cuda)" >&2

set +e
"${PROJECT_PYTHON}" - "${MODEL_NAME}" "${SMOKE_RESULT_TMP}" <<'PYEOF'
import json
import os
import sys
import time

# The locked interpreter is an editable install of
# osm-polygon_sentence_relevance from exact commit B; we import
# directly without touching sys.path. No REPO_ROOT, no REPO_ROOT_HINT.
import torch  # noqa: E402  -- expected import after the shebang-line
from osm_polygon_sentence_relevance.sentences.sat import SaTSentenceSegmenter

model_name = sys.argv[1]
result_path = sys.argv[2]

# Assert CUDA is still available immediately before constructing SaT.
if torch.cuda.is_available() is False:
    sys.exit("cuda identity lost: torch.cuda.is_available() is False")
if torch.cuda.device_count() != 1:
    sys.exit("cuda identity lost: device_count is not exactly 1")
cuda_device_name = torch.cuda.get_device_name(0)
torch_version = torch.__version__
torch_cuda_runtime_version = torch.version.cuda

texts = [
    "This is a short English sentence. Another English clause here.",
    "Ceci est une phrase courte en francais. Une autre proposition ici.",
    "\u0391\u03c5\u03c4\u03b7 \u03b5\u03b9\u03bd\u03b1\u03b9 "
    "\u03bc\u03b9\u03b1 \u03c3\u03c5\u03bd\u03c4\u03bf\u03bc\u03b7 "
    "\u03c0\u03c1\u03bf\u03c4\u03b1\u03c3\u03b7 \u03c3\u03c4\u03b1 "
    "\u03b5\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03b1. "
    "\u039c\u03b9\u03b1 \u03b1\u03bb\u03bb\u03b7 \u03b5\u03b4\u03c9.",
]
languages = ["en", "fr", "el"]

segmenter = SaTSentenceSegmenter(
    model_name=model_name,
    device="cuda",
)

start = time.time()
out = segmenter.split_batch(texts, languages)
elapsed = time.time() - start

assert segmenter.resolved_device == "cuda", (
    f"resolved_device must be cuda, got {segmenter.resolved_device!r}"
)
assert len(out) == len(texts), (
    f"output group count {len(out)} != input count {len(texts)}"
)
for group in out:
    assert any(s.strip() for s in group), "output group has no non-blank sentence"

result = {
    "resolved_device": segmenter.resolved_device,
    "model_name": model_name,
    "input_count": len(texts),
    "sentence_counts": [len(group) for group in out],
    "elapsed_seconds": round(elapsed, 3),
    "torch_version": torch_version,
    "torch_cuda_runtime_version": torch_cuda_runtime_version,
    "cuda_device_name": cuda_device_name,
}
# Final sorted-key JSON so the schema is deterministic.
with open(result_path, "w", encoding="utf-8") as fh:
    json.dump(result, fh, sort_keys=True)
# Stable completion label -- never echoes the supplied result path.
sys.stderr.write("[smoke] result generated\n")
PYEOF
inference_rc=$?
set -e

if [ "${inference_rc}" -ne 0 ]; then
    echo "run_gpu_smoke: inference failed" >&2
    exit 1
fi

# Validate the smoke-result JSON contract.
set +e
"${PROJECT_PYTHON}" "${ARTIFACT_VALIDATOR}" smoke-result \
    "${SMOKE_RESULT_TMP}"
validator_rc=$?
set -e

if [ "${validator_rc}" -ne 0 ]; then
    echo "run_gpu_smoke: smoke_result.json contract validation failed" >&2
    exit 1
fi

# Atomic no-clobber install of the smoke result.
set +e
"${PROJECT_PYTHON}" "${ARTIFACT_VALIDATOR}" install \
    "${SMOKE_RESULT_TMP}" "${SMOKE_RESULT_JSON}"
install_rc=$?
set -e

if [ "${install_rc}" -ne 0 ]; then
    echo "run_gpu_smoke: smoke_result.json install failed" >&2
    exit 1
fi

trap - EXIT INT TERM

echo "[smoke] done" >&2
