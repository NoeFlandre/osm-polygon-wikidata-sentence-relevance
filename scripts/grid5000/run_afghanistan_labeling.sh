#!/usr/bin/env bash
# Run/resume Afghanistan labeling inside an allocated CUDA OAR job.

set -euo pipefail
umask 077

if [ "$#" -ne 11 ]; then
    echo "run_afghanistan_labeling: exactly eleven arguments are required" >&2
    exit 2
fi

REPO_ROOT="$1"; readonly REPO_ROOT
INPUT_PARQUET="$2"; readonly INPUT_PARQUET
WORK_DIR="$3"; readonly WORK_DIR
OUTPUT_DIR="$4"; readonly OUTPUT_DIR
MODEL_FILE="$5"; readonly MODEL_FILE
TOKENIZER_DIR="$6"; readonly TOKENIZER_DIR
MODEL_REVISION="$7"; readonly MODEL_REVISION
INPUT_REVISION="$8"; readonly INPUT_REVISION
SOURCE_COMMIT="$9"; readonly SOURCE_COMMIT
DATASET_ID="${10}"; readonly DATASET_ID
BATCH_SIZE="${11}"; readonly BATCH_SIZE

: "${OAR_JOB_ID:?run_afghanistan_labeling requires an OAR allocation}"
case "${OAR_JOB_ID}" in (*[!0-9]*|'') echo "run_afghanistan_labeling: invalid OAR job ID" >&2; exit 2;; esac
command -v nvidia-smi >/dev/null || { echo "run_afghanistan_labeling: CUDA tooling unavailable" >&2; exit 1; }
nvidia-smi -L >/dev/null || { echo "run_afghanistan_labeling: no visible CUDA GPU" >&2; exit 1; }

case "${MODEL_FILE}" in (*Qwen3.6-27B-Q4_K_M.gguf) ;; (*) echo "run_afghanistan_labeling: expected pinned Q4_K_M model file" >&2; exit 2;; esac
test -f "${INPUT_PARQUET}" || { echo "run_afghanistan_labeling: input Parquet missing" >&2; exit 2; }
test -f "${MODEL_FILE}" || { echo "run_afghanistan_labeling: model file missing" >&2; exit 2; }
test -d "${TOKENIZER_DIR}" || { echo "run_afghanistan_labeling: tokenizer directory missing" >&2; exit 2; }

LABEL_CLI="${REPO_ROOT}/.venv/bin/osm-polygon-label-sentences"
test -x "${LABEL_CLI}" || { echo "run_afghanistan_labeling: labeling CLI missing" >&2; exit 2; }
MODEL_SHA256=$(sha256sum "${MODEL_FILE}" | awk '{print $1}')
readonly MODEL_SHA256
PORT=8000; readonly PORT
SERVER_PID=""

cleanup() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

health() {
    local attempts=0
    while [ "${attempts}" -lt 120 ]; do
        if curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            return 1
        fi
        sleep 1
        attempts=$((attempts + 1))
    done
    return 1
}

ENGINE=""
ENGINE_VERSION=""
if command -v vllm >/dev/null; then
    vllm serve "${MODEL_FILE}" \
        --tokenizer "${TOKENIZER_DIR}" \
        --hf-config-path "${TOKENIZER_DIR}" \
        --host 127.0.0.1 --port "${PORT}" \
        --max-model-len 4096 --gpu-memory-utilization 0.92 \
        --enable-prefix-caching \
        >"${WORK_DIR}.vllm.stdout.log" 2>"${WORK_DIR}.vllm.stderr.log" &
    SERVER_PID=$!
    if health; then
        ENGINE=vllm
        ENGINE_VERSION=$(vllm --version | head -1)
    else
        cleanup
        SERVER_PID=""
    fi
fi

if [ -z "${ENGINE}" ]; then
    command -v llama-server >/dev/null || { echo "run_afghanistan_labeling: vLLM canary failed and llama.cpp is unavailable" >&2; exit 1; }
    llama-server --model "${MODEL_FILE}" --host 127.0.0.1 --port "${PORT}" \
        --ctx-size 4096 --n-gpu-layers 999 --parallel 32 \
        >"${WORK_DIR}.llama.stdout.log" 2>"${WORK_DIR}.llama.stderr.log" &
    SERVER_PID=$!
    health || { echo "run_afghanistan_labeling: llama.cpp canary failed" >&2; exit 1; }
    ENGINE=llama.cpp
    ENGINE_VERSION=$(llama-server --version 2>&1 | head -1)
fi
readonly ENGINE ENGINE_VERSION

LABEL_RESULT="${WORK_DIR}.label-result.json"
"${LABEL_CLI}" label \
    --input-parquet "${INPUT_PARQUET}" --work-dir "${WORK_DIR}" \
    --input-dataset-revision "${INPUT_REVISION}" \
    --model-revision "${MODEL_REVISION}" --model-file-sha256 "${MODEL_SHA256}" \
    --source-commit "${SOURCE_COMMIT}" --engine "${ENGINE}" \
    --engine-version "${ENGINE_VERSION}" --batch-size "${BATCH_SIZE}" \
    --endpoint "http://127.0.0.1:${PORT}/v1/chat/completions" \
    >"${LABEL_RESULT}"

if grep -q '"interrupted": true' "${LABEL_RESULT}"; then
    echo "run_afghanistan_labeling: safely interrupted; resume with identical arguments" >&2
    exit 0
fi

"${LABEL_CLI}" finalize \
    --input-parquet "${INPUT_PARQUET}" --work-dir "${WORK_DIR}" \
    --output-dir "${OUTPUT_DIR}" --dataset-id "${DATASET_ID}" \
    --input-dataset-revision "${INPUT_REVISION}" \
    --model-revision "${MODEL_REVISION}" --model-file-sha256 "${MODEL_SHA256}" \
    --source-commit "${SOURCE_COMMIT}" --engine "${ENGINE}" \
    --engine-version "${ENGINE_VERSION}" --batch-size "${BATCH_SIZE}"

"${LABEL_CLI}" publish --output-dir "${OUTPUT_DIR}" --dataset-id "${DATASET_ID}"
