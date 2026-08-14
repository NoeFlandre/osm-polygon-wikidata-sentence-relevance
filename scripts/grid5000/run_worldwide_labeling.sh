#!/usr/bin/env bash
# Run one resumable V2 worldwide binary-label allocation on a CUDA node.

set -euo pipefail
umask 077

if [ "$#" -ne 19 ]; then
    echo "run_worldwide_labeling: exactly nineteen arguments are required" >&2
    exit 2
fi

REPO_ROOT="$1"; INPUT_PARQUET="$2"; WORK_DIR="$3"; OUTPUT_DIR="$4"
MODEL_FILE="$5"; TOKENIZER_DIR="$6"; MODEL_REVISION="$7"; INPUT_REVISION="$8"
SOURCE_COMMIT="$9"; DATASET_ID="${10}"; BATCH_SIZE="${11}"; ROW_LIMIT="${12}"
LLAMA_PARALLEL="${13}"; LLAMA_PER_SLOT_CONTEXT="${14}"
REQUEST_CONCURRENCY="${15}"; SAMPLING_TARGET="${16}"
SAMPLING_SEED="${17}"; SAMPLING_H3_RESOLUTION="${18}"
LABEL_LANE="${19}"
readonly REPO_ROOT INPUT_PARQUET WORK_DIR OUTPUT_DIR MODEL_FILE TOKENIZER_DIR
readonly MODEL_REVISION INPUT_REVISION SOURCE_COMMIT DATASET_ID BATCH_SIZE ROW_LIMIT
readonly LLAMA_PARALLEL LLAMA_PER_SLOT_CONTEXT REQUEST_CONCURRENCY SAMPLING_TARGET
readonly SAMPLING_SEED SAMPLING_H3_RESOLUTION
readonly LABEL_LANE

: "${OAR_JOB_ID:?run_worldwide_labeling requires an OAR allocation}"
if ! [[ "${OAR_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "run_worldwide_labeling: OAR_JOB_ID must be numeric" >&2
    exit 2
fi
if ! [[ "${MODEL_REVISION}" =~ ^[0-9a-f]{40}$ ]] || \
   ! [[ "${INPUT_REVISION}" =~ ^[0-9a-f]{40}$ ]] || \
   ! [[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
   ! [[ "${DATASET_ID}" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]]; then
    echo "run_worldwide_labeling: immutable identity or dataset ID is invalid" >&2
    exit 2
fi
case "$(basename -- "${MODEL_FILE}")" in
    Qwen3.6-27B-Q4_K_M.gguf) ;;
    *) echo "run_worldwide_labeling: expected pinned Q4_K_M model file" >&2; exit 2 ;;
esac
command -v nvidia-smi >/dev/null || { echo "run_worldwide_labeling: CUDA tooling unavailable" >&2; exit 1; }
nvidia-smi -L >/dev/null || { echo "run_worldwide_labeling: no visible CUDA GPU" >&2; exit 1; }
if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${ROW_LIMIT}" =~ ^(0|[1-9][0-9]*)$ ]] || \
   ! [[ "${SAMPLING_TARGET}" =~ ^[1-9][0-9]*$ ]] || \
   [ -z "${SAMPLING_SEED}" ] || [[ "${SAMPLING_SEED}" == *[[:space:]]* ]] || \
   ! [[ "${SAMPLING_H3_RESOLUTION}" =~ ^3$ ]]; then
    echo "run_worldwide_labeling: invalid batch, target, seed, or H3 configuration" >&2
    exit 2
fi
case "${LABEL_LANE}" in
    smoke) [ "${ROW_LIMIT}" -gt 0 ] || { echo "run_worldwide_labeling: smoke lane requires a row limit" >&2; exit 2; } ;;
    production) [ "${ROW_LIMIT}" -eq 0 ] || { echo "run_worldwide_labeling: production lane requires the full target" >&2; exit 2; } ;;
    *) echo "run_worldwide_labeling: label lane is invalid" >&2; exit 2 ;;
esac
case "${LLAMA_PARALLEL}" in 1|2|4|8|16|32) ;; *) echo "run_worldwide_labeling: unsupported parallelism" >&2; exit 2;; esac
if ! [[ "${LLAMA_PER_SLOT_CONTEXT}" =~ ^[1-9][0-9]*$ ]] || \
   [ "${LLAMA_PER_SLOT_CONTEXT}" -lt 4096 ] || \
   ! [[ "${REQUEST_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || \
   [ "${REQUEST_CONCURRENCY}" -gt "${LLAMA_PARALLEL}" ]; then
    echo "run_worldwide_labeling: invalid context or concurrency" >&2
    exit 2
fi
for path in "${INPUT_PARQUET}" "${MODEL_FILE}"; do
    [ -f "${path}" ] && [ ! -L "${path}" ] || {
        echo "run_worldwide_labeling: required file is missing or unsafe" >&2
        exit 2
    }
done
[ -d "${TOKENIZER_DIR}" ] || { echo "run_worldwide_labeling: tokenizer directory is missing" >&2; exit 2; }
RUN_ID="$(basename -- "$(dirname -- "${WORK_DIR}")")"; readonly RUN_ID
if ! [[ "${RUN_ID}" =~ ^[0-9a-f]{20}$ ]]; then
    echo "run_worldwide_labeling: work directory must be nested under a 20-hex run directory" >&2
    exit 2
fi
# The checkpoint mirror has one immutable branch per operator run.  Smoke and
# production remain isolated by their local work/output roots and Trackio run
# names; adding the lane here would violate the mirror's branch contract.
CHECKPOINT_NAMESPACE="checkpoints/${RUN_ID}"; readonly CHECKPOINT_NAMESPACE
LABEL_CLI="${REPO_ROOT}/.venv/bin/osm-polygon-label-sentences"; readonly LABEL_CLI
[ -x "${LABEL_CLI}" ] || { echo "run_worldwide_labeling: labeling CLI is missing" >&2; exit 2; }
MODEL_SHA256="$(sha256sum "${MODEL_FILE}" | awk '{print $1}')"; readonly MODEL_SHA256
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
        if curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then return 0; fi
        if [ -n "${SERVER_PID}" ] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then return 1; fi
        sleep 1; attempts=$((attempts + 1))
    done
    return 1
}
RUN_ROOT="$(cd "${REPO_ROOT}/.." && pwd -P)"; readonly RUN_ROOT
LLAMA_SERVER_DIR="${RUN_ROOT}/llama-server-bin"; readonly LLAMA_SERVER_DIR
[ ! -L "${LLAMA_SERVER_DIR}" ] || {
    echo "run_worldwide_labeling: llama-server directory must not be a symlink" >&2
    exit 2
}
[ -x "${LLAMA_SERVER_DIR}/llama-server" ] || {
    echo "run_worldwide_labeling: staged llama-server is unavailable" >&2
    exit 1
}
export PATH="${LLAMA_SERVER_DIR}:${PATH}"
export LD_LIBRARY_PATH="${LLAMA_SERVER_DIR}:${LD_LIBRARY_PATH:-}"
command -v llama-server >/dev/null || { echo "run_worldwide_labeling: llama-server is unavailable" >&2; exit 1; }
LLAMA_TOTAL_CONTEXT=$((LLAMA_PARALLEL * LLAMA_PER_SLOT_CONTEXT)); readonly LLAMA_TOTAL_CONTEXT
# The persisted label identity describes the logical request contract.  Keep
# it unchanged while giving llama.cpp enough per-slot admission capacity for
# unusually long, otherwise-valid prompts observed in the worldwide input.
SERVER_PER_SLOT_CONTEXT="${LLAMA_PER_SLOT_CONTEXT}"
if [ "${SERVER_PER_SLOT_CONTEXT}" -lt 12288 ]; then
    SERVER_PER_SLOT_CONTEXT=12288
fi
readonly SERVER_PER_SLOT_CONTEXT
SERVER_TOTAL_CONTEXT=$((LLAMA_PARALLEL * SERVER_PER_SLOT_CONTEXT)); readonly SERVER_TOTAL_CONTEXT
llama-server --model "${MODEL_FILE}" --alias "ggml-org/Qwen3.6-27B-GGUF" \
    --host 127.0.0.1 --port "${PORT}" --ctx-size "${SERVER_TOTAL_CONTEXT}" \
    --parallel "${LLAMA_PARALLEL}" --n-gpu-layers 999 \
    >"${WORK_DIR}.llama.stdout.log" 2>"${WORK_DIR}.llama.stderr.log" &
SERVER_PID=$!
health || { echo "run_worldwide_labeling: llama-server failed health check" >&2; exit 1; }
"${LABEL_CLI}" probe --input-parquet "${INPUT_PARQUET}" --engine llama.cpp \
    --endpoint "http://127.0.0.1:${PORT}/v1/chat/completions" --sample-size 4 \
    --llama-parallel "${LLAMA_PARALLEL}" --llama-per-slot-context "${LLAMA_PER_SLOT_CONTEXT}" \
    --llama-total-context "${LLAMA_TOTAL_CONTEXT}" --request-concurrency "${REQUEST_CONCURRENCY}" \
    --release-lane v2-worldwide >"${WORK_DIR}.llama.probe.json" 2>"${WORK_DIR}.llama.probe.stderr.log"
ENGINE_VERSION="$(llama-server --version 2>&1 | sed -n '1p')"; readonly ENGINE_VERSION
LABEL_RESULT="${WORK_DIR}.label-result.json"
"${LABEL_CLI}" label --input-parquet "${INPUT_PARQUET}" --work-dir "${WORK_DIR}" \
    --input-dataset-revision "${INPUT_REVISION}" --model-revision "${MODEL_REVISION}" \
    --model-file-sha256 "${MODEL_SHA256}" --source-commit "${SOURCE_COMMIT}" \
    --engine llama.cpp --engine-version "${ENGINE_VERSION}" --batch-size "${BATCH_SIZE}" \
    --concurrency "${REQUEST_CONCURRENCY}" --llama-parallel "${LLAMA_PARALLEL}" \
    --llama-per-slot-context "${LLAMA_PER_SLOT_CONTEXT}" --llama-total-context "${LLAMA_TOTAL_CONTEXT}" \
    --request-concurrency "${REQUEST_CONCURRENCY}" --row-limit "${ROW_LIMIT}" \
    --sampling-target "${SAMPLING_TARGET}" --sampling-seed "${SAMPLING_SEED}" \
    --h3-resolution "${SAMPLING_H3_RESOLUTION}" --checkpoint-dataset-id "${DATASET_ID}" \
    --checkpoint-namespace "${CHECKPOINT_NAMESPACE}" --checkpoint-drain-seconds 30 \
    --release-lane v2-worldwide --trackio-project worldwide-stratified-labeling \
    --trackio-run-name "run-${RUN_ID}-${LABEL_LANE}" --trackio-space-id NoeFlandre/worldwide-stratified-labeling-trackio \
    --endpoint "http://127.0.0.1:${PORT}/v1/chat/completions" >"${LABEL_RESULT}"
if grep -q '"interrupted": true' "${LABEL_RESULT}"; then
    echo "run_worldwide_labeling: safely interrupted; resume with identical identity" >&2
    exit 0
fi
"${LABEL_CLI}" finalize --input-parquet "${INPUT_PARQUET}" --work-dir "${WORK_DIR}" \
    --output-dir "${OUTPUT_DIR}" --dataset-id "${DATASET_ID}" \
    --input-dataset-revision "${INPUT_REVISION}" --model-revision "${MODEL_REVISION}" \
    --model-file-sha256 "${MODEL_SHA256}" --source-commit "${SOURCE_COMMIT}" \
    --engine llama.cpp --engine-version "${ENGINE_VERSION}" --batch-size "${BATCH_SIZE}" \
    --row-limit "${ROW_LIMIT}" --sampling-target "${SAMPLING_TARGET}" \
    --sampling-seed "${SAMPLING_SEED}" --h3-resolution "${SAMPLING_H3_RESOLUTION}" \
    --llama-parallel "${LLAMA_PARALLEL}" --llama-per-slot-context "${LLAMA_PER_SLOT_CONTEXT}" \
    --llama-total-context "${LLAMA_TOTAL_CONTEXT}" --request-concurrency "${REQUEST_CONCURRENCY}" \
    --release-lane v2-worldwide
if [ "${LABEL_LANE}" = "production" ]; then
    "${LABEL_CLI}" publish --output-dir "${OUTPUT_DIR}" --dataset-id "${DATASET_ID}"
else
    echo "V2 canary complete; publication intentionally skipped: ${OUTPUT_DIR}"
fi
