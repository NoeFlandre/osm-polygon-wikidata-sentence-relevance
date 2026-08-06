#!/usr/bin/env bash
# Run/resume one deterministic label sample inside an allocated CUDA OAR job.
#
# The historical filename is retained for V1 compatibility. The payload is
# lane-neutral: a zero sampling target is the preserved Afghanistan V1 lane;
# a positive target is the worldwide V2 lane and uses its dedicated Trackio
# Space and ``v2-worldwide`` publication namespace.
#
# The payload launches the real llama-server binary directly with the
# validated parallelism and total context. There is no vLLM attempt, no
# wrapper script, and no reliance on PYTHONPATH or sitecustomize. The
# client concurrency is pinned to the parallel slot count so the request
# stream cannot outpace the server's bounded slots.

set -euo pipefail
umask 077

if [ "$#" -ne 18 ]; then
    echo "run_afghanistan_labeling: exactly eighteen arguments are required" >&2
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
ROW_LIMIT="${12}"; readonly ROW_LIMIT
LLAMA_PARALLEL="${13}"; readonly LLAMA_PARALLEL
LLAMA_PER_SLOT_CONTEXT="${14}"; readonly LLAMA_PER_SLOT_CONTEXT
REQUEST_CONCURRENCY="${15}"; readonly REQUEST_CONCURRENCY
SAMPLING_TARGET="${16}"; readonly SAMPLING_TARGET
SAMPLING_SEED="${17}"; readonly SAMPLING_SEED
SAMPLING_H3_RESOLUTION="${18}"; readonly SAMPLING_H3_RESOLUTION

: "${OAR_JOB_ID:?run_afghanistan_labeling requires an OAR allocation}"
case "${OAR_JOB_ID}" in (*[!0-9]*|'') echo "run_afghanistan_labeling: invalid OAR job ID" >&2; exit 2;; esac
command -v nvidia-smi >/dev/null || { echo "run_afghanistan_labeling: CUDA tooling unavailable" >&2; exit 1; }
nvidia-smi -L >/dev/null || { echo "run_afghanistan_labeling: no visible CUDA GPU" >&2; exit 1; }
if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${ROW_LIMIT}" =~ ^(0|[1-9][0-9]*)$ ]] || \
   ! [[ "${SAMPLING_TARGET}" =~ ^(0|[1-9][0-9]*)$ ]] || \
   ! [[ "${SAMPLING_SEED}" != *[[:space:]]* ]] || [ -z "${SAMPLING_SEED}" ] || \
   ! [[ "${SAMPLING_H3_RESOLUTION}" =~ ^(0|[1-9]|1[0-5])$ ]]; then
    echo "run_afghanistan_labeling: batch, row-limit, or sampling arguments are invalid" >&2
    exit 2
fi
case "${LLAMA_PARALLEL}" in
    1|2|4|8|16|32) ;;
    *) echo "run_afghanistan_labeling: LLAMA_PARALLEL must be one of 1, 2, 4, 8, 16, 32" >&2; exit 2;;
esac
if ! [[ "${LLAMA_PER_SLOT_CONTEXT}" =~ ^[1-9][0-9]*$ ]] || \
   [ "${LLAMA_PER_SLOT_CONTEXT}" -lt 4096 ] || \
   ! [[ "${REQUEST_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || \
   [ "${REQUEST_CONCURRENCY}" -gt "${LLAMA_PARALLEL}" ]; then
    echo "run_afghanistan_labeling: invalid context or concurrency" >&2
    exit 2
fi
LLAMA_TOTAL_CONTEXT=$((LLAMA_PARALLEL * LLAMA_PER_SLOT_CONTEXT)); readonly LLAMA_TOTAL_CONTEXT

case "${MODEL_FILE}" in (*Qwen3.6-27B-Q4_K_M.gguf) ;; (*) echo "run_afghanistan_labeling: expected pinned Q4_K_M model file" >&2; exit 2;; esac
test -f "${INPUT_PARQUET}" || { echo "run_afghanistan_labeling: input Parquet missing" >&2; exit 2; }
test -f "${MODEL_FILE}" || { echo "run_afghanistan_labeling: model file missing" >&2; exit 2; }
test -d "${TOKENIZER_DIR}" || { echo "run_afghanistan_labeling: tokenizer directory missing" >&2; exit 2; }
RUN_ID="$(basename -- "$(dirname -- "${WORK_DIR}")")"; readonly RUN_ID
if ! [[ "${RUN_ID}" =~ ^[0-9a-f]{20}$ ]]; then
    echo "run_afghanistan_labeling: work directory must be nested under a 20-hex run directory" >&2
    exit 2
fi
CHECKPOINT_BRANCH="checkpoints/${RUN_ID}"; readonly CHECKPOINT_BRANCH
RELEASE_LANE="v1-afghanistan"
if [ "${SAMPLING_TARGET}" -gt 0 ]; then
    RELEASE_LANE="v2-worldwide"
fi
readonly RELEASE_LANE
TRACKIO_ARGS=()
if [ "${RELEASE_LANE}" = "v2-worldwide" ]; then
    TRACKIO_ARGS=(
        --trackio-project "worldwide-stratified-labeling"
        --trackio-run-name "run-${RUN_ID}"
        --trackio-space-id "NoeFlandre/worldwide-stratified-labeling-trackio"
    )
fi

LABEL_CLI="${REPO_ROOT}/.venv/bin/osm-polygon-label-sentences"
test -x "${LABEL_CLI}" || { echo "run_afghanistan_labeling: labeling CLI missing" >&2; exit 2; }
MODEL_SHA256=$(sha256sum "${MODEL_FILE}" | awk '{print $1}')
readonly MODEL_SHA256
PORT=8000; readonly PORT
MODEL_REPO_ID="unsloth/Qwen3.6-27B-MTP-GGUF"; readonly MODEL_REPO_ID
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

probe_engine() {
    "${LABEL_CLI}" probe \
        --input-parquet "${INPUT_PARQUET}" \
        --engine llama.cpp \
        --endpoint "http://127.0.0.1:${PORT}/v1/chat/completions" \
        --sample-size 4 \
        --llama-parallel "${LLAMA_PARALLEL}" \
        --llama-per-slot-context "${LLAMA_PER_SLOT_CONTEXT}" \
        --llama-total-context "${LLAMA_TOTAL_CONTEXT}" \
        --request-concurrency "${REQUEST_CONCURRENCY}" \
        >"${WORK_DIR}.llama.probe.json" \
        2>"${WORK_DIR}.llama.probe.stderr.log"
}

command -v llama-server >/dev/null || { echo "run_afghanistan_labeling: llama-server is unavailable" >&2; exit 1; }
llama-server --model "${MODEL_FILE}" --alias "${MODEL_REPO_ID}" \
    --host 127.0.0.1 --port "${PORT}" \
    --ctx-size "${LLAMA_TOTAL_CONTEXT}" --parallel "${LLAMA_PARALLEL}" \
    --n-gpu-layers 999 \
    >"${WORK_DIR}.llama.stdout.log" 2>"${WORK_DIR}.llama.stderr.log" &
SERVER_PID=$!
if ! health; then
    echo "run_afghanistan_labeling: llama-server failed to become healthy" >&2
    exit 1
fi
if ! probe_engine; then
    echo "run_afghanistan_labeling: llama.cpp canary failed" >&2
    exit 1
fi
ENGINE=llama.cpp
readonly ENGINE
ENGINE_VERSION=$(llama-server --version 2>&1 | sed -n '1p')
readonly ENGINE_VERSION

LABEL_RESULT="${WORK_DIR}.label-result.json"
"${LABEL_CLI}" label \
    --input-parquet "${INPUT_PARQUET}" --work-dir "${WORK_DIR}" \
    --input-dataset-revision "${INPUT_REVISION}" \
    --model-revision "${MODEL_REVISION}" --model-file-sha256 "${MODEL_SHA256}" \
    --source-commit "${SOURCE_COMMIT}" --engine "${ENGINE}" \
    --engine-version "${ENGINE_VERSION}" --batch-size "${BATCH_SIZE}" \
    --concurrency "${REQUEST_CONCURRENCY}" \
    --llama-parallel "${LLAMA_PARALLEL}" \
    --llama-per-slot-context "${LLAMA_PER_SLOT_CONTEXT}" \
    --llama-total-context "${LLAMA_TOTAL_CONTEXT}" \
    --request-concurrency "${REQUEST_CONCURRENCY}" \
    --row-limit "${ROW_LIMIT}" \
    --sampling-target "${SAMPLING_TARGET}" \
    --sampling-seed "${SAMPLING_SEED}" \
    --h3-resolution "${SAMPLING_H3_RESOLUTION}" \
    --checkpoint-dataset-id "${DATASET_ID}" \
    --checkpoint-namespace "${CHECKPOINT_BRANCH}" \
    --checkpoint-drain-seconds "30" \
    --release-lane "${RELEASE_LANE}" \
    "${TRACKIO_ARGS[@]}" \
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
    --engine-version "${ENGINE_VERSION}" --batch-size "${BATCH_SIZE}" \
    --row-limit "${ROW_LIMIT}" \
    --sampling-target "${SAMPLING_TARGET}" \
    --sampling-seed "${SAMPLING_SEED}" \
    --h3-resolution "${SAMPLING_H3_RESOLUTION}" \
    --llama-parallel "${LLAMA_PARALLEL}" \
    --llama-per-slot-context "${LLAMA_PER_SLOT_CONTEXT}" \
    --llama-total-context "${LLAMA_TOTAL_CONTEXT}" \
    --request-concurrency "${REQUEST_CONCURRENCY}" \
    --release-lane "${RELEASE_LANE}"

if [ "${ROW_LIMIT}" -eq 0 ]; then
    "${LABEL_CLI}" publish --output-dir "${OUTPUT_DIR}" --dataset-id "${DATASET_ID}"
else
    echo "Canary complete; publication intentionally skipped: ${OUTPUT_DIR}"
fi
