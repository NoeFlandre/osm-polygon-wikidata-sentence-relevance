from __future__ import annotations

from pathlib import Path

SCRIPT = Path("scripts/grid5000/run_afghanistan_labeling.sh")


def test_payload_requires_oar_and_cuda_without_fallback() -> None:
    text = SCRIPT.read_text()
    assert "OAR_JOB_ID" in text
    assert "nvidia-smi" in text
    assert "--device cpu" not in text
    assert "--device mps" not in text
    assert "CUDA_VISIBLE_DEVICES=" not in text


def test_payload_launches_llama_server_directly() -> None:
    text = SCRIPT.read_text()
    # No vLLM attempt; the test for the legacy fallback is intentionally gone.
    assert "vllm serve" not in text
    assert "vllm --version" not in text
    assert "ENGINE=vllm" not in text
    assert "ENGINE=llama.cpp" in text
    assert '--hf-config-path "${TOKENIZER_DIR}"' not in text
    assert '--served-model-name "${MODEL_REPO_ID}"' not in text
    assert '--alias "${MODEL_REPO_ID}"' in text
    assert "health" in text
    assert text.index("probe_engine") < text.index('"${LABEL_CLI}" label')


def test_payload_runs_label_finalize_publish_in_order() -> None:
    text = SCRIPT.read_text()
    label = text.index('"${LABEL_CLI}" label')
    finalize = text.index('"${LABEL_CLI}" finalize')
    publish = text.index('"${LABEL_CLI}" publish')
    assert label < finalize < publish
    assert "--dataset-id" in text
    assert '"main"' not in text  # publication command has no mutable input revision
    assert 'if [ "${ROW_LIMIT}" -eq 0 ]; then' in text


def test_payload_uses_q4_k_m_and_pinned_local_files() -> None:
    text = SCRIPT.read_text()
    assert "Qwen3.6-27B-Q4_K_M.gguf" in text
    assert "--model-revision" in text
    assert "--model-file-sha256" in text
    assert "--input-dataset-revision" in text
    assert "hf download" not in text


def test_payload_supports_nonpublishing_representative_canary() -> None:
    text = SCRIPT.read_text()
    assert "exactly fifteen arguments" in text
    assert 'ROW_LIMIT="${12}"' in text
    assert '--row-limit "${ROW_LIMIT}"' in text
    assert 'LLAMA_PARALLEL="${13}"' in text
    assert '--llama-parallel "${LLAMA_PARALLEL}"' in text
    assert "LLAMA_TOTAL_CONTEXT=$((LLAMA_PARALLEL * LLAMA_PER_SLOT_CONTEXT))" in text
    assert "Canary complete; publication intentionally skipped" in text
