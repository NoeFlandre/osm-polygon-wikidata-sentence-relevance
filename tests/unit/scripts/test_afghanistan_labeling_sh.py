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


def test_payload_attempts_vllm_then_llama_cpp_only_on_failed_canary() -> None:
    text = SCRIPT.read_text()
    assert text.index("vllm serve") < text.index("llama-server")
    assert '--hf-config-path "${TOKENIZER_DIR}"' in text
    assert "health" in text
    assert "ENGINE=vllm" in text
    assert "ENGINE=llama.cpp" in text


def test_payload_runs_label_finalize_publish_in_order() -> None:
    text = SCRIPT.read_text()
    label = text.index('"${LABEL_CLI}" label')
    finalize = text.index('"${LABEL_CLI}" finalize')
    publish = text.index('"${LABEL_CLI}" publish')
    assert label < finalize < publish
    assert "--dataset-id" in text
    assert '"main"' not in text  # publication command has no mutable input revision


def test_payload_uses_q4_k_m_and_pinned_local_files() -> None:
    text = SCRIPT.read_text()
    assert "Qwen3.6-27B-Q4_K_M.gguf" in text
    assert "--model-revision" in text
    assert "--model-file-sha256" in text
    assert "--input-dataset-revision" in text
    assert "hf download" not in text
