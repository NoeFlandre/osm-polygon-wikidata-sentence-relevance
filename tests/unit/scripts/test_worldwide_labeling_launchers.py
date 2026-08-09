"""Contract tests for the dedicated V2 worldwide Grid'5000 launchers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRID = ROOT / "scripts" / "grid5000"


def _text(name: str) -> str:
    return (GRID / name).read_text()


def test_worldwide_launchers_are_executable_and_lane_specific() -> None:
    for name in (
        "submit_worldwide_labeling.sh",
        "run_worldwide_labeling_job.sh",
        "run_worldwide_labeling.sh",
    ):
        assert os.stat(GRID / name).st_mode & 0o111
    payload = _text("run_worldwide_labeling.sh")
    assert '"ggml-org/Qwen3.6-27B-GGUF"' in payload
    assert "--release-lane v2-worldwide" in payload
    assert "--trackio-project worldwide-stratified-labeling" in payload
    assert "unsloth/Qwen3.6-27B-MTP-GGUF" not in payload


def test_payload_uses_first_token_logit_lane_and_never_json_repair() -> None:
    payload = _text("run_worldwide_labeling.sh")
    assert "--checkpoint-namespace" in payload
    assert "--sampling-target" in payload
    assert '--h3-resolution "${SAMPLING_H3_RESOLUTION}"' in payload
    assert "grep -q '\"interrupted\": true'" in payload
    assert "--release-lane v2-worldwide" in payload


def test_payload_revalidates_immutable_identity_and_pinned_model() -> None:
    payload = _text("run_worldwide_labeling.sh")
    assert '[[ "${MODEL_REVISION}" =~ ^[0-9a-f]{40}$ ]]' in payload
    assert '[[ "${INPUT_REVISION}" =~ ^[0-9a-f]{40}$ ]]' in payload
    assert '[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]' in payload
    assert '[[ "${DATASET_ID}" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]]' in payload
    assert 'case "$(basename -- "${MODEL_FILE}")" in' in payload
    assert "Qwen3.6-27B-Q4_K_M.gguf" in payload


def test_wrapper_keeps_lane_aware_front_contract_and_resumable_lock() -> None:
    wrapper = _text("run_worldwide_labeling_job.sh")
    assert "exactly twenty-two arguments are required" in wrapper
    assert "validate_clean_checkout" in wrapper
    assert "flock -n" in wrapper
    assert "labeling.exit_code" in wrapper
    normalized = " ".join(wrapper.replace("\\", "").split())
    expected = (
        'deadline_helper_run "${DEADLINE_DURATION}" "${DEADLINE_GRACE}" '
        '"${PAYLOAD}" "${REPO_ROOT}" "$4" "$5" "$6" "$7" "$8" "$9" '
        '"${10}" "${11}" "${12}" "${13}" "${14}" "${15}" '
        '"${16}" "${17}" "${18}" "${19}" "${20}" "${21}"'
    )
    assert expected in normalized


def test_wrapper_rebuilds_environment_on_compute_node_and_marks_failure() -> None:
    wrapper = _text("run_worldwide_labeling_job.sh")
    assert "_checkout_guard.sh" in wrapper
    assert "prepare_compute_environment" in wrapper
    assert "trap mark_failed_on_exit EXIT" in wrapper
    assert 'HF_TOKEN_FILE="${RUN_ROOT}/.hf-token"' in wrapper
    assert 'export HF_TOKEN="$(cat -- "${HF_TOKEN_FILE}")"' in wrapper
    assert 'stat -c %a -- "${HF_TOKEN_FILE}"' in wrapper
    assert 'LABEL_LANE="${21}"' in wrapper
    assert 'EXECUTION_COMMIT="${22}"' in wrapper
    assert 'rev-parse HEAD)" != "${EXECUTION_COMMIT}' in wrapper


def test_submitter_uses_shared_policy_helper_and_single_submission() -> None:
    submit = _text("submit_worldwide_labeling.sh")
    assert "exactly twenty-two arguments are required" in submit
    assert 'case "${21}" in smoke|production)' in submit
    assert '"40000" "00:55:00"' in submit
    assert '"${command_string}"' in submit
    assert '"${HELPER}"' in submit
    assert "exec oarsub" not in submit


def test_submitter_rejects_mutable_revision_before_scheduler_call(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    fake_bin = root / "bin"
    fake_bin.mkdir()
    calls = root / "calls"
    (fake_bin / "_submit_gpu_job.sh").write_text(
        f"#!/usr/bin/env bash\nprintf '%s\\n' called > \"{calls}\"\n"
    )
    (fake_bin / "_submit_gpu_job.sh").chmod(0o700)
    repo = root / "repo"
    (repo / "scripts/grid5000").mkdir(parents=True)
    for name in ("submit_worldwide_labeling.sh", "run_worldwide_labeling_job.sh"):
        destination = repo / "scripts/grid5000" / name
        destination.write_bytes((GRID / name).read_bytes())
        destination.chmod((GRID / name).stat().st_mode)
    paths = [
        root / name
        for name in ("hf", "logs", "input", "work", "output", "model", "tokenizer")
    ]
    for path in paths:
        path.mkdir(parents=True)
    (paths[2] / "sentences.parquet").write_text("data")
    (paths[5] / "model.gguf").write_text("model")
    args = [
        str(repo),
        str(paths[0]),
        str(paths[1]),
        str(paths[2] / "sentences.parquet"),
        str(paths[3]),
        str(paths[4]),
        str(paths[5] / "model.gguf"),
        str(paths[6]),
        "not-a-revision",
        "b" * 40,
        "c" * 40,
        "owner/dataset",
        "128",
        "0",
        "16",
        "4096",
        "16",
        "200000",
        "seed",
        "3",
        "production",
        "d" * 40,
    ]
    result = subprocess.run(
        ["bash", str(GRID / "submit_worldwide_labeling.sh"), *args],
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert not calls.exists()


def test_payload_isolates_lane_checkpoint_and_tracking_namespaces() -> None:
    payload = _text("run_worldwide_labeling.sh")

    assert "exactly nineteen arguments are required" in payload
    assert 'case "${LABEL_LANE}" in' in payload
    assert "smoke)" in payload
    assert "production)" in payload
    assert 'CHECKPOINT_NAMESPACE="checkpoints/${RUN_ID}/${LABEL_LANE}"' in payload
    assert '--trackio-run-name "run-${RUN_ID}-${LABEL_LANE}"' in payload
    assert 'if [ "${LABEL_LANE}" = "production" ]; then' in payload
