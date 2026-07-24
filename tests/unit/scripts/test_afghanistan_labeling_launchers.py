from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRID = ROOT / "scripts" / "grid5000"


def _text(name: str) -> str:
    return (GRID / name).read_text()


def test_labeling_launchers_are_executable() -> None:
    for name in (
        "submit_afghanistan_labeling.sh",
        "run_afghanistan_labeling_job.sh",
        "run_afghanistan_labeling.sh",
    ):
        assert os.stat(GRID / name).st_mode & 0o111


def test_submitter_requests_one_fast_large_cuda_gpu_once() -> None:
    text = _text("submit_afghanistan_labeling.sh")
    assert text.count("exec oarsub ") == 1
    assert "gpu=1,walltime=01:00:00" in text
    assert "gpu_mem>=60000" in text
    assert " -I" not in text


def test_job_wrapper_verifies_checkout_gpu_and_persists_logs() -> None:
    text = _text("run_afghanistan_labeling_job.sh")
    assert "gpu_preflight.py" in text
    assert "git -C" in text
    assert "status --porcelain" in text
    assert 'JOB_LOG_DIR="${LOG_ROOT}/${OAR_JOB_ID}"' in text
    assert "labeling.exit_code" in text
    assert "run_afghanistan_labeling.sh" in text


def test_job_wrapper_translates_submit_arguments_to_payload_contract() -> None:
    text = _text("run_afghanistan_labeling_job.sh")
    normalized = " ".join(text.replace("\\", "").split())
    expected = (
        '"${PAYLOAD}" "${REPO_ROOT}" "$4" "$5" "$6" "$7" "$8" "$9" '
        '"${10}" "${11}" "${12}" "${13}" "${14}"'
    )
    assert expected in normalized
    assert '"${PAYLOAD}" "$@"' not in text


def test_canary_launch_contract_never_publishes() -> None:
    text = _text("run_afghanistan_labeling.sh")
    assert 'if [ "${ROW_LIMIT}" -eq 0 ]; then' in text
    assert '"${LABEL_CLI}" publish' in text
