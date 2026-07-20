"""Operational contracts for the Grid'5000 streaming launchers."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
GRID = ROOT / "scripts" / "grid5000"


def _text(name: str) -> str:
    return (GRID / name).read_text(encoding="utf-8")


def test_streaming_launchers_are_executable() -> None:
    for name in (
        "submit_streaming_build.sh",
        "run_streaming_build_job.sh",
        "run_streaming_build.sh",
    ):
        assert os.stat(GRID / name).st_mode & 0o111


def test_payload_uses_locked_python_real_inventory_and_explicit_cuda() -> None:
    text = _text("run_streaming_build.sh")
    assert 'PYTHON="${REPO_ROOT}/.venv/bin/python"' in text
    assert 'exec "${PYTHON}" "${args[@]}"' in text
    assert "--device cuda" in text
    assert '--shard "all"' not in text
    assert "unset HF_HUB_OFFLINE" in text


def test_job_uses_allocation_bound_scratch_and_never_bare_python() -> None:
    text = _text("run_streaming_build_job.sh")
    assert (
        'SCRATCH_BASE="${LOCALSCRATCH:-${OAR_JOB_SCRATCH_DIR:-/tmp/oar-${OAR_JOB_ID}}}"'
        in text
    )
    assert 'WORK_DIR="${SCRATCH_BASE}/osm_streaming_${RUN_ID}"' in text
    assert 'PYTHON="${REPO_ROOT}/.venv/bin/python"' in text
    assert "python3" not in text
    assert "export HF_HUB_OFFLINE" not in text


def test_submitter_submits_one_noninteractive_gpu_job() -> None:
    text = _text("submit_streaming_build.sh")
    assert text.count("exec oarsub ") == 1
    assert "gpu=1,walltime=12:00:00" in text
    assert " -I" not in text
    assert "device auto" not in text
