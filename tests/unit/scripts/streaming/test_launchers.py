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
    assert "unset TRANSFORMERS_OFFLINE" in text
    assert "export TRANSFORMERS_OFFLINE=1" not in text


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


def test_job_rebuilds_environment_on_compute_node() -> None:
    text = _text("run_streaming_build_job.sh")
    assert "_checkout_guard.sh" in text
    assert "prepare_compute_environment" in text


def test_finalization_job_rebuilds_environment_on_compute_node() -> None:
    text = _text("run_streaming_finalization_job.sh")
    assert "_checkout_guard.sh" in text
    assert "prepare_compute_environment" in text
    assert "trap mark_failed_on_exit EXIT" in text


def test_split_job_uses_short_resumable_walltime_and_deadline() -> None:
    text = _text("run_streaming_build_job.sh")
    assert 'deadline_helper_run 20m 10m "${PAYLOAD}"' in text


def test_job_marks_failed_managed_root_for_later_cleanup() -> None:
    text = _text("run_streaming_build_job.sh")
    assert 'status":"failed"' in text
    assert "trap mark_failed_on_exit EXIT" in text


def test_submitter_uses_shared_gpu_resource_helper() -> None:
    text = _text("submit_streaming_build.sh")
    assert 'HELPER="${REPO_ROOT}/scripts/grid5000/_submit_gpu_job.sh"' in text
    assert 'exec "${HELPER}" "40000" "00:30:00" "${policy_type}"' in text
    assert "exec oarsub" not in text
    assert '"00:30:00"' in text
    assert "TZ=Europe/Paris date" in text
    assert "policy_type=night" in text
    assert "policy_type=day" in text
    assert " -I" not in text
    assert "device auto" not in text


def test_finalization_submissions_are_night_bound() -> None:
    text = _text("submit_streaming_finalization.sh")
    assert text.count("exec oarsub ") == 2
    for line in text.splitlines():
        if "exec oarsub " in line:
            assert "-q default" in line
            assert "-t night" in line
