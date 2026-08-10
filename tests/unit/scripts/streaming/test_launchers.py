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


def test_job_keeps_work_dir_in_persistent_run_root() -> None:
    text = _text("run_streaming_build_job.sh")
    assert (
        'SCRATCH_BASE="${LOCALSCRATCH:-${OAR_JOB_SCRATCH_DIR:-/tmp/oar-${OAR_JOB_ID}}}"'
        in text
    )
    assert 'WORK_DIR="${RUN_ROOT}/work"' in text
    assert 'mkdir -p -m 0700 -- "${WORK_DIR}"' in text
    assert 'WORK_DIR="${SCRATCH_BASE}/osm_streaming_${RUN_ID}"' not in text
    assert 'PYTHON="${REPO_ROOT}/.venv/bin/python"' in text
    assert "python3" not in text
    assert "export HF_HUB_OFFLINE" not in text


def test_streaming_split_separates_checkout_and_data_source_commits() -> None:
    for name in (
        "submit_streaming_build.sh",
        "run_streaming_build_job.sh",
        "run_streaming_build.sh",
    ):
        text = _text(name)
        assert "DATA_SOURCE_COMMIT" in text
        assert "twelve" in text or '"$#" -ne 12' in text


def test_job_rebuilds_environment_on_compute_node() -> None:
    text = _text("run_streaming_build_job.sh")
    assert "_checkout_guard.sh" in text
    assert "prepare_compute_environment" in text
    assert 'HF_TOKEN_FILE="${RUN_ROOT}/.hf-token"' in text
    assert 'export HF_TOKEN="$(cat -- "${HF_TOKEN_FILE}")"' in text
    assert 'stat -c %a -- "${HF_TOKEN_FILE}"' in text


def test_finalization_job_rebuilds_environment_on_compute_node() -> None:
    text = _text("run_streaming_finalization_job.sh")
    assert "_checkout_guard.sh" in text
    assert "prepare_compute_environment" in text
    assert "trap mark_failed_on_exit EXIT" in text


def test_finalization_payload_treats_all_as_inventory_sentinel() -> None:
    text = _text("run_streaming_finalization.sh")
    assert 'if [ "${EXPECTED_SHARD}" != "all" ]; then' in text
    assert 'args+=(--expected-shard "${EXPECTED_SHARD}")' in text
    assert 'args+=(--sampling-target "${SAMPLING_TARGET}")' in text
    assert '--sampling-seed "${SAMPLING_SEED}"' in text


def test_split_job_uses_short_resumable_walltime_and_deadline() -> None:
    text = _text("run_streaming_build_job.sh")
    assert 'if [ "${COMPUTE_ENVIRONMENT_REUSED:-0}" -eq 1 ]; then' in text
    assert "grace_seconds=60" in text
    assert "grace_seconds=240" in text
    assert 'DEADLINE_GRACE="${grace_seconds}s"' in text
    assert (
        'deadline_helper_run "${DEADLINE_DURATION}" "${DEADLINE_GRACE}" '
        '"${PAYLOAD}"' in text
    )
    assert "one scheduler minute" in text


def test_job_marks_failed_managed_root_for_later_cleanup() -> None:
    text = _text("run_streaming_build_job.sh")
    assert 'status":"failed"' in text
    assert "trap mark_failed_on_exit EXIT" in text


def test_submitter_uses_shared_gpu_resource_helper() -> None:
    text = _text("submit_streaming_build.sh")
    assert 'HELPER="${REPO_ROOT}/scripts/grid5000/_submit_gpu_job.sh"' in text
    assert 'exec "${HELPER}" "40000" "${walltime}" "${policy_type}"' in text
    assert "exec oarsub" not in text
    assert 'WALLTIME_SECONDS="${13:-1800}"' in text
    assert "STREAMING_WALLTIME_SECONDS=${WALLTIME_SECONDS}" in text
    assert "TZ=Europe/Paris date" in text
    assert "policy_type=night" in text
    assert "policy_type=day" in text
    assert " -I" not in text
    assert "device auto" not in text


def test_streaming_wrapper_derives_deadline_from_scheduler_walltime() -> None:
    text = _text("run_streaming_build_job.sh")
    assert 'STREAMING_WALLTIME_SECONDS="${STREAMING_WALLTIME_SECONDS:-1800}"' in text
    assert (
        "deadline_seconds=$((STREAMING_WALLTIME_SECONDS - grace_seconds - 60))" in text
    )
    assert 'DEADLINE_DURATION="${deadline_seconds}s"' in text


def test_finalization_submissions_are_night_bound() -> None:
    text = _text("submit_streaming_finalization.sh")
    assert text.count("exec oarsub ") == 2
    for line in text.splitlines():
        if "exec oarsub " in line:
            assert "-q default" in line
            assert "-t night" in line
