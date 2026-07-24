"""Repository-level contracts for the supported production surface."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RETAINED_SHELL = {
    "_finalize_persist.sh",
    "run_afghanistan_labeling.sh",
    "run_afghanistan_labeling_job.sh",
    "run_streaming_build.sh",
    "run_streaming_build_job.sh",
    "run_streaming_finalization.sh",
    "run_streaming_finalization_job.sh",
    "submit_streaming_build.sh",
    "submit_streaming_finalization.sh",
    "submit_afghanistan_labeling.sh",
}


def test_grid5000_contains_only_production_shell_entrypoints() -> None:
    actual = {path.name for path in (ROOT / "scripts/grid5000").glob("*.sh")}
    assert actual == RETAINED_SHELL


def test_one_off_audit_scripts_are_absent() -> None:
    assert not (ROOT / "scripts/audit").exists()
    assert not (ROOT / "scripts/audit_upstream_correction.py").exists()


def test_current_test_files_use_contract_names() -> None:
    forbidden = ("amendment", "phase", "pause")
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("test_*.py")
        if any(token in path.name.lower() for token in forbidden)
    ]
    assert offenders == []
