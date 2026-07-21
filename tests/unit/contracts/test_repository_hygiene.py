"""Repository-level contracts for the supported production surface."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RETAINED_SHELL = {
    "_finalize_persist.sh",
    "run_streaming_build.sh",
    "run_streaming_build_job.sh",
    "run_streaming_finalization.sh",
    "run_streaming_finalization_job.sh",
    "submit_streaming_build.sh",
    "submit_streaming_finalization.sh",
}


def test_grid5000_contains_only_production_shell_entrypoints() -> None:
    actual = {path.name for path in (ROOT / "scripts/grid5000").glob("*.sh")}
    assert actual == RETAINED_SHELL


def test_one_off_audit_scripts_are_absent() -> None:
    assert not (ROOT / "scripts/audit").exists()
    assert not (ROOT / "scripts/audit_upstream_correction.py").exists()
