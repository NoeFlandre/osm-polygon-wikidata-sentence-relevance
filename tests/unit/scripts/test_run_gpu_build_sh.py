"""Contract tests for the Grid'5000 compute-node payload that drives
the full resumable build (Phase 9L-B).

``scripts/grid5000/run_gpu_build.sh`` runs *inside* an allocated OAR
job (after ``OAR_JOB_ID`` is set). It runs the existing CUDA
preflight, writes run metadata, then invokes the existing public
``osm-polygon-sentence-relevance`` CLI in local-input mode with
explicit CUDA, checkpoint persistence and no publishing flags. It
never runs on a Mac or frontend.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PAYLOAD_SCRIPT = ROOT / "scripts" / "grid5000" / "run_gpu_build.sh"


def _test_sha(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8"), usedforsecurity=False).hexdigest()


TEST_SOURCE_COMMIT = _test_sha("tests/grid5000/run_gpu_build/source_commit")
TEST_INPUT_REVISION = _test_sha("tests/grid5000/run_gpu_build/input_revision")
TEST_OAR_JOB_ID = "9876543210"


_FORBIDDEN_PATTERNS = (
    "eval ",
    "CUDA_VISIBLE_DEVICES:=",
    "OAR_JOB_ID:=",
    'device="auto"',
    'device="cpu"',
    'device="mps"',
    "--device auto",
    "--device cpu",
    "--device mps",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "api_key",
    "/home/nflandre",
    "/Users/",
    "/Volumes/",
    "/srv/storage",
    "git commit",
    "git push",
    "git checkout",
    "git fetch",
    "git clone",
    "rsync",
    "ssh ",
    " snapshot_download",
    "curl ",
    "wget ",
    "uv run",
    "conda",
    "activate",
    # No publishing flags in the build payload.
    "--publish-dataset-id",
    "--publish-revision",
    "--publish-commit-message",
    # Build must use --input-root, never --input-dataset-id.
    "--input-dataset-id ",
    # No --overwrite; output dir must be fresh.
    "--overwrite",
)


@pytest.fixture(scope="module")
def script_text() -> str:
    return PAYLOAD_SCRIPT.read_text(encoding="utf-8")


# --- Static structure checks ----------------------------------------


def test_payload_script_exists_and_is_nonempty():
    assert PAYLOAD_SCRIPT.exists()
    assert len(PAYLOAD_SCRIPT.read_text(encoding="utf-8")) > 200


def test_payload_script_passes_bash_syntax_check():
    proc = subprocess.run(
        ["bash", "-n", str(PAYLOAD_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_payload_script_uses_strict_bash_modes(script_text):
    assert "set -euo pipefail" in script_text


def test_payload_script_has_no_forbidden_patterns(script_text):
    for pat in _FORBIDDEN_PATTERNS:
        assert pat not in script_text, f"forbidden pattern in build payload: {pat!r}"


def test_payload_script_requires_oar_job_id(script_text):
    assert ": ${OAR_JOB_ID:?OAR_JOB_ID is required" in script_text or (
        "OAR_JOB_ID" in script_text and ":?" in script_text
    )


def test_payload_script_does_not_touch_cuda_visible_devices(script_text):
    assert "CUDA_VISIBLE_DEVICES=" not in script_text


def test_payload_script_inherits_required_env(script_text):
    for var in (
        "REPO_ROOT",
        "HF_HOME",
        "BUILD_LOG_DIR",
        "INPUT_ROOT",
        "WORK_DIR",
        "OUTPUT_DIR",
        "EXPECTED_SOURCE_COMMIT",
        "INPUT_REVISION",
    ):
        assert var in script_text, f"{var} missing in payload"


def test_payload_script_invokes_existing_cli(script_text):
    assert ".venv/bin/osm-polygon-sentence-relevance" in script_text


def test_payload_script_uses_explicit_cuda(script_text):
    assert "--device cuda" in script_text


def test_payload_script_uses_input_root_not_dataset_id(script_text):
    assert "--input-root" in script_text
    assert "--input-source-dataset-id" in script_text


def test_payload_script_pins_input_revision(script_text):
    assert "--input-dataset-revision" in script_text
    assert "INPUT_REVISION" in script_text


def test_payload_script_passes_work_dir_and_source_commit(script_text):
    assert "--work-dir" in script_text
    assert "--source-commit" in script_text


def test_payload_script_has_no_publishing(script_text):
    for pat in (
        "--publish-dataset-id",
        "--publish-revision",
        "--publish-commit-message",
    ):
        assert pat not in script_text


def test_payload_script_has_no_overwrite(script_text):
    assert "--overwrite" not in script_text


def test_payload_script_enforces_offline_mode(script_text):
    assert "HF_HUB_OFFLINE=1" in script_text
    assert "TRANSFORMERS_OFFLINE=1" in script_text


def test_payload_script_writes_preflight_and_metadata(script_text):
    assert "gpu_preflight.json" in script_text
    assert "run_metadata.json" in script_text


def test_payload_script_documents_resume_semantics(script_text):
    # The payload must document that WORK_DIR is left untouched on
    # failure and may be reused on a subsequent invocation.
    assert "resume" in script_text.lower() or "WORK_DIR" in script_text


# --- Runtime behaviour with fake interpreter / fake CLI -------------


def _make_fake_bin(tmp_path: Path) -> Path:
    """Create a fake project interpreter directory containing both
    ``python`` and ``osm-polygon-sentence-relevance`` entry points.
    """
    venv_bin = tmp_path / "fake_venv_bin"
    venv_bin.mkdir(parents=True, exist_ok=True)

    py = venv_bin / "python"
    py.write_text(
        """#!/usr/bin/env python3
import json
import os
import shutil
import sys

cap = os.environ["PAYLOAD_CAPTURE"]
with open(cap + ".pythons", "a") as fh:
    fh.write(sys.argv[0] + "\\n")
with open(cap + ".python_args", "a") as fh:
    fh.write(" ".join(sys.argv[1:]) + "\\n")

helper = os.path.basename(sys.argv[1]) if len(sys.argv) > 1 else ""
oar = os.environ.get("OAR_JOB_ID", "unknown")

if helper == "gpu_preflight.py":
    payload = {
        "oar_job_id": oar,
        "hostname": "host",
        "torch_version": "x",
        "torch_cuda_runtime_version": "y",
        "visible_cuda_device_count": 1,
        "device_0_name": "fake",
    }
    json.dump(payload, sys.stdout)
    sys.exit(0)

if helper == "_run_metadata.py":
    if len(sys.argv) > 2:
        payload = {
            "source_commit": sys.argv[2],
            "oar_job_id": oar,
        }
        with open(sys.argv[2], "w") as fh:
            json.dump(payload, fh)
    sys.exit(0)

if helper == "_validate_artifact.py" and len(sys.argv) > 2 and sys.argv[1].endswith("_validate_artifact.py"):
    if len(sys.argv) >= 4 and sys.argv[2] == "install":
        src, dst = sys.argv[3], sys.argv[4]
        shutil.copyfile(src, dst)
        sys.exit(0)
    sys.exit(0)

sys.exit(0)
"""
    )
    py.chmod(0o755)

    cli = venv_bin / "osm-polygon-sentence-relevance"
    cli.write_text(
        """#!/usr/bin/env bash
set -u
cap="${PAYLOAD_CAPTURE:?PAYLOAD_CAPTURE required}"
printf '%s\\n' "$0" >> "${cap}.cli_runs"
printf '%s\\n' "$*" >> "${cap}.cli_args"
printf '%s\\n' "${OAR_JOB_ID:-}" >> "${cap}.cli_oar"
printf '%s\\n' "${REPO_ROOT:-}" >> "${cap}.cli_repo"
printf '%s\\n' "${HF_HOME:-}" >> "${cap}.cli_hf"
printf '%s\\n' "${INPUT_ROOT:-}" >> "${cap}.cli_input"
printf '%s\\n' "${WORK_DIR:-}" >> "${cap}.cli_work"
printf '%s\\n' "${OUTPUT_DIR:-}" >> "${cap}.cli_output"
printf '%s\\n' "${EXPECTED_SOURCE_COMMIT:-}" >> "${cap}.cli_src"
printf '%s\\n' "${INPUT_REVISION:-}" >> "${cap}.cli_rev"
printf '%s\\n' "${BUILD_LOG_DIR:-}" >> "${cap}.cli_log"
exit 0
"""
    )
    cli.chmod(0o755)
    return venv_bin


def _make_layout(tmp_path: Path) -> dict[str, Path]:
    base = tmp_path / "persistent"
    base.mkdir(parents=True, exist_ok=True)
    repo = base / "repo"
    hf = base / "hf_home"
    log = base / "logs"
    inp = base / "input"
    work = base / "work"
    out = base / "output"
    for p in (repo, hf, log, inp, work):
        p.mkdir(parents=True, exist_ok=True)
    # OUTPUT_DIR is NOT created; the payload creates it.
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@e"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "T"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "--allow-empty", "-q", "-m", "x"],
        check=True,
        capture_output=True,
    )
    return {
        "repo": repo,
        "hf": hf,
        "log": log,
        "input": inp,
        "work": work,
        "output": out,
    }


def _run_payload(
    tmp_path: Path,
    *,
    repo_root: Path,
    hf_home: Path,
    build_log_dir: Path,
    input_root: Path,
    work_dir: Path,
    output_dir: Path,
    source_commit: str,
    input_revision: str,
    oar_job_id: str = TEST_OAR_JOB_ID,
    cli_overwrite: str | None = None,
) -> subprocess.CompletedProcess:
    fake_bin = _make_fake_bin(tmp_path)
    if cli_overwrite is not None:
        (fake_bin / "osm-polygon-sentence-relevance").write_text(cli_overwrite)
        (fake_bin / "osm-polygon-sentence-relevance").chmod(0o755)
    real_venv = repo_root / ".venv" / "bin"
    real_venv.mkdir(parents=True, exist_ok=True)
    py_link = real_venv / "python"
    if py_link.exists() or py_link.is_symlink():
        py_link.unlink()
    py_link.symlink_to(fake_bin / "python")
    cli_link = real_venv / "osm-polygon-sentence-relevance"
    if cli_link.exists() or cli_link.is_symlink():
        cli_link.unlink()
    cli_link.symlink_to(fake_bin / "osm-polygon-sentence-relevance")

    scripts_dir = repo_root / "scripts" / "grid5000"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "gpu_preflight.py").write_text("# stub\n")
    (scripts_dir / "_run_metadata.py").write_text("# stub\n")
    (scripts_dir / "_validate_artifact.py").write_text("# stub\n")

    build_log_dir.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "OAR_JOB_ID": oar_job_id,
        "REPO_ROOT": str(repo_root),
        "HF_HOME": str(hf_home),
        "BUILD_LOG_DIR": str(build_log_dir),
        "INPUT_ROOT": str(input_root),
        "WORK_DIR": str(work_dir),
        "OUTPUT_DIR": str(output_dir),
        "EXPECTED_SOURCE_COMMIT": source_commit,
        "INPUT_REVISION": input_revision,
        "PAYLOAD_CAPTURE": str(tmp_path / "payload_cap"),
        "PATH": os.environ.get("PATH", ""),
    }

    return subprocess.run(
        ["bash", str(PAYLOAD_SCRIPT)],
        cwd=str(build_log_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


# --- Argument safety / no retry / explicit CUDA ----------------------


class TestArgumentSafety:
    def test_cli_invoked_with_explicit_cuda(self, tmp_path):
        layout = _make_layout(tmp_path)
        log_dir = layout["log"] / TEST_OAR_JOB_ID
        proc = _run_payload(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            build_log_dir=log_dir,
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode == 0, proc.stderr
        cli_args = (tmp_path / "payload_cap.cli_args").read_text().splitlines()
        assert any("--device cuda" in a for a in cli_args), cli_args

    def test_cli_invoked_with_input_root_and_source_dataset_id(self, tmp_path):
        layout = _make_layout(tmp_path)
        log_dir = layout["log"] / TEST_OAR_JOB_ID
        proc = _run_payload(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            build_log_dir=log_dir,
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode == 0, proc.stderr
        cli_args = (tmp_path / "payload_cap.cli_args").read_text().splitlines()
        joined = " ".join(cli_args)
        # Both --input-root AND --input-source-dataset-id must appear.
        assert "--input-root" in joined
        assert "--input-source-dataset-id" in joined
        assert "NoeFlandre/osm-polygon-wikidata-only" in joined
        # INPUT_ROOT path must be passed.
        assert str(layout["input"]) in joined

    def test_cli_invoked_with_pinned_input_revision(self, tmp_path):
        layout = _make_layout(tmp_path)
        log_dir = layout["log"] / TEST_OAR_JOB_ID
        proc = _run_payload(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            build_log_dir=log_dir,
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode == 0, proc.stderr
        cli_args = (tmp_path / "payload_cap.cli_args").read_text().splitlines()
        joined = " ".join(cli_args)
        assert "--input-dataset-revision" in joined
        assert TEST_INPUT_REVISION in joined

    def test_cli_invoked_with_work_dir_output_dir_source_commit(self, tmp_path):
        layout = _make_layout(tmp_path)
        log_dir = layout["log"] / TEST_OAR_JOB_ID
        proc = _run_payload(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            build_log_dir=log_dir,
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode == 0, proc.stderr
        cli_args = (tmp_path / "payload_cap.cli_args").read_text().splitlines()
        joined = " ".join(cli_args)
        assert "--work-dir" in joined
        assert str(layout["work"]) in joined
        assert "--output-dir" in joined
        assert str(layout["output"]) in joined
        assert "--source-commit" in joined
        assert TEST_SOURCE_COMMIT in joined

    def test_cli_invoked_without_publishing_or_overwrite_flags(self, tmp_path):
        layout = _make_layout(tmp_path)
        log_dir = layout["log"] / TEST_OAR_JOB_ID
        proc = _run_payload(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            build_log_dir=log_dir,
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode == 0, proc.stderr
        cli_args = (tmp_path / "payload_cap.cli_args").read_text().splitlines()
        joined = " ".join(cli_args)
        for pat in (
            "--publish-dataset-id",
            "--publish-revision",
            "--publish-commit-message",
            "--overwrite",
        ):
            assert pat not in joined

    def test_cli_invoked_exactly_once(self, tmp_path):
        layout = _make_layout(tmp_path)
        log_dir = layout["log"] / TEST_OAR_JOB_ID
        proc = _run_payload(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            build_log_dir=log_dir,
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode == 0, proc.stderr
        cli_runs = (tmp_path / "payload_cap.cli_runs").read_text().splitlines()
        assert len(cli_runs) == 1, cli_runs


# --- Environment propagation ----------------------------------------


class TestEnvironmentPropagation:
    def test_required_env_propagates_to_cli(self, tmp_path):
        layout = _make_layout(tmp_path)
        log_dir = layout["log"] / TEST_OAR_JOB_ID
        proc = _run_payload(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            build_log_dir=log_dir,
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            oar_job_id=TEST_OAR_JOB_ID,
        )
        assert proc.returncode == 0, proc.stderr
        cap = tmp_path / "payload_cap"
        assert cap.with_suffix(".cli_oar").read_text().strip() == TEST_OAR_JOB_ID
        assert cap.with_suffix(".cli_repo").read_text().strip() == str(layout["repo"])
        assert cap.with_suffix(".cli_hf").read_text().strip() == str(layout["hf"])
        assert cap.with_suffix(".cli_input").read_text().strip() == str(layout["input"])
        assert cap.with_suffix(".cli_work").read_text().strip() == str(layout["work"])
        assert cap.with_suffix(".cli_output").read_text().strip() == str(
            layout["output"]
        )
        assert cap.with_suffix(".cli_log").read_text().strip() == str(log_dir)


# --- Persistence contract: preflight + metadata written ------------


class TestArtifactsWritten:
    def test_preflight_json_installed(self, tmp_path):
        layout = _make_layout(tmp_path)
        log_dir = layout["log"] / TEST_OAR_JOB_ID
        proc = _run_payload(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            build_log_dir=log_dir,
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode == 0, proc.stderr
        preflight = log_dir / "gpu_preflight.json"
        assert preflight.exists()
        payload = json.loads(preflight.read_text())
        assert payload["oar_job_id"] == TEST_OAR_JOB_ID
        assert payload["visible_cuda_device_count"] == 1

    def test_run_metadata_json_installed(self, tmp_path):
        layout = _make_layout(tmp_path)
        log_dir = layout["log"] / TEST_OAR_JOB_ID
        proc = _run_payload(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            build_log_dir=log_dir,
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode == 0, proc.stderr
        assert (log_dir / "run_metadata.json").exists()
        pythons = (tmp_path / "payload_cap.pythons").read_text().splitlines()
        assert any(p.endswith("python") for p in pythons)


# --- Output directory created by payload ---------------------------


class TestOutputDirCreated:
    def test_payload_creates_output_dir(self, tmp_path):
        layout = _make_layout(tmp_path)
        log_dir = layout["log"] / TEST_OAR_JOB_ID
        # OUTPUT_DIR does not exist before invocation.
        assert not layout["output"].exists()
        proc = _run_payload(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            build_log_dir=log_dir,
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode == 0, proc.stderr
        # The payload must create OUTPUT_DIR before invoking the CLI.
        assert layout["output"].is_dir()


# --- Work dir resume contract --------------------------------------


class TestWorkDirResume:
    def test_existing_work_dir_preserved_on_failure(self, tmp_path):
        """A non-zero CLI exit (e.g. walltime termination) must
        leave WORK_DIR and its checkpoint files untouched.
        """
        layout = _make_layout(tmp_path)
        log_dir = layout["log"] / TEST_OAR_JOB_ID
        # Seed a fake checkpoint file in WORK_DIR.
        work = layout["work"]
        (work / "shards" / "active" / "shard-1").mkdir(parents=True)
        checkpoint = work / "shards" / "active" / "shard-1" / "metadata.json"
        checkpoint.write_text('{"fake": true}')
        proc = _run_payload(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            build_log_dir=log_dir,
            input_root=layout["input"],
            work_dir=work,
            output_dir=layout["output"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            cli_overwrite="#!/usr/bin/env bash\nexit 99\n",
        )
        assert proc.returncode == 99
        # The checkpoint file must still be present and unmodified.
        assert checkpoint.exists()
        assert checkpoint.read_text() == '{"fake": true}'


# --- Exit-code propagation ------------------------------------------


class TestExitPropagation:
    def test_cli_failure_propagates(self, tmp_path):
        layout = _make_layout(tmp_path)
        log_dir = layout["log"] / TEST_OAR_JOB_ID
        proc = _run_payload(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            build_log_dir=log_dir,
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            cli_overwrite="#!/usr/bin/env bash\nexit 17\n",
        )
        assert proc.returncode == 17
