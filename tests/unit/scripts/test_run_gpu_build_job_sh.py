"""Contract tests for the Grid'5000 non-interactive batch entrypoint
for the *full resumable build* (Phase 9L-B).

``scripts/grid5000/run_gpu_build_job.sh`` is the *compute-node
wrapper*. It is invoked by ``oarsub`` inside an allocated OAR job
(after ``OAR_JOB_ID`` is set). It does NOT submit a job; it is the
job payload.

It mirrors the safety contract of ``run_gpu_smoke_job.sh`` and adds
the local-input, output-dir and work-dir contracts from Phase 9L-A/B.

Eight positional arguments:

    $1 = REPO_ROOT
    $2 = HF_HOME
    $3 = LOG_ROOT
    $4 = INPUT_ROOT
    $5 = WORK_DIR
    $6 = OUTPUT_DIR
    $7 = EXPECTED_SOURCE_COMMIT (40 lowercase hex)
    $8 = INPUT_REVISION          (40 lowercase hex; never "main")
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
JOB_SCRIPT = ROOT / "scripts" / "grid5000" / "run_gpu_build_job.sh"


def _test_sha(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8"), usedforsecurity=False).hexdigest()


TEST_SOURCE_COMMIT = _test_sha("tests/grid5000/run_gpu_build_job/source_commit")
TEST_INPUT_REVISION = _test_sha("tests/grid5000/run_gpu_build_job/input_revision")
TEST_OAR_JOB_ID = "1234567890"


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
    "oardel",
    "oarhold",
    "oarresume",
    "oarsub",
    # No publishing flags in the build payload.
    "--publish-dataset-id",
    "--publish-revision",
    "--publish-commit-message",
    # Build must use --input-root, never --input-dataset-id.
    "--input-dataset-id ",
)


@pytest.fixture(scope="module")
def script_text() -> str:
    return JOB_SCRIPT.read_text(encoding="utf-8")


# --- Static structure checks ----------------------------------------


def test_job_script_exists_and_is_nonempty():
    assert JOB_SCRIPT.exists()
    assert len(JOB_SCRIPT.read_text(encoding="utf-8")) > 150


def test_job_script_passes_bash_syntax_check():
    proc = subprocess.run(
        ["bash", "-n", str(JOB_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_job_script_uses_strict_bash_modes(script_text):
    assert "set -euo pipefail" in script_text


def test_job_script_has_no_forbidden_patterns(script_text):
    for pat in _FORBIDDEN_PATTERNS:
        assert pat not in script_text, f"forbidden pattern in job wrapper: {pat!r}"


def test_job_script_requires_oar_job_id(script_text):
    assert ": ${OAR_JOB_ID:?OAR_JOB_ID is required" in script_text or (
        "OAR_JOB_ID" in script_text and ":?" in script_text
    )


def test_job_script_does_not_touch_cuda_visible_devices(script_text):
    assert "CUDA_VISIBLE_DEVICES=" not in script_text


def test_job_script_documents_eight_positional_arguments(script_text):
    for n in range(1, 9):
        assert f"${n}" in script_text
    assert '"$#" -ne 8' in script_text or "[ $# -ne 8 ]" in script_text, (
        "expected guard requiring exactly eight positional arguments"
    )


def test_job_script_requires_commit_format(script_text):
    assert "[0-9a-f]{40}" in script_text


def test_job_script_requires_immutable_input_revision(script_text):
    assert "INPUT_REVISION" in script_text


def test_job_script_targets_build_payload(script_text):
    assert "run_gpu_build.sh" in script_text
    assert "run_gpu_smoke.sh" not in script_text


def test_job_script_rejects_ephemeral_storage(script_text):
    assert "/tmp" in script_text
    assert "/var/tmp" in script_text
    assert "/dev/shm" in script_text


def test_job_script_enforces_canonical_paths(script_text):
    assert "_canonicalise_directory" in script_text or "pwd -P" in script_text


def test_job_script_enforces_dir_mode_0700(script_text):
    assert "0700" in script_text


def test_job_script_enforces_file_mode_0600(script_text):
    assert "0600" in script_text


def test_job_script_documents_path_overlap_checks(script_text):
    # The wrapper must reject overlapping input/output/work paths.
    assert "overlap" in script_text.lower()


def test_job_script_documents_input_must_exist(script_text):
    assert "INPUT_ROOT" in script_text


def test_job_script_documents_output_must_be_fresh(script_text):
    # Output dir must NOT exist before submission (fresh build only).
    assert "OUTPUT_DIR" in script_text


def test_job_script_documents_work_may_exist_for_resume(script_text):
    # Work dir may pre-exist (resumable checkpoint contract).
    assert "WORK_DIR" in script_text


# --- Runtime behaviour with a fake payload --------------------------


def _make_fake_payload(tmp_path: Path) -> Path:
    """Create a fake build payload that records its eight
    arguments + the environment it received.
    """
    repo = tmp_path / "fake_repo"
    scripts_dir = repo / "scripts" / "grid5000"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    payload = scripts_dir / "run_gpu_build.sh"
    payload.write_text(
        "#!/usr/bin/env bash\n"
        'cap="${PAYLOAD_CAPTURE:?PAYLOAD_CAPTURE required}"\n'
        'printf "%s\\n" "${OAR_JOB_ID:-}" > "${cap}.oar_job_id"\n'
        'printf "%s\\n" "${REPO_ROOT:-}" > "${cap}.repo_root"\n'
        'printf "%s\\n" "${HF_HOME:-}" > "${cap}.hf_home"\n'
        'printf "%s\\n" "${BUILD_LOG_DIR:-}" > "${cap}.build_log_dir"\n'
        'printf "%s\\n" "${INPUT_ROOT:-}" > "${cap}.input_root"\n'
        'printf "%s\\n" "${WORK_DIR:-}" > "${cap}.work_dir"\n'
        'printf "%s\\n" "${OUTPUT_DIR:-}" > "${cap}.output_dir"\n'
        'printf "%s\\n" "${EXPECTED_SOURCE_COMMIT:-}" > "${cap}.expected"\n'
        'printf "%s\\n" "${INPUT_REVISION:-}" > "${cap}.input_revision"\n'
        'touch "${cap}.ran"\n'
        "exit 0\n"
    )
    payload.chmod(0o755)
    return payload


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
    # OUTPUT_DIR is NOT created: the wrapper requires it to be fresh.
    return {
        "repo": repo,
        "hf": hf,
        "log": log,
        "input": inp,
        "work": work,
        "output": out,
    }


def _seed_repo(repo_root: Path) -> str:
    """Seed a git repo and return its HEAD commit SHA."""
    subprocess.run(
        ["git", "init", "-q", str(repo_root)], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "config", "user.email", "t@e"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "config", "user.name", "T"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "commit", "--allow-empty", "-q", "-m", "init"],
        check=True,
        capture_output=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _run_job(
    tmp_path: Path,
    *,
    repo_root: Path,
    hf_home: Path,
    log_root: Path,
    input_root: Path,
    work_dir: Path,
    output_dir: Path,
    input_revision: str,
    oar_job_id: str = TEST_OAR_JOB_ID,
    payload_present: bool = True,
    payload_executable: bool = True,
    interpreter_present: bool = True,
    interpreter_executable: bool = True,
    existing_log_dir: bool = False,
    existing_output_dir: bool = False,
    existing_work_dir_files: bool = False,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Invoke the job wrapper. Source commit is derived from
    the repo's HEAD after fixtures are committed. Tests that
    need to inject a custom source commit (or intentionally
    break the git HEAD check) should call ``_seed_repo`` and
    pass an invalid commit; this helper commits the staged
    fixtures so the working tree is clean.
    """
    payload = _make_fake_payload(tmp_path)
    target_payload = Path(repo_root) / "scripts" / "grid5000" / "run_gpu_build.sh"
    if payload_present:
        target_payload.parent.mkdir(parents=True, exist_ok=True)
        target_payload.write_text(payload.read_text())
        target_payload.chmod(0o755 if payload_executable else 0o644)

    if interpreter_present:
        venv_bin = Path(repo_root) / ".venv" / "bin"
        venv_bin.mkdir(parents=True, exist_ok=True)
        py = venv_bin / "python"
        py.write_text("#!/usr/bin/env bash\nexit 0\n")
        py.chmod(0o755 if interpreter_executable else 0o644)

    # The job wrapper enforces a clean working tree. The freshly
    # staged payload + interpreter must be committed before the
    # run so git status --porcelain is empty.
    subprocess.run(
        ["git", "-C", str(repo_root), "add", "-A"], check=False, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "commit", "-q", "-m", "stage-test-fixtures"],
        check=False,
        capture_output=True,
    )

    job_log_dir = Path(log_root) / oar_job_id
    if existing_log_dir:
        job_log_dir.mkdir(parents=True, exist_ok=True)

    if existing_output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    if existing_work_dir_files:
        # Drop a fake checkpoint file inside WORK_DIR to simulate
        # resume state.
        (work_dir / "shards").mkdir(parents=True, exist_ok=True)
        (work_dir / "shards" / "active").mkdir(parents=True, exist_ok=True)
        (work_dir / "shards" / "active" / "fake-shard").mkdir(
            parents=True, exist_ok=True
        )
        (work_dir / "shards" / "active" / "fake-shard" / "metadata.json").write_text(
            "{}"
        )

    run_env = {
        **os.environ,
        "OAR_JOB_ID": oar_job_id,
        "PAYLOAD_CAPTURE": str(tmp_path / "payload_cap"),
        "PATH": os.environ.get("PATH", ""),
    }

    # Always pass the current HEAD as EXPECTED_SOURCE_COMMIT so the
    # wrapper's HEAD check passes. Tests that intentionally inject
    # a mismatched commit invoke the script directly via subprocess.
    current_head = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    args = [
        "bash",
        str(JOB_SCRIPT),
        str(repo_root),
        str(hf_home),
        str(log_root),
        str(input_root),
        str(work_dir),
        str(output_dir),
        current_head,
        input_revision,
    ]
    if extra_args:
        args.extend(extra_args)

    return subprocess.run(
        args,
        cwd=str(tmp_path),
        env=run_env,
        capture_output=True,
        text=True,
        timeout=30,
    )


# --- Argument-count guard --------------------------------------------


class TestArgumentCount:
    def test_three_args_fails(self, tmp_path):
        proc = subprocess.run(
            ["bash", str(JOB_SCRIPT), "/tmp/a", "/tmp/b", "/tmp/c"],
            cwd=str(tmp_path),
            env={**os.environ, "OAR_JOB_ID": TEST_OAR_JOB_ID},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode != 0
        assert "exactly eight positional arguments" in proc.stderr

    def test_nine_args_fails(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        proc = _run_job(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            input_revision=TEST_INPUT_REVISION,
            extra_args=["extra-positional"],
        )
        assert proc.returncode != 0


# --- Scheduler variables --------------------------------------------


class TestSchedulerVariables:
    def test_missing_oar_job_id_fails(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        proc = subprocess.run(
            [
                "bash",
                str(JOB_SCRIPT),
                str(layout["repo"]),
                str(layout["hf"]),
                str(layout["log"]),
                str(layout["input"]),
                str(layout["work"]),
                str(layout["output"]),
                TEST_SOURCE_COMMIT,
                TEST_INPUT_REVISION,
            ],
            cwd=str(tmp_path),
            env={**os.environ, "PATH": os.environ.get("PATH", "")},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode != 0
        assert "OAR_JOB_ID" in proc.stderr

    def test_non_numeric_oar_job_id_fails(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        proc = _run_job(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            input_revision=TEST_INPUT_REVISION,
            oar_job_id="not-numeric",
        )
        assert proc.returncode != 0
        assert "OAR_JOB_ID" in proc.stderr


# --- Path guards -----------------------------------------------------


class TestPathGuards:
    def test_relative_repo_root_rejected(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        proc = subprocess.run(
            [
                "bash",
                str(JOB_SCRIPT),
                "relative/path",
                str(layout["hf"]),
                str(layout["log"]),
                str(layout["input"]),
                str(layout["work"]),
                str(layout["output"]),
                TEST_SOURCE_COMMIT,
                TEST_INPUT_REVISION,
            ],
            cwd=str(tmp_path),
            env={
                **os.environ,
                "OAR_JOB_ID": TEST_OAR_JOB_ID,
                "PATH": os.environ.get("PATH", ""),
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode != 0
        assert "REPO_ROOT" in proc.stderr

    def test_ephemeral_work_dir_rejected(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        proc = _run_job(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=layout["input"],
            work_dir=Path("/dev/shm/work"),
            output_dir=layout["output"],
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode != 0
        assert "WORK_DIR" in proc.stderr

    def test_ephemeral_output_dir_rejected(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        proc = _run_job(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=Path("/tmp/output"),
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode != 0
        assert "OUTPUT_DIR" in proc.stderr

    def test_ephemeral_input_root_rejected(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        proc = _run_job(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=Path("/tmp/input"),
            work_dir=layout["work"],
            output_dir=layout["output"],
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode != 0
        assert "INPUT_ROOT" in proc.stderr

    def test_existing_job_log_dir_rejected(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        proc = _run_job(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            input_revision=TEST_INPUT_REVISION,
            existing_log_dir=True,
        )
        assert proc.returncode != 0
        assert (
            "refusing to reuse" in proc.stderr.lower()
            or "job log" in proc.stderr.lower()
        )


# --- Input/output/work relationship guards --------------------------


class TestPathRelationships:
    def test_reused_output_dir_rejected(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        proc = _run_job(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            input_revision=TEST_INPUT_REVISION,
            existing_output_dir=True,
        )
        assert proc.returncode != 0
        assert "OUTPUT_DIR" in proc.stderr

    def test_overlap_output_under_work_rejected(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        # OUTPUT_DIR nested under WORK_DIR
        overlap_out = layout["work"] / "nested_output"
        proc = _run_job(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=overlap_out,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode != 0
        assert "overlap" in proc.stderr.lower() or "OUTPUT_DIR" in proc.stderr

    def test_overlap_work_under_input_rejected(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        # WORK_DIR nested under INPUT_ROOT
        overlap_work = layout["input"] / "nested_work"
        overlap_work.mkdir(parents=True, exist_ok=True)
        proc = _run_job(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=layout["input"],
            work_dir=overlap_work,
            output_dir=layout["output"],
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode != 0
        assert "overlap" in proc.stderr.lower() or "WORK_DIR" in proc.stderr

    def test_existing_work_dir_accepted_for_resume(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        proc = _run_job(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            input_revision=TEST_INPUT_REVISION,
            existing_work_dir_files=True,
        )
        # WORK_DIR with pre-existing checkpoint files is the resume
        # case; the wrapper must accept it.
        assert proc.returncode == 0, proc.stderr


# --- Commit / revision format guard --------------------------------


class TestCommitAndRevisionGuard:
    def test_nonhex_commit_rejected(self, tmp_path):
        layout = _make_layout(tmp_path)
        # Call the wrapper directly so we can inject a non-hex
        # source commit without _run_job auto-correcting it.
        proc = subprocess.run(
            [
                "bash",
                str(JOB_SCRIPT),
                str(layout["repo"]),
                str(layout["hf"]),
                str(layout["log"]),
                str(layout["input"]),
                str(layout["work"]),
                str(layout["output"]),
                "Z" * 40,
                TEST_INPUT_REVISION,
            ],
            cwd=str(tmp_path),
            env={
                **os.environ,
                "OAR_JOB_ID": TEST_OAR_JOB_ID,
                "PATH": os.environ.get("PATH", ""),
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode != 0
        assert "EXPECTED_SOURCE_COMMIT" in proc.stderr

    def test_main_input_revision_rejected(self, tmp_path):
        layout = _make_layout(tmp_path)
        head = _seed_repo(layout["repo"])
        # Call the wrapper directly so we can inject "main" as the
        # input revision without _run_job auto-correcting it.
        proc = subprocess.run(
            [
                "bash",
                str(JOB_SCRIPT),
                str(layout["repo"]),
                str(layout["hf"]),
                str(layout["log"]),
                str(layout["input"]),
                str(layout["work"]),
                str(layout["output"]),
                head,
                "main",
            ],
            cwd=str(tmp_path),
            env={
                **os.environ,
                "OAR_JOB_ID": TEST_OAR_JOB_ID,
                "PATH": os.environ.get("PATH", ""),
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode != 0
        assert "INPUT_REVISION" in proc.stderr


# --- Payload guards --------------------------------------------------


class TestPayloadGuards:
    def test_missing_payload_rejected(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        proc = _run_job(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            input_revision=TEST_INPUT_REVISION,
            payload_present=False,
        )
        assert proc.returncode != 0
        assert "payload" in proc.stderr.lower() or "wrapper" in proc.stderr.lower()

    def test_non_executable_payload_rejected(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        # Stage the payload with non-executable mode and commit
        # the dirty tree so _run_job can pass the HEAD check.
        payload_target = layout["repo"] / "scripts" / "grid5000" / "run_gpu_build.sh"
        payload_target.parent.mkdir(parents=True, exist_ok=True)
        payload_target.write_text("#!/usr/bin/env bash\nexit 0\n")
        payload_target.chmod(0o644)
        subprocess.run(
            ["git", "-C", str(layout["repo"]), "add", "-A"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(layout["repo"]), "commit", "-q", "-m", "fixture"],
            check=True,
            capture_output=True,
        )
        new_head = subprocess.check_output(
            ["git", "-C", str(layout["repo"]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        proc = subprocess.run(
            [
                "bash",
                str(JOB_SCRIPT),
                str(layout["repo"]),
                str(layout["hf"]),
                str(layout["log"]),
                str(layout["input"]),
                str(layout["work"]),
                str(layout["output"]),
                new_head,
                TEST_INPUT_REVISION,
            ],
            cwd=str(tmp_path),
            env={
                **os.environ,
                "OAR_JOB_ID": TEST_OAR_JOB_ID,
                "PATH": os.environ.get("PATH", ""),
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode != 0


# --- Interpreter guards ---------------------------------------------


class TestInterpreterGuards:
    def test_missing_interpreter_rejected(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        proc = _run_job(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            input_revision=TEST_INPUT_REVISION,
            interpreter_present=False,
        )
        assert proc.returncode != 0
        assert "interpreter" in proc.stderr.lower() or "python" in proc.stderr.lower()


# --- Success path: payload is invoked with all eight args ------------


class TestPayloadInvocation:
    def test_payload_invoked_with_all_eight_args(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        proc = _run_job(
            tmp_path,
            repo_root=layout["repo"],
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode == 0, proc.stderr

        cap = tmp_path / "payload_cap"
        # _run_job commits staged fixtures before invoking the
        # wrapper, so EXPECTED_SOURCE_COMMIT is the current HEAD
        # (not the empty-commit HEAD returned by _seed_repo).
        final_head = subprocess.check_output(
            ["git", "-C", str(layout["repo"]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        assert cap.with_suffix(".ran").exists()
        assert cap.with_suffix(".oar_job_id").read_text().strip() == TEST_OAR_JOB_ID
        assert cap.with_suffix(".repo_root").read_text().strip() == str(layout["repo"])
        assert cap.with_suffix(".hf_home").read_text().strip() == str(layout["hf"])
        assert cap.with_suffix(".input_root").read_text().strip() == str(
            layout["input"]
        )
        assert cap.with_suffix(".work_dir").read_text().strip() == str(layout["work"])
        assert cap.with_suffix(".output_dir").read_text().strip() == str(
            layout["output"]
        )
        assert cap.with_suffix(".expected").read_text().strip() == final_head
        assert (
            cap.with_suffix(".input_revision").read_text().strip()
            == TEST_INPUT_REVISION
        )
        assert cap.with_suffix(".build_log_dir").read_text().strip() == str(
            layout["log"] / TEST_OAR_JOB_ID
        )


# --- Exit-code propagation ------------------------------------------


class TestExitPropagation:
    def test_payload_failure_propagates(self, tmp_path):
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        repo = layout["repo"]
        # Replace the payload with one that exits 42 and commit it
        # so the resulting HEAD is known and stable.
        target = repo / "scripts" / "grid5000" / "run_gpu_build.sh"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("#!/usr/bin/env bash\nexit 42\n")
        target.chmod(0o755)
        subprocess.run(
            ["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-q", "-m", "failing"],
            check=True,
            capture_output=True,
        )
        proc = _run_job(
            tmp_path,
            repo_root=repo,
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=layout["input"],
            work_dir=layout["work"],
            output_dir=layout["output"],
            input_revision=TEST_INPUT_REVISION,
            payload_present=False,
        )
        assert proc.returncode == 42

    def test_work_dir_preserved_after_failure(self, tmp_path):
        """A non-zero exit (e.g. walltime termination) must leave
        WORK_DIR untouched. A subsequent resume invocation can
        reuse the same WORK_DIR.
        """
        layout = _make_layout(tmp_path)
        _seed_repo(layout["repo"])
        repo = layout["repo"]
        # Seed a fake checkpoint file in WORK_DIR before the run.
        work = layout["work"]
        (work / "shards" / "active" / "shard-1").mkdir(parents=True)
        checkpoint = work / "shards" / "active" / "shard-1" / "metadata.json"
        checkpoint.write_text('{"fake": true}')
        # Replace the payload with one that exits non-zero.
        target = repo / "scripts" / "grid5000" / "run_gpu_build.sh"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("#!/usr/bin/env bash\nexit 99\n")
        target.chmod(0o755)
        subprocess.run(
            ["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-q", "-m", "failing"],
            check=True,
            capture_output=True,
        )
        proc = _run_job(
            tmp_path,
            repo_root=repo,
            hf_home=layout["hf"],
            log_root=layout["log"],
            input_root=layout["input"],
            work_dir=work,
            output_dir=layout["output"],
            input_revision=TEST_INPUT_REVISION,
            payload_present=False,
        )
        assert proc.returncode == 99
        # The checkpoint file must still be present.
        assert checkpoint.exists()
        assert checkpoint.read_text() == '{"fake": true}'
