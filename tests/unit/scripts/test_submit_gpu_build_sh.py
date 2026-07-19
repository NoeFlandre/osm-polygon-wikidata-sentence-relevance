"""Contract tests for the Grid'5000 non-interactive OAR submission
adapter for the *full resumable build* (Phase 9L-B).

The submission helper ``scripts/grid5000/submit_gpu_build.sh`` is a
frontend-only helper. It converts the eight required positional
arguments into exactly one command-string argument that Nancy's
``oarsub`` accepts.

Eight positional arguments (REPO_ROOT, HF_HOME, LOG_ROOT, INPUT_ROOT,
WORK_DIR, OUTPUT_DIR, EXPECTED_SOURCE_COMMIT, INPUT_REVISION):

    $1 = REPO_ROOT
    $2 = HF_HOME
    $3 = LOG_ROOT
    $4 = INPUT_ROOT
    $5 = WORK_DIR
    $6 = OUTPUT_DIR
    $7 = EXPECTED_SOURCE_COMMIT  (40 lowercase hex)
    $8 = INPUT_REVISION          (40 lowercase hex; never "main")

It never imports Python, never performs inference, never polls,
cancels, SSHes, mutates git, downloads, retries, or cleans up.

These tests run on the Mac with only a fake ``oarsub`` on PATH.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SUBMIT_SCRIPT = ROOT / "scripts" / "grid5000" / "submit_gpu_build.sh"


def _test_sha(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8"), usedforsecurity=False).hexdigest()


TEST_SOURCE_COMMIT = _test_sha("tests/grid5000/submit_gpu_build/source_commit")
TEST_INPUT_REVISION = _test_sha("tests/grid5000/submit_gpu_build/input_revision")


_FORBIDDEN_PATTERNS = (
    "oarsub -I",
    "--interactive",
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
    "oarstat",
    "oardel",
    "oarhold",
    "oarresume",
    # No publishing flags may appear in the build adapter.
    "--publish-dataset-id",
    "--publish-revision",
    "--publish-commit-message",
    # The build must always use --input-root; never --input-dataset-id.
    "--input-dataset-id ",
)


@pytest.fixture(scope="module")
def script_text() -> str:
    return SUBMIT_SCRIPT.read_text(encoding="utf-8")


# --- Static structure checks ----------------------------------------


def test_submit_script_exists_and_is_nonempty():
    assert SUBMIT_SCRIPT.exists()
    assert len(SUBMIT_SCRIPT.read_text(encoding="utf-8")) > 150


def test_submit_script_passes_bash_syntax_check():
    proc = subprocess.run(
        ["bash", "-n", str(SUBMIT_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_submit_script_uses_strict_bash_modes(script_text):
    assert "set -euo pipefail" in script_text


def test_submit_script_has_no_forbidden_patterns(script_text):
    for pat in _FORBIDDEN_PATTERNS:
        assert pat not in script_text, f"forbidden pattern in submit helper: {pat!r}"


def test_submit_script_documents_nine_positional_arguments(script_text):
    for n in range(1, 10):
        assert f"${n}" in script_text
    assert '"$#" -ne 9' in script_text or "[ $# -ne 9 ]" in script_text, (
        "expected guard requiring exactly nine positional arguments"
    )


def test_submit_script_requires_oarsub_on_path(script_text):
    assert "command -v oarsub" in script_text


def test_submit_script_does_not_export_scheduler_variables(script_text):
    assert "export CUDA_VISIBLE_DEVICES=" not in script_text
    assert "export OAR_JOB_ID=" not in script_text


def test_submit_script_uses_production_queue(script_text):
    assert "-q production" in script_text


def test_submit_script_passes_walltime_to_oarsub(script_text):
    assert "walltime=" in script_text


def test_submit_script_passes_gpu_count_to_oarsub(script_text):
    assert "gpu=1" in script_text


def test_submit_script_rejects_ephemeral_storage(script_text):
    assert "/tmp" in script_text
    assert "/var/tmp" in script_text
    assert "/dev/shm" in script_text


def test_submit_script_requires_commit_format(script_text):
    assert "[0-9a-f]{40}" in script_text


def test_submit_script_requires_immutable_input_revision(script_text):
    # INPUT_REVISION must be 40 lowercase hex; "main" must be
    # explicitly rejected.
    assert "INPUT_REVISION" in script_text
    # The script must reject "main" as an INPUT_REVISION.
    assert "main" in script_text


def test_submit_script_targets_build_wrapper(script_text):
    assert "run_gpu_build_job.sh" in script_text


def test_submit_script_documents_storage_agnosticism(script_text):
    assert "INPUT_ROOT" in script_text
    assert "OUTPUT_DIR" in script_text


def test_submit_script_walltime_is_proposed_pending_benchmark(script_text):
    # The walltime must not be claimed as validated; the comment
    # must say "proposed" or "pending benchmark".
    assert (
        "pending benchmark" in script_text.lower() or "proposed" in script_text.lower()
    ), "walltime must be described as proposed / pending benchmark"


# --- Fake oarsub capturing harness ----------------------------------


def _make_fake_oarsub(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "fake_bin"
    bin_dir.mkdir(exist_ok=True)
    oarsub = bin_dir / "oarsub"
    oarsub.write_text(
        "#!/usr/bin/env bash\n"
        'cap="${CAPTURE_FILE:?CAPTURE_FILE required}"\n'
        'printf "%s\\n" "$@" > "${cap}.argv"\n'
        'printf "%s\\n" "$#" > "${cap}.count"\n'
        'printf "%s\\n" "$#" >> "${cap}.invocations"\n'
        'touch "${cap}.ran"\n'
        'exit "${FAKE_OARSUB_RC:-0}"\n'
    )
    oarsub.chmod(0o755)
    return bin_dir


def _make_fake_compute_wrapper(tmp_path: Path) -> Path:
    repo = tmp_path / "fake_repo"
    repo.mkdir(parents=True, exist_ok=True)
    scripts_dir = repo / "scripts" / "grid5000"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    wrapper = scripts_dir / "run_gpu_build_job.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        'cap="${WRAPPER_CAPTURE:?WRAPPER_CAPTURE required}"\n'
        'printf "%s\\n" "$1" > "${cap}.arg1"\n'
        'printf "%s\\n" "$2" > "${cap}.arg2"\n'
        'printf "%s\\n" "$3" > "${cap}.arg3"\n'
        'printf "%s\\n" "$4" > "${cap}.arg4"\n'
        'printf "%s\\n" "$5" > "${cap}.arg5"\n'
        'printf "%s\\n" "$6" > "${cap}.arg6"\n'
        'printf "%s\\n" "$7" > "${cap}.arg7"\n'
        'printf "%s\\n" "$8" > "${cap}.arg8"\n'
        'printf "%s\\n" "$9" > "${cap}.arg9"\n'
        'printf "%s\\n" "$#" > "${cap}.count"\n'
        'touch "${cap}.ran"\n'
        "exit 0\n"
    )
    wrapper.chmod(0o755)
    return scripts_dir / "run_gpu_build_job.sh"


def _make_persistent_layout(tmp_path: Path) -> dict[str, Path]:
    """Create five canonical absolute persistent directories under a
    private temp root. INPUT_ROOT is also created. OUTPUT_DIR is
    NOT created: the submission adapter requires OUTPUT_DIR to be
    fresh (it does not need to exist; the operator creates it on
    the compute node).
    """
    base = tmp_path / "persistent"
    base.mkdir(parents=True, exist_ok=True)
    repo_root = base / "repo"
    hf_home = base / "hf_home"
    log_root = base / "logs"
    input_root = base / "input"
    work_dir = base / "work"
    output_dir = base / "output"
    for p in (repo_root, hf_home, log_root, input_root, work_dir):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "repo_root": repo_root,
        "hf_home": hf_home,
        "log_root": log_root,
        "input_root": input_root,
        "work_dir": work_dir,
        "output_dir": output_dir,
    }


def _run_submit(
    tmp_path: Path,
    *,
    repo_root: Path,
    hf_home: Path,
    log_root: Path,
    input_root: Path,
    work_dir: Path,
    output_dir: Path,
    source_commit: str,
    input_revision: str,
    batch_size: str = "128",
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    wrapper_present: bool = True,
    wrapper_executable: bool = True,
) -> subprocess.CompletedProcess:
    fake_bin = _make_fake_oarsub(tmp_path)
    capture = tmp_path / "oarsub_capture"
    if wrapper_present:
        wrapper = _make_fake_compute_wrapper(tmp_path)
        target = Path(repo_root) / "scripts" / "grid5000" / "run_gpu_build_job.sh"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(wrapper.read_text())
        target.chmod(0o755 if wrapper_executable else 0o644)

    run_env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "CAPTURE_FILE": str(capture),
    }
    if env:
        run_env.update(env)

    args = [
        "bash",
        str(SUBMIT_SCRIPT),
        str(repo_root),
        str(hf_home),
        str(log_root),
        str(input_root),
        str(work_dir),
        str(output_dir),
        source_commit,
        input_revision,
        batch_size,
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
    def test_zero_args_fails(self, tmp_path):
        proc = subprocess.run(
            ["bash", str(SUBMIT_SCRIPT)],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": os.environ.get("PATH", "")},
            timeout=30,
        )
        assert proc.returncode != 0
        assert "exactly nine positional arguments" in proc.stderr

    def test_three_args_fails(self, tmp_path):
        proc = subprocess.run(
            ["bash", str(SUBMIT_SCRIPT), "/tmp/a", "/tmp/b", "/tmp/c"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": os.environ.get("PATH", "")},
            timeout=30,
        )
        assert proc.returncode != 0

    def test_ten_args_fails(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            extra_args=["extra-positional"],
        )
        assert proc.returncode != 0


# --- Path guards -----------------------------------------------------


class TestPathGuards:
    def test_relative_repo_root_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        fake_bin = _make_fake_oarsub(tmp_path)
        proc = subprocess.run(
            [
                "bash",
                str(SUBMIT_SCRIPT),
                "relative/path",
                str(layout["hf_home"]),
                str(layout["log_root"]),
                str(layout["input_root"]),
                str(layout["work_dir"]),
                str(layout["output_dir"]),
                TEST_SOURCE_COMMIT,
                TEST_INPUT_REVISION,
                "128",
            ],
            cwd=str(tmp_path),
            env={
                **os.environ,
                "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode != 0
        assert "REPO_ROOT" in proc.stderr

    def test_ephemeral_hf_home_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=Path("/tmp/hf_home"),
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode != 0
        assert "HF_HOME" in proc.stderr

    def test_ephemeral_input_root_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=Path("/dev/shm/input"),
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode != 0
        assert "INPUT_ROOT" in proc.stderr

    def test_ephemeral_work_dir_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=Path("/var/tmp/work"),
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode != 0
        assert "WORK_DIR" in proc.stderr

    def test_ephemeral_output_dir_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=Path("/tmp/output"),
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode != 0
        assert "OUTPUT_DIR" in proc.stderr

    def test_traversal_in_path_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        traversal_repo = tmp_path / "ok_parent" / ".." / "evil"
        proc = _run_submit(
            tmp_path,
            repo_root=traversal_repo,
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode != 0


# --- Commit / revision format guard ---------------------------------


class TestCommitAndRevisionGuard:
    def test_nonhex_commit_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit="Z" * 40,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode != 0
        assert "EXPECTED_SOURCE_COMMIT" in proc.stderr

    def test_39_char_commit_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT[:39],
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode != 0

    def test_main_input_revision_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision="main",
        )
        assert proc.returncode != 0
        assert "INPUT_REVISION" in proc.stderr

    def test_nonhex_input_revision_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision="Z" * 40,
        )
        assert proc.returncode != 0
        assert "INPUT_REVISION" in proc.stderr

    def test_39_char_input_revision_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION[:39],
        )
        assert proc.returncode != 0


# --- Batch-size guards (Phase 9M-B) -----------------------------------


class TestBatchSizeGuard:
    def test_zero_batch_size_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            batch_size="0",
        )
        assert proc.returncode != 0
        assert "BATCH_SIZE" in proc.stderr

    def test_negative_batch_size_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            batch_size="-1",
        )
        assert proc.returncode != 0
        assert "BATCH_SIZE" in proc.stderr

    def test_decimal_batch_size_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            batch_size="12.5",
        )
        assert proc.returncode != 0
        assert "BATCH_SIZE" in proc.stderr

    def test_bool_true_batch_size_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            batch_size="true",
        )
        assert proc.returncode != 0
        assert "BATCH_SIZE" in proc.stderr

    def test_bool_false_batch_size_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            batch_size="False",
        )
        assert proc.returncode != 0
        assert "BATCH_SIZE" in proc.stderr

    def test_leading_zero_batch_size_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            batch_size="0128",
        )
        assert proc.returncode != 0
        assert "BATCH_SIZE" in proc.stderr

    def test_non_integer_batch_size_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            batch_size="abc",
        )
        assert proc.returncode != 0
        assert "BATCH_SIZE" in proc.stderr

    def test_positive_integer_batch_size_accepted(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            batch_size="16",
        )
        assert proc.returncode == 0, proc.stderr


# --- Wrapper-presence guards -----------------------------------------


class TestWrapperGuards:
    def test_missing_wrapper_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            wrapper_present=False,
        )
        assert proc.returncode != 0
        assert "wrapper" in proc.stderr.lower()

    def test_non_executable_wrapper_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            wrapper_executable=False,
        )
        assert proc.returncode != 0


# --- oarsub-presence guard -------------------------------------------


class TestOarsubGuard:
    def test_missing_oarsub_rejected(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        wrapper = _make_fake_compute_wrapper(tmp_path)
        target = (
            Path(layout["repo_root"]) / "scripts" / "grid5000" / "run_gpu_build_job.sh"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(wrapper.read_text())
        target.chmod(0o755)

        empty_bin = tmp_path / "empty_bin"
        empty_bin.mkdir()
        system_path = ":".join(
            p
            for p in os.environ.get("PATH", "").split(":")
            if "/usr/bin" in p or "/bin" in p
        )
        proc = subprocess.run(
            [
                "bash",
                str(SUBMIT_SCRIPT),
                str(layout["repo_root"]),
                str(layout["hf_home"]),
                str(layout["log_root"]),
                str(layout["input_root"]),
                str(layout["work_dir"]),
                str(layout["output_dir"]),
                TEST_SOURCE_COMMIT,
                TEST_INPUT_REVISION,
                "128",
            ],
            cwd=str(tmp_path),
            env={
                **os.environ,
                "PATH": f"{empty_bin}:{system_path}",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode != 0
        assert "oarsub" in proc.stderr.lower()


# --- Single-submission contract (no retry) ---------------------------


class TestExactlyOnceSubmission:
    def test_oarsub_invoked_exactly_once(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        invocations = (tmp_path / "oarsub_capture.invocations").read_text().splitlines()
        assert len(invocations) == 1, invocations

    def test_oarsub_receives_exactly_one_positional_command_string(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        argv = (tmp_path / "oarsub_capture.argv").read_text().splitlines()
        opts_with_values = {"-q", "-l"}
        positionals = []
        i = 0
        while i < len(argv):
            if argv[i] in opts_with_values:
                i += 2
                continue
            if argv[i].startswith("-") and argv[i] != "-":
                i += 1
                continue
            positionals.append(argv[i])
            i += 1
        assert len(positionals) == 1, (
            f"oarsub must receive exactly one positional command string, "
            f"got {positionals!r} from {argv!r}"
        )
        assert "run_gpu_build_job.sh" in positionals[0]

    def test_oarsub_receives_resource_request(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        argv = (tmp_path / "oarsub_capture.argv").read_text().splitlines()
        assert "-q" in argv
        q_idx = argv.index("-q")
        assert argv[q_idx + 1] == "production"
        assert "-l" in argv
        l_idx = argv.index("-l")
        assert argv[l_idx + 1].startswith("gpu=1,walltime="), argv

    def test_oarsub_exit_propagates(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        fake_bin = _make_fake_oarsub(tmp_path)
        wrapper = _make_fake_compute_wrapper(tmp_path)
        target = (
            Path(layout["repo_root"]) / "scripts" / "grid5000" / "run_gpu_build_job.sh"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(wrapper.read_text())
        target.chmod(0o755)
        proc = subprocess.run(
            [
                "bash",
                str(SUBMIT_SCRIPT),
                str(layout["repo_root"]),
                str(layout["hf_home"]),
                str(layout["log_root"]),
                str(layout["input_root"]),
                str(layout["work_dir"]),
                str(layout["output_dir"]),
                TEST_SOURCE_COMMIT,
                TEST_INPUT_REVISION,
                "128",
            ],
            cwd=str(tmp_path),
            env={
                **os.environ,
                "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
                "CAPTURE_FILE": str(tmp_path / "oarsub_capture"),
                "FAKE_OARSUB_RC": "37",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 37


# --- Serialization contract: the eight arguments survive intact ----


class TestNineArgSerialization:
    def test_nine_args_survive_quoting(self, tmp_path):
        layout = _make_persistent_layout(tmp_path)
        _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
            batch_size="128",
        )
        cmd = (tmp_path / "oarsub_capture.argv").read_text().splitlines()[-1]
        wrapper_cap_prefix = tmp_path / "wrapper_cap"
        run_env = {
            **os.environ,
            "WRAPPER_CAPTURE": str(wrapper_cap_prefix),
            "PATH": f"{tmp_path / 'fake_bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        }
        res = subprocess.run(
            ["bash", "-c", cmd],
            cwd=str(layout["repo_root"]),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0, (
            f"decode failed: rc={res.returncode}, stderr={res.stderr}, cmd={cmd!r}"
        )
        # The fake wrapper writes ${WRAPPER_CAPTURE}.arg1, etc.
        assert wrapper_cap_prefix.with_suffix(".arg1").read_text().strip() == str(
            layout["repo_root"]
        )
        assert wrapper_cap_prefix.with_suffix(".arg2").read_text().strip() == str(
            layout["hf_home"]
        )
        assert wrapper_cap_prefix.with_suffix(".arg3").read_text().strip() == str(
            layout["log_root"]
        )
        assert wrapper_cap_prefix.with_suffix(".arg4").read_text().strip() == str(
            layout["input_root"]
        )
        assert wrapper_cap_prefix.with_suffix(".arg5").read_text().strip() == str(
            layout["work_dir"]
        )
        assert wrapper_cap_prefix.with_suffix(".arg6").read_text().strip() == str(
            layout["output_dir"]
        )
        assert (
            wrapper_cap_prefix.with_suffix(".arg7").read_text().strip()
            == TEST_SOURCE_COMMIT
        )
        assert (
            wrapper_cap_prefix.with_suffix(".arg8").read_text().strip()
            == TEST_INPUT_REVISION
        )
        assert wrapper_cap_prefix.with_suffix(".arg9").read_text().strip() == "128"

    def test_special_chars_in_path_preserve(self, tmp_path):
        """A path containing a single quote must still be delivered
        verbatim to the compute-node wrapper.
        """
        layout = _make_persistent_layout(tmp_path)
        tricky = tmp_path / "persistent" / "hf'odd"
        try:
            tricky.mkdir(parents=True, exist_ok=False)
        except OSError:
            pytest.skip("filesystem refuses single quote in path")
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=tricky,
            log_root=layout["log_root"],
            input_root=layout["input_root"],
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode == 0
        cmd = (tmp_path / "oarsub_capture.argv").read_text().splitlines()[-1]
        wrapper_cap_prefix = tmp_path / "wrapper_cap"
        run_env = {
            **os.environ,
            "WRAPPER_CAPTURE": str(wrapper_cap_prefix),
            "PATH": f"{tmp_path / 'fake_bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        }
        res = subprocess.run(
            ["bash", "-c", cmd],
            cwd=str(layout["repo_root"]),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0, (
            f"decode failed: rc={res.returncode}, stderr={res.stderr}, cmd={cmd!r}"
        )
        assert wrapper_cap_prefix.with_suffix(".arg2").read_text().strip() == str(
            tricky
        )

    def test_injection_in_path_does_not_execute(self, tmp_path):
        """A path containing shell metacharacters must remain literal
        and must NOT execute. This is the canonical hostile-input
        safety test for the single-quoted command string.
        """
        layout = _make_persistent_layout(tmp_path)
        hostile = tmp_path / "persistent" / "evil$(touch pwned)"
        hostile.mkdir(parents=True, exist_ok=False)
        proc = _run_submit(
            tmp_path,
            repo_root=layout["repo_root"],
            hf_home=layout["hf_home"],
            log_root=layout["log_root"],
            input_root=hostile,
            work_dir=layout["work_dir"],
            output_dir=layout["output_dir"],
            source_commit=TEST_SOURCE_COMMIT,
            input_revision=TEST_INPUT_REVISION,
        )
        assert proc.returncode == 0
        # Decode the command string through a real shell; the
        # injection marker must never be created.
        cmd = (tmp_path / "oarsub_capture.argv").read_text().splitlines()[-1]
        wrapper_cap_prefix = tmp_path / "wrapper_cap"
        run_env = {
            **os.environ,
            "WRAPPER_CAPTURE": str(wrapper_cap_prefix),
            "PATH": f"{tmp_path / 'fake_bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        }
        subprocess.run(
            ["bash", "-c", cmd],
            cwd=str(layout["repo_root"]),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert not (layout["repo_root"] / "pwned").exists(), (
            "shell injection in INPUT_ROOT executed; quoting failed"
        )
        assert wrapper_cap_prefix.with_suffix(".arg4").read_text().strip() == str(
            hostile
        )
