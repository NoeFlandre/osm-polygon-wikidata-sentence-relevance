"""Contract tests for the Grid'5000 non-interactive OAR submission
adapter (Phase 9F).

The submission helper ``scripts/grid5000/submit_gpu_smoke.sh`` is a
frontend-only helper. It converts the four required positional
arguments into exactly one command-string argument that Nancy's
``oarsub`` accepts (this OAR build rejects positional arguments after
the script path). It never imports Python, never performs inference,
never polls, cancels, SSHes, mutates git, downloads, retries, or
cleans up.

These tests run on the Mac with only a fake ``oarsub`` on PATH. They
never:

- execute real OAR submissions;
- run on the Grid'5000 frontend;
- construct SaT, download weights, or perform inference;
- import Torch, send network traffic, or mutate git state.

All fakes live under ``tmp_path``. Production paths are referenced
only as canonical documentation; no real account, job id, or SHA is
embedded in this file.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SUBMIT_SCRIPT = ROOT / "scripts" / "grid5000" / "submit_gpu_smoke.sh"
DOC = ROOT / "docs" / "guides" / "grid5000.md"


# --- SHA fixtures (deterministic, no opaque literal) ----------------


def _test_sha(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8"), usedforsecurity=False).hexdigest()


TEST_SOURCE_COMMIT = _test_sha("tests/grid5000/submit_gpu_smoke/source_commit")
TEST_NONHEX_BASE = _test_sha("tests/grid5000/submit_gpu_smoke/nonhex_base")


def _with_nonhex_char(digest: str, position: int = 0) -> str:
    chars = list(digest)
    chars[position] = "g"
    return "".join(chars)


# --- Forbidden patterns in the submission helper --------------------


_FORBIDDEN_PATTERNS = (
    "oarsub -I",
    "--interactive",
    "eval ",
    "CUDA_VISIBLE_DEVICES:=",
    "OAR_JOB_ID:=",
    'device="auto"',
    'device="cpu"',
    'device="mps"',
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
    "python3 ",
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


def test_submit_script_documents_four_positional_arguments(script_text):
    assert "$1" in script_text
    assert "$2" in script_text
    assert "$3" in script_text
    assert "$4" in script_text
    # Must require exactly four positional arguments.
    assert (
        '"$#" -ne 4' in script_text
        or "[ $# -ne 4 ]" in script_text
        or all(p in script_text for p in ("${1:?", "${2:?", "${3:?", "${4:?"))
    ), "expected guard requiring exactly four positional arguments"


def test_submit_script_requires_oarsub_on_path(script_text):
    # Must require oarsub to be discoverable before invoking it.
    assert "command -v oarsub" in script_text


def test_submit_script_does_not_export_scheduler_variables(script_text):
    assert "export CUDA_VISIBLE_DEVICES=" not in script_text
    assert "export OAR_JOB_ID=" not in script_text


# --- Fake oarsub capturing harness ----------------------------------


def _make_fake_oarsub(tmp_path: Path) -> Path:
    """Create a fake ``oarsub`` on a private bin dir that records
    the exact argv it received and returns a configurable exit code.
    The fake never touches the scheduler/network.

    Returns the directory containing the fake ``oarsub`` (to be
    prepended to PATH).
    """
    bin_dir = tmp_path / "fake_bin"
    bin_dir.mkdir(exist_ok=True)
    oarsub = bin_dir / "oarsub"
    oarsub.write_text(
        "#!/usr/bin/env bash\n"
        'cap="${CAPTURE_FILE:?CAPTURE_FILE required}"\n'
        "# Record the exact argv (excluding argv[0]) one token per line.\n"
        'printf "%s\\n" "$@" > "${cap}.argv"\n'
        'printf "%s\\n" "$#" > "${cap}.count"\n'
        "# Append one invocation record per process call so the test can\n"
        "# assert exactly-once submission (no retry).\n"
        'printf "%s\\n" "$#" >> "${cap}.invocations"\n'
        'touch "${cap}.ran"\n'
        'exit "${FAKE_OARSUB_RC:-0}"\n'
    )
    oarsub.chmod(0o755)
    return bin_dir


def _make_fake_compute_wrapper(tmp_path: Path) -> Path:
    """Create a fake compute-node wrapper that records the exact
    four arguments it receives. This stands in for
    ``run_gpu_smoke_job.sh`` inside the single command string; the
    test executes the captured command string to prove the four
    original arguments survive serialization intact.
    """
    repo = tmp_path / "fake_repo"
    repo.mkdir(parents=True, exist_ok=True)
    scripts_dir = repo / "scripts" / "grid5000"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    wrapper = scripts_dir / "run_gpu_smoke_job.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        'cap="${WRAPPER_CAPTURE:?WRAPPER_CAPTURE required}"\n'
        'printf "%s\\n" "$1" > "${cap}.arg1"\n'
        'printf "%s\\n" "$2" > "${cap}.arg2"\n'
        'printf "%s\\n" "$3" > "${cap}.arg3"\n'
        'printf "%s\\n" "$4" > "${cap}.arg4"\n'
        'printf "%s\\n" "$#" > "${cap}.count"\n'
        'touch "${cap}.ran"\n'
        "exit 0\n"
    )
    wrapper.chmod(0o755)
    return scripts_dir / "run_gpu_smoke_job.sh"


def _make_persistent_layout(
    tmp_path: Path,
    *,
    repo_root: Path | None = None,
    hf_home: Path | None = None,
    log_root: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Create canonical absolute persistent directories under a
    private temp root. Idempotent. Also seeds byte-exact 40-char
    refs/main files for the SaT model and XLM-R tokenizer so the
    pre-submission cache-ref validator passes (Phase 9M amendment).
    """
    base = tmp_path / "persistent"
    base.mkdir(parents=True, exist_ok=True)
    repo_root = repo_root or (base / "repo")
    hf_home = hf_home or (base / "hf_home")
    log_root = log_root or (base / "logs")
    repo_root.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    _seed_clean_hf_refs(hf_home)
    return repo_root, hf_home, log_root


def _seed_clean_hf_refs(hf_home: Path) -> None:
    """Write byte-exact 40-byte refs/main for the two cached repos."""
    import os

    pairs = [
        (
            "models--segment-any-text--sat-3l-sm",
            "137da054051ad9f1eac42025f758db4ac9f22535",
        ),
        (
            "models--facebookAI--xlm-roberta-base",
            "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        ),
    ]
    for slug, sha in pairs:
        refs_dir = hf_home / "hub" / slug / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        path = refs_dir / "main"
        # Use os.write for byte-exact 40-byte semantics (no newline).
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, sha.encode("ascii"))
        finally:
            os.close(fd)


def _ensure_target_wrapper(repo_root: Path, source: Path) -> None:
    target_wrapper = Path(repo_root) / "scripts" / "grid5000" / "run_gpu_smoke_job.sh"
    target_wrapper.parent.mkdir(parents=True, exist_ok=True)
    target_wrapper.write_text(source.read_text())
    target_wrapper.chmod(0o755)


def _run_submit(
    tmp_path: Path,
    *,
    repo_root: Path,
    hf_home: Path,
    log_root: Path,
    source_commit: str,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    wrapper_present: bool = True,
    wrapper_executable: bool = True,
) -> subprocess.CompletedProcess:
    """Run the submission helper with a fake ``oarsub`` on PATH and a
    project-local ``run_gpu_smoke_job.sh`` reachable from REPO_ROOT.

    ``wrapper_present=False`` omits the compute-node wrapper entirely
    (to test the missing-wrapper guard); ``wrapper_executable=False``
    writes it without the executable bit (to test the non-executable
    guard).
    """
    fake_bin = _make_fake_oarsub(tmp_path)
    capture = tmp_path / "oarsub_capture"
    # Always seed the cache-ref validator so the submit script can
    # source it, even on tests that intentionally omit the wrapper
    # (the validator check runs before the wrapper-presence check).
    validator_src = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "grid5000"
        / "_cache_ref_validator.sh"
    )
    validator_dst = Path(repo_root) / "scripts" / "grid5000" / "_cache_ref_validator.sh"
    if validator_src.exists():
        validator_dst.parent.mkdir(parents=True, exist_ok=True)
        validator_dst.write_text(validator_src.read_text())
        validator_dst.chmod(0o755)
    if wrapper_present:
        wrapper = _make_fake_compute_wrapper(tmp_path)
        target_wrapper = (
            Path(repo_root) / "scripts" / "grid5000" / "run_gpu_smoke_job.sh"
        )
        target_wrapper.parent.mkdir(parents=True, exist_ok=True)
        target_wrapper.write_text(wrapper.read_text())
        mode = 0o755 if wrapper_executable else 0o644
        target_wrapper.chmod(mode)

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
        source_commit,
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


def _decode_capture(
    tmp_path: Path, repo_root: Path, wrapper_cap_name: str
) -> subprocess.CompletedProcess:
    """Execute the single captured command string through a real
    shell with the fake oarsub on PATH and a fresh wrapper capture
    target; returns the decode subprocess."""
    cmd = (tmp_path / "oarsub_capture.argv").read_text().splitlines()[-1]
    wrapper_cap = tmp_path / wrapper_cap_name
    run_env = {
        **os.environ,
        "WRAPPER_CAPTURE": str(wrapper_cap),
        "PATH": f"{tmp_path / 'fake_bin'}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    return subprocess.run(
        ["bash", "-c", cmd],
        cwd=str(repo_root),
        env=run_env,
        capture_output=True,
        text=True,
        timeout=30,
    )


# --- Core serialization contract ------------------------------------


def test_fake_oarsub_invoked_exactly_once(tmp_path):
    repo, hf, log = _make_persistent_layout(tmp_path)
    proc = _run_submit(
        tmp_path,
        repo_root=repo,
        hf_home=hf,
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    capture = tmp_path / "oarsub_capture"
    assert (capture.with_suffix(".ran")).exists(), "fake oarsub did not run"
    assert capture.with_suffix(".argv").exists()
    # Exactly one oarsub process must have run (no retry). The
    # invocation record file has one line per process call.
    invocations = (capture.with_suffix(".invocations")).read_text().splitlines()
    assert len(invocations) == 1, (
        f"oarsub must run exactly once, got {len(invocations)} calls: {invocations!r}"
    )


def test_oarsub_receives_exact_queue_and_resource_options(tmp_path):
    repo, hf, log = _make_persistent_layout(tmp_path)
    _run_submit(
        tmp_path,
        repo_root=repo,
        hf_home=hf,
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
    )
    argv = (tmp_path / "oarsub_capture.argv").read_text().splitlines()
    assert "-q" in argv, f"missing -q option: {argv!r}"
    q_idx = argv.index("-q")
    assert argv[q_idx + 1] == "production", f"queue must be production: {argv!r}"
    assert "-l" in argv, f"missing -l option: {argv!r}"
    l_idx = argv.index("-l")
    assert argv[l_idx + 1] == "gpu=1,walltime=00:30:00", (
        f"resource request mismatch: {argv!r}"
    )


def test_oarsub_receives_exactly_one_positional_command_string(tmp_path):
    repo, hf, log = _make_persistent_layout(tmp_path)
    _run_submit(
        tmp_path,
        repo_root=repo,
        hf_home=hf,
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
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
    cmd = positionals[0]
    assert "run_gpu_smoke_job.sh" in cmd, f"command string missing wrapper: {cmd!r}"


def test_command_string_decodes_to_exactly_four_arguments(tmp_path):
    """Executing the captured single command string must deliver
    exactly the four original positional arguments to a fake
    compute-node wrapper. This proves serialization preserves the
    argument count (Nancy's oarsub strips to a single string)."""
    repo, hf, log = _make_persistent_layout(tmp_path)
    _run_submit(
        tmp_path,
        repo_root=repo,
        hf_home=hf,
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
    )
    decoded = _decode_capture(tmp_path, repo, "wrapper_cap")
    assert decoded.returncode == 0, decoded.stderr
    wrapper_cap = tmp_path / "wrapper_cap"
    assert (wrapper_cap.with_suffix(".ran")).exists()
    count = (wrapper_cap.with_suffix(".count")).read_text().strip()
    assert count == "4", f"decoded command must pass exactly 4 args, got {count!r}"
    assert (wrapper_cap.with_suffix(".arg1")).read_text().strip() == str(repo)
    assert (wrapper_cap.with_suffix(".arg2")).read_text().strip() == str(hf)
    assert (wrapper_cap.with_suffix(".arg3")).read_text().strip() == str(log)
    assert (wrapper_cap.with_suffix(".arg4")).read_text().strip() == TEST_SOURCE_COMMIT


def test_command_string_starts_with_exec(tmp_path):
    repo, hf, log = _make_persistent_layout(tmp_path)
    _run_submit(
        tmp_path,
        repo_root=repo,
        hf_home=hf,
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
    )
    cmd = (tmp_path / "oarsub_capture.argv").read_text().splitlines()[-1]
    assert cmd.lstrip().startswith("exec "), f"command must start with exec: {cmd!r}"


def test_oarsub_receives_no_interactive_flag(tmp_path):
    repo, hf, log = _make_persistent_layout(tmp_path)
    _run_submit(
        tmp_path,
        repo_root=repo,
        hf_home=hf,
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
    )
    argv = (tmp_path / "oarsub_capture.argv").read_text().splitlines()
    assert "-I" not in argv, f"oarsub must not receive -I: {argv!r}"
    assert "--interactive" not in argv


def test_no_injection_marker_file_created_for_hostile_values(tmp_path):
    """Hostile values containing spaces and a glob must remain
    literal inside the single command string and must NOT execute.
    We craft a REPO_ROOT whose final component contains these
    characters and assert that decoding the command string does not
    create a marker file named by an injected command."""
    base = tmp_path / "persistent"
    base.mkdir(parents=True, exist_ok=True)
    hostile_repo = base / "repo with space and *star*"
    hostile_repo.mkdir(parents=True)
    hf = base / "hf_home"
    hf.mkdir()
    _seed_clean_hf_refs(hf)
    log = base / "logs"
    log.mkdir()
    proc = _run_submit(
        tmp_path,
        repo_root=hostile_repo,
        hf_home=hf,
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    cmd = (tmp_path / "oarsub_capture.argv").read_text().splitlines()[-1]
    assert "repo with space and *star*" in cmd
    decoded = _decode_capture(tmp_path, hostile_repo, "wrapper_cap_inj")
    assert decoded.returncode == 0, decoded.stderr
    wrapper_cap = tmp_path / "wrapper_cap_inj"
    arg1 = (wrapper_cap.with_suffix(".arg1")).read_text().strip()
    assert arg1 == str(hostile_repo), f"injection altered arg1: {arg1!r}"
    injected = list(tmp_path.glob("**/injected"))
    assert not injected, f"command injection created marker file: {injected!r}"


def test_hostile_values_with_single_quote_and_command_substitution(tmp_path):
    """A path containing a literal single quote (escaped via the
    portable '\\'') and a value containing $()/backticks must remain
    literal. We use HF_HOME with a single quote and a $() token and
    verify the captured command quotes it so the wrapper receives it
    back byte-for-byte and no substitution executes."""
    base = tmp_path / "persistent"
    base.mkdir(parents=True, exist_ok=True)
    repo = base / "repo"
    repo.mkdir()
    hf = base / "hf'a$(touch injected_now)b`date`"
    hf.mkdir()
    _seed_clean_hf_refs(hf)
    log = base / "logs"
    log.mkdir()
    proc = _run_submit(
        tmp_path,
        repo_root=repo,
        hf_home=hf,
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    decoded = _decode_capture(tmp_path, repo, "wrapper_cap_quote")
    assert decoded.returncode == 0, decoded.stderr
    wrapper_cap = tmp_path / "wrapper_cap_quote"
    arg2 = (wrapper_cap.with_suffix(".arg2")).read_text().strip()
    assert arg2 == str(hf), f"single-quote/$() value not preserved literally: {arg2!r}"
    injected = list(tmp_path.glob("**/injected_now"))
    assert not injected, f"command substitution executed: {injected!r}"


# --- Pre-submission validation failures -----------------------------


def test_missing_first_positional_fails_before_oarsub(tmp_path):
    fake_bin = _make_fake_oarsub(tmp_path)
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        cwd=str(tmp_path),
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "CAPTURE_FILE": str(tmp_path / "cap"),
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode != 0
    assert not (tmp_path / "cap.ran").exists(), "oarsub ran despite missing args"


def test_too_many_positional_arguments_fail_before_oarsub(tmp_path):
    repo, hf, log = _make_persistent_layout(tmp_path)
    proc = _run_submit(
        tmp_path,
        repo_root=repo,
        hf_home=hf,
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
        extra_args=["unexpected_fifth"],
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not (tmp_path / "oarsub_capture.ran").exists(), (
        "oarsub ran despite five positional arguments"
    )


def test_invalid_commit_format_fails_before_oarsub(tmp_path):
    repo, hf, log = _make_persistent_layout(tmp_path)
    bad = _with_nonhex_char(TEST_NONHEX_BASE, 0)
    assert len(bad) == 40
    assert "g" in bad
    proc = _run_submit(
        tmp_path, repo_root=repo, hf_home=hf, log_root=log, source_commit=bad
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not (tmp_path / "oarsub_capture.ran").exists(), (
        "oarsub ran despite non-hex commit"
    )


def test_noncanonical_ephemeral_path_fails_before_oarsub(tmp_path):
    base = tmp_path / "persistent"
    base.mkdir(parents=True, exist_ok=True)
    repo = base / "repo"
    repo.mkdir()
    hf = base / "hf_home"
    hf.mkdir()
    log = base / "logs"
    log.mkdir()
    proc = _run_submit(
        tmp_path,
        repo_root=repo,
        hf_home=Path("/tmp/bad_hf"),
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not (tmp_path / "oarsub_capture.ran").exists(), (
        "oarsub ran despite ephemeral HF_HOME"
    )


def test_symlink_repo_root_fails_before_oarsub(tmp_path):
    base = tmp_path / "persistent"
    base.mkdir(parents=True, exist_ok=True)
    real_repo = base / "real_repo"
    real_repo.mkdir()
    link_repo = base / "link_repo"
    link_repo.symlink_to(real_repo)
    hf = base / "hf_home"
    hf.mkdir()
    log = base / "logs"
    log.mkdir()
    proc = _run_submit(
        tmp_path,
        repo_root=link_repo,
        hf_home=hf,
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not (tmp_path / "oarsub_capture.ran").exists(), (
        "oarsub ran despite symlinked REPO_ROOT"
    )


def test_missing_compute_wrapper_fails_before_oarsub(tmp_path):
    """The compute-node wrapper must exist and be executable under
    REPO_ROOT/scripts/grid5000/run_gpu_smoke_job.sh; otherwise the
    helper must abort before invoking oarsub."""
    repo, hf, log = _make_persistent_layout(tmp_path)
    proc = _run_submit(
        tmp_path,
        repo_root=repo,
        hf_home=hf,
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
        wrapper_present=False,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not (tmp_path / "oarsub_capture.ran").exists(), (
        "oarsub ran despite missing compute-node wrapper"
    )


def test_missing_wrapper_error_does_not_leak_wrapper_path(tmp_path):
    """The missing-wrapper error message must be a stable path-free
    string. Even when REPO_ROOT is a sensitive synthetic path, the
    wrapper location (under REPO_ROOT/scripts/grid5000/) must NOT
    appear in the helper's stdout or stderr."""
    sensitive_root = tmp_path / "persistent" / "secret-org-prod-pipeline"
    sensitive_root.mkdir(parents=True)
    hf_home = tmp_path / "persistent" / "hf_home"
    hf_home.mkdir()
    _seed_clean_hf_refs(hf_home)
    log_root = tmp_path / "persistent" / "logs"
    log_root.mkdir()
    proc = _run_submit(
        tmp_path,
        repo_root=sensitive_root,
        hf_home=hf_home,
        log_root=log_root,
        source_commit=TEST_SOURCE_COMMIT,
        wrapper_present=False,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr
    assert "secret-org-prod-pipeline" not in combined, (
        "helper leaked the sensitive REPO_ROOT path: " + combined
    )
    assert "scripts/grid5000" not in combined, (
        "helper leaked the wrapper path component: " + combined
    )


def test_non_executable_wrapper_error_does_not_leak_wrapper_path(tmp_path):
    """The non-executable-wrapper error message must be a stable
    path-free string. The sensitive REPO_ROOT and the wrapper
    component must NOT appear in the helper's stdout or stderr."""
    sensitive_root = tmp_path / "persistent" / "secret-org-prod-pipeline"
    sensitive_root.mkdir(parents=True)
    hf_home = tmp_path / "persistent" / "hf_home"
    hf_home.mkdir()
    _seed_clean_hf_refs(hf_home)
    log_root = tmp_path / "persistent" / "logs"
    log_root.mkdir()
    proc = _run_submit(
        tmp_path,
        repo_root=sensitive_root,
        hf_home=hf_home,
        log_root=log_root,
        source_commit=TEST_SOURCE_COMMIT,
        wrapper_present=True,
        wrapper_executable=False,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr
    assert "secret-org-prod-pipeline" not in combined, (
        "helper leaked the sensitive REPO_ROOT path: " + combined
    )
    assert "scripts/grid5000" not in combined, (
        "helper leaked the wrapper path component: " + combined
    )


def test_non_executable_compute_wrapper_fails_before_oarsub(tmp_path):
    repo, hf, log = _make_persistent_layout(tmp_path)
    proc = _run_submit(
        tmp_path,
        repo_root=repo,
        hf_home=hf,
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
        wrapper_present=True,
        wrapper_executable=False,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not (tmp_path / "oarsub_capture.ran").exists(), (
        "oarsub ran despite non-executable compute-node wrapper"
    )


def test_missing_oarsub_on_path_fails(tmp_path):
    """Deterministic: invoke the helper directly with an absolute
    Bash executable and a minimal controlled PATH that contains
    the basic utilities the helper needs but no ``oarsub``. The
    test must pass even on a Grid'5000 frontend where a real
    ``oarsub`` lives outside the controlled PATH.
    """
    import shutil

    bash_abs = shutil.which("bash")
    assert bash_abs is not None, "bash must be available for this test"

    repo, hf, log = _make_persistent_layout(tmp_path)
    wrapper = repo / "scripts" / "grid5000" / "run_gpu_smoke_job.sh"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
    wrapper.chmod(0o755)

    # Build a controlled PATH directory that contains only the
    # coreutils the helper might reference. Bash is invoked
    # absolutely above, so PATH is consulted only for
    # ``command -v oarsub``, which must see no oarsub and cause the
    # helper to abort.
    control_bin = tmp_path / "control_bin"
    control_bin.mkdir()
    for tool in (
        "pwd",
        "mkdir",
        "chmod",
        "stat",
        "echo",
        "printf",
        "touch",
        "rm",
        "cat",
    ):
        src = shutil.which(tool)
        if src is not None:
            (control_bin / tool).symlink_to(src)

    proc = subprocess.run(
        [
            bash_abs,
            str(SUBMIT_SCRIPT),
            str(repo),
            str(hf),
            str(log),
            TEST_SOURCE_COMMIT,
        ],
        cwd=str(tmp_path),
        env={**os.environ, "PATH": str(control_bin)},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode != 0, (
        f"helper must abort when oarsub is not on PATH, got "
        f"rc={proc.returncode} stdout={proc.stdout!r} "
        f"stderr={proc.stderr!r}"
    )


# --- Exit-code and output forwarding ---------------------------------


def test_fake_oarsub_nonzero_exit_is_returned(tmp_path):
    repo, hf, log = _make_persistent_layout(tmp_path)
    proc = _run_submit(
        tmp_path,
        repo_root=repo,
        hf_home=hf,
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
        env={"FAKE_OARSUB_RC": "7"},
    )
    assert proc.returncode == 7, proc.stdout + proc.stderr
    # Even on a non-zero oarsub exit, exactly one oarsub process
    # must have run (no retry, no fallback).
    invocations = (tmp_path / "oarsub_capture.invocations").read_text().splitlines()
    assert len(invocations) == 1, (
        f"oarsub must run exactly once even on failure, got "
        f"{len(invocations)} calls: {invocations!r}"
    )


def test_oarsub_stdout_stderr_not_rewritten(tmp_path):
    """A fake oarsub that emits distinctive stdout/stderr must have
    those streams forwarded unchanged by the helper. Exact equality:
    the helper must not prefix/suffix status text or echo the
    command string. We compare the helper's full captured stdout
    and stderr byte-for-byte against the fake's emissions."""
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir(exist_ok=True)
    oarsub = fake_bin / "oarsub"
    oarsub.write_text(
        "#!/usr/bin/env bash\n"
        'cap="${CAPTURE_FILE:?}"\n'
        'printf "%s\\n" "$@" > "${cap}.argv"\n'
        'printf "%s\\n" "$#" >> "${cap}.invocations"\n'
        'touch "${cap}.ran"\n'
        'echo "OAR_STDOUT_MARKER_123"\n'
        'echo "OAR_STDERR_MARKER_456" >&2\n'
        'exit "${FAKE_OARSUB_RC:-0}"\n'
    )
    oarsub.chmod(0o755)
    repo, hf, log = _make_persistent_layout(tmp_path)
    capture = tmp_path / "oarsub_capture"
    wrapper = repo / "scripts" / "grid5000" / "run_gpu_smoke_job.sh"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
    wrapper.chmod(0o755)
    # Copy the cache-ref validator so the submit script can source it.
    validator_src = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "grid5000"
        / "_cache_ref_validator.sh"
    )
    if validator_src.exists():
        (wrapper.parent / "_cache_ref_validator.sh").write_text(
            validator_src.read_text()
        )
        (wrapper.parent / "_cache_ref_validator.sh").chmod(0o755)
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), str(repo), str(hf), str(log), TEST_SOURCE_COMMIT],
        cwd=str(tmp_path),
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "CAPTURE_FILE": str(capture),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # Exact equality (no helper-side prefix/suffix/status text).
    assert proc.stdout == "OAR_STDOUT_MARKER_123\n", (
        f"helper added status text to stdout: {proc.stdout!r}"
    )
    assert proc.stderr == "OAR_STDERR_MARKER_456\n", (
        f"helper added status text to stderr: {proc.stderr!r}"
    )


def test_no_retry_on_failure(tmp_path):
    """A failing oarsub invocation must not be retried: exactly one
    oarsub process must run."""
    repo, hf, log = _make_persistent_layout(tmp_path)
    proc = _run_submit(
        tmp_path,
        repo_root=repo,
        hf_home=hf,
        log_root=log,
        source_commit=TEST_SOURCE_COMMIT,
        env={"FAKE_OARSUB_RC": "9"},
    )
    assert proc.returncode == 9
    assert (tmp_path / "oarsub_capture.ran").exists()
    invocations = (tmp_path / "oarsub_capture.invocations").read_text().splitlines()
    assert len(invocations) == 1, (
        f"helper retried oarsub on failure: {len(invocations)} calls"
    )


# --- Documentation consistency ---------------------------------------


def _extract_canonical_fenced_command(text: str) -> str:
    """Extract the canonical submission fenced bash block.

    Locates the fenced bash block under the *Phase 9D/9F: ...*
    section heading and returns its inner lines (continuations
    joined with spaces, newlines stripped). Fails the test if the
    fenced block cannot be found.
    """
    # Find the Phase 9D/9F heading and its following fenced block.
    import re

    head = re.search(r"^### Phase 9D/?9?F?:.*$", text, flags=re.MULTILINE)
    assert head is not None, "canonical Phase 9D/9F heading missing in doc"
    start = head.end()
    fence_match = re.search(
        r"^```bash\s*\n(.*?)\n```",
        text[start:],
        flags=re.MULTILINE | re.DOTALL,
    )
    assert fence_match is not None, (
        "canonical Phase 9D/9F fenced bash command missing in doc"
    )
    inner = fence_match.group(1)
    # Collapse continuations: a trailing backslash newline joins.
    cleaned = re.sub(r"\\\s*\n\s*", " ", inner)
    return cleaned.strip()


def test_doc_canonical_command_invokes_submit_helper():
    """The doc's canonical automated command must invoke the new
    submit helper (not the raw wrapper, not a direct
    oarsub wrapper args... form)."""
    text = DOC.read_text(encoding="utf-8")
    assert "submit_gpu_smoke.sh" in text, (
        "doc must reference submit_gpu_smoke.sh in the canonical command"
    )
    # The invalid direct form must not appear anywhere.
    assert 'run_gpu_smoke_job.sh" \\' not in text or ("submit_gpu_smoke.sh" in text)


def test_doc_canonical_command_executable_is_submit_helper():
    """The canonical frontend command must invoke
    ``submit_gpu_smoke.sh`` directly — it must NOT be prefixed by
    ``oarsub`` (the helper itself performs the single oarsub
    invocation). The compute-node wrapper ``run_gpu_smoke_job.sh``
    must NOT appear as a directly-invoked token in this canonical
    frontend command.
    """
    import re

    text = DOC.read_text(encoding="utf-8")
    cmd = _extract_canonical_fenced_command(text)
    # The executable is the first whitespace-delimited token; the
    # doc quotes it (a leading and trailing `"`), so strip quotes
    # before comparing.
    executable = cmd.split()[0]
    executable_unquoted = executable.strip('"').strip("'")
    assert executable_unquoted.endswith("submit_gpu_smoke.sh"), (
        f"canonical frontend command executable must be "
        f"submit_gpu_smoke.sh, got: {executable!r} from cmd: {cmd!r}"
    )
    # No `oarsub` token may precede the executable on this line.
    oarsub_matches = re.findall(r"\boarsub\b", cmd)
    assert not oarsub_matches, (
        f"canonical frontend command must NOT contain an oarsub token "
        f"(the helper performs the oarsub call itself): {cmd!r}"
    )
    # The compute-node wrapper must not appear as a direct invocation
    # in this canonical frontend command.
    assert "run_gpu_smoke_job.sh" not in cmd, (
        f"canonical frontend command must not directly invoke "
        f"run_gpu_smoke_job.sh: {cmd!r}"
    )
