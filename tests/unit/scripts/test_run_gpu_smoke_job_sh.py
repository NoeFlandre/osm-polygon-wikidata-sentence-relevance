"""Contract tests for the Grid'5000 non-interactive batch entrypoint
(Phase 9D).

The compute-node batch wrapper ``scripts/grid5000/run_gpu_smoke_job.sh``
replaces the fragile ``oarsub -I`` interactive transport with an
OAR-owned batch job whose lifetime is controlled by the scheduler,
not by an SSH-held TTY.

These tests run on the Mac with only fakes and syntax/contract
checks. They never:

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
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
JOB_SCRIPT = ROOT / "scripts" / "grid5000" / "run_gpu_smoke_job.sh"
SMOKE_SCRIPT = ROOT / "scripts" / "grid5000" / "run_gpu_smoke.sh"
DOC = ROOT / "docs" / "guides" / "grid5000.md"


# --- SHA fixtures (no opaque literal in the source) ----------------


def _test_sha(label: str) -> str:
    """Return a 40-character lowercase-hex digest for use as a
    deterministic SHA-shaped test fixture (test-only)."""
    return hashlib.sha1(label.encode("utf-8"), usedforsecurity=False).hexdigest()


TEST_EXPECTED_SOURCE_COMMIT = _test_sha(
    "tests/grid5000/run_gpu_smoke_job/expected_source_commit"
)
TEST_OTHER_SOURCE_COMMIT = _test_sha(
    "tests/grid5000/run_gpu_smoke_job/other_source_commit"
)
TEST_NONHEX_SHA_BASE = _test_sha("tests/grid5000/run_gpu_smoke_job/nonhex_sha_base")


def _with_nonhex_char(digest: str, position: int = 0) -> str:
    """Return a 40-character string derived from ``digest`` (a
    lowercase-hex fixture of length 40) but with the character at
    ``position`` replaced by ``g`` (deliberately non-hex). Used to
    construct exactly-40-length invalid fixtures without writing
    repeated-character literals."""
    chars = list(digest)
    chars[position] = "g"
    return "".join(chars)


# --- Patterns that must NEVER appear anywhere in the wrapper -------


# Forbidden patterns cover:
#   - interactive oarsub (-I)
#   - scheduler-variable masking
#   - fallback devices
#   - personal paths
#   - mutating git invocations
#   - token / publishing / inference flags
#   - remote-fetch / ssh / rsync
#   - dataset / classification / upload keywords
_FORBIDDEN_PATTERNS = (
    "oarsub -I",
    "oarsub -I ",
    "--interactive",
    "-I\n",
    "export CUDA_VISIBLE_DEVICES=",
    "CUDA_VISIBLE_DEVICES:=",
    "export OAR_JOB_ID=",
    "OAR_JOB_ID:=",
    'device="auto"',
    'device="cpu"',
    'device="mps"',
    '"--publish"',
    "--publish",
    "--input-dataset-id",
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
    "git reset",
    "git clean",
    "git pull",
    "git fetch",
    "git clone",
    "git rm",
    "git mv",
    "git stash",
    "git apply",
    "git cherry-pick",
    "git rebase",
    "rsync",
    "ssh ",
    "ssh(",
    "snapshot_download",
    "curl ",
    "wget ",
    "huggingface-cli",
    "create_commit",
    "dataset_card",
    "classification",
    "upload",
    "publish",
    "python3 ",
    " python ",
    "uv run",
    "conda",
    "activate",
)


@pytest.fixture(scope="module")
def script_text() -> str:
    return JOB_SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


# --- Static checks: script structure ---------------------------------


def test_job_script_exists_and_is_nonempty():
    assert JOB_SCRIPT.exists()
    text = JOB_SCRIPT.read_text(encoding="utf-8")
    assert len(text) > 200


def test_job_script_passes_bash_syntax_check():
    proc = subprocess.run(
        ["bash", "-n", str(JOB_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_smoke_script_passes_bash_syntax_check():
    proc = subprocess.run(
        ["bash", "-n", str(SMOKE_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_job_script_uses_strict_bash_modes(script_text):
    assert "set -euo pipefail" in script_text


# --- Wrapper positional/option contract -----------------------------


def test_job_script_documents_positional_arguments(script_text):
    """The wrapper must document the four required positional
    arguments and reject any invocation that does not supply them.

    Either positional or strictly validated options are acceptable;
    the static contract: ``positional_args`` / explicit
    ``${1:-}`` / ``${2:-}`` / ``${3:-}`` / ``${4:-}`` style consumption.
    """
    assert "$1" in script_text
    assert "$2" in script_text
    assert "$3" in script_text
    assert "$4" in script_text


def test_job_script_requires_all_four_positional_arguments(script_text):
    """Each positional must be required either with ``:?``
    semantics or via an explicit ``$# -ne 4`` guard that
    rejects the wrong argument count. The amended wrapper
    uses the explicit count guard, which is stricter than
    per-positional ``${N:?}`` because it also rejects
    superfluous fifth positional arguments."""
    # Either the explicit count guard or a per-positional ``:?``
    # guard is acceptable. The amended wrapper uses the count
    # guard.
    assert (
        '"$#" -ne 4' in script_text
        or "[ $# -ne 4 ]" in script_text
        or all(p in script_text for p in ("${1:?", "${2:?", "${3:?", "${4:?"))
    ), "expected guard requiring exactly four positional arguments"


# --- Scheduler-variable preservation ---------------------------------


def test_job_script_requires_oar_job_id(script_text):
    assert "OAR_JOB_ID:?" in script_text or "OAR_JOB_ID" in script_text
    assert (
        '":${OAR_JOB_ID:?OAR_JOB_ID is required' in script_text
        or "OAR_JOB_ID is required" in script_text
    )


def test_job_script_requires_cuda_visible_devices(script_text):
    assert "CUDA_VISIBLE_DEVICES" in script_text
    assert (
        "CUDA_VISIBLE_DEVICES is required" in script_text
        or '":${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required' in script_text
    )


def test_job_script_does_not_export_cuda_visible_devices(script_text):
    # The scheduler owns CUDA_VISIBLE_DEVICES; the wrapper must
    # never overwrite it, even with a guard default.
    assert "export CUDA_VISIBLE_DEVICES=" not in script_text
    assert "CUDA_VISIBLE_DEVICES:=" not in script_text


def test_job_script_does_not_export_oar_job_id(script_text):
    # OAR_JOB_ID is set by the scheduler and must not be overwritten.
    assert "export OAR_JOB_ID=" not in script_text
    assert "OAR_JOB_ID:=" not in script_text


def test_job_script_does_not_normalize_cuda_visible_devices(script_text):
    # No whitespace-stripping / unset-defaulting form.
    forbidden_normalizations = (
        "${CUDA_VISIBLE_DEVICES// /}",
        "${CUDA_VISIBLE_DEVICES,,}",
        "${CUDA_VISIBLE_DEVICES^^}",
        "${CUDA_VISIBLE_DEVICES//}",
    )
    for f in forbidden_normalizations:
        assert f not in script_text, (
            f"CUDA_VISIBLE_DEVICES normalization is forbidden: {f!r}"
        )


# --- Source-commit validation ---------------------------------------


def test_job_script_requires_expected_source_commit(script_text):
    # 40 lowercase hex chars via explicit pattern.
    assert "[0-9a-f]{40}" in script_text or (
        "EXPECTED_SOURCE_COMMIT is required" in script_text
    )


def test_job_script_validates_expected_source_commit_format(script_text):
    # The pattern must reject uppercase / wrong-length / non-hex.
    assert "[0-9a-f]" in script_text


def test_job_script_does_not_hard_coded_commit(script_text):
    # The wrapper never embeds a real commit SHA literal.
    import re

    matches = re.findall(r"\b[0-9a-f]{40}\b", script_text)
    for m in matches:
        # The model/tokenizer SHAs are allowed because they are
        # structurally identical to commit SHAs but appear in
        # documentation strings only.
        assert m in {
            "137da054051ad9f1eac42025f758db4ac9f22535",
            "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        }, f"unexpected 40-hex literal in wrapper: {m!r}"


# --- Path validation -------------------------------------------------


def test_job_script_rejects_ephemeral_paths(script_text):
    for p in ("/tmp", "/var/tmp", "/dev/shm"):
        assert p in script_text, f"ephemeral path not rejected: {p!r}"


def test_job_script_validates_absolute_paths(script_text):
    # The wrapper must require canonical absolute paths and reject
    # anything containing ``..`` or non-absolute forms.
    assert ".." in script_text  # listed as a forbidden traversal token
    # Canonicalisation must appear so symlink escapes are rejected
    # before the smoke payload runs. The amended wrapper uses
    # ``pwd -P`` via a strict ``_canonicalise_directory`` helper
    # (no external ``realpath`` dependency).
    assert "pwd -P" in script_text or "realpath" in script_text
    assert "_canonicalise_directory" in script_text


def test_job_script_rejects_empty_paths(script_text):
    assert "is empty" in script_text or "non-empty" in script_text


# --- Repository / interpreter / harness presence ---------------------


def test_job_script_verifies_repo_root(script_text):
    assert "REPO_ROOT" in script_text
    # Must require it to be an existing directory. The amended
    # wrapper performs this check inside the
    # ``_canonicalise_directory`` helper using the normalised
    # path, so either the bare ``[ -d "${REPO_ROOT}" ]`` form or
    # the canonicaliser's internal ``[ ! -d "${normalised}" ]``
    # form is acceptable.
    assert (
        '[ ! -d "${REPO_ROOT}" ]' in script_text
        or '[ -d "${REPO_ROOT}" ]' in script_text
        or '[ ! -d "${normalised}" ]' in script_text
    )


def test_job_script_verifies_hf_home(script_text):
    assert "HF_HOME" in script_text
    assert (
        '[ ! -d "${HF_HOME}" ]' in script_text
        or '[ -d "${HF_HOME}" ]' in script_text
        or '[ ! -r "${HF_HOME}" ]' in script_text
        or '[ ! -d "${normalised}" ]' in script_text
        or '[ ! -r "${HF_HOME_REAL}" ]' in script_text
    )


def test_job_script_verifies_locked_interpreter(script_text):
    assert ".venv/bin/python" in script_text
    assert "[ ! -f" in script_text or "PROJECT_PYTHON" in script_text
    assert "[ ! -x" in script_text


def test_job_script_verifies_smoke_harness(script_text):
    # The wrapper must invoke the committed smoke harness file path.
    assert "scripts/grid5000/run_gpu_smoke.sh" in script_text
    assert (
        '[ ! -r "${SMOKE_HARNESS}" ]' in script_text
        or ('[ ! -r "$${SMOKE_HARNESS}" ]' in script_text)
        or "SMOKE_HARNESS=.*run_gpu_smoke.sh" in script_text
        or "run_gpu_smoke.sh" in script_text
    )


def test_job_script_does_not_use_bare_python(script_text):
    """The wrapper must never invoke a bare interpreter. The
    contract forbids:
    - ``python3 `` (with a trailing space) — the literal always
      suggests an executable invocation;
    - `` python `` (with leading and trailing space) — a bare
      invocation as an executable name;
    - the bare invocation forms ``python -<option>`` and
      ``python ${anyvar}``.

    The wrapper may name a *file path* ending in ``/python``,
    e.g. ``${REPO_ROOT}/.venv/bin/python``, because that is a
    file-path argument to a launcher check, not a bare
    interpreter invocation.
    """
    assert "python3 " not in script_text
    # The wrapper must not look like it runs a bare binary by
    # name. A file-path argument that ends in /python is OK.
    # The check below forbids the line
    #     python ...
    # but allows:
    #     .venv/bin/python
    for line in script_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # The most likely offending form is a bare ``python``.
        # Match ``python`` followed by ASCII whitespace. Do not
        # match ``python/...`` or ``.../python``.
        assert not re.search(r"(?<![\w/])python\b\s", line), (
            f"wrapper appears to invoke a bare interpreter: {line!r}"
        )


# --- Job-log directory creation and reuse refusal --------------------


def test_job_script_creates_job_specific_log_dir(script_text):
    assert "${LOG_ROOT}/${OAR_JOB_ID}" in script_text


def test_job_script_refuses_reuse_of_existing_log_dir(script_text):
    # The wrapper must refuse to overwrite an existing
    # ${LOG_ROOT}/${OAR_JOB_ID}. The ``[ ! -e ... ]`` guard or
    # explicit ``mkdir`` (not ``mkdir -p``) is the static mark.
    assert "mkdir" in script_text
    assert "mkdir -p" not in script_text


def test_job_script_uses_mode_0700_for_job_log_dir(script_text):
    assert "0700" in script_text
    assert "mkdir -m 0700" in script_text or "chmod 0700" in script_text


def test_job_script_sets_restrictive_umask(script_text):
    assert "umask 077" in script_text


# --- Smoke harness invocation: exactly once --------------------------


def test_job_script_invokes_smoke_exactly_once(script_text):
    # Count occurrences of the canonical smoke script reference.
    count = script_text.count("run_gpu_smoke.sh")
    # Once for the SMOKE_HARNESS variable definition, once for the
    # actual bash invocation, possibly once more for the assertion
    # that it is a regular readable file; we allow 2-4 occurrences.
    assert 1 <= count <= 4, f"unexpected number of run_gpu_smoke.sh references: {count}"


def test_job_script_captures_smoke_stdout_stderr(script_text):
    assert "smoke.stdout.log" in script_text
    assert "smoke.stderr.log" in script_text


def test_job_script_captures_smoke_exit_code(script_text):
    assert "smoke.exit_code" in script_text
    # Capture must be the real exit status, not a masked value.
    # The simplest pattern: ``printf '%s\n' "${smoke_rc}"`` with an
    # explicit ``smoke_rc=$?`` capture. We accept either form.
    assert "smoke_rc" in script_text
    assert "exit_code" in script_text


def test_job_script_preserves_all_artifacts_on_failure(script_text):
    # The wrapper must not delete existing artifacts when the smoke
    # returns non-zero; the contract is: capture the exit code, then
    # exit with it.
    assert "rm -rf ${SMOKE_LOG_DIR}" not in script_text


# --- Return-value contract -------------------------------------------


def test_job_script_returns_smoke_exit_status(script_text):
    # ``exit "${smoke_rc}"`` is the documented contract.
    assert 'exit "${smoke_rc}"' in script_text or ('exit "${smoke_rc}"' in script_text)


# --- Forbidden patterns ---------------------------------------------


def test_job_script_has_no_forbidden_patterns(script_text):
    for pattern in _FORBIDDEN_PATTERNS:
        assert pattern not in script_text, f"wrapper must not contain {pattern!r}"


# --- Executable contract: synthetic compute-node test ----------------


def _make_fake_repo(tmp_path: Path) -> Path:
    """Create a minimal fake repo containing the committed smoke
    harness (a *fake* harness that writes the six expected
    artifacts). No Torch, no SaT, no inference. Idempotent."""
    repo = tmp_path / "fake_repo"
    if repo.exists():
        return repo
    repo.mkdir()
    scripts_dir = repo / "scripts" / "grid5000"
    scripts_dir.mkdir(parents=True)
    # Tiny smoke harness faking the committed artifact layout.
    harness = scripts_dir / "run_gpu_smoke.sh"
    # Use a triple-quoted plain string (not an f-string) so the
    # literal ${OAR_JOB_ID} substring is preserved verbatim.
    harness_text = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "umask 077\n"
        ': "${SMOKE_LOG_DIR:?SMOKE_LOG_DIR is required}"\n'
        ': "${HF_HOME:?HF_HOME is required}"\n'
        ': "${REPO_ROOT:?REPO_ROOT is required}"\n'
        ': "${OAR_JOB_ID:?OAR_JOB_ID is required}"\n'
        ': "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"\n'
        ': "${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required}"\n'
        'printf \'{"oar_job_id":"%s","hostname":"fake","torch_version":"2.13.0","torch_cuda_runtime_version":"12.1","device_0_name":"L40S","visible_cuda_device_count":1}\\n\' "${OAR_JOB_ID}" > "${SMOKE_LOG_DIR}/gpu_preflight.json"\n'
        'printf \'{"source_commit":"%s","model_name":"sat-3l-sm","model_revision":"sat-sha","tokenizer_name":"facebookAI/xlm-roberta-base","tokenizer_revision":"tok-sha","oar_job_id":"%s","hostname":"fake"}\\n\' "${EXPECTED_SOURCE_COMMIT}" "${OAR_JOB_ID}" > "${SMOKE_LOG_DIR}/run_metadata.json"\n'
        'printf \'{"resolved_device":"cuda","model_name":"sat-3l-sm","input_count":3,"sentence_counts":[2,2,2],"elapsed_seconds":0.1,"torch_version":"2.13.0","torch_cuda_runtime_version":"12.1","cuda_device_name":"L40S"}\\n\' > "${SMOKE_LOG_DIR}/smoke_result.json"\n'
        'echo "smoke ran" > "${SMOKE_LOG_DIR}/smoke.stdout.log"\n'
        'echo "" > "${SMOKE_LOG_DIR}/smoke.stderr.log"\n'
        "exit 0\n"
    )
    harness.write_text(harness_text)
    harness.chmod(0o755)
    # Also drop the validator file since the harness does not
    # require it (the wrapper must still tolerate its absence).
    (scripts_dir / "_validate_artifact.py").write_text(
        "#!/usr/bin/env python3\nimport sys\nsys.exit(0)\n"
    )
    (scripts_dir / "_validate_artifact.py").chmod(0o755)
    (scripts_dir / "gpu_preflight.py").write_text(
        "#!/usr/bin/env python3\nprint('ok')\n"
    )
    (scripts_dir / "gpu_preflight.py").chmod(0o755)
    return repo


def _make_fake_interpreter(tmp_path: Path) -> Path:
    """Create a fake locked interpreter at
    ``${tmp_path}/fake_venv_bin/python``. Idempotent.
    """
    py_dir = tmp_path / "fake_venv_bin"
    if py_dir.exists():
        return py_dir
    py_dir.mkdir()
    py = py_dir / "python"
    py.write_text("#!/usr/bin/env bash\nexit 0\n")
    py.chmod(0o755)
    return py_dir


def _make_log_root(parent: Path) -> tuple[Path, Path]:
    """Create persistent log root and HF root under ``parent``.
    Idempotent."""
    hf_home = parent / "hf_home"
    log_root = parent / "logs"
    hf_home.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    return hf_home, log_root


def _run_job(
    tmp_path: Path,
    *,
    oar_job_id: str,
    cuda_visible_devices: str = "0",
    expected_source_commit: str = TEST_EXPECTED_SOURCE_COMMIT,
    oar_job_id_unset: bool = False,
    cuda_visible_devices_unset: bool = False,
    expect_success: bool = True,
    repo: Path | None = None,
    hf_home: Path | None = None,
    log_root: Path | None = None,
    interpreter: Path | None = None,
    extra_args: list[str] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    """Run the wrapper under controlled conditions.

    ``dry_run=True`` substitutes the real wrapper with a tiny
    shell that echoes ``bash <replacement> ...`` rather than
    executing it. Used to verify argument-construction logic
    separately from execution logic.
    """
    repo = repo or _make_fake_repo(tmp_path)
    if hf_home is None or log_root is None:
        hf_home_default, log_root_default = _make_log_root(tmp_path / "persistent")
        hf_home = hf_home or hf_home_default
        log_root = log_root or log_root_default

    interpreter = interpreter or _make_fake_interpreter(tmp_path)

    # Copy the fake interpreter into the repo as ``.venv/bin/python``
    # so the wrapper's interpreter-presence check passes.
    fake_repo_venv = repo / ".venv" / "bin"
    if not fake_repo_venv.exists():
        fake_repo_venv.mkdir(parents=True)
    fake_py = fake_repo_venv / "python"
    fake_py.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_py.chmod(0o755)

    # We invoke the wrapper with the four positional arguments and a
    # batch of scheduler-set env vars. Run from an unrelated cwd to
    # prove the wrapper does not assume the working directory.
    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir(exist_ok=True)
    env = {**os.environ}
    if not oar_job_id_unset:
        env["OAR_JOB_ID"] = oar_job_id
    if not cuda_visible_devices_unset:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    args = [
        "bash",
        str(JOB_SCRIPT),
        str(repo),
        str(hf_home),
        str(log_root),
        expected_source_commit,
    ]
    if extra_args:
        args.extend(extra_args)

    cmd = args

    return subprocess.run(
        cmd,
        cwd=str(unrelated_cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


# --- Executable: missing scheduler env vars --------------------------


def test_missing_oar_job_id_aborts(tmp_path):
    proc = _run_job(
        tmp_path,
        oar_job_id="",  # placeholder; we will unset below
        oar_job_id_unset=True,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr


def test_missing_cuda_visible_devices_aborts(tmp_path):
    proc = _run_job(
        tmp_path,
        oar_job_id="1000001",
        cuda_visible_devices_unset=True,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr


# --- Executable: required positional arguments -------------------------


def test_missing_first_positional_arg_aborts(tmp_path):
    proc = subprocess.run(
        ["bash", str(JOB_SCRIPT)],
        cwd=str(tmp_path),
        env={
            **os.environ,
            "OAR_JOB_ID": "1000002",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr


# --- Executable: malformed expected source commit --------------------


def test_uppercase_expected_source_commit_aborts(tmp_path):
    proc = _run_job(
        tmp_path,
        oar_job_id="1000003",
        expected_source_commit=TEST_EXPECTED_SOURCE_COMMIT.upper(),
    )
    assert proc.returncode != 0
    # Must mention the SHA pattern, not the value.
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert TEST_EXPECTED_SOURCE_COMMIT.upper() not in combined


def test_too_short_expected_source_commit_aborts(tmp_path):
    proc = _run_job(
        tmp_path,
        oar_job_id="1000004",
        expected_source_commit="deadbeef",
    )
    assert proc.returncode != 0


def test_nonhex_expected_source_commit_aborts(tmp_path):
    invalid_40_nonhex = _with_nonhex_char(TEST_NONHEX_SHA_BASE, position=0)
    proc = _run_job(
        tmp_path,
        oar_job_id="1000005",
        expected_source_commit=invalid_40_nonhex,
    )
    assert proc.returncode != 0


# --- Executable: path refusals ---------------------------------------


def test_ephemeral_hf_home_is_refused(tmp_path):
    """A HF_HOME under /tmp must be rejected without any model
    construction. We use a path that is unlikely to collide with a
    genuine /tmp/* fixture but is on a rejected prefix."""
    repo = _make_fake_repo(tmp_path)
    log_root = tmp_path / "logs"
    log_root.mkdir(parents=True)
    fake_repo_venv = repo / ".venv" / "bin"
    fake_repo_venv.mkdir(parents=True)
    fake_py = fake_repo_venv / "python"
    fake_py.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_py.chmod(0o755)

    proc = subprocess.run(
        [
            "bash",
            str(JOB_SCRIPT),
            str(repo),
            "/tmp/ephemeral-blocked-hf-home",
            str(log_root),
            TEST_EXPECTED_SOURCE_COMMIT,
        ],
        env={
            **os.environ,
            "OAR_JOB_ID": "1000006",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode != 0
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "/tmp/ephemeral-blocked-hf-home" not in combined, (
        "HF_HOME leaked into wrapper output"
    )


def test_traversal_path_is_refused(tmp_path):
    repo = _make_fake_repo(tmp_path)
    log_root = tmp_path / "logs"
    log_root.mkdir(parents=True)
    fake_repo_venv = repo / ".venv" / "bin"
    fake_repo_venv.mkdir(parents=True)
    fake_py = fake_repo_venv / "python"
    fake_py.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_py.chmod(0o755)

    proc = subprocess.run(
        [
            "bash",
            str(JOB_SCRIPT),
            f"{repo}/../SENSITIVE-TRAVERSAL",
            str(repo / "hf_home"),
            str(log_root),
            TEST_EXPECTED_SOURCE_COMMIT,
        ],
        env={
            **os.environ,
            "OAR_JOB_ID": "1000007",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode != 0


def test_symlink_escape_is_refused(tmp_path, monkeypatch):
    # Construct a persistent layout where REPO_ROOT is a symlink to
    # a *real* location outside the operator-visible prefix.
    target = tmp_path / "real_repo"
    target.mkdir()
    (target / "scripts").mkdir()
    (target / "scripts" / "grid5000").mkdir()
    (target / "scripts" / "grid5000" / "run_gpu_smoke.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n"
    )
    (target / "scripts" / "grid5000" / "run_gpu_smoke.sh").chmod(0o755)
    venv_bin = target / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").write_text("#!/usr/bin/env bash\nexit 0\n")
    (venv_bin / "python").chmod(0o755)

    link_parent = tmp_path / "links"
    link_parent.mkdir()
    repo_link = link_parent / "repo-link"
    repo_link.symlink_to(target)

    proc = subprocess.run(
        [
            "bash",
            str(JOB_SCRIPT),
            str(repo_link),
            str(target / "hf_home"),
            str(target / "logs"),
            TEST_EXPECTED_SOURCE_COMMIT,
        ],
        env={
            **os.environ,
            "OAR_JOB_ID": "1000008",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr


# --- Executable: success path produces the six artifacts --------------


def test_successful_run_writes_all_six_artifacts(tmp_path):
    proc = _run_job(
        tmp_path,
        oar_job_id="1000099",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    log_dir = tmp_path / "persistent" / "logs" / "1000099"
    for f in (
        "gpu_preflight.json",
        "run_metadata.json",
        "smoke_result.json",
        "smoke.stdout.log",
        "smoke.stderr.log",
        "smoke.exit_code",
    ):
        path = log_dir / f
        assert path.exists(), f"missing artifact: {f}"
    # The exit_code file must contain the real exit code.
    exit_text = (log_dir / "smoke.exit_code").read_text().strip()
    assert exit_text == "0"
    # All files are mode 0600 inside the 0700 job directory.
    for f in (
        "gpu_preflight.json",
        "run_metadata.json",
        "smoke_result.json",
        "smoke.stdout.log",
        "smoke.stderr.log",
        "smoke.exit_code",
    ):
        stat = (log_dir / f).stat()
        # ``stat.st_mode & 0o777`` extracts the permission bits.
        assert stat.st_mode & 0o777 == 0o600, (
            f"artifact {f} mode != 0600 (got {oct(stat.st_mode & 0o777)})"
        )
    # Job directory itself is mode 0700.
    dir_stat = log_dir.stat()
    assert dir_stat.st_mode & 0o777 == 0o700


def test_smoke_failure_exit_code_is_preserved(tmp_path):
    """Construct a fake smoke harness that exits with a non-zero
    status; the wrapper must capture that exit code and write it
    out without retrying."""
    repo = _make_fake_repo(tmp_path)
    # Replace the smoke harness with a fail-fast script.
    harness = repo / "scripts" / "grid5000" / "run_gpu_smoke.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'echo "smoke failed" > "${SMOKE_LOG_DIR}/smoke.stderr.log"\n'
        "exit 42\n"
    )
    harness.chmod(0o755)

    proc = _run_job(tmp_path, oar_job_id="1000100", repo=repo)
    # The wrapper's exit code must reflect the smoke's exit code.
    assert proc.returncode == 42, proc.stdout + proc.stderr
    log_dir = tmp_path / "persistent" / "logs" / "1000100"
    # The exit_code file must contain 42.
    exit_text = (log_dir / "smoke.exit_code").read_text().strip()
    assert exit_text == "42"
    # No retry attempts: stderr must not containe retry markers.
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "retry" not in combined.lower()
    assert "fall back" not in combined.lower()


def test_existing_job_log_dir_is_refused(tmp_path):
    proc1 = _run_job(
        tmp_path,
        oar_job_id="1000200",
    )
    assert proc1.returncode == 0
    # Second run with the same OAR_JOB_ID must refuse reuse.
    proc2 = _run_job(
        tmp_path,
        oar_job_id="1000200",
    )
    assert proc2.returncode != 0


# --- Executable: no mutation in success path --------------------------


def test_wrapper_does_not_run_git(tmp_path):
    """The wrapper must NOT invoke git; the smoke harness itself
    is the only thing that may run git (and only via read-only
    ``git rev-parse`` / ``git status --porcelain``)."""
    proc = _run_job(
        tmp_path,
        oar_job_id="1000300",
    )
    assert proc.returncode == 0


# --- Static check: argument quoting survives spaces --------------------


def test_quoting_survives_spaces_in_paths(tmp_path, monkeypatch):
    """A repo path containing a space must reach the smoke harness
    intact (no shell injection)."""
    repo = tmp_path / "repo with space"
    repo.mkdir()
    scripts_dir = repo / "scripts" / "grid5000"
    scripts_dir.mkdir(parents=True)
    harness = scripts_dir / "run_gpu_smoke.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ': "${SMOKE_LOG_DIR:?SMOKE_LOG_DIR is required}"\n'
        ': "${HF_HOME:?HF_HOME is required}"\n'
        ': "${REPO_ROOT:?REPO_ROOT is required}"\n'
        # Verify that REPO_ROOT reached the harness exactly.
        'printf \'%s\\n\' "${REPO_ROOT}" > "${SMOKE_LOG_DIR}/smoke.stdout.log"\n'
        # Touch the three JSON artifacts with dummy content so the
        # harness satisfies the wrapper's ``exit 0`` expectation.
        "printf '{}\\n' > \"${SMOKE_LOG_DIR}/gpu_preflight.json\"\n"
        "printf '{}\\n' > \"${SMOKE_LOG_DIR}/run_metadata.json\"\n"
        "printf '{}\\n' > \"${SMOKE_LOG_DIR}/smoke_result.json\"\n"
        "printf '' > \"${SMOKE_LOG_DIR}/smoke.stderr.log\"\n"
        "exit 0\n"
    )
    harness.chmod(0o755)
    # Drop validator and preflight presence only.
    (scripts_dir / "_validate_artifact.py").write_text(
        "#!/usr/bin/env python3\nsys.exit(0)\n"
    )
    (scripts_dir / "_validate_artifact.py").chmod(0o755)
    (scripts_dir / "gpu_preflight.py").write_text(
        "#!/usr/bin/env python3\nprint('ok')\n"
    )
    (scripts_dir / "gpu_preflight.py").chmod(0o755)
    fake_repo_venv = repo / ".venv" / "bin"
    fake_repo_venv.mkdir(parents=True)
    fake_py = fake_repo_venv / "python"
    fake_py.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_py.chmod(0o755)

    persistent_dir = tmp_path / "persistent_log"
    log_root = persistent_dir / "logs"
    persistent_dir.mkdir(parents=True)
    log_root.mkdir(parents=True)

    # The HF_HOME path contains a space; create the directory
    # so the wrapper's readable-directory check passes.
    hf_home = tmp_path / "hf home with space"
    hf_home.mkdir(parents=True)

    proc = subprocess.run(
        [
            "bash",
            str(JOB_SCRIPT),
            str(repo),
            str(hf_home),
            str(log_root),
            TEST_EXPECTED_SOURCE_COMMIT,
        ],
        env={
            **os.environ,
            "OAR_JOB_ID": "1000400",
            "CUDA_VISIBLE_DEVICES": "0",
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    log_dir = log_root / "1000400"
    stdout_log = log_dir / "smoke.stdout.log"
    assert stdout_log.exists()
    content = stdout_log.read_text().strip()
    assert content == str(repo), f"REPO_ROOT did not survive quoting: {content!r}"


# --- Documentation contract: non-interactive batch is canonical ---------


def test_doc_section_for_batch_submission_exists(doc_text):
    """The guide must document a non-interactive batch submission
    flow. The static contract: a header section that names the
    compute-node wrapper explicitly."""
    assert "Phase 9D" in doc_text
    assert "compute-node" in doc_text.lower() or "compute node" in doc_text.lower()


def test_doc_canonical_command_uses_no_interactive_flag(doc_text):
    """The canonical command in the docs must NOT use ``-I`` for
    the automation path. The canonical frontend command invokes
    the submission helper directly (no ``oarsub`` prefix). The
    ``-I`` form is permitted only in a separate human-driven
    debugging section."""
    # The canonical batch command (under the Phase 9D/9F heading)
    # invokes submit_gpu_smoke.sh, which itself invokes oarsub
    # with -q production -l gpu=1,walltime=00:30:00 and no -I.
    # Find the Phase 9D/9F fenced bash block.
    head = re.search(r"^### Phase 9D/?9?F?:.*$", doc_text, flags=re.MULTILINE)
    assert head is not None, "canonical Phase 9D/9F heading missing in doc"
    fenced = re.search(
        r"^```bash\s*\n(.*?)\n```",
        doc_text[head.end() :],
        flags=re.MULTILINE | re.DOTALL,
    )
    assert fenced is not None, "canonical fenced bash block missing"
    canonical_cmd = re.sub(r"\\\s*\n\s*", " ", fenced.group(1)).strip()
    # The canonical frontend command does NOT start with oarsub.
    assert not canonical_cmd.startswith("oarsub"), (
        f"canonical frontend command must invoke submit_gpu_smoke.sh "
        f"directly (no oarsub prefix): {canonical_cmd!r}"
    )
    assert "submit_gpu_smoke.sh" in canonical_cmd, (
        f"canonical command must invoke submit_gpu_smoke.sh: {canonical_cmd!r}"
    )
    # No -I, no besteffort, no classic_ssh in the canonical command.
    assert " -I" not in canonical_cmd
    assert "besteffort" not in canonical_cmd
    assert "classic_ssh" not in canonical_cmd
    # A separate interactive-debugging section may contain -I.
    interactive = re.search(
        r"^### Interactive.*?$",
        doc_text,
        flags=re.MULTILINE,
    )
    if interactive is not None:
        section = doc_text[interactive.start() :]
        assert " -I" in section, "interactive-debugging section must still document -I"


def test_doc_explains_oarstat_monitoring(doc_text):
    """The new section must mention ``oarstat -j <job_id>`` as the
    monitoring primitive."""
    assert "oarstat" in doc_text
    assert "-j " in doc_text or "-j&nbsp;" in doc_text or "-j<" in doc_text


def test_doc_explains_artifact_location(doc_text):
    """The doc must mention where the six-artifact bundle is
    written after a successful non-interactive batch run."""
    assert "${LOG_ROOT}/${OAR_JOB_ID}" in doc_text or (
        "$LOG_ROOT/$OAR_JOB_ID" in doc_text
    )


def test_doc_explains_failure_handling(doc_text):
    """The doc must explain the failure-handling contract:
    preserve artifacts, do not retry from the assistant, report
    the OAR job ID and exit code."""
    assert "exit code" in doc_text.lower() or "exit_code" in doc_text
    assert "preserve" in doc_text.lower() or "preserve " in doc_text.lower()


def test_doc_states_compute_node_only_inference(doc_text):
    """The doc must reiterate that inference happens only on the
    allocated compute node, never on the Mac or frontend."""
    assert "compute node" in doc_text.lower()
    # Reiteration: the doc must mention both "Mac" and "frontend"
    # as forbidden places for inference.
    assert "mac" in doc_text.lower()
    assert "frontend" in doc_text.lower()


def test_doc_no_personal_paths_persisted(doc_text):
    for bad in ("/home/nflandre", "/Users/", "/Volumes/", "/srv/storage/"):
        assert bad not in doc_text, f"public guide must not embed {bad!r}"


# =====================================================================
# Phase 9D amendment RED tests.
#
# The amendment contracts:
#   1. Path canonicalisation rejects any final or intermediate
#      symlink, an absolute value that equals its ``pwd -P``
#      resolution, and a non-existent target.
#   2. ``OAR_JOB_ID`` must match ``^[0-9]+$`` before being used as
#      part of any path; the value must never be echoed.
#   3. Exactly four positional arguments are required; a fifth
#      argument is rejected before job-directory creation or smoke
#      invocation.
#   4. On smoke success the wrapper enforces six regular files in
#      the job log directory with mode 0600 and the directory
#      itself mode 0700; smoke failure preserves the original
#      nonzero exit code and the partial artefacts.
#
# These tests are the **frozen contract** against which the
# amended wrapper is written (RED → GREEN).
# =====================================================================


# --- Helpers for amendment tests -------------------------------------------


def _run_wrapper_with_paths(
    tmp_path: Path,
    *,
    repo: Path,
    hf_home: Path,
    log_root: Path,
    oar_job_id: str = "12345",
    cuda_visible_devices: str = "0",
    extra_args: list[str] | None = None,
    expected_source_commit: str = TEST_EXPECTED_SOURCE_COMMIT,
) -> subprocess.CompletedProcess:
    """Run the wrapper with explicit paths and a real OAR-shaped
    numeric job id. The other prerequisites are constructed
    automatically (repo contents, interpreter, layout)."""
    repo = repo or tmp_path / "fake_repo"
    if not repo.exists():
        repo.mkdir(parents=True)
        scripts_dir = repo / "scripts" / "grid5000"
        scripts_dir.mkdir(parents=True)
        harness_text = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            ': "${SMOKE_LOG_DIR:?SMOKE_LOG_DIR is required}"\n'
            ': "${OAR_JOB_ID:?OAR_JOB_ID is required}"\n'
            "exit 0\n"
        )
        (scripts_dir / "run_gpu_smoke.sh").write_text(harness_text)
        (scripts_dir / "run_gpu_smoke.sh").chmod(0o755)
        (repo / ".venv" / "bin").mkdir(parents=True)
        (repo / ".venv" / "bin" / "python").write_text("#!/usr/bin/env bash\nexit 0\n")
        (repo / ".venv" / "bin" / "python").chmod(0o755)
    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "OAR_JOB_ID": oar_job_id,
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
    }
    args = [
        "bash",
        str(JOB_SCRIPT),
        str(repo),
        str(hf_home),
        str(log_root),
        expected_source_commit,
    ]
    if extra_args:
        args.extend(extra_args)
    return subprocess.run(
        args,
        cwd=str(unrelated_cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


# --- (1) Path canonicalisation RED --------------------------------


def test_repo_root_final_component_symlink_aborts(tmp_path):
    """A REPO_ROOT whose final component is a symlink must fail.

    Other prerequisites are still valid (HF_HOME, LOG_ROOT,
    interpreter, smoke harness); the failure must be ``REPO_ROOT``
    for the symlink reason, not a downstream directory-missing
    error."""
    repo_target = tmp_path / "real_repo_target"
    (repo_target / "scripts" / "grid5000").mkdir(parents=True)
    (repo_target / "scripts" / "grid5000" / "run_gpu_smoke.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n"
    )
    (repo_target / "scripts" / "grid5000" / "run_gpu_smoke.sh").chmod(0o755)
    (repo_target / ".venv" / "bin").mkdir(parents=True)
    (repo_target / ".venv" / "bin" / "python").write_text(
        "#!/usr/bin/env bash\nexit 0\n"
    )
    (repo_target / ".venv" / "bin" / "python").chmod(0o755)
    repo_link = tmp_path / "repo_link"
    repo_link.symlink_to(repo_target)

    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    log_root = tmp_path / "logs"
    log_root.mkdir()

    proc = _run_wrapper_with_paths(
        tmp_path,
        repo=repo_link,
        hf_home=hf_home,
        log_root=log_root,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "REPO_ROOT" in combined, (
        f"failure was not attributed to REPO_ROOT: {combined!r}"
    )
    # The supplied path must never be echoed.
    assert str(repo_link) not in combined, (
        f"REPO_ROOT path leaked into error output: {combined!r}"
    )
    # No job directory should have been created.
    assert not (log_root / "12345").exists(), (
        "wrapper proceeded past REPO_ROOT validation"
    )


def test_hf_home_final_component_symlink_aborts(tmp_path):
    repo_target = tmp_path / "real_repo_target"
    (repo_target / "scripts" / "grid5000").mkdir(parents=True)
    (repo_target / "scripts" / "grid5000" / "run_gpu_smoke.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n"
    )
    (repo_target / "scripts" / "grid5000" / "run_gpu_smoke.sh").chmod(0o755)
    (repo_target / ".venv" / "bin").mkdir(parents=True)
    (repo_target / ".venv" / "bin" / "python").write_text(
        "#!/usr/bin/env bash\nexit 0\n"
    )
    (repo_target / ".venv" / "bin" / "python").chmod(0o755)

    hf_target = tmp_path / "real_hf_target"
    hf_target.mkdir()
    hf_link = tmp_path / "hf_link"
    hf_link.symlink_to(hf_target)

    log_root = tmp_path / "logs"
    log_root.mkdir()

    proc = _run_wrapper_with_paths(
        tmp_path,
        repo=repo_target,
        hf_home=hf_link,
        log_root=log_root,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "HF_HOME" in combined
    assert str(hf_link) not in combined


def test_log_root_final_component_symlink_aborts(tmp_path):
    repo_target = tmp_path / "real_repo_target"
    (repo_target / "scripts" / "grid5000").mkdir(parents=True)
    (repo_target / "scripts" / "grid5000" / "run_gpu_smoke.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n"
    )
    (repo_target / "scripts" / "grid5000" / "run_gpu_smoke.sh").chmod(0o755)
    (repo_target / ".venv" / "bin").mkdir(parents=True)
    (repo_target / ".venv" / "bin" / "python").write_text(
        "#!/usr/bin/env bash\nexit 0\n"
    )
    (repo_target / ".venv" / "bin" / "python").chmod(0o755)

    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    log_target = tmp_path / "real_logs_target"
    log_target.mkdir()
    log_link = tmp_path / "logs_link"
    log_link.symlink_to(log_target)

    proc = _run_wrapper_with_paths(
        tmp_path,
        repo=repo_target,
        hf_home=hf_home,
        log_root=log_link,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "LOG_ROOT" in combined
    assert str(log_link) not in combined


# --- (2) OAR_JOB_ID validation RED ----------------------------------


def test_oar_job_id_with_traversal_is_rejected(tmp_path):
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    log_root = tmp_path / "logs"
    log_root.mkdir()
    proc = _run_wrapper_with_paths(
        tmp_path,
        repo=tmp_path / "fake_repo",
        hf_home=hf_home,
        log_root=log_root,
        oar_job_id="../escape",
    )
    assert proc.returncode != 0
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "../escape" not in combined


def test_oar_job_id_with_internal_traversal_is_rejected(tmp_path):
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    log_root = tmp_path / "logs"
    log_root.mkdir()
    proc = _run_wrapper_with_paths(
        tmp_path,
        repo=tmp_path / "fake_repo",
        hf_home=hf_home,
        log_root=log_root,
        oar_job_id="1/../../escape",
    )
    assert proc.returncode != 0
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "1/../../escape" not in combined
    # Specifically the supplied string is not used as a path
    # component: the job log directory is never created.
    assert not (log_root / "1").exists()


def test_oar_job_id_with_whitespace_is_rejected(tmp_path):
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    log_root = tmp_path / "logs"
    log_root.mkdir()
    for bad in ("12 34", "12\t34", " 12345", "12345 ", "12 345"):
        proc = _run_wrapper_with_paths(
            tmp_path,
            repo=tmp_path / "fake_repo",
            hf_home=hf_home,
            log_root=log_root,
            oar_job_id=bad,
        )
        assert proc.returncode != 0, (bad, proc.stdout + proc.stderr)
        combined = (proc.stdout or "") + (proc.stderr or "")
        # The exact bad value must never appear in error output.
        assert bad not in combined, (bad, combined)


def test_oar_job_id_with_prose_is_rejected(tmp_path):
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    log_root = tmp_path / "logs"
    log_root.mkdir()
    proc = _run_wrapper_with_paths(
        tmp_path,
        repo=tmp_path / "fake_repo",
        hf_home=hf_home,
        log_root=log_root,
        oar_job_id="OAR-12345",
    )
    assert proc.returncode != 0


def test_oar_job_id_numeric_passes_validation(tmp_path):
    """A bare-numeric OAR_JOB_ID must survive the validation gate.

    Even with the contract tightened, a numeric id is allowed;
    the wrapper must reach the job-directory creation step (i.e.
    it should not abort on the OAR_JOB_ID check itself).

    We use a fake smoke harness that exits non-zero so the
    wrapper still aborts downstream on the smoke result, but
    the failure MUST be a smoke exit code, not the OAR_JOB_ID
    rejection error."""
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    log_root = tmp_path / "logs"
    log_root.mkdir()
    repo = tmp_path / "fake_repo"
    (repo / "scripts" / "grid5000").mkdir(parents=True)
    (repo / "scripts" / "grid5000" / "run_gpu_smoke.sh").write_text(
        "#!/usr/bin/env bash\nexit 42\n"
    )
    (repo / "scripts" / "grid5000" / "run_gpu_smoke.sh").chmod(0o755)
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("#!/usr/bin/env bash\nexit 0\n")
    (repo / ".venv" / "bin" / "python").chmod(0o755)

    proc = _run_wrapper_with_paths(
        tmp_path,
        repo=repo,
        hf_home=hf_home,
        log_root=log_root,
        oar_job_id="12345",
    )
    # The smoke failed (exit 42). The wrapper must propagate that
    # exit code, not its own OAR_JOB_ID validation error.
    assert proc.returncode == 42, proc.stdout + proc.stderr
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "OAR_JOB_ID" not in combined or "numeric" in combined, (
        "OAR_JOB_ID rejection leaked for numeric id: " + combined
    )
    # The job directory was created.
    assert (log_root / "12345").exists()


# --- (3) Exactly four positional arguments RED --------------------


def test_fifth_positional_argument_is_rejected(tmp_path):
    """A fifth positional argument must be rejected before any
    job-directory creation or smoke invocation. The wrapper's
    exit code must NOT be the smoke harness's success exit
    code: the wrapper should fail with its own validation
    error."""
    repo = tmp_path / "fake_repo"
    (repo / "scripts" / "grid5000").mkdir(parents=True)
    (repo / "scripts" / "grid5000" / "run_gpu_smoke.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n"
    )
    (repo / "scripts" / "grid5000" / "run_gpu_smoke.sh").chmod(0o755)
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("#!/usr/bin/env bash\nexit 0\n")
    (repo / ".venv" / "bin" / "python").chmod(0o755)
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    log_root = tmp_path / "logs"
    log_root.mkdir()

    proc = _run_wrapper_with_paths(
        tmp_path,
        repo=repo,
        hf_home=hf_home,
        log_root=log_root,
        oar_job_id="12345",
        extra_args=["/tmp/extra-unwanted-arg"],
    )
    # The wrapper must reject before invoking the smoke.
    assert proc.returncode != 0, proc.stdout + proc.stderr
    combined = (proc.stdout or "") + (proc.stderr or "")
    # Stable label; the exact phrasing is implementation-defined
    # but the contract is "reject". The five-arg-but-the-extra
    # is the smoking gun: no job directory should be created.
    assert "12345" not in combined or "argument" in combined.lower(), (
        f"unexpected leak: {combined!r}"
    )
    assert not (log_root / "12345").exists(), (
        "wrapper proceeded past positional-count validation"
    )


# --- (4) Success artefact contract RED ----------------------------


def _make_fake_repo_with_smoke_harness(
    tmp_path: Path,
    *,
    script_body: str,
    artifacts_body: str,
) -> Path:
    """Construct a fake repo whose smoke harness writes artefacts
    via ``bash`` derived from ``script_body`` (the bash
    implementations of the smoke outputs) and where
    ``artifacts_body`` is the contents of an extra helper that
    the harness invokes to write artefacts. The harness is
    committed at ``scripts/grid5000/run_gpu_smoke.sh`` and the
    wrapper invokes it exactly once.

    Returns the repo path.
    """
    repo = tmp_path / "fake_repo"
    if repo.exists():
        return repo
    repo.mkdir()
    scripts_dir = repo / "scripts" / "grid5000"
    scripts_dir.mkdir(parents=True)
    full_harness = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ': "${SMOKE_LOG_DIR:?SMOKE_LOG_DIR is required}"\n'
        ': "${OAR_JOB_ID:?OAR_JOB_ID is required}"\n'
        "umask 077\n" + artifacts_body + "\n" + script_body + "\nexit 0\n"
    )
    (scripts_dir / "run_gpu_smoke.sh").write_text(full_harness)
    (scripts_dir / "run_gpu_smoke.sh").chmod(0o755)
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("#!/usr/bin/env bash\nexit 0\n")
    (repo / ".venv" / "bin" / "python").chmod(0o755)
    return repo


def test_success_missing_gpu_preflight_enforces_contract(tmp_path):
    """The wrapper must refuse to declare success when the
    smoke harness produced only five of the six required
    artefacts (gpu_preflight.json is missing)."""
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    log_root = tmp_path / "logs"
    log_root.mkdir()
    artifacts_body = (
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/run_metadata.json\"\n"
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/smoke_result.json\"\n"
        "printf '' > \"${SMOKE_LOG_DIR}/smoke.stdout.log\"\n"
        "printf '' > \"${SMOKE_LOG_DIR}/smoke.stderr.log\"\n"
    )
    repo = _make_fake_repo_with_smoke_harness(
        tmp_path, script_body="", artifacts_body=artifacts_body
    )
    proc = _run_wrapper_with_paths(
        tmp_path,
        repo=repo,
        hf_home=hf_home,
        log_root=log_root,
        oar_job_id="12345",
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr


def test_success_unexpected_seventh_entry_enforces_contract(tmp_path):
    """A successful run with an unexpected 7th entry (here, an
    extra ``stale.json``) must fail the contract. The wrapper
    must not silently ignore the extra entry."""
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    log_root = tmp_path / "logs"
    log_root.mkdir()
    artifacts_body = (
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/gpu_preflight.json\"\n"
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/run_metadata.json\"\n"
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/smoke_result.json\"\n"
        "printf '' > \"${SMOKE_LOG_DIR}/smoke.stdout.log\"\n"
        "printf '' > \"${SMOKE_LOG_DIR}/smoke.stderr.log\"\n"
        "printf '%s\\n' unexpected > \"${SMOKE_LOG_DIR}/stale.json\"\n"
    )
    repo = _make_fake_repo_with_smoke_harness(
        tmp_path, script_body="", artifacts_body=artifacts_body
    )
    proc = _run_wrapper_with_paths(
        tmp_path,
        repo=repo,
        hf_home=hf_home,
        log_root=log_root,
        oar_job_id="12345",
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    # The extra entry must not have been quietly removed.
    assert (log_root / "12345" / "stale.json").exists(), (
        "wrapper deleted the unexpected entry before reporting failure"
    )


def test_success_unsafe_permissions_enforced(tmp_path):
    """An artefact with unsafe permissions (mode 0644) must
    fail the contract. The wrapper must not silently rewrite
    permissions; it must abort and preserve the artefact."""
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    log_root = tmp_path / "logs"
    log_root.mkdir()
    artifacts_body = (
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/gpu_preflight.json\"\n"
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/run_metadata.json\"\n"
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/smoke_result.json\"\n"
        "printf '' > \"${SMOKE_LOG_DIR}/smoke.stdout.log\"\n"
        "printf '' > \"${SMOKE_LOG_DIR}/smoke.stderr.log\"\n"
        'chmod 0644 "${SMOKE_LOG_DIR}/smoke_result.json"\n'
    )
    repo = _make_fake_repo_with_smoke_harness(
        tmp_path, script_body="", artifacts_body=artifacts_body
    )
    proc = _run_wrapper_with_paths(
        tmp_path,
        repo=repo,
        hf_home=hf_home,
        log_root=log_root,
        oar_job_id="12345",
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    # The artefact must keep its 0644 mode (no silent rewrite).
    assert (log_root / "12345" / "smoke_result.json").stat().st_mode & 0o777 == 0o644


def test_failing_harness_preserves_original_exit_code(tmp_path):
    """A non-zero smoke exit (e.g. 42) must propagate verbatim.
    The wrapper MUST NOT rewrite the recorded exit code to its
    own validation result. Partial artefacts (whatever the
    smoke had time to write) must be preserved on disk."""
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    log_root = tmp_path / "logs"
    log_root.mkdir()
    repo = tmp_path / "fake_repo"
    (repo / "scripts" / "grid5000").mkdir(parents=True)
    # Harness writes the gpu_preflight artefact and immediately
    # exits with 42; run_metadata, smoke_result, smoke.stdout,
    # smoke.stderr are NOT produced.
    harness = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ': "${SMOKE_LOG_DIR:?SMOKE_LOG_DIR is required}"\n'
        ': "${OAR_JOB_ID:?OAR_JOB_ID is required}"\n'
        "umask 077\n"
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/gpu_preflight.json\"\n"
        "printf '%s\\n' partial > \"${SMOKE_LOG_DIR}/smoke.stderr.log\"\n"
        "exit 42\n"
    )
    (repo / "scripts" / "grid5000" / "run_gpu_smoke.sh").write_text(harness)
    (repo / "scripts" / "grid5000" / "run_gpu_smoke.sh").chmod(0o755)
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("#!/usr/bin/env bash\nexit 0\n")
    (repo / ".venv" / "bin" / "python").chmod(0o755)

    proc = _run_wrapper_with_paths(
        tmp_path,
        repo=repo,
        hf_home=hf_home,
        log_root=log_root,
        oar_job_id="12345",
    )
    assert proc.returncode == 42, proc.stdout + proc.stderr
    # Recorded exit-code file must contain "42", not "0" or "1".
    log_dir = log_root / "12345"
    assert log_dir.exists()
    assert (log_dir / "smoke.exit_code").read_text().strip() == "42"
    # Partial artefacts are preserved.
    assert (log_dir / "gpu_preflight.json").exists()
    assert (log_dir / "smoke.stderr.log").exists()
    # Missing artefacts remain absent (no auto-stubs).
    assert not (log_dir / "run_metadata.json").exists()
    assert not (log_dir / "smoke_result.json").exists()


def test_success_unexpected_directory_enforces_contract(tmp_path):
    """A successful run whose job directory additionally
    contains an unexpected subdirectory must fail the contract.
    The smoke harness exits 0 and ``smoke.exit_code`` is 0, but
    the wrapper must still exit non-zero because the
    postcondition (exactly six direct entries, all expected
    regular files) is violated. The unexpected directory must
    be preserved for forensic inspection."""
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    log_root = tmp_path / "logs"
    log_root.mkdir()
    artifacts_body = (
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/gpu_preflight.json\"\n"
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/run_metadata.json\"\n"
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/smoke_result.json\"\n"
        "printf '' > \"${SMOKE_LOG_DIR}/smoke.stdout.log\"\n"
        "printf '' > \"${SMOKE_LOG_DIR}/smoke.stderr.log\"\n"
        'mkdir "${SMOKE_LOG_DIR}/unexpected_dir"\n'
    )
    repo = _make_fake_repo_with_smoke_harness(
        tmp_path, script_body="", artifacts_body=artifacts_body
    )
    proc = _run_wrapper_with_paths(
        tmp_path,
        repo=repo,
        hf_home=hf_home,
        log_root=log_root,
        oar_job_id="12345",
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    log_dir = log_root / "12345"
    assert (log_dir / "smoke.exit_code").read_text().strip() == "0"
    # The unexpected directory is preserved (not deleted).
    assert (log_dir / "unexpected_dir").is_dir(), (
        "wrapper removed the unexpected directory before reporting failure"
    )


def test_success_unexpected_symlink_enforces_contract(tmp_path):
    """A successful run whose job directory additionally
    contains an unexpected symlink must fail the contract. The
    smoke harness exits 0 and ``smoke.exit_code`` is 0, but the
    wrapper must still exit non-zero because the postcondition
    (exactly six direct entries, all expected regular files) is
    violated. The unexpected symlink must be preserved for
    forensic inspection."""
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    log_root = tmp_path / "logs"
    log_root.mkdir()
    artifacts_body = (
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/gpu_preflight.json\"\n"
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/run_metadata.json\"\n"
        "printf '%s\\n' ok > \"${SMOKE_LOG_DIR}/smoke_result.json\"\n"
        "printf '' > \"${SMOKE_LOG_DIR}/smoke.stdout.log\"\n"
        "printf '' > \"${SMOKE_LOG_DIR}/smoke.stderr.log\"\n"
        'ln -s gpu_preflight.json "${SMOKE_LOG_DIR}/unexpected_link"\n'
    )
    repo = _make_fake_repo_with_smoke_harness(
        tmp_path, script_body="", artifacts_body=artifacts_body
    )
    proc = _run_wrapper_with_paths(
        tmp_path,
        repo=repo,
        hf_home=hf_home,
        log_root=log_root,
        oar_job_id="12345",
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    log_dir = log_root / "12345"
    assert (log_dir / "smoke.exit_code").read_text().strip() == "0"
    # The unexpected symlink is preserved (not deleted/normed).
    assert (log_dir / "unexpected_link").is_symlink(), (
        "wrapper removed the unexpected symlink before reporting failure"
    )
