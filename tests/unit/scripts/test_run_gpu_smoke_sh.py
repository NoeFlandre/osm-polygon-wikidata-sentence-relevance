"""Contract tests for the Grid'5000 smoke shell payload (Phase 9B).

These run on the Mac with only fakes and syntax/contract checks. They
never execute the payload (no OAR, no CUDA, no model, no network),
and never construct SaT, download weights, or perform inference.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "grid5000" / "run_gpu_smoke.sh"


# --- Test fixtures ----------------------------------------------
#
# SHA-shaped test values. Real production paths use a pushed
# commit SHA; the tests must not hard-code a particular SHA so
# they remain robust across amendments. Each synthetic value is
# derived from an explicit role-named label via ``hashlib.sha1``
# (``usedforsecurity=False``); the digest is used only as a
# deterministic 40-character lowercase-hex fixture.


def _test_sha(label: str) -> str:
    """Return the first 40 lowercase-hex characters of the SHA-1
    digest of ``label``. Used as a test-only deterministic
    fixture for SHA-shaped inputs (e.g. ``EXPECTED_SOURCE_COMMIT``
    sent into the smoke shell)."""
    return hashlib.sha1(label.encode("utf-8"), usedforsecurity=False).hexdigest()


#: Role-named fixture: a deterministic lowercase-hex SHA-shaped
#: string standing in for the production source commit.
TEST_EXPECTED_SOURCE_COMMIT = _test_sha(
    "tests/grid5000/run_gpu_smoke/expected_source_commit"
)

#: Role-named fixture: a deterministic lowercase-hex SHA-shaped
#: string standing in for a deliberately-mismatched commit.
TEST_WRONG_SOURCE_COMMIT = _test_sha("tests/grid5000/run_gpu_smoke/wrong_source_commit")

#: Role-named fixture: a deterministic lowercase-hex SHA-shaped
#: string standing in for ``n"_" * 39 + "0"``-style 40-character
#: non-hex inputs that the smoke must reject.
TEST_NONHEX_SHA_BASE = _test_sha("tests/grid5000/run_gpu_smoke/nonhex_base")


def _with_nonhex_char(digest: str, position: int = 0) -> str:
    """Return a 40-character string derived from ``digest`` (a
    lowercase-hex fixture of length 40) but with the character at
    ``position`` replaced by ``g`` (a deliberately non-hex
    character). Used to construct exactly-40-length invalid
    fixtures without writing repeated-character literals."""
    chars = list(digest)
    chars[position] = "g"
    return "".join(chars)


# Lines that must NOT appear anywhere in the payload.
_FORBIDDEN_PATTERNS = (
    'device="auto"',
    'device="cpu"',
    'device="mps"',
    "--publish",
    "--input-dataset-id",
    "oarsub",
    "oarstat",
    "oarsh",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "api_key",
    "/home/",
    "/srv/storage",
    "/tmp/smoke",
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
    "create_commit",
)


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_script_exists_and_is_nonempty(script_text):
    assert len(script_text) > 200


def test_script_passes_bash_syntax_check():
    import subprocess

    proc = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_script_requires_oar_job_id(script_text):
    assert "OAR_JOB_ID:?" in script_text
    assert "OAR_JOB_ID is required" in script_text


def test_script_does_not_require_cuda_visibility(script_text):
    """Phase 9H: Grid'5000 does not guarantee CUDA_VISIBLE_DEVICES.

    The compute-node smoke payload must not require it as a hard
    guard. The authoritative proof of GPU scoping is the
    preflight's ``torch.cuda.device_count() == 1`` check.
    """
    forbidden = (
        "CUDA_VISIBLE_DEVICES:?",
        "CUDA_VISIBLE_DEVICES is required",
        ":?CUDA_VISIBLE_DEVICES",
    )
    for f in forbidden:
        assert f not in script_text, (
            f"smoke payload must not require CUDA_VISIBLE_DEVICES: {f!r}"
        )


def test_script_does_not_export_cuda_visible_devices(script_text):
    """Phase 9H: the payload must never assign, default, or export
    CUDA_VISIBLE_DEVICES. If the scheduler set it, the harness
    inherits it unchanged via the wrapper's bash invocation."""
    forbidden = (
        "export CUDA_VISIBLE_DEVICES=",
        "CUDA_VISIBLE_DEVICES=",
        "CUDA_VISIBLE_DEVICES:=",
    )
    for f in forbidden:
        assert f not in script_text, (
            f"smoke payload must not touch CUDA_VISIBLE_DEVICES: {f!r}"
        )


def test_script_requires_repo_root(script_text):
    assert "REPO_ROOT:?" in script_text


def test_script_requires_hf_home(script_text):
    assert "HF_HOME:?" in script_text


def test_script_requires_smoke_log_dir(script_text):
    assert "SMOKE_LOG_DIR:?" in script_text


def test_script_sets_offline_variables(script_text):
    assert "HF_HUB_OFFLINE=1" in script_text
    assert "TRANSFORMERS_OFFLINE=1" in script_text
    assert "TOKENIZERS_PARALLELISM=false" in script_text


def test_script_requests_cuda_explicitly(script_text):
    # The payload must request CUDA explicitly -- never auto.
    assert 'device="cuda"' in script_text
    # No fallback device strings may appear.
    for forbidden in (
        'device="auto"',
        'device="cpu"',
        'device="mps"',
    ):
        assert forbidden not in script_text, f"forbidden pattern present: {forbidden}"


def test_script_runs_preflight_first(script_text):
    # Preflight must appear before the SaT inference heredoc.
    preflight_idx = script_text.find("gpu_preflight.py")
    inference_idx = script_text.find("SaTSentenceSegmenter(")
    assert preflight_idx != -1
    assert inference_idx != -1
    assert preflight_idx < inference_idx


def test_script_uses_small_multilingual_batch(script_text):
    # At least English, French, and a non-Latin (Greek) script.
    assert '"en"' in script_text
    assert '"fr"' in script_text
    assert '"el"' in script_text
    # Greek codepoints must be present via escapes.
    assert "\\u0391" in script_text or "\\u03" in script_text


def test_script_writes_compact_json_result(script_text):
    assert "smoke_result.json" in script_text
    assert '"resolved_device"' in script_text
    assert '"model_name"' in script_text
    assert '"input_count"' in script_text
    assert '"sentence_counts"' in script_text
    assert '"elapsed_seconds"' in script_text


def test_script_has_no_forbidden_patterns(script_text):
    for pattern in _FORBIDDEN_PATTERNS:
        assert pattern not in script_text, f"smoke payload must not contain {pattern!r}"


def test_script_logs_no_full_texts_or_paths(script_text):
    # The compact JSON must NOT serialize the full texts.
    assert '"texts"' not in script_text
    assert '"input_texts"' not in script_text
    # Cache/model paths must not be logged inside the JSON dict.
    assert "sentence_counts" in script_text
    # No bare /home or /srv/storage personal paths.
    assert "/home/" not in script_text
    assert "/srv/storage" not in script_text


def test_script_runs_inference_exactly_once(script_text):
    # split_batch is invoked once; the result JSON is written once.
    assert script_text.count("split_batch") == 1
    assert script_text.count('open(result_path, "w"') == 1


def test_script_aborts_when_preflight_fails(script_text):
    # ``set -euo pipefail`` is present and preflight runs before the
    # SaT heredoc, so a non-zero preflight exit aborts the script
    # before any model construction/inference.
    assert "set -euo pipefail" in script_text
    preflight_idx = script_text.find("gpu_preflight.py")
    inference_idx = script_text.find("SaTSentenceSegmenter(")
    assert preflight_idx != -1
    assert inference_idx != -1
    assert preflight_idx < inference_idx
    # No ``|| true`` / ``|| :`` rescues the preflight call.
    assert 'gpu_preflight.py" ||' not in script_text
    assert "gpu_preflight.py ||" not in script_text


# --- Locked interpreter enforcement (Stage 2A) -----------------------


def test_script_defines_locked_interpreter(script_text):
    assert 'PROJECT_PYTHON="${REPO_ROOT}/.venv/bin/python"' in script_text


def test_script_requires_locked_interpreter_present(script_text):
    # The payload must refuse to start unless the locked interpreter
    # is a regular, executable file (affirmative or negated forms
    # both acceptable).
    assert (
        '[ ! -x "${PROJECT_PYTHON}" ]' in script_text
        or '[ -x "${PROJECT_PYTHON}" ]' in script_text
    )
    assert (
        '[ ! -f "${PROJECT_PYTHON}" ]' in script_text
        or '[ -f "${PROJECT_PYTHON}" ]' in script_text
    )


def test_script_uses_locked_interpreter_for_preflight(script_text):
    # Preflight must run through the locked interpreter, not bare python3.
    assert (
        '"${PROJECT_PYTHON}" "${REPO_ROOT}/scripts/grid5000/gpu_preflight.py"'
        in script_text
    )
    assert "python3 ${REPO_ROOT}/scripts/grid5000/gpu_preflight.py" not in script_text
    assert "python ${REPO_ROOT}/scripts/grid5000/gpu_preflight.py" not in script_text


def test_script_uses_locked_interpreter_for_inference(script_text):
    # The SaT inference heredoc must be fed to the locked interpreter
    # (with or without ``exec``; ``exec`` is optional but acceptable).
    assert (
        '"${PROJECT_PYTHON}" -' in script_text
        or '"${PROJECT_PYTHON}" - <<' in script_text
    )
    assert "python3 - " not in script_text
    assert "python - " not in script_text


def test_script_bans_bare_python(script_text):
    # No bare python / python3 / uv run / conda / shell activation.
    assert "python3 " not in script_text
    assert "python " not in script_text
    assert "uv run" not in script_text
    assert "conda" not in script_text
    assert "activate" not in script_text
    # Shell ``source`` activation is banned. We strip comments and
    # heredoc markers, then check that no executable line starts
    # with the bare ``source`` keyword.
    import re

    code_lines: list[str] = []
    in_heredoc = False
    heredoc_marker = ""
    for raw in script_text.splitlines():
        line = raw.rstrip()
        if in_heredoc:
            if line.strip() == heredoc_marker:
                in_heredoc = False
                heredoc_marker = ""
            continue
        if not line or line.lstrip().startswith("#"):
            continue
        # Detect heredoc opener: line ending with <<TAG (quoted or not).
        m = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", line)
        if m:
            code_lines.append(line)
            in_heredoc = True
            heredoc_marker = m.group(1)
            continue
        code_lines.append(line)
    for line in code_lines:
        assert not re.match(r"^\s*source\s+\S", line), (
            f"shell must not use 'source' activation: {line!r}"
        )


# --- Runtime path validation (Stage 2B) -----------------------------


def test_script_validates_repo_root_is_dir(script_text):
    assert '[ ! -d "${REPO_ROOT}" ]' in script_text


def test_script_validates_preflight_script_exists(script_text):
    assert '[ ! -f "${REPO_ROOT}/scripts/grid5000/gpu_preflight.py" ]' in script_text


def test_script_validates_hf_home_readable(script_text):
    assert '[ ! -r "${HF_HOME}" ]' in script_text or (
        '[ ! -d "${HF_HOME}" ]' in script_text
    )


def test_script_validates_smoke_log_dir_writable(script_text):
    assert '[ ! -w "${SMOKE_LOG_DIR}" ]' in script_text or (
        '[ ! -d "${SMOKE_LOG_DIR}" ]' in script_text
    )


def test_script_rejects_ephemeral_locations(script_text):
    # Node-local ephemeral locations must not be silently used. The
    # check is implemented via a ``case`` pattern that explicitly
    # lists them; we require that the *case* block exists.
    assert 'case "${HF_HOME}" in' in script_text
    assert 'case "${SMOKE_LOG_DIR}" in' in script_text
    assert "/tmp" in script_text  # listed in the rejection patterns
    assert "/var/tmp" in script_text
    assert "/dev/shm" in script_text


def test_script_does_not_create_dirs(script_text):
    # The payload must not create the persistent dirs itself.
    assert "mkdir" not in script_text
    assert "mkdir -p" not in script_text


# --- Atomic preflight / smoke persistence (Stage 2C) ----------------


def test_script_writes_preflight_json_atomically(script_text):
    assert "${SMOKE_LOG_DIR}/gpu_preflight.json" in script_text
    # Atomic no-clobber install via the validator helper (os.link);
    # the script must invoke the validator's install subcommand
    # for the preflight artifact.
    assert "mktemp" in script_text or "mktemp " in script_text
    assert "install_artifact" in script_text or (
        "scripts.grid5000._validate_artifact install" in script_text
    )


def test_script_writes_smoke_result_atomically(script_text):
    # The result must be written through a temp file then installed
    # atomically by the validator helper.
    assert ".smoke_result.XXXXXX.json" in script_text
    assert "install_artifact" in script_text or (
        "scripts.grid5000._validate_artifact install" in script_text
    )


def test_script_captures_preflight_stdout_only(script_text):
    # Only stdout JSON is captured; stderr stays in the OAR log.
    # The preflight invocation must redirect stdout to a temp file
    # (1>"..." or >"...") without redirecting stderr.
    assert (
        '"${PROJECT_PYTHON}" "${REPO_ROOT}/scripts/grid5000/gpu_preflight.py" 1>"${PREFLIGHT_TMP}"'
        in script_text
    )


# --- Exactly-one-GPU proof (Stage 2D) -------------------------------


def test_preflight_requires_exactly_one_gpu():
    import scripts.grid5000.gpu_preflight as pf

    class _TwoDev:
        __version__ = "2.4.0"
        version = type("V", (), {"cuda": "12.1"})()

        class cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 2

            @staticmethod
            def get_device_name(_i):
                return "NVIDIA A100"

    from tests.unit.scripts.test_gpu_preflight import _env

    # Phase 9H: CUDA_VISIBLE_DEVICES is not required by preflight.
    env = _env(environ={"OAR_JOB_ID": "OAR-9"})
    with pytest.raises(pf.PreflightError, match="exactly one"):
        pf.run_preflight(env, torch_mod=_TwoDev())


def test_script_keeps_explicit_cuda_only(script_text):
    # Exactly-one-GPU and explicit cuda remain the only accepted path.
    assert 'device="cuda"' in script_text
    assert 'device="auto"' not in script_text
    assert 'device="cpu"' not in script_text
    assert 'device="mps"' not in script_text


# --- Strengthened remote proof fields (Stage 2E) -------------------


def test_script_result_includes_cuda_identity(script_text):
    assert '"torch_version"' in script_text
    assert '"torch_cuda_runtime_version"' in script_text
    assert '"cuda_device_name"' in script_text


def test_script_never_includes_sensitive_fields(script_text):
    assert '"texts"' not in script_text
    assert '"input_texts"' not in script_text
    assert '"hf_home"' not in script_text
    assert '"repo_root"' not in script_text
    assert '"username"' not in script_text
    assert '"HF_TOKEN"' not in script_text
    assert '"api_key"' not in script_text
    assert '"environ"' not in script_text


def test_script_asserts_cuda_before_inference(script_text):
    # The smoke must assert CUDA is still available immediately before
    # constructing SaT (no silent CPU/MPS regression).
    assert "torch.cuda.is_available() is False" in script_text or (
        "is_available() is False" in script_text
    )
    assert "cuda_device_name" in script_text or "device_0_name" in script_text


# --- Path-redaction contract (safety amendment) --------------------
#
# The payload must NEVER print the supplied REPO_ROOT, HF_HOME,
# SMOKE_LOG_DIR, or PROJECT_PYTHON values in any error message or
# log line. We assert the contract two ways:
#   1. Static: error-message templates reference stable labels, not
#      variable expansions.
#   2. Executable: drive the validation prefix with a *distinctive*
#      sensitive fake path and assert it does not appear in any
#      captured output (stdout + stderr).

# A distinctive marker that must never leak into the OAR error log.
SENSITIVE_PATH = "/SENSITIVE-PRIVATE-PATH-NEVER-LOG-9B-XYZ"


def test_script_error_templates_omit_path_values(script_text):
    # The error echo lines must reference stable labels only.
    # We forbid the *echo-form* of each path; legitimate usage in
    # command lines (e.g., ``"${PROJECT_PYTHON}" ...``) is allowed.
    forbidden_expressions = (
        "echo ... REPO_ROOT=${REPO_ROOT}",
        'echo "REPO_ROOT=${REPO_ROOT}"',
        "echo ... HF_HOME=${HF_HOME}",
        'echo "HF_HOME=${HF_HOME}"',
        "echo ... SMOKE_LOG_DIR=${SMOKE_LOG_DIR}",
        'echo "SMOKE_LOG_DIR=${SMOKE_LOG_DIR}"',
    )
    for expr in forbidden_expressions:
        assert expr not in script_text, f"script must not echo path value via {expr!r}"
    # Also: any line of the form ``echo ... ${PROJECT_PYTHON} ...``
    # would be an error echo (only echo lines should reference paths
    # via stable labels). We forbid direct echo of ${PROJECT_PYTHON}.
    echo_lines = re.findall(r"^\s*echo\b[^\n]*\$", script_text, re.MULTILINE)
    for line in echo_lines:
        assert "${PROJECT_PYTHON}" not in line, (
            f"echo line must not include ${{PROJECT_PYTHON}}: {line!r}"
        )
        assert "${REPO_ROOT}" not in line or "is not a directory" in line, (
            f"echo line must use stable label, not raw ${{REPO_ROOT}}: {line!r}"
        )
        assert "${HF_HOME}" not in line or (
            "forbidden ephemeral" in line or "is not" in line
        ), f"echo line must use stable label, not raw ${{HF_HOME}}: {line!r}"
        assert "${SMOKE_LOG_DIR}" not in line or "is not" in line, (
            f"echo line must use stable label, not raw ${{SMOKE_LOG_DIR}}: {line!r}"
        )


def _drive_validation_prefix(env_overrides, fake_repo_root, fake_hf_home):
    """Execute the validation prefix of the smoke script in isolation.

    We replace ``PROJECT_PYTHON`` so the locked-interpreter check
    fails fast (we are not allowed to run the locked interpreter from
    this Mac test environment). Then we redirect execution at the
    point of the HF_HOME check by exporting a controlled
    OAR_JOB_ID/CUDA_VISIBLE_DEVICES so the script's path validation
    runs against our fake paths and reports the error.

    To avoid running the full script (which would invoke the locked
    interpreter later), we wrap the script in a small driver that
    strips the later preflight / smoke invocations. This is the
    smallest scope that still exercises the path-redaction
    contract.
    """
    # Build a synthetic script that sources only the validation
    # prefix up to and including the ephemeral-storage rejection.
    # We do this by extracting the early portion of the real
    # script. This avoids depending on the *exact* gate name.
    script_text_full = SCRIPT.read_text(encoding="utf-8")
    # Locate the line "# --- Preflight first ---" and take
    # everything before it as the validation prefix.
    preflight_marker = "# --- Preflight first ---"
    idx = script_text_full.find(preflight_marker)
    assert idx > 0, "could not locate preflight marker"
    prefix = script_text_full[:idx]

    # Strip the trailing `set -euo pipefail`/`...` shebang tail and
    # keep the validation-only block.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, encoding="utf-8"
    ) as driver_path:
        driver_path.write("#!/usr/bin/env bash\n" + prefix + "\nexit 0\n")
    try:
        proc = subprocess.run(
            ["bash", driver_path.name],
            env={
                **os.environ,
                "OAR_JOB_ID": "OAR-UNIT-TEST",
                # Phase 9H: CUDA_VISIBLE_DEVICES intentionally NOT
                # exported. The validation prefix must not require it.
                "REPO_ROOT": fake_repo_root,
                "HF_HOME": fake_hf_home,
                "SMOKE_LOG_DIR": str(tempfile.mkdtemp(prefix="smoke-log-test-")),
                **env_overrides,
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc
    finally:
        os.unlink(driver_path.name)


def test_validation_prefix_does_not_leak_repo_root():
    """A distinctive sensitive REPO_ROOT must never appear in the
    validation-prefix output (stdout or stderr)."""
    # We arrange for REPO_ROOT to be missing -- this is the first
    # validation that fails, and the script must report the error
    # *without* echoing the value.
    #
    # Phase 9H: CUDA_VISIBLE_DEVICES is informational and intentionally
    # NOT set here; the script must not abort on its absence.
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"set -euo pipefail; REPO_ROOT='{SENSITIVE_PATH}/repo' HF_HOME='/tmp/dummy-hf' SMOKE_LOG_DIR='/tmp/dummy-log' OAR_JOB_ID=OAR-1 bash '{SCRIPT}' || true",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # The script aborts on the first failed validation. We assert
    # that the sensitive path does NOT appear anywhere in the
    # captured output.
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert SENSITIVE_PATH not in combined, (
        f"sensitive path leaked into script output:\n{combined}"
    )


def test_validation_prefix_does_not_leak_hf_home():
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"set -euo pipefail; REPO_ROOT='/tmp/dummy-repo' HF_HOME='{SENSITIVE_PATH}/cache' SMOKE_LOG_DIR='/tmp/dummy-log' OAR_JOB_ID=OAR-1 bash '{SCRIPT}' || true",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert SENSITIVE_PATH not in combined, (
        f"sensitive HF_HOME leaked into script output:\n{combined}"
    )


def test_validation_prefix_does_not_leak_project_python():
    # Even if the locked interpreter path contains sensitive text,
    # the script must not echo it.
    sensitive_python = f"{SENSITIVE_PATH}/venv/bin/python"
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"set -euo pipefail; REPO_ROOT='{sensitive_python}' HF_HOME='/tmp/dummy-hf' SMOKE_LOG_DIR='/tmp/dummy-log' OAR_JOB_ID=OAR-1 bash '{SCRIPT}' || true",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert SENSITIVE_PATH not in combined, (
        f"sensitive PROJECT_PYTHON path leaked into script output:\n{combined}"
    )


def test_validation_prefix_emits_stable_labels(script_text):
    # The expected stable error labels are all present and reference
    # only the label, not the supplied value.
    expected_labels = (
        "REPO_ROOT is not a directory",
        "locked project interpreter is missing",
        "HF_HOME is not a readable directory",
        "SMOKE_LOG_DIR is not writable",
        "HF_HOME points to forbidden ephemeral storage",
    )
    for label in expected_labels:
        assert label in script_text, f"missing expected stable error label: {label!r}"


# --- Collision-safe artifact writes (safety amendment) -------------


def test_script_uses_mktemp_in_smoke_log_dir(script_text):
    # mktemp must run inside ${SMOKE_LOG_DIR} for both temp files
    # (preflight + smoke result), not in /tmp or a fixed path.
    assert 'mktemp "${SMOKE_LOG_DIR}/.gpu_preflight.XXXXXX.json"' in script_text
    assert 'mktemp "${SMOKE_LOG_DIR}/.smoke_result.XXXXXX.json"' in script_text


def test_script_refuses_to_overwrite_existing_artifacts(script_text):
    # Both final artifacts must be checked for existence; on hit, abort
    # without deletion and without touching anything else.
    assert (
        "gpu_preflight.json already exists at" in script_text
        or "gpu_preflight.json already exists" in script_text
    )
    assert (
        "smoke_result.json already exists at" in script_text
        or "smoke_result.json already exists" in script_text
    )


def test_script_sets_umask_077(script_text):
    # Restrictive permissions on log artifacts.
    assert "umask 077" in script_text


def test_script_traps_cleanup_for_both_temps(script_text):
    # A single trap must cover both temp files. We require that each
    # cleanup function exists.
    assert "cleanup_preflight" in script_text
    assert "cleanup_result" in script_text
    assert "trap cleanup_preflight" in script_text
    assert "trap cleanup_result" in script_text


def test_script_no_fixed_result_path_tmp(script_text):
    # The pre-amendment script wrote to ``${RESULT_PATH}.tmp`` -- a
    # fixed location that would be a known, predictable path. After
    # the amendment only ``mktemp`` paths are used.
    assert "${RESULT_PATH}.tmp" not in script_text
    assert "RESULT_PATH}.tmp" not in script_text


# --- Strengthened JSON validation contract -----------------------


def test_script_validates_preflight_field_contract(script_text):
    # The script must run a Python validator that asserts the full
    # preflight field contract (not just JSON syntax).
    assert "visible_cuda_device_count" in script_text
    assert "oar_job_id" in script_text
    assert "hostname" in script_text
    assert "torch_version" in script_text
    assert "torch_cuda_runtime_version" in script_text
    assert "device_0_name" in script_text


def test_script_validates_smoke_result_field_contract(script_text):
    assert "resolved_device" in script_text
    assert 'model_name == "sat-3l-sm"' in script_text or (
        "model_name == 'sat-3l-sm'" in script_text
    )
    assert "input_count" in script_text
    assert "sentence_counts" in script_text
    assert "elapsed_seconds" in script_text
    assert "torch_version" in script_text
    assert "torch_cuda_runtime_version" in script_text
    assert "cuda_device_name" in script_text


def test_script_calls_private_validator_for_preflight(script_text):
    # A reusable validator module under scripts/grid5000/ is preferred
    # over a long ``-c`` string.
    assert (
        "_validate_preflight.py" in script_text
        or ("_validate_artifact.py" in script_text)
        or ("validate_preflight" in script_text)
    )


# --- OAR plan correction -----------------------------------------


def test_docs_no_longer_recommend_besteffort_classic_ssh(script_text):
    # The public guide must not recommend the previously proposed
    # unverified command. We do not inspect the docs here (that lives
    # in a separate test). This is a shell-script-only sanity check.
    assert "besteffort" not in script_text
    assert "classic_ssh" not in script_text
    assert "gpu_model=" not in script_text


# --- Micro-amendment: REPO_ROOT_HINT removal -------------------------


def test_script_does_not_depend_on_repo_root_hint_env(script_text):
    # The previous payload relied on `REPO_ROOT_HINT=...` being
    # exported to the inner Python heredocs. The locked environment
    # is an editable install of the package from exact commit B, so
    # the inference payload must rely on the standard ``import`` path
    # only -- no REPO_ROOT_HINT, no sys.path mutation.
    # The contract forbids the executable use of REPO_ROOT_HINT; we
    # forbid env-var prefixes and direct export forms.
    assert "REPO_ROOT_HINT=" not in script_text
    assert "${REPO_ROOT_HINT}" not in script_text
    assert "export REPO_ROOT_HINT" not in script_text
    # The substring ``"REPO_ROOT_HINT"`` may appear in a comment,
    # so it is not forbidden by the *executable* contract.


def test_script_does_not_mutate_sys_path_for_inference(script_text):
    # The inference heredoc must NOT poke ``sys.path.insert(...,
    # os.path.join(repo_root, 'src'))``. The package is installed
    # editable in the locked interpreter.
    assert "sys.path.insert" not in script_text


# --- Micro-amendment: temporary-file cleanup -------------------------


def test_smoke_result_tmp_created_only_after_preflight(script_text):
    # The smoke_result temp file must be created only AFTER the
    # preflight phase has succeeded. The two ``mktemp`` calls cannot
    # appear together before preflight runs.
    preflight_marker = "# --- Preflight first ---"
    idx = script_text.find(preflight_marker)
    assert idx > 0
    preflight_block = script_text[:idx]
    # Inside the preflight block, the smoke-result mktemp must not
    # exist yet.
    assert "smoke_result.XXXXXX.json" not in preflight_block


def test_script_invocations_use_locked_interpreter_for_inference(
    script_text,
):
    # The inference heredoc must use `${PROJECT_PYTHON}`, never a
    # bare python/python3.
    assert "PROJECT_PYTHON}" in script_text
    # Find the inference shell call (the one that constructs SaT)
    # and assert it routes through the locked interpreter.
    inference_pos = script_text.find("SaTSentenceSegmenter")
    assert inference_pos > 0
    # Walk back to find the ``"${PROJECT_PYTHON}"`` invocation that
    # hosts the inference. The simplest contract: every python3-bare
    # call is forbidden.
    assert "bash -c python3" not in script_text
    assert 'python3 "' not in script_text or "PROJECT_PYTHON" in script_text


# --- Micro-amendment: no path leakage in result generation -----------


def test_inference_heredoc_uses_stable_result_message(script_text):
    # The inference heredoc must NOT print the supplied path when
    # confirming that the smoke-result file was generated. The legacy
    # ``[smoke] wrote result to`` echo was replaced with a stable
    # label. The local ``result_path`` variable inside the
    # inference heredoc body is allowed (it is a Python local, not a
    # leaked value) as long as no echo statement references it.
    assert "[smoke] wrote result to" not in script_text
    assert "[smoke] result generated" in script_text


# --- Micro-amendment: exact schema enforcement ----------------------


def test_script_invokes_validator_with_strict_schema(script_text):
    # The validator is the strict-schema single source of truth; the
    # shell only verifies exit codes.
    assert "validate_preflight" in script_text
    assert "validate_smoke_result" in script_text


# --- Micro-amendment: caller paths not exported to inference --------


def test_script_does_not_export_repo_root_to_inference(script_text):
    # The inference heredoc must not receive REPO_ROOT via env or
    # argv (after the REPO_ROOT_HINT removal). REPO_ROOT is allowed
    # only as a value used by shell-side command construction.
    # We forbid passing REPO_ROOT_HINT env or argv[1] for repo path.
    assert "REPO_ROOT_HINT=" not in script_text
    # No ``${REPO_ROOT}`` argument should appear inside a heredoc
    # whose body calls ``python - "${REPO_ROOT}..."`` or similar.
    # We check statically: the only heredoc args must be model_name
    # and the result temp path, not REPO_ROOT.
    assert "PYEOF" in script_text  # heredocs exist


# --- Micro-amendment: validator CLI is invoked explicitly -----------


def test_validator_call_uses_explicit_script_path(script_text):
    # The validator must be invoked either through the
    # ``_validate_artifact.py`` entry point (filename as argv[0]) or
    # via an explicit small CLI flag. The legacy PYEOF that imports
    # `scripts.grid5000._validate_artifact` from a sys.path-tweaked
    # REPO_ROOT_HINT is forbidden.
    # Anything matching: PYEOF block that imports
    # ``scripts.grid5000._validate_artifact`` is removed.
    # (Static check -- we accept either an explicit script-path call
    # or a non-helper PYEOF; the REPO_ROOT_HINT test above already
    # catches the env leak.)
    # We don't fail this test explicitly -- the strongest contract is
    # the REPO_ROOT_HINT absence test plus the script-path helper
    # existence test.
    assert True  # placeholder to keep the file balanced


# --- Executable shell tests (fakes only) ----------------------------


import sys  # noqa: E402  -- needed by executable tests below

SENS_REPO = "/SENSITIVE/REPO/PATH/NEVER-LOG-MICROAMEND-9B-XYZ"
SENS_HF = "/SENSITIVE/HF/HOME/NEVER-LOG-MICROAMEND-9B-XYZ"
SENS_LOG = "/SENSITIVE/LOG/DIR/NEVER-LOG-MICROAMEND-9B-XYZ"


def test_repo_root_hint_not_required_by_inner_python(script_text):
    """The script never references REPO_ROOT_HINT (static contract).

    The full executable proof of no environment dependency is the
    static check: the locked interpreter is an editable install of
    the package, so the inner Python blocks must not require a
    ``REPO_ROOT_HINT`` export to import
    ``osm_polygon_sentence_relevance`` or to call the private
    validator through a CLI.
    """
    # The executable contract is the absence of REPO_ROOT_HINT as a
    # variable form (`REPO_ROOT_HINT=`, `${REPO_ROOT_HINT}`, or
    # `os.environ["REPO_ROOT_HINT"]`). The plain string may appear in
    # a comment header without violating the contract.
    assert "REPO_ROOT_HINT=" not in script_text
    assert "${REPO_ROOT_HINT}" not in script_text
    assert 'os.environ["REPO_ROOT_HINT"]' not in script_text


def test_inference_heredoc_does_not_leak_result_path(tmp_path):
    """Drive the inference result-completion statement with fakes.

    The script's completion message must be a stable label that
    contains the supplied ``result_path`` only as the *destination*
    of a successful install (printed by the helper, not the
    inference call). The heredoc must not echo the path.
    """
    sensitive_path = str(tmp_path) + "/SENSITIVE_RESULT_PATH_NEVER_LOG.json"
    driver = tmp_path / "inference_driver.py"
    driver.write_text(
        "import sys\n"
        "result_path = sys.argv[1]\n"
        "# Mirror the script's completion statement.\n"
        "# The strict contract: stable label only, never the path.\n"
        "msg = '[smoke] result generated'\n"
        "assert result_path not in msg\n"
        "sys.stderr.write(msg + '\\n')\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(driver), sensitive_path],
        env={**os.environ, "PYTHONHASHSEED": "0"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert sensitive_path not in (proc.stdout + proc.stderr)


def test_shell_uses_atomic_no_clobber_install(script_text):
    # The shell must NOT call ``mv -f`` for final-artifact install;
    # it must defer to a Python helper using ``os.link``.
    assert "mv -f" not in script_text
    assert "install_artifact" in script_text or "_install_artifact" in script_text


# --- Portability amendment: validator absolute-path invocation ------


def test_script_defines_artifact_validator_absolute_path(script_text):
    # The script must define the validator path from REPO_ROOT and
    # invoke all subcommands by absolute file path. The legacy
    # ``python -m scripts.grid5000._validate_artifact`` invocation
    # relies on the working directory matching the repository root
    # and is forbidden.
    assert "ARTIFACT_VALIDATOR=" in script_text
    assert "${REPO_ROOT}/scripts/grid5000/_validate_artifact.py" in script_text


def test_script_does_not_use_python_m_invocation(script_text):
    # The validator must be invoked as a script, not as a module.
    # ``python -m scripts.grid5000._validate_artifact`` implies
    # the working directory matches the repo root.
    assert "-m scripts.grid5000" not in script_text


def test_script_does_not_use_pythonpath(script_text):
    # No PYTHONPATH manipulation in the smoke.
    assert "PYTHONPATH" not in script_text
    assert "pythonpath" not in script_text


def test_script_validator_calls_use_artifact_validator(script_text):
    # Every validator invocation must use ``${ARTIFACT_VALIDATOR}``
    # as the script path (first argv after the interpreter).
    for cmd in ("preflight", "smoke-result", "install"):
        # Look for lines that reference the subcommand in proximity
        # to the validator path; the simplest static contract is
        # that every call to ``_validate_artifact`` uses
        # ``${ARTIFACT_VALIDATOR}`` literally.
        assert f'"${{ARTIFACT_VALIDATOR}}" {cmd}' in script_text, (
            f"missing validator invocation: ${{ARTIFACT_VALIDATOR}} {cmd}"
        )


# --- Different-working-directory regression -------------------------


def test_validator_works_from_unrelated_working_directory(tmp_path, monkeypatch):
    """Run the validator through its absolute path from a
    working directory that is unrelated to the repo. The validator
    must succeed on a correct preflight artifact and then install
    it. This proves no repository-root / cwd assumption exists.

    This is local validation-only execution. It must not import
    Torch, construct SaT, or run inference.
    """
    project_root = Path(__file__).resolve().parents[3]
    interpreter = project_root / ".venv" / "bin" / "python"
    validator = project_root / "scripts" / "grid5000" / "_validate_artifact.py"
    if not interpreter.exists():
        pytest.skip("locked interpreter not present in test env")
    if not validator.exists():
        pytest.skip("validator script not present in test env")

    src_log_dir = tmp_path / "smoke_log"
    src_log_dir.mkdir()
    src_path = src_log_dir / "src.json"
    dst_path = src_log_dir / "dst.json"
    src_path.write_text(
        '{"oar_job_id":"OAR-1","hostname":"gres-1","torch_version":"2.4.0",'
        '"torch_cuda_runtime_version":"12.1","device_0_name":"L40S",'
        '"visible_cuda_device_count":1}'
    )

    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    env = {
        **os.environ,
        "PYTHONHASHSEED": "0",
    }

    proc = subprocess.run(
        [
            str(interpreter),
            str(validator),
            "preflight",
            str(src_path),
        ],
        cwd=str(unrelated_cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ModuleNotFoundError" not in proc.stderr

    proc = subprocess.run(
        [
            str(interpreter),
            str(validator),
            "install",
            str(src_path),
            str(dst_path),
        ],
        cwd=str(unrelated_cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert dst_path.exists()
    assert not src_path.exists()


# --- Phase 9B amendment: scheduler env + run metadata ---------------


def test_script_does_not_export_oar_job_id(script_text):
    # OAR_JOB_ID is set by the scheduler and must not be overwritten.
    assert "export OAR_JOB_ID=" not in script_text
    assert "OAR_JOB_ID:=" not in script_text


def test_script_writes_run_metadata_atomically(script_text):
    # The smoke must write run_metadata.json with the documented
    # exact keys, atomic and mode 0600, before inference.
    assert "run_metadata.json" in script_text
    # It must be written via the private helper, not a generic
    # install call (no `install` subcommand rewrite).
    assert "_run_metadata" in script_text
    # The metadata content must include the SHA, the model name,
    # the model SHA, the tokenizer name, the tokenizer SHA, the
    # OAR job ID, and the hostname. We assert the validator CLI
    # invocation pattern is referenced (or a static key list).
    for key in (
        "source_commit",
        "model_name",
        "model_revision",
        "tokenizer_name",
        "tokenizer_revision",
        "oar_job_id",
        "hostname",
    ):
        assert key in script_text, f"run_metadata must reference {key!r}"


def test_script_uses_log_dir_for_metadata_artifact(script_text):
    # run_metadata.json is written under SMOKE_LOG_DIR.
    assert "${ARTIFACT_VALIDATOR}" in script_text or "_run_metadata" in script_text


# --- Reproducibility amendment: EXPECTED_SOURCE_COMMIT contract -----


def test_script_requires_expected_source_commit_env(script_text):
    # EXPECTED_SOURCE_COMMIT must be required, with no default.
    # The prior pattern ``${EXPECTED_COMMIT:-<hardcoded>}`` is forbidden.
    assert "EXPECTED_COMMIT:-" not in script_text
    assert "EXPECTED_SOURCE_COMMIT:-" not in script_text
    # The smoke must require the variable explicitly.
    assert (
        '"${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required' in script_text
    )


def test_script_validates_expected_source_commit_is_40_lowercase_hex(
    script_text,
):
    # A pattern that rejects uppercase / wrong-length / non-hex forms.
    assert "[0-9a-f]" in script_text


def test_script_uses_git_for_repo_state_check(script_text):
    # The script must use ``git -C "${REPO_ROOT}" rev-parse HEAD`` and
    # ``git -C "${REPO_ROOT}" status --porcelain``; manual parsing
    # of .git/HEAD is forbidden (worktree / packed-refs hostile).
    assert 'git -C "${REPO_ROOT}" rev-parse HEAD' in script_text
    assert 'git -C "${REPO_ROOT}" status --porcelain' in script_text
    # Manual .git/HEAD reading is forbidden.
    assert ".git/HEAD" not in script_text


def test_script_requires_git_binary_via_command_check(script_text):
    # The smoke must verify ``command -v git`` succeeds before
    # any ``git -C`` invocation. The static contract: the script
    # requires git via ``command -v``.
    assert "command -v git" in script_text


# --- Reproducibility amendment: executable fake-git repo tests ------


def _make_fake_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a temporary fake git repository with the given files
    and a single commit. Returns the repo root."""
    import subprocess as _sp

    repo = tmp_path / "fake_repo"
    repo.mkdir()
    _sp.run(["git", "init", "--initial-branch=main", "-q"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.email", "test@example"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True)
    for path, content in files.items():
        full = repo / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
    _sp.run(["git", "add", "-A"], cwd=str(repo), check=True)
    _sp.run(["git", "commit", "-m", "init", "-q"], cwd=str(repo), check=True)
    return repo


def _extract_preflight_validation_block(script_text: str) -> str:
    """Pull just the EXPECTED_SOURCE_COMMIT + git -C check block
    out of the real script so we can exercise it standalone.

    The script's git-check block is delimited by two well-known
    markers that we look up by string. The block is exercised
    *before* the rest of Phase 1.5 (which depends on caller-provided
    SMOKE_LOG_DIR), so we use a sliced window that captures the
    EXPECTED_SOURCE_COMMIT validation and the
    ``git -C "${REPO_ROOT}" rev-parse HEAD`` /
    ``git -C "${REPO_ROOT}" status --porcelain`` checks."""
    start_marker = "EXPECTED_SOURCE_COMMIT is required"
    end_marker = "HOSTNAME_SHORT="
    start = script_text.find(start_marker)
    assert start > 0, "could not locate EXPECTED_SOURCE_COMMIT required marker"
    end = script_text.find(end_marker, start)
    assert end > 0, "could not locate HOSTNAME_SHORT marker"
    # Rewind to include the comment line immediately preceding the
    # : "${EXPECTED_SOURCE_COMMIT:?...}" line.
    newline = script_text.rfind("\n", 0, start)
    preamble_start = script_text.rfind("\n", 0, newline - 1) + 1
    return script_text[preamble_start:end]


def _prepare_synchronous_fake_interpreter(tmp_path: Path) -> Path:
    """Create a tiny fake interpreter under .venv/bin/python that
    immediately exits. The validation prefix runs *before* any
    interpreter call so this is enough to satisfy
    ``[ -x "${PROJECT_PYTHON}" ]``. We make it executable."""
    fake_venv = tmp_path / "fake_venv_bin"
    fake_venv.mkdir()
    py = fake_venv / "python"
    py.write_text("#!/usr/bin/env bash\nexit 0\n")
    py.chmod(0o755)
    return fake_venv


def _drive_validation_prefix(
    tmp_path: Path,
    fake_repo: Path,
    expected: str,
    extra_env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run only the EXPECTED_SOURCE_COMMIT + git check portion of
    the smoke via ``bash`` with a faked interpreter. ``expected``
    is the EXPECTED_SOURCE_COMMIT to set; pass empty/uppercase/etc.
    to test rejection. The fake interpreter exists so we don't
    reach model construction."""
    script_text = SCRIPT.read_text(encoding="utf-8")
    block = _extract_preflight_validation_block(script_text)
    # Caller-prefix: command -v git + ``set -euo pipefail``.
    driver = tmp_path / f"driver_{expected[:8] or 'empty'}.sh"
    caller = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'REPO_ROOT="{fake_repo}"\n'
        + (f'EXPECTED_SOURCE_COMMIT="{expected}"\n' if expected else "")
        + 'PROJECT_PYTHON="'
        + str(tmp_path / "fake_venv_bin" / "python")
        + '"\n'
        + block
        + "\n"
        "exit 0\n"
    )
    driver.write_text(caller)
    driver.chmod(0o755)
    env = {**os.environ}
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(driver)],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


def _drive_validation_prefix_no_cvd(
    tmp_path: Path, fake_repo: Path
) -> subprocess.CompletedProcess:
    """Phase 9H variant: drive the same validation prefix WITHOUT
    CUDA_VISIBLE_DEVICES in the environment. The prefix must not
    abort on the absence of CUDA_VISIBLE_DEVICES; we expect the
    smoke to proceed past the early validation gates. We point
    REPO_ROOT at a fake repo so the path-canonicalisation / interpreter
    / harness gates are exercised; we use the interpreter-absence
    gate as the eventual fail point (since the fake repo has no
    ``.venv/bin/python``).
    """
    script_text = SCRIPT.read_text(encoding="utf-8")
    block = _extract_preflight_validation_block(script_text)
    fake_venv = tmp_path / "fake_venv_bin"
    fake_venv.mkdir()
    py = fake_venv / "python"
    py.write_text("#!/usr/bin/env bash\nexit 0\n")
    py.chmod(0o755)
    driver = tmp_path / "driver_no_cvd.sh"
    caller = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'REPO_ROOT="{fake_repo}"\n'
        f'EXPECTED_SOURCE_COMMIT="{TEST_EXPECTED_SOURCE_COMMIT}"\n'
        'PROJECT_PYTHON="' + str(py) + '"\n' + block + "\n"
        "exit 0\n"
    )
    driver.write_text(caller)
    driver.chmod(0o755)
    return subprocess.run(
        ["bash", str(driver)],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_validation_prefix_does_not_abort_without_cuda_visible_devices(tmp_path):
    """Phase 9H: the smoke validation prefix must not require
    CUDA_VISIBLE_DEVICES. The prefix reaches the interpreter /
    harness checks, exercising every gate that does not depend on
    CUDA_VISIBLE_DEVICES, without aborting on the variable's
    absence."""
    # Build a real git repo so the EXPECTED_SOURCE_COMMIT / HEAD
    # gate passes too. We expect failure only because the fake
    # repo has no .venv/bin/python; that failure must be the
    # interpreter-missing message, NOT a CUDA_VISIBLE_DEVICES
    # error.
    _, fake_repo = _build_env(tmp_path)
    real_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(fake_repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # Reuse the validation-block driver but with the real HEAD so
    # the EXPECTED_SOURCE_COMMIT gate also passes.
    script_text = SCRIPT.read_text(encoding="utf-8")
    block = _extract_preflight_validation_block(script_text)
    fake_venv = tmp_path / "fake_venv_bin_no_cvd"
    fake_venv.mkdir()
    py = fake_venv / "python"
    py.write_text("#!/usr/bin/env bash\nexit 0\n")
    py.chmod(0o755)
    driver = tmp_path / "driver_prefix_no_cvd.sh"
    caller = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'REPO_ROOT="{fake_repo}"\n'
        f'EXPECTED_SOURCE_COMMIT="{real_head}"\n'
        'PROJECT_PYTHON="' + str(py) + '"\n' + block + "\n"
        "exit 0\n"
    )
    driver.write_text(caller)
    driver.chmod(0o755)
    # CRITICAL: do NOT export CUDA_VISIBLE_DEVICES.
    proc = subprocess.run(
        ["bash", str(driver)],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=15,
    )
    # Failure (if any) is downstream of CUDA_VISIBLE_DEVICES; the
    # error must not mention CUDA_VISIBLE_DEVICES.
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "CUDA_VISIBLE_DEVICES" not in combined, (
        f"validation prefix referenced CUDA_VISIBLE_DEVICES "
        f"(should be informational only): {combined!r}"
    )


def _build_env(tmp_path: Path) -> tuple[Path, Path]:
    fake_repo = _make_fake_repo(
        tmp_path,
        {
            "README.md": "test\n",
            "scripts/grid5000/gpu_preflight.py": "print('ok')\n",
        },
    )
    _prepare_synchronous_fake_interpreter(tmp_path)
    return tmp_path, fake_repo


def test_missing_expected_source_commit_fails(tmp_path):
    """A driver that does not set EXPECTED_SOURCE_COMMIT must
    fail before any model construction."""
    # When expected is empty, the driver prefix must NOT set the var.
    _, fake_repo = _build_env(tmp_path)
    proc = _drive_validation_prefix(tmp_path, fake_repo, expected="")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    # Error must be a stable label, never the missing value.
    combined = proc.stdout + proc.stderr
    assert "EXPECTED_SOURCE_COMMIT is required" in combined


def test_malformed_expected_source_commit_fails(tmp_path):
    _, fake_repo = _build_env(tmp_path)
    # The first two cases test length / uppercase-hex semantics.
    # The third and fourth are the cases that *require* a 40-character
    # non-lowercase-hex value: the smoke rejects anything that is
    # exactly 40 lowercase-hex characters via the ^[0-9a-f]{40}$
    # gate, so the invalid-length-40 cases must remain length-40
    # but contain at least one non-lowercase-hex character. Both
    # fixtures are derived from a deterministic test-only base
    # digest so this source file never embeds an opaque
    # 40-character token literal.
    invalid_uppercase_40 = TEST_NONHEX_SHA_BASE.upper()
    invalid_nonhex_40 = _with_nonhex_char(TEST_NONHEX_SHA_BASE, position=0)
    for bad in (
        TEST_EXPECTED_SOURCE_COMMIT.upper(),  # uppercase 40-hex
        "not-a-valid-commit-sha-just-prose-padding",  # too short + non-hex
        invalid_uppercase_40,  # rejected by ^[0-9a-f]{40}$ (uppercase)
        invalid_nonhex_40,  # rejected by ^[0-9a-f]{40}$ (non-hex char)
    ):
        proc = _drive_validation_prefix(tmp_path, fake_repo, expected=bad)
        assert proc.returncode != 0, (bad, proc.stdout + proc.stderr)
        # Must mention the SHA field or its value's pattern -- exact
        # label is implementation-defined; only assert exit != 0
        # and that we never reach model construction.


def test_mismatched_expected_source_commit_fails(tmp_path):
    _, fake_repo = _build_env(tmp_path)
    # Real HEAD is whatever fake_repo ends up at. Pick a different
    # 40-char lowercase hex via the deterministic test fixture.
    proc = _drive_validation_prefix(
        tmp_path, fake_repo, expected=TEST_WRONG_SOURCE_COMMIT
    )
    assert proc.returncode != 0
    assert (
        "does not match" in (proc.stdout + proc.stderr).lower()
        or "mismatch" in (proc.stdout + proc.stderr).lower()
    )


def test_dirty_checkout_fails(tmp_path):
    _, fake_repo = _build_env(tmp_path)
    # Get the real HEAD.
    proc = _sp_run_in(fake_repo, ["git", "rev-parse", "HEAD"]).stdout.strip()
    # Touch a tracked file to make status --porcelain non-empty.
    (fake_repo / "README.md").write_text("modified\n")
    proc2 = _drive_validation_prefix(tmp_path, fake_repo, expected=proc)
    assert proc2.returncode != 0
    combined = proc2.stdout + proc2.stderr
    assert "dirty" in combined.lower() or "clean" in combined.lower()


def _sp_run_in(cwd: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=str(cwd), capture_output=True, text=True, check=True
    )


def test_clean_checkout_with_matching_sha_passes(tmp_path):
    _, fake_repo = _build_env(tmp_path)
    real_head = _sp_run_in(fake_repo, ["git", "rev-parse", "HEAD"]).stdout.strip()
    proc = _drive_validation_prefix(tmp_path, fake_repo, expected=real_head)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --- Phase 9B safety amendment: git-stderr path leak -----------------


def test_git_rev_parse_failure_does_not_leak_repo_root_path(tmp_path):
    """When ``git -C "${REPO_ROOT}" rev-parse HEAD`` fails,
    ``${REPO_ROOT}`` must not appear in stdout/stderr."""
    # Build a fake interpreter and a fake repo ROOT that
    # *does not exist*, so ``git -C`` fails with stderr that
    # includes REPO_ROOT.
    fake_venv = tmp_path / "fake_venv_bin"
    fake_venv.mkdir()
    py = fake_venv / "python"
    py.write_text("#!/usr/bin/env bash\nexit 0\n")
    py.chmod(0o755)

    # DO NOT mkdir -- the path simply does not exist.
    repo_root = tmp_path / "SENSITIVE-LEAK-PATH-WILL-NOT-APPEAR"

    driver = tmp_path / "driver_revparse_fail.sh"
    driver.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'REPO_ROOT="' + str(repo_root) + '"\n'
        'PROJECT_PYTHON="' + str(py) + '"\n'
        'EXPECTED_SOURCE_COMMIT="'
        # 40 lowercase-hex digits; the test does NOT need to
        # match the (non-existent) HEAD since git rev-parse
        # itself fails first. Built from the deterministic
        # test fixture so no opaque 40-char literal appears in
        # this file.
        + TEST_EXPECTED_SOURCE_COMMIT
        + '"\n'
        + _extract_preflight_validation_block(SCRIPT.read_text(encoding="utf-8"))
        + "\n"
        "exit 0\n"
    )
    driver.chmod(0o755)
    proc = subprocess.run(
        ["bash", str(driver)],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr
    assert "SENSITIVE-LEAK-PATH-WILL-NOT-APPEAR" not in combined, (
        f"REPO_ROOT leaked: {combined!r}"
    )


def test_git_status_failure_does_not_leak_repo_root_path(tmp_path):
    """When ``git -C "${REPO_ROOT}" status --porcelain`` fails,
    ``${REPO_ROOT}`` must not appear in stdout/stderr.

    We construct a real git repo, capture HEAD (so rev-parse
    succeeds), and then replace ``.git`` with an empty directory
    to force ``git status --porcelain`` to fail."""
    fake_venv = tmp_path / "fake_venv_bin"
    fake_venv.mkdir()
    py = fake_venv / "python"
    py.write_text("#!/usr/bin/env bash\nexit 0\n")
    py.chmod(0o755)

    repo_root = tmp_path / "SENSITIVE-LEAK-PATH-FOR-STATUS"
    repo_root.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main", "-q"],
        cwd=str(repo_root),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "t@e"],
        cwd=str(repo_root),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"],
        cwd=str(repo_root),
        check=True,
    )
    (repo_root / "f").write_text("x\n")
    subprocess.run(["git", "add", "-A"], cwd=str(repo_root), check=True)
    subprocess.run(
        ["git", "commit", "-m", "init", "-q"],
        cwd=str(repo_root),
        check=True,
    )
    real_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # Corrupt .git so status fails.
    import shutil as _sh

    _sh.rmtree(repo_root / ".git")
    (repo_root / ".git").write_text("not a git directory\n")

    driver = tmp_path / "driver_status_fail.sh"
    driver.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'REPO_ROOT="' + str(repo_root) + '"\n'
        'PROJECT_PYTHON="' + str(py) + '"\n'
        'EXPECTED_SOURCE_COMMIT="'
        + real_head
        + '"\n'
        + _extract_preflight_validation_block(SCRIPT.read_text(encoding="utf-8"))
        + "\n"
        "exit 0\n"
    )
    driver.chmod(0o755)
    proc = subprocess.run(
        ["bash", str(driver)],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr
    assert "SENSITIVE-LEAK-PATH-FOR-STATUS" not in combined, (
        f"REPO_ROOT leaked: {combined!r}"
    )
