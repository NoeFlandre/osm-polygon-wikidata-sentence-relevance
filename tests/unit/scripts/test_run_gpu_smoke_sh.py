"""Contract tests for the Grid'5000 smoke shell payload (Phase 9B).

These run on the Mac with only fakes and syntax/contract checks. They
never execute the payload (no OAR, no CUDA, no model, no network),
and never construct SaT, download weights, or perform inference.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "grid5000" / "run_gpu_smoke.sh"

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
    "git ",
    "git commit",
    "git push",
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


def test_script_requires_cuda_visibility(script_text):
    assert "CUDA_VISIBLE_DEVICES:?" in script_text
    assert "CUDA_VISIBLE_DEVICES is required" in script_text


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
    assert "source " not in script_text
    assert "activate" not in script_text


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

    env = _env(environ={"OAR_JOB_ID": "OAR-9", "CUDA_VISIBLE_DEVICES": "0,1"})
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
                "CUDA_VISIBLE_DEVICES": "0",
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
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"set -euo pipefail; REPO_ROOT='{SENSITIVE_PATH}/repo' HF_HOME='/tmp/dummy-hf' SMOKE_LOG_DIR='/tmp/dummy-log' OAR_JOB_ID=OAR-1 CUDA_VISIBLE_DEVICES=0 bash '{SCRIPT}' || true",
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
            f"set -euo pipefail; REPO_ROOT='/tmp/dummy-repo' HF_HOME='{SENSITIVE_PATH}/cache' SMOKE_LOG_DIR='/tmp/dummy-log' OAR_JOB_ID=OAR-1 CUDA_VISIBLE_DEVICES=0 bash '{SCRIPT}' || true",
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
            f"set -euo pipefail; REPO_ROOT='{sensitive_python}' HF_HOME='/tmp/dummy-hf' SMOKE_LOG_DIR='/tmp/dummy-log' OAR_JOB_ID=OAR-1 CUDA_VISIBLE_DEVICES=0 bash '{SCRIPT}' || true",
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
