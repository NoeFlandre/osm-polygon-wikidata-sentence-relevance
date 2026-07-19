"""Subprocess-level CLI tests for the Grid'5000 run metadata helper
(Phase 9I-fix).

Phase 9B introduced ``scripts/grid5000/_run_metadata.py`` exposing
``write_run_metadata(...)`` as a top-level function. The smoke shell
payload invokes that module as a CLI process with nine positional
arguments, but the module never wires ``sys.argv`` into the function.

This file proves the fix at the *subprocess* boundary so future
refactors of the CLI dispatcher cannot silently regress to a no-op
that exits 0 without writing the destination.

Tests run on the Mac with only the locked interpreter and the
committed helper. No torch, no SaT, no network, no Grid'5000
contact.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts" / "grid5000" / "_run_metadata.py"

INTERPRETER = ROOT / ".venv" / "bin" / "python"
if not INTERPRETER.exists():
    INTERPRETER = Path(
        subprocess.run(
            ["which", "python"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )


# --- Deterministic SHA-shaped test fixtures (no opaque literal) -------


def _test_sha(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8"), usedforsecurity=False).hexdigest()


SOURCE_COMMIT = _test_sha("tests/grid5000/run_metadata_cli/source_commit")
MODEL_REVISION = _test_sha("tests/grid5000/run_metadata_cli/model_revision")
TOKENIZER_REVISION = _test_sha("tests/grid5000/run_metadata_cli/tokenizer_revision")
# 7-digit decimal oar_job_id fixture, deterministic, not a real job.
OAR_JOB_ID = "9" + _test_sha("tests/grid5000/run_metadata_cli/oar_job_id")[:6]
# Synthetic hostname fixture (deterministic, not a real Grid'5000 node).
HOSTNAME_FIXTURE = "host-" + _test_sha("tests/grid5000/run_metadata_cli/hostname")[:8]
# Synthetic sensitive path marker (deterministic, not a real path).
SENSITIVE_DIR = (
    "sensitive-fixture-dir-"
    + _test_sha("tests/grid5000/run_metadata_cli/sensitive_dir")[:8]
)


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke the helper as a real subprocess with ``args`` as the
    user-supplied positional arguments (no program name)."""
    return subprocess.run(
        [str(INTERPRETER), str(HELPER), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


# --- Static checks -----------------------------------------------------


def test_helper_exists_and_is_nonempty():
    assert HELPER.exists()
    assert HELPER.stat().st_size > 200


def test_helper_syntax_is_valid_python():
    proc = subprocess.run(
        [
            str(INTERPRETER),
            "-c",
            "import py_compile; py_compile.compile(str(__import__('pathlib').Path('"
            + str(HELPER)
            + "')), doraise=True)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


def test_helper_has_main_entry_point():
    """The helper must expose a CLI adapter (main + __main__)."""
    text = HELPER.read_text(encoding="utf-8")
    assert "def main(" in text, "helper must define a main(argv) -> int entry point"
    assert "__main__" in text, "helper must dispatch via if __name__ == '__main__':"
    assert "SystemExit(main())" in text or "raise SystemExit" in text, (
        "helper must call main() under if __name__ == '__main__'"
    )


# --- Successful invocation --------------------------------------------


def test_cli_creates_seven_key_json(tmp_path):
    dst = tmp_path / "run_metadata.json"
    proc = _run_cli(
        str(dst),
        SOURCE_COMMIT,
        "sat-3l-sm",
        MODEL_REVISION,
        "facebookAI/xlm-roberta-base",
        TOKENIZER_REVISION,
        OAR_JOB_ID,
        HOSTNAME_FIXTURE,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert dst.exists(), "destination was not created"
    payload = json.loads(dst.read_text())
    assert set(payload.keys()) == {
        "source_commit",
        "model_name",
        "model_revision",
        "tokenizer_name",
        "tokenizer_revision",
        "oar_job_id",
        "hostname",
    }
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["model_name"] == "sat-3l-sm"
    assert payload["model_revision"] == MODEL_REVISION
    assert payload["tokenizer_name"] == "facebookAI/xlm-roberta-base"
    assert payload["tokenizer_revision"] == TOKENIZER_REVISION
    assert payload["oar_job_id"] == OAR_JOB_ID
    assert payload["hostname"] == HOSTNAME_FIXTURE


def test_cli_writes_atomic_no_clobber_with_mode_0600(tmp_path):
    parent = tmp_path / "logdir"
    parent.mkdir()
    parent.chmod(0o700)
    dst = parent / "run_metadata.json"
    proc = _run_cli(
        str(dst),
        SOURCE_COMMIT,
        "sat-3l-sm",
        MODEL_REVISION,
        "facebookAI/xlm-roberta-base",
        TOKENIZER_REVISION,
        OAR_JOB_ID,
        HOSTNAME_FIXTURE,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert dst.exists()
    mode = dst.stat().st_mode & 0o777
    assert mode == 0o600, f"expected mode 0600, got {oct(mode)}"


def test_cli_refuses_to_overwrite_existing_destination(tmp_path):
    dst = tmp_path / "run_metadata.json"
    dst.write_text("preserved\n")
    dst.chmod(0o600)
    proc = _run_cli(
        str(dst),
        SOURCE_COMMIT,
        "sat-3l-sm",
        MODEL_REVISION,
        "facebookAI/xlm-roberta-base",
        TOKENIZER_REVISION,
        OAR_JOB_ID,
        HOSTNAME_FIXTURE,
    )
    assert proc.returncode != 0
    # The original content is preserved; the CLI never overwrites.
    assert dst.read_text() == "preserved\n"


# --- Usage / validation errors ----------------------------------------


def test_cli_wrong_argument_count_exits_2(tmp_path):
    proc = _run_cli(str(tmp_path / "run_metadata.json"))
    assert proc.returncode == 2
    assert proc.stderr  # stable usage label on stderr


def test_cli_invalid_source_commit_format_exits_nonzero(tmp_path):
    """Non-40-hex source_commit must be rejected with a stable,
    path-free error."""
    proc = _run_cli(
        str(tmp_path / "run_metadata.json"),
        "not-a-valid-commit",
        "sat-3l-sm",
        MODEL_REVISION,
        "facebookAI/xlm-roberta-base",
        TOKENIZER_REVISION,
        OAR_JOB_ID,
        HOSTNAME_FIXTURE,
    )
    assert proc.returncode != 0
    # Stable label reference, not the supplied value.
    assert "field 'source_commit'" in proc.stderr
    assert "not-a-valid-commit" not in proc.stderr


def test_cli_invalid_model_name_exits_nonzero(tmp_path):
    proc = _run_cli(
        str(tmp_path / "run_metadata.json"),
        SOURCE_COMMIT,
        "wrong-model",
        MODEL_REVISION,
        "facebookAI/xlm-roberta-base",
        TOKENIZER_REVISION,
        OAR_JOB_ID,
        HOSTNAME_FIXTURE,
    )
    assert proc.returncode != 0
    assert "model_name" in proc.stderr
    assert "wrong-model" not in proc.stderr


def test_cli_invalid_model_revision_exits_nonzero(tmp_path):
    proc = _run_cli(
        str(tmp_path / "run_metadata.json"),
        SOURCE_COMMIT,
        "sat-3l-sm",
        "not-a-valid-sha",
        "facebookAI/xlm-roberta-base",
        TOKENIZER_REVISION,
        OAR_JOB_ID,
        HOSTNAME_FIXTURE,
    )
    assert proc.returncode != 0
    assert "model_revision" in proc.stderr


def test_cli_invalid_tokenizer_name_exits_nonzero(tmp_path):
    proc = _run_cli(
        str(tmp_path / "run_metadata.json"),
        SOURCE_COMMIT,
        "sat-3l-sm",
        MODEL_REVISION,
        "wrong/tokenizer",
        TOKENIZER_REVISION,
        OAR_JOB_ID,
        HOSTNAME_FIXTURE,
    )
    assert proc.returncode != 0
    assert "tokenizer_name" in proc.stderr


def test_cli_invalid_tokenizer_revision_exits_nonzero(tmp_path):
    proc = _run_cli(
        str(tmp_path / "run_metadata.json"),
        SOURCE_COMMIT,
        "sat-3l-sm",
        MODEL_REVISION,
        "facebookAI/xlm-roberta-base",
        "not-a-valid-sha",
        OAR_JOB_ID,
        HOSTNAME_FIXTURE,
    )
    assert proc.returncode != 0
    assert "tokenizer_revision" in proc.stderr


# --- Path-free error contract -----------------------------------------


def test_cli_error_messages_never_leak_destination_path(tmp_path):
    """A distinctive sensitive destination must never appear in
    stdout or stderr on any failure path."""
    # Place the sensitive marker INSIDE tmp_path so the OS allows
    # the helper to attempt writes there. The path itself remains
    # distinctive and never appears in any CLI output.
    sensitive = tmp_path / SENSITIVE_DIR
    sensitive.mkdir()
    sensitive_dst = sensitive / "run_metadata.json"
    cases = [
        # Pre-existing destination (refuse-to-overwrite).
        (
            [
                str(sensitive_dst),
                SOURCE_COMMIT,
                "sat-3l-sm",
                MODEL_REVISION,
                "facebookAI/xlm-roberta-base",
                TOKENIZER_REVISION,
                OAR_JOB_ID,
                HOSTNAME_FIXTURE,
            ],
            True,  # pre-create destination to trigger the refusal branch
        ),
        # Bad source_commit.
        (
            [
                str(sensitive_dst),
                "bad",
                "sat-3l-sm",
                MODEL_REVISION,
                "facebookAI/xlm-roberta-base",
                TOKENIZER_REVISION,
                OAR_JOB_ID,
                HOSTNAME_FIXTURE,
            ],
            False,
        ),
    ]
    for args, precreate in cases:
        if precreate:
            # Pre-create the destination so the overwrite-refusal
            # branch fires; the contents must be preserved.
            sensitive_dst.write_text("untouched\n")
            sensitive_dst.chmod(0o600)
        proc = _run_cli(*args)
        combined = (proc.stdout or "") + (proc.stderr or "")
        assert SENSITIVE_DIR not in combined, (
            f"destination path leaked into CLI output: {combined!r}"
        )


# --- Failure-mode exit codes are non-zero for every error path --------


@pytest.mark.parametrize(
    "args",
    [
        # Missing required args.
        ["/tmp/whatever"],
        ["/tmp/whatever", SOURCE_COMMIT],
        # Bad source_commit (non-hex).
        [
            "/tmp/whatever",
            "not-40-hex",
            "sat-3l-sm",
            MODEL_REVISION,
            "facebookAI/xlm-roberta-base",
            TOKENIZER_REVISION,
            "1",
            "host",
        ],
        # Bad model_revision (non-hex).
        [
            "/tmp/whatever",
            SOURCE_COMMIT,
            "sat-3l-sm",
            "not-40-hex",
            "facebookAI/xlm-roberta-base",
            TOKENIZER_REVISION,
            "1",
            "host",
        ],
        # Bad tokenizer_revision (non-hex).
        [
            "/tmp/whatever",
            SOURCE_COMMIT,
            "sat-3l-sm",
            MODEL_REVISION,
            "facebookAI/xlm-roberta-base",
            "not-40-hex",
            "1",
            "host",
        ],
    ],
)
def test_cli_every_failure_path_exits_nonzero(tmp_path, args):
    # Bind the destination to a path under tmp_path so the test does
    # not write to absolute system locations.
    args = [str(tmp_path / "run_metadata.json"), *args[1:]]
    proc = _run_cli(*args)
    assert proc.returncode != 0, (
        f"expected non-zero exit for args={args!r}, got rc={proc.returncode}, "
        f"stdout={proc.stdout!r}, stderr={proc.stderr!r}"
    )


# --- Smoke-payload style invocation (mirrors run_gpu_smoke.sh) -------


def test_cli_invocation_pattern_matches_smoke_payload(tmp_path):
    """The smoke shell payload invokes the helper as:

        python <helper> <RUN_METADATA_PATH> \\
            <SOURCE_COMMIT> \\
            "sat-3l-sm" \\
            <MODEL_REVISION> \\
            "facebookAI/xlm-roberta-base" \\
            <TOKENIZER_REVISION> \\
            <OAR_JOB_ID> \\
            <HOSTNAME_SHORT>

    This test pins that exact CLI shape.
    """
    dst = tmp_path / "run_metadata.json"
    proc = subprocess.run(
        [
            str(INTERPRETER),
            str(HELPER),
            str(dst),
            SOURCE_COMMIT,
            "sat-3l-sm",
            MODEL_REVISION,
            "facebookAI/xlm-roberta-base",
            TOKENIZER_REVISION,
            OAR_JOB_ID,
            HOSTNAME_FIXTURE,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(dst.read_text())
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["oar_job_id"] == OAR_JOB_ID


# --- No environment-dependent behaviour -------------------------------


def test_cli_unrelated_environ_is_ignored(tmp_path):
    """The CLI must not depend on or mutate os.environ. The smoke
    payload invokes it with HF_HUB_OFFLINE etc. exported; this
    test verifies the helper ignores such variables.
    """
    dst = tmp_path / "run_metadata.json"
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "RANDOM_VAR_THAT_SHOULD_NOT_BE_READ": "secret",
    }
    proc = subprocess.run(
        [
            str(INTERPRETER),
            str(HELPER),
            str(dst),
            SOURCE_COMMIT,
            "sat-3l-sm",
            MODEL_REVISION,
            "facebookAI/xlm-roberta-base",
            TOKENIZER_REVISION,
            OAR_JOB_ID,
            HOSTNAME_FIXTURE,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RANDOM_VAR_THAT_SHOULD_NOT_BE_READ" not in (proc.stdout + proc.stderr)
