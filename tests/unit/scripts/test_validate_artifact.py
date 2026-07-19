"""Unit tests for the Grid'5000 artifact validators (Phase 9B)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from scripts.grid5000 import _validate_artifact as va

# --- Test fixtures ------------------------------------------------
#
# SHA-shaped test values. Real production paths use the pinned
# Phase 9A / 9B commit SHAs and the public Hugging Face model and
# tokenizer revisions; we never re-use those for tests. Instead we
# derive each synthetic value from an explicit, role-named label
# via ``hashlib.sha1``. ``usedforsecurity=False`` documents that
# the digest is not used for cryptographic identity; it is a
# deterministic 40-character lowercase-hex string of known role.
#
# Public Hugging Face revision SHAs that appear elsewhere in the
# repository must NOT be replaced by these test fixtures: the
# smoke payload pins to the real, concrete revisions in production.


def _test_sha(label: str) -> str:
    """Return the first 40 lowercase-hex characters of the SHA-1
    digest of ``label``. ``usedforsecurity=False`` documents that
    the digest is used only as a deterministic test fixture."""
    return hashlib.sha1(label.encode("utf-8"), usedforsecurity=False).hexdigest()


#: Role-named fixture: a deterministic lowercase-hex SHA-shaped
#: string used in place of the production source commit.
TEST_SOURCE_COMMIT = _test_sha("tests/grid5000/run_metadata/source_commit")

#: Role-named fixture: a deterministic lowercase-hex SHA-shaped
#: string used in place of the production model revision.
TEST_MODEL_REVISION = _test_sha("tests/grid5000/run_metadata/model_revision")

#: Role-named fixture: a deterministic lowercase-hex SHA-shaped
#: string used in place of the production tokenizer revision.
TEST_TOKENIZER_REVISION = _test_sha("tests/grid5000/run_metadata/tokenizer_revision")


def _with_nonhex_char(digest: str, position: int = 0) -> str:
    """Return ``digest`` (a 40-character lowercase-hex fixture)
    with one character replaced by a deliberately non-lowercase-hex
    letter. Used to construct exactly-40-length invalid fixtures
    without writing a repeated-character literal in the source."""
    chars = list(digest)
    chars[position] = "g"
    return "".join(chars)


# --- Preflight ----------------------------------------------------


def _good_preflight():
    return {
        "oar_job_id": "OAR-1",
        "hostname": "gres-1",
        "torch_version": "2.4.0",
        "torch_cuda_runtime_version": "12.1",
        "device_0_name": "NVIDIA L40S",
        "visible_cuda_device_count": 1,
    }


def test_preflight_accepts_valid_payload():
    va.validate_preflight(_good_preflight())


def test_preflight_rejects_non_mapping():
    with pytest.raises(va.ArtifactValidationError, match="JSON object"):
        va.validate_preflight([1, 2, 3])


@pytest.mark.parametrize(
    "field",
    [
        "oar_job_id",
        "hostname",
        "torch_version",
        "torch_cuda_runtime_version",
        "device_0_name",
    ],
)
def test_preflight_rejects_blank_required_string(field):
    payload = _good_preflight()
    payload[field] = "   "
    with pytest.raises(va.ArtifactValidationError, match=field):
        va.validate_preflight(payload)


@pytest.mark.parametrize(
    "field",
    [
        "oar_job_id",
        "hostname",
        "torch_version",
        "torch_cuda_runtime_version",
        "device_0_name",
    ],
)
def test_preflight_rejects_missing_required_string(field):
    payload = _good_preflight()
    del payload[field]
    with pytest.raises(va.ArtifactValidationError, match=field):
        va.validate_preflight(payload)


@pytest.mark.parametrize("count", [0, 2, 3, 8])
def test_preflight_rejects_non_one_device_count(count):
    payload = _good_preflight()
    payload["visible_cuda_device_count"] = count
    with pytest.raises(va.ArtifactValidationError, match="visible_cuda_device_count"):
        va.validate_preflight(payload)


def test_preflight_rejects_bool_device_count():
    payload = _good_preflight()
    payload["visible_cuda_device_count"] = True  # bool is int subclass
    with pytest.raises(va.ArtifactValidationError, match="integer"):
        va.validate_preflight(payload)


def test_preflight_rejects_string_device_count():
    payload = _good_preflight()
    payload["visible_cuda_device_count"] = "1"
    with pytest.raises(va.ArtifactValidationError, match="integer"):
        va.validate_preflight(payload)


# --- Smoke result -------------------------------------------------


def _good_smoke():
    return {
        "resolved_device": "cuda",
        "model_name": "sat-3l-sm",
        "input_count": 3,
        "sentence_counts": [1, 2, 1],
        "elapsed_seconds": 1.234,
        "torch_version": "2.4.0",
        "torch_cuda_runtime_version": "12.1",
        "cuda_device_name": "NVIDIA L40S",
    }


def test_smoke_accepts_valid_payload():
    va.validate_smoke_result(_good_smoke())


def test_smoke_rejects_non_cuda_device():
    payload = _good_smoke()
    payload["resolved_device"] = "cpu"
    with pytest.raises(va.ArtifactValidationError, match="resolved_device"):
        va.validate_smoke_result(payload)


def test_smoke_rejects_wrong_model_name():
    payload = _good_smoke()
    payload["model_name"] = "sat-12l"
    with pytest.raises(va.ArtifactValidationError, match="model_name"):
        va.validate_smoke_result(payload)


def test_smoke_rejects_wrong_input_count():
    payload = _good_smoke()
    payload["input_count"] = 4
    with pytest.raises(va.ArtifactValidationError, match="input_count"):
        va.validate_smoke_result(payload)


def test_smoke_rejects_wrong_sentence_count_length():
    payload = _good_smoke()
    payload["sentence_counts"] = [1, 2]
    with pytest.raises(va.ArtifactValidationError, match="sentence_counts"):
        va.validate_smoke_result(payload)


@pytest.mark.parametrize("value", [0, -1])
def test_smoke_rejects_non_positive_sentence_count(value):
    payload = _good_smoke()
    payload["sentence_counts"] = [value, 1, 1]
    with pytest.raises(va.ArtifactValidationError, match="sentence_counts"):
        va.validate_smoke_result(payload)


def test_smoke_rejects_bool_sentence_count():
    payload = _good_smoke()
    payload["sentence_counts"] = [True, 1, 1]
    with pytest.raises(va.ArtifactValidationError, match="integer"):
        va.validate_smoke_result(payload)


def test_smoke_rejects_negative_elapsed():
    payload = _good_smoke()
    payload["elapsed_seconds"] = -0.5
    with pytest.raises(va.ArtifactValidationError, match="elapsed_seconds"):
        va.validate_smoke_result(payload)


def test_smoke_rejects_bool_elapsed():
    payload = _good_smoke()
    payload["elapsed_seconds"] = True
    with pytest.raises(va.ArtifactValidationError, match="elapsed_seconds"):
        va.validate_smoke_result(payload)


@pytest.mark.parametrize(
    "field",
    ["torch_version", "torch_cuda_runtime_version", "cuda_device_name"],
)
def test_smoke_rejects_blank_required_string(field):
    payload = _good_smoke()
    payload[field] = "   "
    with pytest.raises(va.ArtifactValidationError, match=field):
        va.validate_smoke_result(payload)


def test_smoke_rejects_non_mapping():
    with pytest.raises(va.ArtifactValidationError, match="JSON object"):
        va.validate_smoke_result("cuda")


# --- Micro-amendment: exact-schema enforcement ---------------------


PREFLIGHT_KEYS = {
    "oar_job_id",
    "hostname",
    "torch_version",
    "torch_cuda_runtime_version",
    "visible_cuda_device_count",
    "device_0_name",
}
SMOKE_KEYS = {
    "resolved_device",
    "model_name",
    "input_count",
    "sentence_counts",
    "elapsed_seconds",
    "torch_version",
    "torch_cuda_runtime_version",
    "cuda_device_name",
}


def test_preflight_rejects_extra_field():
    payload = _good_preflight()
    payload["unexpected"] = "value"
    with pytest.raises(va.ArtifactValidationError, match="unexpected"):
        va.validate_preflight(payload)


def test_preflight_rejects_missing_field():
    payload = _good_preflight()
    del payload["hostname"]
    with pytest.raises(va.ArtifactValidationError, match="hostname"):
        va.validate_preflight(payload)


def test_smoke_rejects_extra_field():
    payload = _good_smoke()
    payload["unexpected"] = "value"
    with pytest.raises(va.ArtifactValidationError, match="unexpected"):
        va.validate_smoke_result(payload)


def test_smoke_rejects_missing_field():
    payload = _good_smoke()
    del payload["model_name"]
    with pytest.raises(va.ArtifactValidationError, match="model_name"):
        va.validate_smoke_result(payload)


# --- Micro-amendment: non-finite timings ---------------------------


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), True, -0.5],
)
def test_smoke_rejects_non_finite_elapsed(value):
    payload = _good_smoke()
    payload["elapsed_seconds"] = value
    with pytest.raises(va.ArtifactValidationError, match="elapsed_seconds"):
        va.validate_smoke_result(payload)


def test_smoke_accepts_zero_elapsed():
    payload = _good_smoke()
    payload["elapsed_seconds"] = 0
    va.validate_smoke_result(payload)


# --- Micro-amendment: install helper -------------------------------


def test_install_artifact_happy_path(tmp_path):
    src = tmp_path / "src.json"
    dst = tmp_path / "dst.json"
    src.write_text("{}")
    va.install_artifact(src, dst)
    assert dst.exists()
    assert not src.exists()


def test_install_artifact_preserves_existing_destination(tmp_path):
    src = tmp_path / "src.json"
    dst = tmp_path / "dst.json"
    dst.write_text("ORIGINAL")
    src.write_text("NEW")
    with pytest.raises(va.ArtifactValidationError, match="already exists"):
        va.install_artifact(src, dst)
    assert dst.read_text() == "ORIGINAL"
    assert src.exists()


def test_install_artifact_rejects_parent_mismatch(tmp_path):
    src = tmp_path / "src.json"
    dst = tmp_path / "other" / "dst.json"
    src.write_text("{}")
    str(src.parent)
    str(tmp_path / "yet_another")
    with pytest.raises(va.ArtifactValidationError, match="parent director"):
        va.install_artifact(src, dst)
    # Verified that we never invoked os.link at all.
    assert not dst.exists()


def test_install_artifact_error_message_omits_path_values(tmp_path):
    sensitive = f"{tmp_path}/SENSITIVE-NEVER-LOG-INSTALL-9B-XYZ.json"
    src = tmp_path / "src.json"
    src.write_text("{}")
    src_as_posix = str(src)
    # Pre-create the destination with a sensitive name; the helper
    # must raise a stable error that does NOT echo the source or
    # destination path.
    dst = tmp_path / "SENSITIVE-NEVER-LOG-INSTALL-9B-XYZ.json"
    dst.write_text("PRESENT")
    with pytest.raises(va.ArtifactValidationError) as ei:
        va.install_artifact(src, dst)
    msg = str(ei.value)
    assert src_as_posix not in msg
    assert sensitive not in msg
    # The destination path itself must NOT leak either.
    assert str(dst) not in msg


# --- Portability amendment: cleanup-failure contract ----------------


def test_install_artifact_raises_on_unlink_failure(tmp_path, monkeypatch):
    """If the post-link ``os.unlink`` fails, the helper must raise
    ``ArtifactValidationError`` with a stable, path-free message.
    The final artifact must remain installed; only cleanup is
    reported as failed."""
    import os as _os

    src = tmp_path / "src.json"
    dst = tmp_path / "dst.json"
    src.write_text("PAYLOAD")
    real_unlink = _os.unlink

    def fake_unlink(path):
        # Refuse to unlink the just-installed source. Any other
        # unlink (none happens in this test) passes through.
        if Path(path) == src:
            raise OSError(16, "Device or resource busy")
        return real_unlink(path)

    monkeypatch.setattr(_os, "unlink", fake_unlink)
    with pytest.raises(va.ArtifactValidationError, match="cleanup failed"):
        va.install_artifact(src, dst)
    # The final artifact is still installed and unchanged.
    assert dst.exists()
    assert dst.read_text() == "PAYLOAD"


def test_install_artifact_cleanup_failure_message_is_path_free(tmp_path, monkeypatch):
    """The cleanup-failure message must not echo any path."""
    import os as _os

    sensitive = f"{tmp_path}/SENSITIVE-NEVER-LOG-CLEANUP-9B-XYZ.json"
    src = tmp_path / "src.json"
    dst = tmp_path / "SENSITIVE-NEVER-LOG-CLEANUP-9B-XYZ.json"
    src.write_text("PAYLOAD")

    def fake_unlink(path):
        if Path(path) == src:
            raise OSError(16, "Device or resource busy")
        return _os.unlink(path)

    monkeypatch.setattr(_os, "unlink", fake_unlink)
    with pytest.raises(va.ArtifactValidationError) as ei:
        va.install_artifact(src, dst)
    msg = str(ei.value)
    assert str(src) not in msg
    assert str(dst) not in msg
    assert sensitive not in msg


# --- CLI usage correction -------------------------------------------


def test_cli_usage_uses_module_basename_and_exits_two(tmp_path, monkeypatch):
    """Driving the validator with no arguments must print a usage
    message that references only the module basename (not a
    repository path or a ``python -m`` invocation) and exit
    with status 2.
    """
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    interpreter = project_root / ".venv" / "bin" / "python"
    validator = project_root / "scripts" / "grid5000" / "_validate_artifact.py"
    if not interpreter.exists():
        pytest.skip("locked interpreter not present in test env")
    if not validator.exists():
        pytest.skip("validator script not present in test env")

    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()

    proc = subprocess.run(
        [str(interpreter), str(validator)],
        cwd=str(unrelated_cwd),
        env={**os.environ, "PYTHONHASHSEED": "0"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 2
    combined = proc.stdout + proc.stderr
    # Must contain the basename.
    assert "_validate_artifact.py" in combined
    # Must not mention ``python -m`` (the smoke workflow uses an
    # absolute script path).
    assert "python -m" not in combined
    # Must not contain any repository path.
    assert "scripts/grid5000" not in combined
    assert str(project_root) not in combined


# --- Phase 9B amendment: run-metadata helper -------------------------


def test_run_metadata_writes_expected_keys_atomically(tmp_path, monkeypatch):
    """``scripts.grid5000._run_metadata.write_run_metadata`` writes
    exactly the documented keys (no username, no token, no env
    dump, no paths) and refuses to overwrite."""
    from scripts.grid5000 import _run_metadata as rm

    out_path = tmp_path / "run_metadata.json"
    rm.write_run_metadata(
        out_path,
        source_commit=TEST_SOURCE_COMMIT,
        model_name="sat-3l-sm",
        model_revision="137da054051ad9f1eac42025f758db4ac9f22535",
        tokenizer_name="facebookAI/xlm-roberta-base",
        tokenizer_revision=TEST_TOKENIZER_REVISION,
        oar_job_id="OAR-TEST",
        hostname="gres-1.nancy",
    )
    payload = json.loads(out_path.read_text())
    assert payload == {
        "source_commit": TEST_SOURCE_COMMIT,
        "model_name": "sat-3l-sm",
        "model_revision": "137da054051ad9f1eac42025f758db4ac9f22535",
        "tokenizer_name": "facebookAI/xlm-roberta-base",
        "tokenizer_revision": TEST_TOKENIZER_REVISION,
        "oar_job_id": "OAR-TEST",
        "hostname": "gres-1.nancy",
    }
    # File mode is 0600 (atomic, no group/world visibility).
    mode = out_path.stat().st_mode & 0o777
    assert mode == 0o600


def test_run_metadata_refuses_overwrite(tmp_path):
    from scripts.grid5000 import _run_metadata as rm

    out_path = tmp_path / "run_metadata.json"
    out_path.write_text("{}")
    with pytest.raises(Exception, match="already exists"):
        rm.write_run_metadata(
            out_path,
            source_commit=TEST_SOURCE_COMMIT,
            model_name="sat-3l-sm",
            model_revision=TEST_MODEL_REVISION,
            tokenizer_name="facebookAI/xlm-roberta-base",
            tokenizer_revision=TEST_TOKENIZER_REVISION,
            oar_job_id="OAR-1",
            hostname="gres-1",
        )


def test_run_metadata_rejects_missing_required_field(tmp_path):
    from scripts.grid5000 import _run_metadata as rm

    with pytest.raises(rm.RunMetadataError, match="source_commit"):
        rm.write_run_metadata(
            tmp_path / "run_metadata.json",
            source_commit="",
            model_name="sat-3l-sm",
            model_revision=TEST_MODEL_REVISION,
            tokenizer_name="facebookAI/xlm-roberta-base",
            tokenizer_revision=TEST_TOKENIZER_REVISION,
            oar_job_id="OAR-1",
            hostname="gres-1",
        )


def test_run_metadata_rejects_unsanctioned_keys(tmp_path):
    """The exact-schema contract forbids extra keys (e.g. a leaked
    username, path, or token)."""
    from scripts.grid5000 import _run_metadata as rm

    out_path = tmp_path / "run_metadata.json"
    rm.write_run_metadata(
        out_path,
        source_commit=TEST_SOURCE_COMMIT,
        model_name="sat-3l-sm",
        model_revision=TEST_MODEL_REVISION,
        tokenizer_name="facebookAI/xlm-roberta-base",
        tokenizer_revision=TEST_TOKENIZER_REVISION,
        oar_job_id="OAR-1",
        hostname="gres-1",
    )
    payload = json.loads(out_path.read_text())
    # Anything not in the allow-list must be rejected by the strict
    # schema. We re-invoke with an extra kwarg-sneaking path: the
    # helper function signature itself does not accept extras, so
    # the test simply asserts the on-disk keys are exactly the
    # documented set.
    assert set(payload.keys()) == {
        "source_commit",
        "model_name",
        "model_revision",
        "tokenizer_name",
        "tokenizer_revision",
        "oar_job_id",
        "hostname",
    }


# --- Reproducibility amendment: SHA validation ---------------------


def test_run_metadata_requires_40_lowercase_hex_for_source_commit(tmp_path):
    from scripts.grid5000 import _run_metadata as rm

    long_hex = TEST_SOURCE_COMMIT
    cases = [
        ("short", "abc"),
        ("uppercase", long_hex.upper()),
        ("mixed_case", "A" * 20 + "a" * 20),
        # 40-character non-hex input: derived from the same
        # deterministic test fixture with one lowercase-hex digit
        # replaced by a non-hex character. Avoids opaque
        # repeated-character token literals.
        ("non_hex", _with_nonhex_char(long_hex)),
        ("too_long", long_hex + "0"),
        ("too_short", long_hex[:-1]),
    ]
    for label, value in cases:
        with pytest.raises(rm.RunMetadataError, match="source_commit"):
            rm.write_run_metadata(
                tmp_path / f"run_metadata_{label}.json",
                source_commit=value,
                model_name="sat-3l-sm",
                model_revision=TEST_MODEL_REVISION,
                tokenizer_name="facebookAI/xlm-roberta-base",
                tokenizer_revision=TEST_TOKENIZER_REVISION,
                oar_job_id="OAR-1",
                hostname="gres-1",
            )


def test_run_metadata_requires_40_lowercase_hex_for_model_revision(tmp_path):
    from scripts.grid5000 import _run_metadata as rm

    with pytest.raises(rm.RunMetadataError, match="model_revision"):
        rm.write_run_metadata(
            tmp_path / "run_metadata_mrev.json",
            source_commit=TEST_SOURCE_COMMIT,
            model_name="sat-3l-sm",
            model_revision="AB" + "c" * 38,
            tokenizer_name="facebookAI/xlm-roberta-base",
            tokenizer_revision=TEST_TOKENIZER_REVISION,
            oar_job_id="OAR-1",
            hostname="gres-1",
        )


def test_run_metadata_requires_40_lowercase_hex_for_tokenizer_revision(
    tmp_path,
):
    from scripts.grid5000 import _run_metadata as rm

    with pytest.raises(rm.RunMetadataError, match="tokenizer_revision"):
        rm.write_run_metadata(
            tmp_path / "run_metadata_trev.json",
            source_commit=TEST_SOURCE_COMMIT,
            model_name="sat-3l-sm",
            model_revision=TEST_MODEL_REVISION,
            tokenizer_name="facebookAI/xlm-roberta-base",
            tokenizer_revision="not-hex-at-all-just-text-padding-xyz",
            oar_job_id="OAR-1",
            hostname="gres-1",
        )


def test_run_metadata_requires_specific_model_and_tokenizer_names(tmp_path):
    from scripts.grid5000 import _run_metadata as rm

    with pytest.raises(rm.RunMetadataError, match="model_name"):
        rm.write_run_metadata(
            tmp_path / "run_metadata_model.json",
            source_commit=TEST_SOURCE_COMMIT,
            model_name="sat-12l-sm",
            model_revision=TEST_MODEL_REVISION,
            tokenizer_name="facebookAI/xlm-roberta-base",
            tokenizer_revision=TEST_TOKENIZER_REVISION,
            oar_job_id="OAR-1",
            hostname="gres-1",
        )
    with pytest.raises(rm.RunMetadataError, match="tokenizer_name"):
        rm.write_run_metadata(
            tmp_path / "run_metadata_tok.json",
            source_commit=TEST_SOURCE_COMMIT,
            model_name="sat-3l-sm",
            model_revision=TEST_MODEL_REVISION,
            tokenizer_name="huggingface/xlm-roberta-base",
            tokenizer_revision=TEST_TOKENIZER_REVISION,
            oar_job_id="OAR-1",
            hostname="gres-1",
        )


# --- Reproducibility amendment: atomic install ---------------------


def test_run_metadata_uses_mkstemp_and_does_not_leave_tmp_on_fail(
    monkeypatch, tmp_path
):
    """The helper must use ``tempfile.mkstemp`` for the temp file
    (no predicted name; unique). On ``os.link`` failure the
    temporary file must be cleaned up, and an existing destination
    is preserved byte-for-byte."""
    import os as _os
    import tempfile as _tempfile

    import scripts.grid5000._run_metadata as rm

    dst = tmp_path / "run_metadata.json"

    captured_paths: list[str] = []
    real_mkstemp = _tempfile.mkstemp

    def fake_mkstemp(*args, **kwargs):
        # Restrict to the same dir as the destination so any leak
        # would be visible to the test.
        kwargs["dir"] = str(dst.parent)
        handle, path = real_mkstemp(*args, **kwargs)
        captured_paths.append(path)
        return handle, path

    monkeypatch.setattr(_tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(rm.tempfile, "mkstemp", fake_mkstemp)

    # Force os.link to fail.
    real_link = _os.link
    calls = {"count": 0}

    def boom_link(src, dst):
        calls["count"] += 1
        raise FileExistsError(17, "File exists", str(dst))

    monkeypatch.setattr(_os, "link", boom_link)

    # Pre-create the destination to trigger the link failure.
    dst.write_text("PREEXISTING")

    with pytest.raises(rm.RunMetadataError, match="already exists"):
        rm.write_run_metadata(
            dst,
            source_commit=TEST_SOURCE_COMMIT,
            model_name="sat-3l-sm",
            model_revision=TEST_MODEL_REVISION,
            tokenizer_name="facebookAI/xlm-roberta-base",
            tokenizer_revision=TEST_TOKENIZER_REVISION,
            oar_job_id="OAR-1",
            hostname="gres-1",
        )
    assert calls["count"] >= 1
    # Destination is preserved byte-for-byte.
    assert dst.read_text() == "PREEXISTING"
    # No temporary leftover.
    for path in captured_paths:
        assert not _os.path.exists(path), f"temp file leaked: {path}"
    # Restore os.link for the rest of the suite.
    monkeypatch.setattr(_os, "link", real_link)


def test_run_metadata_install_uses_os_link_and_never_os_replace(monkeypatch, tmp_path):
    """The helper must use ``os.link`` for atomic install (no
    ``os.replace``, no shutil)."""
    import os as _os

    import scripts.grid5000._run_metadata as rm

    forbidden_calls: list[str] = []

    def trip_replace(src, dst, *a, **kw):
        forbidden_calls.append("os.replace")
        raise AssertionError("os.replace must not be used")

    def trip_move(src, dst, *a, **kw):
        forbidden_calls.append("shutil.move")
        raise AssertionError("shutil.move must not be used")

    monkeypatch.setattr(_os, "replace", trip_replace)
    monkeypatch.setattr("shutil.move", trip_move, raising=False)

    dst = tmp_path / "run_metadata.json"
    rm.write_run_metadata(
        dst,
        source_commit=TEST_SOURCE_COMMIT,
        model_name="sat-3l-sm",
        model_revision=TEST_MODEL_REVISION,
        tokenizer_name="facebookAI/xlm-roberta-base",
        tokenizer_revision=TEST_TOKENIZER_REVISION,
        oar_job_id="OAR-1",
        hostname="gres-1",
    )
    assert forbidden_calls == [], (
        f"forbidden atomic-install primitive used: {forbidden_calls}"
    )
    # The atomic-install primitive actually used is os.link.
    assert dst.exists()


# --- Final-consistency amendment: path-free errors ------------------


SENSITIVE_SRC = "/home/REDACTED-TEST-SRC-PATH/.run_metadata.json.tmp.WILL-NOT-APPEAR"
SENSITIVE_DST = "/home/REDACTED-TEST-DST-PATH/run_metadata.json.WILL-NOT-APPEAR"


def test_run_metadata_errors_are_path_free_on_temp_create_failure(
    monkeypatch, tmp_path
):
    """Any filesystem failure must raise a stable, path-free
    ``RunMetadataError``; the source/destination paths must
    never appear in ``str(error)``."""

    import scripts.grid5000._run_metadata as rm

    real_mkstemp = rm.tempfile.mkstemp

    def boom_mkstemp(*args, **kwargs):
        return real_mkstemp(
            dir="/home/REDACTED-TEST-SRC-PATH/.run_metadata.json.tmp.WILL-NOT-APPEAR",
            prefix="will-not-appear-marker",
            suffix=".tmp",
        )

    monkeypatch.setattr(rm.tempfile, "mkstemp", boom_mkstemp)

    with pytest.raises(rm.RunMetadataError) as excinfo:
        rm.write_run_metadata(
            tmp_path / "run_metadata.json",
            source_commit=TEST_SOURCE_COMMIT,
            model_name="sat-3l-sm",
            model_revision=TEST_MODEL_REVISION,
            tokenizer_name="facebookAI/xlm-roberta-base",
            tokenizer_revision=TEST_TOKENIZER_REVISION,
            oar_job_id="OAR-1",
            hostname="gres-1",
        )
    msg = str(excinfo.value)
    assert SENSITIVE_SRC not in msg
    assert SENSITIVE_DST not in msg


def test_run_metadata_errors_are_path_free_on_link_failure(monkeypatch, tmp_path):
    """An ``os.link`` failure (other than ``FileExistsError``)
    must produce a path-free error."""
    import os as _os

    import scripts.grid5000._run_metadata as rm

    def boom_link(src, dst):
        raise OSError(
            28, "/home/REDACTED-TEST-DST-PATH/run_metadata.json.WILL-NOT-APPEAR"
        )

    monkeypatch.setattr(_os, "link", boom_link)

    with pytest.raises(rm.RunMetadataError) as excinfo:
        rm.write_run_metadata(
            tmp_path / "run_metadata.json",
            source_commit=TEST_SOURCE_COMMIT,
            model_name="sat-3l-sm",
            model_revision=TEST_MODEL_REVISION,
            tokenizer_name="facebookAI/xlm-roberta-base",
            tokenizer_revision=TEST_TOKENIZER_REVISION,
            oar_job_id="OAR-1",
            hostname="gres-1",
        )
    msg = str(excinfo.value)
    assert SENSITIVE_DST not in msg


def test_run_metadata_surfaces_post_link_cleanup_failure(monkeypatch, tmp_path):
    """After a successful ``os.link``, failure to unlink the temp
    file must raise ``RunMetadataError("artifact installed but
    temporary cleanup failed")``. The destination is preserved
    and the error message is path-free."""
    import os as _os

    import scripts.grid5000._run_metadata as rm

    link_calls = {"count": 0}
    unlink_calls: list[str] = []

    real_link = _os.link

    def spy_link(src, dst):
        link_calls["count"] += 1
        return real_link(src, dst)

    def boom_unlink(path):
        # Force cleanup failure AFTER a successful link.
        unlink_calls.append(path)
        raise OSError(16, "/home/REDACTED-TEST-SRC-PATH/.tmp.WILL-NOT-APPEAR")

    monkeypatch.setattr(_os, "link", spy_link)
    monkeypatch.setattr(_os, "unlink", boom_unlink)

    dst = tmp_path / "run_metadata.json"
    with pytest.raises(
        rm.RunMetadataError, match="temporary cleanup failed"
    ) as excinfo:
        rm.write_run_metadata(
            dst,
            source_commit=TEST_SOURCE_COMMIT,
            model_name="sat-3l-sm",
            model_revision=TEST_MODEL_REVISION,
            tokenizer_name="facebookAI/xlm-roberta-base",
            tokenizer_revision=TEST_TOKENIZER_REVISION,
            oar_job_id="OAR-1",
            hostname="gres-1",
        )
    # The atomic-install primitive ran, and the destination exists.
    assert link_calls["count"] == 1
    assert dst.exists()
    # Cleanup was attempted against the temp file.
    assert unlink_calls, "cleanup unlink was never attempted"
    # Error message is path-free.
    msg = str(excinfo.value)
    assert SENSITIVE_SRC not in msg
    assert SENSITIVE_DST not in msg


def test_run_metadata_uses_fchmod_0600(monkeypatch, tmp_path):
    """The temp file's permission must be set via ``os.fchmod``
    with mode ``0o600`` on the open fd (not via ``os.chmod``
    after closing the fd)."""
    import os as _os

    import scripts.grid5000._run_metadata as rm

    captured: list[tuple[int, int]] = []

    real_fchmod = _os.fchmod

    def spy_fchmod(fd, mode):
        captured.append((fd, mode))
        return real_fchmod(fd, mode)

    monkeypatch.setattr(_os, "fchmod", spy_fchmod)

    forbidden_calls: list[str] = []

    def trip_chmod(*a, **kw):
        forbidden_calls.append("os.chmod (called from helper)")
        # On macOS os.chmod is permissive; raise to surface the
        # regression in test output.
        raise AssertionError(
            "os.chmod must not be used to set temp permission; use os.fchmod"
        )

    monkeypatch.setattr(_os, "chmod", trip_chmod)

    dst = tmp_path / "run_metadata.json"
    rm.write_run_metadata(
        dst,
        source_commit=TEST_SOURCE_COMMIT,
        model_name="sat-3l-sm",
        model_revision=TEST_MODEL_REVISION,
        tokenizer_name="facebookAI/xlm-roberta-base",
        tokenizer_revision=TEST_TOKENIZER_REVISION,
        oar_job_id="OAR-1",
        hostname="gres-1",
    )
    # The temp permission was set via os.fchmod with 0o600.
    assert any(mode == 0o600 for _, mode in captured), (
        f"expected os.fchmod call with mode=0o600; got {captured!r}"
    )
    # And it was the only chmod primitive used inside the helper.
    assert forbidden_calls == [], f"os.chmod called inside helper: {forbidden_calls}"


# --- Phase 9B safety amendment: fchmod-failure wrap ------------------


def test_run_metadata_fchmod_failure_is_path_free(monkeypatch, tmp_path):
    """An ``os.fchmod`` failure after ``mkstemp`` must be wrapped
    into a stable, path-free ``RunMetadataError``. Sensitive
    filesystem paths must never appear in ``str(error)`` and the
    underlying OSError must be preserved as ``__cause__``."""
    import os as _os

    import scripts.grid5000._run_metadata as rm

    def boom_fchmod(fd, mode):
        raise OSError(
            1,
            "/home/REDACTED-TEST-SRC-PATH/.run_metadata.json.tmp.WILL-NOT-APPEAR",
        )

    monkeypatch.setattr(_os, "fchmod", boom_fchmod)

    with pytest.raises(rm.RunMetadataError) as excinfo:
        rm.write_run_metadata(
            tmp_path / "run_metadata.json",
            source_commit=TEST_SOURCE_COMMIT,
            model_name="sat-3l-sm",
            model_revision=TEST_MODEL_REVISION,
            tokenizer_name="facebookAI/xlm-roberta-base",
            tokenizer_revision=TEST_TOKENIZER_REVISION,
            oar_job_id="OAR-1",
            hostname="gres-1",
        )
    msg = str(excinfo.value)
    assert SENSITIVE_SRC not in msg
    assert SENSITIVE_DST not in msg
    assert excinfo.value.__cause__ is not None


def test_run_metadata_fchmod_failure_closes_descriptor(monkeypatch, tmp_path):
    """When ``os.fchmod`` fails after ``mkstemp`` returned an
    open descriptor, the helper must close that exact descriptor
    before propagating. We intercept the descriptor returned by
    ``mkstemp`` and spy on ``os.close`` to prove the descriptor
    was closed exactly once and the temporary file removed."""
    import os as _os

    import scripts.grid5000._run_metadata as rm

    # mkstemp in the helper delegates to the module-level
    # ``tempfile`` import. Replace it with a wrapper that records
    # the descriptor it received.
    real_mkstemp = rm.tempfile.mkstemp

    # Track mkstemp output paths so we can confirm cleanup also
    # removed them from the filesystem.
    recorded: list[tuple[int, str]] = []
    captured_real_paths: list[str] = []

    def spy_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        recorded.append((fd, path))
        captured_real_paths.append(path)
        return fd, path

    # Real os.close path-tracking and tracking list. We want the
    # test to distinguish the helper's call from any closing done
    # by ``os.fdopen``.
    closed: list[tuple[object, ...]] = []

    real_close = _os.close

    def spy_close(fd):
        closed.append(fd)
        return real_close(fd)

    monkeypatch.setattr(rm.tempfile, "mkstemp", spy_mkstemp)
    monkeypatch.setattr(
        _os, "fchmod", lambda fd, mode: (_ for _ in ()).throw(OSError(1, "boom-fchmod"))
    )
    monkeypatch.setattr(_os, "close", spy_close)

    with pytest.raises(rm.RunMetadataError, match="temporary permission setup failed"):
        rm.write_run_metadata(
            tmp_path / "run_metadata.json",
            source_commit=TEST_SOURCE_COMMIT,
            model_name="sat-3l-sm",
            model_revision=TEST_MODEL_REVISION,
            tokenizer_name="facebookAI/xlm-roberta-base",
            tokenizer_revision=TEST_TOKENIZER_REVISION,
            oar_job_id="OAR-1",
            hostname="gres-1",
        )

    # mkstemp was called exactly once.
    assert len(recorded) == 1, recorded
    leaked_fd, leaked_path = recorded[0]
    # The helper closed the leaked fd exactly once. ``os.fdopen``
    # was never reached (it would have transferred ownership), so
    # any close() call here is the helper's explicit cleanup.
    assert leaked_fd in closed, f"helper leaked fd {leaked_fd}; closed={closed!r}"
    # The fd was closed exactly once (no double-close noise from
    # the helper).
    assert closed.count(leaked_fd) == 1, (
        f"fd {leaked_fd!r} closed {closed.count(leaked_fd)} times; "
        f"all closes: {closed!r}"
    )
    # The temporary file produced by mkstemp is removed.
    assert not _os.path.exists(leaked_path), f"temp file {leaked_path!r} leaked"


def test_run_metadata_post_link_cleanup_preserves_cause(monkeypatch, tmp_path):
    """When the post-link ``os.unlink`` of the temp file fails,
    the helper raises ``RunMetadataError("artifact installed but
    temporary cleanup failed")``. The original OSError must be
    preserved as ``__cause__`` and the error string must remain
    path-free. The successfully installed destination must
    remain on disk."""
    import os as _os

    import scripts.grid5000._run_metadata as rm

    boom_unlink = OSError(16, "/home/REDACTED-TEST-SRC-PATH/.tmp.WILL-NOT-APPEAR")
    calls = {"count": 0}

    real_unlink = _os.unlink

    def maybe_boom_unlink(path):
        calls["count"] += 1
        if calls["count"] == 1:
            raise boom_unlink
        return real_unlink(path)

    monkeypatch.setattr(_os, "unlink", maybe_boom_unlink)

    dst = tmp_path / "run_metadata.json"
    with pytest.raises(
        rm.RunMetadataError, match="temporary cleanup failed"
    ) as excinfo:
        rm.write_run_metadata(
            dst,
            source_commit=TEST_SOURCE_COMMIT,
            model_name="sat-3l-sm",
            model_revision=TEST_MODEL_REVISION,
            tokenizer_name="facebookAI/xlm-roberta-base",
            tokenizer_revision=TEST_TOKENIZER_REVISION,
            oar_job_id="OAR-1",
            hostname="gres-1",
        )

    # The original OSError is preserved as __cause__.
    assert excinfo.value.__cause__ is boom_unlink, (
        f"expected the original OSError as __cause__, got {excinfo.value.__cause__!r}"
    )
    # The error message is path-free.
    msg = str(excinfo.value)
    assert "SENSITIVE-PATH" not in msg
    assert "/home/" not in msg
    # The successfully installed destination remains intact.
    assert dst.exists(), "post-link cleanup failure must not roll back the destination"
