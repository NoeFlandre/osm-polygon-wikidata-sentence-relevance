"""Unit tests for the Grid'5000 artifact validators (Phase 9B)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from scripts.grid5000 import _validate_artifact as va

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
