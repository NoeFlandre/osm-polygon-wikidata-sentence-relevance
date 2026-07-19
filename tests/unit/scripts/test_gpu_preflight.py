"""Unit tests for the Grid'5000 CUDA preflight (Phase 9B + Phase 9H).

Phase 9H contract:
  * CUDA_VISIBLE_DEVICES is informational only. The preflight does
    NOT require, read, or assert on it; the authoritative runtime
    proof of GPU scoping is ``torch.cuda.is_available() is True``
    AND ``torch.cuda.device_count() == 1``.
  * The preflight does not mutate ``os.environ``.
  * The result schema is unchanged from Phase 9B (no new field).
  * Required conditions: Linux, non-blank OAR_JOB_ID, Torch import,
    ``torch.cuda.is_available() is True``, exactly one device, and
    ``torch.cuda.get_device_name(0)`` succeeds.

Tests run on the Mac without CUDA; every platform/environment/
Torch touch is injected through ``PreflightEnv`` facades and a
fake ``torch`` module. No real GPU, no network access, no real
Torch import.
"""

from __future__ import annotations

import json
import os

import pytest
from scripts.grid5000 import gpu_preflight as pf

# --- Fake Torch fixtures ---------------------------------------------


class _FakeTorch:
    """Minimal torch stub with a healthy CUDA backend."""

    __version__ = "2.4.0"
    version = type("Version", (), {"cuda": "12.1"})()

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(_idx: int) -> str:
            return "NVIDIA L40S"


class _FakeTorchNoCuda:
    """torch whose CUDA backend reports unavailable."""

    __version__ = "2.4.0"
    version = type("Version", (), {"cuda": "12.1"})()

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

        @staticmethod
        def get_device_name(_idx: int) -> str:
            return "NVIDIA L40S"


def _env(
    *,
    system="Linux",
    hostname="gpu-node-1",
    environ=None,
    torch_factory=None,
):
    """Build a PreflightEnv facade over a plain dict + function."""

    environ = dict(environ or {})

    class _Env:
        @property
        def system(self) -> str:
            return system

        @property
        def node_name(self) -> str:
            return hostname

        def getenv(self, name, default=None):
            return environ.get(name, default)

        def torch_factory(self):
            if torch_factory is not None:
                return torch_factory()
            raise ImportError("torch not available")

    return _Env()


# --- Rejections -------------------------------------------------------


def test_rejects_non_linux():
    env = _env(system="Darwin")
    with pytest.raises(pf.PreflightError, match="Linux"):
        pf.run_preflight(env, torch_mod=_FakeTorch())


def test_rejects_missing_oar_job_id():
    # Phase 9H: CUDA_VISIBLE_DEVICES is NOT set here. Preflight must
    # succeed past its absence and fail on OAR_JOB_ID instead.
    env = _env()
    with pytest.raises(pf.PreflightError, match="OAR_JOB_ID"):
        pf.run_preflight(env, torch_mod=_FakeTorch())


def test_rejects_blank_oar_job_id():
    env = _env(environ={"OAR_JOB_ID": "   "})
    with pytest.raises(pf.PreflightError, match="OAR_JOB_ID"):
        pf.run_preflight(env, torch_mod=_FakeTorch())


def test_rejects_unavailable_cuda():
    env = _env(environ={"OAR_JOB_ID": "123"})
    with pytest.raises(pf.PreflightError, match="is_available"):
        pf.run_preflight(env, torch_mod=_FakeTorchNoCuda())


def test_rejects_zero_devices():
    class _FakeTorchZero:
        __version__ = "2.4.0"
        version = type("Version", (), {"cuda": "12.1"})()

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 0

            @staticmethod
            def get_device_name(_idx: int) -> str:
                return "NVIDIA L40S"

    env = _env(environ={"OAR_JOB_ID": "123"})
    with pytest.raises(pf.PreflightError, match="exactly one"):
        pf.run_preflight(env, torch_mod=_FakeTorchZero())


def test_rejects_more_than_one_device():
    class _FakeTorchTwo:
        __version__ = "2.4.0"
        version = type("Version", (), {"cuda": "12.1"})()

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 2

            @staticmethod
            def get_device_name(_idx: int) -> str:
                return "NVIDIA A100"

    env = _env(environ={"OAR_JOB_ID": "123"})
    with pytest.raises(pf.PreflightError, match="exactly one"):
        pf.run_preflight(env, torch_mod=_FakeTorchTwo())


# --- Phase 9H: CUDA_VISIBLE_DEVICES is informational only -------------


@pytest.mark.parametrize(
    "cuda_visible_value", [None, "", "   ", "0", "0,1,GPU-deadbeef"]
)
def test_cuda_visible_devices_absence_does_not_affect_success(cuda_visible_value):
    """CUDA_VISIBLE_DEVICES must be informational. Any value
    (or absence) is acceptable as long as the real preflight
    conditions (Torch reports exactly one usable CUDA device)
    hold."""
    environ = {"OAR_JOB_ID": "123"}
    if cuda_visible_value is not None:
        environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_value
    env = _env(environ=environ)
    # Must NOT raise on the CUDA_VISIBLE_DEVICES gate.
    result = pf.run_preflight(env, torch_mod=_FakeTorch())
    assert result.visible_cuda_device_count == 1
    assert result.device_0_name == "NVIDIA L40S"


def test_preflight_does_not_read_cuda_visible_devices_internally():
    """The preflight must not call ``env.getenv("CUDA_VISIBLE_DEVICES")``
    or ``os.getenv("CUDA_VISIBLE_DEVICES")`` at all. Any future
    code that does so is forbidden by this test (kept as a
    static check that no string-literal form reads the
    variable). The CUDA_VISIBLE_DEVICES identifier may still
    appear in module docstrings or comments as a historical
    reference."""
    import inspect

    source = inspect.getsource(pf)
    # Strip docstrings and comments so we only check executable code.
    import re

    # Remove triple-quoted docstrings.
    stripped = re.sub(r"\"\"\".*?\"\"\"", "", source, flags=re.DOTALL)
    stripped = re.sub(r"'''.*?'''", "", stripped, flags=re.DOTALL)
    # Strip trailing comments on lines.
    stripped = re.sub(r"#[^\n]*", "", stripped)
    assert '"CUDA_VISIBLE_DEVICES"' not in stripped
    assert "'CUDA_VISIBLE_DEVICES'" not in stripped


def test_preflight_does_not_mutate_os_environ():
    """The preflight must not mutate ``os.environ``. The facade
    pattern is the only supported way to read environment state."""
    # Snapshot the environment, run the preflight, ensure the
    # snapshot is unchanged.
    snapshot = dict(os.environ)
    env = _env(environ={"OAR_JOB_ID": "123"})
    pf.run_preflight(env, torch_mod=_FakeTorch())
    # Also exercise the CLI entry point.
    pf.run_with(env, torch_mod=_FakeTorch())
    assert dict(os.environ) == snapshot


# --- Success ----------------------------------------------------------


def test_emits_expected_json():
    env = _env(environ={"OAR_JOB_ID": "OAR-456"})
    result = pf.run_preflight(env, torch_mod=_FakeTorch())
    payload = json.loads(result.to_json())
    assert payload == {
        "oar_job_id": "OAR-456",
        "hostname": "gpu-node-1",
        "torch_version": "2.4.0",
        "torch_cuda_runtime_version": _FakeTorch().version.cuda,
        "visible_cuda_device_count": 1,
        "device_0_name": "NVIDIA L40S",
    }


def test_torch_cuda_runtime_version_present():
    env = _env(environ={"OAR_JOB_ID": "OAR-456"})
    result = pf.run_preflight(env, torch_mod=_FakeTorch())
    assert isinstance(result.torch_cuda_runtime_version, str)
    assert result.torch_cuda_runtime_version


# --- Schema stability: no new field -----------------------------------


def test_result_schema_is_unchanged():
    """Phase 9H: the JSON schema MUST NOT change. The set of
    documented keys is the same six as Phase 9B."""
    env = _env(environ={"OAR_JOB_ID": "OAR-456"})
    result = pf.run_preflight(env, torch_mod=_FakeTorch())
    payload = json.loads(result.to_json())
    expected_keys = {
        "oar_job_id",
        "hostname",
        "torch_version",
        "torch_cuda_runtime_version",
        "visible_cuda_device_count",
        "device_0_name",
    }
    assert set(payload.keys()) == expected_keys, (
        f"schema drift: got {sorted(payload.keys())!r}, "
        f"expected {sorted(expected_keys)!r}"
    )


# --- main() exit codes ------------------------------------------------


def test_main_returns_zero_on_success(capsys):
    env = _env(environ={"OAR_JOB_ID": "OAR-1"})
    code = pf.run_with(env, torch_mod=_FakeTorch())
    assert code == 0
    out = capsys.readouterr().out.strip()
    assert json.loads(out)["oar_job_id"] == "OAR-1"


def test_main_returns_nonzero_on_failure(capsys):
    env = _env(system="Darwin")
    code = pf.run_with(env, torch_mod=_FakeTorch())
    assert code != 0
    err = capsys.readouterr().err
    assert "Linux" in err
