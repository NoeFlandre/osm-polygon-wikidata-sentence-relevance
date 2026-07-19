"""Unit tests for the Grid'5000 CUDA preflight (Phase 9B).

These run on the Mac without CUDA. Every platform/environment/Torch
touch is injected through ``PreflightEnv`` facades and a fake
``torch`` module, so no real GPU is required and no network access
is attempted.
"""

from __future__ import annotations

import json

import pytest
from scripts.grid5000 import gpu_preflight as pf


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
    env = _env(environ={"CUDA_VISIBLE_DEVICES": "0"})
    with pytest.raises(pf.PreflightError, match="OAR_JOB_ID"):
        pf.run_preflight(env, torch_mod=_FakeTorch())


def test_rejects_blank_oar_job_id():
    env = _env(environ={"OAR_JOB_ID": "   ", "CUDA_VISIBLE_DEVICES": "0"})
    with pytest.raises(pf.PreflightError, match="OAR_JOB_ID"):
        pf.run_preflight(env, torch_mod=_FakeTorch())


def test_rejects_missing_cuda_visibility():
    env = _env(environ={"OAR_JOB_ID": "123"})
    with pytest.raises(pf.PreflightError, match="CUDA_VISIBLE_DEVICES"):
        pf.run_preflight(env, torch_mod=_FakeTorch())


def test_rejects_blank_cuda_visibility():
    env = _env(environ={"OAR_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": ""})
    with pytest.raises(pf.PreflightError, match="CUDA_VISIBLE_DEVICES"):
        pf.run_preflight(env, torch_mod=_FakeTorch())


def test_rejects_unavailable_cuda():
    env = _env(environ={"OAR_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"})
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

    env = _env(environ={"OAR_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"})
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

    env = _env(environ={"OAR_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0,1"})
    with pytest.raises(pf.PreflightError, match="exactly one"):
        pf.run_preflight(env, torch_mod=_FakeTorchTwo())


# --- Success ----------------------------------------------------------


def test_emits_expected_json():
    env = _env(environ={"OAR_JOB_ID": "OAR-456", "CUDA_VISIBLE_DEVICES": "0"})
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
    env = _env(environ={"OAR_JOB_ID": "OAR-456", "CUDA_VISIBLE_DEVICES": "0"})
    result = pf.run_preflight(env, torch_mod=_FakeTorch())
    # The runtime version is a non-empty string (the fake exposes one).
    assert isinstance(result.torch_cuda_runtime_version, str)
    assert result.torch_cuda_runtime_version


# --- main() exit codes ------------------------------------------------


def test_main_returns_zero_on_success(capsys):
    env = _env(environ={"OAR_JOB_ID": "OAR-1", "CUDA_VISIBLE_DEVICES": "0"})
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
