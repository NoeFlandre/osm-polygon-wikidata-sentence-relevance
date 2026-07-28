from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "grid5000" / "_submit_gpu_job.sh"


def _run(
    tmp_path: Path,
    *,
    exotic: str,
    memory: int = 49_140,
    production: str = "NO",
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "oarsub.calls"
    inventory = {
        "node-1": {
            "state": "Alive",
            "gpu_count": 2,
            "gpu_mem": memory,
            "gpu_compute_capability_major": 8,
            "exotic": exotic,
            "production": production,
        }
    }
    fake_bin.joinpath("oarnodes").write_text(
        f"#!/usr/bin/env bash\nprintf '%s\\n' '{json.dumps(inventory)}'\n",
        encoding="utf-8",
    )
    fake_bin.joinpath("oarnodes").chmod(0o700)
    fake_bin.joinpath("oarsub").write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" > "{calls}"\n'
        "printf 'OAR_JOB_ID=123\\n'\n",
        encoding="utf-8",
    )
    fake_bin.joinpath("oarsub").chmod(0o700)
    return (
        subprocess.run(
            ["bash", str(SCRIPT), "40000", "00:55:00", "day", "exec true"],
            env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
            text=True,
            capture_output=True,
            check=False,
        ),
        calls,
    )


def test_submit_helper_is_executable() -> None:
    assert SCRIPT.stat().st_mode & 0o111


def test_standard_gpu_omits_exotic_type(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, exotic="NO")
    assert result.returncode == 0, result.stderr
    command = calls.read_text(encoding="utf-8")
    assert "-t exotic" not in command
    assert "-t day" in command
    assert "gpu_mem>=40000" in command


def test_exotic_gpu_adds_exotic_type(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, exotic="YES")
    assert result.returncode == 0, result.stderr
    assert "-t exotic -t day" in calls.read_text(encoding="utf-8")


def test_production_gpu_uses_production_queue_and_property(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, exotic="NO", production="YES")
    assert result.returncode == 0, result.stderr
    command = calls.read_text(encoding="utf-8")
    assert "-q production" in command
    assert "production='YES'" in command
    assert "-t exotic" not in command


def test_no_matching_gpu_refuses_before_oarsub(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, exotic="NO", memory=39_999)
    assert result.returncode != 0
    assert "no compatible live GPU resource" in result.stderr
    assert not calls.exists()
