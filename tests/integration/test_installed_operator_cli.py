"""Smoke tests for the operator's installed-package boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_operator_cli_imports_without_source_only_scripts_package(
    tmp_path: Path,
) -> None:
    """The installed operator must not depend on the excluded scripts tree."""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from osm_polygon_sentence_relevance.operator.cli import app; "
            "assert app.info.help",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "operator CLI must import without the source-only scripts package\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
