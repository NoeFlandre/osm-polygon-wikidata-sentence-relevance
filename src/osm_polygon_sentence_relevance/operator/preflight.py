"""Local checkout and remote resource checks used before operator work."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import PurePosixPath
from typing import Any

from osm_polygon_sentence_relevance.operator.config import (
    INPUT_DATASET_ID,
    OUTPUT_DATASET_ID,
    Stage,
)
from osm_polygon_sentence_relevance.operator.ssh import SshClient


def _result_text(result: Any) -> str:
    """Return compatible text from local subprocess and SSH result objects."""

    text_attr = getattr(result, "text", None)
    if text_attr is not None:
        return text_attr
    return getattr(result, "stdout", "")


def git_head() -> str:
    """Return the current commit when the checkout is immutable and clean."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    value = _result_text(result).strip()
    if len(value) != 40:
        raise RuntimeError("current source checkout has no immutable commit")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    if dirty.stdout:
        raise RuntimeError("current source checkout must be clean")
    return value


def resolve_input_revision(explicit: str | None, stage: str) -> str:
    """Resolve the immutable Hub revision for the selected workflow stage."""

    if explicit:
        return explicit
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "the hub extra is required to resolve the input revision"
        ) from exc
    dataset_id = OUTPUT_DATASET_ID if stage == Stage.LABEL.value else INPUT_DATASET_ID
    sha = HfApi().dataset_info(dataset_id, revision="main").sha
    if not sha:
        raise RuntimeError("input dataset main did not resolve to a commit")
    return sha


def remote_home(ssh: SshClient) -> PurePosixPath:
    """Read and validate the remote user's home directory."""

    result = ssh.run('printf "%s\\n" "$HOME"')
    value = _result_text(result).strip()
    if not value.startswith("/") or "\n" in value or ".." in value.split("/"):
        raise RuntimeError("remote home path is invalid")
    return PurePosixPath(value)


def usage_policy_preflight(ssh: SshClient, site: str) -> None:
    """Fail closed unless Grid'5000's live usage-policy checks succeed."""

    if re.fullmatch(r"[a-z][a-z0-9-]*", site) is None:
        raise ValueError("Grid'5000 site name is invalid")
    quoted_site = shlex.quote(site)
    ssh.run(
        "command -v usagepolicycheck >/dev/null && "
        f"usagepolicycheck -l --sites {quoted_site} >/dev/null && "
        "usagepolicycheck -t >/dev/null"
    )


__all__ = [
    "git_head",
    "remote_home",
    "resolve_input_revision",
    "usage_policy_preflight",
]
