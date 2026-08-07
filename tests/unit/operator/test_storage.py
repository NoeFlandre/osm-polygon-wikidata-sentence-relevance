"""Contracts for production Grid'5000 remote-storage management.

These tests pin the public storage API that the operator CLI delegates to:

* staging-headroom reservation (label/all stages need at least 22 GiB);
* the diagnosis that cleanup alone could restore site compatibility;
* read-only managed-run cleanup (preview and execute);
* home-quota headroom enforcement (check, clean, recheck, fail closed).

All tests use fake ``SshClient`` objects and never contact a real frontend.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator.sites import SiteProbe, SiteRequirements
from osm_polygon_sentence_relevance.operator.ssh import SshError
from osm_polygon_sentence_relevance.operator.storage import (
    LABEL_STAGING_HEADROOM_BYTES,
    cleanup_can_restore_compatibility,
    cleanup_managed_runs,
    ensure_home_headroom,
    required_staging_headroom,
)

_GIB = 1024**3
_HEADROOM_QUOTA = "0 25000000 100000000\n"
_OVER_QUOTA = "30000000* 25000000 100000000\n"
_OK_QUOTA = "10000000 25000000 100000000\n"


class _ScriptSsh:
    """Fake SSH that records every command and returns canned stdout."""

    def __init__(self, outputs: Iterable[object]) -> None:
        self._outputs = list(outputs)
        self.commands: list[str] = []

    def run(self, command: str) -> SimpleNamespace:
        self.commands.append(command)
        value = self._outputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        return SimpleNamespace(stdout=value)


def _compatible_probe(*, persistent_free_bytes: int) -> SiteProbe:
    return SiteProbe(
        "nancy",
        "nancy",
        True,
        80_000,
        (8, 0),
        persistent_free_bytes,
        0,
    )


# -----------------------------------------------------------------------
# required_staging_headroom
# -----------------------------------------------------------------------


def test_split_stage_uses_requested_headroom() -> None:
    assert required_staging_headroom("split", 8 * _GIB) == 8 * _GIB


def test_label_stage_reserves_at_least_22_gib() -> None:
    assert required_staging_headroom("label", 8 * _GIB) == LABEL_STAGING_HEADROOM_BYTES
    assert LABEL_STAGING_HEADROOM_BYTES == 22 * _GIB


def test_all_stage_keeps_larger_request() -> None:
    assert required_staging_headroom("all", 24 * _GIB) == 24 * _GIB


def test_negative_headroom_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        required_staging_headroom("label", -1)


# -----------------------------------------------------------------------
# cleanup_can_restore_compatibility
# -----------------------------------------------------------------------


def test_cleanup_can_help_when_only_storage_fails() -> None:
    requirements = SiteRequirements(
        gpu_memory_mb=40_000,
        persistent_free_bytes=10_000,
    )
    low_storage = _compatible_probe(persistent_free_bytes=1)
    assert cleanup_can_restore_compatibility([low_storage], requirements)


def test_cleanup_cannot_help_when_gpu_fails() -> None:
    requirements = SiteRequirements(
        gpu_memory_mb=40_000,
        persistent_free_bytes=10_000,
    )
    no_gpu = SiteProbe("x", "x", True, 0, None, 100_000, 0)
    assert not cleanup_can_restore_compatibility([no_gpu], requirements)


def test_cleanup_cannot_help_when_storage_is_sufficient() -> None:
    requirements = SiteRequirements(
        gpu_memory_mb=40_000,
        persistent_free_bytes=10_000,
    )
    ok = _compatible_probe(persistent_free_bytes=100_000)
    assert not cleanup_can_restore_compatibility([ok], requirements)


# -----------------------------------------------------------------------
# ensure_home_headroom
# -----------------------------------------------------------------------


def test_sufficient_quota_skips_cleanup() -> None:
    ssh = _ScriptSsh([_OK_QUOTA])
    ensure_home_headroom(
        ssh,  # type: ignore[arg-type]
        protected_root=PurePosixPath("/h/current"),
        minimum_headroom_bytes=1024,
    )
    assert len(ssh.commands) == 1
    assert "quota" in ssh.commands[0]


def test_insufficient_quota_invokes_cleanup_once_and_rechecks() -> None:
    ssh = _ScriptSsh([_OVER_QUOTA, "/home/old\n", _OK_QUOTA])
    ensure_home_headroom(
        ssh,  # type: ignore[arg-type]
        protected_root=PurePosixPath("/h/current"),
        minimum_headroom_bytes=15_000_000 * 1024,
    )
    assert len(ssh.commands) == 3
    assert "quota" in ssh.commands[0]
    assert "rm -rf" in ssh.commands[1]
    assert "current" in ssh.commands[1]
    assert "quota" in ssh.commands[2]


def test_continued_insufficiency_fails_closed() -> None:
    ssh = _ScriptSsh([_OVER_QUOTA, "", _OVER_QUOTA])
    with pytest.raises(RuntimeError, match="soft quota"):
        ensure_home_headroom(
            ssh,  # type: ignore[arg-type]
            protected_root=PurePosixPath("/h/current"),
            minimum_headroom_bytes=15_000_000 * 1024,
        )


def test_negative_minimum_headroom_rejected() -> None:
    ssh = _ScriptSsh([])
    with pytest.raises(ValueError, match="non-negative"):
        ensure_home_headroom(
            ssh,  # type: ignore[arg-type]
            protected_root=PurePosixPath("/h/current"),
            minimum_headroom_bytes=-1,
        )


def test_quota_error_propagates() -> None:
    ssh = _ScriptSsh(
        [SshError("down", category="transport", returncode=255, attempts=1)]
    )
    with pytest.raises(SshError):
        ensure_home_headroom(
            ssh,  # type: ignore[arg-type]
            protected_root=PurePosixPath("/h/current"),
            minimum_headroom_bytes=1024,
        )


# -----------------------------------------------------------------------
# cleanup_managed_runs
# -----------------------------------------------------------------------


def test_preview_reports_paths_without_deletion() -> None:
    ssh = _ScriptSsh(["/home/run-a\n/home/run-b\n"])
    result = cleanup_managed_runs(ssh, execute=False)  # type: ignore[arg-type]
    assert result == ("/home/run-a", "/home/run-b")
    assert "preview" in ssh.commands[0]
    assert "delete" not in ssh.commands[0].split("preview")[0]


def test_execute_deletes_eligible_paths() -> None:
    ssh = _ScriptSsh(["/home/run-a\n"])
    cleanup_managed_runs(ssh, execute=True)  # type: ignore[arg-type]
    assert "delete" in ssh.commands[0]
    assert "rm -rf" in ssh.commands[0]


def test_protected_root_is_excluded_in_script() -> None:
    ssh = _ScriptSsh([""])
    protected = PurePosixPath("/home/osm-polygon-operator/current")
    cleanup_managed_runs(
        ssh,  # type: ignore[arg-type]
        execute=True,
        protected_root=protected,
    )
    assert "current" in ssh.commands[0]


def test_no_protected_root_when_omitted() -> None:
    ssh = _ScriptSsh([""])
    cleanup_managed_runs(ssh, execute=False)  # type: ignore[arg-type]
    # When no protected root, the script still runs but the guard uses empty.
    assert "osm-polygon-operator" in ssh.commands[0]


def test_empty_results() -> None:
    ssh = _ScriptSsh([""])
    assert cleanup_managed_runs(ssh, execute=False) == ()  # type: ignore[arg-type]


def test_missing_managed_root_exits_cleanly() -> None:
    ssh = _ScriptSsh([""])
    cleanup_managed_runs(ssh, execute=True)  # type: ignore[arg-type]
    assert "[ -d" in ssh.commands[0]
    assert "exit 0" in ssh.commands[0]


def test_only_complete_and_failed_are_eligible() -> None:
    ssh = _ScriptSsh([""])
    cleanup_managed_runs(ssh, execute=False)  # type: ignore[arg-type]
    assert "complete|failed" in ssh.commands[0]


def test_failed_roots_with_checkpoints_are_preserved() -> None:
    ssh = _ScriptSsh([""])
    cleanup_managed_runs(ssh, execute=True)  # type: ignore[arg-type]
    command = ssh.commands[0]
    assert "checkpoints" in command
    assert "progress.json" in command


def test_active_and_unknown_status_excluded() -> None:
    ssh = _ScriptSsh([""])
    cleanup_managed_runs(ssh, execute=False)  # type: ignore[arg-type]
    # The case statement continues (skips) anything not complete|failed.
    script = ssh.commands[0]
    assert "*) continue" in script


def test_missing_marker_excluded() -> None:
    ssh = _ScriptSsh([""])
    cleanup_managed_runs(ssh, execute=False)  # type: ignore[arg-type]
    script = ssh.commands[0]
    assert ".operator-managed.json" in script
    assert "[ -f" in script


def test_symlink_candidate_excluded() -> None:
    ssh = _ScriptSsh([""])
    cleanup_managed_runs(ssh, execute=False)  # type: ignore[arg-type]
    script = ssh.commands[0]
    # The candidate guard inspects "$candidate" specifically, distinct from
    # the marker guard which inspects "$marker" below.
    assert '[ ! -L "$candidate" ]' in script


def test_symlink_marker_excluded() -> None:
    ssh = _ScriptSsh([""])
    cleanup_managed_runs(ssh, execute=False)  # type: ignore[arg-type]
    script = ssh.commands[0]
    # The marker guard requires a real, non-symlink file at "$marker".
    assert '[ -f "$marker" ]' in script
    assert '[ ! -L "$marker" ]' in script


def test_whitespace_safe_paths() -> None:
    """Paths with spaces are shell-quoted in the generated script."""

    protected = PurePosixPath("/home/my runs/current")
    ssh = _ScriptSsh([""])
    cleanup_managed_runs(
        ssh,  # type: ignore[arg-type]
        execute=False,
        protected_root=protected,
    )
    # shlex.quote wraps the path in single quotes.
    assert "'/home/my runs/current'" in ssh.commands[0]


def test_managed_root_is_never_removed() -> None:
    ssh = _ScriptSsh([""])
    cleanup_managed_runs(ssh, execute=True)  # type: ignore[arg-type]
    script = ssh.commands[0]
    # find uses -mindepth 1 so the root directory itself is never a candidate.
    assert "-mindepth 1" in script


def test_cleanup_confined_to_direct_children() -> None:
    ssh = _ScriptSsh([""])
    cleanup_managed_runs(ssh, execute=False)  # type: ignore[arg-type]
    script = ssh.commands[0]
    assert "-mindepth 1" in script
    assert "-maxdepth 1" in script


def test_cleanup_returns_only_nonempty_stdout_path_lines() -> None:
    ssh = _ScriptSsh(["/home/run-a\n/home/run-b\n\n/home/run-c\n\n"])
    result = cleanup_managed_runs(ssh, execute=False)  # type: ignore[arg-type]
    assert result == ("/home/run-a", "/home/run-b", "/home/run-c")
    assert len(ssh.commands) == 1
