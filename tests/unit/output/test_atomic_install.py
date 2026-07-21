"""Tests for the rollback-safe atomic installation helper.

The corrective release's publication contract depends on
``install_atomic`` correctly handling three scenarios:

1. a clean swap (no existing output): backup is None;
2. a swap over an existing output: backup is created and returned;
3. a swap failure that triggers backup restoration.

These tests pin the contract so the validator's ``os.rename``-based
swap cannot silently regress.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from osm_polygon_sentence_relevance.contracts.errors import ExportError
from osm_polygon_sentence_relevance.output.atomic import (
    cleanup_on_failure,
    install_atomic,
    remove_backup,
)


class TestInstallAtomic:
    def test_clean_install_with_no_existing_output(self, tmp_path: Path) -> None:
        """No existing output means no backup directory."""
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        (new_dir / "marker").write_text("hi")
        target = tmp_path / "out"

        backup = install_atomic(new_dir, target)
        assert backup is None
        assert target.is_dir()
        assert (target / "marker").read_text() == "hi"
        # The tmp source directory was consumed by os.rename.
        assert not new_dir.exists()

    def test_install_over_existing_output_creates_backup(self, tmp_path: Path) -> None:
        """An existing output is renamed aside before the swap."""
        existing = tmp_path / "out"
        existing.mkdir()
        (existing / "old.txt").write_text("old")
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        (new_dir / "new.txt").write_text("new")

        backup = install_atomic(new_dir, existing)
        # The backup must exist; the caller removes it after success.
        assert backup is not None
        assert backup.exists()
        assert (backup / "old.txt").read_text() == "old"
        # The target now holds the new content.
        assert (existing / "new.txt").read_text() == "new"

    def test_install_failure_restores_backup(self, tmp_path: Path) -> None:
        """A rename failure restores the backup and surfaces the error."""
        existing = tmp_path / "out"
        existing.mkdir()
        (existing / "old.txt").write_text("old")
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        (new_dir / "new.txt").write_text("new")

        # Patch os.rename to fail on the second call (the new→target
        # rename), forcing the backup-restore branch.
        real_rename = os.rename
        calls = {"count": 0}

        def flaky_rename(src, dst):
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("simulated swap failure")
            return real_rename(src, dst)

        # Patch the name imported by ``install_atomic``.
        import osm_polygon_sentence_relevance.output.atomic as atomic_mod

        original_module_rename = atomic_mod.os.rename
        atomic_mod.os.rename = flaky_rename
        try:
            with pytest.raises(OSError, match="simulated swap failure"):
                install_atomic(new_dir, existing)
            # The existing directory was restored with its old content.
            assert existing.is_dir()
            assert (existing / "old.txt").read_text() == "old"
        finally:
            atomic_mod.os.rename = original_module_rename

    def test_install_failure_when_restore_also_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A swap failure plus a restore failure raises ExportError
        with the backup path so the caller can recover."""
        existing = tmp_path / "out"
        existing.mkdir()
        new_dir = tmp_path / "new"
        new_dir.mkdir()

        # First rename (existing→backup) succeeds, second rename
        # (new→existing) fails, third rename (backup→existing)
        # also fails. The helper must raise ExportError.
        real_rename = os.rename
        calls = {"count": 0}

        def both_fail(src, dst):
            calls["count"] += 1
            if calls["count"] == 1:
                # Existing → backup succeeds.
                return real_rename(src, dst)
            if calls["count"] == 2:
                raise OSError("swap failed")
            # 3rd call: backup restore fails.
            raise OSError("restore failed")

        import osm_polygon_sentence_relevance.output.atomic as atomic_mod

        original_module_rename = atomic_mod.os.rename
        atomic_mod.os.rename = both_fail
        try:
            with pytest.raises(ExportError, match="preserved at"):
                install_atomic(new_dir, existing)
        finally:
            atomic_mod.os.rename = original_module_rename


class TestRemoveBackup:
    def test_remove_backup_succeeds(self, tmp_path: Path) -> None:
        backup = tmp_path / "b"
        backup.mkdir()
        (backup / "x").write_text("x")
        remove_backup(backup)
        assert not backup.exists()

    def test_remove_backup_failure_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ExportError, match="failed to delete"):
            remove_backup(tmp_path / "ghost")


class TestCleanupOnFailure:
    def test_cleanup_removes_tmp_dir(self, tmp_path: Path) -> None:
        tmp = tmp_path / "tmp"
        tmp.mkdir()
        cleanup_on_failure(tmp, None)
        assert not tmp.exists()

    def test_cleanup_removes_backup(self, tmp_path: Path) -> None:
        backup = tmp_path / "b"
        backup.mkdir()
        cleanup_on_failure(None, backup)
        assert not backup.exists()

    def test_cleanup_no_op_when_paths_missing(self, tmp_path: Path) -> None:
        # Calling with non-existent paths must not raise.
        cleanup_on_failure(tmp_path / "ghost", tmp_path / "ghost")
