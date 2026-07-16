"""Integration tests for the built distribution contents.

Always rebuilds a fresh sdist + wheel via ``uv build``, then runs the
stdlib-only ``scripts/verify_distribution.py`` against the produced
artifacts and asserts that the helper exits successfully.

Cleanup contract:
- The test never leaves ``dist/``, ``build/``, or generated egg-info
  directories behind.
- Any pre-existing ``dist/``, ``build/``, or
  ``src/osm_polygon_sentence_relevance.egg-info`` is preserved (via a
  temporary copy) and restored after the test, so running it is safe
  even in environments where those directories already contain real
  artifacts.

The test is **skipped** only when the ``uv`` executable is genuinely
unavailable; build failures, missing artifacts, and verifier failures
are hard test failures with captured stdout/stderr.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_distribution.py"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
# ``uv build`` writes egg-info as a sibling of the package source root,
# i.e. ``src/<package-name>.egg-info``. This must match exactly or the
# cleanup contract silently no-ops.
EGG_INFO = ROOT / "src" / "osm_polygon_sentence_relevance.egg-info"


def _snapshot(path: Path) -> Path | None:
    """Return a temporary *copy* of *path* if it exists, else ``None``.

    Used to preserve pre-existing ``dist/`` and ``build/`` contents.
    """
    if not path.exists():
        return None
    backup = Path(tempfile.mkdtemp(prefix=f".{path.name}.bak."))
    shutil.copytree(path, backup / path.name)
    return backup


def _restore(path: Path, backup: Path | None) -> None:
    """Restore *path* from *backup* (a copy created by :func:`_snapshot`)."""
    if backup is None:
        return
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    shutil.copytree(backup / path.name, path)
    shutil.rmtree(backup)


def _clear(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _run_distribution_cycle() -> None:
    """Run a full build + verify cycle inside the cleanup context.

    Snapshots and restores ``dist/``, ``build/``, and the package
    ``egg-info`` directory so neither the test run nor any failure
    during the run destroys pre-existing user content.
    """
    if shutil.which("uv") is None:
        pytest.skip("The 'uv' executable is not available on this machine")

    dist_backup = _snapshot(DIST)
    build_backup = _snapshot(BUILD)
    egg_info_backup = _snapshot(EGG_INFO)

    try:
        # Always rebuild from clean state so the test is never satisfied
        # by stale artifacts from a previous run.
        _clear(DIST)
        _clear(BUILD)
        _clear(EGG_INFO)

        build = subprocess.run(
            ["uv", "build"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert build.returncode == 0, (
            f"'uv build' failed (exit {build.returncode}).\n"
            f"STDOUT:\n{build.stdout}\nSTDERR:\n{build.stderr}"
        )

        wheels = sorted(DIST.glob("*.whl"))
        sdists = sorted(DIST.glob("*.tar.gz"))
        assert wheels, (
            f"'uv build' reported success but no wheel is in {DIST}.\n"
            f"STDOUT:\n{build.stdout}\nSTDERR:\n{build.stderr}"
        )
        assert sdists, (
            f"'uv build' reported success but no sdist is in {DIST}.\n"
            f"STDOUT:\n{build.stdout}\nSTDERR:\n{build.stderr}"
        )
        wheel, sdist = wheels[-1], sdists[-1]

        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(wheel), str(sdist)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, (
            f"distribution verification failed.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        assert "OK" in proc.stdout
    finally:
        # Always clean up generated artifacts, then restore any pre-existing
        # dist/, build/, and egg-info that the user had on disk before the
        # test. Cleanup must run on success AND every failure path.
        _clear(DIST)
        _clear(BUILD)
        _clear(EGG_INFO)
        if dist_backup is not None:
            _restore(DIST, dist_backup)
        if build_backup is not None:
            _restore(BUILD, build_backup)
        if egg_info_backup is not None:
            _restore(EGG_INFO, egg_info_backup)
        # Final invariant: no stray egg-info directories left in src/
        # unless one pre-existed before the test (in which case it's
        # backed up and restored). Without EGG_INFO pointing at the
        # real generated path this glob would find a leftover.
        leftovers = sorted((EGG_INFO.parent).glob("*.egg-info"))
        if egg_info_backup is None:
            assert not leftovers, (
                f"_run_distribution_cycle left stray egg-info dirs in "
                f"{EGG_INFO.parent}: {leftovers}"
            )


def test_verify_distribution_passes_on_fresh_build():
    _run_distribution_cycle()


def test_egg_info_constant_points_to_real_generated_path():
    """Sanity guard: ``EGG_INFO`` must point to the directory ``uv build``
    actually writes. Without this guard a path drift would silently
    disable the cleanup contract.
    """
    assert EGG_INFO.name.endswith(".egg-info"), (
        f"EGG_INFO = {EGG_INFO} does not look like the generated egg-info dir"
    )
    assert EGG_INFO.parent.name == "src", f"EGG_INFO = {EGG_INFO} parent is not 'src'"


def test_pre_existing_egg_info_is_preserved_by_cleanup():
    """Regression: a pre-existing ``egg-info`` directory must round-trip.

    ``uv build`` may write into ``src/osm_polygon_sentence_relevance.egg-info``
    on every run. The cleanup contract must snapshot any pre-existing
    contents, let the build run, and restore them unchanged at the end:
    same directory, same set of files, same file contents.
    """
    # Skip if 'uv' is unavailable: the regression cannot be exercised.
    if shutil.which("uv") is None:
        pytest.skip("The 'uv' executable is not available on this machine")

    # Skip if the user truly has an egg-info on disk that contains a
    # real PKG-INFO (we don't want to clobber a real one). The cleanup
    # contract still guarantees that one would be preserved; we just
    # can't run an in-place regression against an unexpected state.
    if EGG_INFO.exists() and (EGG_INFO / "PKG-INFO").is_file():
        pytest.skip("egg-info contains a real PKG-INFO; skipping regression")

    marker_name = "user-marker.txt"
    marker = EGG_INFO / marker_name
    marker_payload = "USER-MARKER-PRE-EXISTING\n"

    # Set up a pre-existing egg-info with a stable marker.
    EGG_INFO.mkdir(exist_ok=True)
    marker.write_text(marker_payload, encoding="utf-8")

    # Capture the directory listing and a content fingerprint before
    # the cycle so we can detect any drift (uv build may add PKG-INFO,
    # SOURCES.txt, etc. alongside the user's marker if egg-info is not
    # snapshotted and restored by the cleanup contract).
    before_names = sorted(p.name for p in EGG_INFO.iterdir())
    before_hashes = {p.name: p.read_bytes() for p in EGG_INFO.iterdir()}

    try:
        # Drive a full distribution cycle (includes uv build + cleanup).
        _run_distribution_cycle()

        # After the test, the user's egg-info and marker must round-trip.
        assert EGG_INFO.is_dir(), f"pre-existing {EGG_INFO} was removed by the cleanup"
        after_names = sorted(p.name for p in EGG_INFO.iterdir())
        assert after_names == before_names, (
            f"pre-existing {EGG_INFO} contents drifted across the cycle: "
            f"before={before_names}, after={after_names}"
        )
        after_hashes = {p.name: p.read_bytes() for p in EGG_INFO.iterdir()}
        assert after_hashes == before_hashes, (
            f"pre-existing {EGG_INFO} file contents were modified by the cleanup"
        )
        assert marker.is_file(), f"pre-existing {marker} was removed by the cleanup"
        assert marker.read_text(encoding="utf-8") == marker_payload, (
            f"pre-existing {marker} contents were modified by the cleanup"
        )

        # Also assert that any temporary backup dirs left behind by
        # _snapshot (which uses tempfile.mkdtemp) are NOT lingering in
        # the repository tree under the package source dir.
        leftover_backups = list(EGG_INFO.parent.glob(".egg-info.bak.*"))
        assert not leftover_backups, (
            f"leftover egg-info backups under {EGG_INFO.parent}: {leftover_backups}"
        )
    finally:
        # Best-effort cleanup of the marker and the empty egg-info we
        # created. If the snapshot/restore behavior is correct the
        # user's marker is still here and we remove it; if a snapshot
        # was taken (because something else had pre-created egg-info)
        # the restore will have replaced it cleanly already.
        if marker.is_file():
            marker.unlink()
        if EGG_INFO.is_dir():
            with contextlib.suppress(OSError):
                # Restored/pre-existing artifacts may remain; that's fine.
                EGG_INFO.rmdir()
