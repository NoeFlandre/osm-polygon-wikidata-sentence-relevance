"""Executable shell tests for the strict clean-checkout guard.

The guard must inspect tracked, staged, and untracked entries and reject
every dirty entry except the single explicitly approved ``.venv``
deployment entry. These tests build real temporary Git repositories with
controlled dirty states and invoke the guard directly.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GUARD = ROOT / "scripts" / "grid5000" / "_checkout_guard.sh"


def _run_guard(
    repo_root: Path, approved_root: str = ""
) -> subprocess.CompletedProcess[str]:
    script = (
        f"set +e\n"
        f". '{GUARD}'\n"
        f"validate_clean_checkout '{repo_root}' '{approved_root}'\n"
        f'echo "exit=$?"\n'
    )
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _init_repo(path: Path) -> None:
    """Create an initialised repository with one tracked file."""

    path.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init", "-q", str(path)], check=True, env=env)
    (path / "tracked.txt").write_text("hi\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "tracked.txt"],
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t"],
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init", "-q"],
        check=True,
        env=env,
    )


def _create_venv(path: Path) -> None:
    """Create a minimal ``.venv`` directory with the required binaries."""

    bin_dir = path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "python").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "python").chmod(0o755)
    (bin_dir / "osm-polygon-label-sentences").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "osm-polygon-label-sentences").chmod(0o755)


def _expect_pass(result: subprocess.CompletedProcess[str]) -> None:
    assert "exit=0" in result.stdout, (
        f"expected pass but got stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def _expect_fail(result: subprocess.CompletedProcess[str]) -> None:
    assert "exit=0" not in result.stdout, (
        f"expected fail but got stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_guard_accepts_clean_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    result = _run_guard(repo)
    _expect_pass(result)


def test_guard_accepts_approved_venv_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _create_venv(repo / ".venv")
    result = _run_guard(repo)
    _expect_pass(result)


def test_guard_accepts_nested_ignored_entries_within_venv(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _create_venv(repo / ".venv")
    nested = repo / ".venv" / "lib"
    nested.mkdir()
    (nested / "site-packages").mkdir(parents=True)
    (nested / "site-packages" / "evil.pth").write_text("/tmp/evil\n")
    result = _run_guard(repo)
    _expect_pass(result)


def test_guard_accepts_grid5000_oar_output(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "OAR.123456.stdout").write_text("")
    (repo / "OAR.123456.stderr").write_text("")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.write_text("OAR.*.stdout\nOAR.*.stderr\n")
    result = _run_guard(repo)
    _expect_pass(result)


def test_guard_accepts_python_bytecode_caches(tmp_path: Path) -> None:
    """Generated ``__pycache__`` directories must not poison a reuse run."""

    repo = tmp_path / "repo"
    _init_repo(repo)
    package = repo / "src" / "package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n")
    subprocess.run(
        ["git", "-C", str(repo), "add", "src/package/__init__.py"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "add-package", "-q"],
        check=True,
    )
    cache = repo / "src" / "package" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "module.cpython-312.pyc").write_bytes(b"bytecode")
    (repo / ".git" / "info" / "exclude").write_text("__pycache__/\n")
    result = _run_guard(repo)
    _expect_pass(result)


def test_guard_rejects_symlinked_python_bytecode_cache(tmp_path: Path) -> None:
    """A cache path must not be a symlink that escapes the checkout."""

    repo = tmp_path / "repo"
    _init_repo(repo)
    package = repo / "src" / "package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n")
    subprocess.run(
        ["git", "-C", str(repo), "add", "src/package/__init__.py"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "add-package", "-q"],
        check=True,
    )
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    os.symlink(str(outside), str(package / "__pycache__"))
    (repo / ".git" / "info" / "exclude").write_text("__pycache__/\n")
    result = _run_guard(repo)
    _expect_fail(result)


def test_guard_rejects_non_oar_ignored_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "random.tmp").write_text("")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.write_text("*.tmp\n")
    result = _run_guard(repo)
    _expect_fail(result)
    assert "ignored entry is not allowed" in result.stderr


def test_guard_accepts_approved_venv_symlink_into_run_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    run_root = tmp_path / "run-root"
    run_root.mkdir()
    venv_target = run_root / ".venv"
    _create_venv(venv_target)
    os.symlink(str(venv_target), str(repo / ".venv"))
    result = _run_guard(repo, str(run_root))
    _expect_pass(result)


def test_guard_rejects_tracked_modification(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "tracked.txt").write_text("changed\n")
    result = _run_guard(repo)
    _expect_fail(result)
    assert "checkout is dirty" in result.stderr


def test_guard_rejects_staged_addition(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "new.txt").write_text("new\n")
    subprocess.run(
        ["git", "-C", str(repo), "add", "new.txt"],
        check=True,
        capture_output=True,
    )
    result = _run_guard(repo)
    _expect_fail(result)


def test_guard_rejects_untracked_python_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "sitecustomize.py").write_text("#!/usr/bin/env python\n")
    result = _run_guard(repo)
    _expect_fail(result)


def test_guard_rejects_untracked_shell_script(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "wrapper.sh").write_text("#!/bin/sh\nexit 0\n")
    (repo / "wrapper.sh").chmod(0o755)
    result = _run_guard(repo)
    _expect_fail(result)


def test_guard_rejects_untracked_pth_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "evil.pth").write_text("/tmp/evil\n")
    result = _run_guard(repo)
    _expect_fail(result)


def test_guard_rejects_usercustomize_py(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "usercustomize.py").write_text("#!/usr/bin/env python\n")
    result = _run_guard(repo)
    _expect_fail(result)


def test_guard_rejects_package_shadow_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    shadow = repo / "osm_polygon_sentence_relevance"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("# shadow\n")
    result = _run_guard(repo)
    _expect_fail(result)


def test_guard_rejects_two_untracked_venvs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _create_venv(repo / ".venv")
    nested = repo / "nested" / ".venv"
    _create_venv(nested)
    result = _run_guard(repo)
    _expect_fail(result)


def test_guard_rejects_venv_symlink_target_outside_run_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    outside = tmp_path / "outside-venv"
    _create_venv(outside)
    os.symlink(str(outside), str(repo / ".venv"))
    run_root = tmp_path / "approved-run-root"
    run_root.mkdir()
    result = _run_guard(repo, str(run_root))
    _expect_fail(result)


def test_guard_rejects_broken_venv_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    os.symlink(str(tmp_path / "missing-venv"), str(repo / ".venv"))
    run_root = tmp_path / "approved-run-root"
    run_root.mkdir()
    result = _run_guard(repo, str(run_root))
    _expect_fail(result)


def test_guard_rejects_venv_symlink_chained_outside_root(tmp_path: Path) -> None:
    """A symlink inside the run root that points outside must fail closed."""

    repo = tmp_path / "repo"
    _init_repo(repo)
    run_root = tmp_path / "approved-run-root"
    run_root.mkdir()
    intermediate = run_root / "link-source"
    outside = tmp_path / "outside-venv"
    _create_venv(outside)
    os.symlink(str(outside), str(intermediate))
    os.symlink(str(intermediate), str(repo / ".venv"))
    result = _run_guard(repo, str(run_root))
    _expect_fail(result)


def test_guard_rejects_venv_without_installed_cli(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    bin_dir = repo / ".venv" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "python").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "python").chmod(0o755)
    result = _run_guard(repo)
    _expect_fail(result)


def test_guard_rejects_regular_file_named_venv(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".venv").write_text("not a directory\n")
    result = _run_guard(repo)
    _expect_fail(result)


def test_guard_rejects_venv_named_with_subpath(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "subdir").mkdir()
    _create_venv(repo / "subdir" / ".venv")
    result = _run_guard(repo)
    _expect_fail(result)


def test_guard_rejects_untracked_empty_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "scratch").mkdir()
    (repo / "scratch" / ".keep").write_text("")
    result = _run_guard(repo)
    _expect_fail(result)


def test_guard_includes_untracked_files_check(tmp_path: Path) -> None:
    """The guard must inspect untracked entries; ``--untracked-files=no`` would miss them."""

    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "sitecustomize.py").write_text("#!/usr/bin/env python\n")
    # Sanity-check: ``git status --porcelain --untracked-files=no`` does NOT list it.
    hidden = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    )
    assert hidden.stdout == ""
    # The guard must still reject it.
    result = _run_guard(repo)
    _expect_fail(result)
