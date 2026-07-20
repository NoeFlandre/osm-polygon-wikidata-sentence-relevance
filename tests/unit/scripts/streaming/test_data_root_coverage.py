"""Coverage-targeted tests for the data_root module uncovered branches."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from scripts.streaming.data_root import (
    _FORBIDDEN_LITERAL_PREFIXES,
    _FORBIDDEN_PAIRS,
    _FORBIDDEN_PATH_NAMES,
    DataRootRejected,
    _is_forbidden_tmp,
    check_data_root,
)

# ---------------------------------------------------------------------------
# _is_forbidden_tmp: prefix matching paths.
# ---------------------------------------------------------------------------


def test_is_forbidden_tmp_detects_trailing_slash_prefix(tmp_path: Path) -> None:
    """A trailing slash on the canonical prefix must still match."""
    p = tmp_path / "subdir"
    p.mkdir()
    # Pretend the realpath resolves to /tmp/subdir (with trailing slash
    # being implicit) by mocking realpath.
    with mock.patch(
        "scripts.streaming.data_root.os.path.realpath",
        return_value="/tmp/foo",
    ):
        assert _is_forbidden_tmp(p) is True


def test_is_forbidden_tmp_detects_private_var_tmp(tmp_path: Path) -> None:
    """A realpath under /private/var/tmp must be detected via pair check."""
    p = tmp_path / "x"
    with mock.patch(
        "scripts.streaming.data_root.os.path.realpath",
        return_value="/private/var/tmp/foo",
    ):
        assert _is_forbidden_tmp(p) is True


def test_is_forbidden_tmp_detects_dev_shm_pair(tmp_path: Path) -> None:
    """A realpath under /dev/shm is detected via (dev, shm) pair check."""
    p = tmp_path / "x"
    with mock.patch(
        "scripts.streaming.data_root.os.path.realpath",
        return_value="/dev/shm/foo",
    ):
        assert _is_forbidden_tmp(p) is True


def test_is_forbidden_tmp_accepts_normal_path(tmp_path: Path) -> None:
    """A normal /home/... path is accepted."""
    p = tmp_path / "normal"
    with mock.patch(
        "scripts.streaming.data_root.os.path.realpath",
        return_value=str(p),
    ):
        assert _is_forbidden_tmp(p) is False


def test_is_forbidden_tmp_handles_empty_string(tmp_path: Path) -> None:
    """An empty realpath candidate is skipped without raising."""
    p = tmp_path / "x"
    with mock.patch(
        "scripts.streaming.data_root.os.path.realpath",
        return_value="",
    ):
        assert _is_forbidden_tmp(p) is False


def test_is_forbidden_tmp_handles_relative_path(tmp_path: Path) -> None:
    """A relative realpath skips the component-pair branch safely."""
    p = tmp_path / "x"
    with mock.patch(
        "scripts.streaming.data_root.os.path.realpath",
        return_value="relative/path",
    ):
        assert _is_forbidden_tmp(p) is False


# ---------------------------------------------------------------------------
# check_data_root: error paths.
# ---------------------------------------------------------------------------


def test_check_data_root_rejects_role_none(tmp_path: Path) -> None:
    real = tmp_path / "ok"
    real.mkdir()
    with pytest.raises(ValueError, match="role"):
        check_data_root(real, role=None, min_free_bytes=0)


def test_check_data_root_rejects_role_non_string(tmp_path: Path) -> None:
    real = tmp_path / "ok"
    real.mkdir()
    with pytest.raises(ValueError, match="role"):
        check_data_root(real, role=123, min_free_bytes=0)  # type: ignore[arg-type]


def test_check_data_root_rejects_missing_dir(tmp_path: Path) -> None:
    p = tmp_path / "missing-dir"
    with pytest.raises(DataRootRejected) as ei:
        check_data_root(p, role="scratch", min_free_bytes=0)
    assert ei.value.reason == "NOT_REGULAR_DIR"


def test_check_data_root_rejects_symlink(tmp_path: Path) -> None:
    """A symlink that points to a missing target is NOT_REGULAR_DIR.

    We use a broken symlink because ``os.path.realpath`` follows
    valid symlinks; only broken ones survive as symlinks through
    realpath.
    """
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "does-not-exist")
    with pytest.raises(DataRootRejected) as ei:
        check_data_root(link, role="scratch", min_free_bytes=0)
    assert ei.value.reason == "NOT_REGULAR_DIR"


def test_check_data_root_rejects_file_not_dir(tmp_path: Path) -> None:
    p = tmp_path / "file.txt"
    p.write_text("hi")
    with pytest.raises(DataRootRejected) as ei:
        check_data_root(p, role="scratch", min_free_bytes=0)
    assert ei.value.reason == "NOT_REGULAR_DIR"


def test_check_data_root_rejects_below_ceiling(tmp_path: Path) -> None:
    real = tmp_path / "ok"
    real.mkdir()
    huge = 1 << 60  # 1 EiB
    with (
        mock.patch(
            "scripts.streaming.data_root.shutil.disk_usage",
            return_value=mock.Mock(free=huge - 1, total=huge, used=1),
        ),
        pytest.raises(DataRootRejected) as ei,
    ):
        check_data_root(real, role="scratch", min_free_bytes=huge)
    assert ei.value.reason == "BELOW_CEILING"


def test_check_data_root_accepts_with_sufficient_free_space(tmp_path: Path) -> None:
    real = tmp_path / "ok"
    real.mkdir()
    with mock.patch(
        "scripts.streaming.data_root.shutil.disk_usage",
        return_value=mock.Mock(free=1 << 30, total=1 << 35, used=1 << 30),
    ):
        out = check_data_root(real, role="scratch", min_free_bytes=1 << 20)
    assert out == real.resolve()


# ---------------------------------------------------------------------------
# Constants: structural sanity.
# ---------------------------------------------------------------------------


def test_forbidden_constants_contain_expected_values() -> None:
    assert "/tmp" in _FORBIDDEN_LITERAL_PREFIXES
    assert ("var", "tmp") in _FORBIDDEN_PAIRS
    assert ("dev", "shm") in _FORBIDDEN_PAIRS
    assert "tmp" in _FORBIDDEN_PATH_NAMES
    assert "dev" in _FORBIDDEN_PATH_NAMES


# ---------------------------------------------------------------------------
# discover_oar_scratch_dir & safe_cleanup_scratch branches.
# ---------------------------------------------------------------------------


def test_discover_oar_scratch_dir_raises_not_in_oar_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.streaming.data_root import discover_oar_scratch_dir

    monkeypatch.delenv("OAR_JOB_ID", raising=False)
    with pytest.raises(DataRootRejected) as ei:
        discover_oar_scratch_dir()
    assert ei.value.reason == "NOT_IN_OAR_JOB"


def test_discover_oar_scratch_dir_success_with_localscratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.streaming.data_root import discover_oar_scratch_dir

    scratch_target = tmp_path / "oar-12345"
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    monkeypatch.setenv("LOCALSCRATCH", str(scratch_target))
    res = discover_oar_scratch_dir(min_free_bytes=100)
    assert res == scratch_target.resolve()


def test_discover_oar_scratch_dir_raises_when_no_candidate_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.streaming.data_root import discover_oar_scratch_dir

    monkeypatch.setenv("OAR_JOB_ID", "99999")
    monkeypatch.setenv("LOCALSCRATCH", "/nonexistent/impossible/path/99999")
    with mock.patch(
        "scripts.streaming.data_root.shutil.disk_usage", side_effect=OSError("no du")
    ):
        with pytest.raises(DataRootRejected) as ei:
            discover_oar_scratch_dir(min_free_bytes=1 << 30)
        assert ei.value.reason == "NO_VALID_OAR_SCRATCH"


def test_safe_cleanup_scratch_rejects_protected_system_path() -> None:
    from scripts.streaming.data_root import safe_cleanup_scratch

    with pytest.raises(ValueError, match="protected system path"):
        safe_cleanup_scratch(Path("/tmp"))


def test_safe_cleanup_scratch_rejects_missing_prefix(tmp_path: Path) -> None:
    from scripts.streaming.data_root import safe_cleanup_scratch

    p = tmp_path / "noprefix_dir"
    p.mkdir()
    with pytest.raises(ValueError, match="missing prefix"):
        safe_cleanup_scratch(p, prefix_requirement="osm_")


def test_safe_cleanup_scratch_rejects_wrong_uid(tmp_path: Path) -> None:
    from scripts.streaming.data_root import safe_cleanup_scratch

    p = tmp_path / "osm_scratch" / "dir"
    p.mkdir(parents=True)
    fake_stat = mock.Mock(st_uid=99999)
    with (
        mock.patch("pathlib.Path.stat", return_value=fake_stat),
        pytest.raises(ValueError, match="owned by another UID"),
    ):
        safe_cleanup_scratch(p, prefix_requirement="osm_")
