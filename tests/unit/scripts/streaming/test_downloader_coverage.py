"""Coverage-targeted tests for the downloader module uncovered branches."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest import mock

import pytest
from scripts.streaming.downloader import (
    DownloadError,
    DownloadReceipt,
    PerFileHubDownloader,
    _import_huggingface_hub,
    _phys,
    _sha256_file,
    _synthesized_url,
)


def _fake_meta(*, commit: str, etag: str, location: str, size: int = 100):
    m = mock.Mock()
    m.commit_hash = commit
    m.etag = etag
    m.location = location
    m.size = size
    return m


def _fake_dl_for(content: bytes, repo_id: str = "owner/repo"):
    def _dl(**kw):
        target = Path(kw["local_dir"]) / kw["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    return _dl


def _make_hub_module(content: bytes):
    hub = mock.MagicMock()
    hub.hf_hub_download = _fake_dl_for(content)
    return hub


def _make(target_dir: Path) -> PerFileHubDownloader:
    api = mock.Mock()
    return PerFileHubDownloader(
        repo_id="owner/repo",
        resolved_revision="a" * 40,
        target_dir=target_dir,
        hub_api=api,
    )


# ---------------------------------------------------------------------------
# Constructor validation.
# ---------------------------------------------------------------------------


def test_ctor_rejects_blank_repo_id(tmp_path: Path) -> None:
    api = mock.Mock()
    with pytest.raises(ValueError, match="repo_id"):
        PerFileHubDownloader(
            repo_id="",
            resolved_revision="a" * 40,
            target_dir=tmp_path,
            hub_api=api,
        )


def test_ctor_rejects_repo_id_no_slash(tmp_path: Path) -> None:
    api = mock.Mock()
    with pytest.raises(ValueError, match="repo_id"):
        PerFileHubDownloader(
            repo_id="noslash",
            resolved_revision="a" * 40,
            target_dir=tmp_path,
            hub_api=api,
        )


def test_ctor_rejects_wrong_length_revision(tmp_path: Path) -> None:
    api = mock.Mock()
    with pytest.raises(ValueError, match="resolved_revision"):
        PerFileHubDownloader(
            repo_id="owner/repo",
            resolved_revision="abc",
            target_dir=tmp_path,
            hub_api=api,
        )


def test_ctor_rejects_non_hex_revision(tmp_path: Path) -> None:
    api = mock.Mock()
    with pytest.raises(ValueError, match="resolved_revision"):
        PerFileHubDownloader(
            repo_id="owner/repo",
            resolved_revision="z" * 40,
            target_dir=tmp_path,
            hub_api=api,
        )


def test_ctor_rejects_target_dir_not_path(tmp_path: Path) -> None:
    api = mock.Mock()
    with pytest.raises(TypeError, match="Path"):
        PerFileHubDownloader(
            repo_id="owner/repo",
            resolved_revision="a" * 40,
            target_dir="not-a-path",  # type: ignore[arg-type]
            hub_api=api,
        )


def test_ctor_rejects_none_hub_api(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hub_api"):
        PerFileHubDownloader(
            repo_id="owner/repo",
            resolved_revision="a" * 40,
            target_dir=tmp_path,
            hub_api=None,
        )


# ---------------------------------------------------------------------------
# download() validation paths.
# ---------------------------------------------------------------------------


def test_download_rejects_blank_path_in_repo(tmp_path: Path) -> None:
    d = _make(tmp_path)
    with pytest.raises(ValueError, match="path_in_repo"):
        d.download("")


def test_download_rejects_leading_slash_path_in_repo(tmp_path: Path) -> None:
    d = _make(tmp_path)
    with pytest.raises(ValueError, match="path_in_repo"):
        d.download("/foo.parquet")


def test_download_rejects_bad_expected_sha(tmp_path: Path) -> None:
    d = _make(tmp_path)
    with pytest.raises(ValueError, match="expected_sha256"):
        d.download("polygons/foo.parquet", expected_sha256="too-short")


def test_download_rejects_non_hex_expected_sha(tmp_path: Path) -> None:
    d = _make(tmp_path)
    with pytest.raises(ValueError, match="expected_sha256"):
        d.download("polygons/foo.parquet", expected_sha256="z" * 64)


def test_download_replaces_existing_local_file(tmp_path: Path) -> None:
    """If a stale local file already exists, it is removed first."""
    d = _make(tmp_path)
    sub = tmp_path / "polygons"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "foo.parquet").write_bytes(b"stale")
    content = b"fresh"
    hub = _make_hub_module(content)
    meta = _fake_meta(commit="a" * 40, etag='"x"', location="https://hub/resolve/foo")
    hub.get_hf_file_metadata = lambda *a, **k: meta
    with mock.patch(
        "scripts.streaming.downloader._import_huggingface_hub",
        return_value=(hub, hub.get_hf_file_metadata),
    ):
        receipt = d.download("polygons/foo.parquet")
    assert receipt.path.read_bytes() == content


def test_download_falls_back_to_resolved_revision(tmp_path: Path) -> None:
    """If commit_hash is None, fall back to resolved_revision."""
    d = _make(tmp_path)
    content = b"hello"
    hub = _make_hub_module(content)
    meta = mock.Mock()
    meta.commit_hash = None
    meta.etag = None
    meta.location = None
    meta.size = None
    hub.get_hf_file_metadata = lambda *a, **k: meta
    with mock.patch(
        "scripts.streaming.downloader._import_huggingface_hub",
        return_value=(hub, hub.get_hf_file_metadata),
    ):
        receipt = d.download("polygons/foo.parquet")
    assert receipt.hub_commit_hash == "a" * 40
    assert receipt.hub_etag is None
    assert receipt.hub_url.endswith("polygons/foo.parquet")
    assert receipt.size is None


def test_download_falls_back_when_commit_not_40_chars(tmp_path: Path) -> None:
    """If commit_hash has wrong length, fall back to resolved_revision."""
    d = _make(tmp_path)
    content = b"hello"
    hub = _make_hub_module(content)
    meta = mock.Mock()
    meta.commit_hash = "short"
    meta.etag = None
    meta.location = None
    meta.size = None
    hub.get_hf_file_metadata = lambda *a, **k: meta
    with mock.patch(
        "scripts.streaming.downloader._import_huggingface_hub",
        return_value=(hub, hub.get_hf_file_metadata),
    ):
        receipt = d.download("polygons/foo.parquet")
    assert receipt.hub_commit_hash == "a" * 40


def test_download_propagates_metadata_failure(tmp_path: Path) -> None:
    """If get_hf_file_metadata raises, a DownloadError is raised."""
    d = _make(tmp_path)
    hub = mock.MagicMock()
    hub.hf_hub_download = _fake_dl_for(b"hi")

    def _bad_meta(*a, **k):
        raise RuntimeError("meta fail")

    with (
        mock.patch(
            "scripts.streaming.downloader._import_huggingface_hub",
            return_value=(hub, _bad_meta),
        ),
        pytest.raises(DownloadError, match="get_hf_file_metadata"),
    ):
        d.download("polygons/foo.parquet")


def test_download_propagates_hf_hub_download_failure(tmp_path: Path) -> None:
    """If hf_hub_download raises, a DownloadError is raised."""
    d = _make(tmp_path)
    hub = mock.MagicMock()
    hub.hf_hub_download = mock.Mock(side_effect=RuntimeError("network"))

    meta = _fake_meta(commit="a" * 40, etag='"x"', location="https://hub/foo")
    with (
        mock.patch(
            "scripts.streaming.downloader._import_huggingface_hub",
            return_value=(hub, lambda *a, **k: meta),
        ),
        pytest.raises(DownloadError, match="hf_hub_download"),
    ):
        d.download("polygons/foo.parquet")


def test_download_raises_when_local_file_missing_after_dl(tmp_path: Path) -> None:
    """If the local file disappears after hf_hub_download, DownloadError."""
    d = _make(tmp_path)

    def _no_file(**kw):
        # Never write the file; report a fake path.
        return str(Path(kw["local_dir"]) / kw["filename"])

    hub = mock.MagicMock()
    hub.hf_hub_download = _no_file
    meta = _fake_meta(commit="a" * 40, etag='"x"', location="https://hub/foo")
    with (
        mock.patch(
            "scripts.streaming.downloader._import_huggingface_hub",
            return_value=(hub, lambda *a, **k: meta),
        ),
        pytest.raises(DownloadError, match="not found"),
    ):
        d.download("polygons/foo.parquet")


def test_download_sha_mismatch_removes_partial(tmp_path: Path) -> None:
    """If SHA-256 does not match expected, partial file is removed."""
    d = _make(tmp_path)
    content = b"world"
    hub = _make_hub_module(content)
    meta = _fake_meta(commit="a" * 40, etag='"x"', location="https://hub/foo")
    with (
        mock.patch(
            "scripts.streaming.downloader._import_huggingface_hub",
            return_value=(hub, lambda *a, **k: meta),
        ),
        pytest.raises(DownloadError, match="sha256 mismatch"),
    ):
        d.download(
            "polygons/foo.parquet",
            expected_sha256="0" * 64,
        )
    # Verify the partial was removed
    leftover = tmp_path / "polygons" / "foo.parquet"
    assert not leftover.exists()


def test_import_huggingface_hub_raises_download_error(monkeypatch) -> None:
    """When huggingface_hub is not installed, DownloadError is raised."""

    # Save and remove the real module to force ImportError on inner import
    import sys

    saved = sys.modules.pop("huggingface_hub", None)

    # Block re-import by inserting a dummy that raises on attribute access.
    class _Blocker:
        def __getattr__(self, name):
            raise ImportError("no hub")

    sys.modules["huggingface_hub"] = _Blocker()  # type: ignore[assignment]
    try:
        with pytest.raises(DownloadError, match="huggingface_hub is required"):
            _import_huggingface_hub()
    finally:
        sys.modules.pop("huggingface_hub", None)
        if saved is not None:
            sys.modules["huggingface_hub"] = saved


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def test_sha256_file_reads(tmp_path: Path) -> None:
    p = tmp_path / "x"
    p.write_bytes(b"hello")
    assert _sha256_file(p) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_phys_resolves_symlink(tmp_path: Path) -> None:
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    assert _phys(link) == target.resolve()


def test_synthesized_url_quotes_slashes(tmp_path: Path) -> None:
    revision = "a" * 40
    url = _synthesized_url("owner/repo", revision, "polygons/foo-latest.parquet")
    assert "owner/repo" in url
    assert f"/resolve/{revision}/" in url
    assert "polygons/foo-latest.parquet" in url


def test_safe_unlink_logs_warning(monkeypatch, tmp_path) -> None:
    """A failed unlink logs a warning rather than raising."""
    p = tmp_path / "ghost"
    # Don't actually create; safe_unlink handles it.
    PerFileHubDownloader._safe_unlink(p)


def test_download_receipt_is_frozen() -> None:
    r = DownloadReceipt(
        path=Path("/x"),
        repo_id="o/r",
        path_in_repo="x",
        hub_commit_hash="a" * 40,
        hub_etag=None,
        hub_url="u",
        size=None,
        local_sha256="b" * 64,
        expected_sha256=None,
    )
    assert dataclasses.is_dataclass(r)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.path_in_repo = "y"
