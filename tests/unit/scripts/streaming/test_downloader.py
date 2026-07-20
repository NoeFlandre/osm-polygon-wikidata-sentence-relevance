"""RED tests for the per-file Hub downloader.

Contract:

* Uses ``huggingface_hub.hf_hub_download`` for the actual byte fetch.
* Uses ``huggingface_hub.get_hf_file_metadata`` to retrieve Hub
  commit_hash, etag, location, size (no invented ``repo_info``
  parameters).
* Verifies the local file's SHA-256 after download and rejects on
  mismatch.
* Records resolved repo commit, blob identity/etag, size, path and
  local SHA-256 in a structured ``DownloadReceipt``.
* Handles interruption cleanly: a partial file under a controlled
  ``ResumeToken`` is atomically renamed before re-download when the
  Hub metadata is fetched fresh per file (we never re-hand-roll
  Range headers because ``hf_hub_download`` already exposes a
  resumable partial blob cache).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "streaming"
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from scripts.streaming.downloader import (  # noqa: E402
    DownloadError,
    DownloadReceipt,
    PerFileHubDownloader,
)

# ---------------------------------------------------------------------------
# Fixture: fake hub_api and fake downloader
# ---------------------------------------------------------------------------


def _fake_hf_file_metadata(url: str, **_: object) -> mock.Mock:
    """Return a HfFileMetadata-shaped mock for one URL."""
    m = mock.Mock()
    m.commit_hash = "abcdef0123456789abcdef0123456789abcdef01"
    m.etag = '"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"'
    m.location = url
    m.size = 11  # "hello world"
    m.xet_file_data = None
    return m


def _fake_hf_hub_download(
    *,
    repo_id: str,
    filename: str,
    local_dir: str | None = None,
    content: bytes = b"hello world",
    **_: object,
) -> str:
    """Minimal stand-in for ``hf_hub_download``.

    Writes ``content`` to ``filename`` under ``local_dir`` so the
    downloader's target_dir is honoured. The returned string matches
    what the real library returns.
    """
    from pathlib import Path as _P

    target = _P(local_dir or ".") / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return str(target)


# ---------------------------------------------------------------------------
# GREEN contract: a happy-path download produces a valid receipt.
# ---------------------------------------------------------------------------


def test_per_file_download_happy_path(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    downloader = PerFileHubDownloader(
        repo_id="owner/repo",
        resolved_revision="abcdef0123456789abcdef0123456789abcdef01",
        target_dir=work,
        hub_api=mock.Mock(),
    )
    with mock.patch(
        "scripts.streaming.downloader._import_huggingface_hub",
        return_value=(
            mock.MagicMock(hf_hub_download=lambda **kw: _fake_hf_hub_download(**kw)),
            _fake_hf_file_metadata,
        ),
    ):
        receipt = downloader.download("polygons/foo-latest.parquet")

    assert isinstance(receipt, DownloadReceipt)
    assert receipt.path == work / "polygons" / "foo-latest.parquet"
    assert receipt.expected_sha256 is None or isinstance(receipt.expected_sha256, str)
    assert receipt.local_sha256 is not None
    assert receipt.local_sha256 != ""
    assert receipt.hub_commit_hash == "abcdef0123456789abcdef0123456789abcdef01"
    assert receipt.hub_etag.startswith('"0')
    # hello world is 11 bytes.
    assert receipt.size == 11


# ---------------------------------------------------------------------------
# GREEN contract: SHA-256 mismatch removes the partial file and raises.
# ---------------------------------------------------------------------------


def test_per_file_download_sha_mismatch_removes_partial(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    downloader = PerFileHubDownloader(
        repo_id="owner/repo",
        resolved_revision="abcdef0123456789abcdef0123456789abcdef01",
        target_dir=work,
        hub_api=mock.Mock(),
    )
    # Give the fake a distinct "expected" content that the caller will
    # then verify against a different hash.
    with (
        mock.patch(
            "scripts.streaming.downloader._import_huggingface_hub",
            return_value=(
                mock.MagicMock(
                    hf_hub_download=lambda **kw: _fake_hf_hub_download(
                        content=b"corrupted bytes", **kw
                    )
                ),
                _fake_hf_file_metadata,
            ),
        ),
        pytest.raises(DownloadError),
    ):
        downloader.download(
            "polygons/foo-latest.parquet",
            expected_sha256=(
                # SHA-256 of "hello world"
                "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
            ),
        )

    target = work / "polygons" / "foo-latest.parquet"
    assert not target.exists(), "partial file must be removed on SHA mismatch"


# ---------------------------------------------------------------------------
# GREEN contract: explicit expected_sha256 is verified after download.
# ---------------------------------------------------------------------------


def test_per_file_download_with_expected_sha256(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    downloader = PerFileHubDownloader(
        repo_id="owner/repo",
        resolved_revision="abcdef0123456789abcdef0123456789abcdef01",
        target_dir=work,
        hub_api=mock.Mock(),
    )
    sha = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    with mock.patch(
        "scripts.streaming.downloader._import_huggingface_hub",
        return_value=(
            mock.MagicMock(hf_hub_download=lambda **kw: _fake_hf_hub_download(**kw)),
            _fake_hf_file_metadata,
        ),
    ):
        receipt = downloader.download(
            "polygons/foo-latest.parquet", expected_sha256=sha
        )
    assert receipt.local_sha256 == sha
    assert receipt.expected_sha256 == sha


# ---------------------------------------------------------------------------
# RED contract: unknown ``commit_hash`` from Hub metadata is allowed
# (the metadata API may return None) but the receipt records it as the
# resolved revision instead so the download is still audit-traceable.
# ---------------------------------------------------------------------------


def test_per_file_download_records_resolved_revision_when_commit_hash_missing(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    downloader = PerFileHubDownloader(
        repo_id="owner/repo",
        resolved_revision="abcdef0123456789abcdef0123456789abcdef01",
        target_dir=work,
        hub_api=mock.Mock(),
    )

    def no_commit(*_a: object, **_kw: object) -> mock.Mock:
        m = mock.Mock()
        m.commit_hash = None
        m.etag = '"deadbeef"'
        m.location = "https://example/foo"
        m.size = 11
        m.xet_file_data = None
        return m

    with mock.patch(
        "scripts.streaming.downloader._import_huggingface_hub",
        return_value=(
            mock.MagicMock(hf_hub_download=lambda **kw: _fake_hf_hub_download(**kw)),
            no_commit,
        ),
    ):
        receipt = downloader.download("polygons/foo-latest.parquet")
    # The receipt must NOT invent a commit hash; it falls back to the
    # resolved_revision recorded by the caller.
    assert receipt.hub_commit_hash == "abcdef0123456789abcdef0123456789abcdef01"


def test_metadata_lookup_is_pinned_to_resolved_revision(tmp_path: Path) -> None:
    revision = "abcdef0123456789abcdef0123456789abcdef01"
    downloader = PerFileHubDownloader(
        repo_id="owner/repo",
        resolved_revision=revision,
        target_dir=tmp_path,
        hub_api=mock.Mock(),
    )
    observed: list[str] = []

    def capture(url: str, **_: object) -> mock.Mock:
        observed.append(url)
        return _fake_hf_file_metadata(url)

    with mock.patch(
        "scripts.streaming.downloader._import_huggingface_hub",
        return_value=(
            mock.MagicMock(hf_hub_download=lambda **kw: _fake_hf_hub_download(**kw)),
            capture,
        ),
    ):
        downloader.download("polygons/foo-latest.parquet")

    assert observed == [
        f"https://huggingface.co/datasets/owner/repo/resolve/{revision}/"
        "polygons/foo-latest.parquet"
    ]
