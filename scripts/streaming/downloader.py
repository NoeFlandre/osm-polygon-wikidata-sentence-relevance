"""Per-file Hugging Face downloader for the per-shard streaming workflow.

This is a thin wrapper over ``huggingface_hub.hf_hub_download`` plus a
SHA-256 verification step. It deliberately does NOT hand-roll HTTP
range requests; ``hf_hub_download`` already supports resumable
downloads through its internal cache. We add:

* explicit SHA-256 verification of the downloaded bytes
* an atomic cleanup of the partial file on any failure
* a structured ``DownloadReceipt`` carrying resolved repo commit,
  blob identity / etag, size, path and local SHA-256.

The ``huggingface_hub`` dependency is imported lazily so that
importing this module as part of the package surface does not break
the ``no_optional_deps`` smoke tests.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """Raised when a per-file download fails for any reason.

    ``partial_path`` is the (now-removed) on-disk partial file when
    one existed; ``None`` otherwise.
    """

    def __init__(self, message: str, *, partial_path: Path | None) -> None:
        super().__init__(message)
        self.partial_path = partial_path


@dataclass(frozen=True, slots=True)
class DownloadReceipt:
    """Audit record for one downloaded file.

    Mirrors the download-receipt schema recorded alongside the
    upstream file (the same provenance facts are persisted by the
    driver into the checkpoint manifest).
    """

    path: Path
    repo_id: str
    path_in_repo: str
    hub_commit_hash: str
    hub_etag: str | None
    hub_url: str
    size: int | None
    local_sha256: str
    expected_sha256: str | None


def _phys(path: Path) -> Path:
    return Path(os.path.realpath(str(path)))


def _sha256_file(path: Path) -> str:
    """Compute the lowercase SHA-256 hex digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def _import_huggingface_hub():
    """Lazy ``huggingface_hub`` import so importing this module does
    not pollute ``sys.modules`` for tests that assert the package
    loads without optional dependencies."""
    try:
        import huggingface_hub
        from huggingface_hub import get_hf_file_metadata
    except ImportError as exc:  # pragma: no cover
        raise DownloadError(
            "huggingface_hub is required for per-file download; "
            "install it via ``pip install huggingface_hub``",
            partial_path=None,
        ) from exc
    return huggingface_hub, get_hf_file_metadata


class PerFileHubDownloader:
    """Fetch a single file from the Hub with SHA-256 verification."""

    def __init__(
        self,
        *,
        repo_id: str,
        resolved_revision: str,
        target_dir: Path,
        hub_api: Any,
    ) -> None:
        if not isinstance(repo_id, str) or not repo_id.strip() or "/" not in repo_id:
            raise ValueError("repo_id must be a non-blank owner/name string")
        if (
            not isinstance(resolved_revision, str)
            or len(resolved_revision) != 40
            or not all(c in "0123456789abcdef" for c in resolved_revision.lower())
        ):
            raise ValueError(
                "resolved_revision must be a 40-character lowercase hex commit SHA"
            )
        if not isinstance(target_dir, Path):
            raise TypeError("target_dir must be a Path")
        if hub_api is None:
            raise ValueError("hub_api must be supplied (typically HfApi())")
        self.repo_id = repo_id
        self.resolved_revision = resolved_revision.lower()
        self.target_dir = _phys(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.hub_api = hub_api

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        path_in_repo: str,
        *,
        expected_sha256: str | None = None,
        url: str | None = None,
    ) -> DownloadReceipt:
        """Download one file with a fresh fetch, verify SHA-256, and
        produce a ``DownloadReceipt``.

        ``expected_sha256``, if supplied, must be a 64-character
        lowercase hex string. The downloaded bytes are hashed and
        compared; on mismatch the partial file is removed and a
        ``DownloadError`` is raised.

        ``url`` is optional and used only for ``get_hf_file_metadata``
        if supplied; otherwise the URL is synthesised from the
        ``repo_id`` and ``path_in_repo`` via the Hub resolution.
        """
        if not isinstance(path_in_repo, str) or not path_in_repo.strip():
            raise ValueError("path_in_repo must be a non-blank string")
        if path_in_repo.startswith("/"):
            raise ValueError("path_in_repo must not start with '/'")
        if expected_sha256 is not None and (
            len(expected_sha256) != 64
            or not all(c in "0123456789abcdef" for c in expected_sha256.lower())
        ):
            raise ValueError(
                "expected_sha256 must be a 64-character lowercase hex string"
            )

        local_path = self.target_dir / path_in_repo
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if local_path.exists() or local_path.is_symlink():
                local_path.unlink()
        except OSError:
            pass

        # Lazy imports: keep huggingface_hub out of sys.modules for
        # tests that import this module without invoking ``download``.
        huggingface_hub, get_hf_file_metadata = _import_huggingface_hub()

        metadata_url = (
            url
            if url is not None
            else _synthesized_url(self.repo_id, self.resolved_revision, path_in_repo)
        )
        try:
            meta = get_hf_file_metadata(metadata_url)
        except Exception as exc:
            raise DownloadError(
                f"get_hf_file_metadata failed for {path_in_repo!r}: {exc}",
                partial_path=None,
            ) from exc

        hub_commit_hash: str = (
            getattr(meta, "commit_hash", None) or self.resolved_revision
        )
        if not isinstance(hub_commit_hash, str) or len(hub_commit_hash) != 40:
            hub_commit_hash = self.resolved_revision
        hub_commit_hash = hub_commit_hash.lower()

        hub_etag = getattr(meta, "etag", None)
        hub_url = getattr(meta, "location", None) or metadata_url
        hub_size = getattr(meta, "size", None)

        try:
            local_path_str = huggingface_hub.hf_hub_download(
                repo_id=self.repo_id,
                filename=path_in_repo,
                revision=self.resolved_revision,
                repo_type="dataset",
                local_dir=str(self.target_dir),
                force_download=True,
            )
        except Exception as exc:
            self._safe_unlink(local_path)
            raise DownloadError(
                f"hf_hub_download failed for {path_in_repo!r}: {exc}",
                partial_path=local_path,
            ) from exc

        local_path = _phys(Path(local_path_str))
        try:
            if local_path.exists():
                computed = _sha256_file(local_path)
            else:
                self._safe_unlink(local_path)
                raise DownloadError(
                    f"downloaded bytes not found at {local_path!r}",
                    partial_path=local_path,
                )
        except OSError as exc:
            self._safe_unlink(local_path)
            raise DownloadError(
                f"sha256 read failed for {path_in_repo!r}: {exc}",
                partial_path=local_path,
            ) from exc

        expected = expected_sha256.lower() if isinstance(expected_sha256, str) else None
        if expected is not None and computed != expected:
            self._safe_unlink(local_path)
            raise DownloadError(
                f"sha256 mismatch for {path_in_repo!r}: "
                f"expected {expected!r}, got {computed!r}",
                partial_path=local_path,
            )

        return DownloadReceipt(
            path=_phys(local_path) if local_path.exists() else local_path,
            repo_id=self.repo_id,
            path_in_repo=path_in_repo,
            hub_commit_hash=hub_commit_hash,
            hub_etag=hub_etag,
            hub_url=hub_url,
            size=hub_size,
            local_sha256=computed,
            expected_sha256=expected,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            if path.exists() or path.is_symlink():
                path.unlink()
        except OSError:
            log.warning("could not unlink partial download at %s", path)


def _synthesized_url(repo_id: str, revision: str, path_in_repo: str) -> str:
    """Synthesize a Hub file URL without private library parameters.

    Only used as a fallback when no URL is provided. Uses the public
    ``huggingface.co`` endpoint.
    """
    from urllib.parse import quote

    return (
        f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/"
        f"{quote(path_in_repo, safe='/')}"
    )
