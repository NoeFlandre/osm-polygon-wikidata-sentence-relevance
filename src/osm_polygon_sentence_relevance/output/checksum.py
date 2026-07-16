"""Streaming SHA-256 checksumming for exported files."""

from __future__ import annotations

from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Return the lowercase hex SHA-256 digest of *path* read in chunks."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()
