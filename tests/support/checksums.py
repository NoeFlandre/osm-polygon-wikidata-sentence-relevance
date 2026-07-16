"""Checksum helpers for exporter/atomic-install tests."""

from __future__ import annotations

import hashlib
from pathlib import Path


def get_checksum(file_path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of *file_path* (streamed)."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()
