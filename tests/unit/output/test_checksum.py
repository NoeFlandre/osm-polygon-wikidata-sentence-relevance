"""Contracts for streamed output checksums."""

from __future__ import annotations

import hashlib
from pathlib import Path

from osm_polygon_sentence_relevance.output import checksum


def test_sha256_file_reads_fixed_sized_chunks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reads: list[object] = []

    class FakeFile:
        def __enter__(self) -> FakeFile:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: object) -> bytes:
            reads.append(size)
            return b"abc" if len(reads) == 1 else b""

    def fake_open(path: object, mode: str) -> FakeFile:
        assert path == tmp_path / "payload.bin"
        assert mode == "rb"
        return FakeFile()

    monkeypatch.setattr("builtins.open", fake_open)

    assert checksum.sha256_file(tmp_path / "payload.bin") == hashlib.sha256(
        b"abc"
    ).hexdigest()
    assert reads == [65_536, 65_536]
