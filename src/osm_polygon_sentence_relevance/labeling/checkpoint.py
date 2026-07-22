"""Atomic, identity-bound labeling checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import LabelRecord, LabelValue, RunIdentity


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be trusted."""


LABEL_SCHEMA = pa.schema(
    [
        pa.field("sentence_id", pa.string(), nullable=False),
        pa.field("landuse_relevance", pa.string(), nullable=False),
        pa.field("polygon_relevance", pa.string(), nullable=False),
        pa.field("landuse_reason", pa.string(), nullable=False),
        pa.field("polygon_reason", pa.string(), nullable=False),
        pa.field("evidence", pa.string(), nullable=False),
    ]
)
_ENTRY = re.compile(r"batch-(\d{6})\.(parquet|json)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        tmp.unlink(missing_ok=True)
        raise


class CheckpointStore:
    """Persist and validate bounded result batches."""

    def __init__(self, root: Path, identity: RunIdentity) -> None:
        self.root = Path(root)
        self.identity = identity
        self.directory = self.root / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.mkdir(exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        os.chmod(self.directory, 0o700)

    def write_batch(self, index: int, records: list[LabelRecord]) -> None:
        """Atomically publish one Parquet batch and its metadata."""

        if index < 0 or not records:
            raise CheckpointError("checkpoint batch must be non-empty")
        stem = f"batch-{index:06d}"
        parquet = self.directory / f"{stem}.parquet"
        metadata = self.directory / f"{stem}.json"
        if parquet.exists() or metadata.exists():
            raise CheckpointError("checkpoint batch already exists")
        table = pa.Table.from_pylist(
            [
                {
                    "sentence_id": r.sentence_id,
                    "landuse_relevance": r.landuse_relevance.value,
                    "polygon_relevance": r.polygon_relevance.value,
                    "landuse_reason": r.landuse_reason,
                    "polygon_reason": r.polygon_reason,
                    "evidence": r.evidence,
                }
                for r in records
            ],
            schema=LABEL_SCHEMA,
        )
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, compression="zstd")
        _atomic_bytes(parquet, sink.getvalue().to_pybytes())
        payload = {
            "schema_version": 1,
            "identity": self.identity.to_dict(),
            "row_count": len(records),
            "parquet_sha256": _sha256(parquet),
        }
        try:
            _atomic_bytes(
                metadata,
                (
                    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode(),
            )
        except BaseException:
            parquet.unlink(missing_ok=True)
            raise

    def _batch_indexes(self) -> list[int]:
        indexes: dict[int, set[str]] = {}
        for path in self.directory.iterdir():
            if path.is_symlink() or not path.is_file():
                raise CheckpointError("unexpected checkpoint entry")
            match = _ENTRY.fullmatch(path.name)
            if match is None:
                raise CheckpointError("unexpected checkpoint entry")
            indexes.setdefault(int(match.group(1)), set()).add(match.group(2))
        if any(kinds != {"parquet", "json"} for kinds in indexes.values()):
            raise CheckpointError("checkpoint batch is incomplete")
        return sorted(indexes)

    def load_all(self) -> list[LabelRecord]:
        """Load all batches after strict identity, hash, and schema checks."""

        result: list[LabelRecord] = []
        seen: set[str] = set()
        for index in self._batch_indexes():
            stem = f"batch-{index:06d}"
            parquet = self.directory / f"{stem}.parquet"
            metadata_path = self.directory / f"{stem}.json"
            try:
                metadata: Any = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise CheckpointError("checkpoint metadata is invalid") from exc
            if metadata.get("identity") != self.identity.to_dict():
                raise CheckpointError("checkpoint identity mismatch")
            if metadata.get("parquet_sha256") != _sha256(parquet):
                raise CheckpointError("checkpoint Parquet SHA-256 mismatch")
            try:
                table = pq.read_table(parquet)
            except (OSError, pa.ArrowException) as exc:
                raise CheckpointError("checkpoint Parquet is invalid") from exc
            if not table.schema.equals(LABEL_SCHEMA):
                raise CheckpointError("checkpoint schema mismatch")
            if table.num_rows != metadata.get("row_count"):
                raise CheckpointError("checkpoint row count mismatch")
            for row in table.to_pylist():
                sentence_id = row["sentence_id"]
                if sentence_id in seen:
                    raise CheckpointError("duplicate sentence ID in checkpoints")
                seen.add(sentence_id)
                result.append(
                    LabelRecord(
                        sentence_id=sentence_id,
                        landuse_relevance=LabelValue(row["landuse_relevance"]),
                        polygon_relevance=LabelValue(row["polygon_relevance"]),
                        landuse_reason=row["landuse_reason"],
                        polygon_reason=row["polygon_reason"],
                        evidence=row["evidence"],
                    )
                )
        return result

    def completed_ids(self) -> set[str]:
        """Return IDs from all validated checkpoints."""

        return {record.sentence_id for record in self.load_all()}

    def write_progress(
        self, *, completed: int, total: int, elapsed_seconds: float
    ) -> None:
        """Atomically update factual progress and ETA."""

        rate = completed / elapsed_seconds if elapsed_seconds > 0 else 0.0
        eta = (total - completed) / rate if rate > 0 else None
        payload = {
            "identity": self.identity.to_dict(),
            "completed": completed,
            "remaining": total - completed,
            "elapsed_seconds": elapsed_seconds,
            "rows_per_second": rate,
            "eta_seconds": eta,
        }
        _atomic_bytes(
            self.root / "progress.json",
            (
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode(),
        )

    def write_timing(self, payload: dict[str, float | int | bool]) -> None:
        """Atomically persist the final timing summary."""

        _atomic_bytes(
            self.root / "timing.json",
            (
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode(),
        )


__all__ = ["LABEL_SCHEMA", "CheckpointError", "CheckpointStore"]
