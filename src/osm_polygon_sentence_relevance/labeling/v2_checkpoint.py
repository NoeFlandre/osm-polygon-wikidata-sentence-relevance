"""Strict durable checkpoints for V2 score-based labels."""

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

from .contracts import RunIdentity
from .v2_contracts import V2LogitRecord

V2_LOGIT_SCHEMA = pa.schema(
    [
        pa.field("sentence_id", pa.string(), nullable=False),
        pa.field("place_relevance", pa.string(), nullable=False),
        pa.field("yes_logprob", pa.float64(), nullable=False),
        pa.field("no_logprob", pa.float64(), nullable=False),
        pa.field("logit_margin", pa.float64(), nullable=False),
        pa.field("two_class_probability", pa.float64(), nullable=False),
    ]
)
_ENTRY = re.compile(r"batch-(\d{6})\.(parquet|json)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path: Path, data: bytes) -> None:
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        temporary.unlink(missing_ok=True)
        raise


class V2CheckpointStore:
    """Atomically persist and validate V2 score batches."""

    def __init__(self, root: Path, identity: RunIdentity) -> None:
        self.root = Path(root)
        self.identity = identity
        self.directory = self.root / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.mkdir(exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        os.chmod(self.directory, 0o700)

    def _indexes(self) -> list[int]:
        entries: dict[int, set[str]] = {}
        for path in self.directory.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("unexpected V2 checkpoint entry")
            match = _ENTRY.fullmatch(path.name)
            if match is None:
                raise ValueError("unexpected V2 checkpoint entry")
            entries.setdefault(int(match.group(1)), set()).add(match.group(2))
        if any(kinds != {"parquet", "json"} for kinds in entries.values()):
            raise ValueError("V2 checkpoint batch is incomplete")
        return sorted(entries)

    def batch_indexes(self) -> list[int]:
        """Return validated checkpoint indexes in deterministic order."""

        return self._indexes()

    def write_batch(self, index: int, records: list[V2LogitRecord]) -> None:
        if index < 0 or not records:
            raise ValueError("V2 checkpoint batch must be non-empty")
        record_ids = [record.sentence_id for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("duplicate sentence ID in V2 checkpoint batch")
        parquet = self.directory / f"batch-{index:06d}.parquet"
        metadata = self.directory / f"batch-{index:06d}.json"
        if parquet.exists() or metadata.exists():
            raise ValueError("V2 checkpoint batch already exists")
        table = pa.Table.from_pylist(
            [
                {
                    "sentence_id": record.sentence_id,
                    "place_relevance": record.place_relevance,
                    "yes_logprob": record.yes_logprob,
                    "no_logprob": record.no_logprob,
                    "logit_margin": record.logit_margin,
                    "two_class_probability": record.two_class_probability,
                }
                for record in records
            ],
            schema=V2_LOGIT_SCHEMA,
        )
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, compression="zstd")
        _atomic(parquet, sink.getvalue().to_pybytes())
        try:
            _atomic(
                metadata,
                (
                    json.dumps(
                        {
                            "schema_version": 1,
                            "identity": self.identity.checkpoint_dict(),
                            "row_count": len(records),
                            "parquet_sha256": _sha256(parquet),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode(),
            )
        except BaseException:
            parquet.unlink(missing_ok=True)
            raise

    def load_all(self) -> list[V2LogitRecord]:
        result: list[V2LogitRecord] = []
        seen: set[str] = set()
        for index in self._indexes():
            parquet = self.directory / f"batch-{index:06d}.parquet"
            metadata_path = self.directory / f"batch-{index:06d}.json"
            try:
                metadata: Any = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("V2 checkpoint metadata is invalid") from exc
            if not isinstance(metadata, dict):
                raise ValueError("V2 checkpoint metadata is invalid")
            if metadata.get("schema_version") != 1:
                raise ValueError("V2 checkpoint metadata schema is invalid")
            if metadata.get("identity") != self.identity.checkpoint_dict():
                raise ValueError("V2 checkpoint identity mismatch")
            if metadata.get("parquet_sha256") != _sha256(parquet):
                raise ValueError("V2 checkpoint SHA-256 mismatch")
            table = pq.read_table(parquet)
            if not table.schema.equals(V2_LOGIT_SCHEMA):
                raise ValueError("V2 checkpoint schema mismatch")
            row_count = metadata.get("row_count")
            if isinstance(row_count, bool) or not isinstance(row_count, int):
                raise ValueError("V2 checkpoint row count is invalid")
            if table.num_rows != row_count:
                raise ValueError("V2 checkpoint row count mismatch")
            for row in table.to_pylist():
                if row["sentence_id"] in seen:
                    raise ValueError("duplicate sentence ID in V2 checkpoints")
                seen.add(row["sentence_id"])
                record = V2LogitRecord(
                    sentence_id=row["sentence_id"],
                    place_relevance=row["place_relevance"],
                    yes_logprob=row["yes_logprob"],
                    no_logprob=row["no_logprob"],
                )
                if (
                    record.logit_margin != row["logit_margin"]
                    or record.two_class_probability != row["two_class_probability"]
                ):
                    raise ValueError("V2 checkpoint derived score mismatch")
                result.append(record)
        return result

    def completed_ids(self) -> set[str]:
        return {record.sentence_id for record in self.load_all()}


__all__ = ["V2CheckpointStore", "V2_LOGIT_SCHEMA"]
