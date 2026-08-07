"""Deterministic local review sample for the V2 binary labels."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import pyarrow as pa

from .v2_contracts import V2LogitRecord


def _section_title(value: object) -> str:
    if isinstance(value, list):
        return str(value[-1]) if value else "none"
    if isinstance(value, str):
        return value
    raise ValueError("section_path must be a list or string")


def write_v2_manual_eval(
    table: pa.Table,
    records: list[V2LogitRecord],
    path: Path,
    *,
    limit: int = 100,
) -> None:
    """Write a stable, editable JSONL sample without changing publication data."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("manual evaluation limit must be positive")
    required = {
        "sentence_id",
        "sentence_text_raw",
        "previous_sentence",
        "next_sentence",
        "page_title",
        "section_path",
    }
    if missing := sorted(required.difference(table.column_names)):
        raise ValueError(f"manual evaluation input is missing columns: {missing}")
    by_id: dict[str, V2LogitRecord] = {}
    for record in records:
        if record.sentence_id in by_id:
            raise ValueError("manual evaluation contains duplicate scores")
        by_id[record.sentence_id] = record
    rows = sorted(table.to_pylist(), key=lambda row: str(row["sentence_id"]))
    selected = rows[:limit]
    output: list[dict[str, Any]] = []
    for row in selected:
        sentence_id = row["sentence_id"]
        if not isinstance(sentence_id, str) or sentence_id not in by_id:
            raise ValueError("manual evaluation is missing a score")
        record = by_id[sentence_id]
        output.append(
            {
                "sentence_id": sentence_id,
                "page_title": str(row["page_title"]),
                "section_title": _section_title(row["section_path"]),
                "previous_sentence": row["previous_sentence"],
                "sentence_text": str(row["sentence_text_raw"]),
                "next_sentence": row["next_sentence"],
                "model_label": record.place_relevance,
                "yes_logprob": record.yes_logprob,
                "no_logprob": record.no_logprob,
                "logit_margin": record.logit_margin,
                "two_class_probability": record.two_class_probability,
                "human_label": "",
                "notes": "",
            }
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in output:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        temporary.unlink(missing_ok=True)
        raise


__all__ = ["write_v2_manual_eval"]
