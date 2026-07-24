"""Representative, deterministic row selection for labeling canaries."""

from __future__ import annotations

import hashlib

import pyarrow as pa


def select_canary_rows(table: pa.Table, row_limit: int) -> pa.Table:
    """Select a bounded representative subset while preserving input order."""

    if row_limit == 0 and not isinstance(row_limit, bool):
        return table
    if (
        isinstance(row_limit, bool)
        or not isinstance(row_limit, int)
        or row_limit < 1
        or row_limit >= table.num_rows
    ):
        raise ValueError("row limit must be zero or smaller than the input row count")
    required = {"sentence_id", "source", "language", "region"}
    if missing := required.difference(table.column_names):
        raise ValueError(f"canary input is missing required columns: {sorted(missing)}")
    if set(table["region"].to_pylist()) != {"afghanistan"}:
        raise ValueError("canary input must contain only Afghanistan rows")

    rows = table.select(["sentence_id", "source", "language"]).to_pylist()
    chosen: list[int] = []
    chosen_set: set[int] = set()
    chosen_languages: set[str] = set()

    def add(index: int) -> None:
        if index not in chosen_set and len(chosen) < row_limit:
            chosen.append(index)
            chosen_set.add(index)
            chosen_languages.add(str(rows[index]["language"]))

    for source in sorted({str(row["source"]) for row in rows}):
        candidates = [
            index for index, row in enumerate(rows) if str(row["source"]) == source
        ]
        fresh = [
            index
            for index in candidates
            if str(rows[index]["language"]) not in chosen_languages
        ]
        add((fresh or candidates)[0])

    language_first: dict[str, int] = {}
    for index, row in enumerate(rows):
        language_first.setdefault(str(row["language"]), index)
    for language in sorted(
        language_first,
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    ):
        add(language_first[language])

    for index in sorted(
        range(table.num_rows),
        key=lambda value: hashlib.sha256(
            str(rows[value]["sentence_id"]).encode()
        ).hexdigest(),
    ):
        add(index)

    return table.take(pa.array(sorted(chosen), type=pa.int64()))


__all__ = ["select_canary_rows"]
