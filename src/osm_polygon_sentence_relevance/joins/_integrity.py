"""Generic join-key and referential-integrity validation.

These helpers are shared by the Wikipedia and Wikivoyage joins and raise
:class:`~osm_polygon_sentence_relevance.errors.JoinIntegrityError` on any
violation.  They do not depend on either source's specific algorithm.
"""

from __future__ import annotations

import pyarrow as pa

from osm_polygon_sentence_relevance.errors import JoinIntegrityError


def _check_non_empty(
    table: pa.Table,
    column: str,
    source: str,
    table_name: str,
) -> None:
    """Raise JoinIntegrityError if *column* has nulls or empty strings."""
    col = table.column(column)
    null_count = col.null_count
    if null_count > 0:
        raise JoinIntegrityError(
            source,
            table_name,
            column,
            "contains nulls",
            ["<null>"] * min(null_count, 5),
        )
    values = col.to_pylist()
    empties = [v for v in values if v == ""]
    if empties:
        raise JoinIntegrityError(
            source,
            table_name,
            column,
            "contains empty strings",
            empties[:5],
        )


def _check_unique(
    table: pa.Table,
    column: str,
    source: str,
    table_name: str,
) -> None:
    """Raise JoinIntegrityError if *column* has duplicate values."""
    values = table.column(column).to_pylist()
    seen: set = set()
    dupes: list[str] = []
    for v in values:
        if v is not None:
            if v in seen and v not in dupes:
                dupes.append(str(v))
            seen.add(v)
    if dupes:
        raise JoinIntegrityError(
            source,
            table_name,
            column,
            "contains duplicates",
            dupes[:5],
        )


def _check_unique_pairs(
    table: pa.Table,
    col_a: str,
    col_b: str,
    source: str,
    table_name: str,
) -> None:
    """Raise JoinIntegrityError if (col_a, col_b) pairs are not unique."""
    a_vals = table.column(col_a).to_pylist()
    b_vals = table.column(col_b).to_pylist()
    seen: set[tuple] = set()
    dupes: list[str] = []
    for a, b in zip(a_vals, b_vals, strict=True):
        pair = (a, b)
        if pair in seen:
            rep = f"({a!r}, {b!r})"
            if rep not in dupes:
                dupes.append(rep)
        seen.add(pair)
    if dupes:
        raise JoinIntegrityError(
            source,
            table_name,
            f"{col_a}, {col_b}",
            "contains duplicate pairs",
            dupes[:5],
        )


def _check_section_index(
    table: pa.Table,
    column: str,
    source: str,
    table_name: str,
) -> None:
    """Raise JoinIntegrityError if any value in *column* is null or negative."""
    col = table.column(column)
    null_count = col.null_count
    if null_count > 0:
        raise JoinIntegrityError(
            source,
            table_name,
            column,
            "contains nulls",
            ["<null>"] * min(null_count, 5),
        )
    values = col.to_pylist()
    negatives = [str(v) for v in values if v is not None and v < 0]
    if negatives:
        raise JoinIntegrityError(
            source,
            table_name,
            column,
            "contains negative values",
            negatives[:5],
        )


def _check_no_orphans(
    child_table: pa.Table,
    child_key: str,
    parent_table: pa.Table,
    parent_key: str,
    source: str,
    child_table_name: str,
    context: str,
) -> None:
    """Raise JoinIntegrityError if child has keys absent from parent.

    Uses string representation to avoid sorting comparison errors.
    """
    child_vals = {
        str(v) for v in child_table.column(child_key).to_pylist() if v is not None
    }
    parent_vals = {
        str(v) for v in parent_table.column(parent_key).to_pylist() if v is not None
    }
    orphans = sorted(child_vals - parent_vals)
    if orphans:
        raise JoinIntegrityError(
            source,
            child_table_name,
            child_key,
            f"orphan keys not found in {context}",
            orphans[:5],
        )
