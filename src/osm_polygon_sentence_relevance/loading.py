"""Schema-validated local Parquet loading.

Reads a single Parquet file, validates its physical schema against the
Phase 1 contract, and optionally projects to a subset of columns.
Contains no join logic.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_sentence_relevance.schemas import SCHEMA_REGISTRY, validate_table_schema


def load_validated_table(
    table_name: str,
    path: Path,
    columns: tuple[str, ...] | None = None,
) -> pa.Table:
    """Read and validate a Parquet file against the schema contract.

    Parameters
    ----------
    table_name
        Logical table name (must exist in :data:`SCHEMA_REGISTRY`).
    path
        Path to the ``.parquet`` file.
    columns
        Optional column projection.  When provided, only these columns
        are read after validation.

    Returns
    -------
    pyarrow.Table

    Raises
    ------
    UnknownTableError
        If *table_name* is not in the schema registry.
    MissingColumnsError
        If the Parquet file is missing required contract columns.
    IncompatibleTypesError
        If column types don't match the contract.
    ValueError
        If *columns* contains names not present in the Parquet file.
    """
    # Read just the schema (metadata) first for validation.
    parquet_schema = pq.read_schema(path)
    pa_schema = pa.schema(parquet_schema)

    # Validate the *complete* physical schema before projection.
    validate_table_schema(table_name, pa_schema)

    # Validate projection columns exist.
    if columns is not None:
        file_columns = {f.name for f in pa_schema}
        unknown = sorted(set(columns) - file_columns)
        if unknown:
            raise ValueError(
                f"Unknown projection columns for {table_name!r}: {unknown}"
            )

    # Read with optional projection.
    return pq.read_table(path, columns=list(columns) if columns is not None else None)
