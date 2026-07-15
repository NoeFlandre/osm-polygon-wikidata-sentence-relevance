"""Custom exceptions for the pipeline.

Kept deliberately narrow: configuration problems and schema-contract
violations each get their own hierarchy so callers can catch at the right
granularity.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------
class ConfigurationError(Exception):
    """Raised when pipeline settings are invalid."""


# ---------------------------------------------------------------------------
# Schema-contract errors
# ---------------------------------------------------------------------------
class SchemaContractError(Exception):
    """Base class for schema-validation failures."""


class UnknownTableError(SchemaContractError):
    """Raised when a table name is not found in the schema registry."""

    def __init__(self, table_name: str) -> None:
        self.table_name = table_name
        super().__init__(f"Unknown table name: {table_name!r}")


class MissingColumnsError(SchemaContractError):
    """Raised when required columns are absent from an actual schema."""

    def __init__(self, table_name: str, missing: list[str]) -> None:
        self.table_name = table_name
        self.missing = missing
        super().__init__(
            f"Table {table_name!r} is missing required columns: {missing}"
        )


class IncompatibleTypesError(SchemaContractError):
    """Raised when a column's actual type does not match the contract."""

    def __init__(
        self,
        table_name: str,
        mismatches: list[tuple[str, str, str]],
    ) -> None:
        self.table_name = table_name
        self.mismatches = mismatches  # [(column, expected, actual), ...]
        details = "; ".join(
            f"{col}: expected {exp}, got {act}" for col, exp, act in mismatches
        )
        super().__init__(
            f"Table {table_name!r} has incompatible column types: {details}"
        )


# ---------------------------------------------------------------------------
# Preprocessing errors (Phase 3)
# ---------------------------------------------------------------------------
class PreprocessingError(Exception):
    """Raised when deterministic preprocessing of raw values fails."""


# ---------------------------------------------------------------------------
# Segmentation errors (Phase 3)
# ---------------------------------------------------------------------------
class SegmentationError(Exception):
    """Raised when sentence segmentation fails or violates its contract."""


# ---------------------------------------------------------------------------
# Shard-discovery errors (Phase 2)
# ---------------------------------------------------------------------------
class ShardDiscoveryError(Exception):
    """Raised when local shard discovery encounters an invalid layout."""

    def __init__(self, shard_key: str, detail: str) -> None:
        self.shard_key = shard_key
        self.detail = detail
        super().__init__(f"Shard {shard_key!r}: {detail}")


# ---------------------------------------------------------------------------
# Join-integrity errors (Phase 2)
# ---------------------------------------------------------------------------
class JoinIntegrityError(Exception):
    """Raised when join-key integrity checks fail."""

    def __init__(
        self,
        source: str,
        table_name: str,
        key: str,
        violation: str,
        sample: list[str],
    ) -> None:
        self.source = source
        self.table_name = table_name
        self.key = key
        self.violation = violation
        self.sample = sample
        detail = ", ".join(repr(v) for v in sample[:5])
        super().__init__(
            f"{source}/{table_name} integrity violation on {key!r}: "
            f"{violation} (sample: [{detail}])"
        )


# ---------------------------------------------------------------------------
# Finalization errors (Phase 4)
# ---------------------------------------------------------------------------
class FinalizationError(ValueError):
    """Raised when sentence finalization fails or violates its contract."""


# ---------------------------------------------------------------------------
# Export errors (Phase 5A)
# ---------------------------------------------------------------------------
class ExportError(ValueError):
    """Raised when local dataset export fails."""
