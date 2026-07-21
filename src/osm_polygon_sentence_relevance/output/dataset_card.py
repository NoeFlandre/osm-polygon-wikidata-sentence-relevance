"""Stable public facade for deterministic dataset-card generation."""

from ._card.rendering import (
    render_dataset_card,
    render_dataset_card_from_profile,
    schema_field_documentation,
    schema_has_map_types,
)
from ._card.statistics import (
    STATISTICS_VERSION,
    DatasetStatistics,
    compute_parquet_statistics,
    compute_statistics,
    statistics_from_dict,
    statistics_to_dict,
)

__all__ = [
    "STATISTICS_VERSION",
    "DatasetStatistics",
    "compute_parquet_statistics",
    "compute_statistics",
    "render_dataset_card",
    "render_dataset_card_from_profile",
    "schema_field_documentation",
    "schema_has_map_types",
    "statistics_from_dict",
    "statistics_to_dict",
]
