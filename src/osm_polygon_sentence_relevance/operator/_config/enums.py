"""Operator scope and stage values."""

from enum import StrEnum


class Scope(StrEnum):
    """Operator execution scope."""

    REGION = "region"
    ALL = "all"


class Stage(StrEnum):
    """Operator stage within the end-to-end pipeline."""

    SPLIT = "split"
    LABEL = "label"
    ALL = "all"


__all__ = ["Scope", "Stage"]
