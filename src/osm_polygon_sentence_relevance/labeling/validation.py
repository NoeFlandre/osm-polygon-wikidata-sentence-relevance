"""Strict validation of constrained model responses."""

from __future__ import annotations

import json
from typing import Any

from .contracts import LabelValue, SentenceLabel
from .prompt import (
    LANDUSE_REASON_VALUES,
    POLYGON_REASON_VALUES,
    REQUIRED_RESPONSE_FIELDS,
)


class LabelValidationError(ValueError):
    """Raised when a model response violates the labeling contract."""


_FIELDS = frozenset(REQUIRED_RESPONSE_FIELDS)
_REASONS = {
    "landuse_reason": frozenset(LANDUSE_REASON_VALUES),
    "polygon_reason": frozenset(POLYGON_REASON_VALUES),
}
_CONSISTENT = {
    "landuse_reason": {
        LabelValue.YES: frozenset(
            {"explicit_land_use", "explicit_land_cover", "built_or_managed_feature"}
        ),
        LabelValue.NO: frozenset({"no_landuse_or_cover"}),
        LabelValue.UNCERTAIN: frozenset({"insufficient_evidence"}),
    },
    "polygon_reason": {
        LabelValue.YES: frozenset(
            {
                "direct_polygon_reference",
                "context_resolved_reference",
                "place_description",
            }
        ),
        LabelValue.NO: frozenset(
            {"nearby_or_broader_area", "navigation_or_reference_text", "unrelated_fact"}
        ),
        LabelValue.UNCERTAIN: frozenset({"insufficient_evidence"}),
    },
}


def parse_label_response(raw: str, *, target_sentence: str) -> SentenceLabel:
    """Parse one closed-schema response and validate its evidence."""

    try:
        value: Any = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LabelValidationError("response must be valid JSON object") from exc
    if not isinstance(value, dict):
        raise LabelValidationError("response must be a JSON object")
    if set(value) != _FIELDS:
        raise LabelValidationError("response fields do not match the required schema")
    labels: dict[str, LabelValue] = {}
    for field in ("landuse_relevance", "polygon_relevance"):
        candidate = value[field]
        try:
            labels[field] = (
                LabelValue(candidate) if isinstance(candidate, str) else LabelValue("")
            )
        except ValueError as exc:
            raise LabelValidationError(f"{field} has an invalid value") from exc
    for field in ("landuse_reason", "polygon_reason"):
        if not isinstance(value[field], str) or value[field] not in _REASONS[field]:
            raise LabelValidationError(f"{field} has an invalid value")
        label_field = (
            "landuse_relevance" if field == "landuse_reason" else "polygon_relevance"
        )
        if value[field] not in _CONSISTENT[field][labels[label_field]]:
            raise LabelValidationError(f"{field} is inconsistent with {label_field}")
    evidence = value["evidence"]
    if not isinstance(evidence, str):
        raise LabelValidationError("evidence must be a string")
    if len(evidence) > 240:
        raise LabelValidationError("evidence must contain at most 240 characters")
    if evidence and evidence not in target_sentence:
        raise LabelValidationError(
            "evidence must be an exact substring of target sentence"
        )
    return SentenceLabel(
        landuse_relevance=labels["landuse_relevance"],
        polygon_relevance=labels["polygon_relevance"],
        landuse_reason=value["landuse_reason"],
        polygon_reason=value["polygon_reason"],
        evidence=evidence,
    )


__all__ = ["LabelValidationError", "parse_label_response"]
