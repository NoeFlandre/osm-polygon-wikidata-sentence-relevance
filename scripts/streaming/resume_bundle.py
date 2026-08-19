"""Compatibility import for the packaged split-resume implementation."""

from __future__ import annotations

import sys

from osm_polygon_sentence_relevance.operator import resume_bundle as _implementation
from osm_polygon_sentence_relevance.operator.resume_bundle import (
    MANIFEST_NAME,
    ResumeBundle,
    ResumeBundleError,
    ResumeMergeResult,
    create_resume_bundle,
    merge_resume_bundle,
    valid_streaming_shard_key,
    validate_resume_bundle,
    validate_streaming_state,
)

__all__ = [
    "MANIFEST_NAME",
    "ResumeBundle",
    "ResumeBundleError",
    "ResumeMergeResult",
    "create_resume_bundle",
    "merge_resume_bundle",
    "valid_streaming_shard_key",
    "validate_resume_bundle",
    "validate_streaming_state",
]

sys.modules[__name__] = _implementation
