from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from osm_polygon_sentence_relevance.contracts.errors import AcquisitionError
from osm_polygon_sentence_relevance.ingestion.discovery import discover_shards

# Keep allow/ignore patterns as module-level immutable constants
ALLOW_PATTERNS: tuple[str, ...] = (
    "polygons/*.parquet",
    "polygon_articles/*.parquet",
    "wikipedia/documents/*.parquet",
    "wikipedia/sections/*.parquet",
    "wikivoyage/documents/*.parquet",
    "wikivoyage/sections/*.parquet",
)

IGNORE_PATTERNS: tuple[str, ...] = ("articles/*",)


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """The result of read-only Hugging Face dataset snapshot acquisition."""

    dataset_id: str
    requested_revision: str
    resolved_sha: str
    snapshot_path: Path
    discovered_region_count: int


def acquire_dataset_snapshot(
    dataset_id: str,
    requested_revision: str,
    *,
    hub_api: object | None = None,
    download_fn: Callable[..., str] | None = None,
) -> AcquisitionResult:
    """Download a dataset snapshot from the Hugging Face Hub under strict path validation rules."""
    # 1. Validate dataset_id and requested_revision types and blank CLI strings first
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-blank string")
    if not isinstance(requested_revision, str) or not requested_revision.strip():
        raise ValueError("requested_revision must be a non-blank string")

    # 2. Lazy-import only when an uninjected HfApi or snapshot_download is actually required
    if hub_api is None or download_fn is None:
        try:
            import huggingface_hub
        except ImportError as exc:
            raise AcquisitionError(
                "huggingface_hub is not installed. Please install it using: "
                "uv sync --extra hub"
            ) from exc

    # 3. Resolve revision using HfApi
    if hub_api is None:
        try:
            api = huggingface_hub.HfApi()
        except Exception as exc:
            raise AcquisitionError("Failed to initialize HfApi") from exc
    else:
        api = hub_api

    try:
        repo_info = api.repo_info(
            repo_id=dataset_id,
            repo_type="dataset",
            revision=requested_revision,
        )
        resolved_sha = repo_info.sha
        if isinstance(resolved_sha, str):
            resolved_sha = resolved_sha.lower()
    except Exception as exc:
        raise AcquisitionError(
            f"Failed to resolve revision {requested_revision!r} for dataset {dataset_id!r}"
        ) from exc

    # 4. Validate and normalize the resolved SHA-1 (must be exactly 40 hex characters)
    if (
        not resolved_sha
        or not isinstance(resolved_sha, str)
        or not re.match(r"^[a-fA-F0-9]{40}$", resolved_sha)
    ):
        raise AcquisitionError(
            f"Resolved SHA {resolved_sha!r} is invalid: it must be exactly 40 hexadecimal characters"
        )

    # 5. Check resolved revision mismatch BEFORE download if requested_revision is a full SHA-1 hash (40 hex chars)
    is_sha1 = len(requested_revision) == 40 and bool(
        re.match(r"^[a-fA-F0-9]{40}$", requested_revision)
    )
    if is_sha1 and resolved_sha != requested_revision.lower():
        raise AcquisitionError(
            "Resolved commit SHA does not match the requested commit SHA-1 hash"
        )

    # 6. Download the exact dataset snapshot (Parquet files only, exclude articles/)
    if download_fn is None:
        download_fn = huggingface_hub.snapshot_download

    try:
        snapshot_dir = download_fn(
            repo_id=dataset_id,
            revision=resolved_sha,
            repo_type="dataset",
            allow_patterns=list(ALLOW_PATTERNS),
            ignore_patterns=list(IGNORE_PATTERNS),
        )
    except Exception as exc:
        raise AcquisitionError(
            f"Failed to download snapshot of dataset {dataset_id!r} at commit {resolved_sha!r}"
        ) from exc

    snapshot_path = Path(snapshot_dir)

    # 7. Validate downloaded layout via discover_shards
    try:
        shards = discover_shards(snapshot_path)
    except Exception as exc:
        raise AcquisitionError(
            f"Discovered shards validation failed for downloaded snapshot at {snapshot_path}: {exc}"
        ) from exc

    if not shards:
        raise AcquisitionError(
            f"Discovered shards validation failed for downloaded snapshot at {snapshot_path}: no regional shards found (missing required shards)"
        )

    return AcquisitionResult(
        dataset_id=dataset_id,
        requested_revision=requested_revision,
        resolved_sha=resolved_sha,
        snapshot_path=snapshot_path,
        discovered_region_count=len(shards),
    )
