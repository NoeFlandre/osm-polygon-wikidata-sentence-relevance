"""Atomic Hugging Face publication for complete labeled datasets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .finalization import validate_labeled_publication


class LabelPublicationError(RuntimeError):
    """Raised when validated labeled publication fails."""


@dataclass(frozen=True, slots=True)
class LabelPublicationResult:
    commit_id: str
    commit_url: str
    row_count: int
    parquet_sha256: str


def publish_labeled_dataset(
    directory: Path,
    dataset_id: str,
    *,
    target_revision: str = "main",
    hub_api: Any | None = None,
    operation_factory: Callable[..., Any] | None = None,
    readback_downloader: Callable[[str, str], Path] | None = None,
) -> LabelPublicationResult:
    """Validate, atomically publish, and verify the exact Hub commit."""

    if not dataset_id.strip() or not target_revision.strip():
        raise LabelPublicationError("dataset ID and target revision must be non-blank")
    validated = validate_labeled_publication(directory)
    if hub_api is None or operation_factory is None:
        try:
            from huggingface_hub import CommitOperationAdd, HfApi
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LabelPublicationError(
                "install the hub extra to publish labels"
            ) from exc
        hub_api = hub_api or HfApi()
        operation_factory = operation_factory or CommitOperationAdd
    operations = [
        operation_factory(
            path_in_repo=str(path.relative_to(validated.directory)),
            path_or_fileobj=str(path),
        )
        for path in validated.files
    ]
    try:
        info = hub_api.create_commit(
            repo_id=dataset_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Publish {validated.row_count} Afghanistan relevance labels",
            revision=target_revision,
        )
    except Exception as exc:
        raise LabelPublicationError("Hugging Face label publication failed") from exc
    oid = getattr(info, "oid", None)
    url = getattr(info, "commit_url", None)
    if not isinstance(oid, str) or not oid or not isinstance(url, str) or not url:
        raise LabelPublicationError("Hugging Face returned an invalid commit response")
    if readback_downloader is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LabelPublicationError(
                "install the hub extra to verify the published labels"
            ) from exc

        def readback_downloader(repo_id: str, revision: str) -> Path:
            return Path(
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                    allow_patterns=[
                        "sentences.parquet",
                        "manifest.json",
                        "README.md",
                        "assets/label_distribution.png",
                        "assets/positive_languages.png",
                    ],
                )
            )

    try:
        readback = validate_labeled_publication(readback_downloader(dataset_id, oid))
    except Exception as exc:
        raise LabelPublicationError("Hub readback validation failed") from exc
    if (
        readback.parquet_sha256 != validated.parquet_sha256
        or readback.row_count != validated.row_count
    ):
        raise LabelPublicationError("Hub readback does not match the uploaded dataset")
    return LabelPublicationResult(
        commit_id=oid,
        commit_url=url,
        row_count=validated.row_count,
        parquet_sha256=validated.parquet_sha256,
    )


__all__ = ["LabelPublicationError", "LabelPublicationResult", "publish_labeled_dataset"]
