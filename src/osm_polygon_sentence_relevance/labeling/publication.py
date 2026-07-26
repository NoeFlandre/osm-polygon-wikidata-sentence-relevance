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


# The closed release layout the publisher maintains on the Hub.
_LABELED_RELEASE_FILES: tuple[str, ...] = (
    "sentences.parquet",
    "manifest.json",
    "README.md",
    "assets/label_distribution.png",
    "assets/positive_languages.png",
)
# ``.gitattributes`` must always be preserved verbatim; it is never
# part of the add/replace/delete set so it survives across releases.
_GITATTRIBUTES_NAME = ".gitattributes"
# Obsolete files left behind by earlier releases that the new commit
# is allowed to delete. Anything outside this set is unexpected and
# the publisher must refuse publication.
_OBSOLETE_DELETE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "assets/geographic_coverage.png",
        "assets/language_distribution.png",
    }
)


def _require_main_revision(revision: str) -> None:
    if revision != "main":
        raise LabelPublicationError("label publication only targets the main branch")


def _classify_remote_path(path: str) -> str:
    """Return ``expected``, ``obsolete``, ``gitattributes``, or ``unexpected``."""

    if path == _GITATTRIBUTES_NAME:
        return "gitattributes"
    if path in _LABELED_RELEASE_FILES:
        return "expected"
    if path in _OBSOLETE_DELETE_ALLOWLIST:
        return "obsolete"
    return "unexpected"


def _default_operation_factory() -> Callable[..., Any]:
    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LabelPublicationError("install the hub extra to publish labels") from exc

    def factory(*, op: str, path_in_repo: str, path_or_fileobj: str | Path) -> Any:
        if op == "add":
            return CommitOperationAdd(
                path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj
            )
        return CommitOperationDelete(path_in_repo=path_in_repo)

    return factory


def _default_list_remote_files(hub_api: Any, dataset_id: str) -> list[str]:
    try:
        return list(
            hub_api.list_repo_files(
                repo_id=dataset_id,
                repo_type="dataset",
                revision="main",
            )
        )
    except Exception as exc:
        raise LabelPublicationError("Hugging Face list_repo_files failed") from exc


def _default_hub_api() -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LabelPublicationError("install the hub extra to publish labels") from exc
    return HfApi()


def _default_readback_downloader(repo_id: str, revision: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LabelPublicationError(
            "install the hub extra to verify the published labels"
        ) from exc
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=list(_LABELED_RELEASE_FILES) + [_GITATTRIBUTES_NAME],
        )
    )


def publish_labeled_dataset(
    directory: Path,
    dataset_id: str,
    *,
    target_revision: str = "main",
    hub_api: Any | None = None,
    operation_factory: Callable[..., Any] | None = None,
    readback_downloader: Callable[[str, str], Path] | None = None,
    list_remote_files: Callable[[Any, str], list[str]] | None = None,
) -> LabelPublicationResult:
    """Validate, atomically publish, and verify the exact Hub commit."""

    if not dataset_id.strip() or not target_revision.strip():
        raise LabelPublicationError("dataset ID and target revision must be non-blank")
    _require_main_revision(target_revision)
    validated = validate_labeled_publication(directory)

    if hub_api is None:
        hub_api = _default_hub_api()
    if operation_factory is None:
        operation_factory = _default_operation_factory()
    if readback_downloader is None:
        readback_downloader = _default_readback_downloader
    if list_remote_files is None:
        list_remote_files = _default_list_remote_files

    remote_files = list_remote_files(hub_api, dataset_id)
    unexpected = sorted(
        path for path in remote_files if _classify_remote_path(path) == "unexpected"
    )
    if unexpected:
        raise LabelPublicationError(
            "remote tree contains unexpected files: " + ", ".join(unexpected)
        )

    operations: list[dict[str, Any]] = []
    for path in validated.files:
        rel = path.relative_to(validated.directory)
        operations.append(
            {
                "op": "add",
                "path_in_repo": str(rel),
                "path_or_fileobj": str(path),
            }
        )
    for remote_path in remote_files:
        if _classify_remote_path(remote_path) == "obsolete":
            operations.append(
                {
                    "op": "delete",
                    "path_in_repo": remote_path,
                    "path_or_fileobj": "",
                }
            )

    constructed_ops: list[Any] = []
    for spec in operations:
        try:
            constructed_ops.append(
                operation_factory(
                    op=spec["op"],
                    path_in_repo=spec["path_in_repo"],
                    path_or_fileobj=spec["path_or_fileobj"],
                )
            )
        except Exception as exc:
            raise LabelPublicationError(
                f"failed to construct {spec['op']!r} operation for "
                f"{spec['path_in_repo']!r}: {exc}"
            ) from exc

    try:
        info = hub_api.create_commit(
            repo_id=dataset_id,
            repo_type="dataset",
            operations=constructed_ops,
            commit_message=(
                f"Publish {validated.row_count} Afghanistan relevance labels"
            ),
            revision=target_revision,
        )
    except Exception as exc:
        raise LabelPublicationError("Hugging Face label publication failed") from exc

    oid = getattr(info, "oid", None)
    url = getattr(info, "commit_url", None)
    if not isinstance(oid, str) or not oid or not isinstance(url, str) or not url:
        raise LabelPublicationError("Hugging Face returned an invalid commit response")

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


__all__ = [
    "LabelPublicationError",
    "LabelPublicationResult",
    "publish_labeled_dataset",
]
