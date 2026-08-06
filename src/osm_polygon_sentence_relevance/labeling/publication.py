"""Atomic Hugging Face publication for complete labeled datasets.

Both releases are committed to the dataset's single ``main`` revision. V1
keeps its historical root paths; V2 is mapped below ``v2-worldwide/``. This
prevents a worldwide continuation from replacing the Afghanistan files while
avoiding a second public release branch.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .finalization import validate_labeled_publication
from .releases import (
    V2_REMOTE_PREFIX,
    ReleaseLane,
    release_lane,
    remote_release_path,
)


class LabelPublicationError(RuntimeError):
    """Raised when validated labeled publication fails."""


@dataclass(frozen=True, slots=True)
class LabelPublicationResult:
    commit_id: str
    commit_url: str
    row_count: int
    parquet_sha256: str


# The closed release layout the publisher maintains on the Hub.
_BASE_RELEASE_FILES: tuple[str, ...] = (
    "sentences.parquet",
    "manifest.json",
    "README.md",
    "assets/label_distribution.png",
    "assets/positive_languages.png",
    "assets/joint_label_heatmap.png",
    "assets/polygon_coverage_funnel.png",
    "assets/reason_code_distribution.png",
)
_V2_RELEASE_FILES: tuple[str, ...] = tuple(
    remote_release_path(ReleaseLane.V2_WORLDWIDE, path) for path in _BASE_RELEASE_FILES
) + ("v2-worldwide/assets/h3_sentence_distribution.png",)
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
        "assets/slice_yield.html",
        "v2-worldwide/assets/slice_yield.html",
    }
)


def _expected_revision(directory: Path) -> str:
    """Return the only public Hugging Face revision used by this dataset."""

    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelPublicationError("cannot read label publication manifest") from exc
    identity = manifest.get("run_identity")
    if not isinstance(identity, dict):
        raise LabelPublicationError("label publication identity is missing")
    expected = "main"
    recorded = manifest.get("publication_revision")
    if recorded is not None and recorded != expected:
        raise LabelPublicationError(
            "publication revision is inconsistent with identity"
        )
    return expected


def _require_release_revision(revision: str, expected: str) -> None:
    if revision != "main":
        raise LabelPublicationError("all label releases must use the main revision")
    if revision != expected:
        raise LabelPublicationError("label publication revision is not main")


def _classify_remote_path(path: str) -> str:
    """Classify release, checkpoint, and unexpected remote paths."""

    if path == _GITATTRIBUTES_NAME:
        return "gitattributes"
    if path in _BASE_RELEASE_FILES or path in _V2_RELEASE_FILES:
        return "preserve"
    if path.startswith(".pipeline/checkpoints/"):
        return "preserve"
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


def _default_list_remote_files(
    hub_api: Any, dataset_id: str, revision: str
) -> list[str]:
    try:
        return list(
            hub_api.list_repo_files(
                repo_id=dataset_id,
                repo_type="dataset",
                revision=revision,
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


def _default_readback_downloader(
    repo_id: str, revision: str, *, allow_patterns: list[str] | None = None
) -> Path:
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
            allow_patterns=allow_patterns
            or [*_BASE_RELEASE_FILES, _GITATTRIBUTES_NAME],
        )
    )


def _release_snapshot(root: Path, lane: ReleaseLane) -> Path:
    """Materialize one release from a same-main Hub snapshot for validation."""

    if (root / "manifest.json").is_file():
        return root
    source = root / V2_REMOTE_PREFIX if lane is ReleaseLane.V2_WORLDWIDE else root
    if not (source / "manifest.json").is_file():
        raise LabelPublicationError("Hub readback is missing the selected release")
    target = Path(tempfile.mkdtemp(prefix="label-readback-"))
    for path in source.rglob("*"):
        if path.is_file():
            relative = path.relative_to(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    return target


def publish_labeled_dataset(
    directory: Path,
    dataset_id: str,
    *,
    target_revision: str | None = None,
    hub_api: Any | None = None,
    operation_factory: Callable[..., Any] | None = None,
    readback_downloader: Callable[[str, str], Path] | None = None,
    list_remote_files: Callable[[Any, str], list[str]] | None = None,
) -> LabelPublicationResult:
    """Validate, atomically publish, and verify the exact Hub commit."""

    if not dataset_id.strip() or (
        target_revision is not None and not target_revision.strip()
    ):
        raise LabelPublicationError("dataset ID and target revision must be non-blank")
    validated = validate_labeled_publication(directory)
    expected_revision = _expected_revision(directory)
    target_revision = target_revision or expected_revision
    _require_release_revision(target_revision, expected_revision)
    try:
        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        lane = release_lane(manifest["run_identity"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise LabelPublicationError("label publication lane is missing") from exc

    if hub_api is None:
        hub_api = _default_hub_api()
    if operation_factory is None:
        operation_factory = _default_operation_factory()
    if readback_downloader is None:
        patterns = (
            list(_V2_RELEASE_FILES)
            if lane is ReleaseLane.V2_WORLDWIDE
            else list(_BASE_RELEASE_FILES)
        )
        patterns.append(_GITATTRIBUTES_NAME)

        def readback_downloader(repo: str, revision: str) -> Path:
            return _default_readback_downloader(repo, revision, allow_patterns=patterns)

    if list_remote_files is None:

        def list_remote_files(api: Any, dataset: str) -> list[str]:
            return _default_list_remote_files(api, dataset, target_revision)

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
                "path_in_repo": remote_release_path(lane, str(rel)),
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
                f"Publish {validated.row_count} {lane.value} relevance labels"
            ),
            revision="main",
        )
    except Exception as exc:
        raise LabelPublicationError("Hugging Face label publication failed") from exc

    oid = getattr(info, "oid", None)
    url = getattr(info, "commit_url", None)
    if not isinstance(oid, str) or not oid or not isinstance(url, str) or not url:
        raise LabelPublicationError("Hugging Face returned an invalid commit response")

    try:
        readback_root = readback_downloader(dataset_id, oid)
        readback = validate_labeled_publication(
            _release_snapshot(Path(readback_root), lane)
        )
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
