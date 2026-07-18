"""Hugging Face dataset publishing of a validated local export.

Publishes exactly the three verified artifacts
(``sentences.parquet``, ``manifest.json``, and the auto-generated
``README.md`` dataset card) to an existing Hugging Face dataset
repository in a single ``create_commit`` call.

Design contract
---------------

Two separate dependencies, both injectable:

- ``hub_api`` owns the network: it exposes ``create_commit(repo_id,
  repo_type, operations, commit_message, revision)``. It is always
  called exactly once.
- ``commit_operation_factory(path_in_repo, path_or_fileobj)`` builds one
  add operation per local file. The two returned objects are passed
  unchanged to ``hub_api.create_commit``.

If either dependency is absent, only the missing one is imported lazily
from ``huggingface_hub``. Fully-injected calls perform zero Hub work
and never import the library.

Validation order is fixed:

1. Public arguments (dataset_id, target_revision, commit_message) are
   checked for non-blank strings before any other work.
2. ``validate_export_directory`` runs *before* any Hub import or
   factory / API call. A corrupt export cannot reach the network.
3. The two operations are constructed via the (possibly defaulted)
   ``commit_operation_factory``.
4. ``hub_api.create_commit`` is invoked exactly once.
5. The response ``(oid, commit_url)`` from ``huggingface_hub.CommitInfo``
   is validated and a frozen ``PublicationResult`` is returned.

The repository is assumed to already exist; this function does not
create one, does not accept a token, and does not retry. The CLI is
intentionally not extended here; publishing remains a programmatic,
single-commit operation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_sentence_relevance.contracts.errors import PublicationError
from osm_polygon_sentence_relevance.output.validation import (
    validate_export_directory,
)

# File names established by the existing exporter contract.
_PARQUET_NAME = "sentences.parquet"
_MANIFEST_NAME = "manifest.json"
_CARD_NAME = "README.md"

# Optional Hub extra. Lazy import only when an uninjected dependency
# must be filled in.
_HUB_EXTRA_HINT = (
    "huggingface_hub is not installed. Install it with: uv sync --extra hub"
)


@dataclass(frozen=True, slots=True)
class PublicationResult:
    """Verified facts about a single published Hub commit.

    Every field is either returned by the Hub API (``commit_id``,
    ``commit_url``) or confirmed against the validated export
    (``row_count``, ``sha256``) before the instance is constructed.
    """

    dataset_id: str
    target_revision: str
    commit_id: str
    commit_url: str
    row_count: int
    sha256: str


# Callable signatures used in the public API.
OperationFactory = Callable[..., Any]
"""Builds one commit operation from ``(path_in_repo, path_or_fileobj)``."""


def _require_non_blank_string(value: object, field: str) -> str:
    """Return a string after rejecting non-strings, blanks, and whitespace."""
    if not isinstance(value, str) or not value.strip():
        raise PublicationError(f"{field} must be a non-blank string")
    return value


def _import_hub() -> Any:
    """Lazy-import ``huggingface_hub`` with an actionable error."""
    try:
        import huggingface_hub
    except ImportError as err:
        raise PublicationError(_HUB_EXTRA_HINT) from err
    return huggingface_hub


def _default_commit_operation_factory() -> OperationFactory:
    """Return a factory that builds ``huggingface_hub.CommitOperationAdd``.

    The returned factory accepts keyword arguments ``path_in_repo`` and
    ``path_or_fileobj`` and returns a library ``CommitOperationAdd``
    instance. ``huggingface_hub`` is imported lazily on first call.
    """
    hub = _import_hub()

    def factory(*, path_in_repo: str, path_or_fileobj: str) -> Any:
        return hub.CommitOperationAdd(
            path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj
        )

    return factory


def _default_hub_api() -> Any:
    """Return a default ``HfApi`` instance, importing ``huggingface_hub`` lazily."""
    hub = _import_hub()
    try:
        return hub.HfApi()
    except Exception as err:
        raise PublicationError(
            "Failed to construct default Hugging Face HfApi client"
        ) from err


def _extract_commit_id_and_url(info: Any) -> tuple[str, str]:
    """Validate a ``huggingface_hub.CommitInfo`` response and return its
    ``(oid, commit_url)`` fields.

    The real ``huggingface_hub.CommitInfo`` exposes ``oid`` and
    ``commit_url`` (NOT ``url``). Objects exposing only a generic
    ``url`` attribute are rejected.
    """
    commit_id = getattr(info, "oid", None)
    commit_url = getattr(info, "commit_url", None)
    if not isinstance(commit_id, str) or not commit_id.strip():
        raise PublicationError(
            "Hugging Face create_commit returned a response without a "
            "non-blank 'oid' (commit id)"
        )
    if not isinstance(commit_url, str) or not commit_url.strip():
        raise PublicationError(
            "Hugging Face create_commit returned a response without a "
            "non-blank 'commit_url' (commit url)"
        )
    return commit_id, commit_url


def publish_export_directory(
    export_dir: str | Path,
    dataset_id: object,
    *,
    target_revision: object = "main",
    commit_message: object | None = None,
    hub_api: Any | None = None,
    commit_operation_factory: OperationFactory | None = None,
) -> PublicationResult:
    """Validate then publish a local export directory in a single Hub commit.

    Parameters
    ----------
    export_dir : str | Path
        The local export directory to publish (must already contain a
        validated ``sentences.parquet`` + ``manifest.json``).
    dataset_id : str
        Target Hugging Face dataset ID, e.g. ``"owner/name"``. Must be
        a non-blank string.
    target_revision : str, default ``"main"``
        Target branch / revision on the Hub repository.
    commit_message : str | None
        Commit message; if ``None`` a deterministic default derived
        from the validated checksum is used.
    hub_api : object | None
        Optional ``HfApi``-like object exposing ``create_commit``. If
        ``None``, a default ``HfApi`` is constructed via lazy import.
    commit_operation_factory : callable | None
        Optional factory with signature
        ``(*, path_in_repo, path_or_fileobj) -> operation``. If
        ``None``, ``huggingface_hub.CommitOperationAdd`` is used (lazy
        import).

    Returns
    -------
    PublicationResult
        Verified facts about the published commit.

    Raises
    ------
    PublicationError
        If any public argument is invalid, the export fails
        validation, the Hub extra is missing, the operation factory
        or ``hub_api.create_commit`` fails, or the commit response is
        malformed.
    """
    # 1. Public-argument validation BEFORE lazy import or Hub access.
    dataset_id_s = _require_non_blank_string(dataset_id, "dataset_id")
    target_revision_s = _require_non_blank_string(target_revision, "target_revision")

    if commit_message is None:
        default_message: str | None = None
    else:
        default_message = _require_non_blank_string(commit_message, "commit_message")

    # 2. Validate the export directory (read-only). This MUST happen
    #    before any lazy import or Hub call. ExportError propagates as
    #    is to preserve the validation contract.
    validated = validate_export_directory(export_dir)

    # 3. Resolve the default commit message if none was provided.
    if default_message is None:
        default_message = (
            f"Publish {_PARQUET_NAME}, {_MANIFEST_NAME}, and {_CARD_NAME} "
            f"({validated.row_count} rows, sha256={validated.sha256})"
        )

    # 4. Build the two add-only operations using local files directly.
    #    If an operation factory was injected it owns construction; the
    #    only translation (to ``CommitOperationAdd``) happens inside the
    #    default factory and never touches the production path here.
    if commit_operation_factory is None:
        commit_operation_factory = _default_commit_operation_factory()

    try:
        parquet_op = commit_operation_factory(
            path_in_repo=_PARQUET_NAME,
            path_or_fileobj=str(validated.parquet_path),
        )
        manifest_op = commit_operation_factory(
            path_in_repo=_MANIFEST_NAME,
            path_or_fileobj=str(validated.manifest_path),
        )
        card_op = commit_operation_factory(
            path_in_repo=_CARD_NAME,
            path_or_fileobj=str(validated.card_path),
        )
    except PublicationError:
        raise
    except Exception as err:
        raise PublicationError(
            f"Failed to construct commit operations for {dataset_id_s!r}: {err}"
        ) from err

    operations: list[Any] = [parquet_op, manifest_op, card_op]

    # 5. Resolve the Hub API. An injected API takes precedence; the
    #    default path constructs ``HfApi`` lazily.
    if hub_api is None:
        hub_api = _default_hub_api()

    # 6. Perform exactly one commit. Library/remote failures are
    #    re-wrapped as PublicationError with the original cause preserved.
    try:
        commit_info = hub_api.create_commit(
            repo_id=dataset_id_s,
            repo_type="dataset",
            operations=operations,
            commit_message=default_message,
            revision=target_revision_s,
        )
    except PublicationError:
        raise
    except Exception as err:
        raise PublicationError(
            f"Hugging Face publish failed for {dataset_id_s!r} at "
            f"revision {target_revision_s!r}: {err}"
        ) from err

    # 7. Validate the response shape and construct the result.
    commit_id, commit_url = _extract_commit_id_and_url(commit_info)

    return PublicationResult(
        dataset_id=dataset_id_s,
        target_revision=target_revision_s,
        commit_id=commit_id,
        commit_url=commit_url,
        row_count=validated.row_count,
        sha256=validated.sha256,
    )


__all__ = ["PublicationResult", "publish_export_directory"]
