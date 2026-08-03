"""Hardened Hugging Face atomic publication tests.

The publisher must:

1. validate the local labeled release first;
2. inspect the target repository's current tree;
3. construct one Hub commit that:
   - adds/replaces:
     sentences.parquet, manifest.json, README.md,
     assets/label_distribution.png, assets/positive_languages.png,
     assets/joint_label_heatmap.png, assets/polygon_coverage_funnel.png,
     assets/reason_code_distribution.png, assets/slice_yield.html
   - deletes only explicitly allowlisted obsolete paths
     (assets/geographic_coverage.png, assets/language_distribution.png)
   - preserves .gitattributes
4. refuse publication if unexpected remote content exists;
5. after the commit, snapshot_download the immutable commit tree and
   verify exactly .gitattributes plus the labeled-release files;
6. independently validate every artifact (row count, every SHA);
7. verify the target revision is ``main``;
8. do not create a branch or repository;
9. do not accept a token argument.

The tests use fully-injected fake Hub APIs so zero network calls are
made. Each fake captures the operations passed to ``create_commit`` and
the arguments to ``snapshot_download`` so we can assert the contract.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
from osm_polygon_sentence_relevance.labeling.contracts import (
    LabelRecord,
    LabelValue,
    RunIdentity,
)
from osm_polygon_sentence_relevance.labeling.finalization import (
    finalize_labeled_dataset,
)
from osm_polygon_sentence_relevance.labeling.publication import (
    LabelPublicationError,
    publish_labeled_dataset,
)

# ---------------------------------------------------------------------------
# Fake Hub infrastructure
# ---------------------------------------------------------------------------


def _fake_op_factory() -> Callable[..., Any]:
    """Return a factory that records every operation specification."""

    def factory(*, op: str, path_in_repo: str, path_or_fileobj: str | None) -> dict:
        return {
            "op": op,
            "path_in_repo": path_in_repo,
            "path_or_fileobj": path_or_fileobj,
        }

    return factory


def _fake_readback_downloader(source_publication: Path) -> Callable[[str, str], Path]:
    """Return a snapshot_download that copies the local publication to a temp dir."""

    def downloader(repo_id: str, revision: str) -> Path:
        safe_id = repo_id.replace("/", "_")
        target = source_publication.parent / f"readback_{safe_id}_{revision}"
        target.mkdir(parents=True, exist_ok=True)
        for src in source_publication.rglob("*"):
            if src.is_file():
                rel = src.relative_to(source_publication)
                (target / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target / rel)
        return target

    return downloader


@dataclass
class _FakeCommitInfo:
    oid: str = "f" * 40
    commit_url: str = "https://example.com/commit/f" + "f" * 40


@dataclass
class _RecordingHub:
    """Records ``list_repo_files``, ``create_commit`` and supports readback."""

    files: list[str] = field(default_factory=list)
    create_commit_calls: list[dict[str, Any]] = field(default_factory=list)
    snapshot_download_calls: list[dict[str, Any]] = field(default_factory=list)
    downloads_dir: Path | None = None
    fail_create_commit: Exception | None = None
    fail_snapshot_download: Exception | None = None

    def list_repo_files(
        self, *, repo_id: str, repo_type: str, revision: str
    ) -> list[str]:
        return list(self.files)

    def create_commit(self, **kwargs: Any) -> Any:
        if self.fail_create_commit is not None:
            raise self.fail_create_commit
        self.create_commit_calls.append(kwargs)
        return _FakeCommitInfo()

    def snapshot_download(self, **kwargs: Any) -> str:
        if self.fail_snapshot_download is not None:
            raise self.fail_snapshot_download
        self.snapshot_download_calls.append(kwargs)
        assert self.downloads_dir is not None
        repo_id = kwargs["repo_id"]
        revision = kwargs["revision"]
        target = self.downloads_dir / f"{repo_id.replace('/', '_')}_{revision}"
        target.mkdir(parents=True, exist_ok=True)
        return str(target)


def _identity(input_sha256: str = "a" * 64) -> RunIdentity:
    return RunIdentity(
        input_sha256=input_sha256,
        input_dataset_revision="b" * 40,
        model_repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        model_revision="c" * 40,
        model_file="Qwen3.6-27B-Q4_K_M.gguf",
        model_file_sha256="d" * 64,
        prompt_version="afghanistan-landuse-polygon-v2",
        source_commit="e" * 40,
        engine="llama.cpp",
        engine_version="1",
        batch_size=2,
        llama_parallel=16,
        llama_per_slot_context=4096,
        llama_total_context=65536,
        request_concurrency=16,
    )


def _write_input(path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "sentence_id": ["s1", "s2"],
                "region": ["afghanistan", "afghanistan"],
                "language": ["en", "fa"],
                "sentence_text_raw": ["farming", "history"],
            }
        ),
        path,
    )


def _build_publication(tmp_path: Path) -> Path:
    input_path = tmp_path / "input.parquet"
    _write_input(input_path)
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    store = CheckpointStore(tmp_path / "work", _identity(digest))
    store.write_batch(
        0,
        [
            LabelRecord(
                "s1",
                LabelValue.YES,
                LabelValue.YES,
                "explicit_land_use",
                "direct_polygon_reference",
                "farming",
            ),
            LabelRecord(
                "s2",
                LabelValue.NO,
                LabelValue.YES,
                "no_landuse_or_cover",
                "direct_polygon_reference",
                "history",
            ),
        ],
    )
    store.write_timing(
        {
            "total_wall_seconds": 12.5,
            "initial_inference_seconds": 10.0,
            "repair_inference_seconds": 0.0,
            "inference_seconds": 10.0,
            "checkpoint_and_validation_seconds": 2.5,
        }
    )
    output = tmp_path / "publication"
    finalize_labeled_dataset(
        input_path=input_path,
        store=store,
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )
    return output


# ---------------------------------------------------------------------------
# 1. Clean replacement: only the labeled-release files exist remotely.
# ---------------------------------------------------------------------------


def test_clean_replacement_atomically_replaces(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    downloads = tmp_path / "downloads"
    hub = _RecordingHub(
        files=[".gitattributes", "README.md"],
        downloads_dir=downloads,
    )
    result = publish_labeled_dataset(
        output,
        "owner/dataset",
        hub_api=hub,
        operation_factory=_fake_op_factory(),
        readback_downloader=_fake_readback_downloader(output),
        target_revision="main",
    )
    assert result.commit_id == "f" * 40
    # One create_commit invocation, never more.
    assert len(hub.create_commit_calls) == 1
    kwargs = hub.create_commit_calls[0]
    assert kwargs["repo_id"] == "owner/dataset"
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["revision"] == "main"
    # Exactly five add operations plus any required deletes.
    operations = kwargs["operations"]
    expected_paths = {
        "sentences.parquet",
        "manifest.json",
        "README.md",
        "assets/label_distribution.png",
        "assets/positive_languages.png",
        "assets/joint_label_heatmap.png",
        "assets/polygon_coverage_funnel.png",
        "assets/reason_code_distribution.png",
        "assets/slice_yield.html",
    }
    add_paths = {op["path_in_repo"] for op in operations if op["op"] == "add"}
    delete_paths = {op["path_in_repo"] for op in operations if op["op"] == "delete"}
    assert add_paths == expected_paths
    # The remote had no obsolete assets so no delete operations.
    assert delete_paths == set()


# ---------------------------------------------------------------------------
# 2. Obsolete assets present: the publisher deletes them in the same commit.
# ---------------------------------------------------------------------------


def test_obsolete_assets_are_deleted_in_the_same_commit(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    downloads = tmp_path / "downloads"
    hub = _RecordingHub(
        files=[
            ".gitattributes",
            "README.md",
            "assets/geographic_coverage.png",
            "assets/language_distribution.png",
        ],
        downloads_dir=downloads,
    )
    publish_labeled_dataset(
        output,
        "owner/dataset",
        hub_api=hub,
        operation_factory=_fake_op_factory(),
        readback_downloader=_fake_readback_downloader(output),
        target_revision="main",
    )
    operations = hub.create_commit_calls[0]["operations"]
    delete_paths = {op["path_in_repo"] for op in operations if op["op"] == "delete"}
    assert delete_paths == {
        "assets/geographic_coverage.png",
        "assets/language_distribution.png",
    }


def test_obsolete_assets_absent_means_no_delete(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    downloads = tmp_path / "downloads"
    hub = _RecordingHub(
        files=[".gitattributes", "README.md"],
        downloads_dir=downloads,
    )
    publish_labeled_dataset(
        output,
        "owner/dataset",
        hub_api=hub,
        operation_factory=_fake_op_factory(),
        readback_downloader=_fake_readback_downloader(output),
        target_revision="main",
    )
    operations = hub.create_commit_calls[0]["operations"]
    delete_paths = [op for op in operations if op["op"] == "delete"]
    assert delete_paths == []


# ---------------------------------------------------------------------------
# 3. Unexpected remote file: the publisher refuses publication.
# ---------------------------------------------------------------------------


def test_unexpected_remote_file_is_rejected(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    downloads = tmp_path / "downloads"
    hub = _RecordingHub(
        files=[
            ".gitattributes",
            "README.md",
            "extra_unauthorized.txt",
        ],
        downloads_dir=downloads,
    )
    with pytest.raises(LabelPublicationError, match="unexpected"):
        publish_labeled_dataset(
            output,
            "owner/dataset",
            hub_api=hub,
            operation_factory=_fake_op_factory(),
            readback_downloader=_fake_readback_downloader(output),
            target_revision="main",
        )
    # The publisher must NOT have created a commit.
    assert hub.create_commit_calls == []


def test_remote_gitattributes_only_is_accepted(tmp_path: Path) -> None:
    """``.gitattributes`` alone in the remote is a valid empty predecessor."""

    output = _build_publication(tmp_path)
    downloads = tmp_path / "downloads"
    hub = _RecordingHub(
        files=[".gitattributes"],
        downloads_dir=downloads,
    )
    result = publish_labeled_dataset(
        output,
        "owner/dataset",
        hub_api=hub,
        operation_factory=_fake_op_factory(),
        readback_downloader=_fake_readback_downloader(output),
        target_revision="main",
    )
    assert result.commit_id == "f" * 40


# ---------------------------------------------------------------------------
# 4. Deletion failure: the create_commit call fails.
# ---------------------------------------------------------------------------


def test_deletion_failure_propagates(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    downloads = tmp_path / "downloads"
    hub = _RecordingHub(
        files=[
            ".gitattributes",
            "README.md",
            "assets/geographic_coverage.png",
        ],
        downloads_dir=downloads,
        fail_create_commit=OSError("remote refused"),
    )
    with pytest.raises(LabelPublicationError):
        publish_labeled_dataset(
            output,
            "owner/dataset",
            hub_api=hub,
            operation_factory=_fake_op_factory(),
            readback_downloader=_fake_readback_downloader(output),
            target_revision="main",
        )


# ---------------------------------------------------------------------------
# 5. Atomic create_commit failure must propagate as a LabelPublicationError.
# ---------------------------------------------------------------------------


def test_create_commit_failure_propagates(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    downloads = tmp_path / "downloads"
    hub = _RecordingHub(
        files=[".gitattributes"],
        downloads_dir=downloads,
        fail_create_commit=OSError("network down"),
    )
    with pytest.raises(LabelPublicationError):
        publish_labeled_dataset(
            output,
            "owner/dataset",
            hub_api=hub,
            operation_factory=_fake_op_factory(),
            readback_downloader=_fake_readback_downloader(output),
            target_revision="main",
        )


# ---------------------------------------------------------------------------
# 6. Immutable-tree mismatch: snapshot_download returns the wrong files.
# ---------------------------------------------------------------------------


def test_immutable_tree_mismatch_is_rejected(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    downloads = tmp_path / "downloads"

    def bad_readback(repo_id: str, revision: str) -> Path:
        target = downloads / f"{repo_id.replace('/', '_')}_{revision}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "sentences.parquet").write_bytes(b"tampered")
        (target / ".gitattributes").write_text("* filter=lfs\n")
        return target

    hub = _RecordingHub(
        files=[".gitattributes"],
        downloads_dir=downloads,
    )
    with pytest.raises(LabelPublicationError, match="readback"):
        publish_labeled_dataset(
            output,
            "owner/dataset",
            hub_api=hub,
            operation_factory=_fake_op_factory(),
            readback_downloader=bad_readback,
            target_revision="main",
        )


# ---------------------------------------------------------------------------
# 7. Readback hash mismatch: bytes differ from the local publication.
# ---------------------------------------------------------------------------


def test_readback_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    downloads = tmp_path / "downloads"

    class _TamperingHub(_RecordingHub):
        pass

    hub = _TamperingHub(
        files=[".gitattributes"],
        downloads_dir=downloads,
    )

    # Persistent readback: copy once, then the test can tamper with it
    # without the readback downloader overwriting the changes.
    state: dict[str, Path] = {}

    def mutable_readback(repo_id: str, revision: str) -> Path:
        key = f"{repo_id}_{revision}"
        target = downloads / key.replace("/", "_")
        if key not in state:
            target.mkdir(parents=True, exist_ok=True)
            for src in output.rglob("*"):
                if src.is_file():
                    rel = src.relative_to(output)
                    (target / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, target / rel)
            state[key] = target
        return state[key]

    publish_labeled_dataset(
        output,
        "owner/dataset",
        hub_api=hub,
        operation_factory=_fake_op_factory(),
        readback_downloader=mutable_readback,
        target_revision="main",
    )
    # Tamper with the readback parquet bytes; the second call must fail.
    readback = state["owner/dataset_" + "f" * 40]
    parquet = readback / "sentences.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"tamper")
    with pytest.raises(LabelPublicationError, match="readback"):
        publish_labeled_dataset(
            output,
            "owner/dataset",
            hub_api=hub,
            operation_factory=_fake_op_factory(),
            readback_downloader=mutable_readback,
            target_revision="main",
        )


# ---------------------------------------------------------------------------
# 8. Branch other than ``main`` is rejected.
# ---------------------------------------------------------------------------


def test_non_main_target_revision_is_rejected(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    with pytest.raises(LabelPublicationError, match="main"):
        publish_labeled_dataset(
            output,
            "owner/dataset",
            hub_api=_RecordingHub(
                files=[".gitattributes"], downloads_dir=tmp_path / "downloads"
            ),
            operation_factory=_fake_op_factory(),
            readback_downloader=_fake_readback_downloader(output),
            target_revision="feature-branch",
        )


# ---------------------------------------------------------------------------
# 9. ``.gitattributes`` is preserved across the commit.
# ---------------------------------------------------------------------------


def test_gitattributes_is_preserved(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    downloads = tmp_path / "downloads"
    hub = _RecordingHub(
        files=[".gitattributes", "README.md"],
        downloads_dir=downloads,
    )

    # Use the hub's snapshot_download to populate the readback with both
    # the labeled release and ``.gitattributes``.
    def hub_readback(repo_id: str, revision: str) -> Path:
        target = downloads / f"{repo_id.replace('/', '_')}_{revision}"
        target.mkdir(parents=True, exist_ok=True)
        # Mirror the local publication into the readback tree.
        for src in output.rglob("*"):
            if src.is_file():
                rel = src.relative_to(output)
                (target / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target / rel)
        # And write ``.gitattributes`` because the remote still has it.
        (target / ".gitattributes").write_text("* filter=lfs\n")
        hub.snapshot_download_calls.append({"repo_id": repo_id, "revision": revision})
        return target

    publish_labeled_dataset(
        output,
        "owner/dataset",
        hub_api=hub,
        operation_factory=_fake_op_factory(),
        readback_downloader=hub_readback,
        target_revision="main",
    )
    operations = hub.create_commit_calls[0]["operations"]
    # ``.gitattributes`` must NOT be deleted or replaced.
    delete_paths = {op["path_in_repo"] for op in operations if op["op"] == "delete"}
    assert ".gitattributes" not in delete_paths
    # The readback directory includes ``.gitattributes``.
    final = downloads / ("owner_dataset_" + "f" * 40)
    assert (final / ".gitattributes").exists()


# ---------------------------------------------------------------------------
# 10. No token, branch, or repository creation is supported.
# ---------------------------------------------------------------------------


def test_publisher_does_not_accept_token(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    with pytest.raises((TypeError, LabelPublicationError)):
        publish_labeled_dataset(
            output,
            "owner/dataset",
            hub_api=_RecordingHub(
                files=[".gitattributes"], downloads_dir=tmp_path / "downloads"
            ),
            operation_factory=_fake_op_factory(),
            readback_downloader=_fake_readback_downloader(output),
            token="hf_secret",  # type: ignore[call-arg]
        )


def test_publisher_does_not_create_branch(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    downloads = tmp_path / "downloads"
    hub = _RecordingHub(
        files=[".gitattributes"],
        downloads_dir=downloads,
    )
    publish_labeled_dataset(
        output,
        "owner/dataset",
        hub_api=hub,
        operation_factory=_fake_op_factory(),
        readback_downloader=_fake_readback_downloader(output),
        target_revision="main",
    )
    # The publisher must use ``revision`` (a SHA or branch), not create_repo.
    for kwargs in hub.create_commit_calls:
        assert "create_branch" not in kwargs
        assert "create_repo" not in kwargs


# ---------------------------------------------------------------------------
# 11. The publisher never imports ``huggingface_hub`` when fully injected.
# ---------------------------------------------------------------------------


def test_publisher_does_not_import_hub_when_fully_injected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _build_publication(tmp_path)
    downloads = tmp_path / "downloads"
    hub = _RecordingHub(
        files=[".gitattributes"],
        downloads_dir=downloads,
    )
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    readback_calls: list[tuple[str, str]] = []

    def tracking_readback(repo_id: str, revision: str) -> Path:
        readback_calls.append((repo_id, revision))
        return _fake_readback_downloader(output)(repo_id, revision)

    publish_labeled_dataset(
        output,
        "owner/dataset",
        hub_api=hub,
        operation_factory=_fake_op_factory(),
        readback_downloader=tracking_readback,
        target_revision="main",
    )
    # No network access occurred; the operation count is bounded.
    assert len(hub.create_commit_calls) == 1
    assert len(readback_calls) == 1
