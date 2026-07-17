"""Tests for the programmatic Hugging Face export publishing (Phase 7C).

These tests use fully-injected Hub API and commit-operation factory
fixtures to perform zero network calls. Validation and argument rules
must complete before any lazy import or Hub activity.

Public contract under test:

- ``hub_api`` owns ``create_commit(...)``.
- ``commit_operation_factory(path_in_repo, path_or_fileobj)`` constructs
  one add operation and returns it.
- The two constructed operation objects are passed unchanged to
  ``hub_api.create_commit`` exactly once.
- If either dependency is missing, ``huggingface_hub`` is imported lazily
  to fill the gap; fully-injected calls do not import it.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import FrozenInstanceError, dataclass, field
from pathlib import Path

import pytest

from osm_polygon_sentence_relevance.errors import ExportError
from osm_polygon_sentence_relevance.finalization import (
    finalize_sentence_dataset,
)
from osm_polygon_sentence_relevance.schemas import SEGMENTED_SENTENCES_SCHEMA
from tests.helpers import make_segmented_row

# ===================================================================
# Helpers: reuse validation test patterns to build a real export dir.
# ===================================================================


def _rows_to_table(rows: list[dict]):
    if not rows:
        return SEGMENTED_SENTENCES_SCHEMA.empty_table()
    import pyarrow as pa

    data = {}
    for schema_field in SEGMENTED_SENTENCES_SCHEMA:
        data[schema_field.name] = pa.array(
            [r[schema_field.name] for r in rows], type=schema_field.type
        )
    return pa.table(data, schema=SEGMENTED_SENTENCES_SCHEMA)


def _make_valid_export(tmpdir: str, *, n_rows: int = 2) -> Path:
    from osm_polygon_sentence_relevance.output import export_finalized_dataset

    rows = [
        make_segmented_row(sentence_text_normalized=f"sentence-{i}")
        for i in range(n_rows)
    ]
    table = _rows_to_table(rows)
    dataset = finalize_sentence_dataset(
        table, input_dataset_revision="rev-7c", pipeline_version="ver-7c"
    )
    res = export_finalized_dataset(dataset, tmpdir)
    assert res.parquet_path.exists()
    assert res.manifest_path.exists()
    return Path(tmpdir)


def _checksum(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


# ===================================================================
# Fake dependency fixtures
# ===================================================================


@dataclass
class FakeCommitInfo:
    """Mimics the real ``huggingface_hub.CommitInfo`` shape.

    The real type exposes ``oid`` and ``commit_url`` (NOT ``url``).
    """

    oid: str = "deadbeef" * 5
    commit_url: str = "https://huggingface.co/datasets/my/dataset/commit/deadbeef"


@dataclass
class FakeOperation:
    """Mimics ``huggingface_hub.CommitOperationAdd`` for assertions."""

    kind: str = "add"
    path_in_repo: str = ""
    path_or_fileobj: str = ""


@dataclass
class RecordingOperationFactory:
    """Records every per-operation call and returns a fresh ``FakeOperation``.

    The production API invokes this factory exactly twice (once per file).
    """

    calls: list[dict] = field(default_factory=list)
    raise_with: Exception | None = None

    def __call__(self, *, path_in_repo: str, path_or_fileobj: str):
        self.calls.append(
            {"path_in_repo": path_in_repo, "path_or_fileobj": path_or_fileobj}
        )
        if self.raise_with is not None:
            raise self.raise_with
        return FakeOperation(
            kind="add",
            path_in_repo=path_in_repo,
            path_or_fileobj=path_or_fileobj,
        )


@dataclass
class RecordingHubApi:
    """Records ``create_commit`` calls and returns a fake commit info."""

    create_commit_calls: list[dict] = field(default_factory=list)
    raise_with: Exception | None = None

    def create_commit(self, **kwargs):
        self.create_commit_calls.append(kwargs)
        if self.raise_with is not None:
            raise self.raise_with
        return FakeCommitInfo()


# ===================================================================
# Corrected public-contract tests
# ===================================================================


class TestHubApiIsAlwaysTheCommitExecutor:
    def test_hub_api_create_commit_called_exactly_once(self):
        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            publish_export_directory(
                export_dir,
                "my/dataset",
                hub_api=api,
                commit_operation_factory=factory,
            )
        assert len(api.create_commit_calls) == 1

    def test_create_commit_uses_repo_type_dataset(self):
        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            publish_export_directory(
                export_dir,
                "my/dataset",
                hub_api=api,
                commit_operation_factory=factory,
            )
        assert api.create_commit_calls[0]["repo_type"] == "dataset"


class TestCommitOperationFactoryContract:
    def test_factory_called_exactly_twice_with_correct_paths(self):
        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            publish_export_directory(
                export_dir,
                "my/dataset",
                hub_api=api,
                commit_operation_factory=factory,
            )
            calls = factory.calls
            assert len(calls) == 2
            paths = sorted(c["path_in_repo"] for c in calls)
            assert paths == ["manifest.json", "sentences.parquet"]
            # Both calls reference existing local files.
            for c in calls:
                assert Path(c["path_or_fileobj"]).is_file()
                assert Path(c["path_or_fileobj"]).name == c["path_in_repo"]

    def test_factory_returned_operations_passed_unchanged_to_create_commit(self):
        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            publish_export_directory(
                export_dir,
                "my/dataset",
                hub_api=api,
                commit_operation_factory=factory,
            )
        # The exact objects returned by the factory must appear in the
        # create_commit ``operations`` list, in order.
        passed_ops = api.create_commit_calls[0]["operations"]
        assert len(passed_ops) == 2
        # Every passed op is one of the factory's returned objects.
        assert len(factory.calls) == 2
        # We don't have the return values stored on the factory; instead,
        # assert each passed op equals a FakeOperation with the matching
        # path_in_repo/path_or_fileobj the factory was called with.
        paths_in_commit = sorted(
            (op.path_in_repo, op.path_or_fileobj) for op in passed_ops
        )
        paths_from_factory = sorted(
            (c["path_in_repo"], c["path_or_fileobj"]) for c in factory.calls
        )
        assert paths_in_commit == paths_from_factory

    def test_create_commit_no_delete_operations(self):
        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            publish_export_directory(
                export_dir,
                "my/dataset",
                hub_api=api,
                commit_operation_factory=factory,
            )
        ops = api.create_commit_calls[0]["operations"]
        assert all(getattr(op, "kind", None) == "add" for op in ops)


# ===================================================================
# Dependency injection boundary
# ===================================================================


class TestFullyInjectedNoHubImport:
    def test_fully_injected_works_with_huggingface_hub_blocked(self, monkeypatch):
        """When both ``hub_api`` and ``commit_operation_factory`` are
        injected, ``huggingface_hub`` must not be imported at all.
        """
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)

        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            publish_export_directory(
                export_dir,
                "my/dataset",
                hub_api=api,
                commit_operation_factory=factory,
            )
        assert len(api.create_commit_calls) == 1
        assert len(factory.calls) == 2

    def test_hub_only_injected_fetches_real_commit_operation_add(self, monkeypatch):
        """With only ``hub_api`` injected, the library is imported to obtain
        ``CommitOperationAdd``; genuine/fake library operation objects reach
        the injected API.
        """
        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        class _RealOp:
            def __init__(self, *, path_in_repo, path_or_fileobj):
                self.path_in_repo = path_in_repo
                self.path_or_fileobj = path_or_fileobj

        captured_ops = []

        class _RealHub:
            def create_commit(self, **kwargs):
                captured_ops.extend(kwargs["operations"])
                return FakeCommitInfo()

        class _FakeHub:
            @staticmethod
            def CommitOperationAdd(*, path_in_repo, path_or_fileobj):
                return _RealOp(
                    path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj
                )

        monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHub())

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            publish_export_directory(export_dir, "my/dataset", hub_api=_RealHub())

        # The injected hub_api received the genuine/fake library op objects.
        assert len(captured_ops) == 2
        assert all(isinstance(op, _RealOp) for op in captured_ops)

    def test_operation_factory_only_injected_fetches_real_hfapi(self, monkeypatch):
        """With only ``commit_operation_factory`` injected, the library is
        imported to construct ``HfApi``.
        """
        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        class _FakeHub:
            @staticmethod
            def HfApi():
                class _Api:
                    def create_commit(self, **kwargs):
                        return FakeCommitInfo()

                return _Api()

        monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHub())

        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            publish_export_directory(
                export_dir,
                "my/dataset",
                commit_operation_factory=factory,
            )
        # The injected factory was used to build both operations.
        assert len(factory.calls) == 2

    def test_neither_injected_uses_default_factory(self, monkeypatch):
        """With neither dependency injected, both ``HfApi`` and
        ``CommitOperationAdd`` come from the lazy import.
        """
        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        class _RealOp:
            def __init__(self, *, path_in_repo, path_or_fileobj):
                self.path_in_repo = path_in_repo
                self.path_or_fileobj = path_or_fileobj

        captured_ops = []

        class _FakeHub:
            HfApi = staticmethod(
                lambda: type(
                    "_Api",
                    (),
                    {
                        "create_commit": lambda self, **kw: (
                            captured_ops.extend(kw["operations"]),
                            FakeCommitInfo(),
                        )[-1]
                    },
                )()
            )

            @staticmethod
            def CommitOperationAdd(*, path_in_repo, path_or_fileobj):
                return _RealOp(
                    path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj
                )

        monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHub())

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            publish_export_directory(export_dir, "my/dataset")

        assert len(captured_ops) == 2
        assert all(isinstance(op, _RealOp) for op in captured_ops)


# ===================================================================
# Missing-dependency messaging
# ===================================================================


class TestMissingHuggingfaceHub:
    def test_missing_hub_when_neither_injected(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            publish_export_directory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            with pytest.raises(PublicationError) as exc:
                publish_export_directory(export_dir, "my/dataset")
        assert "uv sync --extra hub" in str(exc.value)
        assert "huggingface_hub" in str(exc.value)
        assert isinstance(exc.value.__cause__, ImportError)

    def test_missing_hub_when_only_hub_api_injected(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            publish_export_directory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            with pytest.raises(PublicationError) as exc:
                publish_export_directory(
                    export_dir, "my/dataset", hub_api=RecordingHubApi()
                )
        assert "uv sync --extra hub" in str(exc.value)
        assert isinstance(exc.value.__cause__, ImportError)

    def test_missing_hub_when_only_operation_factory_injected(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            publish_export_directory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            with pytest.raises(PublicationError) as exc:
                publish_export_directory(
                    export_dir,
                    "my/dataset",
                    commit_operation_factory=RecordingOperationFactory(),
                )
        assert "uv sync --extra hub" in str(exc.value)
        assert isinstance(exc.value.__cause__, ImportError)


# ===================================================================
# Operation-construction and create_commit failure wrapping
# ===================================================================


class TestConstructionAndCommitFailures:
    def test_operation_construction_failure_wrapped_with_preserved_cause(self):
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            publish_export_directory,
        )

        api = RecordingHubApi()
        factory = RecordingOperationFactory(
            raise_with=ValueError("bad op inputs"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            with pytest.raises(PublicationError) as exc:
                publish_export_directory(
                    export_dir,
                    "my/dataset",
                    hub_api=api,
                    commit_operation_factory=factory,
                )
        assert isinstance(exc.value.__cause__, ValueError)
        # create_commit must NOT have been called.
        assert api.create_commit_calls == []

    def test_create_commit_failure_wrapped_with_preserved_cause(self):
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            publish_export_directory,
        )

        api = RecordingHubApi(raise_with=ConnectionError("network down"))
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            with pytest.raises(PublicationError) as exc:
                publish_export_directory(
                    export_dir,
                    "my/dataset",
                    hub_api=api,
                    commit_operation_factory=factory,
                )
        assert isinstance(exc.value.__cause__, ConnectionError)
        # The operations were constructed but the commit failed.
        assert len(factory.calls) == 2
        assert len(api.create_commit_calls) == 1


# ===================================================================
# Validation-first ordering and argument hygiene
# ===================================================================


class TestValidationOrdering:
    def test_validation_completes_before_any_hub_activity(self):
        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            (export_dir / "sentences.parquet").unlink()
            with pytest.raises(ExportError):
                publish_export_directory(
                    export_dir,
                    "my/dataset",
                    hub_api=api,
                    commit_operation_factory=factory,
                )
        assert api.create_commit_calls == []
        assert factory.calls == []

    def test_invalid_export_causes_zero_hub_activity(self):
        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            import json

            manifest_path = export_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sha256"] = "0" * 64
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with pytest.raises(ExportError):
                publish_export_directory(
                    export_dir,
                    "my/dataset",
                    hub_api=api,
                    commit_operation_factory=factory,
                )
        assert api.create_commit_calls == []
        assert factory.calls == []


class TestPublicArgumentsRejectedBeforeHub:
    @pytest.mark.parametrize(
        "bad_dataset_id",
        ["", " ", "\t", 123, None, [], {}],
        ids=["empty", "space", "tab", "int", "none", "list", "dict"],
    )
    def test_blank_or_non_string_dataset_id_rejected_before_hub(self, bad_dataset_id):
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            publish_export_directory,
        )

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            with pytest.raises((PublicationError, TypeError)):
                publish_export_directory(
                    export_dir,
                    bad_dataset_id,
                    hub_api=api,
                    commit_operation_factory=factory,
                )
        assert api.create_commit_calls == []
        assert factory.calls == []

    @pytest.mark.parametrize(
        "bad_revision",
        ["", " ", "\t", 123, None, [], {}],
        ids=["empty", "space", "tab", "int", "none", "list", "dict"],
    )
    def test_blank_or_non_string_target_revision_rejected_before_hub(
        self, bad_revision
    ):
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            publish_export_directory,
        )

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            with pytest.raises((PublicationError, TypeError)):
                publish_export_directory(
                    export_dir,
                    "my/dataset",
                    target_revision=bad_revision,
                    hub_api=api,
                    commit_operation_factory=factory,
                )
        assert api.create_commit_calls == []
        assert factory.calls == []

    @pytest.mark.parametrize(
        "bad_msg",
        ["", " ", "\t", 123, [], {}],
        ids=["empty", "space", "tab", "int", "list", "dict"],
    )
    def test_blank_or_non_string_commit_message_rejected_before_hub(self, bad_msg):
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            publish_export_directory,
        )

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            with pytest.raises((PublicationError, TypeError)):
                publish_export_directory(
                    export_dir,
                    "my/dataset",
                    commit_message=bad_msg,
                    hub_api=api,
                    commit_operation_factory=factory,
                )
        assert api.create_commit_calls == []
        assert factory.calls == []


# ===================================================================
# Response validation
# ===================================================================


class TestCreateCommitResponseValidation:
    def test_missing_oid_rejected(self):
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            publish_export_directory,
        )

        @dataclass
        class BadInfo:
            url: str = "https://huggingface.co/datasets/my/dataset/commit/abc"

        api = RecordingHubApi(raise_with=None)
        # Override the recorded response by wrapping create_commit.
        api.create_commit = lambda **kw: BadInfo()  # type: ignore[assignment]
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            with pytest.raises(PublicationError, match="oid"):
                publish_export_directory(
                    export_dir,
                    "my/dataset",
                    hub_api=api,
                    commit_operation_factory=factory,
                )

    def test_blank_oid_rejected(self):
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            publish_export_directory,
        )

        @dataclass
        class BlankOid:
            oid: str = "   "
            url: str = "https://huggingface.co/datasets/my/dataset/commit/abc"

        api = RecordingHubApi()
        api.create_commit = lambda **kw: BlankOid()  # type: ignore[assignment]
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            with pytest.raises(PublicationError, match="oid"):
                publish_export_directory(
                    export_dir,
                    "my/dataset",
                    hub_api=api,
                    commit_operation_factory=factory,
                )

    def test_missing_commit_url_rejected(self):
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            publish_export_directory,
        )

        @dataclass
        class NoCommitUrl:
            oid: str = "abcdef"

        api = RecordingHubApi()
        api.create_commit = lambda **kw: NoCommitUrl()  # type: ignore[assignment]
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            with pytest.raises(PublicationError, match="commit_url"):
                publish_export_directory(
                    export_dir,
                    "my/dataset",
                    hub_api=api,
                    commit_operation_factory=factory,
                )


# ===================================================================
# Defaults and result contents
# ===================================================================


class TestDefaults:
    def test_default_target_revision_is_main(self):
        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            publish_export_directory(
                export_dir,
                "my/dataset",
                hub_api=api,
                commit_operation_factory=factory,
            )
        assert api.create_commit_calls[0]["revision"] == "main"

    def test_default_commit_message_is_deterministic_and_identical(self):
        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        api_a, api_b = RecordingHubApi(), RecordingHubApi()
        fa, fb = RecordingOperationFactory(), RecordingOperationFactory()
        with (
            tempfile.TemporaryDirectory() as tmpdir_a,
            tempfile.TemporaryDirectory() as tmpdir_b,
        ):
            export_a = _make_valid_export(tmpdir_a, n_rows=2)
            export_b = _make_valid_export(tmpdir_b, n_rows=2)
            publish_export_directory(
                export_a,
                "my/dataset",
                hub_api=api_a,
                commit_operation_factory=fa,
            )
            publish_export_directory(
                export_b,
                "my/dataset",
                hub_api=api_b,
                commit_operation_factory=fb,
            )
        msg_a = api_a.create_commit_calls[0]["commit_message"]
        msg_b = api_b.create_commit_calls[0]["commit_message"]
        assert msg_a == msg_b
        assert isinstance(msg_a, str)
        assert msg_a.strip()

    def test_explicit_commit_message_used_verbatim(self):
        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            publish_export_directory(
                export_dir,
                "my/dataset",
                commit_message="explicit message",
                hub_api=api,
                commit_operation_factory=factory,
            )
        assert api.create_commit_calls[0]["commit_message"] == "explicit message"


class TestPublicationResult:
    def test_publication_result_contains_validated_row_count_and_checksum(self):
        from osm_polygon_sentence_relevance.publishing import (
            PublicationResult,
            publish_export_directory,
        )

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir, n_rows=3)
            parquet_path = export_dir / "sentences.parquet"
            expected_sha = _checksum(parquet_path)
            result = publish_export_directory(
                export_dir,
                "my/dataset",
                target_revision="refs/heads/main",
                hub_api=api,
                commit_operation_factory=factory,
            )

        assert isinstance(result, PublicationResult)
        assert result.dataset_id == "my/dataset"
        assert result.target_revision == "refs/heads/main"
        assert result.commit_id == FakeCommitInfo().oid
        assert result.commit_url == FakeCommitInfo().commit_url
        assert result.row_count == 3
        assert result.sha256 == expected_sha

    def test_inputs_not_mutated_files_byte_for_byte_unchanged(self):
        from osm_polygon_sentence_relevance.publishing import publish_export_directory

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir, n_rows=2)
            parquet_path = export_dir / "sentences.parquet"
            manifest_path = export_dir / "manifest.json"

            before = {
                parquet_path: _checksum(parquet_path),
                manifest_path: _checksum(manifest_path),
            }

            publish_export_directory(
                export_dir,
                "my/dataset",
                hub_api=api,
                commit_operation_factory=factory,
            )

            after = {
                parquet_path: _checksum(parquet_path),
                manifest_path: _checksum(manifest_path),
            }
            assert before == after

    def test_publication_result_is_frozen_and_slotted(self):
        from osm_polygon_sentence_relevance.publishing import (
            PublicationResult,
            publish_export_directory,
        )

        api = RecordingHubApi()
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            result = publish_export_directory(
                export_dir,
                "my/dataset",
                hub_api=api,
                commit_operation_factory=factory,
            )
        assert isinstance(result, PublicationResult)
        assert result.__class__.__slots__ is not None
        with pytest.raises(FrozenInstanceError):
            result.commit_id = "tampered"  # type: ignore[misc]


class TestNoPrivateStandInOperationType:
    def test_publishing_module_does_not_expose_private_addop_type(self):
        """The production path must not depend on a private stand-in
        operation type; only the public API is acceptable.
        """
        import osm_polygon_sentence_relevance.publishing.huggingface as hf

        # No public _AddOp attribute.
        assert not hasattr(hf, "_AddOp")

    def test_publishing_module_does_not_accept_old_commit_op_factory_keyword(self):
        """The corrected public parameter is ``commit_operation_factory``;
        the old incorrect ``commit_op_factory`` is not retained.
        """
        import inspect

        sig = inspect.signature(
            __import__(
                "osm_polygon_sentence_relevance.publishing.huggingface",
                fromlist=["publish_export_directory"],
            ).publish_export_directory
        )
        assert "commit_operation_factory" in sig.parameters
        assert "commit_op_factory" not in sig.parameters


# ===================================================================
# Realistic CommitInfo contract
# ===================================================================


@dataclass
class RealisticCommitInfo:
    """Mirrors the real ``huggingface_hub.CommitInfo`` shape exactly.

    Real CommitInfo exposes ``oid`` and ``commit_url`` (NOT ``url``).
    """

    oid: str
    commit_url: str


@dataclass
class _OnlyGenericUrl:
    """An object with a valid ``oid`` and a valid generic ``url`` but no
    ``commit_url`` must not be accepted as a valid ``CommitInfo``.
    """

    oid: str
    url: str


class TestRealisticCommitInfoContract:
    """The publisher must read ``info.commit_url``, not ``info.url``."""

    def test_realistic_response_with_commit_url_succeeds(self):
        from osm_polygon_sentence_relevance.publishing import (
            PublicationResult,
            publish_export_directory,
        )

        realistic_url = (
            "https://huggingface.co/datasets/my/dataset/commit/abcdef0123456789"
        )
        realistic = RealisticCommitInfo(
            oid="abcdef0123456789" * 2,
            commit_url=realistic_url,
        )

        api = RecordingHubApi()
        api.create_commit = lambda **kw: realistic  # type: ignore[assignment]
        factory = RecordingOperationFactory()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            result = publish_export_directory(
                export_dir,
                "my/dataset",
                hub_api=api,
                commit_operation_factory=factory,
            )

        assert isinstance(result, PublicationResult)
        assert result.commit_id == realistic.oid
        assert result.commit_url == realistic.commit_url
        assert result.commit_url == realistic_url

    def test_response_missing_commit_url_rejected(self):
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            publish_export_directory,
        )

        @dataclass
        class NoCommitUrl:
            oid: str = "abcdef0123456789"

        api = RecordingHubApi()
        api.create_commit = lambda **kw: NoCommitUrl()  # type: ignore[assignment]
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            with pytest.raises(PublicationError, match="commit_url"):
                publish_export_directory(
                    export_dir,
                    "my/dataset",
                    hub_api=api,
                    commit_operation_factory=factory,
                )

    @pytest.mark.parametrize(
        "bad_value",
        ["", " ", "\t", 123, None, [], {}],
        ids=["empty", "space", "tab", "int", "none", "list", "dict"],
    )
    def test_blank_or_non_string_commit_url_rejected(self, bad_value):
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            publish_export_directory,
        )

        @dataclass
        class BadCommitUrl:
            oid: str = "abcdef0123456789"
            commit_url: object = None

            def __init__(self, url):
                self.commit_url = url

        api = RecordingHubApi()
        api.create_commit = lambda **kw: BadCommitUrl(  # type: ignore[assignment]
            bad_value
        )
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            with pytest.raises(PublicationError, match="commit_url"):
                publish_export_directory(
                    export_dir,
                    "my/dataset",
                    hub_api=api,
                    commit_operation_factory=factory,
                )

    def test_object_with_only_generic_url_attribute_rejected(self):
        """An object exposing only a generic ``url`` (not ``commit_url``)
        must NOT be treated as a valid ``CommitInfo``.

        The fixture also supplies a valid ``oid`` so the rejection must
        come from the ``commit_url`` check, not the earlier ``oid``
        check, and the error message must mention ``commit_url``.
        """
        from osm_polygon_sentence_relevance.publishing import (
            PublicationError,
            publish_export_directory,
        )

        api = RecordingHubApi()
        api.create_commit = lambda **kw: _OnlyGenericUrl(  # type: ignore[assignment]
            oid="abcdef0123456789",
            url="https://huggingface.co/datasets/my/dataset/commit/abcdef",
        )
        factory = RecordingOperationFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            with pytest.raises(PublicationError, match="commit_url"):
                publish_export_directory(
                    export_dir,
                    "my/dataset",
                    hub_api=api,
                    commit_operation_factory=factory,
                )
