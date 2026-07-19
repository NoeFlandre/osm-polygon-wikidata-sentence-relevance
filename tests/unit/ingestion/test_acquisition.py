from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.acquisition import (
    AcquisitionResult,
    acquire_dataset_snapshot,
)
from osm_polygon_sentence_relevance.errors import AcquisitionError

# ===================================================================
# Helper to build conforming dataset snapshot layout
# ===================================================================


def make_table_from_rows(schema: pa.Schema, rows: list[dict]) -> pa.Table:
    data = {}
    for field in schema:
        col_values = []
        for r in rows:
            if field.name in r:
                col_values.append(r[field.name])
            else:
                if pa.types.is_string(field.type):
                    col_values.append("")
                elif pa.types.is_integer(field.type):
                    col_values.append(0)
                elif pa.types.is_floating(field.type):
                    col_values.append(0.0)
                elif pa.types.is_boolean(field.type):
                    col_values.append(False)
                else:
                    col_values.append(None)
        data[field.name] = pa.array(col_values, type=field.type)
    return pa.table(data, schema=schema)


def write_dummy_layout(root: Path, shard_key="reg-1") -> None:
    # 1. polygons
    from osm_polygon_sentence_relevance.schemas import POLYGONS_SCHEMA

    polygons_dir = root / "polygons"
    polygons_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        make_table_from_rows(
            POLYGONS_SCHEMA,
            [{"polygon_id": "poly-1", "wikidata": "Q1", "region": shard_key}],
        ),
        polygons_dir / f"{shard_key}.parquet",
    )

    # 2. polygon_articles
    from osm_polygon_sentence_relevance.schemas import POLYGON_ARTICLES_SCHEMA

    pa_dir = root / "polygon_articles"
    pa_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        make_table_from_rows(
            POLYGON_ARTICLES_SCHEMA,
            [{"polygon_id": "poly-1", "article_id": "art-1", "wikidata": "Q1"}],
        ),
        pa_dir / f"{shard_key}.parquet",
    )

    # 3. wikipedia/documents
    from osm_polygon_sentence_relevance.schemas import WIKIPEDIA_DOCUMENTS_SCHEMA

    wp_doc_dir = root / "wikipedia/documents"
    wp_doc_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        make_table_from_rows(WIKIPEDIA_DOCUMENTS_SCHEMA, []),
        wp_doc_dir / f"{shard_key}.parquet",
    )

    # 4. wikipedia/sections
    from osm_polygon_sentence_relevance.schemas import SECTIONS_SCHEMA

    wp_sec_dir = root / "wikipedia/sections"
    wp_sec_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        make_table_from_rows(SECTIONS_SCHEMA, []), wp_sec_dir / f"{shard_key}.parquet"
    )


# ===================================================================
# Mocks
# ===================================================================


class FakeRepoInfo:
    def __init__(self, sha: str):
        self.sha = sha


class MockHfApi:
    def __init__(self, sha="mock-sha-12345"):
        self.sha = sha
        self.calls = []

    def repo_info(self, repo_id, repo_type, revision):
        self.calls.append((repo_id, repo_type, revision))
        if self.sha is None:
            raise RuntimeError("Repository not found or unresolved revision")
        return FakeRepoInfo(self.sha)


# ===================================================================
# Test Suite for Dataset Snapshot Acquisition (Phase 6B)
# ===================================================================


class TestAcquisition:
    def test_missing_dependency(self, monkeypatch):
        # Simulate missing optional dependency locally only
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(AcquisitionError) as exc_info:
            acquire_dataset_snapshot("my/dataset", "main")
        assert "uv sync --extra hub" in str(exc_info.value)
        assert "huggingface_hub" in str(exc_info.value)

    def test_lazy_hfapi_construction_failure_wrapped(self, monkeypatch):
        """When ``hub_api`` is not injected and ``HfApi()`` itself
        raises (e.g. due to a network/auth error), the function must
        wrap the error in ``AcquisitionError``.
        """

        class _BoomHfApi:
            def __init__(self):
                raise RuntimeError("network/auth boom")

        # Monkey-patch the ``huggingface_hub`` import to return a
        # module whose ``HfApi`` constructor raises.
        class _FakeModule:
            HfApi = _BoomHfApi

        monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeModule())
        with pytest.raises(AcquisitionError, match="Failed to initialize HfApi"):
            acquire_dataset_snapshot("my/dataset", "main")

    def test_blank_or_non_string_arguments_fail_early(self, monkeypatch):
        # Block huggingface_hub import locally to prove it fails BEFORE imports
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)

        with pytest.raises(ValueError, match="dataset_id"):
            acquire_dataset_snapshot("", "main")
        with pytest.raises(ValueError, match="dataset_id"):
            acquire_dataset_snapshot(123, "main")
        with pytest.raises(ValueError, match="requested_revision"):
            acquire_dataset_snapshot("my/dataset", "  ")
        with pytest.raises(ValueError, match="requested_revision"):
            acquire_dataset_snapshot("my/dataset", None)

    def test_surrounding_whitespace_dataset_id_rejected_before_lazy_import(
        self, monkeypatch
    ):
        """A dataset identifier with leading/trailing whitespace is
        rejected with ``ValueError`` BEFORE the lazy import of
        ``huggingface_hub`` or any Hub/API call. The acquisition fn
        must never silently trim the identifier.
        """
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)

        download_calls = []

        def download_fn(*args, **kwargs):
            download_calls.append(args)
            return "/nonexistent/path"

        api_calls = []

        class TrackingApi:
            def repo_info(self, *args, **kwargs):
                api_calls.append(args)
                raise AssertionError(
                    "HfApi.repo_info must not be called for an invalid dataset_id"
                )

        # ``hub_api`` and ``download_fn`` are both injected so the only
        # available failure mode is the explicit validation step. Any
        # trimming/normalization inside acquisition would still record
        # the call.
        with pytest.raises(ValueError, match="surrounding whitespace"):
            acquire_dataset_snapshot(
                "  my/dataset  ",
                "main",
                hub_api=TrackingApi(),
                download_fn=download_fn,
            )
        # The download must never run, and the API must never be called.
        assert len(download_calls) == 0
        assert len(api_calls) == 0

    def test_acquisition_works_with_injected_even_if_blocked(self, monkeypatch):
        # Block import locally
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)

        api = MockHfApi(sha="a" * 40)
        with tempfile.TemporaryDirectory() as tmp_download:
            download_dir = Path(tmp_download)
            write_dummy_layout(download_dir, "region-x")

            def mock_download(*a, **kw):
                return str(download_dir)

            res = acquire_dataset_snapshot(
                "my/dataset", "main", hub_api=api, download_fn=mock_download
            )
            assert res.resolved_sha == "a" * 40

    def test_invalid_commit_sha_format(self):
        api = MockHfApi(sha="invalid-sha")

        def dummy_download(*_args, **_kwargs):
            return "fake_dir"

        with pytest.raises(AcquisitionError, match="exactly 40 hexadecimal characters"):
            acquire_dataset_snapshot(
                "my/dataset", "main", hub_api=api, download_fn=dummy_download
            )

        api_39 = MockHfApi(sha="a" * 39)
        with pytest.raises(AcquisitionError, match="exactly 40 hexadecimal characters"):
            acquire_dataset_snapshot(
                "my/dataset", "main", hub_api=api_39, download_fn=dummy_download
            )

    def test_explicit_sha_mismatch_fails_before_download(self):
        sha_req = "a" * 40
        sha_res = "b" * 40
        api = MockHfApi(sha=sha_res)

        download_called = False

        def mock_download(*args, **kwargs):
            nonlocal download_called
            download_called = True
            return "fake_dir"

        with pytest.raises(
            AcquisitionError, match="Resolved commit SHA does not match"
        ):
            acquire_dataset_snapshot(
                "my/dataset", sha_req, hub_api=api, download_fn=mock_download
            )

        assert not download_called

    def test_allow_ignore_patterns_immutable_constants(self):
        from osm_polygon_sentence_relevance.acquisition import (
            ALLOW_PATTERNS,
            IGNORE_PATTERNS,
        )

        assert isinstance(ALLOW_PATTERNS, tuple)
        assert isinstance(IGNORE_PATTERNS, tuple)
        assert len(ALLOW_PATTERNS) == 6
        for pat in ALLOW_PATTERNS:
            assert not pat.startswith("articles/")
            assert "/articles/" not in pat

    def test_uppercase_sha_normalization_succeeds(self):
        sha_upper = "A" * 40
        api = MockHfApi(sha=sha_upper)

        with tempfile.TemporaryDirectory() as tmp_download:
            download_dir = Path(tmp_download)
            write_dummy_layout(download_dir, "region-x")

            download_calls = []

            def mock_download(
                repo_id, revision, repo_type, allow_patterns, ignore_patterns
            ):
                download_calls.append(revision)
                return str(download_dir)

            res = acquire_dataset_snapshot(
                "my/dataset", sha_upper, hub_api=api, download_fn=mock_download
            )

            # Succeeded and normalized resolved SHA to lowercase
            assert res.resolved_sha == "a" * 40
            assert download_calls == ["a" * 40]

    def test_successful_acquisition(self):
        api = MockHfApi(sha="a" * 40)
        download_calls = []

        with tempfile.TemporaryDirectory() as tmp_download:
            download_dir = Path(tmp_download)
            write_dummy_layout(download_dir, "region-x")

            def mock_download(
                repo_id, revision, repo_type, allow_patterns, ignore_patterns
            ):
                download_calls.append(
                    {
                        "repo_id": repo_id,
                        "revision": revision,
                        "repo_type": repo_type,
                        "allow_patterns": allow_patterns,
                        "ignore_patterns": ignore_patterns,
                    }
                )
                return str(download_dir)

            res = acquire_dataset_snapshot(
                "my/dataset", "main", hub_api=api, download_fn=mock_download
            )

            assert isinstance(res, AcquisitionResult)
            assert res.dataset_id == "my/dataset"
            assert res.requested_revision == "main"
            assert res.resolved_sha == "a" * 40
            assert res.snapshot_path == download_dir
            assert res.discovered_region_count == 1

            assert api.calls == [("my/dataset", "dataset", "main")]
            assert len(download_calls) == 1
            assert download_calls[0]["revision"] == "a" * 40

    def test_unresolved_revision_fails(self):
        api = MockHfApi(sha=None)

        def dummy_download(*_args, **_kwargs):
            return "fake_dir"

        with pytest.raises(AcquisitionError, match="Failed to resolve revision"):
            acquire_dataset_snapshot(
                "my/dataset", "invalid-rev", hub_api=api, download_fn=dummy_download
            )

    def test_layout_validation_fails(self):
        api = MockHfApi(sha="a" * 40)

        with tempfile.TemporaryDirectory() as tmp_download:
            download_dir = Path(tmp_download)

            def mock_download(*args, **kwargs):
                return str(download_dir)

            with pytest.raises(
                AcquisitionError, match="Discovered shards validation failed"
            ):
                acquire_dataset_snapshot(
                    "my/dataset", "main", hub_api=api, download_fn=mock_download
                )
