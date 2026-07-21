"""Read-only validation of an exported dataset directory (Phase 7B).

These tests assert the public ``validate_export_directory`` contract:
an already-exported directory must be internally consistent (Parquet
present, manifest present and well-formed, checksum and row count
matching, and schema equal to ``OUTPUT_SENTENCE_SCHEMA``) before any
future publication/upload step is allowed to touch it.

Validation performs no writes and leaves all files byte-for-byte
unchanged. No network access.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.errors import ExportError
from osm_polygon_sentence_relevance.finalization import (
    finalize_sentence_dataset,
)
from osm_polygon_sentence_relevance.output.dataset_card import (
    compute_parquet_statistics,
)
from osm_polygon_sentence_relevance.schemas import SEGMENTED_SENTENCES_SCHEMA
from tests.helpers import make_segmented_row

# ===================================================================
# Helpers
# ===================================================================


def _rows_to_table(rows: list[dict]) -> pa.Table:
    """Build a SEGMENTED_SENTENCES_SCHEMA table from row dicts."""
    if not rows:
        return SEGMENTED_SENTENCES_SCHEMA.empty_table()
    data = {}
    for field in SEGMENTED_SENTENCES_SCHEMA:
        col_values = [r[field.name] for r in rows]
        data[field.name] = pa.array(col_values, type=field.type)
    return pa.table(data, schema=SEGMENTED_SENTENCES_SCHEMA)


def _make_valid_export(tmpdir: str, *, n_rows: int = 2) -> Path:
    """Produce a real export via the existing exporter and return its dir."""
    from osm_polygon_sentence_relevance.output import export_finalized_dataset

    rows = [
        make_segmented_row(sentence_text_normalized=f"sentence-{i}")
        for i in range(n_rows)
    ]
    table = _rows_to_table(rows)
    dataset = finalize_sentence_dataset(
        table, input_dataset_revision="rev-7b", pipeline_version="ver-7b"
    )
    res = export_finalized_dataset(dataset, tmpdir)
    assert res.parquet_path.exists()
    assert res.manifest_path.exists()
    return Path(tmpdir)


def _checksum(path: Path) -> str:
    """Independent streamed SHA-256 for test verification."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def _rewrite_manifest(export_dir: Path, **overrides: object) -> dict:
    """Read the manifest, apply overrides, rewrite deterministically, return it."""
    manifest_path = export_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(overrides)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return manifest


def _rewrite_parquet(
    export_dir: Path,
    *,
    table: pa.Table,
) -> Path:
    """Overwrite the export's Parquet with *table*; return its path."""
    parquet_path = export_dir / "sentences.parquet"
    pq.write_table(table, parquet_path)
    return parquet_path


def _build_output_table(
    *,
    n_rows: int = 2,
    metadata: dict[bytes, bytes] | None = None,
) -> pa.Table:
    """Build an OUTPUT_SENTENCE_SCHEMA table carrying optional schema metadata."""
    segmented_rows = [
        make_segmented_row(sentence_text_normalized=f"sentence-{i}")
        for i in range(n_rows)
    ]
    table = _rows_to_table(segmented_rows)
    dataset = finalize_sentence_dataset(
        table, input_dataset_revision="rev-7b", pipeline_version="ver-7b"
    )
    if metadata is None:
        return dataset.table
    return dataset.table.replace_schema_metadata(metadata)


# ===================================================================
# Validation contract
# ===================================================================


class TestValidateExportDirectory:
    def test_valid_export_validates_successfully(self):
        from osm_polygon_sentence_relevance.output import (
            ValidatedExport,
            validate_export_directory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir, n_rows=3)
            result = validate_export_directory(export_dir)

            assert isinstance(result, ValidatedExport)
            # Frozen + slotted.
            assert result.__class__.__slots__ is not None
            with pytest.raises(FrozenInstanceError):
                result.row_count = 999  # type: ignore[misc]

            assert result.export_dir == export_dir.resolve()
            assert result.parquet_path == export_dir.resolve() / "sentences.parquet"
            assert result.manifest_path == export_dir.resolve() / "manifest.json"
            assert result.row_count == 3
            assert result.sha256 == _checksum(result.parquet_path)

    def test_non_directory_path_rejected_early(self):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "not_a_dir.txt"
            file_path.write_text("hello")

            with pytest.raises(ExportError) as exc:
                validate_export_directory(file_path)
            assert "directory" in str(exc.value).lower()

    def test_non_path_argument_rejected_early(self):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with pytest.raises(TypeError):
            validate_export_directory(12345)  # type: ignore[arg-type]

    def test_missing_parquet_rejected(self):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            (export_dir / "sentences.parquet").unlink()

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "parquet" in str(exc.value).lower()

    def test_missing_manifest_rejected(self):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            (export_dir / "manifest.json").unlink()

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "manifest" in str(exc.value).lower()

    def test_malformed_manifest_rejected(self):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            (export_dir / "manifest.json").write_text("{ not valid json ")

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "manifest" in str(exc.value).lower()

    def test_checksum_mismatch_rejected(self):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            manifest_path = export_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sha256"] = "deadbeef" + "0" * 56
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert (
                "checksum" in str(exc.value).lower()
                or "sha256" in str(exc.value).lower()
            )

    def test_row_count_mismatch_rejected(self):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir, n_rows=2)
            manifest_path = export_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["row_count"] = 999
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "row" in str(exc.value).lower()

    def test_schema_mismatch_rejected(self):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            # Overwrite the Parquet with a wrong-schema table (single int column),
            # then rewrite the manifest so checksum + row_count match the new file,
            # isolating the schema check.
            bad_path = export_dir / "sentences.parquet"
            bad_table = pa.table({"x": pa.array([1, 2], type=pa.int64())})
            pq.write_table(bad_table, bad_path)
            manifest_path = export_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sha256"] = _checksum(bad_path)
            manifest["row_count"] = bad_table.num_rows
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "schema" in str(exc.value).lower()

    def test_validation_performs_no_writes(self):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir, n_rows=4)
            parquet_path = export_dir / "sentences.parquet"
            manifest_path = export_dir / "manifest.json"

            before = {
                parquet_path: _checksum(parquet_path),
                manifest_path: _checksum(manifest_path),
            }

            result = validate_export_directory(export_dir)

            after = {
                parquet_path: _checksum(parquet_path),
                manifest_path: _checksum(manifest_path),
            }
            assert before == after
            assert result.row_count == 4


# ===================================================================
# Parquet corruption / I/O failure boundary
# ===================================================================


class TestParquetIOFailures:
    def test_non_parquet_file_rejected_with_preserved_cause(self):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            # Overwrite the Parquet with non-Parquet bytes.
            (export_dir / "sentences.parquet").write_bytes(b"not a parquet file")

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "parquet" in str(exc.value).lower()
            # The original library exception must be preserved as __cause__.
            assert exc.value.__cause__ is not None

    def test_checksum_read_failure_wrapped_as_export_error(self, monkeypatch):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )
        from osm_polygon_sentence_relevance.output import validation as validation_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)

            def boom(_path: Path) -> str:
                raise OSError("simulated checksum read failure")

            monkeypatch.setattr(validation_mod, "sha256_file", boom)

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "checksum" in str(exc.value).lower()
            assert isinstance(exc.value.__cause__, OSError)


# ===================================================================
# Parquet schema metadata contract
# ===================================================================


class TestParquetMetadataContract:
    """Parquet schema metadata must be present, decodable, and consistent."""

    def test_missing_schema_metadata_rejected(self):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            # Rewrite Parquet with the right physical schema but no metadata.
            table = _build_output_table(n_rows=2, metadata={})
            # An empty metadata dict still strips the keys; force None.
            table = table.replace_schema_metadata(None)
            parquet_path = _rewrite_parquet(export_dir, table=table)
            _rewrite_manifest(
                export_dir,
                sha256=_checksum(parquet_path),
                row_count=table.num_rows,
            )

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "metadata" in str(exc.value).lower()

    @pytest.mark.parametrize(
        "bad_metadata",
        [
            # missing revision key entirely
            {b"pipeline_version": b"ver-7b"},
            # blank revision
            {
                b"input_dataset_revision": b"",
                b"pipeline_version": b"ver-7b",
            },
            # non-string (integer) revision stored as bytes that won't decode
            {
                b"input_dataset_revision": b"\xff\xfe\x00",
                b"pipeline_version": b"ver-7b",
            },
        ],
    )
    def test_bad_input_dataset_revision_metadata_rejected(self, bad_metadata):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            table = _build_output_table(n_rows=2, metadata=bad_metadata)
            parquet_path = _rewrite_parquet(export_dir, table=table)
            _rewrite_manifest(
                export_dir,
                sha256=_checksum(parquet_path),
                row_count=table.num_rows,
            )

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "input_dataset_revision" in str(exc.value)

    @pytest.mark.parametrize(
        "bad_metadata",
        [
            # missing version key entirely
            {b"input_dataset_revision": b"rev-7b"},
            # blank version
            {
                b"input_dataset_revision": b"rev-7b",
                b"pipeline_version": b"",
            },
            # undecodable version
            {
                b"input_dataset_revision": b"rev-7b",
                b"pipeline_version": b"\xff\xfe\x00",
            },
        ],
    )
    def test_bad_pipeline_version_metadata_rejected(self, bad_metadata):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            table = _build_output_table(n_rows=2, metadata=bad_metadata)
            parquet_path = _rewrite_parquet(export_dir, table=table)
            _rewrite_manifest(
                export_dir,
                sha256=_checksum(parquet_path),
                row_count=table.num_rows,
            )

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "pipeline_version" in str(exc.value)


class TestBoundedParquetStatistics:
    def _write(self, tmp_path: Path, table: pa.Table) -> Path:
        path = tmp_path / "sentences.parquet"
        pq.write_table(table, path)
        return path

    def test_matches_in_memory_statistics(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
        )

        table = _build_output_table(n_rows=3)
        path = self._write(tmp_path, table)
        digest = _checksum(path)
        expected = compute_statistics(
            table,
            input_dataset_revision="rev-7b",
            pipeline_version="ver-7b",
            parquet_sha256=digest,
        )
        actual = compute_parquet_statistics(
            path,
            input_dataset_revision="rev-7b",
            pipeline_version="ver-7b",
            parquet_sha256=digest,
            input_dataset_id=None,
            scratch_dir=tmp_path / "scratch",
            batch_size=1,
        )
        assert actual == expected
        assert list((tmp_path / "scratch").glob("*.sqlite3")) == []

    @pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
    def test_rejects_invalid_batch_size(
        self, tmp_path: Path, batch_size: object
    ) -> None:
        path = self._write(tmp_path, _build_output_table())
        with pytest.raises(ValueError, match="batch_size"):
            compute_parquet_statistics(
                path,
                input_dataset_revision="rev-7b",
                pipeline_version="ver-7b",
                parquet_sha256=_checksum(path),
                input_dataset_id=None,
                scratch_dir=tmp_path / "scratch",
                batch_size=batch_size,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        ("revision", "version", "message"),
        [("wrong", "ver-7b", "revision"), ("rev-7b", "wrong", "version")],
    )
    def test_rejects_metadata_identity_mismatch(
        self, tmp_path: Path, revision: str, version: str, message: str
    ) -> None:
        path = self._write(tmp_path, _build_output_table())
        with pytest.raises(ValueError, match=message):
            compute_parquet_statistics(
                path,
                input_dataset_revision=revision,
                pipeline_version=version,
                parquet_sha256=_checksum(path),
                input_dataset_id=None,
                scratch_dir=tmp_path / "scratch",
            )

    def test_rejects_duplicate_sentence_id(self, tmp_path: Path) -> None:
        table = _build_output_table(n_rows=2)
        duplicate = table.set_column(
            table.schema.get_field_index("sentence_id"),
            table.schema.field("sentence_id"),
            pa.array(["same", "same"], type=pa.string()),
        )
        duplicate = duplicate.set_column(
            duplicate.schema.get_field_index("polygon_id"),
            duplicate.schema.field("polygon_id"),
            pa.array(["a", "b"], type=pa.string()),
        )
        path = self._write(tmp_path, duplicate)
        with pytest.raises(ValueError, match="duplicate sentence_id"):
            compute_parquet_statistics(
                path,
                input_dataset_revision="rev-7b",
                pipeline_version="ver-7b",
                parquet_sha256=_checksum(path),
                input_dataset_id=None,
                scratch_dir=tmp_path / "scratch",
            )

    def test_rejects_unsorted_rows(self, tmp_path: Path) -> None:
        table = _build_output_table(n_rows=2).take(pa.array([1, 0]))
        path = self._write(tmp_path, table)
        with pytest.raises(ValueError, match="sorted"):
            compute_parquet_statistics(
                path,
                input_dataset_revision="rev-7b",
                pipeline_version="ver-7b",
                parquet_sha256=_checksum(path),
                input_dataset_id=None,
                scratch_dir=tmp_path / "scratch",
            )

    def test_parquet_revision_differs_from_manifest_rejected(self):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            table = _build_output_table(
                n_rows=2,
                metadata={
                    b"input_dataset_revision": b"rev-from-parquet",
                    b"pipeline_version": b"ver-7b",
                },
            )
            parquet_path = _rewrite_parquet(export_dir, table=table)
            _rewrite_manifest(
                export_dir,
                sha256=_checksum(parquet_path),
                row_count=table.num_rows,
                # manifest still carries the original "rev-7b"
            )

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "input_dataset_revision" in str(exc.value)

    def test_parquet_pipeline_version_differs_from_manifest_rejected(self):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            table = _build_output_table(
                n_rows=2,
                metadata={
                    b"input_dataset_revision": b"rev-7b",
                    b"pipeline_version": b"ver-from-parquet",
                },
            )
            parquet_path = _rewrite_parquet(export_dir, table=table)
            _rewrite_manifest(
                export_dir,
                sha256=_checksum(parquet_path),
                row_count=table.num_rows,
            )

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "pipeline_version" in str(exc.value)


# ===================================================================
# Parquet schema metadata whitespace contract
# ===================================================================


class TestParquetMetadataWhitespace:
    """Whitespace-only Parquet metadata values must be rejected."""

    @pytest.mark.parametrize(
        "ws_value",
        [
            b" ",
            b"\t",
            b"\n",
            b" \t\n",
        ],
        ids=["space", "tab", "newline", "mixed"],
    )
    def test_whitespace_only_input_dataset_revision_metadata_rejected(self, ws_value):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            table = _build_output_table(
                n_rows=2,
                metadata={
                    b"input_dataset_revision": ws_value,
                    b"pipeline_version": b"ver-7b",
                },
            )
            parquet_path = _rewrite_parquet(export_dir, table=table)
            _rewrite_manifest(
                export_dir,
                sha256=_checksum(parquet_path),
                row_count=table.num_rows,
                input_dataset_revision=ws_value.decode("utf-8"),
            )

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "input_dataset_revision" in str(exc.value)

    @pytest.mark.parametrize(
        "ws_value",
        [
            b" ",
            b"\t",
            b"\n",
            b" \t\n",
        ],
        ids=["space", "tab", "newline", "mixed"],
    )
    def test_whitespace_only_pipeline_version_metadata_rejected(self, ws_value):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            table = _build_output_table(
                n_rows=2,
                metadata={
                    b"input_dataset_revision": b"rev-7b",
                    b"pipeline_version": ws_value,
                },
            )
            parquet_path = _rewrite_parquet(export_dir, table=table)
            _rewrite_manifest(
                export_dir,
                sha256=_checksum(parquet_path),
                row_count=table.num_rows,
                pipeline_version=ws_value.decode("utf-8"),
            )

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "pipeline_version" in str(exc.value)


# ===================================================================
# Manifest revision / version string contract
# ===================================================================


class TestManifestRevisionVersion:
    """Manifest-side revision/version must be non-empty strings."""

    @pytest.mark.parametrize(
        "bad_value",
        [None, "", 123, 1.5, True, [], {}],
    )
    def test_manifest_input_dataset_revision_must_be_nonempty_string(self, bad_value):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            _rewrite_manifest(export_dir, input_dataset_revision=bad_value)

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "input_dataset_revision" in str(exc.value)

    @pytest.mark.parametrize(
        "bad_value",
        [None, "", 123, 1.5, True, [], {}],
    )
    def test_manifest_pipeline_version_must_be_nonempty_string(self, bad_value):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            _rewrite_manifest(export_dir, pipeline_version=bad_value)

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "pipeline_version" in str(exc.value)


# ===================================================================
# Manifest revision / version whitespace contract
# ===================================================================


class TestManifestRevisionVersionWhitespace:
    """Whitespace-only manifest values must be rejected.

    Both the Parquet metadata and the manifest carry the same whitespace
    value so the cross-check agrees; the only thing that can reject is the
    whitespace check itself in ``_require_manifest_string`` (or, for the
    Parquet side, ``_decode_meta_value``).
    """

    @pytest.mark.parametrize(
        "ws_value",
        [" ", "\t", "\n", " \t\n"],
        ids=["space", "tab", "newline", "mixed"],
    )
    def test_whitespace_only_manifest_input_dataset_revision_rejected(self, ws_value):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            # Set BOTH the Parquet metadata and the manifest to the same
            # whitespace so the cross-check cannot be what rejects it.
            table = _build_output_table(
                n_rows=2,
                metadata={
                    b"input_dataset_revision": ws_value.encode("utf-8"),
                    b"pipeline_version": b"ver-7b",
                },
            )
            parquet_path = _rewrite_parquet(export_dir, table=table)
            _rewrite_manifest(
                export_dir,
                sha256=_checksum(parquet_path),
                row_count=table.num_rows,
                input_dataset_revision=ws_value,
            )

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "input_dataset_revision" in str(exc.value)

    @pytest.mark.parametrize(
        "ws_value",
        [" ", "\t", "\n", " \t\n"],
        ids=["space", "tab", "newline", "mixed"],
    )
    def test_whitespace_only_manifest_pipeline_version_rejected(self, ws_value):
        from osm_polygon_sentence_relevance.output import validate_export_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _make_valid_export(tmpdir)
            table = _build_output_table(
                n_rows=2,
                metadata={
                    b"input_dataset_revision": b"rev-7b",
                    b"pipeline_version": ws_value.encode("utf-8"),
                },
            )
            parquet_path = _rewrite_parquet(export_dir, table=table)
            _rewrite_manifest(
                export_dir,
                sha256=_checksum(parquet_path),
                row_count=table.num_rows,
                pipeline_version=ws_value,
            )

            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "pipeline_version" in str(exc.value)


# ===================================================================
# Coverage backfill (Phase 9N): narrow defensive-branch tests that
# are easier to express at module scope than inside the existing
# nested test classes. Each test asserts one branch the previous
# suite did not exercise.
# ===================================================================


class TestCoverageBackfill:
    """Targeted tests for branches the existing suite does not yet exercise.

    These exist only to keep the full repository coverage gate at or
    above 95%. They are deliberately narrow: each test asserts one
    defensive branch in the validator or the card/manifest layer.
    """

    def test_manifest_unreadable_oserror_wrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from osm_polygon_sentence_relevance.output import validate_export_directory

        export_dir = _make_valid_export(tmp_path)
        manifest_path = export_dir / "manifest.json"
        original_read_text = Path.read_text

        def trip(self: Path, *args: object, **kwargs: object) -> str:
            if self == manifest_path:
                raise OSError("simulated read failure")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", trip)
        with pytest.raises(ExportError, match="not readable"):
            validate_export_directory(export_dir)

    def test_manifest_is_not_json_object(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.output import validate_export_directory

        export_dir = _make_valid_export(tmp_path)
        (export_dir / "manifest.json").write_text("[1, 2, 3]\n", encoding="utf-8")
        with pytest.raises(ExportError, match="JSON object"):
            validate_export_directory(export_dir)

    def test_manifest_sha256_not_a_string(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.output import validate_export_directory

        export_dir = _make_valid_export(tmp_path)
        _rewrite_manifest(export_dir, sha256=12345)
        with pytest.raises(ExportError, match="sha256"):
            validate_export_directory(export_dir)

    def test_manifest_top_level_row_count_drift_detected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.output import validate_export_directory

        export_dir = _make_valid_export(tmp_path)
        _rewrite_manifest(export_dir, row_count=0)
        with pytest.raises(ExportError, match="row_count"):
            validate_export_directory(export_dir)

    def test_manifest_input_dataset_id_blank_string(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.output import validate_export_directory

        export_dir = _make_valid_export(tmp_path)
        _rewrite_manifest(export_dir, input_dataset_id="   ")
        with pytest.raises(ExportError, match="input_dataset_id"):
            validate_export_directory(export_dir)

    def test_manifest_input_dataset_id_with_surrounding_whitespace(
        self, tmp_path: Path
    ) -> None:
        from osm_polygon_sentence_relevance.output import validate_export_directory

        export_dir = _make_valid_export(tmp_path)
        _rewrite_manifest(export_dir, input_dataset_id="  owner/dataset  ")
        with pytest.raises(ExportError, match="input_dataset_id"):
            validate_export_directory(export_dir)

    def test_card_file_unreadable_wrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from osm_polygon_sentence_relevance.output import validate_export_directory

        export_dir = _make_valid_export(tmp_path)
        card_path = export_dir / "README.md"
        original_read_text = Path.read_text

        def trip(self: Path, *args: object, **kwargs: object) -> str:
            if self == card_path:
                raise OSError("simulated card read failure")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", trip)
        with pytest.raises(ExportError, match="Dataset card"):
            validate_export_directory(export_dir)


# ===================================================================
# Coverage backfill (Phase 9N, second pass): the few remaining
# defensive branches after the first backfill pass.
# ===================================================================


class TestCoverageBackfillSecondPass:
    """Round out the validator coverage gate by exercising the remaining
    defensive branches: parquet-read OSError wrapping, ``pq.ParquetFile``
    construction failure, manifest row_count bool-rejection, and the
    bounded-statistics failure path inside ``validate_export_directory``.
    """

    def test_manifest_row_count_bool_rejected(self, tmp_path: Path) -> None:
        from osm_polygon_sentence_relevance.output import validate_export_directory

        export_dir = _make_valid_export(tmp_path)
        _rewrite_manifest(export_dir, row_count=True)
        with pytest.raises(ExportError, match="row_count"):
            validate_export_directory(export_dir)

    def test_parquet_file_open_failure_wrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from osm_polygon_sentence_relevance.output import validate_export_directory
        from osm_polygon_sentence_relevance.output import validation as validation_mod

        export_dir = _make_valid_export(tmp_path)

        class _BoomMeta:
            metadata = type(
                "m",
                (),
                {"num_rows": 2},
            )()

            def __eq__(self, _other: object) -> bool:
                raise RuntimeError("simulated schema comparison failure")

        class _BoomParquetFile:
            schema_arrow = object()

            @property
            def metadata(self) -> object:  # pragma: no cover - never reached
                return None

        monkeypatch.setattr(
            validation_mod.pq, "ParquetFile", lambda *a, **k: _BoomParquetFile()
        )
        with pytest.raises(ExportError, match="could not be read"):
            validate_export_directory(export_dir)

    def test_parquet_read_iteration_failure_wrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from osm_polygon_sentence_relevance.output import validate_export_directory
        from osm_polygon_sentence_relevance.output import validation as validation_mod

        export_dir = _make_valid_export(tmp_path)

        # Force ``compute_parquet_statistics`` to raise a non-ExportError
        # exception. The validator must wrap it.
        def boom(**_kw: object) -> object:
            raise RuntimeError("simulated bounded-statistics failure")

        monkeypatch.setattr(validation_mod, "compute_parquet_statistics", boom)
        with pytest.raises(ExportError, match="Could not validate"):
            validate_export_directory(export_dir)
