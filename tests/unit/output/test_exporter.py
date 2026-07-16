from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.errors import ExportError
from osm_polygon_sentence_relevance.exporter import (
    ExportResult,
    export_finalized_dataset,
)
from osm_polygon_sentence_relevance.finalization import (
    FinalizedDataset,
    finalize_sentence_dataset,
)
from osm_polygon_sentence_relevance.schemas import (
    OUTPUT_SENTENCE_SCHEMA,
    SEGMENTED_SENTENCES_SCHEMA,
)
from tests.helpers import get_checksum, make_segmented_row

# ===================================================================
# Helpers to construct SEGMENTED_SENTENCES_SCHEMA tables
# ===================================================================


def rows_to_table(rows: list[dict]) -> pa.Table:
    if not rows:
        return SEGMENTED_SENTENCES_SCHEMA.empty_table()
    data = {}
    for field in SEGMENTED_SENTENCES_SCHEMA:
        col_values = [r[field.name] for r in rows]
        data[field.name] = pa.array(col_values, type=field.type)
    return pa.table(data, schema=SEGMENTED_SENTENCES_SCHEMA)


# ===================================================================
# Test Suite for Phase 5A Exporter
# ===================================================================


class TestExporter:
    def test_reject_non_finalized_dataset(self):
        with pytest.raises(TypeError):
            # non-FinalizedDataset input
            export_finalized_dataset("not-a-dataset", "/tmp")

    def test_reject_table_schema_mismatch(self):
        # We pass a FinalizedDataset but the table is just SEGMENTED_SENTENCES_SCHEMA (not OUTPUT_SENTENCE_SCHEMA)
        table = rows_to_table([make_segmented_row()])
        dataset = FinalizedDataset(table=table, report=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ExportError) as exc:
                export_finalized_dataset(dataset, tmpdir)
            assert "schema" in str(exc.value).lower()

    def test_reject_inconsistent_revision_version(self):
        # We manually modify columns in a conforming table to have different values
        row = make_segmented_row()
        table = rows_to_table([row])
        finalize_sentence_dataset(
            table, input_dataset_revision="r1", pipeline_version="v1"
        )

        # Modify the table to have inconsistent revisions across two rows
        row1 = make_segmented_row(sentence_text_normalized="dup1")
        row2 = make_segmented_row(sentence_text_normalized="dup2")
        table_inconsistent = rows_to_table([row1, row2])
        # We finalize them with "r1" and "v1" but then manually hack the Arrow table
        dataset_inc = finalize_sentence_dataset(
            table_inconsistent, input_dataset_revision="r1", pipeline_version="v1"
        )

        # Replace the input_dataset_revision column with two different values
        arr = pa.array(["r1", "r2"])
        bad_table = dataset_inc.table.set_column(
            dataset_inc.table.schema.get_field_index("input_dataset_revision"),
            "input_dataset_revision",
            arr,
        ).cast(OUTPUT_SENTENCE_SCHEMA)
        bad_dataset = FinalizedDataset(table=bad_table, report=dataset_inc.report)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ExportError) as exc:
                export_finalized_dataset(bad_dataset, tmpdir)
            assert "inconsistent" in str(exc.value).lower()

    def test_successful_write_read_round_trip(self):
        row1 = make_segmented_row(sentence_text_normalized="one")
        row2 = make_segmented_row(sentence_text_normalized="two")
        table = rows_to_table([row1, row2])
        dataset = finalize_sentence_dataset(
            table, input_dataset_revision="rev-1", pipeline_version="ver-1"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            res = export_finalized_dataset(dataset, tmpdir)

            assert isinstance(res, ExportResult)
            assert res.parquet_path.name == "sentences.parquet"
            assert res.manifest_path.name == "manifest.json"

            # Read back Parquet and verify it equals dataset table
            loaded_table = pq.read_table(res.parquet_path)
            assert loaded_table.equals(dataset.table)

    def test_exact_schema_preservation(self):
        row = make_segmented_row()
        dataset = finalize_sentence_dataset(
            rows_to_table([row]), input_dataset_revision="r1", pipeline_version="v1"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            res = export_finalized_dataset(dataset, tmpdir)
            loaded_table = pq.read_table(res.parquet_path)

            # Schema must match exactly (field order, type, nullability)
            assert loaded_table.schema.equals(OUTPUT_SENTENCE_SCHEMA)
            assert loaded_table.column_names == list(OUTPUT_SENTENCE_SCHEMA.names)

    def test_correct_manifest_aggregation(self):
        # 3 input rows, 1 duplicate collapses
        # - Row 1: wikipedia, en, reg-1
        # - Row 2: wikipedia, en, reg-1 (duplicate of Row 1)
        # - Row 3: wikivoyage, fr, reg-2
        rows = [
            make_segmented_row(
                source="wikipedia",
                language="en",
                region="reg-1",
                sentence_text_normalized="dup",
            ),
            make_segmented_row(
                source="wikipedia",
                language="en",
                region="reg-1",
                sentence_text_normalized="dup",
            ),
            make_segmented_row(
                source="wikivoyage",
                language="fr",
                region="reg-2",
                sentence_text_normalized="other",
            ),
        ]
        dataset = finalize_sentence_dataset(
            rows_to_table(rows),
            input_dataset_revision="rev-xyz",
            pipeline_version="ver-123",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            res = export_finalized_dataset(dataset, tmpdir)
            manifest = res.manifest_data

            assert manifest["row_count"] == 2
            assert manifest["input_occurrence_count"] == 3
            assert manifest["duplicates_removed"] == 1
            assert manifest["cross_source_duplicate_groups"] == 0

            assert manifest["counts_by_source"] == {"wikipedia": 1, "wikivoyage": 1}
            assert manifest["counts_by_language"] == {"en": 1, "fr": 1}
            assert manifest["counts_by_region"] == {"reg-1": 1, "reg-2": 1}

            assert manifest["input_dataset_revision"] == "rev-xyz"
            assert manifest["pipeline_version"] == "ver-123"

            # Checksum matches the physical file
            actual_checksum = get_checksum(res.parquet_path)
            assert manifest["sha256"] == actual_checksum

    def test_deterministic_manifest_across_repeated_writes(self):
        row = make_segmented_row()
        dataset = finalize_sentence_dataset(
            rows_to_table([row]), input_dataset_revision="r1", pipeline_version="v1"
        )

        with (
            tempfile.TemporaryDirectory() as tmpdir_a,
            tempfile.TemporaryDirectory() as tmpdir_b,
        ):
            res_a = export_finalized_dataset(dataset, tmpdir_a)
            res_b = export_finalized_dataset(dataset, tmpdir_b)

            # Read files as raw text and compare character-for-character
            with open(res_a.manifest_path, encoding="utf-8") as fa:
                content_a = fa.read()
            with open(res_b.manifest_path, encoding="utf-8") as fb:
                content_b = fb.read()

            assert content_a == content_b
            # Stable compact serialization (keys sorted, no space separators, ends with newline)
            assert " " not in content_a
            assert content_a.endswith("\n")

    def test_empty_finalized_dataset(self):
        table = rows_to_table([])
        dataset = finalize_sentence_dataset(
            table, input_dataset_revision="rev-empty", pipeline_version="ver-empty"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            res = export_finalized_dataset(dataset, tmpdir)

            loaded_table = pq.read_table(res.parquet_path)
            assert loaded_table.num_rows == 0
            assert loaded_table.schema.equals(OUTPUT_SENTENCE_SCHEMA)

            manifest = res.manifest_data
            assert manifest["row_count"] == 0
            assert manifest["input_occurrence_count"] == 0
            assert manifest["duplicates_removed"] == 0
            assert manifest["cross_source_duplicate_groups"] == 0
            assert manifest["counts_by_source"] == {}
            assert manifest["counts_by_language"] == {}
            assert manifest["counts_by_region"] == {}
            assert manifest["input_dataset_revision"] == "rev-empty"
            assert manifest["pipeline_version"] == "ver-empty"

    def test_overwrite_policy(self):
        row = make_segmented_row()
        dataset = finalize_sentence_dataset(
            rows_to_table([row]), input_dataset_revision="r1", pipeline_version="v1"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # First write
            export_finalized_dataset(dataset, tmpdir)

            # Second write without overwrite=True must raise ExportError
            with pytest.raises(ExportError) as exc:
                export_finalized_dataset(dataset, tmpdir, overwrite=False)
            assert "exists" in str(exc.value).lower()

            # Second write with overwrite=True succeeds
            res = export_finalized_dataset(dataset, tmpdir, overwrite=True)
            assert res.parquet_path.exists()

    def test_simulated_write_failure_leaves_no_partial_output(self, monkeypatch):
        row = make_segmented_row()
        dataset = finalize_sentence_dataset(
            rows_to_table([row]), input_dataset_revision="r1", pipeline_version="v1"
        )

        def mock_write_table(*args, **kwargs):
            raise OSError("Disk full or connection lost")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            monkeypatch.setattr(pq, "write_table", mock_write_table)

            with pytest.raises(IOError, match="Disk full or connection lost"):
                export_finalized_dataset(dataset, output_dir)

            # Verify no partial directory or files remain
            assert not output_dir.exists()

    def test_input_remains_unchanged(self):
        row = make_segmented_row()
        dataset = finalize_sentence_dataset(
            rows_to_table([row]), input_dataset_revision="r1", pipeline_version="v1"
        )

        orig_num_rows = dataset.table.num_rows
        orig_schema = dataset.table.schema
        orig_report = dataset.report

        with tempfile.TemporaryDirectory() as tmpdir:
            export_finalized_dataset(dataset, tmpdir)

            assert dataset.table.num_rows == orig_num_rows
            assert dataset.table.schema.equals(orig_schema)
            assert dataset.report == orig_report

    def test_failed_replacement_preserves_previous_output(self, monkeypatch):
        # 1. Setup a valid previous export
        row_prev = make_segmented_row(sentence_text_raw="Previous raw text")
        dataset_prev = finalize_sentence_dataset(
            rows_to_table([row_prev]),
            input_dataset_revision="r1",
            pipeline_version="v1",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            export_finalized_dataset(dataset_prev, output_dir)

            # Calculate and save previous checksum and contents
            prev_pq = output_dir / "sentences.parquet"
            prev_checksum = get_checksum(prev_pq)
            prev_table = pq.read_table(prev_pq)

            # 2. Try to export a new dataset but simulate failure during final rename
            row_new = make_segmented_row(sentence_text_raw="New raw text")
            dataset_new = finalize_sentence_dataset(
                rows_to_table([row_new]),
                input_dataset_revision="r1",
                pipeline_version="v1",
            )

            orig_rename = os.rename

            def mock_rename(src, dst):
                if (
                    Path(dst).resolve() == output_dir.resolve()
                    and (Path(src) / "sentences.parquet").exists()
                    and not Path(src).name.startswith(".backup_")
                ):
                    raise OSError("Simulated atomic rename failure")
                return orig_rename(src, dst)

            monkeypatch.setattr(os, "rename", mock_rename)

            with pytest.raises(OSError, match="Simulated atomic rename failure"):
                export_finalized_dataset(dataset_new, output_dir, overwrite=True)

            # 3. Assert previous output is completely unchanged
            assert output_dir.exists()
            assert get_checksum(prev_pq) == prev_checksum
            assert pq.read_table(prev_pq).equals(prev_table)

    def test_no_temporary_or_backup_directories_remain(self, monkeypatch):
        row = make_segmented_row()
        dataset = finalize_sentence_dataset(
            rows_to_table([row]), input_dataset_revision="r1", pipeline_version="v1"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)
            output_dir = parent_dir / "output"

            # 1. Success case
            export_finalized_dataset(dataset, output_dir)
            # Parent dir should only contain "output"
            parent_contents = os.listdir(parent_dir)
            assert len(parent_contents) == 1
            assert parent_contents[0] == "output"

            # 2. Handled failure case during overwrite
            # Mock os.rename to fail during replacement
            orig_rename = os.rename

            def mock_rename(src, dst):
                if (
                    Path(dst).resolve() == output_dir.resolve()
                    and (Path(src) / "sentences.parquet").exists()
                    and not Path(src).name.startswith(".backup_")
                ):
                    raise OSError("Simulated rename failure")
                return orig_rename(src, dst)

            monkeypatch.setattr(os, "rename", mock_rename)
            with pytest.raises(OSError, match="Simulated rename failure"):
                export_finalized_dataset(dataset, output_dir, overwrite=True)

            # Parent dir should still only contain "output" (no leftovers)
            parent_contents = os.listdir(parent_dir)
            assert len(parent_contents) == 1
            assert parent_contents[0] == "output"

    def test_new_output_write_failure_leaves_no_output_dir(self, monkeypatch):
        row = make_segmented_row()
        dataset = finalize_sentence_dataset(
            rows_to_table([row]), input_dataset_revision="r1", pipeline_version="v1"
        )

        # Mock pq.write_table to simulate write failure
        def mock_write_table(*args, **kwargs):
            raise OSError("Disk write failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            parent_dir = Path(tmpdir)
            output_dir = parent_dir / "output"
            monkeypatch.setattr(pq, "write_table", mock_write_table)

            with pytest.raises(IOError, match="Disk write failed"):
                export_finalized_dataset(dataset, output_dir)

            # No output_dir and no temp directories left in parent_dir
            assert not output_dir.exists()
            assert len(os.listdir(parent_dir)) == 0

    def test_failed_replacement_and_failed_restoration_preserves_backup(
        self, monkeypatch
    ):
        row_prev = make_segmented_row(sentence_text_raw="Previous raw text")
        dataset_prev = finalize_sentence_dataset(
            rows_to_table([row_prev]),
            input_dataset_revision="r1",
            pipeline_version="v1",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            export_finalized_dataset(dataset_prev, output_dir)

            row_new = make_segmented_row(sentence_text_raw="New raw text")
            dataset_new = finalize_sentence_dataset(
                rows_to_table([row_new]),
                input_dataset_revision="r1",
                pipeline_version="v1",
            )

            orig_rename = os.rename

            def mock_rename(src, dst):
                if Path(dst).resolve() == output_dir.resolve():
                    raise OSError("Simulated rename failure")
                return orig_rename(src, dst)

            monkeypatch.setattr(os, "rename", mock_rename)

            with pytest.raises(ExportError) as exc:
                export_finalized_dataset(dataset_new, output_dir, overwrite=True)

            assert "Atomic replacement failed" in str(exc.value)
            assert ".backup_" in str(exc.value)

            parent_dir = output_dir.parent
            backup_folders = [
                parent_dir / name
                for name in os.listdir(parent_dir)
                if name.startswith(".backup_")
            ]
            assert len(backup_folders) == 1
            backup_dir = backup_folders[0]
            assert backup_dir.exists()
            assert (backup_dir / "sentences.parquet").exists()

    def test_tmp_dir_creation_failure_no_attribute_error(self, monkeypatch):
        row = make_segmented_row()
        dataset = finalize_sentence_dataset(
            rows_to_table([row]), input_dataset_revision="r1", pipeline_version="v1"
        )

        def mock_mkdtemp(*args, **kwargs):
            raise OSError("mkdtemp failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            export_finalized_dataset(dataset, output_dir)
            prev_pq = output_dir / "sentences.parquet"
            prev_checksum = get_checksum(prev_pq)

            monkeypatch.setattr(tempfile, "mkdtemp", mock_mkdtemp)

            with pytest.raises(OSError, match="mkdtemp failed"):
                export_finalized_dataset(dataset, output_dir, overwrite=True)

            assert output_dir.exists()
            assert get_checksum(prev_pq) == prev_checksum

    def test_existing_non_directory_target_rejected(self):
        row = make_segmented_row()
        dataset = finalize_sentence_dataset(
            rows_to_table([row]), input_dataset_revision="r1", pipeline_version="v1"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "file.txt"
            target_path.write_text("hello")

            with pytest.raises(ExportError) as exc:
                export_finalized_dataset(dataset, target_path, overwrite=True)
            assert "not a directory" in str(exc.value).lower()

    def test_failed_backup_deletion_does_not_rollback(self, monkeypatch):
        row_prev = make_segmented_row(sentence_text_raw="Previous text")
        dataset_prev = finalize_sentence_dataset(
            rows_to_table([row_prev]),
            input_dataset_revision="r1",
            pipeline_version="v1",
        )

        row_new = make_segmented_row(sentence_text_raw="New text")
        dataset_new = finalize_sentence_dataset(
            rows_to_table([row_new]), input_dataset_revision="r1", pipeline_version="v1"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            export_finalized_dataset(dataset_prev, output_dir)

            orig_rmtree = shutil.rmtree

            def mock_rmtree(path, *args, **kwargs):
                if Path(path).name.startswith(".backup_"):
                    raise OSError("Simulated backup deletion failure")
                return orig_rmtree(path, *args, **kwargs)

            monkeypatch.setattr(shutil, "rmtree", mock_rmtree)

            with pytest.raises(ExportError) as exc:
                export_finalized_dataset(dataset_new, output_dir, overwrite=True)
            assert "failed to delete backup directory" in str(exc.value).lower()

            assert output_dir.exists()
            loaded_table = pq.read_table(output_dir / "sentences.parquet")
            assert loaded_table.column("sentence_text_raw").to_pylist() == ["New text"]
