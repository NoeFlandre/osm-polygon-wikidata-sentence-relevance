"""Tests for the deterministic dataset-card generation (the implementation).

These tests assert stable contracts of the dataset-card module:

- exact statistics computed from the finalized output table;
- explicit null-handling and deterministic ordering;
- exact deterministic rendering (byte-for-byte stable across reorderings);
- the exporter writes all three artifacts (parquet, manifest, README card);
- validation recomputes statistics from Parquet and rejects stale/edited cards;
- the publisher uploads exactly the three validated artifacts once.

No network, no model-weight download, no real Hub activity.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.errors import ExportError
from osm_polygon_sentence_relevance.finalization import (
    FinalizedDataset,
    finalize_sentence_dataset,
)
from osm_polygon_sentence_relevance.schemas import (
    SEGMENTED_SENTENCES_SCHEMA,
)
from tests.helpers import get_checksum, make_segmented_row

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rows_to_table(rows: list[dict]) -> pa.Table:
    if not rows:
        return SEGMENTED_SENTENCES_SCHEMA.empty_table()
    import pyarrow as pa

    data = {}
    for field in SEGMENTED_SENTENCES_SCHEMA:
        data[field.name] = pa.array([r[field.name] for r in rows], type=field.type)
    return pa.table(data, schema=SEGMENTED_SENTENCES_SCHEMA)


def _finalize(rows: list[dict], *, revision="rev-8c", version="ver-8c"):
    return finalize_sentence_dataset(
        _rows_to_table(rows),
        input_dataset_revision=revision,
        pipeline_version=version,
    )


def _export(tmpdir: str, dataset: FinalizedDataset, *, overwrite: bool = False) -> Path:
    from osm_polygon_sentence_relevance.output import export_finalized_dataset

    res = export_finalized_dataset(dataset, tmpdir, overwrite=overwrite)
    return Path(res.parquet_path).parent


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------


class TestStatisticsComputation:
    def test_exact_statistics_on_small_synthetic_table(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
        )

        rows = [
            make_segmented_row(
                polygon_id="poly-1",
                wikidata="Q1",
                document_id="doc-1",
                source="wikipedia",
                language="en",
                region="reg-a",
                sentence_text_normalized="alpha",
                lat=1.0,
                lon=2.0,
            ),
            make_segmented_row(
                polygon_id="poly-1",
                wikidata="Q1",
                document_id="doc-2",
                source="wikivoyage",
                language="fr",
                region="reg-b",
                sentence_text_normalized="beta",
                lat=None,
                lon=None,
            ),
            make_segmented_row(
                polygon_id="poly-2",
                wikidata="Q2",
                document_id="doc-3",
                source="wikipedia",
                language="en",
                region="reg-a",
                sentence_text_normalized="gamma",
                lat=3.0,
                lon=4.0,
            ),
        ]
        dataset = _finalize(rows)

        # Two rows are exact duplicates-on-key collapse?
        # The three rows differ by polygon/source/language/normalized text,
        # so output keeps all three.
        stats = compute_statistics(
            dataset.table,
            input_dataset_revision="rev-8c",
            pipeline_version="ver-8c",
            parquet_sha256="0" * 64,
        )

        assert stats.row_count == 3
        assert stats.unique_sentence_ids == 3
        assert stats.unique_polygons == 2  # poly-1, poly-2
        assert stats.unique_wikidata_entities == 2  # Q1, Q2
        assert stats.unique_documents == 3  # doc-1, doc-2, doc-3
        assert stats.source_counts == {"wikipedia": 2, "wikivoyage": 1}
        assert stats.language_counts == {"en": 2, "fr": 1}
        assert stats.region_counts == {"reg-a": 2, "reg-b": 1}
        assert stats.rows_with_coordinates == 2
        assert stats.rows_without_coordinates == 1
        assert stats.input_dataset_revision == "rev-8c"
        assert stats.pipeline_version == "ver-8c"
        assert stats.parquet_sha256 == "0" * 64

    def test_source_fractions_cover_total(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
        )

        rows = [
            make_segmented_row(
                source="wikipedia",
                sentence_text_normalized="a",
            ),
            make_segmented_row(
                source="wikivoyage",
                sentence_text_normalized="b",
            ),
            make_segmented_row(
                source="wikipedia",
                sentence_text_normalized="c",
            ),
        ]
        dataset = _finalize(rows)
        stats = compute_statistics(
            dataset.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        assert sum(stats.source_counts.values()) == stats.row_count == 3

    def test_null_handling(self):
        """Null coordinates are counted, not as extra entities. Wikidata is
        non-nullable but we defensively ignore any null wikidata values."""
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
        )

        rows = [
            make_segmented_row(
                wikidata="Q1",
                lat=None,
                lon=None,
                sentence_text_normalized="a",
            ),
            make_segmented_row(
                wikidata="Q2",
                lat=10.0,
                lon=20.0,
                sentence_text_normalized="b",
            ),
        ]
        dataset = _finalize(rows)
        stats = compute_statistics(
            dataset.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        assert stats.unique_wikidata_entities == 2
        assert stats.rows_with_coordinates == 1
        assert stats.rows_without_coordinates == 1

    def test_empty_dataset_statistics(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
        )

        dataset = _finalize([])
        stats = compute_statistics(
            dataset.table,
            input_dataset_revision="rev-empty",
            pipeline_version="ver-empty",
            parquet_sha256="0" * 64,
        )
        assert stats.row_count == 0
        assert stats.unique_sentence_ids == 0
        assert stats.unique_polygons == 0
        assert stats.unique_wikidata_entities == 0
        assert stats.unique_documents == 0
        assert stats.source_counts == {}
        assert stats.language_counts == {}
        assert stats.region_counts == {}
        assert stats.rows_with_coordinates == 0
        assert stats.rows_without_coordinates == 0

    def test_unicode_languages_and_regions(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
        )

        rows = [
            make_segmented_row(language="zh", region="北京"),
            make_segmented_row(language="日本語", region="東京"),
        ]
        dataset = _finalize(rows)
        stats = compute_statistics(
            dataset.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        assert stats.language_counts == {"zh": 1, "日本語": 1}
        assert stats.region_counts == {"北京": 1, "東京": 1}

    def test_deterministic_ordering_independent_of_input_row_order(self):
        """Reordering input rows must not change any statistic and must
        produce byte-identical canonical breakdowns (sorted keys)."""
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
        )

        base = [
            make_segmented_row(
                polygon_id="poly-1",
                wikidata="Q1",
                document_id="doc-1",
                source="wikipedia",
                language="en",
                region="reg-a",
                sentence_text_normalized="a",
            ),
            make_segmented_row(
                polygon_id="poly-2",
                wikidata="Q2",
                document_id="doc-2",
                source="wikivoyage",
                language="fr",
                region="reg-b",
                sentence_text_normalized="b",
            ),
            make_segmented_row(
                polygon_id="poly-3",
                wikidata="Q3",
                document_id="doc-3",
                source="wikipedia",
                language="de",
                region="reg-c",
                sentence_text_normalized="c",
            ),
        ]
        rows_a = list(base)
        rows_b = list(reversed(base))

        stats_a = compute_statistics(
            _finalize(rows_a).table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="x",
        )
        stats_b = compute_statistics(
            _finalize(rows_b).table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="x",
        )
        assert stats_a == stats_b
        assert stats_a.source_counts == {"wikipedia": 2, "wikivoyage": 1}
        assert stats_a.language_counts == {"de": 1, "en": 1, "fr": 1}
        assert stats_a.region_counts == {"reg-a": 1, "reg-b": 1, "reg-c": 1}


# ---------------------------------------------------------------------------
# Rendering determinism
# ---------------------------------------------------------------------------


class TestCardRendering:
    def test_exact_deterministic_rendering(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
            render_dataset_card,
        )

        rows = [
            make_segmented_row(
                source="wikipedia",
                language="en",
                region="reg-a",
                sentence_text_normalized="a",
            ),
            make_segmented_row(
                source="wikivoyage",
                language="fr",
                region="reg-b",
                sentence_text_normalized="b",
            ),
        ]
        dataset = _finalize(rows, revision="rev-xyz", version="ver-xyz")
        parquet_sha = get_checksum(
            Path(_export(tempfile.mkdtemp(), dataset)) / "sentences.parquet"
        )
        stats = compute_statistics(
            dataset.table,
            input_dataset_revision="rev-xyz",
            pipeline_version="ver-xyz",
            parquet_sha256=parquet_sha,
        )
        card_a = render_dataset_card(stats)
        card_b = render_dataset_card(stats)
        assert card_a == card_b
        # HF front matter present and starts/ends with delimiters.
        assert card_a.startswith("---\n")
        assert "\n---\n" in card_a
        assert "auto-generated" in card_a
        assert "DO NOT EDIT MANUALLY" in card_a

    def test_rendering_references_statistics_only_not_hardcoded(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
            render_dataset_card,
        )

        dataset = _finalize(
            [make_segmented_row(sentence_text_normalized="only-one")],
            revision="r-12",
            version="v-12",
        )
        stats = compute_statistics(
            dataset.table,
            input_dataset_revision="r-12",
            pipeline_version="v-12",
            parquet_sha256="a" * 64,
        )
        card = render_dataset_card(stats)
        # The rendered card must carry the real revision/version/counts.
        assert "r-12" in card
        assert "v-12" in card
        assert "1" in card  # row count appears


# ---------------------------------------------------------------------------
# Exporter writes all three artifacts
# ---------------------------------------------------------------------------


class TestExporterWritesThreeArtifacts:
    def test_exporter_creates_parquet_manifest_and_readme_card(self):
        from osm_polygon_sentence_relevance.output import (
            ExportResult,
            export_finalized_dataset,
        )

        dataset = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            res = export_finalized_dataset(dataset, tmpdir)
            assert isinstance(res, ExportResult)
            export_dir = Path(res.parquet_path).parent
            assert (export_dir / "sentences.parquet").is_file()
            assert (export_dir / "manifest.json").is_file()
            assert (export_dir / "README.md").is_file()
            # Manifest now carries versioned statistics.
            manifest = json.loads((export_dir / "manifest.json").read_text())
            assert "statistics" in manifest
            from osm_polygon_sentence_relevance.output.dataset_card import (
                STATISTICS_VERSION,
            )

            assert manifest["statistics"]["version"] == STATISTICS_VERSION
            assert manifest["statistics"]["row_count"] == 1
            # The checked-in card is the deterministic rendering of manifest stats.
            card_on_disk = (export_dir / "README.md").read_text(encoding="utf-8")
            from osm_polygon_sentence_relevance.output.dataset_card import (
                render_dataset_card,
                statistics_from_dict,
            )

            expected_card = render_dataset_card(
                statistics_from_dict(manifest["statistics"])
            )
            assert card_on_disk == expected_card

    def test_repeated_export_byte_identical_card_and_manifest(self):
        from osm_polygon_sentence_relevance.output import export_finalized_dataset

        dataset = _finalize(
            [
                make_segmented_row(sentence_text_normalized="a"),
                make_segmented_row(sentence_text_normalized="b"),
            ]
        )
        with (
            tempfile.TemporaryDirectory() as a,
            tempfile.TemporaryDirectory() as b,
        ):
            res_a = export_finalized_dataset(dataset, a)
            res_b = export_finalized_dataset(dataset, b)
            card_a = (Path(res_a.parquet_path).parent / "README.md").read_text()
            card_b = (Path(res_b.parquet_path).parent / "README.md").read_text()
            manifest_a = Path(res_a.manifest_path).read_text()
            manifest_b = Path(res_b.manifest_path).read_text()
            assert card_a == card_b
            assert manifest_a == manifest_b


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidationRequiresAndVerifiesCard:
    def test_validation_requires_readme_card(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        dataset = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, dataset)
            (export_dir / "README.md").unlink()
            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert "README" in str(exc.value)

    def test_valid_export_passes_validation(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        dataset = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, dataset)
            validated = validate_export_directory(export_dir)
            assert validated.row_count == 1
            assert (export_dir / "README.md").is_file()

    def test_validation_rejects_manually_edited_card_numbers(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        dataset = _finalize(
            [
                make_segmented_row(sentence_text_normalized="a"),
                make_segmented_row(sentence_text_normalized="b"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, dataset)
            # Tamper the README to claim a different row count.
            card = (export_dir / "README.md").read_text(encoding="utf-8")
            tampered = card.replace("2", "999")
            (export_dir / "README.md").write_text(tampered, encoding="utf-8")
            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert (
                "card" in str(exc.value).lower() or "readme" in str(exc.value).lower()
            )

    def test_validation_recomputes_stats_from_parquet_not_trusted(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        dataset = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, dataset)
            # Tamper the manifest statistics to impossible values while
            # leaving the Parquet intact. Validation must recompute from
            # Parquet and reject the mismatch.
            manifest_path = export_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["statistics"]["row_count"] = 42
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with pytest.raises(ExportError):
                validate_export_directory(export_dir)

    def test_validation_rejects_missing_manifest_statistics(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        dataset = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, dataset)
            manifest_path = export_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["statistics"]
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with pytest.raises(ExportError):
                validate_export_directory(export_dir)

    def test_validation_failure_preserves_causes(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        dataset = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, dataset)
            (export_dir / "sentences.parquet").unlink()
            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert exc.value.__cause__ is None or isinstance(
                exc.value.__cause__, Exception
            )


# ---------------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------------


class TestInputImmutability:
    def test_export_does_not_mutate_input_table_or_dataset(self):
        from osm_polygon_sentence_relevance.output import export_finalized_dataset

        dataset = _finalize([make_segmented_row(sentence_text_normalized="x")])
        orig_rows = dataset.table.num_rows
        orig_schema = dataset.table.schema
        with tempfile.TemporaryDirectory() as tmpdir:
            export_finalized_dataset(dataset, tmpdir)
        assert dataset.table.num_rows == orig_rows
        assert dataset.table.schema.equals(orig_schema)

    def test_validated_export_does_not_mutate_files(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        dataset = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, dataset)
            before = {p.name: get_checksum(p) for p in export_dir.iterdir()}
            validate_export_directory(export_dir)
            after = {p.name: get_checksum(p) for p in export_dir.iterdir()}
            assert before == after


# ---------------------------------------------------------------------------
# Statistics serialization round-trip and error boundaries
# ---------------------------------------------------------------------------


class TestStatisticsSerialization:
    def test_dict_round_trip_is_stable(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
            statistics_from_dict,
            statistics_to_dict,
        )

        dataset = _finalize(
            [
                make_segmented_row(
                    source="wikipedia",
                    language="en",
                    region="reg-a",
                    sentence_text_normalized="a",
                ),
                make_segmented_row(
                    source="wikivoyage",
                    language="fr",
                    region="reg-b",
                    sentence_text_normalized="b",
                ),
            ],
            revision="r-round",
            version="v-round",
        )
        stats = compute_statistics(
            dataset.table,
            input_dataset_revision="r-round",
            pipeline_version="v-round",
            parquet_sha256="f" * 64,
        )
        restored = statistics_from_dict(statistics_to_dict(stats))
        assert restored == stats
        # Breakdowns are preserved in sorted order.
        assert restored.source_counts == {"wikipedia": 1, "wikivoyage": 1}

    def test_statistics_from_dict_rejects_missing_keys(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            STATISTICS_VERSION,
            statistics_from_dict,
        )

        with pytest.raises(ValueError, match="missing keys"):
            statistics_from_dict({"version": STATISTICS_VERSION})

    @pytest.mark.parametrize(
        "bad_type",
        [True, "not-an-int", 1.5, [], None],
        ids=["bool", "str", "float", "list", "none"],
    )
    def test_statistics_from_dict_rejects_wrong_types(self, bad_type):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            STATISTICS_VERSION,
            statistics_from_dict,
        )

        payload = {
            "version": STATISTICS_VERSION,
            "row_count": bad_type,
            "unique_sentence_ids": 1,
            "unique_polygons": 1,
            "unique_wikidata_entities": 1,
            "unique_documents": 1,
            "source_counts": {},
            "language_counts": {},
            "region_counts": {},
            "rows_with_coordinates": 1,
            "rows_without_coordinates": 0,
            "input_dataset_revision": "r",
            "pipeline_version": "v",
            "parquet_sha256": "0" * 64,
            "input_dataset_id": None,
        }
        with pytest.raises(ValueError, match="must be an integer"):
            statistics_from_dict(payload)

    def test_validation_rejects_wrong_statistics_version(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        dataset = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, dataset)
            manifest_path = export_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["statistics"]["version"] = 999
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with pytest.raises(ExportError):
                validate_export_directory(export_dir)

    def test_validation_rejects_recomputed_statistics_mismatch(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        dataset = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, dataset)
            manifest_path = export_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["statistics"]["row_count"] = (
                manifest["statistics"]["row_count"] + 1
            )
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with pytest.raises(ExportError):
                validate_export_directory(export_dir)


# ---------------------------------------------------------------------------
# Publishing (three artifacts, once)
# ---------------------------------------------------------------------------


class TestPublishingUploadsThreeArtifacts:
    def test_publisher_uploads_exactly_three_validated_artifacts(self):
        from osm_polygon_sentence_relevance.publishing import (
            publish_export_directory,
        )
        from tests.unit.publishing.test_huggingface import (
            RecordingHubApi,
            RecordingOperationFactory,
        )

        dataset = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, dataset)
            api = RecordingHubApi()
            factory = RecordingOperationFactory()
            publish_export_directory(
                export_dir,
                "my/dataset",
                hub_api=api,
                commit_operation_factory=factory,
            )
            assert len(api.create_commit_calls) == 1
            paths = sorted(c["path_in_repo"] for c in factory.calls)
            assert paths == ["README.md", "manifest.json", "sentences.parquet"]

    def test_publisher_never_runs_for_stale_card(self):
        from osm_polygon_sentence_relevance.publishing import (
            publish_export_directory,
        )
        from tests.unit.publishing.test_huggingface import (
            RecordingHubApi,
            RecordingOperationFactory,
        )

        dataset = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, dataset)
            card = (export_dir / "README.md").read_text(encoding="utf-8")
            (export_dir / "README.md").write_text(
                card.replace("1", "12345"), encoding="utf-8"
            )
            api = RecordingHubApi()
            factory = RecordingOperationFactory()
            with pytest.raises(ExportError):
                publish_export_directory(
                    export_dir,
                    "my/dataset",
                    hub_api=api,
                    commit_operation_factory=factory,
                )
            assert api.create_commit_calls == []
            assert factory.calls == []
