"""Phase 8C accuracy/robustness regression tests.

Focused contracts only; no exact-prose pinning. Covers:

- factual card claims (no relevance/weighting, no unsupported task
  categories, no false casing/verbatim claims, future-work labels);
- document-identity robustness under cross-source/site collisions;
- strict statistics deserialization (no coercion, invariants enforced);
- no duplicated-statistics drift between manifest and statistics object;
- deterministic card rendering under special characters / empty data.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

from osm_polygon_sentence_relevance.errors import ExportError
from osm_polygon_sentence_relevance.finalization import (
    finalize_sentence_dataset,
)
from osm_polygon_sentence_relevance.schemas import (
    SEGMENTED_SENTENCES_SCHEMA,
)
from tests.helpers import make_segmented_row


def _rows_to_table(rows: list[dict]) -> pa.Table:
    if not rows:
        return SEGMENTED_SENTENCES_SCHEMA.empty_table()
    data = {}
    for field in SEGMENTED_SENTENCES_SCHEMA:
        data[field.name] = pa.array([r[field.name] for r in rows], type=field.type)
    return pa.table(data, schema=SEGMENTED_SENTENCES_SCHEMA)


def _finalize(rows, *, revision="rev-8c", version="ver-8c"):
    return finalize_sentence_dataset(
        _rows_to_table(rows),
        input_dataset_revision=revision,
        pipeline_version=version,
    )


def _export(tmpdir, dataset):
    """Export exactly the supplied dataset; do not synthesize a stub."""
    from osm_polygon_sentence_relevance.output import export_finalized_dataset

    res = export_finalized_dataset(dataset, tmpdir)
    return Path(res.parquet_path).parent


# ---------------------------------------------------------------------------
# Factual card claims
# ---------------------------------------------------------------------------


class TestCardFactualClaims:
    def test_card_does_not_claim_land_use_weighting(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
            render_dataset_card,
        )

        ds = _finalize([make_segmented_row(sentence_text_normalized="x")])
        stats = compute_statistics(
            ds.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        card = render_dataset_card(stats).lower()
        # The card must NOT include statements that the dataset itself
        # is weighted, scored, classified, or labelled. Mentions of these
        # concepts in future-work / licensing context are allowed.
        forbidden_positive_claims = [
            "weighted by land-use relevance",
            "weighted by land use relevance",
            "weighted by relevance",
        ]
        for claim in forbidden_positive_claims:
            assert claim not in card, f"Card must not positively claim {claim!r}"
        # No claim of provided weighting, scoring, or classification.
        assert "this dataset is weighted" not in card
        assert (
            "weighted by " not in card
        )  # covered by 'weighted by relevance' etc. above
        assert "sentence scores" not in card
        assert "scored by" not in card
        # 'land-use relevance labels' is allowed (in future-work context)
        # but the card must NOT assert they are provided.
        assert "this dataset provides" not in card
        assert "these labels are included" not in card
        assert "labeled" not in card
        assert "labels are produced" not in card
        assert "relevance scoring" not in card

    def test_card_excludes_unsupported_task_categories(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
            render_dataset_card,
        )

        ds = _finalize([make_segmented_row(sentence_text_normalized="x")])
        stats = compute_statistics(
            ds.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        card = render_dataset_card(stats)
        assert "sentence-similarity" not in card
        assert "text-retrieval" not in card

    def test_card_preserves_case_and_does_not_claim_verbatim(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
            render_dataset_card,
        )

        ds = _finalize([make_segmented_row(sentence_text_normalized="x")])
        stats = compute_statistics(
            ds.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        card = render_dataset_card(stats).lower()
        # Normalization preserves case -> must not claim it changes casing.
        assert "normaliz" in card
        assert "preserves case" in card
        # Raw section text is stripped (.strip()); not verbatim input.
        assert "verbatim" not in card

    def test_card_omits_land_use_future_work_phrase(self):
        """The card no longer mentions future downstream land-use work.

        Phase 9P removed the original sentence (which talked about
        future work for land-use relevance labels, polygon-description
        classifications, etc.) at the user's request; the test pins
        the omission so a regression cannot quietly re-introduce
        the phrase.
        """
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
            render_dataset_card,
        )

        ds = _finalize([make_segmented_row(sentence_text_normalized="x")])
        stats = compute_statistics(
            ds.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        card = render_dataset_card(stats).lower()
        # The phrase "land-use relevance labels" must not appear.
        assert "land-use relevance labels" not in card
        assert "similarity-pair annotations" not in card
        # The "preview rows" line was also removed; the preview
        # section must show only polygons/Wikidata entities/documents.
        assert "preview rows" not in card
        # The word "articles" must not appear in the introduction
        # paragraph about OpenStreetMap polygons.
        assert "polygon articles" not in card

    def test_card_describes_the_actual_edit_marker_normalization(self):
        """The public card must match ``normalize_sentence`` exactly.

        Normalization removes consecutive leading bracketed MediaWiki
        markers containing a pipe when their closing bracket occurs within
        120 characters. It does not remove templates, list syntax, or
        signature tildes.
        """
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
            render_dataset_card,
        )

        ds = _finalize([make_segmented_row(sentence_text_normalized="x")])
        stats = compute_statistics(
            ds.table,
            input_dataset_revision="rev",
            pipeline_version="1.0.0",
            parquet_sha256="0" * 64,
        )
        card = render_dataset_card(stats)
        prose = " ".join(card.split())

        assert "`[label | target]`" in prose
        assert "within 120 characters" in prose
        assert "consecutive leading" in prose
        assert "signature tildes" not in prose
        assert "{{subst:" not in prose
        assert "numbered list" not in prose


# ---------------------------------------------------------------------------
# Document identity robustness
# ---------------------------------------------------------------------------


class TestDocumentIdentity:
    def test_cross_source_site_same_document_id_are_two_distinct_identities(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
        )

        # Same document_id across two sources/sites with distinct polygons.
        rows = [
            make_segmented_row(
                source="wikipedia",
                site="en.wikipedia.org",
                language="en",
                document_id="DOC-SHARED",
                polygon_id="poly-1",
                sentence_text_normalized="a",
            ),
            make_segmented_row(
                source="wikivoyage",
                site="en.wikivoyage.org",
                language="en",
                document_id="DOC-SHARED",
                polygon_id="poly-2",
                sentence_text_normalized="b",
            ),
        ]
        ds = _finalize(rows)
        stats = compute_statistics(
            ds.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        # The two rows share a raw document_id but differ in provenance
        # tuple, so they are two distinct document identities.
        assert stats.unique_documents == 2

    def test_same_provenance_identity_collapses(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
        )

        rows = [
            make_segmented_row(
                source="wikivoyage",
                site="en.wikivoyage.org",
                language="en",
                document_id="DOC-DUP",
                polygon_id="poly-1",
                sentence_text_normalized="a",
            ),
            make_segmented_row(
                source="wikivoyage",
                site="en.wikivoyage.org",
                language="en",
                document_id="DOC-DUP",
                polygon_id="poly-2",
                sentence_text_normalized="b",
            ),
        ]
        ds = _finalize(rows)
        stats = compute_statistics(
            ds.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        # Identical (source, site, language, document_id) -> one identity.
        assert stats.unique_documents == 1


# ---------------------------------------------------------------------------
# Strict statistics deserialization
# ---------------------------------------------------------------------------


def _valid_stats_payload(**overrides) -> dict:
    from osm_polygon_sentence_relevance.output.dataset_card import (
        STATISTICS_VERSION,
    )

    base = {
        "version": STATISTICS_VERSION,
        "row_count": 1,
        "unique_sentence_ids": 1,
        "unique_polygons": 1,
        "unique_wikidata_entities": 1,
        "unique_documents": 1,
        "source_counts": {"wikipedia": 1},
        "language_counts": {"en": 1},
        "region_counts": {"reg": 1},
        "rows_with_coordinates": 1,
        "rows_without_coordinates": 0,
        "input_dataset_revision": "r",
        "pipeline_version": "v",
        "parquet_sha256": "0" * 64,
        # ``input_dataset_id`` is a required v1 statistics field. The
        # amendment tests focus on other fields; the default ``None``
        # covers the local-mode serialization.
        "input_dataset_id": None,
    }
    base.update(overrides)
    return base


class TestStrictStatisticsDeserialization:
    @pytest.mark.parametrize(
        "bad",
        [
            {"row_count": True},
            {"row_count": 1.5},
            {"row_count": "3"},
            {"row_count": -1},
        ],
        ids=["bool", "float", "numeric-str", "negative"],
    )
    def test_scalar_counts_reject_bad_values(self, bad):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        key = next(iter(bad))
        payload = _valid_stats_payload(**bad)
        with pytest.raises(ValueError, match="integer"):
            statistics_from_dict(payload)
        _ = key

    def test_mapping_values_reject_bad_types(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        for bad_val in [True, 1.5, "2", -1]:
            payload = _valid_stats_payload(source_counts={"en": bad_val})
            with pytest.raises(
                ValueError, match="values must be non-negative integers"
            ):
                statistics_from_dict(payload)

    def test_mapping_keys_must_be_strings(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        payload = _valid_stats_payload(source_counts={1: 1})
        with pytest.raises(ValueError, match="keys must be strings"):
            statistics_from_dict(payload)

    def test_missing_statistics_keys_rejected(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        payload = _valid_stats_payload()
        del payload["parquet_sha256"]
        with pytest.raises(ValueError, match="missing keys"):
            statistics_from_dict(payload)

    def test_unknown_top_level_statistics_key_rejected(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        payload = _valid_stats_payload(bonus_field=5)
        with pytest.raises(ValueError, match="unknown"):
            statistics_from_dict(payload)

    def test_revision_and_version_must_be_nonblank(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        for bad in ["", "   "]:
            with pytest.raises(ValueError, match="must be a non-blank string"):
                statistics_from_dict(_valid_stats_payload(input_dataset_revision=bad))
            with pytest.raises(ValueError, match="must be a non-blank string"):
                statistics_from_dict(_valid_stats_payload(pipeline_version=bad))

    def test_sha256_must_be_lowercase_64_hex(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        for bad in ["0" * 63, "0" * 65, "Z" * 64, "g" * 64, "0" * 63 + "x"]:
            with pytest.raises(ValueError, match="lowercase 64-character hex"):
                statistics_from_dict(_valid_stats_payload(parquet_sha256=bad))

    def test_accounting_invariants_enforced(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        # Coordinates must sum to row count.
        with pytest.raises(ValueError, match="accounting identity violated"):
            statistics_from_dict(
                _valid_stats_payload(
                    rows_with_coordinates=2, rows_without_coordinates=0, row_count=1
                )
            )
        # Source counts must sum to row count.
        with pytest.raises(ValueError, match="source_counts sums to"):
            statistics_from_dict(
                _valid_stats_payload(source_counts={"wikipedia": 5}, row_count=1)
            )
        # Unique counts cannot exceed row count.
        with pytest.raises(ValueError, match="cannot exceed"):
            statistics_from_dict(_valid_stats_payload(unique_polygons=2, row_count=1))

    def test_language_and_region_sum_violation_rejected(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        # Source counts match row_count but language or region do not.
        with pytest.raises(ValueError, match="language_counts sums to"):
            statistics_from_dict(
                _valid_stats_payload(language_counts={"en": 2}, row_count=1)
            )
        with pytest.raises(ValueError, match="region_counts sums to"):
            statistics_from_dict(
                _valid_stats_payload(region_counts={"a": 2}, row_count=1)
            )

    @pytest.mark.parametrize(
        "field",
        ["row_count", "rows_with_coordinates", "rows_without_coordinates"],
    )
    def test_negative_integer_rejected(self, field):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        with pytest.raises(ValueError, match="non-negative integer"):
            statistics_from_dict(_valid_stats_payload(**{field: -1}))

    def test_other_unique_exceeds_row_count_rejected(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        for key in (
            "unique_sentence_ids",
            "unique_wikidata_entities",
            "unique_documents",
        ):
            with pytest.raises(ValueError, match="cannot exceed row_count"):
                statistics_from_dict(_valid_stats_payload(**{key: 5}, row_count=1))

    def test_validation_wraps_deserialization_error_with_cause(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        ds = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, ds)
            manifest_path = export_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["statistics"]["row_count"] = "not-an-int"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with pytest.raises(ExportError) as exc:
                validate_export_directory(export_dir)
            assert isinstance(exc.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# No duplicated-statistics drift
# ---------------------------------------------------------------------------


class TestNoDuplicatedStatisticsDrift:
    def test_manifest_counts_derive_from_statistics(self):
        from osm_polygon_sentence_relevance.output import (
            export_finalized_dataset,
        )

        ds = _finalize(
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
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            res = export_finalized_dataset(ds, tmpdir)
            manifest = json.loads(Path(res.manifest_path).read_text())
            stats = manifest["statistics"]
            assert manifest["counts_by_source"] == stats["source_counts"]
            assert manifest["counts_by_language"] == stats["language_counts"]
            assert manifest["counts_by_region"] == stats["region_counts"]
            assert manifest["row_count"] == stats["row_count"]
            assert manifest["sha256"] == stats["parquet_sha256"]
            assert manifest["input_dataset_revision"] == stats["input_dataset_revision"]
            assert manifest["pipeline_version"] == stats["pipeline_version"]

    def test_altered_legacy_top_level_counts_rejected(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        ds = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, ds)
            # Tamper only the legacy top-level counts, leaving statistics
            # correct. Validation must still reject (the two must agree).
            manifest_path = export_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["counts_by_source"] = {"wikipedia": 999}
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with pytest.raises(ExportError):
                validate_export_directory(export_dir)

    def test_altered_legacy_sha256_rejected(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        ds = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, ds)
            manifest_path = export_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sha256"] = "0" * 64  # differs from statistics.parquet_sha256
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with pytest.raises(ExportError):
                validate_export_directory(export_dir)


# ---------------------------------------------------------------------------
# Deterministic rendering hardening
# ---------------------------------------------------------------------------


class TestCardRenderingHardening:
    def _stats_with(self, **overrides) -> dict:
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
        )

        rows = [
            make_segmented_row(
                source="wikipedia",
                language=overrides.pop("language", "en"),
                region=overrides.pop("region", "reg-a"),
                sentence_text_normalized="a",
            )
        ]
        ds = _finalize(rows)
        stats = compute_statistics(
            ds.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        return stats

    def test_empty_dataset_renders_valid_empty_language_list(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
            render_dataset_card,
        )

        ds = _finalize([])
        stats = compute_statistics(
            ds.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        card = render_dataset_card(stats)
        # Empty language list still yields a valid, present YAML block.
        assert "language:" in card
        assert "\n---\n" in card

    def test_language_value_with_special_chars_is_yaml_safe(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
            render_dataset_card,
        )

        ds = _finalize(
            [
                make_segmented_row(
                    language="zh: 北京 | 東京",
                    sentence_text_normalized="a",
                )
            ]
        )
        stats = compute_statistics(
            ds.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        card = render_dataset_card(stats)
        # The language with a pipe must be quoted so it does not break the
        # YAML sequence; the rendered line should be quoted.
        assert '- "zh: 北京 | 東京"' in card

    def test_markdown_table_cell_special_chars_are_escaped(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
            render_dataset_card,
        )

        # A region containing a pipe / backslash / newline must not break the
        # Markdown table rows.
        ds = _finalize(
            [
                make_segmented_row(
                    region="a|b\\c\nd",
                    sentence_text_normalized="a",
                )
            ]
        )
        stats = compute_statistics(
            ds.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        card = render_dataset_card(stats)
        # The raw pipe/backslash/newline must be escaped (not a bare '|').
        assert "| a&#124;b&#92;c&#10;d |" in card

    def test_identical_statistics_render_byte_identical(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
            render_dataset_card,
        )

        ds = _finalize([make_segmented_row(sentence_text_normalized="x")])
        stats = compute_statistics(
            ds.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        assert render_dataset_card(stats) == render_dataset_card(stats)


# ---------------------------------------------------------------------------
# Strict-module branch coverage
# ---------------------------------------------------------------------------


class TestStrictModuleBranches:
    def test_yaml_quote_escapes_all_control_chars(self):
        from osm_polygon_sentence_relevance.output._card.rendering import (
            _yaml_quote_scalar,
        )

        # Newline, tab, carriage-return, backslash, and quote are all
        # escaped. A control character (NUL) is escaped to ``\0``.
        quoted = _yaml_quote_scalar('a\\b"c\nd\te\rf\0g')
        assert "\n" not in quoted
        assert "\t" not in quoted
        assert "\\n" in quoted
        assert "\\t" in quoted
        assert "\\r" in quoted
        assert "\\\\" in quoted
        assert '\\"' in quoted
        assert "\\0" in quoted

    def test_yaml_quote_uses_double_quotes(self):
        from osm_polygon_sentence_relevance.output._card.rendering import (
            _yaml_quote_scalar,
        )

        # Always double-quoted so the rendering is deterministic and
        # uniform across special-character and ASCII values.
        quoted = _yaml_quote_scalar("en")
        assert quoted.startswith('"')
        assert quoted.endswith('"')

    def test_escape_md_cell_escapes_ampersand(self):
        from osm_polygon_sentence_relevance.output._card.rendering import (
            _escape_md_cell,
        )

        assert (
            _escape_md_cell("a&b|c\\d\re\nf\tg")
            == "a&amp;b&#124;c&#92;d&#13;e&#10;f&#9;g"
        )

    def test_unknown_keys_rejected_with_descriptive_error(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            STATISTICS_VERSION,
            statistics_from_dict,
        )

        with pytest.raises(ValueError, match="unknown keys"):
            statistics_from_dict(
                {
                    "version": STATISTICS_VERSION,
                    "extra_field": True,
                }
            )

    def test_version_mismatch_rejected(self):
        # Provide a version that differs from STATISTICS_VERSION but is
        # otherwise a complete statistics dict, so the version check fires
        # rather than the missing-keys check.
        from osm_polygon_sentence_relevance.output.dataset_card import (
            STATISTICS_VERSION,
            statistics_from_dict,
        )

        good = {
            "version": STATISTICS_VERSION,
            "row_count": 0,
            "unique_sentence_ids": 0,
            "unique_polygons": 0,
            "unique_wikidata_entities": 0,
            "unique_documents": 0,
            "source_counts": {},
            "language_counts": {},
            "region_counts": {},
            "rows_with_coordinates": 0,
            "rows_without_coordinates": 0,
            "input_dataset_revision": "r",
            "pipeline_version": "v",
            "parquet_sha256": "0" * 64,
            "input_dataset_id": None,
        }
        good["version"] = STATISTICS_VERSION + 1
        with pytest.raises(ValueError, match="does not match"):
            statistics_from_dict(good)

    def test_statistics_root_must_be_dict(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        with pytest.raises(ValueError, match="must be a JSON object"):
            statistics_from_dict(None)

    def test_counts_mapping_must_be_dict(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            STATISTICS_VERSION,
            statistics_from_dict,
        )

        with pytest.raises(ValueError, match="must be a JSON object"):
            statistics_from_dict(
                {
                    "version": STATISTICS_VERSION,
                    "row_count": 0,
                    "rows_with_coordinates": 0,
                    "rows_without_coordinates": 0,
                    "source_counts": [],
                    "language_counts": {},
                    "region_counts": {},
                    "unique_sentence_ids": 0,
                    "unique_polygons": 0,
                    "unique_wikidata_entities": 0,
                    "unique_documents": 0,
                    "input_dataset_revision": "r",
                    "pipeline_version": "v",
                    "parquet_sha256": "0" * 64,
                    "input_dataset_id": None,
                }
            )

    def test_drift_check_rejects_altered_revision(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        ds = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, ds)
            manifest_path = export_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["input_dataset_revision"] = "altered-revision"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with pytest.raises(ExportError):
                validate_export_directory(export_dir)

    def test_drift_check_rejects_altered_pipeline_version(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        ds = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, ds)
            manifest_path = export_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["pipeline_version"] = "altered-version"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with pytest.raises(ExportError):
                validate_export_directory(export_dir)

    def test_drift_check_rejects_altered_row_count_top_level(self):
        from osm_polygon_sentence_relevance.output import (
            validate_export_directory,
        )

        ds = _finalize([make_segmented_row(sentence_text_normalized="x")])
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = _export(tmpdir, ds)
            manifest_path = export_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            # Move the statistics-derived top-level row_count so it diverges.
            manifest["statistics"]["row_count"] += 1
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            # Both the row_count drift and the broken accounting identity
            # are detected by validation. Expect either ExportError.
            with pytest.raises(ExportError):
                validate_export_directory(export_dir)
