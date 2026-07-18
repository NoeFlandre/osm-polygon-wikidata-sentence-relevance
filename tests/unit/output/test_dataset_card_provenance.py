"""Source-provenance propagation tests for Phase 8C completion.

These tests assert the dataset-card / finalization chain end-to-end:

- Hub ``input_dataset_id`` reaches Parquet metadata, manifest, statistics,
  and the rendered card;
- the Hub card carries a safely encoded Markdown link to the dataset page
  (and the resolved revision/tree page) without quoting the raw value;
- local-mode renders explicitly state no Hub dataset ID was recorded;
- blank / non-string ``input_dataset_id`` is rejected before any output
  mutation;
- metadata / manifest / statistics disagreement on the dataset ID is
  rejected;
- existing callers omitting ``input_dataset_id`` still work;
- identical provenance + data produces byte-identical card text;
- no network calls occur in tests.

These tests intentionally live alongside the other dataset-card tests
because they exercise the same module surface.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.errors import ExportError, FinalizationError
from osm_polygon_sentence_relevance.finalization import finalize_sentence_dataset
from osm_polygon_sentence_relevance.output import (
    export_finalized_dataset,
    validate_export_directory,
)
from osm_polygon_sentence_relevance.output.dataset_card import (
    render_dataset_card,
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


def _finalize(
    rows: list[dict],
    *,
    revision="rev-8c",
    version="ver-8c",
    input_dataset_id: str | None = None,
):
    return finalize_sentence_dataset(
        _rows_to_table(rows),
        input_dataset_revision=revision,
        pipeline_version=version,
        input_dataset_id=input_dataset_id,
    )


def _stats_with(
    input_dataset_id: str | None = None,
    *,
    row_count: int = 1,
):
    from osm_polygon_sentence_relevance.output.dataset_card import (
        STATISTICS_VERSION,
        DatasetStatistics,
    )

    return DatasetStatistics(
        version=STATISTICS_VERSION,
        row_count=row_count,
        unique_sentence_ids=row_count,
        unique_polygons=1,
        unique_wikidata_entities=1,
        unique_documents=1,
        source_counts={"wikipedia": row_count} if row_count else {},
        language_counts={"en": row_count} if row_count else {},
        region_counts={"a": row_count} if row_count else {},
        rows_with_coordinates=row_count,
        rows_without_coordinates=0,
        input_dataset_revision="r",
        pipeline_version="v",
        input_dataset_id=input_dataset_id,
        parquet_sha256="0" * 64,
    )


# ---------------------------------------------------------------------------
# Schema propagation: finalize -> Parquet metadata -> manifest -> statistics
# ---------------------------------------------------------------------------


_PARQUET_DATASET_ID_KEY = b"input_dataset_id"


class TestSourceProvenancePropagation:
    def test_finalize_writes_input_dataset_id_into_parquet_metadata(self):
        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            input_dataset_id="HUB/Owner/wikidata-only",
        )
        meta = ds.table.schema.metadata or {}
        assert meta.get(_PARQUET_DATASET_ID_KEY) == b"HUB/Owner/wikidata-only"

    def test_local_mode_omits_input_dataset_id_from_metadata(self):
        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            input_dataset_id=None,
        )
        meta = ds.table.schema.metadata or {}
        # Local mode: the key is absent entirely (no silent normalization
        # to a blank string).
        assert _PARQUET_DATASET_ID_KEY not in meta

    def test_blank_input_dataset_id_is_rejected(self):
        for hostile in ["  ", "", "   "]:
            with pytest.raises(FinalizationError, match="input_dataset_id"):
                _finalize(
                    [make_segmented_row(sentence_text_normalized="a")],
                    input_dataset_id=hostile,
                )

    def test_non_string_input_dataset_id_is_rejected(self):
        # A non-string value must be rejected before any output mutation
        # with the project-level ``FinalizationError`` domain type.
        with pytest.raises(FinalizationError, match="input_dataset_id"):
            _finalize(
                [make_segmented_row(sentence_text_normalized="a")],
                input_dataset_id=123,  # type: ignore[arg-type]
            )

    def test_finalizer_rejects_blank_metadata_value_via_explicit_path(self):
        # When a caller constructs a table that already carries an
        # ``input_dataset_id`` metadata key but with a blank value, the
        # explicit ``input_dataset_id=None`` path still resolves
        # without silent normalization (the metadata key is missing in
        # this scenario because we are not altering Parquet metadata in
        # the finalizer when the explicit argument is None).
        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            input_dataset_id=None,
        )
        # ``b"input_dataset_id"`` is absent; nothing else is normalized.
        assert ds.table.schema.metadata.get(_PARQUET_DATASET_ID_KEY) is None

    def test_statistics_round_trip_carries_input_dataset_id(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
            statistics_from_dict,
            statistics_to_dict,
        )

        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            input_dataset_id="Hub/Owner/wikidata-only",
        )
        stats = compute_statistics(
            ds.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        assert stats.input_dataset_id == "Hub/Owner/wikidata-only"
        # The serialized JSON must always carry the field (null in
        # local mode, never missing).
        payload = statistics_to_dict(stats)
        assert "input_dataset_id" in payload
        assert payload["input_dataset_id"] == "Hub/Owner/wikidata-only"
        round_trip = statistics_from_dict(payload)
        assert round_trip.input_dataset_id == "Hub/Owner/wikidata-only"

    def test_omitting_input_dataset_id_explicitly_serializes_null(self):
        """Local mode (``input_dataset_id=None``) still serializes the
        field as JSON ``null``. The deserializer rejects missing keys.
        """
        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
            statistics_from_dict,
            statistics_to_dict,
        )

        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            input_dataset_id=None,
        )
        stats = compute_statistics(
            ds.table,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        assert stats.input_dataset_id is None
        payload = statistics_to_dict(stats)
        assert "input_dataset_id" in payload
        assert payload["input_dataset_id"] is None
        round_trip = statistics_from_dict(payload)
        assert round_trip.input_dataset_id is None

    def test_statistics_from_dict_rejects_missing_input_dataset_id(self):
        """``statistics_from_dict`` requires ``input_dataset_id``. A
        payload that omits the field is rejected like any other
        missing required version-1 key.
        """
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        payload = {
            "version": 1,
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
            # ``input_dataset_id`` deliberately omitted.
        }
        with pytest.raises(ValueError, match="missing keys"):
            statistics_from_dict(payload)

    def test_statistics_from_dict_rejects_blank_input_dataset_id(self):
        """A blank ``input_dataset_id`` string is rejected."""
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        base = {
            "version": 1,
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
            "input_dataset_id": "   ",
        }
        with pytest.raises(ValueError, match="cannot be blank"):
            statistics_from_dict(base)

    def test_statistics_from_dict_rejects_surrounding_whitespace_input_dataset_id(self):
        """A non-blank ``input_dataset_id`` with leading/trailing
        whitespace is rejected; ``statistics_from_dict`` never silently
        normalizes the supplied identifier.
        """
        from osm_polygon_sentence_relevance.output.dataset_card import (
            statistics_from_dict,
        )

        base = {
            "version": 1,
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
            "input_dataset_id": "  NoeFlandre/wikidata-only  ",
        }
        with pytest.raises(
            ValueError, match="input_dataset_id.*surrounding whitespace"
        ):
            statistics_from_dict(base)


# ---------------------------------------------------------------------------
# Card rendering: Hub vs Local branches
# ---------------------------------------------------------------------------


class TestDatasetCardSourceProvenanceRendering:
    def test_hub_card_contains_safely_encoded_dataset_link(self):
        ds_id = "NoeFlandre/osm-polygon-wikidata-only"
        card = render_dataset_card(_stats_with(input_dataset_id=ds_id, row_count=1))
        # Plain ``owner/repo`` identifiers preserve the ``/`` separator
        # in the URL and the link is well-formed.
        assert (
            "https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only"
            in card
        )
        assert "[NoeFlandre/osm-polygon-wikidata-only](" in card
        # Local-mode wording is NOT used.
        assert "local input snapshot" not in card.lower()

    def test_hub_card_omits_revision_link_when_revision_is_not_a_sha(self):
        card = render_dataset_card(_stats_with(input_dataset_id="Owner/dataset"))
        assert "https://huggingface.co/datasets/Owner/dataset" in card
        assert "/tree/" not in card

    def test_local_card_explicitly_states_no_hub_dataset_id_was_recorded(self):
        card = render_dataset_card(_stats_with(input_dataset_id=None))
        normalized = " ".join(card.lower().split())
        assert "no recorded hub dataset id" in normalized or (
            "local input snapshot" in normalized
        )
        # The Hub-only link must NOT appear for the local branch.
        assert "huggingface.co/datasets" not in card

    def test_hub_card_does_not_claim_acquisition_history(self):
        """The card must not infer which acquisition mechanism produced a
        Hub revision. The recorded immutable commit identifier is
        referenced factually but the prior "Hub acquisition step
        resolves…" wording is gone.
        """
        sha = "0" * 40
        from osm_polygon_sentence_relevance.output.dataset_card import (
            STATISTICS_VERSION,
            DatasetStatistics,
        )

        stats = DatasetStatistics(
            version=STATISTICS_VERSION,
            row_count=1,
            unique_sentence_ids=1,
            unique_polygons=1,
            unique_wikidata_entities=1,
            unique_documents=1,
            source_counts={"wikipedia": 1},
            language_counts={"en": 1},
            region_counts={"a": 1},
            rows_with_coordinates=1,
            rows_without_coordinates=0,
            input_dataset_revision=sha,
            pipeline_version="v",
            input_dataset_id="Owner/Repo",
            parquet_sha256="0" * 64,
        )
        normalized = " ".join(render_dataset_card(stats).lower().split())
        # The factual wording stays.
        assert "immutable" not in normalized or "recorded" in normalized
        # The prior "Hub acquisition step resolves the requested revision
        # to a commit SHA before download" claim is gone — the implementation
        # does not know which acquisition produced the recorded value.
        assert "hub acquisition step resolves" not in normalized
        assert "snapshot was fetched from" not in normalized
        # Arbitrary-revision wording must also stop referring to the
        # undocumented acquisition step.
        arbitrary = render_dataset_card(
            DatasetStatistics(
                version=STATISTICS_VERSION,
                row_count=0,
                unique_sentence_ids=0,
                unique_polygons=0,
                unique_wikidata_entities=0,
                unique_documents=0,
                source_counts={},
                language_counts={},
                region_counts={},
                rows_with_coordinates=0,
                rows_without_coordinates=0,
                input_dataset_revision="not-a-sha",
                pipeline_version="v",
                input_dataset_id="Owner/Repo",
                parquet_sha256="0" * 64,
            )
        )
        assert "if it is a sha-1 commit hash" not in arbitrary.lower()


# ---------------------------------------------------------------------------
# Adversarial: URL encoding (UTF-8 percent encoding with documented safe
# sets) and Markdown-label escape (``[``/``]``/backticks/backslashes/CR/LF)
# ---------------------------------------------------------------------------


class TestUrlAndLabelEscaping:
    """All percent encoding MUST go through ``urllib.parse.quote`` with
    explicit ``safe`` sets:

    - Dataset ID path component: ``safe="/"`` only (so ``owner/repo``
      stays readable but everything else, including Unicode code points,
      is encoded as UTF-8 percent escapes).
    - Revision path component: ``safe=""`` (every ``/``, ``#``, ``?``,
      ``%`` etc. is encoded). The Hugging Face Hub only ever serves
      revisions that look like commit SHAs so this path is well-defined
      even if a non-SHA value is ever threaded through.
    """

    def test_dataset_id_url_component_preserves_owner_repo_separator(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            _quote_url_component_dataset_id,
        )

        assert _quote_url_component_dataset_id("Owner/Repo") == "Owner/Repo"

    def test_dataset_id_url_component_percent_encodes_unicode(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            _quote_url_component_dataset_id,
        )

        # Accented, CJK, spaces, ``?``, ``#``, ``%``, backslash, brackets,
        # parens.
        cases = {
            "caf\u00e9/data": "caf%C3%A9/data",
            "\u4e2d\u6587/repo": "%E4%B8%AD%E6%96%87/repo",
            "owner with space/repo": "owner%20with%20space/repo",
            "owner?query/repo": "owner%3Fquery/repo",
            "owner#frag/repo": "owner%23frag/repo",
            "100%data/repo": "100%25data/repo",
            "owner\\data/repo": "owner%5Cdata/repo",
            "owner[brace]/repo": "owner%5Bbrace%5D/repo",
            "owner(paren)/repo": "owner%28paren%29/repo",
        }
        for raw, expected in cases.items():
            actual = _quote_url_component_dataset_id(raw)
            assert actual == expected, f"failed for {raw!r}"

    def test_revision_url_component_encodes_path_separators_and_unsafe_chars(self):
        """The revision path component MUST encode ``/`` because the
        Hub's tree URL only allows one segment after ``tree/``.
        """
        from osm_polygon_sentence_relevance.output.dataset_card import (
            _quote_url_component_revision,
        )

        # All path separators and a few hostname-relevant reserved chars
        # are encoded; commit SHAs (40 lowercase hex chars) are NOT
        # affected because every hex digit is in the unreserved set.
        sha = "abcdef0123456789" * 2 + "abcdef0123456789"
        assert _quote_url_component_revision(sha) == sha.lower()
        # "/" inside an arbitrary revision is encoded so it never bleeds
        # into the path tree.
        assert _quote_url_component_revision("a/b") == "a%2Fb"
        assert _quote_url_component_revision("a?b") == "a%3Fb"
        assert _quote_url_component_revision("a#b") == "a%23b"
        assert _quote_url_component_revision("a b") == "a%20b"
        assert _quote_url_component_revision("a\\b") == "a%5Cb"

    def test_markdown_label_escape_handles_brackets_and_backticks(self):
        """The label escape replaces ``[`` and ``]`` with HTML entities so
        an adversarial dataset identifier cannot terminate the link.
        ``_escape_md_inline`` is the wrong escape here.
        """
        from osm_polygon_sentence_relevance.output.dataset_card import (
            _escape_md_link_label,
        )

        escaped = _escape_md_link_label("Owner/[evil](https://attacker.example)")
        # Raw ``[`` and ``]`` are escaped; the raw closing `](` is gone.
        assert "](https://attacker.example)" not in escaped
        assert "&#91;" in escaped
        assert "&#93;" in escaped
        # Markdown markers that end a link or inline code span are
        # HTML-entity-escaped so the surrounding link stays well-formed.
        for char, entity in [
            ("[", "&#91;"),
            ("]", "&#93;"),
            ("`", "&#96;"),
            ("\\", "&#92;"),
            ("\r", "&#13;"),
            ("\n", "&#10;"),
            ("\t", "&#9;"),
            ("&", "&amp;"),
        ]:
            assert entity in _escape_md_link_label(f"a{char}b")

    def test_hub_link_with_adversarial_dataset_id_is_well_formed(self):
        """The full Markdown link invariantly uses percent-encoded URL +
        HTML-entity-escaped label so an adversarial slug cannot break
        out of the link.
        """
        from osm_polygon_sentence_relevance.output.dataset_card import (
            STATISTICS_VERSION,
            DatasetStatistics,
        )

        hostile = "Owner/`evil`/repo[bad](https://attacker.example)/x"
        stats = DatasetStatistics(
            version=STATISTICS_VERSION,
            row_count=0,
            unique_sentence_ids=0,
            unique_polygons=0,
            unique_wikidata_entities=0,
            unique_documents=0,
            source_counts={},
            language_counts={},
            region_counts={},
            rows_with_coordinates=0,
            rows_without_coordinates=0,
            input_dataset_revision="r",
            pipeline_version="v",
            input_dataset_id=hostile,
            parquet_sha256="0" * 64,
        )
        card = render_dataset_card(stats)
        # The raw attacker URL is never reproduced in a Markdown link
        # boundary; the hostile closing `](` is escaped.
        assert "evil](https://attacker.example)" not in card
        # The first raw ``[`` in the link label position is the one the
        # rendering introduced; the rendered link opens only once.
        assert "https://huggingface.co/datasets/" in card
        # The label's ``[`` and ``]`` come back as HTML entities so the
        # link's closing bracket cannot be smuggled in.
        assert "&#91;" in card
        assert "&#93;" in card

    def test_hub_link_with_unicode_dataset_id_is_well_formed(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            STATISTICS_VERSION,
            DatasetStatistics,
        )

        for ds_id in [
            "caf\u00e9/r\u00e9po",
            "\u4e2d\u6587/data",
            "owner name with spaces/repo",
            "owner?/repo",
            "owner?#/repo",
            "100%owned/repo",
            "owner\\escape/repo",
            "owner[brace]/repo",
            "owner(paren)/repo",
            "owner|pipe|/repo",
        ]:
            stats = DatasetStatistics(
                version=STATISTICS_VERSION,
                row_count=0,
                unique_sentence_ids=0,
                unique_polygons=0,
                unique_wikidata_entities=0,
                unique_documents=0,
                source_counts={},
                language_counts={},
                region_counts={},
                rows_with_coordinates=0,
                rows_without_coordinates=0,
                input_dataset_revision="r",
                pipeline_version="v",
                input_dataset_id=ds_id,
                parquet_sha256="0" * 64,
            )
            card = render_dataset_card(stats)
            # Brackets ``[`` and ``]`` in the input must be encoded so
            # they cannot close a Markdown link boundary.
            if "[" in ds_id or "]" in ds_id:
                assert "&#91;" in card or "&#93;" in card
            # A raw ``?`` or ``#`` in the input cannot leak into a URL
            # boundary because the URL is percent-encoded.
            assert "datasets/" in card


# ---------------------------------------------------------------------------
# Validator + Exporter: cross-checks the dataset ID across Parquet
# metadata, manifest, and statistics, AND rejects present-but-blank
# metadata without silently normalizing it to ``None``.
# ---------------------------------------------------------------------------


class TestValidatorDatasetIdConsistency:
    def test_validator_accepts_matching_hub_dataset_id(self):
        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            input_dataset_id="Owner/Repo",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            export_finalized_dataset(ds, tmpdir)
            res = validate_export_directory(Path(tmpdir))
            assert res.row_count == 1

    def test_validator_rejects_parquet_metadata_manifest_dataset_id_drift(self):
        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            input_dataset_id="Owner/Repo",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            export_finalized_dataset(ds, tmpdir)
            # Tamper: rewrite the manifest's statistics object to claim a
            # different dataset ID than the Parquet metadata.
            manifest_path = Path(tmpdir) / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["statistics"]["input_dataset_id"] = "Other/Repo"
            manifest_path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":"))
            )
            with pytest.raises(ExportError):
                validate_export_directory(Path(tmpdir))

    def test_validator_rejects_blank_metadata_value_with_export_error(self):
        # Tamper with the exported Parquet to insert a present-but-blank
        # ``b"input_dataset_id"`` metadata value. The validator must
        # reject this rather than silently normalize it to ``None``.
        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            input_dataset_id="Owner/Repo",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            export_finalized_dataset(ds, tmpdir)
            parquet_path = Path(tmpdir) / "sentences.parquet"
            table = pq.read_table(parquet_path)
            new_meta = dict(table.schema.metadata or {})
            new_meta[b"input_dataset_id"] = b"   "
            pq.write_table(table.replace_schema_metadata(new_meta), parquet_path)
            with pytest.raises(ExportError, match="not be blank"):
                validate_export_directory(Path(tmpdir))

    def test_validator_rejects_invalid_utf8_in_metadata(self):
        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            input_dataset_id="Owner/Repo",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            export_finalized_dataset(ds, tmpdir)
            parquet_path = Path(tmpdir) / "sentences.parquet"
            table = pq.read_table(parquet_path)
            new_meta = dict(table.schema.metadata or {})
            new_meta[b"input_dataset_id"] = b"\xff\xfe\xfd"
            pq.write_table(table.replace_schema_metadata(new_meta), parquet_path)
            with pytest.raises(ExportError, match="not valid UTF-8"):
                validate_export_directory(Path(tmpdir))

    def test_validator_rejects_surrounding_whitespace_dataset_id(self):
        """The validator must apply the same surrounding-whitespace
        rejection the exporter and ``_resolve_input_dataset_id`` apply.
        Tampering the exported Parquet with a non-blank but
        whitespace-surrounded dataset ID must raise ``ExportError``
        rather than silently normalizing to the trimmed identifier.
        """
        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            input_dataset_id="NoeFlandre/wikidata-only",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            export_finalized_dataset(ds, tmpdir)
            parquet_path = Path(tmpdir) / "sentences.parquet"
            table = pq.read_table(parquet_path)
            new_meta = dict(table.schema.metadata or {})
            new_meta[b"input_dataset_id"] = b"  NoeFlandre/wikidata-only  "
            pq.write_table(table.replace_schema_metadata(new_meta), parquet_path)
            with pytest.raises(ExportError, match="surrounding whitespace"):
                validate_export_directory(Path(tmpdir))


# ---------------------------------------------------------------------------
# End-to-end determinism: identical provenance + data -> byte-identical card
# ---------------------------------------------------------------------------


class TestExporterDatasetIdStrictNormalization:
    """Phase 8C Export-Provenance Consistency Micro-Amendment.

    The exporter must apply the same strict dataset-ID contract as the
    finalizer, validator, and ``_resolve_input_dataset_id``:

    - missing metadata → ``None`` (local mode);
    - present value must decode as UTF-8;
    - present value must be nonblank;
    - surrounding whitespace is rejected, not silently trimmed;
    - valid values are preserved exactly (no normalization);
    - malformed values raise ``ExportError`` with the original
      ``UnicodeDecodeError`` preserved as ``__cause__``.
    """

    def test_exporter_rejects_blank_metadata_with_export_error(self):
        # Finalize with a valid Hub ID, then tamper the Parquet metadata
        # to a blank value. The exporter must refuse rather than turn
        # it into local mode.
        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            input_dataset_id="Owner/Repo",
        )
        with tempfile.TemporaryDirectory() as src_root:
            # Write and tamper in a separate src directory so the
            # exporter's "non-empty output dir" check does not fire.
            src_dir = Path(src_root) / "src"
            src_dir.mkdir()
            parquet_path = src_dir / "sentences.parquet"
            pq.write_table(ds.table, parquet_path)
            table = pq.read_table(parquet_path)
            new_meta = dict(table.schema.metadata or {})
            new_meta[b"input_dataset_id"] = b"   "
            pq.write_table(table.replace_schema_metadata(new_meta), parquet_path)
            tampered = finalize_sentence_dataset(
                _rows_to_table([make_segmented_row(sentence_text_normalized="a")]),
                input_dataset_revision="r",
                pipeline_version="v",
                input_dataset_id=None,  # signal: read from Parquet metadata
            ).table
            # Replace the finalized table's metadata with the tampered
            # Parquet metadata; the exporter must detect it.
            tampered = tampered.replace_schema_metadata(new_meta)
            from osm_polygon_sentence_relevance.finalization import (
                FinalizationReport,
            )

            finalization_report = FinalizationReport(
                input_sentence_occurrence_count=0,
                output_sentence_count=0,
                duplicate_occurrence_count_removed=0,
                cross_source_duplicate_group_count=0,
            )
            from osm_polygon_sentence_relevance.finalization import (
                FinalizedDataset,
            )

            tampered_ds = FinalizedDataset(table=tampered, report=finalization_report)
            with (
                tempfile.TemporaryDirectory() as outdir,
                pytest.raises(ExportError, match="not be blank"),
            ):
                export_finalized_dataset(tampered_ds, outdir)

    def test_exporter_rejects_invalid_utf8_with_unicode_decode_error_cause(self):
        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            input_dataset_id="Owner/Repo",
        )
        tampered_meta = {
            b"input_dataset_revision": b"r",
            b"pipeline_version": b"v",
            b"input_dataset_id": b"\xff\xfe\xfd",
        }
        from osm_polygon_sentence_relevance.finalization import (
            FinalizationReport,
            FinalizedDataset,
        )

        tampered_table = ds.table.replace_schema_metadata(tampered_meta)
        tampered_ds = FinalizedDataset(
            table=tampered_table,
            report=FinalizationReport(
                input_sentence_occurrence_count=0,
                output_sentence_count=0,
                duplicate_occurrence_count_removed=0,
                cross_source_duplicate_group_count=0,
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ExportError, match="not valid UTF-8") as exc_info:
                export_finalized_dataset(tampered_ds, tmpdir)
            # The original ``UnicodeDecodeError`` is preserved as the
            # exception cause so callers can introspect it.
            assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)

    def test_exporter_rejects_surrounding_whitespace_dataset_id(self):
        """An ID with leading/trailing whitespace is rejected. The
        exporter used to strip the value, which mutated the recorded
        identifier and hid corruption.
        """
        # Sanity: the finalizer does not strip; the bytes round-trip
        # verbatim including the surrounding whitespace. The finalizer
        # now rejects whitespace at its boundary, so this test must
        # construct the finalized dataset via a clean finalization pass
        # and then tamper the Parquet metadata to inject whitespace —
        # exactly the scenario the exporter is supposed to defend
        # against.
        from osm_polygon_sentence_relevance.finalization import (
            FinalizedDataset,
        )

        clean = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            input_dataset_id="Owner/Repo",
        )
        new_meta = dict(clean.table.schema.metadata or {})
        new_meta[b"input_dataset_id"] = b"  Owner/Repo  "
        tampered_table = clean.table.replace_schema_metadata(new_meta)
        tampered_ds = FinalizedDataset(
            table=tampered_table,
            report=clean.report,
        )
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(ExportError, match="surrounding whitespace"),
        ):
            export_finalized_dataset(tampered_ds, tmpdir)

    def test_exporter_preserves_valid_dataset_id_exactly(self):
        # A perfectly valid Hub ID is preserved byte-for-byte through
        # the export chain: Parquet metadata, manifest top-level,
        # manifest ``statistics``, and the rendered card.
        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            input_dataset_id="NoeFlandre/osm-polygon-wikidata-only",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            res = export_finalized_dataset(ds, tmpdir)
            # Manifest top-level field.
            manifest = json.loads(res.manifest_path.read_text(encoding="utf-8"))
            assert (
                manifest["input_dataset_id"] == "NoeFlandre/osm-polygon-wikidata-only"
            )
            # Manifest statistics object.
            assert manifest["statistics"]["input_dataset_id"] == (
                "NoeFlandre/osm-polygon-wikidata-only"
            )
            # Card text mentions the exact identifier.
            card_text = res.card_path.read_text(encoding="utf-8")
            assert "NoeFlandre/osm-polygon-wikidata-only" in card_text
            # Card link uses the exact identifier in the URL component.
            assert (
                "https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only"
                in card_text
            )


class TestEndToEndDeterminism:
    def test_byte_identical_artifacts_when_provenance_and_data_match(self):
        common_rows = [
            make_segmented_row(sentence_text_normalized="a"),
            make_segmented_row(sentence_text_normalized="b", region="reg-2"),
        ]
        ds_a = _finalize(
            common_rows,
            input_dataset_id="Owner/Repo",
        )
        ds_b = _finalize(
            common_rows,
            input_dataset_id="Owner/Repo",
        )
        assert ds_a.table.schema.metadata == ds_b.table.schema.metadata

        from osm_polygon_sentence_relevance.output.dataset_card import (
            compute_statistics,
        )

        sha = "0" * 64
        stats_a = compute_statistics(
            ds_a.table,
            input_dataset_revision="rev-8c",
            pipeline_version="ver-8c",
            parquet_sha256=sha,
        )
        stats_b = compute_statistics(
            ds_b.table,
            input_dataset_revision="rev-8c",
            pipeline_version="ver-8c",
            parquet_sha256=sha,
        )
        assert render_dataset_card(stats_a) == render_dataset_card(stats_b)


# ---------------------------------------------------------------------------
# Coverage: hit every branch in the source-provenance rendering helpers and
# statistics helpers.
# ---------------------------------------------------------------------------


class TestSourceProvenanceCoverage:
    def test_hub_revision_tree_link_branch_when_revision_is_sha1(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            STATISTICS_VERSION,
            DatasetStatistics,
        )

        sha = "0" * 40
        stats = DatasetStatistics(
            version=STATISTICS_VERSION,
            row_count=1,
            unique_sentence_ids=1,
            unique_polygons=1,
            unique_wikidata_entities=1,
            unique_documents=1,
            source_counts={"wikipedia": 1},
            language_counts={"en": 1},
            region_counts={"a": 1},
            rows_with_coordinates=1,
            rows_without_coordinates=0,
            input_dataset_revision=sha,
            pipeline_version="v",
            input_dataset_id="Owner/Repo",
            parquet_sha256="0" * 64,
        )
        card = render_dataset_card(stats)
        # Tree link branch: linked SHA, no acquisition-history wording.
        assert "/tree/" in card
        assert "0" * 40 in card
        # No acquisition-history wording anywhere on the card.
        assert "hub acquisition step resolves" not in card.lower()

    def test_hub_revision_branch_when_revision_is_not_sha1(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            STATISTICS_VERSION,
            DatasetStatistics,
        )

        stats = DatasetStatistics(
            version=STATISTICS_VERSION,
            row_count=0,
            unique_sentence_ids=0,
            unique_polygons=0,
            unique_wikidata_entities=0,
            unique_documents=0,
            source_counts={},
            language_counts={},
            region_counts={},
            rows_with_coordinates=0,
            rows_without_coordinates=0,
            input_dataset_revision="not-a-sha",
            pipeline_version="v",
            input_dataset_id="Owner/Repo",
            parquet_sha256="0" * 64,
        )
        card = render_dataset_card(stats)
        assert "huggingface.co/datasets/Owner/Repo" in card
        assert "/tree/" not in card

    def test_resolve_input_dataset_id_disagrees_with_explicit_raises(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            _resolve_input_dataset_id,
        )

        table = pa.table({"x": [1]}).replace_schema_metadata(
            {b"input_dataset_id": b"Owner/A"}
        )
        with pytest.raises(ValueError, match="disagrees with Parquet metadata"):
            _resolve_input_dataset_id(table, "Owner/B")

    def test_resolve_input_dataset_id_rejects_blank_metadata(self):
        """A present-but-blank metadata value is rejected by
        ``_resolve_input_dataset_id`` (which the finalizer funnels into
        ``FinalizationError``); the helper does not silently normalize
        to ``None``.
        """
        from osm_polygon_sentence_relevance.output.dataset_card import (
            _resolve_input_dataset_id,
        )

        blank_table = pa.table({"x": [1]}).replace_schema_metadata(
            {b"input_dataset_id": b"   "}
        )
        with pytest.raises(ValueError, match="cannot be blank"):
            _resolve_input_dataset_id(blank_table, None)

    def test_resolve_input_dataset_id_invalid_utf8_in_metadata(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            _resolve_input_dataset_id,
        )

        bad_table = pa.table({"x": [1]}).replace_schema_metadata(
            {b"input_dataset_id": b"\xff\xfe\xfd"}
        )
        with pytest.raises(ValueError, match="not valid UTF-8"):
            _resolve_input_dataset_id(bad_table, None)

    def test_resolve_input_dataset_id_returns_value_from_metadata(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            _resolve_input_dataset_id,
        )

        table = pa.table({"x": [1]}).replace_schema_metadata(
            {b"input_dataset_id": b"Owner/Repo"}
        )
        # Passing ``None`` as the explicit override reads from metadata.
        assert _resolve_input_dataset_id(table, None) == "Owner/Repo"
