"""Phase 8C Final Correctness Amendment tests.

Focused structural contracts only; no exact-prose pinning. Covers:

- Valid YAML front-matter structure (zero, one, and many languages);
- The export-test helper uses exactly the dataset it is given;
- DatasetStatistics is genuinely immutable;
- Dynamic card text (revision, pipeline version) is escaped so a value
  containing backticks, backslashes, or newlines cannot corrupt the
  rendered Markdown;
- Stable statistics version 1 from the start of Phase 8C;
- Factual wording preservation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

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


def _finalize(rows, *, revision="rev-8c-final", version="ver-8c-final"):
    return finalize_sentence_dataset(
        _rows_to_table(rows),
        input_dataset_revision=revision,
        pipeline_version=version,
    )


def _stats_with(language_counts: dict[str, int]):
    from osm_polygon_sentence_relevance.output.dataset_card import (
        STATISTICS_VERSION,
        DatasetStatistics,
    )

    row_count = sum(language_counts.values()) or 0
    return DatasetStatistics(
        version=STATISTICS_VERSION,
        row_count=row_count,
        unique_sentence_ids=row_count,
        unique_polygons=1,
        unique_wikidata_entities=1,
        unique_documents=1,
        source_counts={"wikipedia": row_count} if row_count else {},
        language_counts=language_counts,
        region_counts={"a": row_count} if row_count else {},
        rows_with_coordinates=row_count,
        rows_without_coordinates=0,
        input_dataset_revision="rev-x",
        pipeline_version="ver-x",
        parquet_sha256="0" * 64,
    )


# ---------------------------------------------------------------------------
# YAML front-matter structural validation (no external dependency)
# ---------------------------------------------------------------------------


def _yaml_front_matter(card: str) -> str:
    """Extract the YAML block between the first two ``---`` lines."""
    # Strip optional leading content before the opening delimiter.
    delim = card.find("---")
    if delim < 0:
        raise ValueError("card has no opening YAML delimiter")
    tail = card[delim + 3 :].lstrip("\n")
    end = tail.find("\n---")
    if end < 0:
        raise ValueError("card has no closing YAML delimiter")
    return tail[:end]


def _parse_simple_yaml_front_matter(block: str) -> dict | None:
    """Parse a strictly-bounded front matter produced by this module.

    The parser is deliberately limited to the subset of YAML 1.2 we emit:
    ``key: scalar`` (the value is everything after the colon up to EOL,
    stripped), block sequences ``- value`` at column 0 (the only kind of
    sequence we emit), and the flow-style empty list ``[]``. Strings
    quoted with double quotes have their literal preserved.  Nested
    mapping blocks (e.g. ``dataset_info:`` followed by indented content)
    are detected and recorded as a list of stripped lines so the top-level
    ``language:`` block sequence remains accessible; this is a
    restricted grammar sufficient for the test contract.
    """
    out: dict = {}
    lines = block.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        # A top-level key (no leading whitespace) with a ``:`` separator.
        if not line.startswith((" ", "\t")) and ":" in line:
            key, _, raw = line.partition(":")
            raw = raw.strip()
            if raw == "[]":
                out[key.strip()] = []
                i += 1
                continue
            if raw:
                out[key.strip()] = raw
                i += 1
                continue
            # ``key:`` with no inline value.  The next non-empty line may
            # either start with ``-`` (block sequence) or be indented
            # (nested mapping block, e.g. ``dataset_info:``).  Collect a
            # block sequence on the immediately following lines, or
            # capture a nested mapping as a list of stripped lines.
            block_items: list = []
            nested_lines: list = []
            j = i + 1
            while j < len(lines):
                seq = lines[j]
                stripped = seq.lstrip()
                if stripped.startswith("-"):
                    content = stripped[1:].strip()
                    if content.startswith('"') and content.endswith('"'):
                        content = content[1:-1]
                    block_items.append(content)
                    j += 1
                    continue
                # An indented (non ``-``) line is a nested mapping block.
                if seq.startswith((" ", "\t")) and seq.strip():
                    nested_lines.append(seq.strip())
                    j += 1
                    continue
                # Any other non-empty line ends the block sequence.
                if seq.strip():
                    break
                j += 1
            if block_items:
                out[key.strip()] = block_items
            elif nested_lines:
                out[key.strip()] = nested_lines
            else:
                out[key.strip()] = []
            i = j
            continue
        # Anything else is unexpected for this restricted grammar.
        return None
    return out


class TestYAMLSequenceStructure:
    def test_empty_languages_renders_as_empty_flow_list(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            render_dataset_card,
        )

        card = render_dataset_card(_stats_with({}))
        fm = _yaml_front_matter(card)
        parsed = _parse_simple_yaml_front_matter(fm)
        assert parsed is not None
        assert parsed["language"] == []

    def test_single_language_renders_as_yaml_block_sequence(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            render_dataset_card,
        )

        card = render_dataset_card(_stats_with({"en": 1}))
        fm = _yaml_front_matter(card)
        # The front matter must form a valid ``key:\n- "x"`` pattern,
        # not ``key: - "x"`` (which is malformed for zero or many
        # values).
        assert "language:" in fm
        assert "language: " not in fm or '\n- "en"' in fm
        parsed = _parse_simple_yaml_front_matter(fm)
        assert parsed is not None
        assert parsed["language"] == ["en"]

    def test_multiple_languages_render_as_yaml_block_sequence(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            render_dataset_card,
        )

        card = render_dataset_card(_stats_with({"en": 1, "fr": 1, "de": 1}))
        fm = _yaml_front_matter(card)
        parsed = _parse_simple_yaml_front_matter(fm)
        assert parsed is not None
        assert sorted(parsed["language"]) == ["de", "en", "fr"]

    def test_quoted_value_with_pipe_does_not_break_yaml(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            render_dataset_card,
        )

        card = render_dataset_card(_stats_with({"zh: 北京 | 東京": 1}))
        fm = _yaml_front_matter(card)
        parsed = _parse_simple_yaml_front_matter(fm)
        assert parsed is not None
        assert parsed["language"] == ["zh: 北京 | 東京"]


# ---------------------------------------------------------------------------
# Export helper audit: it must use the supplied dataset
# ---------------------------------------------------------------------------


class TestExportTestHelperAudit:
    """Demonstrates that the previous ``_export`` helper silently replaced
    the supplied dataset. With the helper fixed, the exporter must be
    called with the dataset the test built; we verify that by reading the
    exported ``input_dataset_revision`` and checking it equals the test's
    chosen revision. The pre-fix helper embedded a hard-coded revision
    and would have reported the wrong value, exposing the regression.
    """

    def test_supplied_dataset_is_what_gets_exported(self):
        from osm_polygon_sentence_relevance.output import export_finalized_dataset

        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="a")],
            revision="THIS-IS-MY-REV-12345",
            version="MY-VERSION-6",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            export_finalized_dataset(ds, tmpdir)
            manifest = (Path(tmpdir) / "manifest.json").read_text(encoding="utf-8")
            assert "THIS-IS-MY-REV-12345" in manifest
            assert "MY-VERSION-6" in manifest

    def test_helper_previously_hidden_revision_unique_is_now_exported(self):
        """A test using a unique revision used to pass only because the
        helper silently swapped the dataset for a stub. After the fix,
        the same call exports the test's own dataset (so the unique
        revision appears in the manifest)."""
        from osm_polygon_sentence_relevance.output import export_finalized_dataset

        sentinel = "AUDIT-SENTINEL-REVISION-DO-NOT-SHORTEN"
        ds = _finalize(
            [make_segmented_row(sentence_text_normalized="x")],
            revision=sentinel,
            version="v-x",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            res = export_finalized_dataset(ds, tmpdir)
            text = Path(res.manifest_path).read_text(encoding="utf-8")
            assert sentinel in text


# ---------------------------------------------------------------------------
# DatasetStatistics immutability
# ---------------------------------------------------------------------------


class TestDatasetStatisticsImmutability:
    def test_mutating_source_counts_after_construction_fails(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            DatasetStatistics,
        )

        stats = DatasetStatistics(
            version=1,
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
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        # Mutation must fail; a plain dict would silently accept it.
        with pytest.raises(TypeError):
            stats.source_counts["wikipedia"] = 999
        with pytest.raises(TypeError):
            stats.language_counts["en"] = 999
        with pytest.raises(TypeError):
            stats.region_counts["a"] = 999

    def test_dataclass_attribute_assignment_still_fails(self):
        from dataclasses import FrozenInstanceError

        from osm_polygon_sentence_relevance.output.dataset_card import (
            DatasetStatistics,
        )

        stats = DatasetStatistics(
            version=1,
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
            parquet_sha256="0" * 64,
        )
        with pytest.raises(FrozenInstanceError):
            stats.row_count = 5  # type: ignore[misc]

    def test_backing_dict_mutation_does_not_leak_through_mapping_proxy(self):
        """A ``MappingProxyType`` is a view over its backing dict; if the
        caller retains that backing dict and mutates it later, the value
        stored inside ``DatasetStatistics`` must still be untouched.
        ``__post_init__`` must take a defensive copy of any input value
        before wrapping it, regardless of whether the input is already
        a proxy or an ordinary dict.
        """
        from types import MappingProxyType

        from osm_polygon_sentence_relevance.output.dataset_card import (
            DatasetStatistics,
        )

        backing_source: dict[str, int] = {"wikipedia": 1}
        backing_language: dict[str, int] = {"en": 1}
        backing_region: dict[str, int] = {"a": 1}
        stats = DatasetStatistics(
            version=1,
            row_count=1,
            unique_sentence_ids=1,
            unique_polygons=1,
            unique_wikidata_entities=1,
            unique_documents=1,
            source_counts=MappingProxyType(backing_source),
            language_counts=MappingProxyType(backing_language),
            region_counts=MappingProxyType(backing_region),
            rows_with_coordinates=1,
            rows_without_coordinates=0,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        # Caller still holds the backing dicts and mutates them.
        backing_source["wikipedia"] = 999
        backing_source["newkey"] = 42
        backing_language["en"] = 999
        backing_region["a"] = 999
        # ``DatasetStatistics`` content is unchanged: the post-init
        # copies are independent of the caller's backing storage.
        assert dict(stats.source_counts) == {"wikipedia": 1}
        assert "newkey" not in stats.source_counts
        assert dict(stats.language_counts) == {"en": 1}
        assert dict(stats.region_counts) == {"a": 1}

    def test_defensive_copy_for_plain_dict_input(self):
        """A plain dict passed in is also defensively copied: subsequent
        mutation of the caller's dict must not leak into the stored
        mapping. This guards against accidental aliasing in the original
        code path that already took ``dict(value)`` before wrapping.
        """
        from osm_polygon_sentence_relevance.output.dataset_card import (
            DatasetStatistics,
        )

        source_in: dict[str, int] = {"wikipedia": 1}
        language_in: dict[str, int] = {"en": 1}
        region_in: dict[str, int] = {"a": 1}
        stats = DatasetStatistics(
            version=1,
            row_count=1,
            unique_sentence_ids=1,
            unique_polygons=1,
            unique_wikidata_entities=1,
            unique_documents=1,
            source_counts=source_in,
            language_counts=language_in,
            region_counts=region_in,
            rows_with_coordinates=1,
            rows_without_coordinates=0,
            input_dataset_revision="r",
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        source_in["wikipedia"] = 999
        source_in["newkey"] = 42
        assert dict(stats.source_counts) == {"wikipedia": 1}
        assert "newkey" not in stats.source_counts


# ---------------------------------------------------------------------------
# Dynamic revision/pipeline-version rendering safety
# ---------------------------------------------------------------------------


class TestRevisionAndVersionEscaping:
    def test_revision_with_backticks_is_escaped_in_render(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            DatasetStatistics,
            render_dataset_card,
        )

        hostile = "abc``````def"
        stats = DatasetStatistics(
            version=1,
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
            input_dataset_revision=hostile,
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        card = render_dataset_card(stats)
        # Raw value appears nowhere in the rendered card; backticks are
        # rendered via ``&#96;`` so the surrounding `` `` `` `` `` ``
        # pair cannot be broken. The value appears twice in the card
        # (the summary bullet and the source-dataset paragraph), so the
        # total entity count is exactly twice the raw count.
        assert hostile not in card
        assert card.count("&#96;") == hostile.count("`") * 2

    def test_revision_with_pipes_or_backslashes_is_escaped(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            DatasetStatistics,
            render_dataset_card,
        )

        hostile = "abc|with-pipes\\and\\backslashes"
        stats = DatasetStatistics(
            version=1,
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
            input_dataset_revision=hostile,
            pipeline_version="v",
            parquet_sha256="0" * 64,
        )
        card = render_dataset_card(stats)
        # Restrict the check to the two inline code spans that hold the
        # revision string; the rest of the card legitimately contains
        # pipes for Markdown table syntax.
        marker = "- **Input dataset revision:** `"
        idx = card.index(marker) + len(marker)
        end = card.index("`", idx)
        span = card[idx:end]
        assert "|" not in span
        assert "\\" not in span
        assert "&#124;" in span
        assert "&#92;" in span

    def test_revision_with_newline_or_cr_is_escaped(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            DatasetStatistics,
            render_dataset_card,
        )

        for hostile in ["newline\npresent", "carriage\rreturn"]:
            stats = DatasetStatistics(
                version=1,
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
                input_dataset_revision=hostile,
                pipeline_version="v",
                parquet_sha256="0" * 64,
            )
            card = render_dataset_card(stats)
            # The raw newline/CR must not appear inside the inline code
            # span area where the revision is rendered. The card already
            # contains some structural newlines (between sections); the
            # safe form ``&#10;`` / ``&#13;`` is the only place the raw
            # char can be substituted.
            # Look at the ``Input dataset revision`` bullet specifically.
            marker = "- **Input dataset revision:** `"
            idx = card.index(marker) + len(marker)
            end = card.index("`", idx)
            span = card[idx:end]
            assert "\n" not in span
            assert "\r" not in span


# ---------------------------------------------------------------------------
# Initial statistics version is 1 (Phase 8C is not yet released)
# ---------------------------------------------------------------------------


class TestStatisticsVersionIsInitialRelease:
    def test_statistics_version_is_one(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            STATISTICS_VERSION,
        )

        assert STATISTICS_VERSION == 1


# ---------------------------------------------------------------------------
# Factual wording tightening
# ---------------------------------------------------------------------------


class TestFactualWordingPreserved:
    def test_card_does_not_imply_global_immutability_of_revision(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            DatasetStatistics,
            render_dataset_card,
        )

        stats = DatasetStatistics(
            version=1,
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
            parquet_sha256="0" * 64,
        )
        # Normalize internal whitespace so phrases spanning a line break
        # in the rendered card still match.
        card = " ".join(render_dataset_card(stats).lower().split())
        # The legacy "in hub mode" wording was replaced by the Hub/local
        # branch introduced in Phase 8C source-provenance completion.
        # For local input (no ``input_dataset_id``), the card explicitly
        # states that no Hub commit SHA is implied.
        assert "no hub commit sha is implied" in card
        # The bare revision wording is honored exactly.
        assert "recorded input revision" in card
        assert "exact, immutable revision" not in card

    def test_zero_width_list_is_narrow(self):
        from osm_polygon_sentence_relevance.output.dataset_card import (
            render_dataset_card,
        )

        card = render_dataset_card(_stats_with({}))
        # Card mentions zero-width removal but does not claim all zero-width
        # code points are removed.
        assert "zero-width" in card.lower()
        # Phrase ``all zero-width code points`` is too broad.
        assert "all zero-width" not in card.lower()
        assert "all zero width" not in card.lower()
