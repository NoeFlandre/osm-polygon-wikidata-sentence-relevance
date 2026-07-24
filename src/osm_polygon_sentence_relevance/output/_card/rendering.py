"""Deterministic Hugging Face dataset-card rendering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import pyarrow as pa

from osm_polygon_sentence_relevance.contracts.schemas import OUTPUT_SENTENCE_SCHEMA

from .statistics import STATISTICS_VERSION, DatasetStatistics

_CARD_NAME = "README.md"


def _yaml_quote_scalar(value: str) -> str:
    """Quote a YAML scalar so special characters cannot break the sequence.

    Emits a double-quoted scalar with backslashes and double quotes
    escaped; non-printable characters are JSON-style escaped so the
    quoting is valid in standard YAML 1.2 libraries as well as in the
    Hugging Face dataset-card parser. Newlines are escaped as ``\\n``
    rather than emitted literally so the rendering stays single-line per
    YAML sequence entry.
    """
    safe = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\x00", "\\0")
        .replace("\x0b", "\\v")
        .replace("\x0c", "\\f")
    )
    return f'"{safe}"'


def _yaml_features_block(stats: DatasetStatistics) -> str:
    """Render the ``dataset_info.features`` block for the YAML front matter.

    The schema is declared in deterministic order so two ``DatasetStatistics``
    inputs with the same logical content produce byte-identical YAML.
    ``osm_tags`` is declared as a ``Sequence`` of ``{key, value}`` structs
    because the Hugging Face Dataset Viewer cannot process the parquet
    ``map<string, string>`` type; the parquet file is encoded to use the
    same logical representation so the declared features match the actual
    bytes.
    """
    feature_lines = [
        "  - name: sentence_id",
        "    dtype: string",
        "  - name: polygon_id",
        "    dtype: string",
        "  - name: wikidata",
        "    dtype: string",
        "  - name: document_id",
        "    dtype: string",
        "  - name: article_id",
        "    dtype: string",
        "  - name: source",
        "    dtype: string",
        "  - name: language",
        "    dtype: string",
        "  - name: site",
        "    dtype: string",
        "  - name: page_title",
        "    dtype: string",
        "  - name: section_id",
        "    dtype: string",
        "  - name: section_index",
        "    dtype: int64",
        "  - name: section_path",
        "    sequence: string",
        "  - name: sentence_index",
        "    dtype: int64",
        "  - name: sentence_text_raw",
        "    dtype: string",
        "  - name: sentence_text_normalized",
        "    dtype: string",
        "  - name: previous_sentence",
        "    dtype: string",
        "  - name: next_sentence",
        "    dtype: string",
        "  - name: url",
        "    dtype: string",
        "  - name: page_id",
        "    dtype: int64",
        "  - name: revision_id",
        "    dtype: int64",
        "  - name: revision_timestamp",
        "    dtype: string",
        "  - name: document_content_hash",
        "    dtype: string",
        "  - name: section_content_hash",
        "    dtype: string",
        "  - name: sentence_content_hash",
        "    dtype: string",
        "  - name: duplicate_occurrence_count",
        "    dtype: int64",
        "  - name: duplicate_sources",
        "    sequence: string",
        "  - name: polygon_name",
        "    dtype: string",
        "  - name: osm_primary_tag",
        "    dtype: string",
        "  - name: osm_tags",
        "    sequence:",
        "    - name: key",
        "      dtype: string",
        "    - name: value",
        "      dtype: string",
        "  - name: region",
        "    dtype: string",
        "  - name: lat",
        "    dtype: float64",
        "  - name: lon",
        "    dtype: float64",
        "  - name: input_dataset_revision",
        "    dtype: string",
        "  - name: pipeline_version",
        "    dtype: string",
    ]
    return "\n".join(feature_lines)


def _yaml_front_matter(stats: DatasetStatistics) -> str:
    """Render valid Hugging Face dataset-card YAML front matter.

    Emits YAML 1.2 block-sequence form for the language list so a
    standard parser produces the same ``list[str]`` in every case:
    ``language:`` followed by ``- "x"`` lines, or ``language: []`` when
    the list is empty. Strings that contain ``:``, ``|``, ``"``, ``\\``,
    or leading/trailing whitespace are emitted as double-quoted scalars to
    keep the rendering parseable and deterministic without a YAML
    dependency.  Includes a ``dataset_info`` block (with ``splits`` before
    ``features``) so the Hugging Face Dataset Viewer can interpret
    ``osm_tags`` as a ``Sequence`` of ``{key, value}`` structs (the
    parquet ``map`` type is not supported by ``datasets``).  ``splits``
    appears before ``features`` so the Hub's strict YAML parser does not
    reject a sibling-key dedent after a deeply-nested feature item.
    """
    if stats.language_counts:
        languages_block = "\n".join(
            f"- {_yaml_quote_scalar(lang)}" for lang in stats.language_counts
        )
        language_section = f"language:\n{languages_block}"
    else:
        language_section = "language: []"
    features_block = _yaml_features_block(stats)
    splits_block = (
        f"  splits:\n"
        f"  - name: train\n"
        f"    num_examples: {stats.row_count}\n"
        f"  features:\n"
        f"{features_block}"
    )
    lines = [
        "---",
        "license: other",
        "pretty_name: OSM Polygon Wikidata Sentence Relevance",
        language_section,
        "dataset_info:",
        splits_block,
        "---",
    ]
    return "\n".join(lines)


def _escape_md_cell(value: str) -> str:
    """Escape a value for safe inclusion in a Markdown table cell.

    Pipes, backslashes, carriage returns, and newlines are replaced with
    HTML character references so the cell cannot break table syntax even
    if the value contains ``|``, ``\\``, ``\\r``, or a literal newline.
    """
    return (
        value.replace("&", "&amp;")
        .replace("|", "&#124;")
        .replace("\\", "&#92;")
        .replace("\r", "&#13;")
        .replace("\n", "&#10;")
        .replace("\t", "&#9;")
    )


def _escape_md_inline(value: str) -> str:
    """Escape a value for safe inclusion inside Markdown inline code.

    Backticks, pipes, backslashes, and control characters are escaped so
    the surrounding `` ` `` `` ` `` pair cannot be broken or split by an
    adversarial revision/version string. The literal text is preserved by
    HTML-entity substitution rather than removal, so the rendered card
    still shows the raw value to a human reader.
    """
    return (
        value.replace("&", "&amp;")
        .replace("`", "&#96;")
        .replace("|", "&#124;")
        .replace("\\", "&#92;")
        .replace("\r", "&#13;")
        .replace("\n", "&#10;")
        .replace("\t", "&#9;")
    )


def _counts_table(title: str, counts: Mapping[str, int]) -> str:
    """Render a deterministic Markdown breakdown table (rows sorted).

    Cell content is HTML-entity-escaped so a key containing a pipe,
    backslash, or newline cannot break the Markdown table structure.
    """
    if not counts:
        return f"### {title}\n\n_No values._\n"
    header = f"### {title}\n\n| Key | Count |\n| --- | --- |"
    body = "\n".join(
        f"| {_escape_md_cell(k)} | {v} |" for k, v in sorted(counts.items())
    )
    return f"{header}\n{body}\n"


def _quote_url_component_dataset_id(value: str) -> str:
    """Percent-encode an upstream dataset identifier for the Hub URL.

    Uses :func:`urllib.parse.quote` with ``safe="/"`` so ``owner/repo``
    style identifiers pass through unchanged, but every other character
    including Unicode code points (which are converted to UTF-8 first)
    is encoded as ``%XX``. This makes the URL component immune to
    smuggled ``?``, ``#``, ``%``, backslashes, brackets, parens, spaces,
    accented characters, and CJK code points.

    The documented slash handling intentionally diverges from the
    ambiguous "Unicode alnum" behavior of an earlier implementation.
    """
    return quote(value, safe="/")


def _quote_url_component_revision(value: str) -> str:
    """Percent-encode a Hub revision path component.

    Uses :func:`urllib.parse.quote` with ``safe=""`` so every path
    separator (``/``), query marker (``?``), fragment marker (``#``),
    percent sign, backslash, bracket, and paren, along with every
    non-ASCII code point (UTF-8 encoded), is percent-encoded. The Hub
    renders the revision as a single segment after ``/tree/``; this
    function makes that contract explicit.
    """
    return quote(value, safe="")


def _escape_md_link_label(value: str) -> str:
    """Escape a value for use as the visible text of a Markdown link.

    Markdown link labels terminate on the first ``]``; the URL inside
    ``(...)`` terminates on the closing ``)``. Both must be
    entity-escaped when the value is untrusted, along with the markers
    that break surrounding inline code spans and the control characters
    that confuse some Markdown renderers.

    The exact entity substitution is deterministic and uses HTML numeric
    character references (``&#NN;``) so any Markdown parser recovers the
    original text on render. Currently escaped:

    - ``&`` → ``&amp;`` (must be first);
    - ``[`` → ``&#91;`` and ``]`` → ``&#93;`` (cannot be skipped);
    - `` `` ` `` → ``&#96;`` (cannot be skipped);
    - ``\\`` → ``&#92;``;
    - ``\r`` / ``\n`` / ``\t`` → ``&#13;`` / ``&#10;`` / ``&#9;``.
    """
    return (
        value.replace("&", "&amp;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("`", "&#96;")
        .replace("\\", "&#92;")
        .replace("\r", "&#13;")
        .replace("\n", "&#10;")
        .replace("\t", "&#9;")
    )


def _markdown_dataset_link(label: str, dataset_id: str) -> str:
    """Build a safely-encoded Markdown link to a Hub dataset page.

    The label is escaped via ``_escape_md_link_label`` so an adversarial
    dataset identifier cannot terminate the link (``[``/``]``) or break
    the surrounding Markdown (backticks/backslashes/CR/LF). The URL
    component is percent-encoded with ``safe="/"`` via
    :func:`urllib.parse.quote` so an adversarial identifier cannot inject
    a URL path or break the URL.
    """
    url = (
        f"https://huggingface.co/datasets/{_quote_url_component_dataset_id(dataset_id)}"
    )
    return f"[{_escape_md_link_label(label)}]({url})"


def _source_provenance_section(stats: DatasetStatistics) -> str:
    """Render the "Source dataset and recorded input revision" section.

    Branch on the recorded ``input_dataset_id``:

    - **Hub mode** (``stats.input_dataset_id is not None``): the
      dataset identifier is shown with a Markdown link to the Hugging
      Face dataset page. When the recorded revision looks like a
      40-character lowercase hex SHA-1, a second link to the
      commit/tree page is also shown. The wording describes the value
      factually ("recorded immutable commit identifier" /
      "recorded reference") and never infers which acquisition
      mechanism produced it.
    - **Local mode** (``stats.input_dataset_id is None``): the card
      states explicitly that the build used a local input snapshot
      with no recorded Hub dataset ID and that the recorded revision
      string is whatever the operator configured at export time.
    """
    revision_rendered = _escape_md_inline(stats.input_dataset_revision)
    dataset_id = stats.input_dataset_id
    if dataset_id:
        # Optional revision-tree link, only when the revision actually
        # looks like an immutable commit SHA-1 (40 lowercase hex chars).
        import re as _re

        if _re.fullmatch(r"[0-9a-f]{40}", stats.input_dataset_revision.strip()):
            dataset_link = _markdown_dataset_link(dataset_id, dataset_id)
            tree_url = (
                "https://huggingface.co/datasets/"
                f"{_quote_url_component_dataset_id(dataset_id)}/tree/"
                f"{_quote_url_component_revision(stats.input_dataset_revision)}"
            )
            tree_link = (
                f"[{_escape_md_link_label(stats.input_dataset_revision)}]({tree_url})"
            )
            return (
                "The sentences were extracted from the input dataset "
                f"{dataset_link} at the recorded input revision "
                f"{tree_link}. This value is a recorded immutable "
                "commit identifier; the card does not assume which "
                "acquisition mechanism produced it."
            )
        return (
            "The sentences were extracted from the input dataset "
            f"{_markdown_dataset_link(dataset_id, dataset_id)} at the "
            f"recorded input revision `{revision_rendered}`. This value "
            "is a recorded reference; it is reproduced exactly as it "
            "was supplied to the build."
        )
    # Local mode: no Hub dataset ID was recorded.
    return (
        "This build used a local input snapshot: there is no recorded "
        "Hugging Face dataset ID for the upstream source. The recorded "
        f"input revision `{revision_rendered}` is the value the "
        "operator configured at export time and is preserved exactly "
        "as recorded; no Hub commit SHA is implied."
    )


def _single_region_preview_section(stats: DatasetStatistics) -> str:
    """Render a prominent "Dataset scope" section when exactly one region
    is in the export.

    Returns ``""`` when ``region_counts`` has zero or multiple keys so
    multi-region and empty exports do not advertise a misleading
    preview scope.  All figures are derived from ``stats``; the region
    display name is the title-cased form of the single region key.
    """
    if len(stats.region_counts) != 1:
        return ""
    region_key, region_rows = next(iter(stats.region_counts.items()))
    display_name = region_key.title()
    sha = stats.parquet_sha256
    revision = _escape_md_inline(stats.input_dataset_revision)
    sha_inline = _escape_md_inline(sha)
    return f"""\
## Dataset scope

This published artifact is the **{display_name}-only preview** of the
OSM Polygon Wikidata Sentence Relevance dataset. It contains
{stats.row_count} deduplicated sentence rows extracted from
{stats.unique_polygons} unique OSM polygons in the
{_escape_md_inline(region_key)} shard, covering
{stats.unique_wikidata_entities} Wikidata entities and
{stats.unique_documents} unique documents. The full multi-region
dataset is published incrementally; the current artifact covers a
single region only and is intended as a canary/validation snapshot
of the production export pipeline.

- **Region:** {display_name}
- **Region key in this preview:** `{_escape_md_inline(region_key)}`
- **Region rows:** {region_rows}
- **Preview polygons:** {stats.unique_polygons}
- **Preview Wikidata entities:** {stats.unique_wikidata_entities}
- **Preview documents:** {stats.unique_documents}
- **Recorded input revision:** `{revision}`
- **Preview Parquet SHA-256:** `{sha_inline}`
"""


def render_dataset_card(stats: DatasetStatistics) -> str:
    """Render the deterministic Hugging Face dataset card.

    The card is derived entirely from ``stats`` (which itself comes from
    the finalized table). It includes a visible note that the quantitative
    sections are generated automatically and must not be edited by hand.
    """
    with_coords = stats.rows_with_coordinates
    without_coords = stats.rows_without_coordinates
    total_coords = with_coords + without_coords

    source_section = _counts_table("Source coverage", stats.source_counts)
    language_section = _counts_table("Language coverage", stats.language_counts)
    region_section = _counts_table("Region coverage", stats.region_counts)
    preview_section = _single_region_preview_section(stats)

    return f"""\
{_yaml_front_matter(stats)}

<!-- GENERATED AUTOMATICALLY DURING EXPORT. DO NOT EDIT MANUALLY. -->
<!-- The quantitative sections below are computed from the exported Parquet -->
<!-- data and must be regenerated whenever the dataset is rebuilt. -->

# OSM Polygon Wikidata Sentence Relevance

This dataset contains normalized sentences extracted from OpenStreetMap
(OSM) polygons and their linked Wikipedia / Wikivoyage pages.
Each row is a deduplicated sentence occurrence scoped to a polygon,
language, and content hash.

{preview_section}## Dataset summary

- **Total sentence rows:** {stats.row_count}
- **Unique sentence IDs:** {stats.unique_sentence_ids}
- **Unique polygons:** {stats.unique_polygons}
- **Unique Wikidata entities:** {stats.unique_wikidata_entities}
- **Unique document identities
  (source, site, language, document_id):** {stats.unique_documents}
- **Rows with coordinates:** {with_coords} / {total_coords}
- **Rows without coordinates:** {without_coords} / {total_coords}
- **Input dataset revision:** `{_escape_md_inline(stats.input_dataset_revision)}`
- **Pipeline version:** `{_escape_md_inline(stats.pipeline_version)}`
- **Exported Parquet SHA-256:** `{_escape_md_inline(stats.parquet_sha256)}`

## Source dataset and recorded input revision

{_source_provenance_section(stats)}

## Wikipedia and Wikivoyage coverage

{source_section}
This dataset combines Wikipedia and Wikivoyage provenance. Wikipedia and
Wikivoyage rows may describe the same Wikidata entity; cross-source
duplicates are collapsed to a single canonical occurrence during
finalization (see *Deterministic IDs, deduplication, and context policy*).
A "document identity" is defined as the tuple
`(source, site, language, document_id)`; a row in one language, site, or
source is not considered the same document as a row with the same raw
`document_id` but different tuple dimensions.

{language_section}
{region_section}
## Output schema (field descriptions)

The export is a single Parquet table (`sentences.parquet`) with the
following schema:

| Field | Type | Description |
| --- | --- | --- |
| `sentence_id` | string | Deterministic ID from polygon, language, and sentence content hash. |
| `polygon_id` | string | OSM polygon identifier. |
| `wikidata` | string | Wikidata entity QID for the polygon/page. |
| `document_id` | string | Document/page identifier within its source/site/language. |
| `article_id` | string (nullable) | Article identifier where available. |
| `source` | string | `wikipedia` or `wikivoyage`. |
| `language` | string | Language code of the sentence. |
| `site` | string | Source site (e.g. `en.wikipedia.org`). |
| `page_title` | string | Page title. |
| `section_id` | string | Section identifier. |
| `section_index` | int64 | Section ordinal within the document. |
| `section_path` | list<string> | Section breadcrumb path. |
| `sentence_index` | int64 | Sentence ordinal within the section. |
| `sentence_text_raw` | string | Segment text after surrounding-whitespace trimming (the segmenter may also trim internal whitespace). |
| `sentence_text_normalized` | string | Normalized sentence text used as the dedup key. |
| `previous_sentence` | string (nullable) | Prior sentence in the section (context). |
| `next_sentence` | string (nullable) | Next sentence in the section (context). |
| `url` | string | Source URL of the document. |
| `page_id` | int64 | Source page ID. |
| `revision_id` | int64 | Source revision ID. |
| `revision_timestamp` | string | Source revision timestamp. |
| `document_content_hash` | string | Hash of the source document. |
| `section_content_hash` | string | Hash of the source section. |
| `sentence_content_hash` | string | Hash of the normalized sentence (dedup key component). |
| `duplicate_occurrence_count` | int64 | Number of source occurrences collapsed into this row. |
| `duplicate_sources` | list<string> | Distinct sources among the collapsed occurrences. |
| `polygon_name` | string (nullable) | Human-readable polygon name. |
| `osm_primary_tag` | string (nullable) | Primary OSM tag of the polygon. |
| `osm_tags` | map<string,string> | OSM tags of the polygon. |
| `region` | string | Input region/extract name. |
| `lat` | float64 (nullable) | Latitude of the polygon centroid, if known. |
| `lon` | float64 (nullable) | Longitude of the polygon centroid, if known. |
| `input_dataset_revision` | string | Exact input revision recorded for reproducibility. |
| `pipeline_version` | string | Pipeline version recorded for reproducibility. |

## Sentence preprocessing and normalization

Sentences are extracted per section. Each segment emitted by the segmenter
has its surrounding whitespace trimmed and is then passed through a
fixed-order normalization pipeline that performs Unicode NFC
normalization, removes the configured zero-width characters (U+200B,
U+2060, U+FEFF), replaces Unicode control characters with spaces,
collapses whitespace, and strips consecutive leading MediaWiki edit
markers such as `[label | target]`. A marker must start at the current
leading position, contain a pipe (`|`), and close with `]` within 120
characters. The pipeline repeats this check until the next leading text is
not a valid marker, then collapses whitespace again. It **preserves case**,
punctuation,
accents, and joiner characters; only `sentence_text_normalized` (the
post-pipeline text) is used as the content hash and dedup key.
`sentence_text_raw` is the segment text after surrounding-whitespace
trimming — the downstream normalizer above is applied to produce
`sentence_text_normalized`, so `sentence_text_raw` is **not** the
original mediawiki source text.

## Deterministic IDs, deduplication, and context policy

Each sentence occurrence is assigned a deterministic `sentence_id` derived
from `polygon_id`, `language`, and the SHA-256 of the normalized text.
Exact duplicates (same polygon, language, and normalized text) are
collapsed into a single canonical occurrence. When a Wikipedia and a
Wikivoyage occurrence collide, Wikipedia is chosen as the canonical
source; the full set of contributing sources is recorded in
`duplicate_sources`. Intra-section previous/next sentences are attached
as context and never alter the identity or content of the row itself.

## Provenance and revision tracking

Every export records `input_dataset_revision` and `pipeline_version` in
both the Parquet schema metadata and the `manifest.json`. The manifest
also stores the Parquet SHA-256, so the content identity of the exact
exported bytes is verifiable independently of this card. The versioned
`statistics` object inside the manifest and the top-level count fields
are derived from the same computation; the validator rejects manifests
where they disagree.

## Intended use

This dataset supports text-research tasks over OSM polygons:
contextual analysis of how places are described across Wikipedia and
Wikivoyage, sentence-level corpus studies, and downstream modeling that
needs sentence text plus article provenance. It is intended for research
and evaluation, not as a substitute for the upstream sources. The
dataset is not a labelled dataset: it does not contain relevance labels,
similarity pairs, or classification outputs, and should not be treated as
one.

## Limitations and known biases

- Extraction depends on upstream OSM / Wikimedia availability, coverage, and
  language balance; over-represented languages and regions will be
  reflected in the statistics above.
- Coordinates are only present when the source polygon carries centroid
  information; see the coordinate counts above.
- Sentence segmentation uses an automatic multilingual model and may
  mis-segment short or mixed-script text.
- Deduplication is exact-match on the normalized text within a polygon
  and language; this collapses identical sentences only and does not in
  any way imply semantic similarity or relevance. The exporter does not
  apply semantic deduplication or sentence-pair scoring.

## Licensing

This dataset combines content from OpenStreetMap and Wikimedia projects.
The repository `LICENSE` covers only the code and this dataset-card
generator; it does not grant rights to the dataset's underlying content.
The dataset's underlying content is governed by the upstream terms of
OpenStreetMap and Wikimedia. Provenance fields and source URLs/revision
identifiers are retained in every row to support attribution; satisfying
the upstream attribution and licence requirements (such as ODbL for
OpenStreetMap contributions and CC BY-SA or project-specific terms for
Wikimedia contributions) remains the responsibility of downstream users.
No single SPDX identifier covers the combined dataset, which is why
`license: other` is used in the front matter.

## Reproducibility

Builds are deterministic for identical inputs, identical code revision,
locked dependencies, and a compatible execution environment. Re-running
the pipeline on the same immutable input revision and pipeline version
reproduces the same Parquet bytes, the same `manifest.json`, and this
auto-generated card.

---

*This dataset card was generated automatically from the exported data. The
statistics above are derived during export and must not be edited manually;
rebuild the dataset to regenerate them.*
"""


def schema_has_map_types(schema: Any) -> bool:
    """Return True if *schema* contains any ``map<...>`` field, recursively.

    The Hugging Face ``datasets`` library cannot ingest
    ``map<string, string>`` columns; any export with such a field will
    fail Viewer-side requests (``/info``, ``/parquet``).  Used by the
    validator to reject exports that have not yet been migrated.
    """
    for field in schema:
        t = field.type
        if pa.types.is_map(t):
            return True
        if pa.types.is_list(t):
            if pa.types.is_map(t.value_type):
                return True
            if pa.types.is_struct(t.value_type):
                for child in t.value_type:
                    if pa.types.is_map(child.type):
                        return True
        elif pa.types.is_struct(t):
            for child in t:
                if pa.types.is_map(child.type):
                    return True
    return False


_SCHEMA_FIELD_DESCRIPTIONS: dict[str, str] = {
    "sentence_id": (
        "Deterministic ID from polygon, language, and sentence content hash."
    ),
    "polygon_id": "OSM polygon identifier.",
    "wikidata": "Wikidata entity QID for the polygon/page.",
    "document_id": ("Document/page identifier within its source/site/language."),
    "article_id": "Article identifier where available.",
    "source": "`wikipedia` or `wikivoyage`.",
    "language": "Language code of the sentence.",
    "site": "Source site (e.g. `en.wikipedia.org`).",
    "page_title": "Page title.",
    "section_id": "Section identifier.",
    "section_index": "Section ordinal within the document.",
    "section_path": "Section breadcrumb path.",
    "sentence_index": "Sentence ordinal within the section.",
    "sentence_text_raw": ("Segment text after surrounding-whitespace trimming."),
    "sentence_text_normalized": ("Normalised sentence text used as the dedup key."),
    "previous_sentence": ("Prior sentence in the section (context)."),
    "next_sentence": "Next sentence in the section (context).",
    "url": "Source URL of the document.",
    "page_id": "Source page ID.",
    "revision_id": "Source revision ID.",
    "revision_timestamp": "Source revision timestamp.",
    "document_content_hash": "Hash of the source document.",
    "section_content_hash": "Hash of the source section.",
    "sentence_content_hash": ("Hash of the normalised sentence (dedup key component)."),
    "duplicate_occurrence_count": (
        "Number of source occurrences collapsed into this row."
    ),
    "duplicate_sources": ("Distinct sources among the collapsed occurrences."),
    "polygon_name": "Human-readable polygon name.",
    "osm_primary_tag": "Primary OSM tag of the polygon.",
    "osm_tags": (
        "OSM tags of the polygon, encoded as a list of "
        "`{key, value}` structs so the Hugging Face Viewer can "
        "ingest the export."
    ),
    "region": "Input region/extract name.",
    "lat": ("Latitude of the polygon centroid, if known."),
    "lon": ("Longitude of the polygon centroid, if known."),
    "input_dataset_revision": ("Exact input revision recorded for reproducibility."),
    "pipeline_version": ("Pipeline version recorded for reproducibility."),
}


def _profile_field_type_label(field_name: str) -> str:
    """Return the on-card type label for *field_name*."""
    f = OUTPUT_SENTENCE_SCHEMA.field(field_name)
    if pa.types.is_list(f.type):
        inner = f.type.value_type
        if pa.types.is_struct(inner):
            children = ", ".join(f"`{c.name}`: string" for c in inner)
            return f"list<struct<{children}>>"
        return f"list<{inner}>"
    return str(f.type)


def schema_field_documentation() -> list[tuple[str, str, str, str]]:
    """Return deterministic ``(name, type, nullable, description)`` rows.

    Order matches ``OUTPUT_SENTENCE_SCHEMA`` so two profiles with the
    same schema emit byte-identical documentation sections.
    """
    rows: list[tuple[str, str, str, str]] = []
    for f in OUTPUT_SENTENCE_SCHEMA:
        nullable = "yes" if f.nullable else "no"
        desc = _SCHEMA_FIELD_DESCRIPTIONS.get(f.name, "(no documentation)")
        rows.append(
            (
                f.name,
                _profile_field_type_label(f.name),
                nullable,
                desc,
            )
        )
    return rows


def _profile_schema_table() -> str:
    """Render the on-card schema field documentation table."""
    rows = schema_field_documentation()
    header = "| Field | Type | Nullable | Description |\n| --- | --- | --- | --- |"
    body = "\n".join(
        f"| `{name}` | `{type_label}` | {nullable} | {desc} |"
        for name, type_label, nullable, desc in rows
    )
    return f"{header}\n{body}\n"


def _profile_yaml(stats: DatasetStatistics, profile: Any) -> str:
    """Render the minimal valid YAML front matter for the on-card.

    Includes ``language`` block, a ``license: other`` declaration, an
    explicit ``configs`` block (with ``data_files`` pointing at
    ``sentences.parquet`` for the ``train`` split), and a
    ``dataset_info.splits`` entry so the Hugging Face Viewer can
    resolve the default config/split.  The explicit
    ``configs[].data_files`` declaration prevents the Viewer from
    interpreting ``assets/*.png`` as dataset rows (imagefolder
    inference) and pins the parquet as the only data source.

    Note: the ``dataset_info.features`` block is intentionally
    omitted.  The Parquet file embeds the canonical
    ``list<struct<{key, value}>>`` Arrow schema, and the Viewer
    uses the Parquet schema (not the YAML features block) to drive
    type-cast.  Adding a misaligned YAML features block triggers a
    ``CastError`` because the Viewer's YAML-derived schema uses
    ``Sequence<key: list<string>, value: list<string>>`` while the
    Parquet's actual schema is
    ``list<struct<key: string, value: string>>``.
    """
    if stats.language_counts:
        languages_block = "\n".join(
            f"- {_yaml_quote_scalar(lang)}" for lang in stats.language_counts
        )
        language_section = f"language:\n{languages_block}"
    else:
        language_section = "language: []"
    lines = [
        "---",
        "license: other",
        "pretty_name: OSM Polygon Wikidata Sentence Relevance",
        language_section,
        "configs:",
        "  - config_name: default",
        "    data_files:",
        "      - split: train",
        "        path: sentences.parquet",
        "dataset_info:",
        "  splits:",
        "  - name: train",
        f"    num_examples: {stats.row_count}",
        "---",
    ]
    return "\n".join(lines)


def _escape_md_inline_profile(value: str) -> str:
    """Escape inline code content for profile-based rendering."""
    return (
        value.replace("&", "&amp;")
        .replace("`", "&#96;")
        .replace("|", "&#124;")
        .replace("\\", "&#92;")
        .replace("\r", "&#13;")
        .replace("\n", "&#10;")
        .replace("\t", "&#9;")
    )


def _profile_preview_section(profile: Any) -> str:
    """Render the single-region preview section.

    Returns the empty string when the profile covers zero or multiple
    regions so multi-region and empty exports do not advertise a
    misleading preview scope.
    """
    if len(profile.region_counts) != 1:
        return ""
    region_key, _region_rows = next(iter(profile.region_counts.items()))
    display_name = region_key.replace("-latest", "").replace("-", " ").title()
    if not display_name:
        display_name = region_key
    return (
        f"## Dataset scope\n\n"
        f"Current release: **{display_name} only** "
        f"(`{_escape_md_inline_profile(region_key)}` shard).\n\n"
    )


def render_dataset_card_from_profile(
    profile: Any,
    *,
    asset_base_url: str | None = None,
) -> str:
    """Render the dataset card from an immutable ``DatasetProfile``.

    This renderer is the canonical profile-driven format. It uses
    profile-derived fields (so the card and manifest cannot drift) and
    embeds two PNG assets, a full real example row, and the schema
    field documentation. All quantitative figures are derived from the
    profile; nothing is hand-typed.
    """
    from osm_polygon_sentence_relevance.output.profile import (
        render_example_row_json,
    )

    stats = DatasetStatistics(
        version=STATISTICS_VERSION,
        row_count=profile.row_count,
        unique_sentence_ids=profile.unique_sentence_ids,
        unique_polygons=profile.unique_polygons,
        unique_wikidata_entities=profile.unique_wikidata_entities,
        unique_documents=profile.unique_documents,
        source_counts=profile.source_counts,
        language_counts=profile.language_counts,
        region_counts=profile.region_counts,
        rows_with_coordinates=profile.rows_with_coordinates,
        rows_without_coordinates=profile.rows_without_coordinates,
        input_dataset_revision=profile.input_dataset_revision,
        pipeline_version=profile.pipeline_version,
        input_dataset_id=profile.input_dataset_id,
        parquet_sha256=profile.parquet_sha256,
    )
    coords = profile.rows_with_coordinates
    total = profile.rows_with_coordinates + profile.rows_without_coordinates

    # Asset embeds. The Hugging Face Hub resolves a relative
    # ``assets/foo.png`` link against the README's location and
    # serves the bytes through its CDN; some renderers, however, do
    # not rewrite that path. To keep the README self-contained and
    # renderable everywhere, the embedded Markdown uses an explicit
    # URL (relative by default; the caller may override with
    # ``asset_base_url``) so the image loads regardless of which
    # path resolver the viewer applies.
    geo = profile.assets.get("geographic_coverage.png")
    lang_png = profile.assets.get("language_distribution.png")

    def _img(alt: str, name: str) -> str:
        if asset_base_url is None:
            return f"![{alt}](assets/{name})"
        base = asset_base_url.rstrip("/")
        return f"![{alt}]({base}/{name})"

    geo_md = (
        _img("Geographic coverage", "geographic_coverage.png")
        if geo is not None
        else "_(Geographic coverage asset unavailable.)_"
    )
    lang_md = (
        _img("Language distribution", "language_distribution.png")
        if lang_png is not None
        else "_(Language distribution asset unavailable.)_"
    )

    example_json = render_example_row_json(profile)

    schema_table = _profile_schema_table()
    preview_section = _profile_preview_section(profile)

    yaml_block = _profile_yaml(stats, profile)

    coords_extent = (
        f"{profile.lat_min:.4f}, {profile.lon_min:.4f}"
        if profile.lat_min is not None
        and profile.lat_max is not None
        and profile.lon_min is not None
        and profile.lon_max is not None
        else "(no coordinates)"
    )
    coords_extent = (
        f"{profile.lat_min:.4f} → {profile.lat_max:.4f}, "
        f"{profile.lon_min:.4f} → {profile.lon_max:.4f}"
    )

    language_table_rows = "\n".join(
        f"| `{_escape_md_inline_profile(lang)}` | {count} |"
        for lang, count in sorted(
            profile.language_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )
    )
    full_language_table = f"| Language | Rows |\n| --- | --- |\n{language_table_rows}\n"

    return f"""\
{yaml_block}

<!-- GENERATED AUTOMATICALLY DURING EXPORT. DO NOT EDIT MANUALLY. -->
<!-- The quantitative sections below are computed from the exported Parquet -->
<!-- data via the immutable DatasetProfile; rebuild the dataset to regenerate. -->

# OSM Polygon Wikidata Sentence Relevance

This dataset contains normalised sentences extracted from OpenStreetMap
(OSM) polygons and their linked Wikipedia / Wikivoyage pages.
Each row is a deduplicated sentence occurrence scoped to a polygon,
language, and content hash.

{preview_section}## Dataset summary

- **Total sentence rows:** {profile.row_count}
- **Input sentence occurrences:** {profile.input_occurrence_count}
- **Duplicates removed:** {profile.duplicates_removed}
- **Cross-source duplicate groups:** {profile.cross_source_duplicate_groups}
- **High-confidence residual boundary violations:** {profile.residual_boundary_violations}
- **Unique polygons:** {profile.unique_polygons}
- **Unique Wikidata entities:** {profile.unique_wikidata_entities}
- **Unique documents:** {profile.unique_documents}
- **Languages:** {len(profile.language_counts)}
- **Rows with coordinates:** {coords} / {total}
- **Sentence length (chars):**
  min {profile.sentence_length_min},
  mean {profile.sentence_length_mean:.2f},
  max {profile.sentence_length_max}
- **Coordinate extent:** {coords_extent}
- **Input dataset revision:** `{_escape_md_inline_profile(profile.input_dataset_revision)}`
- **Pipeline version:** `{_escape_md_inline_profile(profile.pipeline_version)}`
- **Exported Parquet SHA-256:** `{_escape_md_inline_profile(profile.parquet_sha256)}`

{_counts_table("Sources", profile.source_counts)}

## Source dataset and recorded input revision

{_source_provenance_section(stats)}

## Geographic coverage

{geo_md}

One dot represents each unique polygon with coordinates; the map extent
is derived from the data.

## Language coverage

{lang_md}

<details>
<summary>Full language breakdown ({len(profile.language_counts)} languages)</summary>

{full_language_table}
</details>

## Processing method

Sentence boundaries were produced with
`{_escape_md_inline_profile(profile.segmentation_model)}` at revision
`{_escape_md_inline_profile(profile.segmentation_revision)}`. The exact
implementation is available in the
[GitHub source at the producing commit](https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance/tree/{_escape_md_inline_profile(profile.source_commit)}).

After model inference, a conservative residual-boundary repair separates
only high-confidence punctuation boundaries across writing systems. It
preserves short abbreviations, lowercase continuations, numeric values,
and URL query strings. For Arabic-tagged text, a period boundary must
also continue in Arabic script. Publication scans every normalized sentence
with the same predicate and refuses an artifact unless the residual count is
zero.

Each emitted segment has its surrounding whitespace trimmed and is
then passed through a fixed-order normalisation pipeline:

1. Unicode NFC normalisation.
2. Removal of configured zero-width characters (`U+200B`, `U+2060`,
   `U+FEFF`).
3. Replacement of Unicode control characters with spaces.
4. Whitespace collapse.
5. Stripping of consecutive leading MediaWiki edit markers such as
   `[label | target]`. A marker must start at the current leading
   position, contain a pipe (`|`), and close with `]` within 120
   characters. The check repeats until the next leading text is not a
   valid marker, after which whitespace is collapsed again.

Case, punctuation, accents and joiner characters are preserved.
`sentence_id` is derived from `polygon_id`, language and the SHA-256 of
the normalised text. Exact duplicates within that key are collapsed;
Wikipedia is the canonical row when Wikipedia and Wikivoyage collide,
while `duplicate_sources` retains all contributing sources. Adjacent
sentences are context fields and do not affect identity.

## Real example row (from the export)

<details>
<summary>Show one row from the canonical-sorted Parquet</summary>

```json
{example_json}
```

</details>

## Schema

The export is a single Parquet table. `osm_tags` is represented as a
Viewer-compatible list of `{{key, value}}` structs.

<details>
<summary>Show all fields</summary>

{schema_table}

</details>

## Provenance and reproducibility

The Parquet metadata and `manifest.json` record the input dataset,
input revision, pipeline version, model revision, producing source
commit and artifact hashes. Identical pinned inputs and locked
dependencies produce deterministic artifacts.

## Uses and limitations

Suitable uses include multilingual corpus analysis and research on how
places are described across Wikipedia and Wikivoyage. This is not a
labelled relevance, similarity or classification dataset.

- Extraction depends on upstream OSM / Wikimedia availability,
  coverage and language balance.
- Sentence segmentation uses an automatic multilingual model. Publication
  rejects high-confidence residual boundaries, but ambiguous short or
  mixed-script text can still be imperfect.
- Deduplication is exact, not semantic, and does not imply relevance.

## Licensing

The code is covered by the repository licence. Dataset content retains
its upstream terms, including ODbL for OpenStreetMap and CC BY-SA or
project-specific Wikimedia terms. Source URLs and revision identifiers
are retained for attribution; no single SPDX identifier covers the
combined content, so the card uses `license: other`.
"""
