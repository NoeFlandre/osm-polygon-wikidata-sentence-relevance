# Data Contract

This document defines the input/output data the pipeline consumes and
produces. It is authoritative for *current* behavior through Phase 6C.

## Authoritative input paths

The pipeline reads from exactly six subdirectory trees under the input root.
The allowlist is defined in `osm_polygon_sentence_relevance.constants`
(`ALLOWED_INPUT_PATHS`). The downstream acquisition uses these patterns
when filtering the Hugging Face snapshot download.

| Subdirectory | Logical table | Notes |
|---|---|---|
| `polygons/` | `polygons` | OSM polygon metadata, one row per polygon. |
| `polygon_articles/` | `polygon_articles` | Polygon ↔ article link table. |
| `wikipedia/documents/` | `wikipedia_documents` | Wikipedia document metadata. |
| `wikipedia/sections/` | `wikipedia_sections` | Wikipedia section text (joined via article_id). |
| `wikivoyage/documents/` | `wikivoyage_documents` | Wikivoyage document metadata. |
| `wikivoyage/sections/` | `wikivoyage_sections` | Wikivoyage section text (joined via wikidata QID). |

> **Excluded by design.** The obsolete `articles/` directory is
> **intentionally excluded** and must never be used. Both
> `acquire_dataset_snapshot` (`IGNORE_PATTERNS`) and the local
> `osm_polygon_sentence_relevance.constants` allowlist enforce this.

## Afghanistan schema as authoritative structure

The input dataset is published as the
[`NoeFlandre/osm-polygon-wikidata-only`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only)
parquet-per-table layout.  Project-local PyArrow schemas (in
`osm_polygon_sentence_relevance/schemas.py`) are the retained upstream
shape: they are treated as a contract that the input data is validated
against in `loading.py` (`load_validated_table`) before any join runs.

## Wikipedia and Wikivoyage source policy

Both sources are in scope. Each row in the joined occurrence table carries
a fixed `source` label drawn from `ALLOWED_SOURCES`:

- `"wikipedia"` — joined via `polygon_articles.article_id`.
- `"wikivoyage"` — joined via shared `wikidata` QID; empty `article_id`
  in the source row is normalized to `null` during the Wikivoyage join.

Join contracts differ; therefore the implementations are intentionally
separate (see `docs/architecture.md`).

## Sentence normalization and context policy

Sentence-level processing keeps a strict distinction between raw and
normalized text.

`normalize_sentence(text)` performs the following steps in order:

1. Reject non-string input.
2. Unicode NFC normalization.
3. Remove the zero-width characters U+200B, U+2060, U+FEFF.
4. **Preserve** U+200C (ZWNJ) and U+200D (ZWJ).
5. Replace Unicode `Cc` control characters with ASCII spaces.
6. Collapse runs of Unicode whitespace to one ASCII space; trim.
7. Remove consecutive leading MediaWiki edit markers of the form
   `[ text | text ]` whose closing bracket occurs within 120 leading
   characters and whose bracketed content contains a `|`.
8. Collapse and trim whitespace again.

`normalize_sentence` preserves case, punctuation, accents, ZWNJ, and ZWJ.
It does **not** lowercase text.

In `segmentation.py._prepare_section`, `sentence_text_raw` is the
**trimmed** sentence segment emitted by the segmenter — it is *not* the
complete section input. The full section text is consumed once by the
segmenter, and the segmenter's emitted string is what becomes
`sentence_text_raw`. Its `sentence_text_normalized` is
`normalize_sentence(sentence_text_raw)`.

Context policy:

- `previous_sentence` and `next_sentence` are computed **before**
  deduplication.
- Context group key: `(polygon_id, source, document_id, section_id)`.
- Sort key *within* each context group:
  `(sentence_index, stable complete-row representation)`.
- `previous_sentence` of row *i* is `sentence_text_normalized` of the row
  immediately above *i* in the sorted group, or `null` for *i = 0*.
- `next_sentence` of row *i* is `sentence_text_normalized` of the row
  immediately below *i*, or `null` for the last row.

This is **not** the same as the Phase 2 join sort
`(polygon_id, source, language, document_id, section_index, section_id)`,
and it is **not** the same as the dedup-group key.

## Deduplication key and canonical-source policy

- Dedup key: `(polygon_id, language, sentence_text_normalized)`.
- For every group, the canonical occurrence is chosen by this tiebreaker,
  in order:
  1. Wikipedia (`"wikipedia"`) before Wikivoyage (`"wikivoyage"`).
  2. `document_id` ascending.
  3. `section_index` ascending.
  4. `section_id` ascending.
  5. `sentence_index` ascending.
  6. Stable row representation (used as a final tiebreaker for fully
     identical rows).
- `duplicate_occurrence_count_removed` records how many occurrences were
  collapsed into the kept one.
- `cross_source_duplicate_group_count` records how many dedup groups
  contained both Wikipedia and Wikivoyage occurrences.

## Deterministic sentence and content IDs

- `sentence_content_hash` — SHA-256 hex (lower-case) over the canonical
  normalized sentence string.
- `deterministic_sentence_id` — lower-case SHA-256 hex of the **compact
  canonical JSON** of exactly:
  ```json
  {"version": 1, "polygon_id": "...", "language": "...", "sentence_content_hash": "..."}
  ```
  emitted with `sort_keys=True`, `separators=(",", ":")`,
  `ensure_ascii=False`, encoded as UTF-8 before the hash.

  The payload does **not** include `input_dataset_revision` or
  `pipeline_version`. Those values are provenance / schema metadata and
  are recorded in the manifest and in the Parquet schema metadata, but
  they are not part of the sentence ID itself, so changing the input
  revision or pipeline version does **not** change `sentence_id`.

  Because the sentence ID depends only on `(polygon_id, language,
  sentence_content_hash)`, two runs with identical inputs and locked
  dependencies produce identical `sentence_id` values for the same
  `(polygon_id, language, normalized_text)` triple — without relying on
  the manifest for the comparison.

## Output schema ownership and compatibility expectations

The output table conforms to `OUTPUT_SENTENCE_SCHEMA` in
`schemas.py`. There is currently **no separate schema-version field**;
the schema is identified by:

- The committed `OUTPUT_SENTENCE_SCHEMA` PyArrow contract in
  `osm_polygon_sentence_relevance/schemas.py`.
- The package/code revision (git commit).
- `manifest.pipeline_version` and `manifest.input_dataset_revision`.
- The Parquet `sha256` recorded in the manifest.

Owner discipline:

- `sentence_id` is the primary key for downstream joiners.
- The exported dataset's manifest is the authoritative machine-readable
  description of the produced Parquet bytes; consumers should not rely
  on external documentation in place of the manifest.
- All SHA-256 hex digests are lower-case.

## Compatibility expectations for consumers

- The set of field names, types, and nullability in
  `OUTPUT_SENTENCE_SCHEMA` is stable for a given package/code revision
  (a git tag or commit) plus a given `pipeline_version`.
- Consumers should pin the package revision, the `pipeline_version`,
  the `input_dataset_revision`, and treat the manifest `sha256` as the
  byte-level fingerprint for the produced dataset.
- The pipeline will not silently change content for the same
  `(input_dataset_revision, pipeline_version, code revision)` triple:
  deterministic ordering plus content-addressable IDs
  (`sentence_id`, `sentence_content_hash`) make accidental change
  visible.

## Out-of-scope behaviors (current)

The following are **not** part of the data contract today:

- Hugging Face upload/publishing of the produced dataset.
- Sentence classification or labeling.
- Concurrency, resumability, or incremental builds.
