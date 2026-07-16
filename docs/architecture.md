# Architecture

This document describes the module ownership and data flow of the OSM Polygon
– Wikidata Sentence Relevance pipeline. It is authoritative for *current*
behavior through Phase 6C. Later phases (classification, publishing) are
explicitly out of scope and not yet implemented.

## Pipeline stages and module ownership

| Stage | Module | Responsibility |
|-------|--------|----------------|
| Schema contracts | `schemas.py` | Immutable Arrow schemas for all six input tables and the output sentence table. |
| Shard discovery | `discovery.py` | Locate per-region Parquet shards under an input root. |
| Loading | `loading.py` | Project only required columns and validate against schemas. |
| Preprocessing | `preprocessing.py` | Normalize section paths, OSM tags, and sentence text (deterministic). |
| Joins | `joins/` package | Build Wikipedia + Wikivoyage section→polygon occurrences (see below). |
| Segmentation | `segmentation.py`, `sentence_table.py` | Injectable `SentenceSegmenter`; build the segmented sentence table. |
| Finalization | `finalization.py` | Exact deduplication, deterministic IDs, metadata, validation. |
| Export | `exporter.py` facade + `_export/` | Atomic, checksummed Parquet + manifest install. |
| Acquisition | `acquisition.py` | Read-only Hugging Face snapshot acquisition. |
| Orchestration | `pipeline.py` | Tie the above stages together (injected segmenter). |
| CLI | `cli.py` | Console entry point, argument resolution, JSON summary. |

## Local versus Hub input flow

The CLI accepts exactly one input mode (mutually exclusive, both required as
a group):

- `--input-root PATH` — an existing local snapshot root directory.
- `--input-dataset-id DATASET_ID` — an upstream Hugging Face dataset.

In **Hub mode**, `cli._resolve_input` calls `acquire_dataset_snapshot`,
which resolves the requested revision to an immutable commit SHA, downloads
the Parquet-only snapshot (excluding `articles/`), validates it, and returns
an `AcquisitionResult` whose `snapshot_path` becomes the pipeline
`input_root` and whose `resolved_sha` becomes the `input_dataset_revision`
forwarded downstream (never a mutable `main`).

In **local mode**, the supplied `--input-dataset-revision` is forwarded
unchanged.

Both modes require `--input-dataset-revision` and `--pipeline-version`.

## Where Wikipedia and Wikivoyage join

`joins/` is a package facade; `osm_polygon_sentence_relevance.joins`
re-exports the public API:

- `joins/_projection.py` — column-projection tuples per input table.
- `joins/_integrity.py` — generic join-key / referential-integrity checks.
- `joins/_wikipedia.py` — `join_wikipedia_sections` (keys on
  `polygon_articles.article_id` → `wp_documents.article_id`,
  `wp_documents.document_id` → `wp_sections.document_id`,
  `polygon_articles.polygon_id` → `polygons.polygon_id`).
- `joins/_wikivoyage.py` — `join_wikivoyage_sections` (keys on
  `wikivoyage_documents.wikidata` → `polygons.wikidata`,
  `wikivoyage_documents.document_id` → `wikivoyage_sections.document_id`;
  empty `article_id` becomes `null`).
- `joins/_composition.py` — `JoinReport`, `JoinedRegionSections`, and
  the orchestration helper that unions both sources and sorts deterministically.

The two source joins are intentionally **not** forced into a single generic
abstraction because their join contracts differ (Wikipedia keys on article
identity; Wikivoyage keys on Wikidata QID).

## Context-before-deduplication ordering

The Phase 2 joined-sections table is sorted by
`(polygon_id, source, language, document_id, section_index, section_id)`
during composition. That sort order is a property of the join output and
is **separate** from the finalization context grouping below.

In `finalization.py`, context is computed *before* exact deduplication, in
three explicit steps:

1. **Group** the segmented-sentences table by context group:
   `(polygon_id, source, document_id, section_id)`. Note that `language`
   is **not** part of the context group, and `section_index` is **not**
   part of the key.
2. **Sort** each context group by
   `(sentence_index, stable complete-row representation)`. The stable row
   representation is the canonical compact JSON of the complete row's
   fields keyed off `SEGMENTED_SENTENCES_SCHEMA`.
3. **Assign** `previous_sentence` and `next_sentence` to each in-group row
   from the previous/next sorted row's `sentence_text_normalized`.

After context assignment, deduplication then re-groups the same rows by
`(polygon_id, language, sentence_text_normalized)` and selects a canonical
occurrence per group (see `data-contract.md`). Context and deduplication
are not the same grouping, and the Phase 2 join sort is not the same sort
as the context sort.

## Finalization and atomic export boundaries

`finalization.py` performs exact deduplication on
`(polygon_id, language, sentence_text_normalized)` and selects a canonical
occurrence with Wikipedia preferred over Wikivoyage (see `data-contract.md`).
It writes `input_dataset_revision` and `pipeline_version` into the Arrow
schema metadata.

`exporter.py` is the stable public facade. The actual work is delegated
to internal helpers under `_export/` (not part of the public API):

- `_export/manifest.py` — per-source/language/region counts, manifest dict,
  deterministic JSON (sorted keys, no trailing-whitespace beyond one `\n`).
- `_export/checksum.py` — streaming SHA-256 of the Parquet file.
- `_export/atomic.py` — rollback-safe directory swap: build a temp dir,
  rename the existing output aside into a `.backup_<uuid>` dir, rename the
  new dir into place, and only then remove the backup. Any swap failure
  restores the backup; if even restoration fails the backup is preserved
  and surfaced via `ExportError`.

The atomic-swap algorithm is unchanged by the decomposition.

## Optional dependency boundaries

- `wtpsplit` (extra `segmentation`) — only imported lazily inside
  `sat_adapter.SaTSentenceSegmenter._get_model`. Base install never
  constructs the model or downloads weights.
- `huggingface_hub` (extra `hub`) — only imported lazily inside
  `acquire_dataset_snapshot`. The CLI never accepts, prints, or persists
  an HF token; standard library authentication is used.

Model construction (the segmenter) happens **after** acquisition succeeds,
so an acquisition failure never triggers a model-weight download.
