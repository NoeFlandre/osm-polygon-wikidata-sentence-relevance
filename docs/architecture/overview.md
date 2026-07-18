# Architecture

This document describes the module ownership and data flow of the OSM Polygon
– Wikidata Sentence Relevance pipeline. It is authoritative for *current*
behavior. Publishing to the Hugging Face Hub exists both programmatically
(see `osm_polygon_sentence_relevance.publishing`) and via the build CLI's
optional `--publish-dataset-id` flag (post-build, to an existing
repository). Hugging Face repository creation remains unimplemented.

The package is organized into domain packages under
`src/osm_polygon_sentence_relevance/`. Cross-cutting contracts
(`constants`, `errors`, `schemas`, `settings`) now live in the canonical
`contracts/` package; the previous top-level `constants.py`, `errors.py`,
`schemas.py`, and `settings.py` are thin compatibility facades.

## Pipeline stages and module ownership

| Stage | Module | Responsibility |
|-------|--------|----------------|
| Schema contracts | `contracts/schemas/` | Immutable Arrow schemas for all six input tables and the output sentence table. |
| Cross-cutting constants | `contracts/constants.py` | Dataset IDs, pipeline version, allowed sources/paths. |
| Cross-cutting errors | `contracts/errors.py` | Exception hierarchy. |
| Settings | `application/settings.py` | Frozen `PipelineSettings` with portable data-dir precedence. |
| Shard discovery | `ingestion/discovery.py` | Locate per-region Parquet shards under an input root. |
| Loading | `ingestion/loading.py` | Project only required columns and validate against schemas. |
| Acquisition | `ingestion/acquisition.py` | Read-only Hugging Face snapshot acquisition. |
| Preprocessing | `sentences/preprocessing.py` | Normalize section paths, OSM tags, and sentence text (deterministic). |
| Segmentation | `sentences/segmentation.py`, `sentences/table.py` | Injectable `SentenceSegmenter`; build the segmented sentence table. |
| SaT adapter | `sentences/sat.py` | Optional `wtpsplit` SaT segmenter (lazy import). |
| Joins | `joins/` package | Build Wikipedia + Wikivoyage section→polygon occurrences (see below). |
| Finalization | `sentences/finalization.py` | Exact deduplication, deterministic IDs, metadata, validation. |
| Export | `output/exporter.py` facade + `output/` | Atomic, checksummed Parquet + manifest install. |
| Validation | `output/validation.py` | Read-only integrity check of an exported directory. |
| Publishing | `publishing/huggingface.py` | Programmatic one-commit publish of a validated export to an existing Hub dataset. |
| Orchestration | `application/pipeline.py` | Tie the above stages together (injected segmenter). |
| CLI | `application/cli.py` | Console entry point, argument resolution, JSON summary. Optional post-build publishing via `--publish-dataset-id`. |

## Compatibility-facade policy

The implementation lives in the domain packages above (and `contracts/`).
Thin compatibility facades remain at the previous top-level module paths
(`cli`, `pipeline`, `acquisition`, `discovery`, `loading`,
`preprocessing`, `segmentation`, `sat_adapter`, `sentence_table`,
`finalization`, `exporter`, plus `constants`, `errors`, `schemas`,
`settings`). Each facade has only a module docstring, explicit re-exports,
an accurate `__all__`, and no logic, no warnings, and no import-time side
effects. Production code imports via canonical domain paths; never via a
facade. Tests and external consumers may continue to import from the legacy
paths.

## Local versus Hub input flow

The CLI accepts exactly one input mode (mutually exclusive, both required as
a group):

- `--input-root PATH` — an existing local snapshot root directory.
- `--input-dataset-id DATASET_ID` — an upstream Hugging Face dataset.

In **Hub mode**, `application/cli._resolve_input` calls
`ingestion/acquisition.acquire_dataset_snapshot`, which resolves the
requested revision to an immutable commit SHA, downloads the Parquet-only
snapshot (excluding `articles/`), validates it, and returns an
`AcquisitionResult` whose `snapshot_path` becomes the pipeline
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

In `sentences/finalization.py`, context is computed *before* exact
deduplication, in three explicit steps:

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

`sentences/finalization.py` performs exact deduplication on
`(polygon_id, language, sentence_text_normalized)` and selects a canonical
occurrence with Wikipedia preferred over Wikivoyage (see
`reference/data-contract.md`). It writes `input_dataset_revision` and
`pipeline_version` into the Arrow schema metadata.

`output/exporter.py` is the stable public facade. The actual work is
delegated to internal helpers under `output/` (not part of the public
API):

- `output/manifest.py` — per-source/language/region counts, manifest dict,
  deterministic JSON (sorted keys, no trailing-whitespace beyond one `\n`).
- `output/checksum.py` — streaming SHA-256 of the Parquet file.
- `output/atomic.py` — rollback-safe directory swap: build a temp dir,
  rename the existing output aside into a `.backup_<uuid>` dir, rename the
  new dir into place, and only then remove the backup. Any swap failure
  restores the backup; if even restoration fails the backup is preserved
  and surfaced via `ExportError`.

The atomic-swap algorithm is unchanged by the reorganization.

## Optional dependency boundaries

- `wtpsplit` (extra `segmentation`) — only imported lazily inside
  `sentences/sat.SaTSentenceSegmenter._get_model`. Base install never
  constructs the model or downloads weights.
- `torch` (extra `segmentation`) — installed alongside `wtpsplit` to
  supply the PyTorch runtime that `wtpsplit.SaT` requires to
  construct models on the supported Python 3.12 interpreter. SaT
  model weights remain downloaded lazily on first model construction.
- `huggingface_hub` (extra `hub`) — only imported lazily inside
  `ingestion/acquisition.acquire_dataset_snapshot` (input acquisition)
  and inside the publishing default-construction helpers in
  `publishing/huggingface.py` (output publishing). Both modules honor
  fully-injected Hub API / factory objects, so tests need not install
  the extra. The CLI never accepts, prints, or persists an HF token;
  standard library authentication is used.

Model construction (the segmenter) happens **after** acquisition succeeds,
so an acquisition failure never triggers a model-weight download.

## Publishing (programmatic and CLI)

`publishing/huggingface.publish_export_directory` validates a local
export first via `output/validation.validate_export_directory`, then
publishes exactly the two verified files (`sentences.parquet` and
`manifest.json`) to an existing Hub dataset repository in a single
`create_commit` call. It is also reachable from the build CLI:
`application/cli.main` runs it after a successful build when
`--publish-dataset-id` is supplied. The function models two separate
dependencies:

- `hub_api` — owns the network; exposes `create_commit(...)`. Called
  exactly once per publication.
- `commit_operation_factory(*, path_in_repo, path_or_fileobj)` —
  constructs one add operation per local file. The two returned
  objects are passed unchanged to `hub_api.create_commit`.

If either dependency is absent, only the missing one is imported
lazily from `huggingface_hub`. Fully-injected calls never import the
library and perform zero network activity.

The function:

- rejects invalid public arguments (blank / non-string dataset ID,
  revision, or commit message) before any import, validation, or Hub
  activity;
- wraps library/remote failures in `PublicationError` with the
  original exception preserved as `__cause__`;
- does not create repositories, does not accept a token, does not
  retry. When invoked from the CLI via `--publish-dataset-id`, it runs
  strictly post-build and the target repository must already exist.
