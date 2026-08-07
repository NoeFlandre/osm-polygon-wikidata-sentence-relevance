# Changelog

All notable changes are documented here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The Python package
remains pre-1.0; the first public dataset release is `v1.0.0` and is limited
to Afghanistan.

## [1.0.0] - 2026-08-03

- First public dataset release: the complete Afghanistan artifact with 54,462
  labeled sentences, 161 polygons, and 115 languages, published to
  `NoeFlandre/osm-polygon-wikidata-sentence-relevance` on Hugging Face.
- Bound the release to the immutable input, model, prompt, and source revisions
  recorded in its manifest.

## [Unreleased]

- Added a separate worldwide V2 label-sampling contract with a configurable
  target, deterministic H3/language/OSM-primary-tag strata, and nested
  proportional continuation for larger targets. V2 files are namespaced below
  `v2-worldwide/` on the dataset's existing `main` revision; the published
  Afghanistan V1 artifact is unchanged. V2 now uses one binary
  `place_relevance` decision about visual or geographic place description and
  stores first-token yes/no log-probability scores rather than generated JSON.
- Added a durable asynchronous checkpoint mirror for Grid'5000 labeling. Each
  completed local batch can be staged to a run-specific
  `.pipeline/checkpoints/<run-id>/` path on Hugging Face `main`; failed uploads
  remain queued for the next resume and never replace a final release lane.

### Added

- A Mac-side `osm-polygon-grid5000` operator with deterministic run identity,
  durable state, bounded OpenSSH transport, automatic site selection,
  resumable OAR allocations, live terminal logs, exact region selection,
  adaptive llama.cpp context, and validated Hugging Face publication.

- Resumable Afghanistan LLM labeling with independent land-use/land-cover and
  polygon-relevance decisions, strict structured output, vLLM-first CUDA
  serving with a llama.cpp fallback, factual timing/ETA, and atomic labels.
- Automatic labeled-dataset finalization, data-derived card statistics and
  plots, closed-layout validation, and single-commit Hugging Face publication.
- Deterministic final-label analytics and an optional Trackio `track` command
  that logs one static run with KPI cards, tables, plots, and slice yields.
- A guarded, non-publishing Afghanistan labeling canary with deterministic
  source/language coverage, real structured-inference probing, and
  non-interactive high-memory Grid'5000 submission.

- Bounded, resumable per-shard processing for Grid'5000 CUDA allocations,
  backed by identity-bound remote checkpoints.
- Deterministic dataset profiles, generated dataset cards, geographic and
  language-distribution assets, and strict publication validation.
- Programmatic publishing of validated exports to an existing Hugging Face
  dataset in one commit through the `publishing/` domain package, with
  `validate_export_directory` run before upload and `PublicationError` for
  failures.
- Explicit `cpu`, `cuda`, and `mps` device selection with fail-closed placement
  checks for the pinned SaT adapter.

### Changed

- Migrated the Grid'5000 operator command to a typed Typer boundary with
  terminal-safe Rich/tqdm progress while retaining plain machine output, and
  unified local and CI quality commands through uv, just, and pre-commit.
- Production sentence segmentation now defaults to the multilingual
  `sat-12l-sm` model. A conservative post-model repair separates only
  high-confidence residual punctuation boundaries across scripts before
  sentence indexing, while preserving abbreviations, lowercase continuations,
  numeric values, and URL query strings.
- Checkpointing, dataset-card statistics, rendering, profiling, and plotting
  now have focused internal owners behind stable public facades.
- Production operations use only the bounded streaming build and deterministic
  finalization entry points.
- Static typing covers package code and production operational Python.

### Fixed

- Wikipedia joins now accept semicolon-delimited upstream Wikidata aliases
  when the linked polygon or document QID is present, while still rejecting
  unrelated identities.
- Publication now rescans every normalized sentence and refuses an export with
  any high-confidence residual sentence boundary; the factual count is recorded
  in the manifest and generated dataset card.
- Resumable builds validate source-file identities, schemas, hashes, modes, and
  run metadata before reuse.
- Hugging Face publication uses a Viewer-compatible `osm_tags` representation
  and verifies generated assets against the manifest.
- CUDA placement validates the complete pinned `wtpsplit` classifier and never
  silently falls back after an explicit accelerator request.

### Removed

- Superseded diagnostic, audit, full-snapshot, and hardware-probe workflows.
- Historical operational incident notes and obsolete launcher documentation.
- Repository creation, general multi-region classification, and parallel shard processing
  remain outside the product scope; CLI publishing to an existing repository
  is supported.

## [0.1.0]

- Initial pre-release: deterministic OSM-polygon to Wikipedia/Wikivoyage
  sentence dataset construction with acquisition, joins, segmentation,
  finalization, deduplication, deterministic IDs, and atomic export.
