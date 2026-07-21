# Changelog

All notable changes are documented here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and remains a
pre-1.0 package.

## [Unreleased]

### Added

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

- Checkpointing, dataset-card statistics, rendering, profiling, and plotting
  now have focused internal owners behind stable public facades.
- Production operations use only the bounded streaming build and deterministic
  finalization entry points.
- Static typing covers package code and production operational Python.

### Fixed

- Resumable builds validate source-file identities, schemas, hashes, modes, and
  run metadata before reuse.
- Hugging Face publication uses a Viewer-compatible `osm_tags` representation
  and verifies generated assets against the manifest.
- CUDA placement validates the complete pinned `wtpsplit` classifier and never
  silently falls back after an explicit accelerator request.

### Removed

- Superseded diagnostic, audit, full-snapshot, and hardware-probe workflows.
- Historical operational incident notes and obsolete launcher documentation.
- Repository creation, sentence classification, and parallel shard processing
  remain outside the product scope; CLI publishing to an existing repository
  is supported.

## [0.1.0]

- Initial pre-release: deterministic OSM-polygon to Wikipedia/Wikivoyage
  sentence dataset construction with acquisition, joins, segmentation,
  finalization, deduplication, deterministic IDs, and atomic export.
