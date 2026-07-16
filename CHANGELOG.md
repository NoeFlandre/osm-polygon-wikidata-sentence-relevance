# Changelog

All notable changes to this project are documented here. This project
adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
semantic versioning once a stable 1.0 release is cut. Until then the
package remains pre-1.0 (currently `0.1.0`).

## [Unreleased]

### Added
- Domain-package reorganization of the implementation under `src/`:
  `application/` (CLI + pipeline), `ingestion/` (acquisition, discovery,
  loading), `sentences/` (preprocessing, segmentation, SaT adapter,
  table, finalization), `output/` (exporter, atomic install, checksum,
  manifest), and `joins/` (unchanged from Q2).
- Thin compatibility facades at the previous top-level module paths
  (`cli`, `pipeline`, `acquisition`, `discovery`, `loading`,
  `preprocessing`, `segmentation`, `sat_adapter`, `sentence_table`,
  `finalization`, `exporter`) re-export the stable public symbols so
  existing import paths keep working.
- `py.typed` marker for downstream type-checking support; included in
  the wheel.
- `scripts/verify_distribution.py`: stdlib-only distribution-content
  verifier used by CI and documented in `docs/guides/development.md`.
- MIT license (`LICENSE`), `CONTRIBUTING.md`, and `SECURITY.md`.
- `docs/` restructured into `index`, `architecture/`, `guides/`, and
  `reference/` sections.
- `.github/workflows/ci.yml`: lint, type, test (with branch coverage),
  build, distribution-content, and installed-wheel smoke gates.
- Documentation consistency tests (README/dataset IDs, CLI flags,
  version parity, link validity, local-path absence).

### Changed
- Production modules now import each other via canonical domain paths
  rather than the legacy flat module names.
- `MANIFEST.in` extended to ship public docs and project governance
  files in the source distribution while excluding caches, data, model
  weights, and the local-only `.local-docs/` guide.

### Removed
- Obsolete root `config.py` (legacy `get_data_dir`/Seagate path logic)
  and `main.py` (Phase 1 stub). The supported entry point is now the
  installed console command `osm-polygon-sentence-relevance`.

### Not implemented (scope boundaries)
- Hugging Face dataset publishing / upload.
- Sentence classification or labelling.
- Concurrency, resumable builds, or incremental builds.

## [0.1.0]
- Initial pre-release: deterministic OSM-polygon → Wikipedia/Wikivoyage
  sentence-relevance dataset construction with read-only Hugging Face
  snapshot acquisition, joins, segmentation, finalization,
  deduplication, deterministic IDs, and rollback-safe atomic export.
