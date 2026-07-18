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
  `finalization`, `exporter`, and the contract facades `constants`,
  `errors`, `schemas`, `settings`) re-export the stable public symbols so
  existing import paths keep working.
- `contracts/` package as the canonical home for cross-cutting contracts:
  `constants.py`, `errors.py`, and `schemas/` (`__init__.py`, `input.py`,
  `pipeline.py`, `registry.py`). Each schema object is instantiated once
  and re-exported with preserved object identity.
- `application/settings.py` as canonical `PipelineSettings` ownership.
- Read-only export validator (`output.validation.validate_export_directory`):
  proves a local export directory is internally consistent (Parquet
  present, manifest valid, SHA-256 matches, row count matches, schema
  equal to the canonical output schema, schema metadata for
  `input_dataset_revision` and `pipeline_version` present and
  decodable) before publication. Available via the `validation` module
  and re-exported from `osm_polygon_sentence_relevance.output`.
- Dedicated `publishing/` domain package with
  `publishing.huggingface.publish_export_directory` for programmatic,
  one-commit publishing of a validated export to an existing Hugging
  Face dataset repository. Returns a frozen `PublicationResult` with
  the verified `commit_id`, `commit_url`, `row_count`, and `sha256`.
  Adds `PublicationError` to `contracts.errors` for
  publishing-failure handling. Both `hub_api` and
  `commit_operation_factory` are injectable for fully-offline tests.
- Added `torch` to the `segmentation` optional extra
  (`torch>=2.2,<3`) so `uv sync --extra segmentation` installs the
  PyTorch runtime that `wtpsplit.SaT` requires to construct models.
  SaT model weights remain downloaded separately on first model
  construction; core and `hub`-only installs stay lightweight.
- Optional CLI publishing of the completed export via
  `application/cli.main`: `--publish-dataset-id` (plus optional
  `--publish-revision` and `--publish-commit-message`) publishes the
  successfully built export to an existing Hugging Face dataset
  repository, strictly post-build and in a single commit. Validation of
  all publishing relationships happens before acquisition or model
  construction; no token or repository-creation flag exists.
- `scripts/verify_distribution.py`: stdlib-only distribution-content
  verifier used by CI and documented in `docs/guides/development.md`.
- MIT license (`LICENSE`), `CONTRIBUTING.md`, and `SECURITY.md`.
- `docs/` restructured into `index`, `architecture/`, `guides/`, and
  `reference/` sections; ADR 0001 (domain layout) and ADR 0002
  (contracts package).
- `.github/workflows/ci.yml`: lint, type, test (with branch coverage),
  build, distribution-content, and installed-wheel smoke gates.
- Documentation consistency tests (README/dataset IDs, CLI flags,
  version parity, link validity, local-path absence) and structural
  tests (no production imports of facades; facade purity via AST).

### Changed
- Production modules now import each other via canonical domain paths
  (`contracts.constants`, `contracts.errors`, `contracts.schemas`,
  `application.settings`, domain packages); never via a root facade.
- Settings data-directory resolution is now portable and machine-agnostic:
  explicit `data_dir` argument, then nonblank `OSM_DATA_DIR`, then
  `Path.cwd() / "data"`. Whitespace-only `OSM_DATA_DIR` is ignored; a
  leading `~` is expanded. No directory creation, no network access, and
  no probing of personal or platform-specific mount points.
- `MANIFEST.in` extended to ship public docs and project governance
  files in the source distribution while excluding caches, data, model
  weights, and the local-only `.local-docs/` guide.

### Removed
- The hard-coded external-drive data-directory path and all
  machine-specific filesystem probing from settings resolution. (Earlier
  changelog text incorrectly claimed this removal happened in the prior
  reorganization; it is finalized in this pass.)
- Obsolete root `config.py` (legacy `get_data_dir` logic) and `main.py`
  (Phase 1 stub). The supported entry point is the installed console
  command `osm-polygon-sentence-relevance`.

### Not implemented (scope boundaries)
- Hugging Face dataset repository creation: the publisher targets an
  existing repository and never calls `create_repo` or initializes a
  new dataset.
- Sentence classification or labelling.
- Concurrency, resumable builds, or incremental builds.

## [0.1.0]
- Initial pre-release: deterministic OSM-polygon → Wikipedia/Wikivoyage
  sentence-relevance dataset construction with read-only Hugging Face
  snapshot acquisition, joins, segmentation, finalization,
  deduplication, deterministic IDs, and rollback-safe atomic export.
