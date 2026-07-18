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
- Deterministic Hugging Face dataset card (`output/dataset_card.py`):
  computes immutable statistics from the finalized output table
  (totals, unique sentence IDs, polygons, Wikidata entities, documents,
  Wikipedia/Wikivoyage source counts, language/region counts, and
  coordinate presence), renders a valid YAML-front-matter `README.md`
  at export time, and serializes the same figures into a versioned
  `statistics` object in `manifest.json`. The card is regenerated on
  every build and is never hand-edited.
- Strengthened export validation (`output.validation`): the validator
  now also requires the auto-generated `README.md` card, rejects a
  manifest without the `statistics` object, recomputes statistics
  directly from `sentences.parquet`, and rejects any card or manifest
  whose figures do not equal the deterministic render/values.
- Publishing (`publishing/huggingface.publish_export_directory`) now
  uploads all three verified artifacts (`sentences.parquet`,
  `manifest.json`, and the auto-generated `README.md`) in a single Hub
  commit derived from the validated export contract.
- Dataset-card accuracy and robustness amendment
  (`output/dataset_card.py`, `statistics` `STATISTICS_VERSION = 2`):
  unique documents are now counted by the
  `(source, site, language, document_id)` tuple (the input contract does
  not globally guarantee bare `document_id` uniqueness); the card's
  prose no longer claims land-use relevance / weighting / scoring /
  classification or unsupported Hugging Face task categories, no longer
  claims that normalization changes case or that raw text is verbatim,
  and explicitly states that land-use relevance and
  polygon-description labels are future downstream work and are absent;
  `statistics_from_dict` rejects coerced values, unknown keys, malformed
  SHA-256, blank revision/version, breaking accounting identities, and
  over-count uniques; the manifest's top-level count fields, row count,
  checksum, and revision/version are now derived from the single
  `DatasetStatistics` instance and the validator rejects drift between
  them; the card renders YAML-safe quoted language values, an empty
  language list when empty, and HTML-escaped Markdown table cells so
  pipes/backslashes/newlines cannot break the card.
- Source-provenance completion (`finalize_sentence_dataset`,
  `compute_statistics`, `output.manifest.build_manifest_data`,
  `output.exporter`, `output.validation`, `application.pipeline`,
  `application.cli`): an optional `input_dataset_id` is threaded from
  the CLI through the finalizer, Parquet schema metadata
  (`b"input_dataset_id"`), manifest top-level field, manifest
  `statistics.input_dataset_id`, the `DatasetStatistics` dataclass, and
  the auto-generated `README.md` dataset card; the validator
  cross-checks the three surfaces and rejects drift; blank or
  non-string IDs are rejected before any output mutation; existing
  callers omitting the parameter continue to work (local mode); the
  card links to the Hub dataset page (always) and to
  `/datasets/<id>/tree/<sha>` when the recorded revision is a 40-char
  lowercase hex SHA-1; local mode renders an explicit "no recorded
  Hugging Face dataset ID for the upstream source" sentence without
  implying a Hub commit. `STATISTICS_VERSION` remains 1.
- Source-provenance safety amendment
  (`output/dataset_card.py`, validator strict `_decode_meta_value`,
  finalizer): the custom `_quote_url_component` is replaced by
  `urllib.parse.quote` with explicit `safe` sets: dataset IDs keep
  the `owner/repo` separator (`safe="/"`) and every other character
  is percent-encoded as UTF-8; revisions use `safe=""` so every path
  separator, query, fragment, percent, backslash, bracket, paren, and
  Unicode code point is encoded. A dedicated `_escape_md_link_label`
  deterministically escapes `&`, `[`, `]`, backticks, backslashes,
  CR, LF, and tab so an adversarial dataset identifier containing
  ``](https://attacker.example)`` cannot break out of the link. The
  card no longer infers which acquisition mechanism produced a
  recorded value: 40-character lowercase hex revisions are described
  as a "recorded immutable commit identifier" without naming the
  acquisition step. `STATISTICS_VERSION` stays at 1 and
  `_OPTIONAL_STATISTICS_KEYS` is removed: `input_dataset_id` is a
  required v1 key whose value is either `None` or a non-blank
  string. Local exports serialize `null`; missing keys are
  rejected. Present-but-blank Parquet metadata values are now
  rejected by the finalizer, the validator, and `compute_statistics`
  with `FinalizationError` / `ExportError` / `ValueError`
  respectively; no silent normalization to `None` anywhere in the
  export chain.

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
