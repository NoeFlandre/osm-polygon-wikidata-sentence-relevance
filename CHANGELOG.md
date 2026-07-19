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

### Added (Phase 9A — explicit accelerator selection)
- `SaTSentenceSegmenter` accepts a `device` argument (`"auto"`,
  `"cpu"`, `"cuda"`, `"mps"`) and an injectable `caps` capability
  snapshot. `"auto"` prefers CUDA, then MPS, then CPU; explicit
  unavailable accelerators fail with `SegmentationError` rather than
  silently downgrading.
- The build CLI exposes `--device {auto,cpu,cuda,mps}` (default
  `auto`) and performs an early hardware-availability check before
  acquisition or model construction. No Torch import occurs for
  `--help`.
- `--input-source-dataset-id OWNER/DATASET` records the upstream
  source dataset ID for a previously-acquired local snapshot. Only
  valid with `--input-root`; populates the manifest, statistics, and
  generated `README.md` dataset card without triggering any network
  access.
- `sentences.device` exposes `PUBLIC_DEVICE_VALUES`,
  `TorchCapabilities` Protocol, `resolve_device`, and `default_caps`
  for programmatic use. The resolver is pure logic over an injected
  capability snapshot; production callers use a lazy Torch-backed
  default.

### Changed (Phase 9A)
- The wtpsplit placement helper is now narrowly versioned to wtpsplit
  2.2.1 and selects the *complete classifier* owned by the
  `wtpsplit.extract.PyTorchWrapper` (`SubwordXLMForTokenClassification`,
  not its XLM-R backbone). The complete classifier is moved via a
  single `.to(device)` call and verified by reading every parameter
  and buffer device. Knowledge of the wtpsplit wrapper shape is
  isolated to `sentences/_wtpsplit_device.py` (five private helpers)
  with no wtpsplit-specific code remaining in `sentences/sat.py`.
- The segmenter resolves its device **exactly once** per instance,
  immediately before the first model construction. Subsequent batches
  reuse both the resolved device and the placed model.
- Device resolution and model-shape handling are now strictly
  separated. The resolver never inspects the model and never rewrites
  the resolved device based on the loaded model's shape; the
  placement adapter is the only layer that may inspect the model,
  and only to decide whether the model supports the resolved
  device. The one exception is the legacy CPU-only test-double path:
  a `device="cpu"` request against an unrecognised wrapper is a
  no-op; any resolved accelerator (`cuda` / `mps`) against an
  unrecognised wrapper raises `SegmentationError` rather than
  silently running on CPU. This is the guard that prevents a
  Grid'5000 run that selected CUDA from silently computing on CPU
  after placement "fails" on the accelerator.
- The `wtpsplit` entry in the `segmentation` optional extra is now
  pinned to exactly `wtpsplit==2.2.1` (not the previous
  `wtpsplit>=2.2.1,<3`). The lockfile records the pin. The
  placement adapter is intentionally version-specific, so a wider
  range would invite a configuration the adapter has not been
  tested against. The `TestDeclaredVersionAgreement` metadata test
  enforces the contract at the test-suite level.
- `_ResolvedInput` no longer carries a redundant `local_source_dataset_id`
  field; `dataset_id` is the single provenance value used by both Hub
  acquisition and local mode with `--input-source-dataset-id`.

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

### Added (Phase 9B -- Grid'5000 GPU smoke tooling, safety amendment)
- `scripts/grid5000/gpu_preflight.py`: read-only preflight snapshot
  proving Linux, OAR allocation, `torch.cuda.is_available()`, exactly
  one visible NVIDIA device; emits a stable JSON record (OAR job ID,
  hostname, Torch/CUDA versions, device identity) to stdout.
- `scripts/grid5000/run_gpu_smoke.sh`: the OAR job payload. Requires
  `OAR_JOB_ID` and `CUDA_VISIBLE_DEVICES`, the locked interpreter at
  `${REPO_ROOT}/.venv/bin/python`, and caller-provided persistent
  `HF_HOME` and `SMOKE_LOG_DIR`. The validator is invoked by
  absolute script path (`${ARTIFACT_VALIDATOR}`) so the payload
  works from any working directory. Per-phase cleanup traps remove
  every temporary file on success, failure, or interruption.
  Restricted `umask 077` for log permissions.
- `scripts/grid5000/_validate_artifact.py`: private reusable
  validators (`validate_preflight`, `validate_smoke_result`) and
  an atomic no-clobber install helper (`install_artifact` using
  `os.link`). Both validators enforce an exact-schema key set
  (missing or extra keys rejected with a stable, path-free message);
  finite-number and bool-rejection contract; cleanup-failure path
  raises `ArtifactValidationError` without echoing paths.
- Path-redaction, collision-safe atomic writes, OAR plan correction,
  commit-based remote reproducibility, cache staging plan: see
  `docs/guides/grid5000.md` and `tests/unit/scripts/`.

### Added (Phase 9D — non-interactive Grid'5000 GPU smoke jobs)
- `scripts/grid5000/run_gpu_smoke_job.sh`: non-interactive OAR batch
  entrypoint that `oarsub` invokes inside the allocation. The scheduler,
  not the local SSH transport, owns job lifetime, which removes the
  interactive-`-I` frag failure mode. The script requires the scheduler-set
  `OAR_JOB_ID` (validated as exactly decimal digits) and
  `CUDA_VISIBLE_DEVICES`, and the four positional arguments
  `REPO_ROOT HF_HOME LOG_ROOT EXPECTED_SOURCE_COMMIT`. It strictly
  canonicalises each path (absolute, no traversal, no symlink in the
  resolved chain), refuses ephemeral node-local storage, and invokes the
  committed `run_gpu_smoke.sh` payload exactly once, capturing stdout,
  stderr, and the real exit code without masking, retry, or CPU/MPS/auto
  fallback. On a zero smoke exit it enforces an exact six-artifact success
  contract: exactly six direct entries in a mode-0700 directory, each an
  expected regular file (mode 0600); any unexpected or missing entry fails
  the contract while preserving all artefacts for forensic inspection.
- Public docs (`docs/guides/grid5000.md`) restructured so the canonical
  submission path is the non-interactive batch form and the interactive
  `-I` form is retained only as a human-held debugging option, never as the
  smoke command. The canonical `oarsub` line uses the caller's exported
  `${REPO_ROOT}`, `${HF_HOME}`, `${LOG_ROOT}`, and `${SOURCE_COMMIT}`
  variables with quoted positional arguments and no personal absolute paths.
- `tests/unit/scripts/test_run_gpu_smoke_job_sh.py`: RED→GREEN contract
  tests for the batch wrapper (path canonicalisation, `OAR_JOB_ID`
  validation, exact four-positional-argument check, six-artifact success
  contract, failure-code preservation, and a forbidden-pattern audit).
- `MANIFEST.in` now ships the public Grid'5000 shell scripts (`scripts/*.sh`)
  in the sdist; the wheel remains package-code only.

### Added (Phase 9F — OAR submission adapter)
- `scripts/grid5000/submit_gpu_smoke.sh`: frontend-only OAR submission
  helper. Nancy's `oarsub` accepts exactly one positional command
  string and rejects `oarsub <opts> script arg1 arg2 arg3 arg4`, so this
  helper validates the four positional arguments (`REPO_ROOT HF_HOME
  LOG_ROOT EXPECTED_SOURCE_COMMIT`) and serialises them into a single
  quoted command that runs the compute-node wrapper inside the
  allocation. Serialisation uses a portable POSIX single-quote escaper
  (`'\''` idiom, no `eval`, no Bash `%q`) so spaces, semicolons, `$()`,
  backticks, wildcards, and embedded single quotes stay literal and
  cannot execute. The helper requires `command -v oarsub`, the
  compute-node wrapper present and executable, canonical absolute
  persistent directories (rejecting ephemeral storage, traversal, and
  symlinks), and a 40-lowercase-hex commit; it never imports Python,
  performs inference, polls, cancels, SSHes, mutates git, downloads,
  retries, or cleans up, and forwards `oarsub` stdout/stderr and its real
  exit code unchanged without parsing a job ID.
- `docs/guides/grid5000.md` canonical command now invokes
  `submit_gpu_smoke.sh` with the four quoted arguments; the invalid
  direct `oarsub ... run_gpu_smoke_job.sh arg1...` line is removed and the
  interactive `-I` form is retained only for human debugging.
- `tests/unit/scripts/test_submit_gpu_smoke_sh.py`: RED→GREEN contract
  tests with a fake `oarsub` proving single invocation, exact
  queue/resource options, exactly one positional command string, faithful
  four-argument decoding, `exec`-led payload, no `-I`, injection safety,
  pre-submission validation failures, real non-zero exit forwarding, and
  no retry.
- All three public Grid'5000 shell scripts (`run_gpu_smoke.sh`,
  `run_gpu_smoke_job.sh`, `submit_gpu_smoke.sh`) are committed with mode
  `100755`; the distribution verifier now also requires
  `scripts/grid5000/submit_gpu_smoke.sh` in the sdist and still forbids
  `scripts/` paths in the wheel.

## [0.1.0]
- Initial pre-release: deterministic OSM-polygon → Wikipedia/Wikivoyage
  sentence-relevance dataset construction with read-only Hugging Face
  snapshot acquisition, joins, segmentation, finalization,
  deduplication, deterministic IDs, and rollback-safe atomic export.
