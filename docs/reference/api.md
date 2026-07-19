# API reference

This page documents the supported **canonical** imports (implementation in
domain packages) and lists the **compatibility** imports (legacy top-level
facades). Private underscore modules are intentionally not documented.

All canonical modules live under
`src/osm_polygon_sentence_relevance/`.

## Canonical imports

### `osm_polygon_sentence_relevance.contracts` (cross-cutting contracts)

- `contracts.constants` — `INPUT_DATASET_ID`, `OUTPUT_DATASET_ID`,
  `DEFAULT_INPUT_REVISION`, `PIPELINE_VERSION`, `ALLOWED_SOURCES`,
  `SCHEMA_NAMES`, `ALLOWED_INPUT_PATHS`.
- `contracts.errors` — `ConfigurationError`, `SchemaContractError`,
  `UnknownTableError`, `MissingColumnsError`, `IncompatibleTypesError`,
  `PreprocessingError`, `SegmentationError`, `ShardDiscoveryError`,
  `JoinIntegrityError`, `FinalizationError`, `ExportError`,
  `AcquisitionError`, `PublicationError`.
- `contracts.schemas` — all PyArrow schemas (`POLYGONS_SCHEMA`,
  `POLYGON_ARTICLES_SCHEMA`, `WIKIPEDIA_DOCUMENTS_SCHEMA`,
  `WIKIVOYAGE_DOCUMENTS_SCHEMA`, `SECTIONS_SCHEMA`, `OUTPUT_SENTENCE_SCHEMA`,
  `JOINED_SECTIONS_SCHEMA`, `SEGMENTED_SENTENCES_SCHEMA`), `SCHEMA_REGISTRY`,
  and `validate_table_schema`.

### `osm_polygon_sentence_relevance.application`

- `main(args=None, *, model_factory=None, acquisition_fn=None) -> int`
- `PipelineResult` (dataclass)
- `run_pipeline(input_root, output_dir, segmenter, *, input_dataset_revision,
  pipeline_version, batch_size=128, overwrite=False,
  input_dataset_id=None) -> PipelineResult`. `input_dataset_id` is
  the optional upstream dataset identifier (e.g. `OWNER/REPO`); in Hub
  mode it is threaded into the export chain (Parquet schema metadata,
  manifest, statistics, generated `README.md` dataset card). When
  omitted (local mode) no Hub identity is propagated.

### `osm_polygon_sentence_relevance.ingestion`

- `acquisition.AcquisitionResult`
- `acquisition.acquire_dataset_snapshot(dataset_id, requested_revision, *,
  hub_api=None, download_fn=None) -> AcquisitionResult`
  (optional `huggingface_hub` extra; lazy import)
- `acquisition.ALLOW_PATTERNS`, `acquisition.IGNORE_PATTERNS`
  (snapshot allow/ignore glob patterns — these live on
  `ingestion.acquisition`, **not** on `constants`)
- `discovery.RegionShardSet`
- `discovery.discover_shards(root) -> list[RegionShardSet]`
- `loading.load_validated_table(name, path, *, columns=...) -> pa.Table`

### `osm_polygon_sentence_relevance.sentences`

- `preprocessing.normalize_sentence(text) -> str`
- `preprocessing.parse_section_path(text) -> list[str]`
- `preprocessing.parse_osm_tags(text) -> list[tuple[str, str]]`
- `segmentation.SentenceSegmenter` (protocol)
- `segmentation.SegmentationReport`
- `segmentation.split_validated_batch(...)`
- `device.PUBLIC_DEVICE_VALUES` — `frozenset({"auto", "cpu", "cuda", "mps"})`.
- `device.TorchCapabilities` — `Protocol` exposing `cuda_available: bool`
  and `mps_available: bool`. Production callers use
  `device.default_caps()` which lazily probes Torch; tests inject a
  fake.
- `device.resolve_device(value, *, caps) -> str` — pure function.
  `value` must be one of the four public device strings. `"auto"`
  resolves to `cuda` when available, otherwise `mps`, otherwise `cpu`.
  Explicit `cuda` / `mps` raise `SegmentationError` when the backend
  is unavailable; the resolver never silently downgrades.
- `sat.SaTSentenceSegmenter` (optional `segmentation` extra:
  installs the `wtpsplit` adapter **pinned to exactly `wtpsplit==2.2.1`**
  plus its required PyTorch runtime; the SaT model itself is
  constructed lazily and its weights are downloaded separately on
  first model construction).
  Constructor signature: `SaTSentenceSegmenter(model_name="sat-3l-sm",
  *, model_factory=None, model_kwargs=None, split_kwargs=None,
  device="auto", caps=None)`. The device is resolved **exactly once**
  per segmenter, immediately before model construction. The complete
  classifier (the `wtpsplit.extract.PyTorchWrapper`-owned
  `SubwordXLMForTokenClassification`, **not** its backbone) is moved
  to the resolved device and verified by reading every parameter
  and buffer. Hardware selection never alters output schema, IDs,
  hashes, or dataset-card statistics; only the runtime
  accelerator differs. Device resolution and model-shape handling
  are kept separate: the segmenter never inspects the model to
  decide which device to use, and never silently rewrites a
  resolved accelerator (`cuda` / `mps`) to `cpu` after the fact —
  a real wtpsplit model that cannot honour a resolved accelerator
  fails with `SegmentationError` rather than running on CPU.
- `sat.WTPSPLIT_SUPPORTED_VERSION` — the wtpsplit version the
  placement adapter is structurally tested against (`"2.2.1"`).
  This must agree with the segmentation-extras pin in
  `pyproject.toml`; the metadata test
  `TestDeclaredVersionAgreement` enforces the contract.
- `table.SegmentedTableResult`
- `table.segment_joined_sections(...)`
- `finalization.FinalizationReport`
- `finalization.FinalizedDataset`
- `finalization.sentence_content_hash(text) -> str`
- `finalization.deterministic_sentence_id(...) -> str`
- `finalization.finalize_sentence_dataset(table, *, input_dataset_revision,
  pipeline_version, input_dataset_id=None) -> FinalizedDataset`. When
  `input_dataset_id` is a non-blank string the value is recorded as
  Parquet schema metadata under the key `b"input_dataset_id"` and is
  later cross-checked by the validator against the manifest. In CLI
  local mode (`--input-root`) the value is sourced from
  `--input-source-dataset-id`; in Hub mode it is the upstream
  `--input-dataset-id`.

### `osm_polygon_sentence_relevance.joins`

- `build_region_section_occurrences(shards) -> JoinedRegionSections`
- `join_wikipedia_sections(...)`
- `join_wikivoyage_sections(...)`
- `JoinReport`, `JoinedRegionSections`
- projection-column tuples (`POLYGONS_COLS`, `POLYGON_ARTICLES_COLS`,
  `WIKIPEDIA_DOCUMENTS_COLS`, `WIKIPEDIA_SECTIONS_COLS`,
  `WIKIVOYAGE_DOCUMENTS_COLS`, `WIKIVOYAGE_SECTIONS_COLS`)

### `osm_polygon_sentence_relevance.output`

- `ExportResult`
- `export_finalized_dataset(dataset, output_dir, *, overwrite=False)
  -> ExportResult`
- `ValidatedExport` (frozen, slotted dataclass)
- `validate_export_directory(path) -> ValidatedExport`
  (read-only: verifies Parquet + manifest presence, JSON validity,
  SHA-256 checksum, row count, exact `OUTPUT_SENTENCE_SCHEMA`
  compatibility, and that the Parquet schema metadata for
  `input_dataset_revision` and `pipeline_version` is present, UTF-8
  decodable, non-empty, and equal to the corresponding manifest values;
  performs no writes and no network access)

### `osm_polygon_sentence_relevance.publishing`

- `PublicationError` — dedicated publishing-failure error type
  (`ValueError`).
- `PublicationResult` (frozen, slotted dataclass) — verified facts
  about a published Hub commit (`dataset_id`, `target_revision`,
  `commit_id`, `commit_url`, `row_count`, `sha256`).
- `publish_export_directory(export_dir, dataset_id, *,
  target_revision="main", commit_message=None, hub_api=None,
  commit_operation_factory=None) -> PublicationResult` (validates the
  export via `validate_export_directory` first, then publishes
  exactly `sentences.parquet`, `manifest.json`, and the auto-generated
  `README.md` dataset card to the existing Hub dataset in one
  `create_commit` call; no deletes, no repository
  creation, no token handling). Two injectable dependencies model the
  Hub boundary separately: `hub_api` owns `create_commit`, and
  `commit_operation_factory(*, path_in_repo, path_or_fileobj)`
  constructs one add operation per local file. If either is absent,
  only the missing one is imported lazily from `huggingface_hub`;
  fully-injected calls never import the library. Optionally,
  `application.cli` invokes this API after a successful build when
  `--publish-dataset-id` is supplied. See the
  [CLI reference](cli.md) for the exact flags and argument
  relationships.

### Root compatibility facades

These top-level modules are thin facades that re-export the canonical
symbols (implementation lives under `contracts/`, `application/`,
`ingestion/`, `sentences/`, `joins/`, `output/`). They are stable but
new code should prefer the canonical paths above.

- `constants` (facade) — re-exports `contracts.constants`
- `schemas` (facade) — re-exports `contracts.schemas`
- `settings` (facade) — re-exports `application.settings`
- `errors` (facade) — re-exports `contracts.errors`
- `acquisition` (facade) — re-exports `ingestion.acquisition`
- `cli` (facade) — re-exports `application.cli`
- `discovery` (facade) — re-exports `ingestion.discovery`
- `exporter` (facade) — re-exports `output.exporter`
- `finalization` (facade) — re-exports `sentences.finalization`
- `loading` (facade) — re-exports `ingestion.loading`
- `pipeline` (facade) — re-exports `application.pipeline`
- `preprocessing` (facade) — re-exports `sentences.preprocessing`
- `sat_adapter` (facade) — re-exports `sentences.sat`
- `segmentation` (facade) — re-exports `sentences.segmentation`
- `sentence_table` (facade) — re-exports `sentences.table`

Do not rely on private underscore modules (`_export` is gone; the
equivalent helpers now live under `output/`).
