# API reference

This page documents the supported **canonical** imports (implementation in
domain packages) and lists the **compatibility** imports (legacy top-level
facades). Private underscore modules are intentionally not documented.

All canonical modules live under
`src/osm_polygon_sentence_relevance/`.

## Canonical imports

### `osm_polygon_sentence_relevance.application`

- `main(args=None, *, model_factory=None, acquisition_fn=None) -> int`
- `PipelineResult` (dataclass)
- `run_pipeline(input_root, output_dir, segmenter, *, input_dataset_revision,
  pipeline_version, batch_size=128, overwrite=False) -> PipelineResult`

### `osm_polygon_sentence_relevance.ingestion`

- `acquisition.AcquisitionResult`
- `acquisition.acquire_dataset_snapshot(dataset_id, requested_revision, *,
  hub_api=None, download_fn=None) -> AcquisitionResult`
  (optional `huggingface_hub` extra; lazy import)
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
- `sat.SaTSentenceSegmenter` (optional `wtpsplit` extra; lazy import)
- `table.SegmentedTableResult`
- `table.segment_joined_sections(...)`
- `finalization.FinalizationReport`
- `finalization.FinalizedDataset`
- `finalization.sentence_content_hash(text) -> str`
- `finalization.deterministic_sentence_id(...) -> str`
- `finalization.finalize_sentence_dataset(...) -> FinalizedDataset`

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

### Root contracts

- `constants` — `INPUT_DATASET_ID`, `ALLOWED_INPUT_PATHS`, `ALLOW_PATTERNS`,
  `IGNORE_PATTERNS`, `ALLOWED_SOURCES`, `PIPELINE_VERSION`.
- `schemas` — all PyArrow schemas (`OUTPUT_SENTENCE_SCHEMA`,
  `SEGMENTED_SENTENCES_SCHEMA`, `JOINED_SECTIONS_SCHEMA`, input schemas).
- `settings` — `PipelineSettings`.
- `errors` — `ConfigurationError`, `SchemaContractError`,
  `PreprocessingError`, `SegmentationError`, `ShardDiscoveryError`,
  `JoinIntegrityError`, `FinalizationError`, `ExportError`,
  `AcquisitionError`.

## Compatibility imports

The following legacy top-level modules remain as thin facades that
re-export the canonical symbols. They are stable but new code should
prefer the canonical paths above.

- `osm_polygon_sentence_relevance.acquisition`
- `osm_polygon_sentence_relevance.cli`
- `osm_polygon_sentence_relevance.discovery`
- `osm_polygon_sentence_relevance.exporter`
- `osm_polygon_sentence_relevance.finalization`
- `osm_polygon_sentence_relevance.loading`
- `osm_polygon_sentence_relevance.pipeline`
- `osm_polygon_sentence_relevance.preprocessing`
- `osm_polygon_sentence_relevance.sat_adapter`
- `osm_polygon_sentence_relevance.segmentation`
- `osm_polygon_sentence_relevance.sentence_table`

Do not rely on private underscore modules (`_export` is gone; the
equivalent helpers now live under `output/`).
