# CLI reference

Console command (installed via `uv sync`):

```text
osm-polygon-sentence-relevance
```

## Usage

```text
usage: osm-polygon-sentence-relevance [-h]
    (--input-root INPUT_ROOT | --input-dataset-id INPUT_DATASET_ID)
    --output-dir OUTPUT_DIR
    --input-dataset-revision INPUT_DATASET_REVISION
    --pipeline-version PIPELINE_VERSION
    [--batch-size BATCH_SIZE]
    [--sat-model SAT_MODEL] [--overwrite]
```

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-root` | one of the two input modes | — | Existing local input snapshot root directory. |
| `--input-dataset-id` | one of the two input modes | — | Upstream Hugging Face dataset ID to acquire a read-only snapshot from. |
| `--output-dir` | yes | — | Output directory. |
| `--input-dataset-revision` | yes | — | Input dataset revision. In Hub mode a mutable name (e.g. `main`) is resolved to an immutable commit SHA. |
| `--pipeline-version` | yes | — | Pipeline version recorded into the output metadata. |
| `--batch-size` | no | `128` | Batch size for the segmenter (must be a positive integer). |
| `--sat-model` | no | `sat-3l-sm` | wtpsplit SaT model name. |
| `--overwrite` | no | off | Overwrite an existing non-empty output directory. |

`--input-root` and `--input-dataset-id` are mutually exclusive and at least
one is required. Supplying both, or neither, exits with status `2`.

## Exit statuses

| Status | Meaning |
|--------|---------|
| `0` | Success. A single line of JSON is printed to stdout. |
| `1` | Runtime failure (invalid arguments, acquisition failure, export error). A concise message is printed to stderr; no success JSON. |
| `2` | Argument-parsing error (mutually-exclusive or required input mode violated). |

## Local mode

When `--input-root` is given, `--input-dataset-revision` is forwarded
unchanged into the output `input_dataset_revision` metadata. No Hugging
Face access occurs.

## Hub mode

When `--input-dataset-id` is given, the CLI resolves the requested
revision to an immutable commit SHA via
`acquire_dataset_snapshot`, downloads the Parquet-only snapshot
(excluding `articles/`), and forwards the resolved SHA (never a mutable
`main`) as the pipeline `input_dataset_revision`. Acquisition runs before
model construction, so an acquisition failure never triggers a
model-weight download.

## Success JSON

On success, a single JSON object is printed (keys sorted, compact). It
contains `parquet_path`, `manifest_path`, `processed_regions_count`,
`total_joined_section_occurrences`, an `input` object
(`mode`, `dataset_id`, `requested_revision`, `resolved_revision`,
`snapshot_path`), and `segmentation_report` / `finalization_report`
summaries.
