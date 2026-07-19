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
    [--sat-model SAT_MODEL]
    [--device {auto,cpu,cuda,mps}]
    [--input-source-dataset-id OWNER/DATASET]
    [--overwrite]
    [--publish-dataset-id OWNER/DATASET]
    [--publish-revision REVISION] [--publish-commit-message MESSAGE]
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
| `--device` | no | `auto` | Accelerator for SaT inference. One of `auto`, `cpu`, `cuda`, `mps`. `auto` (default) prefers CUDA when available, otherwise MPS, otherwise CPU. Explicit `cuda` or `mps` fail with exit code `1` when the requested backend is unavailable; the CLI never silently downgrades. |
| `--input-source-dataset-id` | no | — | Optional Hugging Face dataset ID of the upstream source for a local input snapshot. Only valid with `--input-root`; populates the source provenance recorded in the manifest and dataset card without triggering any network request. |
| `--overwrite` | no | off | Overwrite an existing non-empty output directory. |
| `--publish-dataset-id` | no | — | Optional Hugging Face dataset ID to publish the completed export to, after the build succeeds. The target repository must already exist; no repository is created. |
| `--publish-revision` | no | `main` | Target Hugging Face dataset revision for publishing. Only used with `--publish-dataset-id`. |
| `--publish-commit-message` | no | — | Optional commit message for the publishing commit. Only used with `--publish-dataset-id`. |

`--input-root` and `--input-dataset-id` are mutually exclusive and at least
one is required. Supplying both, or neither, exits with status `2`.

The three publishing flags are optional and related: `--publish-revision`
and `--publish-commit-message` are only valid together with
`--publish-dataset-id`. Supplying either without a publishing dataset ID
exits with status `1` before any acquisition, model construction, or
pipeline execution. No token or repository-creation flag exists.

`--device` is validated for syntactic correctness and hardware availability
before any acquisition, model construction, or pipeline execution.
Explicit `cuda` or `mps` exit with status `1` when the requested backend
is unavailable on the host.

`--input-source-dataset-id` is only valid together with `--input-root`;
supplying it in Hub mode (`--input-dataset-id`) or with a blank value
exits with status `1` before any acquisition, model construction, or
pipeline execution. When supplied it records the source provenance in
the manifest and generated dataset card.

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
contains `parquet_path`, `manifest_path`, `card_path`,
`processed_regions_count`,
`total_joined_section_occurrences`, an `input` object
(`mode`, `dataset_id`, `requested_revision`, `resolved_revision`,
`snapshot_path`), and `segmentation_report` / `finalization_report`
summaries. `card_path` points at the auto-generated `README.md`
dataset card produced from the validated export.

When `--publish-dataset-id` is supplied, a `publication` object is added
with `dataset_id`, `target_revision`, `commit_id`, `commit_url`,
`row_count`, and `sha256`. When publishing is not requested, the success
JSON is unchanged and has no `publication` key.

In Hub mode (`--input-dataset-id` supplied), the `input.dataset_id` and
`input.resolved_revision` values reflect the upstream dataset identity
that the success JSON also passes through to the manifest, statistics,
Parquet schema metadata, and the generated `README.md` dataset card via
`run_pipeline(input_dataset_id=...)`. In local mode (`--input-root`),
`input.dataset_id` is `null` and no Hub identity is propagated.

## Publishing (optional, post-build)

When `--publish-dataset-id OWNER/DATASET` is supplied, the CLI publishes
the successfully exported `sentences.parquet`, `manifest.json`, and
auto-generated `README.md` dataset card to an **existing** Hugging Face
dataset repository, in a single commit, only
after the build succeeds. The publisher revalidates the local export
before any network call and uses standard Hugging Face authentication
(no token is accepted by this command). `--publish-revision` selects the
target branch (default `main`); `--publish-commit-message` overrides the
default deterministic message.

Publishing is strictly post-build: a pipeline failure yields zero
publishing attempts and exit code `1`, and a publishing failure preserves
the already-created local export and exits `1` with a concise stderr
message. The CLI never creates repositories, retries, or adds token
handling. For programmatic control, use
`osm_polygon_sentence_relevance.publishing.publish_export_directory`
directly.
