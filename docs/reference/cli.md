# CLI reference

Console command (installed via `uv sync`):

```text
osm-polygon-sentence-relevance
```

The Afghanistan labeling proof of concept has a separate focused entry point:

```text
osm-polygon-label-sentences {label,finalize,publish}
```

`label` requires immutable input/model/source revisions, a persistent work
directory, the selected inference engine and version, and a positive batch
size. It checkpoints each validated batch and prints JSON containing completed
and total rows, interruption status, elapsed seconds, and input SHA-256.
`finalize` refuses partial labels and creates the labeled Parquet, manifest,
concise data-derived README, and two plots. `publish` revalidates those five
files and uploads them to the existing dataset in one commit. No command
accepts a token or creates a repository.

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
    [--work-dir WORK_DIR]
    [--source-commit SOURCE_COMMIT]
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
| `--work-dir` | no | — | Optional persistent work directory for shard-level checkpoints and a factual progress heartbeat. When supplied, the pipeline publishes one checkpoint per shard as a whole-directory atomic rename under `${work_dir}/shards/active/<shard_key>/`, and writes a `heartbeat.json` updated at shard boundaries. A subsequent invocation with the same `--work-dir` resumes from the last valid checkpoint; invalid or mismatched checkpoints are moved into `${work_dir}/shards/quarantine/` with a UUID-suffixed unique name and their original bytes are preserved (never deleted). Cannot overlap with `--input-root` or `--output-dir`; ignored when omitted (legacy no-work-directory mode). |
| `--source-commit` | required iff `--work-dir` is set | — | 40-character lowercase hex commit SHA binding each checkpoint and the heartbeat to a specific code revision. Validated against `^[0-9a-f]{40}$` and recorded verbatim into every checkpoint's metadata. When `--work-dir` is omitted the value is ignored. |
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

`--work-dir` is optional and orthogonal to the input/output flags. The
CLI validates the syntactic form; the pipeline then rejects overlap
with `--input-root` / `--output-dir` (including ancestor relationships)
and refuses to write checkpoints for a `--work-dir` equal to or inside
either of those paths. The pipeline also records the current source
commit (from `osm_polygon_sentence_relevance.__version__`) and the
selected `--sat-model` into every shard checkpoint and the heartbeat;
this binds each artifact to the exact code revision and model that
produced it.

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

## Restartable builds (`--work-dir`)

When `--work-dir` is supplied, the pipeline persists per-shard
checkpoints and a factual progress heartbeat so a long build can be
interrupted and resumed without re-running already-validated shards.
Layout:

```text
${WORK_DIR}/
    heartbeat.json                  # atomic, factual-only JSON
    shards/
        inventory.json              # run-level snapshot
        active/
            ${shard_key}/
                segmented.parquet   # mode 0600
                metadata.json       # mode 0600
        quarantine/
            ${shard_key}.${utc}.${hex8}/   # mode 0700; byte-identical to
                                            # the rejected active checkpoint
```

The pipeline is **single-writer**: at most one invocation may own a
given `--work-dir` at a time. Cross-filesystem moves are not supported;
the active and quarantine trees live on the same filesystem.

**Source-file binding.** Each checkpoint carries a `source_files` field
in `metadata.json` whose entries are exactly the six files referenced
by the corresponding `RegionShardSet` discovered for that shard (four
core Parquet files plus the optional Wikivoyage pair). Each entry
records the canonical forward-slash relative path, byte size, and
SHA-256. On resume every referenced source file is re-hashed; any
change in size, SHA-256, presence, or absence triggers quarantine of
that shard's active directory. The pipeline therefore detects edits to
source bytes even when the file path and pinned revision are
unchanged.

**Run-level inventory.** `shards/inventory.json` records the discovered
shard keys and each shard's source manifest. Reconciliation is
**per shard**:

- an added shard is processed alone, the unchanged ones are reused;
- a removed shard quarantines its own orphaned checkpoint;
- a changed shard quarantines only its own active directory;
- unchanged shards with matching manifests reuse their previously
  published bytes — joins and segmentation are not invoked for them.

**Publication.** A new checkpoint is published as a whole-directory
atomic rename from a unique sibling staging directory into `active/`.
The staging directory is preserved on every failure as evidence; the
active slot is never modified in place, never overwritten, never
silently replaced. On any failure mid-publish the previous active
checkpoint (if any) is unaffected.

**Quarantine, never delete.** When a checkpoint is rejected for any
reason — corrupt metadata, SHA mismatch, missing report field,
unexpected directory entries, wrong mode, stale identity, source-file
drift, wrong schema version — its entire active directory is moved
into `quarantine/`. The bytes are preserved exactly (no chmod, no
rewrite) under a UUID-suffixed unique name. **No code path
automatically deletes checkpoints.** Multiple failed attempts
accumulate as multiple quarantine directories.

**Heartbeat.** `heartbeat.json` records factual values only:
`stage`, `total_shards`, `completed_shards`, `current_shard_key`,
`retained_sentence_occurrence_count`, `dropped_empty_raw_count`,
`dropped_empty_normalized_count`, `elapsed_seconds`,
`input_dataset_revision`, `source_commit`. **Heartbeat failures
propagate visibly** — they raise and abort the current run, while
every successfully published checkpoint remains valid for the next
run. The pipeline never silently swallows heartbeat errors.

**Required flag.** `--source-commit` (40-char lowercase hex) is
required when `--work-dir` is set and is recorded into every
checkpoint's metadata and the heartbeat. The CLI rejects non-conforming
values before any acquisition or model construction.

A failure before final export preserves every previously-published
checkpoint and the previously-completed output directory; the resume
reuses the checkpoint bytes and only re-runs the shards that were not
yet validated at the moment of failure.

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
