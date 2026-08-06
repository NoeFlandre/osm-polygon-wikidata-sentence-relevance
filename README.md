![Afghanistan sentence relevance dataset overview](docs/assets/afghanistan-labeling-hero.png)

# OSM Polygon – Wikidata Sentence Relevance

[![CI](https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance/actions/workflows/ci.yml/badge.svg)](https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-MkDocs%20Material-526CFE)](https://noeflandre.github.io/osm-polygon-wikidata-sentence-relevance/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](.python-version)

A sentence-level dataset derived from OpenStreetMap polygon metadata,
Wikipedia, and Wikivoyage article sections. The goal is to produce a flat,
deduplicated table of sentences linked to their source polygon, section,
and document metadata — suitable for downstream relevance modelling.

Public documentation: [MkDocs site](https://noeflandre.github.io/osm-polygon-wikidata-sentence-relevance/).
Documentation source: [`docs/index.md`](docs/index.md).

## One-command Grid'5000 production

Install the locked environment on the Mac, connect the external project disk,
and run:

```bash
uv sync --locked --extra operator --extra hub --extra segmentation
uv run osm-polygon-grid5000 run \
  --scope region \
  --region afghanistan-latest \
  --stage all
```

This command preserves the V1 Afghanistan workflow. To start the worldwide
V2 stratified label workflow, use:

```bash
uv run osm-polygon-grid5000 run --scope all --stage label
```

The operator resolves the input dataset to an immutable revision, selects a
compatible Grid'5000 site, stages a clean checkout, submits CUDA sentence
splitting and labeling allocations, streams their logs in this terminal,
continues from durable checkpoints after an expected wall-time stop, validates
the final artifacts, and publishes to
[`NoeFlandre/osm-polygon-wikidata-sentence-relevance`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
on `main`. Pressing Ctrl-C stops only local monitoring; it does not cancel the
remote job or delete checkpoints. Re-running the same command resumes the same
deterministic run.

Every OAR submission is preceded by the live `usagepolicycheck` checks.
Labeling is split into sequential 55-minute allocations bound to OAR's current
day or night/weekend window; each allocation checkpoints after at most 45
minutes and the next one resumes automatically. Site selection uses the
account's `/home` soft-quota headroom rather than the shared filesystem's free
space. All current Grid'5000 sites are probed; unreachable or incompatible
sites are discarded before selection. When headroom is insufficient, only completed or failed
operator-managed runs are eligible for automatic removal; the active run and
its checkpoints are never deleted.

All Mac-side state is stored under the configured external data root. The
command refuses to fall back to the Mac’s internal disk or to run inference
locally.

## Project Repositories

- **GitHub**: [NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
- **Hugging Face (output dataset)**: [NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
- **Hugging Face (input dataset)**: [NoeFlandre/osm-polygon-wikidata-only](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only)

## V1 dataset release

`v1.0.0` is the first public dataset release. It is intentionally limited to
the complete Afghanistan labeling artifact: **54,462 labeled sentences**, 161
polygons, and 115 languages, published on Hugging Face at
[`NoeFlandre/osm-polygon-wikidata-sentence-relevance`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance).
The Python package remains version `0.1.0`; broader regional processing will be
handled by later releases.

## V2 worldwide stratified labeling

V2 is a separate worldwide release stored below `v2-worldwide/` in the same
Hugging Face dataset `main` revision. It defaults to 200,000 sentences selected
by a deterministic proportional prefix across global H3 resolution 3 cells,
language, and OSM primary tag. Increase `--sampling-target` on `run` or
`resume RUN_ID` to extend the same sample without reshuffling earlier rows; the
run identity and checkpoints stay stable. The generated V2 card includes an H3
hexagon map of labeled-sentence coverage and links to a separate worldwide
Trackio dashboard. V1 root files and their Afghanistan dashboard are never
overwritten by V2.

## Current status

The package is pre-1.0 (version `0.1.0`, alpha). It provides a
deterministic, local-first pipeline that:

- discovers per-region Parquet shards and validates them against immutable
  PyArrow schemas;
- builds deterministic Wikipedia and Wikivoyage section→polygon joins;
- segments sections into sentences with an injected segmenter;
- deduplicates exactly, computes deterministic sentence/content IDs, and
  validates the output (`OUTPUT_SENTENCE_SCHEMA`);
- exports the dataset atomically with a checksummed manifest.

Programmatic publishing of a validated local export to an existing
Hugging Face dataset repository is implemented in
`osm_polygon_sentence_relevance.publishing` (one `create_commit`
call, two add operations, no deletes). The build CLI can optionally
publish the completed export with `--publish-dataset-id` (plus optional
`--publish-revision` and `--publish-commit-message`) after a successful
build. The target repository must already exist and no token is accepted.

Restartable builds are supported through an optional `--work-dir`
flag. Each shard is published as a whole-directory atomic rename under
`${work_dir}/shards/active/${shard_key}/` together with a factual
progress `heartbeat.json` at the work-directory root. A subsequent
invocation with the same `--work-dir` resumes from the last valid
checkpoint; invalid or mismatched checkpoints are moved (never deleted)
into `${work_dir}/shards/quarantine/${shard_key}.${utc}.${hex8}/` with
their original bytes preserved byte-for-byte. Each checkpoint carries
a per-file source manifest: the six source files referenced by the
discovered `RegionShardSet` (paths, sizes, and SHA-256). On resume
every source file is re-hashed; any change in bytes, presence or
absence quarantines that shard's checkpoint. A run-level
`shards/inventory.json` reconciles per shard (added / removed / changed
/ unchanged), so adding or removing a single shard never invalidates
the others. `--source-commit` (40-char lowercase hex) is required when
`--work-dir` is set and is recorded into every checkpoint. Heartbeat
failures propagate visibly; they never silently drop a previously
published checkpoint. Cross-shard global deduplication and report
aggregation remain identical with or without `--work-dir`. See
[`docs/reference/cli.md`](docs/reference/cli.md) and
[`docs/guides/reproducibility.md`](docs/guides/reproducibility.md).

**Not implemented (out of scope):**

- Hugging Face dataset repository creation.
- Concurrency (parallel shard segmentation).

The V1 Afghanistan labeling release is available through
`osm-polygon-label-sentences`. It produces independent land-use/land-cover and
target-polygon relevance labels with exact evidence excerpts. Label batches are
atomic, resumable, identity-bound, and timed. Grid'5000 canaries select
deterministic representative rows, validate real structured inference before
labeling, and never publish; only a complete validated run can generate its
factual dataset card and publish to the existing Hub dataset.
Each Grid'5000 batch is also queued for asynchronous upload to a run-specific
`.pipeline/checkpoints/<run-id>/` path on the dataset's `main` tree. This is a
path namespace, not a second public branch. Failed uploads remain durable and
retry on resume.
Production inference uses the Grid'5000 CUDA workflow documented in
[`docs/guides/grid5000.md`](docs/guides/grid5000.md).

## Final Afghanistan labeling metrics

The published card and manifest are rendered from the final labeled Parquet.
The current Afghanistan release contains **54,462 labeled sentences**, **161
unique polygons**, and **115 languages**. The strong-positive yield, where both
questions are `yes`, is **18.20%**. The card also contains the joint-label
heatmap, polygon coverage funnel, normalized reason-code charts, and a
selector-based slice table for language, source, and `osm_primary_tag`.

After finalization, log those same validated facts as one static Trackio run:

```bash
uv sync --locked --extra tracking
uv run osm-polygon-label-sentences track \
  --output-dir /path/to/label-publication \
  --project afghanistan-labeling
```

The default Space follows the release lane in `manifest.json`: V1 uses the public
[Afghanistan Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/afghanistan-labeling-trackio),
while V2 uses the separate
[worldwide dashboard](https://huggingface.co/spaces/NoeFlandre/worldwide-stratified-labeling-trackio).
Use `--space-id OWNER/SPACE` to override the destination.
The command validates the manifest against the Parquet before initializing
Trackio, then records one step (`step=0`) containing the KPI cards, tables,
and PNG plots. The slice table is available in the Trackio run.

For a visual introduction, see the [Afghanistan dataset presentation](https://noeflandre.github.io/osm-polygon-wikidata-sentence-relevance/presentations/afghanistan-dataset-overview/index.html).
For the implementation, see the [codebase overview](https://noeflandre.github.io/osm-polygon-wikidata-sentence-relevance/presentations/codebase-overview/index.html).

## Development setup

This project uses [uv](https://github.com/astral-sh/uv) for Python
package and environment management. Requires Python 3.12+.

```bash
uv sync --locked --all-extras --dev
just check
```

See [`docs/guides/development.md`](docs/guides/development.md) for the
full contributor workflow and verification gates.

## Building the dataset (CLI)

The CLI is the public entry point: `osm-polygon-sentence-relevance`. It
ships with the base install and accepts two mutually-exclusive input modes.
Both modes require `--input-dataset-revision` and `--pipeline-version`.

Local snapshot example (requires the segmentation extra):

```bash
uv sync --extra segmentation
uv run osm-polygon-sentence-relevance \
  --input-root /path/to/snapshot \
  --output-dir ./out \
  --input-dataset-revision abc123... \
  --pipeline-version 0.1.0
```

Hugging Face example (acquires a read-only snapshot, then builds):

```bash
uv sync --extra hub --extra segmentation
uv run osm-polygon-sentence-relevance \
  --input-dataset-id NoeFlandre/osm-polygon-wikidata-only \
  --output-dir ./out \
  --input-dataset-revision main \
  --pipeline-version 0.1.0
```

In Hub mode, the resolved immutable commit SHA is what enters the
pipeline and the manifest. No HF token is accepted, printed, or persisted;
standard `huggingface_hub` authentication is used.

### Hardware selection

The SaT segmenter supports `--device {auto,cpu,cuda,mps}` (default
`auto`): `auto` prefers CUDA when available, otherwise MPS, otherwise
CPU. Explicit `cuda`/`mps` fail with exit code `1` when the requested
backend is unavailable; the CLI never silently downgrades. Hardware
selection happens after acquisition, only when the model is built, and
it does not alter output schema, IDs, hashes, or dataset-card
statistics. **One GPU only; multi-GPU is not implemented.** Production
Grid'5000 runs should use the bounded streaming workflow documented in
[`docs/guides/grid5000.md`](docs/guides/grid5000.md).

### Local source provenance

`--input-source-dataset-id OWNER/DATASET` records the upstream source
dataset ID for an already-local snapshot. Only valid with
`--input-root`; populates the source provenance threaded into the
manifest, statistics, and the generated `README.md` dataset card without
triggering any network access.

## Optional extras

The base install pulls in only `pyarrow`. Three extras are available:

- `segmentation` (`wtpsplit==2.2.1` + `torch>=2.2,<3`) — installs the
  `wtpsplit` SaT adapter and its PyTorch runtime, as required by the
  default `SaTSentenceSegmenter`. The SaT model weights themselves are
  still downloaded separately on first model construction. **`wtpsplit`
  is pinned to exactly `2.2.1`**: the placement adapter is
  intentionally version-specific (it descends into the
  `PyTorchWrapper` that ships with 2.2.1) and refuses any other
  version at runtime. A wider range would invite a configuration the
  adapter has not been tested against.
- `hub` (`huggingface_hub>=0.20.0`) — required for Hub input acquisition
  through `--input-dataset-id` and for programmatic publishing through
  `publish_export_directory`.
- `tracking` (`trackio==0.26.0`) — required for the explicit `track` command,
  which logs one final static run from a validated labeled publication.

Both extras are imported lazily; importing their respective modules is
side-effect-free when the dependency is not installed.

## Documentation

- [Architecture overview](docs/architecture/overview.md)
- [Getting started](docs/guides/getting-started.md)
- [Development](docs/guides/development.md)
- [Reproducibility](docs/guides/reproducibility.md)
- [API reference](docs/reference/api.md)
- [CLI reference](docs/reference/cli.md)
- [Data contract](docs/reference/data-contract.md)

## Governance

- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [License (MIT)](LICENSE)
