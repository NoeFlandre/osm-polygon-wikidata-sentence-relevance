# OSM Polygon – Wikidata Sentence Relevance

[![CI](https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance/actions/workflows/ci.yml/badge.svg)](https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](.python-version)

A sentence-level dataset derived from OpenStreetMap polygon metadata,
Wikipedia, and Wikivoyage article sections. The goal is to produce a flat,
deduplicated table of sentences linked to their source polygon, section,
and document metadata — suitable for downstream relevance modelling.

Documentation index: [`docs/index.md`](docs/index.md).

## Project Repositories

- **GitHub**: [NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
- **Hugging Face (output dataset)**: [NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
- **Hugging Face (input dataset)**: [NoeFlandre/osm-polygon-wikidata-only](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only)

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

An Afghanistan-only labeling proof of concept is available through
`osm-polygon-label-sentences`. It produces independent land-use/land-cover and
target-polygon relevance labels with exact evidence excerpts. Label batches are
atomic, resumable, identity-bound, and timed; only a complete validated run can
generate its factual dataset card and publish to the existing Hub dataset.
Production inference uses the Grid'5000 CUDA workflow documented in
[`docs/guides/grid5000.md`](docs/guides/grid5000.md).

## Development setup

This project uses [uv](https://github.com/astral-sh/uv) for Python
package and environment management. Requires Python 3.12+.

```bash
uv sync --extra hub --extra segmentation
uv run pytest -q
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

The base install pulls in only `pyarrow`. Two extras are available:

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
