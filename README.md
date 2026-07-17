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
call, two add operations, no deletes). No token parameter is accepted.

**Not implemented (out of scope):**

- CLI publishing flags for the Hugging Face dataset publisher.
- Hugging Face dataset repository creation.
- Sentence classification or labelling.
- Concurrency, resumable, or incremental builds.

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

## Optional extras

The base install pulls in only `pyarrow`. Two extras are available:

- `segmentation` (`wtpsplit>=2.2.1,<3`) — required by the default
  `SaTSentenceSegmenter` adapter.
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
