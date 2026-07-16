# OSM Polygon – Wikidata Sentence Relevance

A sentence-level dataset derived from OpenStreetMap polygon metadata,
Wikipedia, and Wikivoyage article sections. The goal is to produce a flat,
deduplicated table of sentences linked to their source polygon, section,
and document metadata — suitable for downstream relevance modelling.

See [`docs/architecture.md`](docs/architecture.md),
[`docs/reproducibility.md`](docs/reproducibility.md), and
[`docs/data-contract.md`](docs/data-contract.md) for module ownership,
reproduction commands, and the data contract.

## Project Repositories

- **GitHub**: [NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
- **Hugging Face (output dataset)**: [NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
- **Hugging Face (input dataset)**: [NoeFlandre/osm-polygon-wikidata-only](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only)

## Current Implementation Status (through Phase 6C)

Implemented:

- **Phase 1** — src-layout package, PyArrow schema contracts (six input
  tables + output sentence table), immutable pipeline settings.
- **Phase 2** — shard discovery, Parquet loading, deterministic Wikipedia
  and Wikivoyage section→polygon joins (`JOINED_SECTIONS_SCHEMA`).
- **Phase 3** — deterministic preprocessing, injected segmenter table
  transformation (`SEGMENTED_SENTENCES_SCHEMA`) with the optional
  `SaTSentenceSegmenter` adapter for the wtpsplit multilingual SaT model.
- **Phase 4** — exact deduplication, deterministic sentence/content IDs,
  `OUTPUT_SENTENCE_SCHEMA` finalization.
- **Phase 5** — atomic, deterministic local export with rollback-safe
  overwrite and checksummed manifest.
- **Phase 6 A** — local-build CLI with strict path/config validation.
- **Phase 6 B** — read-only Hugging Face input-snapshot acquisition
  (snapshot validation, SHA-1 immutable revision resolution, inclusive
  Parquet-only download with `articles/` exclusion).
- **Phase 6 C** — CLI integration of the two mutually-exclusive input
  modes (local vs Hub); Hub mode resolves the mutable revision to a
  commit SHA before any model construction.

**Not implemented (explicitly out of scope for the current series):**

- Phase 7 and beyond, including Hugging Face publishing / dataset
  upload of the produced dataset.
- Sentence classification or labelling.
- Concurrency, resumability, or incremental rebuilds.
- Performance rewrites.

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python
package and environment management.

### Prerequisites

- Python 3.12+ (managed by `uv`)
- `uv` package manager

### Installation

```bash
uv sync
```

### Running Tests

```bash
uv run pytest -q
```

## Building the Dataset (CLI)

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

## Optional Extras

The base install pulls in only `pyarrow`. Two extras are available:

- `segmentation` (`wtpsplit>=2.2.1,<3`) — required by the default
  `SaTSentenceSegmenter` adapter.
- `hub` (`huggingface_hub>=0.20.0`) — required for `--input-dataset-id`.

Both extras are imported lazily; importing their respective modules is
side-effect-free when the dependency is not installed.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — module ownership, input
  flow, optional-dependency boundaries.
- [`docs/reproducibility.md`](docs/reproducibility.md) — environment
  requirements, exact build commands, verification commands.
- [`docs/data-contract.md`](docs/data-contract.md) — input paths,
  dedup key, deterministic IDs, output schema expectations.
