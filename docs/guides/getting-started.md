# Getting started

This guide covers installation and a minimal first run. The pipeline is
local-first and deterministic; publishing to Hugging Face is **not**
implemented.

## Installation

We use [`uv`](https://github.com/astral-sh/uv). No system Python packages
are required.

```bash
# Core install: schemas, joins, finalization, export, CLI, lazy SaT stubs.
uv sync

# Add the wtpsplit SaT segmentation model (used by the default segmenter).
uv sync --extra segmentation

# Add the read-only Hugging Face Hub acquisition path.
uv sync --extra hub

# Combine extras (local and Hub builds both require the segmentation extra).
uv sync --extra hub --extra segmentation
```

The package version is `0.1.0` (alpha). The installed console command is
`osm-polygon-sentence-relevance`.

## Local build

```bash
uv sync --extra segmentation
uv run osm-polygon-sentence-relevance \
  --input-root /path/to/snapshot \
  --output-dir ./out \
  --input-dataset-revision <revision-or-sha> \
  --pipeline-version 0.1.0
```

## Hub build

`--input-dataset-revision` is **required for both** modes. In Hub mode it
is the name you want to resolve (for example `main` or a commit SHA); the
resolved immutable commit SHA is what flows into the output.

```bash
uv sync --extra hub --extra segmentation
uv run osm-polygon-sentence-relevance \
  --input-dataset-id NoeFlandre/osm-polygon-wikidata-only \
  --output-dir ./out \
  --input-dataset-revision main \
  --pipeline-version 0.1.0
```

## Expected output files

For a successful run, `output-dir/` contains:

- `sentences.parquet` — the finalized sentence table.
- `manifest.json` — per-source/language/region counts, the
  `input_dataset_revision`, the `pipeline_version`, and the lower-cased
  hex SHA-256 of the Parquet file.

No token flags exist. The CLI never accepts, prints, or persists a
Hugging Face token.

## Scope note

Classification, labelling, concurrency, resumable, and incremental builds
are not implemented.
