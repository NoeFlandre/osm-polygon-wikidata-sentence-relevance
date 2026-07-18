# Getting started

This guide covers installation and a minimal first run. The pipeline is
local-first and deterministic. Programmatic publishing of a validated
local export to an existing Hugging Face dataset repository is
implemented (see `osm_polygon_sentence_relevance.publishing`), and the
build CLI can optionally publish the completed export with
`--publish-dataset-id`; Hugging Face repository creation is not.

## Installation

We use [`uv`](https://github.com/astral-sh/uv). No system Python packages
are required.

```bash
# Core install: schemas, joins, finalization, export, CLI, lazy SaT stubs.
uv sync

# Add the wtpsplit SaT adapter/library and its required PyTorch
# runtime (used by the default segmenter). SaT model weights are
# fetched separately when SaT is first constructed.
uv sync --extra segmentation

# Add the Hugging Face Hub extra: enables read-only acquisition and
# programmatic publishing.
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

## Build and publish

Add `--publish-dataset-id` to publish the completed export to an existing
Hugging Face dataset repository after the build succeeds. The target
repository must already exist; no token flag is accepted.

```bash
uv sync --extra hub --extra segmentation
uv run osm-polygon-sentence-relevance \
  --input-dataset-id NoeFlandre/osm-polygon-wikidata-only \
  --output-dir ./out \
  --input-dataset-revision main \
  --pipeline-version 0.1.0 \
  --publish-dataset-id NoeFlandre/osm-polygon-wikidata-sentence-relevance \
  --publish-revision main
```

The summary JSON then gains a `publication` object with the resolved
`commit_id`, `commit_url`, `row_count`, and `sha256`.

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

## Programmatic publishing

Publishing a validated export is a programmatic, single-commit operation
via `osm_polygon_sentence_relevance.publishing`. The target Hugging Face
dataset repository must already exist; standard Hugging Face
authentication is used and **no token is passed to this function**. The
export is validated locally before any Hub call. (For an end-to-end build
that publishes, use the build CLI's `--publish-dataset-id` flag instead;
see [Build and publish](#build-and-publish).)

```bash
uv sync --extra hub   # enables Hugging Face acquisition and programmatic publishing
```

```python
from osm_polygon_sentence_relevance.publishing import (
    publish_export_directory,
)

result = publish_export_directory(
    "./out",                                   # local output dir with sentences.parquet + manifest.json
    "owner/dataset",                           # existing Hugging Face dataset repo id
    target_revision="main",
)

print(result.commit_url)
```

`publish_export_directory` raises `PublicationError` if the export fails
validation, the Hub extra is missing, or the commit response is malformed
(see [the API reference](../reference/api.md) for the full contract). It
does not create repositories and does not retry.
