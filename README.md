# OSM Polygon – Wikidata Sentence Relevance

A sentence-level dataset derived from OpenStreetMap polygon metadata, Wikipedia, and Wikivoyage article sections. The goal is to produce a flat, deduplicated table of sentences linked to their source polygon, section, and document metadata—suitable for downstream relevance modelling.

## Project Repositories

- **GitHub**: [NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
- **Hugging Face (output dataset)**: [NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
- **Hugging Face (input dataset)**: [NoeFlandre/osm-polygon-wikidata-only](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only)

## Input Data

The pipeline reads from six authoritative subdirectories of the input dataset:

| Subdirectory | Description |
|---|---|
| `polygons/` | OSM polygon metadata (one row per polygon) |
| `polygon_articles/` | Polygon ↔ article link table |
| `wikipedia/documents/` | Wikipedia document metadata |
| `wikipedia/sections/` | Wikipedia section text |
| `wikivoyage/documents/` | Wikivoyage document metadata |
| `wikivoyage/sections/` | Wikivoyage section text |

Both **Wikipedia** and **Wikivoyage** are in scope.

> **Note:** The obsolete `articles/` directory is intentionally excluded and must never be used.

## Data Storage

### Local Data Path

Data directory resolution (in order of precedence):

1. `OSM_DATA_DIR` environment variable
2. `/Volumes/Seagate M3/projects/osm-polygon-wikidata-sentence-relevance` (external drive)
3. `data/` in the repository root (git-ignored)

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for Python package and environment management.

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

### Running the Stub Entry Point

```bash
uv run python main.py
```

## Current Status

**Phase 1 (foundation)** is complete: src-layout package, PyArrow schema contracts for all six input tables and the output sentence table, immutable pipeline settings, and comprehensive tests.

**Phase 2 (ingestion & joins)** is complete: shard discovery, Parquet loading, and the deterministic Wikipedia/Wikivoyage section-to-polygon join producing `JOINED_SECTIONS_SCHEMA`.

**Phase 3 (preprocessing & segmentation)** is complete: deterministic preprocessing (section-path parsing, OSM-tag parsing, sentence normalization), the injected-segmenter table transformation (`segment_joined_sections`) that builds `SEGMENTED_SENTENCES_SCHEMA` with batching, validation, and reporting, and an optional `SaTSentenceSegmenter` adapter for the wtpsplit multilingual SaT model (installed via the `segmentation` extra, see below).

Not yet implemented:

- Phase 4 deduplication, classification, dataset upload, and any CLI.

Later phases will add Phase 4 pipeline stages.

## Optional Segmentation Model (wtpsplit / SaT)

The base install (`uv sync`) is intentionally lightweight and does **not**
pull in a heavy ML dependency. A concrete segmenter backed by
[wtpsplit](https://github.com/segment-any-text/wtpsplit)'s multilingual SaT model
is provided as an optional extra:

```bash
uv sync --extra segmentation
```

This installs `wtpsplit>=2.2.1,<3` in addition to the core dependencies.

Notes:

- Model weights are downloaded and cached by the underlying `wtpsplit` library
  on first use. **No model weights are stored in this repository.**
- Importing `osm_polygon_sentence_relevance.sat_adapter` is side-effect-free;
  the model is constructed lazily on the first non-empty `split_batch` call.
- Plain `uv sync` and the full test suite continue to work without
  wtpsplit installed.

Usage example:

```python
from osm_polygon_sentence_relevance.sat_adapter import SaTSentenceSegmenter
from osm_polygon_sentence_relevance.segmentation import split_validated_batch

segmenter = SaTSentenceSegmenter()  # defaults to "sat-3l-sm"
groups = split_validated_batch(segmenter, ["First sentence. Second one."], ["en"])
```

## Building the Dataset (CLI)

The CLI builds from **either** an existing local snapshot **or** the upstream
Hugging Face dataset. The two input modes are mutually exclusive. The input
revision is required for both and, in Hub mode, is resolved to an immutable
commit SHA that is forwarded into the pipeline.

Local snapshot example (requires the segmentation extra for the SaT model):

```bash
uv sync --extra segmentation
uv run osm-polygon-sentence-relevance \
  --input-root /path/to/snapshot \
  --output-dir ./out \
  --input-dataset-revision abc123... \
  --pipeline-version 0.1.0
```

Hugging Face example (acquires a read-only snapshot, then builds; requires
both the hub and segmentation extras):

```bash
uv sync --extra hub --extra segmentation
uv run osm-polygon-sentence-relevance \
  --input-dataset-id NoeFlandre/osm-polygon-wikidata-only \
  --output-dir ./out \
  --input-dataset-revision main \
  --pipeline-version 0.1.0
```

The success JSON includes an `input` object reporting the mode, dataset id,
requested and resolved revisions, and the snapshot path used. No HF token is
accepted, printed, or persisted; standard `huggingface_hub` authentication is
used.
