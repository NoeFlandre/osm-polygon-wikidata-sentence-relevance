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

**Phase 3 (preprocessing & segmentation)** is complete: deterministic preprocessing (section-path parsing, OSM-tag parsing, sentence normalization) and the injected-segmenter table transformation (`segment_joined_sections`) that builds `SEGMENTED_SENTENCES_SCHEMA` with batching, validation, and reporting.

Not yet implemented:

- The real multilingual sentence-segmentation model adapter (only the `SentenceSegmenter` protocol and a validation boundary exist).
- Phase 4 deduplication, classification, dataset upload, and any CLI.

Later phases will add the model adapter and Phase 4 pipeline stages.
