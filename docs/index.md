# OSM Polygon Sentence Relevance — Documentation

This project builds a deterministic, sentence-level relevance dataset from
OSM polygons joined to Wikipedia and Wikivoyage sections. It is currently
pre-1.0 (alpha).

## Audience entry points

- **I want to run it once** → [Getting started](guides/getting-started.md)
- **I am developing or contributing** →
  [Development](guides/development.md)
- **I need to reproduce a build exactly** →
  [Reproducibility](guides/reproducibility.md)
- **I want the architecture and module ownership** →
  [Architecture overview](architecture/overview.md)
- **I want the public API reference** → [API](reference/api.md)
- **I want the CLI reference** → [CLI](reference/cli.md)
- **I want the data contract (schemas, IDs, normalization)** →
  [Data contract](reference/data-contract.md)
- **I want the why behind the package layout** →
  [ADR 0001: Domain package layout](architecture/decisions/0001-domain-package-layout.md)

## Repository layout

- `src/osm_polygon_sentence_relevance/` — the package, organized into
  `application/`, `ingestion/`, `sentences/`, `joins/`, `output/`.
- `tests/` — mirrored unit/integration/compatibility structure with shared
  `tests/support/` factories.
- `docs/` — this documentation tree.
- `scripts/verify_distribution.py` — stdlib-only distribution-content check.
- Root governance files: `README.md`, `LICENSE`, `CHANGELOG.md`,
  `CONTRIBUTING.md`, `SECURITY.md`, `MANIFEST.in`, `pyproject.toml`.

## Scope

Not implemented: Hugging Face dataset publishing/upload, sentence
classification or labelling, concurrency, resumable or incremental builds.
See the [architecture overview](architecture/overview.md) for invariants.
