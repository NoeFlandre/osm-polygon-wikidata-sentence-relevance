# Contributing

Thanks for your interest in improving the OSM Polygon Sentence Relevance
pipeline. This document covers the developer workflow and the expectations
for changes.

## Scope

This is an alpha (pre-1.0) project. The current goal is a deterministic,
local-first pipeline that builds a sentence-level relevance dataset from
OSM polygons joined to Wikipedia and Wikivoyage sections.

Programmatic publishing of a validated export to an existing Hugging
Face dataset repository is implemented under
`osm_polygon_sentence_relevance.publishing` (see
`publishing/huggingface.publish_export_directory`). The build CLI also
publishes the completed export via the `--publish-dataset-id` flag
(strictly post-build, to an existing repository only).

Out of scope (do not add in a normal pull request unless explicitly
planned):
- Hugging Face dataset repository creation (the publisher targets an
  existing repository only).
- Sentence classification or labelling.
- Parallel shard processing.
- Performance rewrites that change output bytes.

## Environment setup

We use [`uv`](https://github.com/astral-sh/uv) for dependency and
environment management. No system Python packages are required.

```bash
uv sync
uv sync --extra segmentation   # install wtpsplit SaT adapter + its PyTorch runtime (SaT weights still download lazily on first model construction)
uv sync --extra hub            # enable read-only Hugging Face acquisition and programmatic publishing
```

## Test-driven development

Every behavior change starts with a failing test (red), then the minimum
implementation to make it pass (green). Structural refactors are protected
by characterization tests before any move.

- Preserve all existing tests; do not weaken or delete them to pass.
- Keep public import paths stable. If a module is relocated, keep a thin
  compatibility facade at the old path re-exporting the stable symbols.
- New tests go under `tests/` mirroring the `src/` package layout
  (`unit/`, `integration/`, `compatibility/`, `support/`).

## Required quality commands

All of these must pass before opening a pull request:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
uv run pytest --cov=osm_polygon_sentence_relevance --cov-branch --cov-report=term-missing
uv build
uv run python scripts/verify_distribution.py <wheel> <sdist>
uv run osm-polygon-sentence-relevance --help
```

## Architecture rules

- Canonical cross-cutting contracts (constants, schemas, errors) live
  under `contracts/`. The legacy top-level modules (`constants.py`,
  `errors.py`, `schemas.py`, `settings.py`) are thin compatibility
  facades that re-export the canonical symbols — they are not
  canonical ownership.
- `PipelineSettings` is owned canonically by `application/settings.py`;
  the top-level `settings.py` is a compatibility facade.
- Operational code lives in domain packages: `application/`,
  `ingestion/`, `sentences/`, `joins/`, `output/`, and `publishing/`.
- Production imports use canonical domain paths, never compatibility
  facades.
- Do not add generic frameworks or speculative abstractions (YAGNI).

## Pull-request checklist

- [ ] Red test added before implementation.
- [ ] `ruff format`, `ruff check`, `mypy`, and `pytest` pass.
- [ ] Branch coverage threshold maintained (see CI).
- [ ] Public API, schemas, hashes, IDs, ordering, and output bytes unchanged
      for the same `(input_dataset_revision, pipeline_version)`.
- [ ] `uv.lock` updated via `uv lock` if dependencies changed.
- [ ] Documentation updated where behavior is described.

## Repository hygiene

Do not commit:
- data, Parquet files, or model weights;
- credentials or tokens (no `.env`, no Hugging Face tokens on disk);
- generated datasets or build artifacts (`dist/`, `build/`, `egg-info/`,
  caches).

The local-only coding-agent handoff guide under `.local-docs/` is
intentionally excluded from version control and must never be staged.
