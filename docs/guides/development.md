# Development

This guide covers the contributor workflow, repository layout, and the
verification gates used in CI.

## Environment setup

```bash
uv sync --locked --all-extras --dev
uv run pre-commit install
```

`pytest-cov`, Ruff, ty, and pre-commit are locked development dependencies.
The root `justfile` provides the same command façade locally and in CI.

## Repository layout

- `src/osm_polygon_sentence_relevance/` — the package.
  - Root: `py.typed`, and thin compatibility facades
    (`constants.py`, `errors.py`, `schemas.py`, `settings.py`, `cli.py`,
    `pipeline.py`, `acquisition.py`, `discovery.py`, `loading.py`,
    `preprocessing.py`, `segmentation.py`, `sat_adapter.py`,
    `sentence_table.py`, `finalization.py`, `exporter.py`).
  - `contracts/` — canonical cross-cutting contracts: `constants.py`,
    `errors.py`, `schemas/` (`__init__.py`, `input.py`, `pipeline.py`,
    `registry.py`).
  - `application/` — `cli.py`, `pipeline.py`, `settings.py`.
  - `ingestion/` — `acquisition.py`, `discovery.py`, `loading.py`.
  - `sentences/` — `preprocessing.py`, `segmentation.py`, `sat.py`,
    `table.py`, `finalization.py`.
  - `joins/` — `_projection.py`, `_integrity.py`, `_wikipedia.py`,
    `_wikivoyage.py`, `_composition.py`, facade `__init__.py`.
  - `output/` — `exporter.py`, `atomic.py`, `checksum.py`, `manifest.py`,
    `validation.py`.
  - `publishing/` — `huggingface.py` (programmatic, one-commit publishing
    of a validated export to an existing Hub dataset; `PublicationResult`,
    `PublicationError`).
  - `operator/` — the Typer-based Mac-side Grid'5000 orchestration command,
    durable state, scheduling, transport, storage, and terminal presentation.
- `tests/` — mirrored `unit/`, `integration/`, `compatibility/`,
  `support/` layout. `tests/support/` holds the shared Arrow factories
  and fake-result builders.
- `docs/` — `index.md`, `architecture/`, `guides/`, `reference/`.
- `scripts/verify_distribution.py` — stdlib-only distribution check.
- Root governance: `README.md`, `LICENSE`, `CHANGELOG.md`,
  `CONTRIBUTING.md`, `SECURITY.md`, `MANIFEST.in`, `pyproject.toml`,
  `.python-version`, `.github/`.

## TDD workflow

Start with a failing test, then the minimum implementation. Structural
refactors are protected by characterization tests before any move. See
`CONTRIBUTING.md` for the full checklist.

## Verification gates

Run all of these before opening a pull request:

```bash
just check
just ci
```

`just check` verifies formatting, Ruff, ty, and the coverage-gated pytest
suite. `just ci` additionally builds and verifies the wheel and sdist.
Use the underlying `uv run ...` commands for focused diagnosis.

## Public API compatibility expectations

- `osm_polygon_sentence_relevance.cli.main(args=None, *, model_factory,
  acquisition_fn)` and its flags/exit codes are stable.
- `osm_polygon_sentence_relevance.pipeline.run_pipeline(...)` signature is
  stable.
- `osm_polygon_sentence_relevance.joins` re-exports the accepted public
  facade (`build_region_section_occurrences`, `join_wikipedia_sections`,
  `join_wikivoyage_sections`, `JoinReport`, `JoinedRegionSections`,
  projection-column tuples).
- `osm_polygon_sentence_relevance.exporter` exports `ExportResult` and
  `export_finalized_dataset(...)`.
- `osm_polygon_sentence_relevance.acquisition` exports `AcquisitionResult`
  and `acquire_dataset_snapshot(...)`.
- Legacy top-level module names remain as compatibility facades; import
  canonical domain paths in new production code.

## Updating dependencies

Edit `pyproject.toml`, then regenerate the lock:

```bash
uv lock
uv sync --locked --all-extras --dev
```

The committed `uv.lock` pins every transitive dependency, so a given
commit plus its `uv.lock` yields a reproducible environment.
