# ADR 0001: Compatibility-first domain package layout

- Status: accepted
- Date: 2026-07-16

## Context

The implementation had grown as a flat set of top-level modules
(`cli.py`, `pipeline.py`, `acquisition.py`, `discovery.py`, `loading.py`,
`preprocessing.py`, `segmentation.py`, `sat_adapter.py`,
`sentence_table.py`, `finalization.py`, `exporter.py`) plus an internal
`_export/` package and a `joins/` package. As behavior accumulated, the
flat layout no longer reflected the pipeline's domain boundaries, and
imports between legacy names obscured ownership.

We needed an organization that:

- keeps cross-cutting contracts (schemas, constants, settings, errors)
  central and stable;
- groups operational code by pipeline stage;
- preserves every existing public import path for downstream consumers
  and the test suite;
- requires no breaking migration.

## Decision

Adopt a domain-package layout under `src/osm_polygon_sentence_relevance/`:

- `application/` — CLI (`cli.py`) and pipeline orchestration
  (`pipeline.py`).
- `ingestion/` — acquisition, discovery, loading.
- `sentences/` — preprocessing, segmentation, SaT adapter, table,
  finalization.
- `joins/` — unchanged from before (projection, integrity, wikipedia,
  wikivoyage, composition).
- `output/` — exporter facade plus `atomic`, `checksum`, `manifest`
  (replacing the former `_export/`).
- Root: `constants.py`, `errors.py`, `schemas.py`, `settings.py`.

Each moved module keeps its canonical name inside its domain package.
Thin compatibility facades remain at every previous top-level path,
re-exporting the stable public symbols with a docstring and `__all__`,
and containing no logic, warnings, or import-time side effects.

## Alternatives considered

- **Breaking import migration.** Rename modules and update all importers
  (production, tests, docs). Rejected: it would break downstream consumers
  and the large existing test suite for no behavioral gain; the project is
  pre-1.0 but stability of import paths is a published expectation.
- **Leave everything flat.** Rejected: the flat layout no longer conveyed
  pipeline-stage ownership and made internal import direction ambiguous.
- **Excessive generic abstractions.** Introduce a generic "stage" or
  "processor" framework to unify ingestion/joins/sentences/output.
  Rejected: YAGNI; the stages have genuinely different contracts
  (especially the Wikipedia vs Wikivoyage join difference) and a generic
  framework would add indirection without removing real duplication.

## Consequences

- Production modules import each other via canonical domain paths; facades
  are never imported by other production modules.
- Public API and import paths are preserved; the test suite required only
  targeted mock-target updates where tests patched relocated symbols.
- New contributors learn ownership from the directory tree directly.
