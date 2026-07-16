# ADR 0002: `contracts/` package for cross-cutting contracts

- Status: accepted
- Date: 2026-07-16
- Supersedes: (none) — extends [ADR 0001: Domain package layout](0001-domain-package-layout.md)

## Context

After ADR 0001, the cross-cutting contracts (`constants`, `errors`,
`schemas`, `settings`) still lived at the package root alongside the
compatibility facades. That kept the root flat and mixed three concerns:

1. **Implementation** (`contracts/*`) vs **compatibility facades**
   (`constants.py` at root re-exporting `contracts.constants`).
2. **Contract ownership** (constants/errors/schemas) vs **application
   settings** (data-directory precedence).
3. **Canonical paths** that production code should import from, distinct
   from the legacy top-level paths.

We needed a single, discoverable canonical home for the contracts so that
"import the contract from `contracts.<x>`" is the unambiguous, documented
default, while the legacy root modules remain as thin facades.

## Decision

Introduce a `contracts/` package under
`src/osm_polygon_sentence_relevance/`:

- `contracts/constants.py` — dataset IDs, pipeline version, allowed
  sources, schema names, allowed input paths.
- `contracts/errors.py` — the exception hierarchy (unchanged).
- `contracts/schemas/` — split by cohesion:
  - `__init__.py`: stable public re-exports only.
  - `input.py`: the six upstream input-table schemas.
  - `pipeline.py`: joined, segmented, and final-output schemas.
  - `registry.py`: `SCHEMA_REGISTRY` and `validate_table_schema`.
- Each schema object is instantiated exactly once; re-exports preserve
  object identity (`canonical.X is legacy.X`).

Move the real `PipelineSettings` implementation to
`application/settings.py` (canonical settings ownership). The legacy root
`settings.py` becomes a thin facade re-exporting `PipelineSettings`.

Remove the hard-coded external-drive data-directory path and all
machine-specific filesystem probing from settings resolution. The portable
data-directory precedence is now:

1. explicit `data_dir` argument;
2. nonblank `OSM_DATA_DIR` environment variable (whitespace-only ignored,
   leading `~` expanded);
3. `Path.cwd() / "data"`.

No directory creation, no network access, and no probing of personal or
platform-specific mount points.

Convert the root `constants.py`, `errors.py`, `schemas.py`, `settings.py`
(and the operational facades from ADR 0001) into uniform logic-free
facades containing only a docstring, explicit canonical imports, and an
accurate `__all__`. Production code imports only canonical paths; a static
AST check fails the build if any production module imports a root facade.

## Alternatives considered

- **Keep contracts at root.** Rejected: the root already held 15 facades;
  adding the real contract modules there kept the mix of
  implementation-vs-facade ambiguous and made the canonical path
  non-obvious.
- **`core/` instead of `contracts/`.** Rejected: `core` is vague and
  collides with common conventions; `contracts` names the actual
  responsibility (cross-cutting data contracts).
- **Dynamic facade via `sys.modules`/import hooks/deprectation shims.**
  Rejected: ADR 0001 already established explicit, lint-clean facades; the
  same policy is extended here.

## Consequences

- `contracts.constants`, `contracts.errors`, `contracts.schemas` are the
  canonical import paths; the root names are stable facades.
- Schema and exception objects are identical across canonical and legacy
  imports (identity-checked by characterization tests).
- Settings resolution is machine-portable and free of personal paths; the
  previously documented external-drive behavior is gone (changelog corrected).
- A structural test guarantees no production module imports a root facade
  and that every facade contains only a docstring, imports, and `__all__`.
