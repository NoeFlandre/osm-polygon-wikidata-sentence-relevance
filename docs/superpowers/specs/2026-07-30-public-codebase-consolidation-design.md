# Public Codebase Consolidation Design

## Purpose

Consolidate the existing production pipeline into a smaller, easier-to-navigate
public codebase without adding features or changing runtime behavior. The work
replaces mypy with Astral `ty`, splits oversized modules by responsibility,
keeps supported imports and commands stable, and updates active documentation
to match the resulting architecture.

## Scope

This pass covers production Python under
`src/osm_polygon_sentence_relevance/`, retained operational Python under
`scripts/streaming/` and `scripts/*.py`, active CI and developer tooling, and
active public documentation.

It does not change:

- dataset schemas, normalization, sentence boundaries, prompts, or labels;
- checkpoint, publication, scheduling, quota, or security contracts;
- public command names, command arguments, or supported import paths;
- Grid'5000 jobs, Hugging Face datasets, local datasets, or remote storage;
- archived design specifications and implementation plans, which remain
  historical records of the tooling used when they were written.

## Quality Boundaries

### File size

Every maintained production Python file must contain at most 500 physical
lines. A repository contract test enforces the ceiling. Vendored data,
generated files, tests, and deliberately tiny compatibility facades are not
special-cased because the ceiling applies only to maintained production
Python.

The ceiling is a guardrail, not a reason to split cohesive code arbitrarily.
Modules are divided only at existing responsibility boundaries. Public
facades remain small and forward to internal packages.

### Compatibility

Existing public modules, imports, entry points, exception types, serialized
state, command construction, log text covered by tests, and deterministic
outputs remain stable. Compatibility tests exercise the public surface before
and after every extraction.

Private helpers may move. Their new locations use responsibility-oriented
subpackages rather than generic `utils` or `helpers` modules.

### Type checking

`ty` becomes the only repository type checker:

- add a pinned development dependency through `uv`;
- configure checked production roots in `[tool.ty]`;
- replace active mypy commands and CI steps with `uv run ty check`;
- remove mypy dependencies and configuration;
- fix every `ty` diagnostic with narrowing, protocols, typed mappings, or
  clearer interfaces;
- do not use project-wide ignored rules, blanket file suppressions, or
  automatic ignore insertion.

Third-party limitations may use the narrowest rule-specific suppression only
when no sound local type boundary is possible, and each such suppression must
have a regression test or explanatory comment.

## Target Architecture

### Operator

Keep `operator/cli.py`, `operator/state.py`, `operator/ssh.py`,
`operator/config.py`, and `operator/relay.py` as compatibility facades where
needed. Move cohesive internals into:

- `operator/_cli/`: argument construction, new-run orchestration, resume
  orchestration, and queued-start optimization;
- `operator/_state/`: models, JSON validation, secure filesystem I/O, and
  store transitions;
- `operator/_ssh/`: target/path validation, transport execution, and
  offset-based log reading;
- `operator/_config/`: immutable models, validation, and identity encoding;
- `operator/_relay/`: manifest validation, checkpoint inspection, and
  transport orchestration.

No module may reach through another internal package to mutate its private
state. Shared contracts live in the nearest public facade or a specifically
named contracts module.

### Application

Keep `application/pipeline.py` as the supported facade. Move internal work to
`application/_pipeline/`:

- planning and path validation;
- checkpoint reconciliation and shard execution;
- aggregation and final export.

The public `run_pipeline` signature and output remain unchanged.

### Output

Retain the existing `output/_card/` boundary and divide it further by rendered
section and factual-statistics responsibility. Separate plot data preparation
from PNG rendering. Split publication validation into filesystem, manifest,
and render-consistency checks while preserving the public validator.

### Labeling and streaming operations

Separate orchestration from record conversion and durable I/O:

- labeling finalization: accumulation, statistics, and publication assembly;
- streaming driver: CLI/configuration, shard loop, and scratch lifecycle;
- streaming offload: Hub protocol, remote validation, and local cache policy.

Operational modules remain sdist-only where currently required.

## Documentation

Each major production domain receives a concise architecture page under
`docs/architecture/packages/` describing:

- its responsibility;
- its public entry points;
- its internal subpackages;
- its dependencies and invariants;
- where its focused tests live.

`docs/architecture/overview.md`, `docs/index.md`, `CONTRIBUTING.md`, active
developer guides, and CI commands are updated. Module and package docstrings
remain the closest API documentation. Documentation avoids duplicated
implementation narratives and links to the canonical page instead.

## Test Strategy

Every extraction follows RED → GREEN:

1. add or strengthen a boundary/compatibility test that fails before the move;
2. record the expected failure;
3. move the smallest coherent responsibility;
4. pass focused tests and `ty`;
5. run related regression tests;
6. keep the complete repository suite green before continuing.

Required final gates:

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
uv build --no-build-isolation
uv run python scripts/verify_distribution.py dist/*.whl dist/*.tar.gz
bash -n scripts/grid5000/*.sh
git diff --check
```

The existing 95% branch-aware coverage threshold remains unchanged.

## Delivery

Implementation is committed in reviewable, responsibility-sized commits.
Before the final push:

- audit every changed path and staged mode;
- ensure no credentials, datasets, models, logs, caches, build products, or
  machine-specific paths are staged;
- require a clean working tree;
- push normally to `origin/main`, never force.
