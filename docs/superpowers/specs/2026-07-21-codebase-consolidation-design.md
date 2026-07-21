# Codebase Consolidation Design

## Objective

Turn the repository into a concise, public-facing implementation of one
product: a deterministic, resumable sentence-level dataset pipeline with a
bounded Grid'5000 production runner and validated Hugging Face publication.
The consolidation adds no product capability and changes no supported data,
CLI, Python API, checkpoint, or publication contract.

## Compatibility Boundary

The following behavior remains supported and regression-tested:

- the `osm-polygon-sentence-relevance` command and every documented option;
- canonical Python imports under the domain subpackages;
- documented root-level compatibility facades such as `constants`, `errors`,
  `schemas`, and `settings`;
- local input and read-only Hugging Face input acquisition;
- explicit CPU, CUDA, and MPS device selection where currently supported;
- deterministic joins, segmentation, normalization, finalization, export,
  checkpoints, dataset-card generation, validation, and publishing;
- the bounded, resumable Grid'5000 streaming production workflow;
- the current output schemas, manifest fields, public exception identities,
  serialized CLI output, and distribution boundaries.

Compatibility modules remain thin re-export surfaces. Canonical ownership
stays in the relevant domain package. No new deprecation framework is added.

## Removal Boundary

Delete material that exists only for past investigation or superseded
operations:

- GPU smoke submission, wrappers, payloads, cache-ref workarounds, and their
  dedicated tests;
- the obsolete full-snapshot Grid'5000 build launcher family superseded by
  bounded streaming;
- one-off join/upstream audit scripts and their remote runner;
- diagnostic-only artifact and metadata helpers not used by the streaming
  production path;
- historical phase/amendment narration, placeholder assertions, debugging
  scaffolding, personal-path guards that only protect deleted scripts, and
  stale operational examples;
- duplicated tests whose behavior is already asserted by a clearer canonical
  contract test.

Removal is dependency-driven. A helper is retained if the supported streaming
runner, package build, publication flow, or acceptance suite uses it.

## Target Repository Structure

### Python package

Keep the existing domain layout:

- `application/`: CLI orchestration, pipeline coordination, checkpointing;
- `contracts/`: schemas, constants, errors, exception formatting;
- `ingestion/`: acquisition, discovery, loading;
- `joins/`: Wikipedia/Wikivoyage composition and integrity;
- `sentences/`: preprocessing, segmentation, device placement, finalization;
- `output/`: export, manifest, dataset card, profile, plots, validation;
- `publishing/`: Hugging Face publication boundary.

Split a module only when it combines independently testable responsibilities
or cannot be reviewed comfortably. Preserve its existing import surface by
re-exporting the same public names. The intended focused extractions are:

- checkpoint lock/path safety, run inventory, checkpoint storage, and recovery;
- dataset-card data models/statistics, prose rendering, plot rendering, and
  publication validation.

No abstraction is introduced solely to reduce line count.

### Operational scripts

Retain only:

- the bounded streaming driver package;
- the Grid'5000 streaming submit/job/payload scripts;
- the streaming finalization submit/job/payload scripts and the persistence
  helper they require;
- GPU preflight used by the production runner;
- deterministic publication rendering;
- distribution verification.

Operational modules remain sdist-only. They must not leak into the wheel.

### Tests

Organize tests by stable component and contract, not implementation phase.
Characterization tests protect the current surface before files move. Remove
tests only when a retained test proves the same behavior more directly.

Large amendment-era files are split by subject where doing so improves test
discovery and review. Test helpers belong in `tests/support/`; production code
must not gain test-only hooks.

### Documentation

The public documentation set is:

- root `README.md`: purpose, status, install, minimal usage, outputs, links;
- `docs/index.md`: short navigation page;
- architecture overview and durable ADRs;
- getting-started, reproducibility, development, and production Grid'5000
  guides;
- CLI, API, and data-contract references;
- contribution, security, license, and changelog documents.

Rewrite the Grid'5000 guide around the single production streaming workflow.
Remove phase numbers and incident history from maintained documentation.
Condense `[Unreleased]` changelog material into user-facing Added, Changed,
Fixed, and Removed entries; preserve genuinely released history if present.
Examples use placeholders, immutable revisions, and current commands only.

## Refactoring Method

Every behavior-affecting cleanup follows strict RED-GREEN-REFACTOR:

1. add or identify a focused characterization/contract test;
2. demonstrate that the new structural or hygiene requirement fails;
3. make the smallest production or repository change;
4. demonstrate the focused test passes;
5. run the affected subsystem before proceeding.

Pure file moves use import/contract tests as the red boundary: tests first
express the target module ownership while preserving the old import path.
Deletions use inventory tests that fail while obsolete public/distribution
entries remain. Documentation cleanup uses link, command-parity, and
forbidden-stale-language tests rather than exact prose snapshots.

## Quality Rules

- No new feature flags, compatibility frameworks, or generalized plugin
  systems.
- No debug prints, `pdb`, breakpoints, temporary markers, placeholder tests,
  personal absolute paths, concrete job IDs, secrets, credentials, data files,
  model weights, caches, or generated artifacts.
- No phase/amendment labels in maintained source, tests, or current docs.
- Errors crossing a public boundary remain typed, stable, and path-safe where
  already guaranteed.
- Public functions and modules have concise responsibility-focused docstrings.
- Optional heavy dependencies remain lazily imported.
- Production runners never fall back from CUDA to local Mac, frontend, CPU,
  MPS, or automatic device selection.
- Published statistics and cards continue to derive from artifact bytes.

## Acceptance Criteria

The consolidation is complete only when all of the following are freshly
verified:

1. baseline public import, callable-signature, error-identity, CLI-help, and
   non-publishing JSON compatibility tests pass;
2. all supported streaming submit/job/payload shell contracts pass;
3. all deterministic pipeline, checkpoint resume, publication, schema, hash,
   dataset-card, and Viewer-compatibility tests pass;
4. the complete test suite passes with branch coverage at or above 95%;
5. Ruff format and lint pass without new suppressions;
6. mypy passes for `src` and the retained operational Python package;
7. wheel and sdist build, validate, and contain exactly their intended
   surfaces;
8. isolated-wheel imports and CLI help work without optional dependencies;
9. every retained shell script passes `bash -n`;
10. all maintained Markdown links resolve and documented CLI flags match the
    parser;
11. repository scans find no obsolete script names, phase/amendment language,
    debug hooks, personal paths, tracked data, or generated artifacts;
12. `git diff --check` passes and the worktree is clean after generated build
    outputs are removed.

External dependency warnings are either eliminated by a compatible dependency
choice or explicitly filtered by a narrow, documented test configuration when
they originate wholly outside this project.

## Delivery Strategy

Implement in independently reviewable commits:

1. characterize the protected public and production surfaces;
2. remove superseded operational tooling and update distribution contracts;
3. consolidate production modules behind unchanged imports;
4. consolidate tests and repository hygiene;
5. rewrite maintained documentation and changelog;
6. run the complete release gate and clean generated artifacts.

The branch is not merged or pushed to `main` until the full release gate is
green and the final diff contains only consolidation work.
