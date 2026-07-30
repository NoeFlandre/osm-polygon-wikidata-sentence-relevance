# Operator CLI and Developer Tooling Design

**Date:** 2026-07-30
**Status:** Proposed
**Scope:** Public Grid'5000 operator CLI and repository developer workflow

## Goal

Make the repository intentionally use:

- `uv` for environments, dependency locking, execution, and builds;
- Ruff for formatting and linting;
- `ty` for static type checking;
- pytest for tests and branch coverage;
- pre-commit for fast local verification;
- Typer for the public `osm-polygon-grid5000` command;
- Rich for readable interactive terminal output;
- tqdm for bounded live progress when a terminal is interactive;
- `just` as the documented command façade;
- GitHub Actions for locked, reproducible continuous integration.

The change must preserve the production pipeline and every existing operator
contract. It is a tooling and presentation improvement, not a workflow rewrite.

## Current Contract

The installed command is:

```text
osm-polygon-grid5000
```

It exposes four subcommands:

- `run`
- `resume`
- `status`
- `cleanup`

The migration must preserve:

- every current option name, positional argument, default, repeatable option,
  choice, and required/optional distinction;
- `main(argv: list[str] | None = None) -> int` for direct and installed use;
- exit `0` on success, `1` on handled operational failure, and `130` on local
  interruption;
- the guarantee that Ctrl+C stops only local monitoring and prints the durable
  resume command when a run identity exists;
- durable state, scheduler, checkpoint, storage, publication, and cleanup
  semantics;
- machine-readable pretty JSON on standard output from `status`;
- stable, non-ANSI output when redirected, captured by tests, or when
  `NO_COLOR` is set;
- current stdout/stderr separation: normal status on stdout and errors or
  interruption guidance on stderr.

The other installed commands remain unchanged:

- `osm-polygon-sentence-relevance`
- `osm-polygon-label-sentences`

## Non-goals

- No new pipeline feature, scheduler policy, model behavior, or data format.
- No SSH, OAR, GPU, Hugging Face, inference, or publication operation during
  implementation or verification.
- No migration of the build CLI or labeling CLI.
- No compatibility alias, duplicate parser, or indefinite argparse/Typer
  coexistence.
- No progress animation in logs, pipes, redirected output, or tests.
- No broad reformat or unrelated refactor.

## Architecture

### Typer command boundary

`operator/cli.py` will retain orchestration helpers and the public `main`
entrypoint. A small Typer application will own only parsing, help, command
dispatch, and conversion to the existing handler inputs.

The command functions will call the existing `_run`, `_resume_handler`,
`_status`, and `_cleanup` workflows. They will not duplicate their business
logic. Once the Typer path is covered, `build_parser` and argparse-only tests
will be removed rather than retained as a second source of truth.

`main(argv)` will invoke the Typer application programmatically without
letting Click's standalone runner hide the repository's established return
codes. It will preserve the current top-level handling for `KeyboardInterrupt`
and operational exceptions.

### Terminal presentation

A focused `operator/console.py` module will own terminal behavior:

- concise `[operator]` milestones;
- job log lines;
- errors and interruption guidance;
- interactive progress construction;
- terminal capability and `NO_COLOR` decisions.

Rich will render human-facing milestones and errors only when appropriate.
Plain output will remain the compatibility baseline. Rich markup must never
interpret remote log content.

tqdm will be used only for progress with a real total and current count, such
as labeling checkpoint progress. It will be disabled when stdout is not a TTY,
when output is captured, or when `NO_COLOR` is set. Unknown-duration scheduler
waits will remain deduplicated textual status updates rather than fake progress
bars.

The presentation layer will accept injected streams and terminal-capability
decisions so tests do not patch global terminal state.

### Machine output

`status` is a machine-readable command. Its JSON bytes, indentation, key
ordering, output stream, and exit behavior will remain plain and stable.
Rich and tqdm must not write around this payload.

### Dependencies

Typer, Rich, and tqdm will become direct runtime dependencies because the
installed operator command imports them.

pre-commit will become a direct development dependency. Ruff, ty, pytest, and
pytest-cov remain direct development dependencies.

`just` is an external command runner, not a Python runtime dependency. The
repository will provide and document a root `justfile`; CI will install a
pinned `just` release before invoking its recipes.

All dependency changes will be made through `uv`, and `uv.lock` will be
updated and verified with `uv sync --locked --all-extras --dev`.

## Developer Command Façade

The root `justfile` will provide small, composable recipes:

```text
sync
format
format-check
lint
typecheck
test
check
build
verify-dist
ci
```

Semantics:

- `sync`: locked synchronization of all extras and development dependencies;
- `format`: apply Ruff formatting;
- `format-check`: verify formatting without mutation;
- `lint`: Ruff lint;
- `typecheck`: `ty check`;
- `test`: full pytest suite with configured coverage;
- `check`: format-check, lint, typecheck, and test;
- `build`: build sdist and wheel with `uv`;
- `verify-dist`: run the existing distribution verifier;
- `ci`: `check`, `build`, and `verify-dist`.

Recipes will use `uv run` and remain thin. They will not reimplement project
configuration or add shell-specific hidden behavior.

## Pre-commit

The root `.pre-commit-config.yaml` will use repository-local hooks that invoke
the same project commands:

- Ruff format check;
- Ruff lint;
- `ty check`;
- a focused fast test set covering the public CLI and tooling contracts.

The full test suite remains a `just test` and CI responsibility. This keeps
commits responsive while CI remains authoritative. Hooks must not download
their own alternate Ruff, ty, or pytest versions; the locked `uv` environment
is the single tool source.

## GitHub Actions

The existing workflow will continue to:

- use pinned action SHAs;
- install `uv`;
- install Python 3.12;
- sync the locked environment;
- run all quality, test, build, distribution, and installed-wheel checks.

After installing a pinned `just`, CI will call the same `just` recipes used
locally instead of maintaining duplicate command sequences. The installed
wheel smoke test will additionally exercise:

```text
osm-polygon-grid5000 --help
```

No secret, remote credential, SSH, OAR, or Hugging Face access is required.

## Documentation

`CONTRIBUTING.md` will make `just` the primary contributor interface and list
the underlying `uv` commands for transparency. The README will mention the
operator command without turning into a developer-tool manual.

Documentation will state:

- `uv sync --locked --all-extras --dev` prepares the project;
- `just check` runs the local acceptance gate;
- `just ci` mirrors continuous integration;
- pre-commit installation uses the locked environment;
- `ty`, not mypy, is the supported type checker.

References to direct commands that remain useful for diagnosis are retained,
but contradictory or duplicate workflows are removed.

## RED-to-GREEN Verification

Tests will be written or tightened before production changes.

### CLI compatibility tests

- exact command set and public option names;
- defaults and repeatable `--site`;
- invalid choices and missing arguments;
- `main(argv)` return/exit behavior;
- help succeeds and names all four commands;
- Ctrl+C preserves the remote job and prints the exact resume command;
- handled exceptions produce a plain `Error: ...` on stderr and return `1`;
- `status` emits only stable JSON;
- cleanup dry-run and execute output remain distinguishable.

### Presentation tests

- interactive mode enables Rich styling and tqdm only for measurable progress;
- redirected, captured, and `NO_COLOR` modes emit no ANSI/control sequences;
- remote log text is rendered literally, never as Rich markup;
- repeated scheduler status remains deduplicated;
- progress completion does not corrupt subsequent lines;
- stdout and stderr contracts are preserved.

### Tooling tests

- required direct dependencies are declared in the appropriate dependency
  group;
- `just --list` exposes the documented recipes;
- pre-commit configuration invokes the locked project tools;
- GitHub Actions delegates to the documented `just` recipes;
- wheel installation exposes `osm-polygon-grid5000 --help`.

## Acceptance Gates

The implementation is complete only when all of these pass:

```bash
uv sync --locked --all-extras --dev
just format-check
just lint
just typecheck
just test
just build
just verify-dist
uv run pre-commit run --all-files
git diff --check
```

Additionally:

- focused new/modified modules must meet at least 95% branch coverage;
- the full repository must keep its configured 95% coverage threshold;
- no `mypy` configuration, dependency, command, or documentation reference may
  be introduced;
- no generated distributions, caches, logs, credentials, runtime data, or
  unrelated files may be committed;
- the final working tree must be clean after an intentional commit and normal
  push to `main`.

## Risks and Controls

### Typer changes help formatting

Help text formatting may differ, but the public command and option contract
must not. Tests will assert semantic help content and parsing behavior rather
than fragile terminal widths.

### Rich or tqdm pollutes automation output

Plain output is the default for non-interactive streams and `NO_COLOR`.
Machine JSON bypasses presentation entirely.

### CLI migration accidentally rewrites orchestration

Command functions remain thin adapters to existing workflows. Orchestration
tests continue to exercise the genuine call path.

### Local and CI commands drift

Both pre-commit and GitHub Actions delegate to the locked `uv` tools, while
GitHub Actions uses the same `just` recipes documented for contributors.
