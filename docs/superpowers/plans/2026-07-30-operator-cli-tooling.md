# Operator CLI and Developer Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Grid'5000 operator's argparse boundary with a compatible Typer interface, add terminal-safe Rich/tqdm presentation, and unify local and CI quality commands through uv, pre-commit, and just.

**Architecture:** Keep all pipeline orchestration in its current functions and introduce two narrow boundaries: `operator/console.py` for terminal presentation and Typer command adapters in `operator/cli.py` for parsing and dispatch. Make `just` the command façade while every recipe delegates to locked `uv` tools; pre-commit and GitHub Actions reuse those same contracts.

**Tech Stack:** Python 3.12, uv, Typer, Click, Rich, tqdm, Ruff, ty, pytest/pytest-cov, pre-commit, just, GitHub Actions.

---

## File Map

### Create

- `src/osm_polygon_sentence_relevance/operator/console.py` — terminal capability detection, plain/Rich messages, literal remote log lines, and optional tqdm progress.
- `tests/unit/operator/test_console.py` — isolated output and progress contracts.
- `tests/unit/test_developer_tooling.py` — declarative contracts for dependencies, just, pre-commit, and CI delegation.
- `.pre-commit-config.yaml` — locked local quality hooks.
- `justfile` — contributor and CI command façade.

### Modify

- `src/osm_polygon_sentence_relevance/operator/cli.py` — replace argparse parsing with Typer command adapters and delegate output to `console.py`.
- `tests/unit/operator/test_cli.py` — characterize and then test the genuine Typer boundary.
- `tests/unit/operator/test_resume_recovery.py` — replace direct argparse namespace construction with explicit test namespaces or genuine command invocation.
- `tests/unit/operator/test_resume_lifecycle.py` — replace direct parser use while preserving workflow assertions.
- `tests/unit/operator/test_earliest_start_cli.py` — preserve output and exit contracts through the Typer path.
- `pyproject.toml` — direct runtime and development dependencies.
- `uv.lock` — locked dependency graph.
- `.github/workflows/ci.yml` — install pinned just and invoke shared recipes.
- `CONTRIBUTING.md` — document the single supported workflow.
- `README.md` — concise operator and contributor entrypoints.
- `CHANGELOG.md` — public tooling and CLI presentation note without claiming pipeline changes.

## Task 1: Characterize the Existing Public CLI

**Files:**
- Modify: `tests/unit/operator/test_cli.py`

- [ ] **Step 1: Add failing or characterization tests for the exact parser contract**

Add a table-driven test that records all commands, arguments, defaults, and
repeatable-site behavior before changing production code:

```python
@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["status", "a" * 20],
            {"command": "status", "run_id": "a" * 20},
        ),
        (
            ["resume", "b" * 20],
            {
                "command": "resume",
                "run_id": "b" * 20,
                "gpu_memory_mb": 40_000,
                "poll_seconds": 30.0,
            },
        ),
        (
            [
                "run",
                "--scope",
                "region",
                "--region",
                "afghanistan-latest",
                "--stage",
                "label",
            ],
            {
                "command": "run",
                "scope": "region",
                "region": "afghanistan-latest",
                "stage": "label",
                "batch_size": 128,
                "row_limit": 0,
                "llama_parallel": 8,
                "llama_per_slot_context": 8192,
                "request_concurrency": None,
                "gpu_memory_mb": 40_000,
                "remote_free_bytes": 8 * 1024**3,
                "poll_seconds": 30.0,
            },
        ),
    ],
)
def test_public_parser_defaults(
    argv: list[str],
    expected: dict[str, object],
) -> None:
    args = cli.build_parser().parse_args(argv)
    for key, value in expected.items():
        assert getattr(args, key) == value
```

Add a dedicated test that determines whether explicit repeated `--site`
values replace or extend `DEFAULT_SITES`:

```python
def test_explicit_sites_have_characterized_argparse_semantics() -> None:
    args = cli.build_parser().parse_args(
        [
            "resume",
            "a" * 20,
            "--site",
            "nancy",
            "--site",
            "nantes",
        ]
    )
    assert args.site == ["nancy", "nantes"]
```

Add tests for the existing help, invalid choice, missing required option,
handled error, and Ctrl+C exit behavior. Use exact exit codes and essential
stream assertions; do not snapshot terminal widths.

- [ ] **Step 2: Run the characterization tests**

Run:

```bash
uv run pytest -o addopts='' tests/unit/operator/test_cli.py -q
```

Expected: PASS. If explicit `--site` differs from the asserted list, update
only that expected value to the observed current behavior and document it in
the test name before proceeding.

- [ ] **Step 3: Commit the characterization boundary**

```bash
git add tests/unit/operator/test_cli.py
git commit -m "Characterize operator CLI contracts"
```

## Task 2: Declare and Lock the Toolchain

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `tests/unit/test_developer_tooling.py`

- [ ] **Step 1: Write a failing dependency contract test**

Create `tests/unit/test_developer_tooling.py`:

```python
from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _names(requirements: list[str]) -> set[str]:
    return {
        requirement.split("[", 1)[0]
        .split("<", 1)[0]
        .split(">", 1)[0]
        .split("=", 1)[0]
        .strip()
        .lower()
        for requirement in requirements
    }


def test_required_runtime_and_development_tools_are_direct() -> None:
    project = _pyproject()
    runtime = _names(project["project"]["dependencies"])
    development = _names(project["dependency-groups"]["dev"])
    assert {"typer", "rich", "tqdm"} <= runtime
    assert {"pytest", "pytest-cov", "ruff", "ty", "pre-commit"} <= development
    assert "mypy" not in runtime | development
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest -o addopts='' tests/unit/test_developer_tooling.py -q
```

Expected: FAIL because Typer, Rich, tqdm, and pre-commit are not direct
dependencies.

- [ ] **Step 3: Add bounded direct dependencies**

Modify `pyproject.toml`:

```toml
dependencies = [
    "pyarrow",
    "rich>=15,<16",
    "tqdm>=4.68,<5",
    "typer>=0.26,<0.27",
]

[dependency-groups]
dev = [
    "pre-commit>=4,<5",
    "pytest",
    "pytest-cov",
    "ruff>=0.14,<0.15",
    "ty>=0.0.65,<0.1",
]
```

Use the versions already present transitively in `uv.lock` where possible.

- [ ] **Step 4: Update and verify the lock**

Run:

```bash
uv lock
uv sync --locked --all-extras --dev
uv run pytest -o addopts='' tests/unit/test_developer_tooling.py -q
```

Expected: lock and sync succeed; the test passes.

- [ ] **Step 5: Commit the locked dependency contract**

```bash
git add pyproject.toml uv.lock tests/unit/test_developer_tooling.py
git commit -m "Declare operator and developer tool dependencies"
```

## Task 3: Add a Terminal-Safe Presentation Boundary

**Files:**
- Create: `src/osm_polygon_sentence_relevance/operator/console.py`
- Create: `tests/unit/operator/test_console.py`

- [ ] **Step 1: Write RED tests for plain and interactive output**

Create tests using `io.StringIO`, not the process terminal:

```python
from __future__ import annotations

from io import StringIO

from osm_polygon_sentence_relevance.operator.console import OperatorConsole


def test_noninteractive_output_is_plain_and_stable() -> None:
    stdout = StringIO()
    stderr = StringIO()
    console = OperatorConsole(stdout=stdout, stderr=stderr, interactive=False)
    console.milestone("Preparing [remote] checkout")
    console.error("failed")
    console.job_lines(42, "[red]literal[/red]\nsecond")
    assert stdout.getvalue() == (
        "[operator] Preparing [remote] checkout\n"
        "[job 42] [red]literal[/red]\n"
        "[job 42] second\n"
    )
    assert stderr.getvalue() == "Error: failed\n"
    assert "\x1b[" not in stdout.getvalue() + stderr.getvalue()


def test_no_color_forces_plain_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    stream = StringIO()
    console = OperatorConsole(stdout=stream, stderr=StringIO(), interactive=True)
    console.milestone("ready")
    assert stream.getvalue() == "[operator] ready\n"
    assert "\x1b[" not in stream.getvalue()
```

Add tests for:

- interruption text and exact resume command on stderr;
- literal remote markup;
- JSON written without Rich;
- progress disabled without a TTY;
- progress advances through an exact bounded total only in interactive mode;
- closing progress leaves the next message on a clean line.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest -o addopts='' tests/unit/operator/test_console.py -q
```

Expected: collection FAIL because `operator.console` does not exist.

- [ ] **Step 3: Implement the minimal presentation API**

Create `operator/console.py` with this public shape:

```python
from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import IO, Any, Self

from rich.console import Console
from tqdm import tqdm


@dataclass(slots=True)
class ProgressView:
    _bar: tqdm[Any] | None

    def __enter__(self) -> Self:
        return self

    def advance(self) -> None:
        if self._bar is not None:
            self._bar.update(1)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class OperatorConsole:
    def __init__(
        self,
        *,
        stdout: IO[str] = sys.stdout,
        stderr: IO[str] = sys.stderr,
        interactive: bool | None = None,
    ) -> None:
        enabled = (
            stdout.isatty() if interactive is None else interactive
        ) and "NO_COLOR" not in os.environ
        self._stdout = stdout
        self._stderr = stderr
        self._interactive = enabled
        self._rich_out = Console(
            file=stdout,
            force_terminal=enabled,
            no_color=not enabled,
            markup=enabled,
            highlight=False,
            soft_wrap=True,
        )
        self._rich_err = Console(
            file=stderr,
            force_terminal=enabled,
            no_color=not enabled,
            markup=enabled,
            highlight=False,
            soft_wrap=True,
        )

    def milestone(self, message: str) -> None:
        if self._interactive:
            self._rich_out.print("[bold cyan][operator][/bold cyan]", message)
        else:
            print(f"[operator] {message}", file=self._stdout, flush=True)

    def error(self, message: str) -> None:
        if self._interactive:
            self._rich_err.print("[bold red]Error:[/bold red]", message)
        else:
            print(f"Error: {message}", file=self._stderr, flush=True)

    def plain(self, message: str, *, error: bool = False) -> None:
        print(
            message,
            file=self._stderr if error else self._stdout,
            flush=True,
        )

    def job_lines(self, job_id: int, text: str) -> None:
        for line in text.splitlines():
            print(f"[job {job_id}] {line}", file=self._stdout, flush=True)

    def json(self, payload: Mapping[str, object]) -> None:
        print(
            json.dumps(payload, indent=2, sort_keys=True),
            file=self._stdout,
            flush=True,
        )

    def progress(self, *, description: str, total: int) -> ProgressView:
        bar = (
            tqdm(
                total=total,
                desc=description,
                file=self._stdout,
                disable=not self._interactive,
                dynamic_ncols=True,
                unit="rows",
            )
            if self._interactive
            else None
        )
        return ProgressView(bar)


__all__ = ["OperatorConsole", "ProgressView"]
```

During implementation, remove unused imports from this skeleton. Do not pass
remote log text through Rich markup.

- [ ] **Step 4: Run focused tests and coverage**

```bash
uv run pytest -o addopts='' tests/unit/operator/test_console.py -q \
  --cov=osm_polygon_sentence_relevance.operator.console \
  --cov-branch --cov-report=term-missing --cov-fail-under=95
```

Expected: PASS with at least 95% branch coverage.

- [ ] **Step 5: Commit the presentation boundary**

```bash
git add src/osm_polygon_sentence_relevance/operator/console.py \
  tests/unit/operator/test_console.py
git commit -m "Add terminal-safe operator presentation"
```

## Task 4: Replace Argparse with the Typer Command Boundary

**Files:**
- Modify: `src/osm_polygon_sentence_relevance/operator/cli.py`
- Modify: `tests/unit/operator/test_cli.py`
- Modify: `tests/unit/operator/test_resume_recovery.py`
- Modify: `tests/unit/operator/test_resume_lifecycle.py`
- Modify: `tests/unit/operator/test_earliest_start_cli.py`

- [ ] **Step 1: Add RED Typer boundary tests**

Use `typer.testing.CliRunner` for parsing/help tests:

```python
from typer.testing import CliRunner


runner = CliRunner()


def test_typer_help_exposes_exact_command_set() -> None:
    result = runner.invoke(cli.app, ["--help"], color=False)
    assert result.exit_code == 0
    for command in ("run", "resume", "status", "cleanup"):
        assert command in result.stdout


def test_status_command_delegates_exact_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(cli, "_status_run_id", lambda run_id: seen.append(run_id) or 0)
    result = runner.invoke(cli.app, ["status", "a" * 20], color=False)
    assert result.exit_code == 0
    assert seen == ["a" * 20]
```

Add command-level tests for:

- all current run defaults;
- explicit repeated sites;
- all `Scope` and `Stage` choices;
- resume defaults;
- cleanup dry-run and `--execute`;
- invalid choice and missing option exit `2`;
- no ANSI when `color=False` or `NO_COLOR=1`.

Expected RED: `cli.app` does not exist.

- [ ] **Step 2: Introduce typed Typer commands as thin adapters**

In `operator/cli.py`, remove the argparse import and add:

```python
from types import SimpleNamespace
from typing import Annotated

import click
import typer

from osm_polygon_sentence_relevance.operator.console import OperatorConsole


app = typer.Typer(
    name="osm-polygon-grid5000",
    help="Run and resume sentence processing on Grid'5000.",
    add_completion=False,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)
_CONSOLE = OperatorConsole()


@app.command("run")
def run_command(
    scope: Annotated[Scope, typer.Option("--scope")],
    stage: Annotated[Stage, typer.Option("--stage")],
    region: Annotated[str | None, typer.Option("--region")] = None,
    input_revision: Annotated[
        str | None, typer.Option("--input-revision")
    ] = None,
    site: Annotated[
        list[str] | None, typer.Option("--site")
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 128,
    row_limit: Annotated[int, typer.Option("--row-limit")] = 0,
    llama_parallel: Annotated[int, typer.Option("--llama-parallel")] = 8,
    llama_per_slot_context: Annotated[
        int, typer.Option("--llama-per-slot-context")
    ] = 8192,
    request_concurrency: Annotated[
        int | None, typer.Option("--request-concurrency")
    ] = None,
    gpu_memory_mb: Annotated[int, typer.Option("--gpu-memory-mb")] = 40_000,
    remote_free_bytes: Annotated[
        int, typer.Option("--remote-free-bytes")
    ] = 8 * 1024**3,
    poll_seconds: Annotated[float, typer.Option("--poll-seconds")] = 30.0,
) -> int:
    args = SimpleNamespace(
        command="run",
        scope=scope.value,
        stage=stage.value,
        region=region,
        input_revision=input_revision,
        site=list(site) if site is not None else list(DEFAULT_SITES),
        batch_size=batch_size,
        row_limit=row_limit,
        llama_parallel=llama_parallel,
        llama_per_slot_context=llama_per_slot_context,
        request_concurrency=request_concurrency,
        gpu_memory_mb=gpu_memory_mb,
        remote_free_bytes=remote_free_bytes,
        poll_seconds=poll_seconds,
    )
    return _run(args)
```

Implement `resume`, `status`, and `cleanup` similarly. Use `SimpleNamespace`
only at this compatibility boundary; do not spread it into domain modules.

If Task 1 proves that explicit sites extend defaults, implement that exact
behavior rather than the replacement shown above.

- [ ] **Step 3: Implement the stable `main(argv)` wrapper**

Use the Click command generated from Typer, while preserving repository exit
handling:

```python
def main(argv: list[str] | None = None) -> int:
    global _ACTIVE_RUN_ID
    prior_active = _ACTIVE_RUN_ID
    _ACTIVE_RUN_ID = None
    command = typer.main.get_command(app)
    try:
        result = command.main(
            args=argv,
            prog_name="osm-polygon-grid5000",
            standalone_mode=False,
        )
        return int(result) if result is not None else 0
    except click.UsageError as exc:
        exc.show(file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    except click.exceptions.Exit as exc:
        raise SystemExit(exc.exit_code) from exc
    except KeyboardInterrupt:
        run_id = _ACTIVE_RUN_ID or prior_active
        _CONSOLE.plain(
            "Local monitoring stopped; the remote job and checkpoints were "
            "preserved.",
            error=True,
        )
        if run_id is not None:
            _CONSOLE.plain(
                f"Resume with: {_resume_command(run_id)}",
                error=True,
            )
        raise SystemExit(130) from None
    except Exception as exc:
        _CONSOLE.error(str(exc))
        return 1
    finally:
        _ACTIVE_RUN_ID = prior_active
```

Verify the exact behavior of Typer's help exit under
`standalone_mode=False`. If it returns `0`, keep that behavior in `main` and
test it explicitly; do not add a fake exception solely to imitate argparse
internals.

- [ ] **Step 4: Remove argparse and parser-only compatibility**

Delete:

- `build_parser`;
- argparse imports;
- parser handler injection;
- `__all__` references to `build_parser`.

Export:

```python
__all__ = ["app", "main"]
```

Replace workflow tests that used `build_parser().parse_args(...)` with a
local explicit namespace factory:

```python
def _resume_args(
    run_id: str,
    *,
    sites: list[str] | None = None,
    gpu_memory_mb: int = 40_000,
    poll_seconds: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        command="resume",
        run_id=run_id,
        site=list(DEFAULT_SITES) if sites is None else sites,
        gpu_memory_mb=gpu_memory_mb,
        poll_seconds=poll_seconds,
    )
```

Keep parser behavior tests in `test_cli.py` on `CliRunner`; keep orchestration
tests focused on orchestration inputs.

- [ ] **Step 5: Route existing output through `OperatorConsole`**

Replace `_milestone` and `_emit` bodies:

```python
def _milestone(message: str) -> None:
    _CONSOLE.milestone(message)


def _emit(progress: LiveProgress) -> None:
    _CONSOLE.job_lines(progress.job_id, progress.text)
```

Replace top-level errors, interruption guidance, status JSON, and cleanup
output with the corresponding console methods. Do not bulk-replace output in
other modules in this task; those modules retain their tested behavior.

- [ ] **Step 6: Use tqdm for the bounded site-probe loop**

Materialize the deduplicated targets once and wrap both the initial probe and
re-probe loops:

```python
targets = tuple(dict.fromkeys(args.site))
probes: list[SiteProbe] = []
with _CONSOLE.progress(
    description="Probing Grid'5000 sites",
    total=len(targets),
) as progress:
    for target in targets:
        _milestone(f"Probing Grid'5000 site: {target}")
        probe = probe_site(target, config.run_id, requirements)
        probes.append(probe)
        if probe.reachable:
            _milestone(
                f"Site {probe.name}: reachable, GPU {probe.gpu_memory_mb} MiB, "
                f"persistent free {probe.persistent_free_bytes // 1024**3} GiB"
            )
        else:
            _milestone(f"Site {target}: unavailable")
        progress.advance()
```

Use description `"Re-probing Grid'5000 sites"` for the cleanup retry. In
plain/non-interactive mode, preserve every existing milestone line and emit no
progress control characters. Do not fabricate a sentence-labeling progress
bar from unstructured remote log text.

- [ ] **Step 7: Run focused CLI and workflow tests**

```bash
uv run pytest -o addopts='' -q \
  tests/unit/operator/test_console.py \
  tests/unit/operator/test_cli.py \
  tests/unit/operator/test_earliest_start_cli.py \
  tests/unit/operator/test_resume_recovery.py \
  tests/unit/operator/test_resume_lifecycle.py
```

Expected: PASS with no network or remote execution.

- [ ] **Step 8: Run module coverage**

```bash
uv run pytest -o addopts='' -q \
  tests/unit/operator/test_console.py \
  tests/unit/operator/test_cli.py \
  --cov=osm_polygon_sentence_relevance.operator.console \
  --cov=osm_polygon_sentence_relevance.operator.cli \
  --cov-branch --cov-report=term-missing --cov-fail-under=95
```

Expected: combined touched-module coverage at least 95%. Add focused tests for
actual missing branches; do not add coverage exclusions.

- [ ] **Step 9: Commit the CLI migration**

```bash
git add \
  src/osm_polygon_sentence_relevance/operator/cli.py \
  tests/unit/operator/test_cli.py \
  tests/unit/operator/test_earliest_start_cli.py \
  tests/unit/operator/test_resume_recovery.py \
  tests/unit/operator/test_resume_lifecycle.py
git commit -m "Migrate Grid5000 operator CLI to Typer"
```

## Task 5: Add just and Pre-commit as Shared Local Contracts

**Files:**
- Create: `justfile`
- Create: `.pre-commit-config.yaml`
- Modify: `tests/unit/test_developer_tooling.py`

- [ ] **Step 1: Write RED configuration tests**

Extend `test_developer_tooling.py`:

```python
def test_justfile_exposes_required_recipes() -> None:
    text = (ROOT / "justfile").read_text(encoding="utf-8")
    for recipe in (
        "sync:",
        "format:",
        "format-check:",
        "lint:",
        "typecheck:",
        "test:",
        "check:",
        "build:",
        "verify-dist:",
        "ci:",
    ):
        assert recipe in text
    assert "mypy" not in text


def test_precommit_uses_locked_project_commands() -> None:
    text = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    for entry in (
        "entry: uv run ruff format --check .",
        "entry: uv run ruff check .",
        "entry: uv run ty check",
        "tests/unit/operator/test_console.py",
        "tests/unit/operator/test_cli.py",
        "tests/unit/test_developer_tooling.py",
    ):
        assert entry in text
    assert text.count("language: system") == 4
    assert text.count("pass_filenames: false") == 4
```

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -o addopts='' tests/unit/test_developer_tooling.py -q
```

Expected: FAIL because `justfile` and `.pre-commit-config.yaml` do not exist.

- [ ] **Step 3: Create the root `justfile`**

```make
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

sync:
    uv sync --locked --all-extras --dev

format:
    uv run ruff format .

format-check:
    uv run ruff format --check .

lint:
    uv run ruff check .

typecheck:
    uv run ty check

test:
    uv run pytest -q

check: format-check lint typecheck test

build:
    uv build

verify-dist: build
    uv run python scripts/verify_distribution.py dist/*.whl dist/*.tar.gz

ci: check verify-dist
```

Do not make `check` mutate source. Keep `format` explicit.

- [ ] **Step 4: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: local
    hooks:
      - id: ruff-format-check
        name: Ruff format check
        entry: uv run ruff format --check .
        language: system
        pass_filenames: false
      - id: ruff-check
        name: Ruff lint
        entry: uv run ruff check .
        language: system
        pass_filenames: false
      - id: ty-check
        name: ty type check
        entry: uv run ty check
        language: system
        pass_filenames: false
      - id: focused-tests
        name: Focused fast tests
        entry: >-
          uv run pytest -q --no-cov
          tests/unit/operator/test_console.py
          tests/unit/operator/test_cli.py
          tests/unit/test_developer_tooling.py
        language: system
        pass_filenames: false
```

- [ ] **Step 5: Run configuration and live tool checks**

```bash
uv lock
uv sync --locked --all-extras --dev
uv run pytest -o addopts='' tests/unit/test_developer_tooling.py -q
just --list
uv run pre-commit validate-config
uv run pre-commit run --all-files
```

Expected: all pass. If `just` is not installed locally, install the pinned
external binary through the user's package manager; do not add a Python package
named `just` to `pyproject.toml`.

- [ ] **Step 6: Commit local tooling**

```bash
git add justfile .pre-commit-config.yaml pyproject.toml uv.lock \
  tests/unit/test_developer_tooling.py
git commit -m "Add shared local quality commands"
```

## Task 6: Make GitHub Actions Use the Shared Recipes

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/unit/test_developer_tooling.py`

- [ ] **Step 1: Add a RED CI delegation test**

```python
def test_ci_uses_just_recipes_and_keeps_locked_sync() -> None:
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "uv sync --locked --all-extras --dev" in text
    assert "cargo install just --locked --version 1.40.0" in text
    assert "run: just check" in text
    assert "run: just verify-dist" in text
    assert "osm-polygon-grid5000 --help" in text
    assert "mypy" not in text
```

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -o addopts='' tests/unit/test_developer_tooling.py -q
```

Expected: FAIL because CI does not install or call just and does not smoke-test
the operator CLI.

- [ ] **Step 3: Update CI without weakening existing checks**

Keep pinned `actions/checkout` and `astral-sh/setup-uv`. After Python setup,
add:

```yaml
      - name: Install just
        run: cargo install just --locked --version 1.40.0
```

Keep the explicit locked sync before any just recipe:

```yaml
      - name: Sync dependencies (locked, all extras + dev)
        run: uv sync --locked --all-extras --dev

      - name: Quality and tests
        run: just check

      - name: Build and verify distributions
        run: just verify-dist
```

Preserve the isolated installed-wheel smoke test and add:

```yaml
      - name: Operator CLI --help
        run: >-
          uv run --isolated --no-project --with dist/*.whl
          osm-polygon-grid5000 --help
```

Remove only the now-duplicated individual Ruff, ty, pytest, build, and
distribution steps.

- [ ] **Step 4: Verify locally**

```bash
uv run pytest -o addopts='' tests/unit/test_developer_tooling.py -q
just ci
```

Expected: PASS.

- [ ] **Step 5: Commit CI delegation**

```bash
git add .github/workflows/ci.yml tests/unit/test_developer_tooling.py
git commit -m "Run shared quality recipes in CI"
```

## Task 7: Update Public Contributor Documentation

**Files:**
- Modify: `CONTRIBUTING.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `tests/unit/test_developer_tooling.py`

- [ ] **Step 1: Add RED documentation contract tests**

```python
def test_contributing_documents_one_supported_toolchain() -> None:
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    for command in (
        "uv sync --locked --all-extras --dev",
        "just check",
        "just ci",
        "uv run pre-commit install",
        "uv run ty check",
    ):
        assert command in text
    assert "mypy" not in text.lower()


def test_readme_names_operator_and_primary_quality_command() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "osm-polygon-grid5000" in text
    assert "just check" in text
```

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -o addopts='' tests/unit/test_developer_tooling.py -q
```

Expected: FAIL on missing documented commands.

- [ ] **Step 3: Update `CONTRIBUTING.md`**

Replace the environment and quality-command sections with:

```markdown
## Environment setup

Install `uv` and `just`, then prepare the locked environment:

```bash
uv sync --locked --all-extras --dev
uv run pre-commit install
```

## Required quality commands

The repository exposes one command façade:

```bash
just format       # apply Ruff formatting
just check        # format check, Ruff, ty, and full pytest suite
just ci           # local equivalent of CI, including build verification
```

The recipes delegate to locked `uv` tools. For diagnosis, the underlying
commands remain available:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
```
```

Retain the existing repository hygiene and architecture rules. Correct the
outdated statement that sentence labeling is out of scope, because the
repository already contains the production labeling pipeline.

- [ ] **Step 4: Update README and changelog tersely**

Add one contributor sentence to README:

```markdown
Run `just check` before submitting changes; see [CONTRIBUTING.md](CONTRIBUTING.md)
for the locked uv, pre-commit, and CI workflow.
```

Add an Unreleased changelog entry:

```markdown
- Migrated the Grid'5000 operator command to a typed Typer boundary with
  terminal-safe Rich/tqdm progress while retaining plain machine output, and
  unified local and CI quality commands through uv, just, and pre-commit.
```

- [ ] **Step 5: Verify documentation contracts**

```bash
uv run pytest -o addopts='' tests/unit/test_developer_tooling.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit documentation**

```bash
git add CONTRIBUTING.md README.md CHANGELOG.md \
  tests/unit/test_developer_tooling.py
git commit -m "Document the unified developer workflow"
```

## Task 8: Full Acceptance and Final Integration

**Files:**
- Review all paths changed by Tasks 1–7.

- [ ] **Step 1: Synchronize exactly from the lock**

```bash
uv sync --locked --all-extras --dev
```

Expected: success with no lock mutation.

- [ ] **Step 2: Run the shared local acceptance gate**

```bash
just format-check
just lint
just typecheck
just test
```

Expected: all pass; full repository branch coverage remains at least 95%.

- [ ] **Step 3: Run build and distribution verification**

```bash
just build
just verify-dist
```

Expected: sdist and wheel build; distribution verifier reports `OK`.

- [ ] **Step 4: Run pre-commit**

```bash
uv run pre-commit run --all-files
```

Expected: all hooks pass without modifying files.

- [ ] **Step 5: Verify installed CLI behavior**

```bash
uv run --isolated --no-project --with dist/*.whl \
  osm-polygon-grid5000 --help
NO_COLOR=1 uv run osm-polygon-grid5000 --help
```

Expected: both succeed; the second contains no ANSI escape sequences.

- [ ] **Step 6: Audit scope and hygiene**

```bash
git diff --check
git status --short
git diff --stat
rg -n "mypy" pyproject.toml uv.lock justfile .pre-commit-config.yaml \
  .github/workflows CONTRIBUTING.md README.md
```

Expected:

- no whitespace errors;
- only intended source, tests, tooling, lock, CI, and documentation paths;
- no mypy references;
- no credentials, data, models, logs, caches, or generated distributions
  staged.

- [ ] **Step 7: Verify history and push normally to main**

```bash
git status --short --branch
git log --oneline --decorate -10
git push origin main
git rev-parse HEAD
git rev-parse origin/main
git status --short --branch
```

Expected:

- normal non-force push;
- `HEAD == origin/main`;
- one branch remains in use: `main`;
- working tree is clean.
