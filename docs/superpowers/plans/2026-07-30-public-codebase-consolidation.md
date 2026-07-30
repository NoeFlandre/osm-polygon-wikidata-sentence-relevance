# Public Codebase Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the existing production pipeline into focused modules of
at most 500 lines, replace mypy with Astral `ty`, and leave all supported
behavior and public interfaces unchanged.

**Architecture:** Preserve current public modules as compatibility facades and
move implementation into responsibility-oriented internal packages. Use one
repository contract to enforce file size and active-tooling consistency.
Extract one responsibility at a time behind existing tests, with a fresh
RED→GREEN boundary test for every move.

**Tech Stack:** Python 3.12, uv, Ruff, Astral ty, pytest/pytest-cov, PyArrow,
Hugging Face Hub, Bash, Grid'5000 OAR.

---

## File map

New internal packages:

- `operator/_cli/`: common primitives, run workflow, resume workflow,
  scheduling optimization, and parser.
- `operator/_state/`: contracts, JSON validation, secure filesystem I/O, and
  store implementation.
- `operator/_ssh/`: contracts, validation/redaction, transport, and log
  protocol.
- `operator/_config/`: validation, immutable models, and run identity.
- `operator/_relay/`: inventory validation, local retrieval, and remote
  staging.
- `application/_pipeline/`: contracts, single-shard execution, orchestration,
  and progress.
- `output/_card/`: YAML, Markdown sections, schema docs, and profile rendering.
- `output/_plots/`: dependencies, geography, and language distribution.
- `output/_publication/`: models, manifest/filesystem validation, and
  render-consistency validation.
- `output/_profile/`: contracts, parsing, and computation.
- `labeling/_finalization/`: manifest/card assembly and validation.
- `scripts/streaming/_driver/`: configuration, orchestration, and CLI.
- `scripts/streaming/_offload/`: contracts/validation, Hub I/O, and discovery.

Existing public module paths remain importable and re-export their documented
surface.

### Task 1: Replace mypy with ty and resolve the type baseline

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.github/workflows/ci.yml`
- Modify: `CONTRIBUTING.md`
- Modify: `docs/guides/development.md`
- Modify: `docs/guides/reproducibility.md`
- Modify: `scripts/verify_distribution.py`
- Modify: the production files named by `uv run ty check`
- Test: `tests/unit/contracts/test_project_metadata.py`
- Test: `tests/unit/contracts/test_documentation_consistency.py`
- Test: `tests/unit/contracts/test_repository_hygiene.py`

- [ ] **Step 1: Write failing tooling-contract tests**

Add assertions equivalent to:

```python
def test_development_tooling_uses_ty_without_mypy() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev = metadata["dependency-groups"]["dev"]
    assert any(item.startswith("ty") for item in dev)
    assert not any(item.startswith("mypy") for item in dev)
    assert "mypy" not in metadata.get("tool", {})


@pytest.mark.parametrize(
    "path",
    [
        ROOT / ".github/workflows/ci.yml",
        ROOT / "CONTRIBUTING.md",
        ROOT / "docs/guides/development.md",
        ROOT / "docs/guides/reproducibility.md",
    ],
)
def test_active_quality_documentation_uses_ty(path: Path) -> None:
    text = path.read_text()
    assert "uv run ty check" in text
    assert "uv run mypy" not in text
```

- [ ] **Step 2: Run the tests and capture RED**

Run:

```bash
uv run pytest -o addopts='' \
  tests/unit/contracts/test_project_metadata.py \
  tests/unit/contracts/test_documentation_consistency.py \
  tests/unit/contracts/test_repository_hygiene.py -q
```

Expected: fail because mypy remains the configured checker and ty is absent.

- [ ] **Step 3: Migrate tooling with uv**

Run:

```bash
uv remove --dev mypy
uv add --dev "ty>=0.0.1a20,<0.1"
```

Add:

```toml
[tool.ty.src]
include = [
    "src",
    "scripts/streaming",
    "scripts/grid5000/gpu_preflight.py",
    "scripts/render_assets.py",
    "scripts/verify_distribution.py",
]
```

Remove all `[tool.mypy]` and `[[tool.mypy.overrides]]` tables. Replace active
commands with `uv run ty check`. Update the distribution hygiene list from
`.mypy_cache/` to `.ty_cache/` only if ty creates such a cache; do not invent
an unused exclusion.

- [ ] **Step 4: Fix every ty diagnostic soundly**

Run `uv run ty check`, then fix diagnostics using:

- explicit `isinstance` narrowing for JSON and PyArrow-derived objects;
- `Mapping[str, object]` for covariant read-only arguments;
- small `Protocol` definitions for dynamic model and API objects;
- explicit optional checks instead of obsolete `type: ignore` comments;
- local typed variables for collection comprehensions.

Do not add global ignored rules, `# ty: ignore` without a named rule, casts
that bypass runtime validation, or weaker public annotations.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
uv run ty check
uv run pytest -o addopts='' \
  tests/unit/contracts/test_project_metadata.py \
  tests/unit/contracts/test_documentation_consistency.py \
  tests/unit/contracts/test_repository_hygiene.py -q
```

Expected: zero diagnostics and all focused tests pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .github CONTRIBUTING.md docs/guides \
  scripts/verify_distribution.py src scripts tests/unit/contracts
git commit -m "Replace mypy with ty"
```

### Task 2: Add structural boundaries and package documentation skeleton

**Files:**

- Modify: `tests/unit/contracts/test_source_structure.py`
- Create: `docs/architecture/packages/operator.md`
- Create: `docs/architecture/packages/application.md`
- Create: `docs/architecture/packages/sentences.md`
- Create: `docs/architecture/packages/labeling.md`
- Create: `docs/architecture/packages/output.md`
- Create: `docs/architecture/packages/publishing.md`
- Create: `docs/architecture/packages/streaming-operations.md`
- Modify: `docs/architecture/overview.md`
- Modify: `docs/index.md`

- [ ] **Step 1: Add the failing production-file ceiling**

```python
PRODUCTION_ROOTS = (
    ROOT / "src/osm_polygon_sentence_relevance",
    ROOT / "scripts/streaming",
)


def test_production_python_files_are_at_most_500_lines() -> None:
    oversized = {
        path.relative_to(ROOT).as_posix(): len(path.read_text().splitlines())
        for root in PRODUCTION_ROOTS
        for path in root.rglob("*.py")
        if len(path.read_text().splitlines()) > 500
    }
    assert oversized == {}
```

- [ ] **Step 2: Run and capture RED**

Run:

```bash
uv run pytest -o addopts='' \
  tests/unit/contracts/test_source_structure.py::test_production_python_files_are_at_most_500_lines -q
```

Expected: report every currently oversized production file.

- [ ] **Step 3: Add concise domain pages**

Each page must contain exactly these canonical headings:

```markdown
# Domain name

## Responsibility
## Public entry points
## Internal structure
## Invariants
## Tests
```

Link the pages from the architecture overview and documentation index. Do not
duplicate CLI tutorials or implementation history.

- [ ] **Step 4: Verify documentation contracts**

Run:

```bash
uv run pytest -o addopts='' \
  tests/unit/contracts/test_documentation_consistency.py \
  tests/unit/contracts/test_source_structure.py -q
```

The size test remains RED until Tasks 3–8 finish; documentation tests must be
GREEN.

- [ ] **Step 5: Commit the guard and documentation**

```bash
git add tests/unit/contracts/test_source_structure.py docs
git commit -m "Define production module boundaries"
```

### Task 3: Split operator configuration, SSH, state, and relay internals

**Files:**

- Create: `src/osm_polygon_sentence_relevance/operator/_config/`
- Create: `src/osm_polygon_sentence_relevance/operator/_ssh/`
- Create: `src/osm_polygon_sentence_relevance/operator/_state/`
- Create: `src/osm_polygon_sentence_relevance/operator/_relay/`
- Modify: `src/osm_polygon_sentence_relevance/operator/config.py`
- Modify: `src/osm_polygon_sentence_relevance/operator/ssh.py`
- Modify: `src/osm_polygon_sentence_relevance/operator/state.py`
- Modify: `src/osm_polygon_sentence_relevance/operator/relay.py`
- Modify: focused tests under `tests/unit/operator/`

- [ ] **Step 1: Add failing facade/boundary tests**

For each facade, assert the supported objects retain their import identity:

```python
def test_state_facade_exports_canonical_store() -> None:
    from osm_polygon_sentence_relevance.operator import state
    from osm_polygon_sentence_relevance.operator._state.store import StateStore

    assert state.StateStore is StateStore
```

Add equivalent tests for `OperatorConfig`, `SshClient`, and relay public
functions. Add import-graph assertions that internal modules do not import
`operator.cli`.

- [ ] **Step 2: Run and capture RED**

Run the four new boundary tests. Expected: internal packages do not exist.

- [ ] **Step 3: Extract config**

Move validation functions to `_config/validation.py`, immutable dataclasses to
`_config/models.py`, and canonical run-identity encoding to
`_config/identity.py`. Keep `config.py` as explicit imports plus `__all__`.

- [ ] **Step 4: Extract SSH**

Move result/error contracts to `_ssh/contracts.py`, target/path/redaction
logic to `_ssh/validation.py`, subprocess retry execution to
`_ssh/transport.py`, and offset log protocol to `_ssh/logs.py`. Keep the
constructor and methods of `SshClient` unchanged.

- [ ] **Step 5: Extract state**

Move phase/state/error contracts to `_state/contracts.py`, JSON validation to
`_state/json.py`, secure filesystem primitives to `_state/filesystem.py`, and
`StateStore` plus lock context to `_state/store.py`. Preserve file modes,
atomic replacement, event schema, and serialized JSON exactly.

- [ ] **Step 6: Extract relay**

Move `RelayInventory` and progress validation to `_relay/inventory.py`, local
generation assembly to `_relay/retrieval.py`, and remote staging to
`_relay/staging.py`. Preserve the `retrieve_to_seagate` and
`stage_to_destination` signatures.

- [ ] **Step 7: Verify GREEN**

Run:

```bash
uv run pytest -q --no-cov tests/unit/operator/test_config.py \
  tests/unit/operator/test_config_persisted.py \
  tests/unit/operator/test_ssh.py \
  tests/unit/operator/test_state.py \
  tests/unit/operator/test_relay.py \
  tests/unit/operator/test_relay_branches.py
uv run ty check
```

- [ ] **Step 8: Commit**

```bash
git add src/osm_polygon_sentence_relevance/operator tests/unit/operator
git commit -m "Split operator persistence and transport internals"
```

### Task 4: Split operator CLI orchestration

**Files:**

- Create: `src/osm_polygon_sentence_relevance/operator/_cli/common.py`
- Create: `src/osm_polygon_sentence_relevance/operator/_cli/resume.py`
- Create: `src/osm_polygon_sentence_relevance/operator/_cli/run.py`
- Create: `src/osm_polygon_sentence_relevance/operator/_cli/scheduling.py`
- Create: `src/osm_polygon_sentence_relevance/operator/_cli/parser.py`
- Modify: `src/osm_polygon_sentence_relevance/operator/cli.py`
- Modify: `tests/unit/operator/test_cli.py`
- Modify: resume/scheduling tests under `tests/unit/operator/`

- [ ] **Step 1: Add failing entry-point and ownership tests**

```python
def test_operator_cli_is_a_thin_public_facade() -> None:
    path = ROOT / "src/osm_polygon_sentence_relevance/operator/cli.py"
    assert len(path.read_text().splitlines()) <= 120


def test_cli_facade_preserves_public_entry_points() -> None:
    from osm_polygon_sentence_relevance.operator import cli
    assert callable(cli.build_parser)
    assert callable(cli.main)
```

Run and capture the expected line-count failure.

- [ ] **Step 2: Extract common primitives**

Move local/remote identity resolution, milestones, usage-policy preflight,
state transition helpers, and attachment construction to `_cli/common.py`.

- [ ] **Step 3: Extract resume and scheduling**

Move classification/continuation/relay logic to `_cli/resume.py`. Move queued
replacement selection and trial management to `_cli/scheduling.py`. Keep
injected clocks, sleepers, clients, and state stores so tests remain
deterministic and no network is used.

- [ ] **Step 4: Extract run and parser**

Move new-run orchestration to `_cli/run.py`; move parser and handler wiring to
`_cli/parser.py`. Keep `cli.main(argv)` responsible only for parsing, invoking
the selected handler, and rendering safe errors.

- [ ] **Step 5: Verify GREEN**

Run all `tests/unit/operator/`, `uv run ty check`, and the CLI integration
tests. Require exact stdout and command serialization tests to remain green.

- [ ] **Step 6: Commit**

```bash
git add src/osm_polygon_sentence_relevance/operator tests/unit/operator \
  tests/integration/test_cli.py
git commit -m "Split operator CLI orchestration"
```

### Task 5: Split application pipeline internals

**Files:**

- Create: `src/osm_polygon_sentence_relevance/application/_pipeline/contracts.py`
- Create: `src/osm_polygon_sentence_relevance/application/_pipeline/shard.py`
- Create: `src/osm_polygon_sentence_relevance/application/_pipeline/run.py`
- Create: `src/osm_polygon_sentence_relevance/application/_pipeline/progress.py`
- Modify: `src/osm_polygon_sentence_relevance/application/pipeline.py`
- Modify: `tests/unit/application/`

- [ ] **Step 1: Add failing facade and size tests**

Assert `PipelineResult`, `ShardCheckpointResult`, `process_single_shard`, and
`run_pipeline` resolve to canonical internal definitions and the facade is at
most 120 lines.

- [ ] **Step 2: Extract contracts and progress**

Move result dataclasses to `contracts.py`; move heartbeat calculations and
writes to `progress.py`.

- [ ] **Step 3: Extract shard execution**

Move `process_single_shard` and its checkpoint publication flow to `shard.py`.
Keep segmentation, report accounting, and exceptions unchanged.

- [ ] **Step 4: Extract orchestration**

Move the locked run, inventory reconciliation, cache reuse, quarantine, and
final export sequence to `run.py`. Keep `run_pipeline` as the public entry.

- [ ] **Step 5: Verify and commit**

Run all application unit/integration tests and `ty`, then:

```bash
git add src/osm_polygon_sentence_relevance/application tests/unit/application \
  tests/integration/test_pipeline.py
git commit -m "Split application pipeline orchestration"
```

### Task 6: Split output profiling, cards, plots, and validation

**Files:**

- Create/modify focused modules under:
  - `src/osm_polygon_sentence_relevance/output/_card/`
  - `src/osm_polygon_sentence_relevance/output/_plots/`
  - `src/osm_polygon_sentence_relevance/output/_profile/`
  - `src/osm_polygon_sentence_relevance/output/_publication/`
- Keep facades:
  - `output/profile.py`
  - `output/plots.py`
  - `output/validation_publication.py`
- Modify: `tests/unit/output/`

- [ ] **Step 1: Add failing import-identity and size tests**

Assert existing public imports point to canonical extracted implementations
and each facade is at most 120 lines.

- [ ] **Step 2: Split card rendering**

Move YAML front matter to `_card/yaml.py`, Markdown escaping/tables to
`_card/markdown.py`, factual dataset sections to `_card/dataset.py`, schema
documentation to `_card/schema.py`, and profile card rendering to
`_card/profile.py`.

- [ ] **Step 3: Split statistics and profile**

Move immutable statistical contracts and serialization separately from
Parquet scanning. Move example-row parsing separately from profile
aggregation.

- [ ] **Step 4: Split plots**

Move optional plotting dependency loading and deterministic setup to
`_plots/runtime.py`, geographic calculations/rendering to
`_plots/geography.py`, and language charts to `_plots/languages.py`.

- [ ] **Step 5: Split publication validation**

Move publication contracts/filesystem inventory to `_publication/contracts.py`
and `_publication/filesystem.py`; move manifest/accounting checks to
`_publication/manifest.py`; move deterministic README/profile re-render checks
to `_publication/rendering.py`.

- [ ] **Step 6: Verify and commit**

Run every output unit test, publication integration tests, `ty`, and the
distribution verifier. Commit:

```bash
git add src/osm_polygon_sentence_relevance/output tests/unit/output
git commit -m "Split output rendering and validation"
```

### Task 7: Split labeling finalization and streaming operations

**Files:**

- Create: `src/osm_polygon_sentence_relevance/labeling/_finalization/`
- Modify: `src/osm_polygon_sentence_relevance/labeling/finalization.py`
- Create: `scripts/streaming/_driver/`
- Modify: `scripts/streaming/driver.py`
- Create: `scripts/streaming/_offload/`
- Modify: `scripts/streaming/offload.py`
- Modify: focused labeling and streaming tests

- [ ] **Step 1: Add failing facade and line-ceiling tests**

Assert existing labeling and `scripts.streaming` imports remain canonical and
facades remain at most 120 lines.

- [ ] **Step 2: Split labeling finalization**

Move plot data, manifest/card assembly, and validation into separate
`_finalization` modules. Keep `finalize_labeled_dataset` and
`validate_labeled_publication` public and byte-deterministic.

- [ ] **Step 3: Split streaming driver**

Move `DriverConfig` and errors to `_driver/contracts.py`, shard orchestration
to `_driver/run.py`, scratch cleanup to `_driver/scratch.py`, and argument
parsing to `_driver/cli.py`.

- [ ] **Step 4: Split streaming offload**

Move metadata/identity validation to `_offload/validation.py`, Hub listing and
download to `_offload/hub.py`, materialization to `_offload/materialize.py`,
and offloader/discovery orchestration to `_offload/service.py`.

- [ ] **Step 5: Verify and commit**

Run labeling, streaming, shell launcher, distribution, and `ty` gates. Commit:

```bash
git add src/osm_polygon_sentence_relevance/labeling scripts/streaming \
  tests/unit/labeling tests/unit/scripts/streaming
git commit -m "Split labeling and streaming finalization"
```

### Task 8: Close structural, documentation, and repository gates

**Files:**

- Modify active documentation as required by the final architecture
- Modify source-structure and documentation tests only for factual final paths
- Do not alter archived specs/plans to rewrite historical tooling

- [ ] **Step 1: Run the 500-line structural contract**

```bash
uv run pytest -o addopts='' \
  tests/unit/contracts/test_source_structure.py::test_production_python_files_are_at_most_500_lines -q
```

Expected: pass with an empty oversized mapping.

- [ ] **Step 2: Audit architecture docs against files**

Verify every documented public entry point imports successfully and every
listed internal package exists. Remove duplicated or stale documentation.

- [ ] **Step 3: Run the complete acceptance gate**

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

Require at least 95% branch-aware coverage and zero failures.

- [ ] **Step 4: Audit staged content**

Confirm no credentials, tokens, datasets, models, checkpoints, OAR logs,
caches, build products, personal paths, or unrelated files are staged.

- [ ] **Step 5: Final commit and push**

Commit any final documentation-only corrections, verify `HEAD` and
`origin/main`, then:

```bash
git push origin main
```

Never force-push. Report the full SHA and clean status.
