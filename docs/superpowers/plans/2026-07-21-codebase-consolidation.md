# Codebase Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove superseded operational and historical clutter, establish focused internal module ownership, and leave one documented production pipeline without changing any supported behavior.

**Architecture:** Keep the installed CLI, canonical domain packages, root compatibility facades, serialized artifacts, and bounded streaming workflow stable. Delete dependency-proven obsolete tooling first, then split the two oversized implementation modules behind unchanged import facades, consolidate tests and docs, and finish with distribution and isolated-wheel verification.

**Tech Stack:** Python 3.12, PyArrow, pytest/pytest-cov, Ruff, mypy, Bash, setuptools/uv, Hugging Face Hub operational APIs.

---

## Target File Map

### Retained public and production surfaces

- `src/osm_polygon_sentence_relevance/application/checkpoint.py`: stable re-export facade.
- `src/osm_polygon_sentence_relevance/application/_checkpoint/locking.py`: work-directory lock ownership.
- `src/osm_polygon_sentence_relevance/application/_checkpoint/inventory.py`: source manifests and run inventories.
- `src/osm_polygon_sentence_relevance/application/_checkpoint/io.py`: strict atomic file installation and directory fsync.
- `src/osm_polygon_sentence_relevance/application/_checkpoint/storage.py`: checkpoint publication, loading, quarantine, and heartbeat.
- `src/osm_polygon_sentence_relevance/application/_checkpoint/validation.py`: metadata/schema/path validation.
- `src/osm_polygon_sentence_relevance/output/dataset_card.py`: stable re-export facade.
- `src/osm_polygon_sentence_relevance/output/_card/statistics.py`: `DatasetStatistics` and statistics serialization/computation.
- `src/osm_polygon_sentence_relevance/output/_card/rendering.py`: legacy and profile-driven Markdown rendering.
- `src/osm_polygon_sentence_relevance/output/profile.py`: profile data and Parquet profiling.
- `src/osm_polygon_sentence_relevance/output/plots.py`: deterministic geographic/language plot rendering.
- `scripts/streaming/`: retained bounded streaming implementation.
- `scripts/grid5000/{submit,run}_streaming_*`: retained production launchers.
- `scripts/grid5000/gpu_preflight.py`: retained production CUDA proof.
- `scripts/grid5000/_finalize_persist.sh`: retained final-artifact persistence helper.
- `scripts/render_assets.py`, `scripts/verify_distribution.py`: retained publication/distribution operations.

### Deleted superseded surfaces

- `scripts/audit/`, `scripts/audit_upstream_correction.py`.
- `scripts/grid5000/_cache_ref_validator.sh`.
- `scripts/grid5000/_run_metadata.py`.
- `scripts/grid5000/_validate_artifact.py`.
- `scripts/grid5000/{submit,run}_gpu_smoke*.sh`.
- `scripts/grid5000/{submit,run}_gpu_build*.sh`.
- their dedicated tests under `tests/unit/scripts/`.

---

### Task 1: Protect the Supported Surface and Define the Production Inventory

**Files:**
- Create: `tests/compatibility/test_supported_surface.py`
- Create: `tests/unit/contracts/test_repository_hygiene.py`
- Modify: `tests/unit/contracts/test_distribution_verifier.py`

- [ ] **Step 1: Write the failing supported-surface characterization test**

```python
from inspect import signature

from osm_polygon_sentence_relevance.application.checkpoint import (
    load_shard_checkpoint,
    publish_shard_checkpoint,
    validate_source_commit,
    validate_work_dir,
)
from osm_polygon_sentence_relevance.application.pipeline import run_pipeline
from osm_polygon_sentence_relevance.output.dataset_card import (
    DatasetStatistics,
    compute_parquet_statistics,
    render_dataset_card,
    render_dataset_card_from_profile,
)


def test_supported_callable_signatures_are_stable() -> None:
    assert "work_dir" in signature(run_pipeline).parameters
    assert list(signature(validate_source_commit).parameters) == ["value"]
    assert list(signature(validate_work_dir).parameters) == ["work_dir"]
    assert "verified_manifest" in signature(publish_shard_checkpoint).parameters
    assert "shard_key" in signature(load_shard_checkpoint).parameters
    assert DatasetStatistics.__name__ == "DatasetStatistics"
    assert callable(compute_parquet_statistics)
    assert callable(render_dataset_card)
    assert callable(render_dataset_card_from_profile)
```

- [ ] **Step 2: Run the characterization test and record GREEN baseline**

Run: `uv run pytest tests/compatibility/test_supported_surface.py -q`

Expected: PASS. This is a characterization exception to RED: it freezes existing behavior before structural moves. No production edit occurs in this step.

- [ ] **Step 3: Write the failing production-inventory hygiene test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RETAINED_SHELL = {
    "_finalize_persist.sh",
    "run_streaming_build.sh",
    "run_streaming_build_job.sh",
    "run_streaming_finalization.sh",
    "run_streaming_finalization_job.sh",
    "submit_streaming_build.sh",
    "submit_streaming_finalization.sh",
}


def test_grid5000_contains_only_production_shell_entrypoints() -> None:
    actual = {path.name for path in (ROOT / "scripts/grid5000").glob("*.sh")}
    assert actual == RETAINED_SHELL


def test_one_off_audit_scripts_are_absent() -> None:
    assert not (ROOT / "scripts/audit").exists()
    assert not (ROOT / "scripts/audit_upstream_correction.py").exists()
```

- [ ] **Step 4: Verify RED**

Run: `uv run pytest tests/unit/contracts/test_repository_hygiene.py -q`

Expected: FAIL listing the smoke, full-snapshot build, cache-ref, and audit paths.

- [ ] **Step 5: Add exact retained-script expectations to the distribution tests**

Add a parametrized assertion that the sdist must contain the seven retained shell scripts above plus `gpu_preflight.py`, all five `scripts/streaming/*.py` modules, `render_assets.py`, and `verify_distribution.py`; assert every other `scripts/grid5000/*.sh` name is absent.

- [ ] **Step 6: Commit the contract boundary**

```bash
git add tests/compatibility/test_supported_surface.py \
  tests/unit/contracts/test_repository_hygiene.py \
  tests/unit/contracts/test_distribution_verifier.py
git commit -m "Define supported production surface"
```

---

### Task 2: Remove Superseded Operational Tooling

**Files:**
- Delete: `scripts/audit/`
- Delete: `scripts/audit_upstream_correction.py`
- Delete: `scripts/grid5000/_cache_ref_validator.sh`
- Delete: `scripts/grid5000/_run_metadata.py`
- Delete: `scripts/grid5000/_validate_artifact.py`
- Delete: `scripts/grid5000/run_gpu_build.sh`
- Delete: `scripts/grid5000/run_gpu_build_job.sh`
- Delete: `scripts/grid5000/submit_gpu_build.sh`
- Delete: `scripts/grid5000/run_gpu_smoke.sh`
- Delete: `scripts/grid5000/run_gpu_smoke_job.sh`
- Delete: `scripts/grid5000/submit_gpu_smoke.sh`
- Delete: `tests/unit/scripts/test_cache_ref_validator_sh.py`
- Delete: `tests/unit/scripts/test_run_gpu_build_job_sh.py`
- Delete: `tests/unit/scripts/test_run_gpu_build_sh.py`
- Delete: `tests/unit/scripts/test_run_gpu_smoke_job_sh.py`
- Delete: `tests/unit/scripts/test_run_gpu_smoke_sh.py`
- Delete: `tests/unit/scripts/test_run_metadata_cli.py`
- Delete: `tests/unit/scripts/test_submit_gpu_build_sh.py`
- Delete: `tests/unit/scripts/test_submit_gpu_smoke_sh.py`
- Delete: `tests/unit/scripts/test_validate_artifact.py`
- Modify: `scripts/verify_distribution.py`
- Modify: `MANIFEST.in`

- [ ] **Step 1: Remove the files named above with `git rm`**

The deletion is intentionally exact. Retain `gpu_preflight.py`, `_finalize_persist.sh`, every `*streaming*` shell script, and the entire `scripts/streaming/` package.

- [ ] **Step 2: Replace the verifier's historical script tuple**

```python
SDIST_PRODUCTION_SHELL_SCRIPTS = (
    "scripts/grid5000/_finalize_persist.sh",
    "scripts/grid5000/run_streaming_build.sh",
    "scripts/grid5000/run_streaming_build_job.sh",
    "scripts/grid5000/run_streaming_finalization.sh",
    "scripts/grid5000/run_streaming_finalization_job.sh",
    "scripts/grid5000/submit_streaming_build.sh",
    "scripts/grid5000/submit_streaming_finalization.sh",
)

SDIST_PRODUCTION_PYTHON_SCRIPTS = (
    "scripts/grid5000/gpu_preflight.py",
    "scripts/render_assets.py",
    "scripts/verify_distribution.py",
    "scripts/streaming/__init__.py",
    "scripts/streaming/data_root.py",
    "scripts/streaming/downloader.py",
    "scripts/streaming/driver.py",
    "scripts/streaming/finalization.py",
    "scripts/streaming/offload.py",
)
```

Verify regular-file mode `0o755` for each retained shell script and forbid every `scripts/` path in the wheel.

- [ ] **Step 3: Narrow the source manifest**

Replace broad `recursive-include scripts *.py` and `*.sh` rules with explicit
production directories/files. Continue excluding data, caches, weights, local
guides, and `docs/superpowers/` design-working documents from the public sdist.

- [ ] **Step 4: Verify GREEN for inventory and distribution unit tests**

Run:

```bash
uv run pytest tests/unit/contracts/test_repository_hygiene.py \
  tests/unit/contracts/test_distribution_verifier.py -q
```

Expected: PASS.

- [ ] **Step 5: Run retained operational tests**

Run:

```bash
uv run pytest tests/unit/scripts/streaming \
  tests/unit/scripts/test_finalize_persist_sh.py \
  tests/unit/scripts/test_gpu_preflight.py \
  tests/unit/scripts/test_submit_streaming_finalization_sh.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add MANIFEST.in scripts tests scripts/verify_distribution.py
git commit -m "Remove superseded operational tooling"
```

---

### Task 3: Split Checkpoint Responsibilities Behind the Stable Facade

**Files:**
- Create: `src/osm_polygon_sentence_relevance/application/_checkpoint/__init__.py`
- Create: `src/osm_polygon_sentence_relevance/application/_checkpoint/locking.py`
- Create: `src/osm_polygon_sentence_relevance/application/_checkpoint/inventory.py`
- Create: `src/osm_polygon_sentence_relevance/application/_checkpoint/io.py`
- Create: `src/osm_polygon_sentence_relevance/application/_checkpoint/validation.py`
- Create: `src/osm_polygon_sentence_relevance/application/_checkpoint/storage.py`
- Replace: `src/osm_polygon_sentence_relevance/application/checkpoint.py`
- Create: `tests/unit/application/test_checkpoint_module_boundaries.py`
- Modify: existing checkpoint tests only for private-module imports that become unnecessary

- [ ] **Step 1: Write the failing module-boundary test**

```python
from osm_polygon_sentence_relevance.application import checkpoint
from osm_polygon_sentence_relevance.application._checkpoint import (
    inventory,
    locking,
    storage,
    validation,
)


def test_checkpoint_facade_reexports_canonical_owners() -> None:
    assert checkpoint.acquire_work_dir_lock is locking.acquire_work_dir_lock
    assert checkpoint.RunInventory is inventory.RunInventory
    assert checkpoint.validate_checkpoint_metadata is validation.validate_checkpoint_metadata
    assert checkpoint.publish_shard_checkpoint is storage.publish_shard_checkpoint
    assert checkpoint.load_shard_checkpoint is storage.load_shard_checkpoint
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/unit/application/test_checkpoint_module_boundaries.py -q`

Expected: FAIL because `_checkpoint` does not exist.

- [ ] **Step 3: Move lock ownership**

Move `WorkDirLock`, `acquire_work_dir_lock`, and `release_work_dir_lock` unchanged to `locking.py`. Keep constants private and preserve exception types/messages.

- [ ] **Step 4: Move inventory ownership**

Move `SourceFileEntry`, `RunInventory`, source hashing/manifest construction,
reconciliation, and inventory read/write/quarantine functions to
`inventory.py`.

- [ ] **Step 5: Move strict atomic I/O ownership**

Move `_fsync_dir_strict`, `_atomic_write_bytes`, and `_atomic_write_parquet` to
`io.py`. Both inventory and storage may depend on this module; it depends only
on stdlib, PyArrow, and public checkpoint errors, preventing a cycle.

- [ ] **Step 6: Move validation ownership**

Move schema fingerprinting, checkpoint exceptions, report serialization, metadata validation, staged layout validation, active-directory scanning, work-dir validation, and source-commit validation to `validation.py`.

- [ ] **Step 7: Move storage ownership**

Move atomic writes, publish/load/quarantine operations, safe active-directory handling, and heartbeat writing to `storage.py`. Depend on `validation.py` and inventory dataclasses; do not import the public facade internally.

- [ ] **Step 8: Replace `checkpoint.py` with explicit re-exports**

```python
"""Stable checkpoint API backed by focused internal modules."""

from ._checkpoint.inventory import (
    RunInventory,
    SourceFileEntry,
    compute_run_inventory,
    compute_shard_source_manifest,
    load_run_inventory,
    load_run_inventory_quarantine_first,
    reconcile_inventory,
    write_run_inventory,
)
from ._checkpoint.locking import (
    WorkDirLock,
    acquire_work_dir_lock,
    release_work_dir_lock,
)
from ._checkpoint.storage import (
    load_shard_checkpoint,
    publish_shard_checkpoint,
    quarantine_shard_checkpoint,
    scan_active_directory,
    write_heartbeat,
)
from ._checkpoint.validation import (
    CheckpointPublicationError,
    CheckpointValidationError,
    segmented_schema_sha256,
    validate_checkpoint_metadata,
    validate_source_commit,
    validate_work_dir,
)

__all__ = [
    "CheckpointPublicationError",
    "CheckpointValidationError",
    "RunInventory",
    "SourceFileEntry",
    "WorkDirLock",
    "acquire_work_dir_lock",
    "compute_run_inventory",
    "compute_shard_source_manifest",
    "load_run_inventory",
    "load_run_inventory_quarantine_first",
    "load_shard_checkpoint",
    "publish_shard_checkpoint",
    "quarantine_shard_checkpoint",
    "reconcile_inventory",
    "release_work_dir_lock",
    "scan_active_directory",
    "segmented_schema_sha256",
    "validate_checkpoint_metadata",
    "validate_source_commit",
    "validate_work_dir",
    "write_heartbeat",
    "write_run_inventory",
]
```

- [ ] **Step 9: Verify GREEN and full checkpoint behavior**

Run:

```bash
uv run pytest tests/unit/application/test_checkpoint_module_boundaries.py \
  tests/unit/application/test_pipeline_checkpoint.py \
  tests/unit/application/test_checkpoint_correctness_amendment.py \
  tests/unit/application/test_pipeline_single_shard_pause2.py -q
```

Expected: PASS with no changed assertion text or exception identity.

- [ ] **Step 10: Commit**

```bash
git add src/osm_polygon_sentence_relevance/application tests/unit/application
git commit -m "Separate checkpoint responsibilities"
```

---

### Task 4: Separate Card Statistics, Rendering, and Plots

**Files:**
- Create: `src/osm_polygon_sentence_relevance/output/_card/__init__.py`
- Create: `src/osm_polygon_sentence_relevance/output/_card/statistics.py`
- Create: `src/osm_polygon_sentence_relevance/output/_card/rendering.py`
- Create: `src/osm_polygon_sentence_relevance/output/plots.py`
- Replace: `src/osm_polygon_sentence_relevance/output/dataset_card.py`
- Modify: `src/osm_polygon_sentence_relevance/output/profile.py`
- Create: `tests/unit/output/test_output_module_boundaries.py`

- [ ] **Step 1: Write the failing output ownership test**

```python
from osm_polygon_sentence_relevance.output import dataset_card, plots
from osm_polygon_sentence_relevance.output._card import rendering, statistics


def test_dataset_card_facade_reexports_focused_implementations() -> None:
    assert dataset_card.DatasetStatistics is statistics.DatasetStatistics
    assert dataset_card.compute_statistics is statistics.compute_statistics
    assert dataset_card.render_dataset_card is rendering.render_dataset_card
    assert (
        dataset_card.render_dataset_card_from_profile
        is rendering.render_dataset_card_from_profile
    )
    assert callable(plots.render_geographic_coverage_png)
    assert callable(plots.render_language_distribution_png)


def test_profile_keeps_plot_compatibility_exports() -> None:
    from osm_polygon_sentence_relevance.output import profile

    assert profile.render_geographic_coverage_png is plots.render_geographic_coverage_png
    assert profile.render_language_distribution_png is plots.render_language_distribution_png
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/unit/output/test_output_module_boundaries.py -q`

Expected: FAIL because `_card` and `plots` do not exist.

- [ ] **Step 3: Move statistics without behavior changes**

Move `DatasetStatistics`, count sorting, input-dataset resolution, table/Parquet statistics, and statistics serialization/deserialization to `_card/statistics.py`.

- [ ] **Step 4: Move all Markdown/YAML rendering**

Move escaping, YAML, count tables, provenance, scope, schema documentation, `render_dataset_card`, and `render_dataset_card_from_profile` to `_card/rendering.py`. Keep prose byte-for-byte unchanged during this structural task.

- [ ] **Step 5: Move deterministic plots out of profiling**

Move plot constants, vendored-outline loading, Matplotlib seeding/PNG encoding,
centroid collection, captions, and both plot renderers to `output/plots.py`.
Keep `DatasetProfile`, `AssetInfo`, `ExampleRow`, Parquet profiling, example
JSON, and `sha256_bytes` in `profile.py`. Re-export the moved plot functions
from `profile.py`; `plots.py` uses `TYPE_CHECKING` plus structural attribute
access so it does not import `profile.py` at runtime and create a cycle.

- [ ] **Step 6: Replace `dataset_card.py` with explicit stable re-exports**

Re-export the pre-refactor public functions/classes and `schema_has_map_types`; do not re-export underscore-prefixed helpers.

- [ ] **Step 7: Verify GREEN across the output subsystem**

Run:

```bash
uv run pytest tests/unit/output/test_output_module_boundaries.py \
  tests/unit/output -q
```

Expected: PASS, including deterministic PNG byte comparisons, generated README equality, manifest accounting, and publication validation.

- [ ] **Step 8: Commit**

```bash
git add src/osm_polygon_sentence_relevance/output tests/unit/output
git commit -m "Separate dataset publication responsibilities"
```

---

### Task 5: Consolidate Tests Around Durable Contracts

**Files:**
- Rename/split: amendment- and phase-named files under `tests/unit/application/`, `tests/unit/output/`, `tests/unit/sentences/`, and `tests/integration/`
- Modify: `tests/support/`
- Delete: `docs/superpowers/plans/2026-07-20-streaming-production-completion.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write a failing test-file hygiene assertion**

Extend `test_repository_hygiene.py`:

```python
def test_current_test_files_use_contract_names() -> None:
    forbidden = ("amendment", "phase", "pause")
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("test_*.py")
        if any(token in path.name.lower() for token in forbidden)
    ]
    assert offenders == []
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/unit/contracts/test_repository_hygiene.py -q`

Expected: FAIL listing amendment/phase/pause filenames.

- [ ] **Step 3: Rename by behavior and split oversized files**

Use names such as:

- `test_checkpoint_locking.py`, `test_checkpoint_inventory.py`,
  `test_checkpoint_storage.py`, `test_checkpoint_recovery.py`;
- `test_card_statistics.py`, `test_card_rendering.py`,
  `test_publication_profile.py`, `test_publication_plots.py`;
- `test_device_placement.py`, `test_segmentation_adapter.py`.

Move shared builders/fakes to `tests/support/`. Preserve all unique assertions.

- [ ] **Step 4: Remove redundant and placeholder tests**

Delete `assert True` placeholders and exact duplicates only after proving the same contract remains in a named retained test with `pytest --collect-only` counts recorded before and after. Do not reduce coverage of public errors, negative validation, atomicity, resume, schema, hash, or publication behavior.

- [ ] **Step 5: Expand static checking to production operational Python**

Set:

```toml
[tool.mypy]
files = ["src", "scripts/streaming", "scripts/grid5000/gpu_preflight.py", "scripts/render_assets.py", "scripts/verify_distribution.py"]
```

Resolve errors without broad `ignore_errors`, `type: ignore`, or new coverage exclusions.

- [ ] **Step 6: Handle external `docopt` warnings narrowly**

Add a pytest warning filter scoped to the installed `docopt` module's invalid-escape `SyntaxWarning`; do not suppress project warnings or all `SyntaxWarning` instances.

- [ ] **Step 7: Verify GREEN**

Run:

```bash
uv run pytest tests/unit/contracts/test_repository_hygiene.py -q
uv run pytest -q
uv run mypy
```

Expected: all tests pass, coverage remains at least 95%, and project output contains no warnings.

- [ ] **Step 8: Commit**

```bash
git add tests pyproject.toml docs/superpowers/plans/2026-07-20-streaming-production-completion.md
git commit -m "Consolidate tests around durable contracts"
```

---

### Task 6: Rewrite Maintained Documentation Around the Product

**Files:**
- Rewrite: `README.md`
- Rewrite: `CHANGELOG.md`
- Modify: `CONTRIBUTING.md`
- Modify: `docs/index.md`
- Modify: `docs/architecture/overview.md`
- Rewrite: `docs/guides/grid5000.md`
- Modify: `docs/guides/getting-started.md`
- Modify: `docs/guides/development.md`
- Modify: `docs/guides/reproducibility.md`
- Modify: `docs/reference/api.md`
- Modify: `docs/reference/cli.md`
- Modify: `docs/reference/data-contract.md`
- Modify: `tests/unit/contracts/test_documentation_consistency.py`
- Modify: `tests/unit/scripts/test_grid5000_doc.py`

- [ ] **Step 1: Add failing current-document hygiene tests**

```python
CURRENT_DOCS = (
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "CHANGELOG.md",
    ROOT / "docs/index.md",
    ROOT / "docs/architecture/overview.md",
    ROOT / "docs/guides/development.md",
    ROOT / "docs/guides/getting-started.md",
    ROOT / "docs/guides/grid5000.md",
    ROOT / "docs/guides/reproducibility.md",
    ROOT / "docs/reference/api.md",
    ROOT / "docs/reference/cli.md",
    ROOT / "docs/reference/data-contract.md",
)
STALE_TERMS = (
    "Phase 9",
    "amendment",
    "smoke test",
    "submit_gpu_build.sh",
    "run_gpu_build.sh",
    "submit_gpu_smoke.sh",
    "run_gpu_smoke.sh",
)


def test_current_docs_contain_no_historical_or_superseded_workflows() -> None:
    offenders = {
        path: term
        for path in CURRENT_DOCS
        for term in STALE_TERMS
        if term.casefold() in path.read_text(encoding="utf-8").casefold()
    }
    assert offenders == {}
```

Add a Grid'5000 test that extracts exactly two canonical frontend commands:
`submit_streaming_build.sh` and `submit_streaming_finalization.sh`.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest tests/unit/contracts/test_documentation_consistency.py \
  tests/unit/scripts/test_grid5000_doc.py -q
```

Expected: FAIL on historical phase and obsolete script references.

- [ ] **Step 3: Rewrite the Grid'5000 guide**

Use this structure only:

1. prerequisites and storage model;
2. immutable source/model revisions;
3. frontend environment preparation;
4. bounded streaming submission;
5. monitoring and resume;
6. deterministic finalization;
7. validation and publication boundary;
8. failure recovery and cleanup.

State that inference runs only on an allocated CUDA node. Remove smoke,
interactive debugging, incident history, old job layouts, and obsolete helpers.

- [ ] **Step 4: Rewrite root navigation and maintained references**

Make README a concise entry point. Ensure API/CLI/data docs describe current
canonical modules, all parser flags, checkpoint semantics, streaming workflow,
and the five-file publication contract without duplicating each other.

- [ ] **Step 5: Condense the changelog**

Replace phase diary entries under `[Unreleased]` with concise product-facing
Added/Changed/Fixed/Removed bullets. Preserve meaningful released version
sections if any exist; do not retain debugging chronology or job identifiers.

- [ ] **Step 6: Verify documentation GREEN**

Run:

```bash
uv run pytest tests/unit/contracts/test_documentation_consistency.py \
  tests/unit/scripts/test_grid5000_doc.py -q
uv run osm-polygon-sentence-relevance --help > /tmp/current-help.txt
```

Expected: links resolve, documented flags match parser output, only current
streaming commands appear, and no stale terms remain.

- [ ] **Step 7: Commit**

```bash
git add README.md CHANGELOG.md CONTRIBUTING.md docs tests/unit/contracts \
  tests/unit/scripts/test_grid5000_doc.py
git commit -m "Rewrite documentation around production workflow"
```

---

### Task 7: Complete the Release Gate and Repository Audit

**Files:**
- No planned source changes. A gate failure returns execution to the task that
  owns the failing contract; add a reproducing RED test there before correction.

- [ ] **Step 1: Run formatting and linting**

```bash
uv run ruff format --check .
uv run ruff check .
```

Expected: zero changes and zero findings.

- [ ] **Step 2: Run typing and full tests**

```bash
uv run mypy
uv run pytest -q
```

Expected: zero typing errors, zero test failures, no project warnings, and branch coverage at least 95%.

- [ ] **Step 3: Build and verify distributions**

```bash
uv build
uv run python scripts/verify_distribution.py dist/*.whl dist/*.tar.gz
```

Expected: wheel and sdist build; verifier reports `OK`.

- [ ] **Step 4: Inspect archive contents independently**

Assert the wheel contains installed package code, `py.typed`, metadata, and no
docs/tests/scripts. Assert the sdist contains current docs, source, tests, and
only the retained production scripts; no data, weights, caches, or local guides.

- [ ] **Step 5: Run isolated-wheel acceptance without optional dependencies**

Create a fresh temporary Python 3.12 environment, install only the wheel, then verify:

```bash
python -c "import osm_polygon_sentence_relevance"
osm-polygon-sentence-relevance --help
```

Assert neither `torch`, `wtpsplit`, nor `huggingface_hub` is imported by help.

- [ ] **Step 6: Validate retained shell scripts**

```bash
for script in scripts/grid5000/*.sh; do bash -n "$script"; done
```

Expected: every retained script exits zero.

- [ ] **Step 7: Run final hygiene scans**

```bash
rg -n '(Phase [0-9]|amendment|TODO|FIXME|HACK|TEMP|DEBUG|breakpoint\(|pdb\.)' \
  src scripts tests README.md CONTRIBUTING.md CHANGELOG.md docs \
  --glob '!docs/superpowers/**'
git ls-files | rg '(\.DS_Store|\.parquet$|\.safetensors$|\.bin$|__pycache__)'
git diff --check
```

Expected: no current-surface matches except durable ADR text explicitly allowed
by the hygiene test; no tracked artifacts; clean whitespace.

- [ ] **Step 8: Remove generated outputs safely**

Remove `dist/`, `build/`, egg-info, coverage output, and tool caches created by
verification. Do not remove user data, model caches, or files outside the
worktree.

- [ ] **Step 9: Re-run the concise final gate after cleanup**

```bash
git status --short
git diff --check
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
```

Expected: only intentional source changes before the final commit; all gates green.

- [ ] **Step 10: Commit the release-gate corrections**

```bash
git add -A
git commit -m "Complete codebase consolidation"
```

Do not merge or push until the complete branch diff is reviewed against the
design specification.
