# Operator Configuration Modularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `operator/config.py` into focused validation and model modules
while preserving its complete public and behavioral contract.

**Architecture:** Keep `operator/config.py` as an explicit compatibility
facade. Put private parsing and normalization in `_config/validation.py` and
immutable constants, enums, and dataclasses in `_config/models.py`; expose the
same supported names through `_config/__init__.py`.

**Tech Stack:** Python 3.12, dataclasses, pytest, Ruff, Astral ty, uv.

---

## File map

- `operator/_config/validation.py`: private scalar, identifier, revision,
  scope, stage, and runtime validation.
- `operator/_config/models.py`: public constants, enums, immutable
  requirements, identity, and operator configuration.
- `operator/_config/__init__.py`: explicit internal package exports.
- `operator/config.py`: stable public re-export facade.
- `tests/unit/operator/test_config_structure.py`: structural and compatibility
  boundary.
- `docs/architecture/packages/operator.md`: current ownership.

### Task 1: Establish the failing structural contract

**Files:**

- Create: `tests/unit/operator/test_config_structure.py`

- [ ] **Step 1: Add the facade and size tests**

Add tests that parse `operator/config.py` with `ast` and require its body after
the module docstring to contain only `ImportFrom` nodes and one literal
`__all__` assignment. Require these files to exist:

```python
PACKAGE = SOURCE / "operator" / "_config"
assert (PACKAGE / "__init__.py").is_file()
assert (PACKAGE / "validation.py").is_file()
assert (PACKAGE / "models.py").is_file()
```

Require every Python file in `operator/_config/` plus `operator/config.py` to
contain at most 500 physical lines. Import `operator.config` and assert its
`__all__` exactly matches the current supported symbol set.

- [ ] **Step 2: Verify RED**

Run:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -o addopts='' \
  tests/unit/operator/test_config_structure.py -q
```

Expected: failure because `_config/` does not exist and `config.py` still
contains implementation.

- [ ] **Step 3: Commit the RED contract**

```bash
git add tests/unit/operator/test_config_structure.py
git commit -m "Test operator config module boundary"
```

### Task 2: Extract validation and immutable models

**Files:**

- Create: `src/osm_polygon_sentence_relevance/operator/_config/__init__.py`
- Create: `src/osm_polygon_sentence_relevance/operator/_config/validation.py`
- Create: `src/osm_polygon_sentence_relevance/operator/_config/models.py`
- Replace: `src/osm_polygon_sentence_relevance/operator/config.py`

- [ ] **Step 1: Move validation without changing it**

Move the existing private regular expressions and helpers into
`validation.py`:

```python
_coerce_int
_validate_nonblank_no_ws
_validate_repo_id
_validate_model_file
_validate_hex
_validate_region
_canonicalize_scope
_canonicalize_stage
_require_run_fields_for_scope
_normalize_runtime_requirements
```

Keep the exact validation branches and messages. Because scope and stage are
models, make the three enum-aware helpers generic over the passed enum classes
rather than importing `models.py`; this preserves the one-way dependency.

- [ ] **Step 2: Move models and defaults**

Move all public defaults, `Scope`, `Stage`, `Grid5000Requirements`,
`RunIdentity`, and `OperatorConfig` into `models.py`. Import only private
helpers from `validation.py`. Preserve declarations, signatures,
`__post_init__`, `build`, `from_persisted`, `to_dict`, `canonical_json`, and
`run_id` byte-for-byte except for required import qualification.

- [ ] **Step 3: Add explicit exports**

In `_config/__init__.py`, explicitly import every supported name from
`models.py` and define the same literal `__all__` as the old public module.

Replace `operator/config.py` with a module docstring, explicit imports from
`operator._config`, and the identical literal `__all__`. Do not use wildcard
imports, `__getattr__`, aliases, or compatibility wrappers.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -o addopts='' \
  tests/unit/operator/test_config_structure.py \
  tests/unit/operator/test_config.py \
  tests/unit/operator/test_config_persisted.py -q
```

Expected: all pass.

- [ ] **Step 5: Verify operator regressions**

Run:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -o addopts='' \
  tests/unit/operator -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run ty check
UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff check .
```

Expected: zero failures and zero diagnostics.

- [ ] **Step 6: Commit implementation**

```bash
git add src/osm_polygon_sentence_relevance/operator/config.py \
  src/osm_polygon_sentence_relevance/operator/_config \
  tests/unit/operator/test_config_structure.py
git commit -m "Modularize operator configuration"
```

### Task 3: Document and perform repository acceptance

**Files:**

- Modify: `docs/architecture/packages/operator.md`

- [ ] **Step 1: Document the internal boundary**

State that `_config/validation.py` owns parsing and validation,
`_config/models.py` owns immutable contracts, and `config.py` is the stable
public facade. Record the one-way dependency from models to validation.

- [ ] **Step 2: Run complete acceptance**

Run:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv sync --locked --all-extras --dev
UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/uv-cache uv run ty check
COVERAGE_FILE=/private/tmp/.coverage-config-refactor \
  UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/uv-cache uv build --no-build-isolation
UV_CACHE_DIR=/private/tmp/uv-cache uv run python \
  scripts/verify_distribution.py dist/*.whl dist/*.tar.gz
for script in scripts/grid5000/*.sh; do bash -n "$script"; done
git diff --check
```

Expected: all existing gates pass and repository coverage remains at least
95%.

- [ ] **Step 3: Audit and commit documentation**

Confirm `git status --short` contains only the planned configuration, tests,
and operator documentation. Confirm the diff contains no credentials, runtime
artifacts, caches, logs, datasets, model files, or generated distributions.

```bash
git add docs/architecture/packages/operator.md
git commit -m "Document operator configuration boundary"
```

- [ ] **Step 4: Push after final verification**

Confirm `HEAD` is on `main`, the tree is clean, and `origin/main` is the
pre-refactor ancestor. Push normally without force, then confirm
`HEAD == origin/main`.
