# Afghanistan Sentence-Boundary Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild and publish Afghanistan with high-quality sentence boundaries and no residual version of the reported Arabic defect.

**Architecture:** Upgrade SaT to the stronger 12-layer supervised-mixture model and add a small language-aware repair after model inference. Validate the boundary invariant before export and reuse the existing resumable CUDA and atomic-publication paths.

**Tech Stack:** Python 3.12, wtpsplit 2.2.1, PyArrow, pytest, Grid'5000 OAR/CUDA, Hugging Face Hub.

---

### Task 1: Residual boundary repair

**Files:**
- Create: `src/osm_polygon_sentence_relevance/sentences/boundaries.py`
- Modify: `src/osm_polygon_sentence_relevance/sentences/segmentation.py`
- Test: `tests/unit/sentences/test_boundaries.py`

- [ ] Write failing tests for the exact Arabic regression, abbreviations,
  decimals, CJK/Indic punctuation, and ordering.
- [ ] Run the focused tests and record the expected failures.
- [ ] Implement the minimal conservative splitter and integrate it after SaT
  output, before sentence normalization/index assignment.
- [ ] Run focused tests and the existing sentence tests.

### Task 2: Production model quality

**Files:**
- Modify: `src/osm_polygon_sentence_relevance/sentences/sat.py`
- Modify: `src/osm_polygon_sentence_relevance/application/cli.py`
- Modify: `scripts/streaming/driver.py`
- Modify: `scripts/streaming/finalization.py`
- Modify: `scripts/grid5000/run_streaming_build.sh`
- Modify: public documentation and model-default tests

- [ ] Change tests to require `sat-12l-sm` and observe RED.
- [ ] Change all production defaults and operational invocations atomically.
- [ ] Update documentation to name the quality-first model and trade-off.
- [ ] Run focused CLI, streaming, placement, and documentation tests.

### Task 3: Export quality gate and factual card

**Files:**
- Modify: `src/osm_polygon_sentence_relevance/output/profile.py`
- Modify: `src/osm_polygon_sentence_relevance/output/validation_publication.py`
- Modify: `src/osm_polygon_sentence_relevance/output/_card/rendering.py`
- Test: focused output/profile/publication tests

- [ ] Add RED tests requiring zero high-confidence residual boundaries in a
  publication and a data-derived boundary-audit statement in the card.
- [ ] Compute the audit while scanning Parquet and reject violations.
- [ ] Render only profile-derived quality facts.
- [ ] Run focused output tests.

### Task 4: Full verification and release

- [ ] Run format, lint, mypy, full pytest with coverage, build, distribution,
  shell syntax, and diff checks.
- [ ] Commit and push `main`.
- [ ] Resolve the newest immutable upstream Afghanistan input and model
  revisions.
- [ ] Run/resume the fresh Afghanistan CUDA build on Grid'5000.
- [ ] Validate row identities, boundary audit, schema, hashes, card, and plots.
- [ ] Publish atomically to Hugging Face `main`.
- [ ] Independently download and validate the returned commit.
- [ ] Verify the Viewer and prove only `main` remains.
