# Afghanistan LLM Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, timed Afghanistan sentence-labeling pipeline using a pinned Qwen3.6 27B quantization through vLLM with a llama.cpp fallback.

**Architecture:** Deterministic prompt and validation code lives in the installable package; a bounded runner talks to an OpenAI-compatible inference server and persists identity-bound Parquet checkpoints. Grid'5000 scripts own CUDA allocation and server startup, while publication remains a separate validated step.

**Tech Stack:** Python 3.12, PyArrow, stdlib HTTP/JSON, pytest, vLLM or llama.cpp on Grid'5000.

---

### Task 1: Label contracts and deterministic prompt

**Files:**
- Create: `src/osm_polygon_sentence_relevance/labeling/contracts.py`
- Create: `src/osm_polygon_sentence_relevance/labeling/prompt.py`
- Test: `tests/unit/labeling/test_prompt.py`

- [ ] Write tests asserting the exact system instructions, independent labels, all OSM tags sorted without filtering, excluded fields, final section title only, and deterministic output.
- [ ] Run the focused tests and capture failures caused by the missing package.
- [ ] Implement immutable enums/contracts and prompt construction minimally.
- [ ] Run the focused tests and retain GREEN evidence.

### Task 2: Strict structured-response validation

**Files:**
- Create: `src/osm_polygon_sentence_relevance/labeling/validation.py`
- Test: `tests/unit/labeling/test_validation.py`

- [ ] Write failing tests for malformed JSON, extra/missing fields, invalid enums/reasons, non-exact evidence, excessive evidence, and valid empty evidence.
- [ ] Run the tests and verify RED.
- [ ] Implement strict parsing with a fixed JSON schema and target-sentence evidence checks.
- [ ] Run the tests and verify GREEN.

### Task 3: Atomic identity-bound checkpoints

**Files:**
- Create: `src/osm_polygon_sentence_relevance/labeling/checkpoint.py`
- Test: `tests/unit/labeling/test_checkpoint.py`

- [ ] Write failing tests for atomic publication, modes, hashes, schema, duplicate IDs, corrupt metadata, identity mismatch, and reusable completed IDs.
- [ ] Run the tests and verify RED.
- [ ] Implement checkpoint metadata, Parquet storage, validation, and atomic filesystem operations.
- [ ] Run the tests and verify GREEN.

### Task 4: OpenAI-compatible engine adapter

**Files:**
- Create: `src/osm_polygon_sentence_relevance/labeling/engine.py`
- Test: `tests/unit/labeling/test_engine.py`

- [ ] Write failing tests using an in-process HTTP server for batch requests, JSON-schema forwarding, ordering, timeouts, HTTP errors, and response-count mismatch.
- [ ] Run the tests and verify RED.
- [ ] Implement the inference protocol and stdlib HTTP adapter without SDK dependencies.
- [ ] Run the tests and verify GREEN.

### Task 5: Resumable runner, graceful stop, and timing

**Files:**
- Create: `src/osm_polygon_sentence_relevance/labeling/runner.py`
- Test: `tests/unit/labeling/test_runner.py`

- [ ] Write failing tests for bounded batches, checkpoint resume, identity rejection, SIGTERM/SIGINT stop semantics, deterministic order, progress heartbeat, rolling throughput, ETA, and final timing phases.
- [ ] Run the tests and verify RED.
- [ ] Implement the minimal runner with injected clock, engine, and stop signal.
- [ ] Run the tests and verify GREEN.

### Task 6: CLI, Afghanistan finalization, factual card, and publication

**Files:**
- Modify: `src/osm_polygon_sentence_relevance/application/cli.py`
- Create: `src/osm_polygon_sentence_relevance/labeling/finalization.py`
- Create: `src/osm_polygon_sentence_relevance/labeling/card.py`
- Create: `src/osm_polygon_sentence_relevance/labeling/publication.py`
- Test: `tests/integration/test_cli_labeling.py`

- [ ] Write failing integration tests for explicit immutable input/model revisions, Afghanistan-only guard, fresh final output, resumable work directory, exact input/output accounting, automatic data-derived card statistics/plots, refusal to publish partial runs, atomic publication, and independent commit readback.
- [ ] Run the tests and verify RED.
- [ ] Add the CLI surface, deterministic labeled-Parquet/manifest finalization, concise factual card/plots, and validated publication command using the existing Hub publisher.
- [ ] Run the tests and verify GREEN.

### Task 7: Grid'5000 vLLM-first operation with llama.cpp fallback

**Files:**
- Create: `scripts/grid5000/run_afghanistan_labeling.sh`
- Create: `scripts/grid5000/run_afghanistan_labeling_job.sh`
- Create: `scripts/grid5000/submit_afghanistan_labeling.sh`
- Test: `tests/unit/scripts/test_afghanistan_labeling_sh.py`

- [ ] Write failing static and executable shell tests for OAR/CUDA guards, pinned model/file revisions, Q4_K_M default, vLLM startup canary, llama.cpp fallback, fixed-engine run identity, cleanup, signals, no CPU/MPS execution, and real exit-code propagation.
- [ ] Run the tests and verify RED.
- [ ] Implement the three small scripts by following the existing production Grid'5000 conventions.
- [ ] Run shell syntax and focused tests and verify GREEN.

### Task 8: Public documentation and complete verification

**Files:**
- Modify: `README.md`
- Modify: `docs/guides/grid5000.md`
- Modify: `docs/reference/cli.md`
- Modify: `docs/architecture/overview.md`
- Modify: `CHANGELOG.md`
- Modify: `scripts/verify_distribution.py`
- Test: `tests/unit/contracts/test_documentation_consistency.py`
- Test: `tests/unit/contracts/test_distribution_verifier.py`

- [ ] Write failing documentation/distribution tests for the new public command, resume contract, engine fallback, timing output, and sdist script modes.
- [ ] Run tests and verify RED.
- [ ] Update concise public documentation and distribution verification.
- [ ] Run focused tests, full pytest with coverage, Ruff format/check, mypy, shell syntax, build, distribution verification, and `git diff --check`.
