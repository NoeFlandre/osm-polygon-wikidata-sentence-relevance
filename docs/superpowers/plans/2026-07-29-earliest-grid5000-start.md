# Earliest Policy-Compliant Grid'5000 Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace a distant queued Grid'5000 allocation only when one compatible site demonstrably starts the same resumable workload within ten minutes.

**Architecture:** Add a pure replacement planner for time and candidate decisions, then a durable replacement coordinator used by `resume`. Keep site probing, preparation, submission, OAR inspection, and cancellation behind existing adapters. Preserve the original job until the replacement is `Running`.

**Tech Stack:** Python 3.12, frozen dataclasses, existing `StateStore`, `SshClient`, `OarClient`, Grid'5000 OAR, pytest, Ruff, mypy.

---

### Task 1: Pure replacement planning

**Files:**
- Create: `src/osm_polygon_sentence_relevance/operator/earliest_start.py`
- Create: `tests/unit/operator/test_earliest_start.py`
- Modify: `src/osm_polygon_sentence_relevance/operator/__init__.py`

- [ ] Write failing tests for parsing OAR forecasts as Europe/Paris timestamps,
  the ten-minute cutoff, daytime/weekend eligibility, filtering to factual idle
  compatible probes, and deterministic candidate ordering.
- [ ] Run:
  `uv run pytest -o addopts='' tests/unit/operator/test_earliest_start.py -q`
  and confirm failures are caused by the missing planner.
- [ ] Implement immutable `ReplacementPlan`, `ReplacementCandidate`, and pure
  functions `should_seek_replacement(...)` and `rank_replacement_candidates(...)`.
- [ ] Re-run the focused tests and require them to pass.

### Task 2: Durable replacement lifecycle

**Files:**
- Modify: `src/osm_polygon_sentence_relevance/operator/cli.py`
- Modify: `src/osm_polygon_sentence_relevance/operator/state.py`
- Create: `tests/unit/operator/test_earliest_start_lifecycle.py`

- [ ] Write failing lifecycle tests proving the fallback is not cancelled on
  preparation failure, policy failure, submission failure, queued timeout, or
  Ctrl+C.
- [ ] Write failing tests proving one replacement is submitted, its job ID is
  persisted before polling, and fallback cancellation happens only after the
  replacement reports `Running`.
- [ ] Run the lifecycle tests and verify the expected RED failures.
- [ ] Implement `_optimize_queued_start(...)` and recovery from recorded
  `replacement_*` facts using existing probe, staging, relay, OAR, policy, and
  quota adapters.
- [ ] Re-run the lifecycle tests and require GREEN.

### Task 3: Remote double-run guard

**Files:**
- Modify: `scripts/grid5000/run_afghanistan_labeling_job.sh`
- Modify: `tests/unit/scripts/test_afghanistan_labeling_launchers.py`

- [ ] Write failing executable tests proving a per-run nonblocking lock is
  acquired before starting llama.cpp or labeling and that lock contention exits
  without modifying checkpoints.
- [ ] Run the focused shell tests and verify RED.
- [ ] Add a persistent run lock using `flock -n` on a mode-0600 file within the
  managed label work directory.
- [ ] Re-run the shell tests and require GREEN.

### Task 4: Terminal reporting and documentation

**Files:**
- Modify: `src/osm_polygon_sentence_relevance/operator/cli.py`
- Modify: `docs/reference/grid5000-operator.md`
- Modify: `tests/unit/operator/test_cli.py`

- [ ] Write failing tests for factual fallback forecast, candidate rejection,
  trial deadline, replacement adoption, and fallback-retained messages.
- [ ] Implement concise progress output without fabricated ETAs.
- [ ] Document the replacement rules and Ctrl+C recovery.
- [ ] Run focused operator tests and require GREEN.

### Task 5: Acceptance and delivery

**Files:**
- Verify all changed files.

- [ ] Run `uv run ruff format --check .`.
- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run mypy src`.
- [ ] Run `uv run pytest -q` and require repository coverage at least 95%.
- [ ] Run `bash -n scripts/grid5000/*.sh`.
- [ ] Run `uv build`.
- [ ] Run `uv run python scripts/verify_distribution.py dist/*.whl dist/*.tar.gz`.
- [ ] Run `git diff --check` and audit the diff for secrets, tokens, personal
  paths, scheduler IDs, and generated runtime artifacts.
- [ ] Commit only the reviewed implementation and push normally to `main`.
- [ ] Inspect job `2895249` read-only. Do not cancel or resubmit it during code
  acceptance.

