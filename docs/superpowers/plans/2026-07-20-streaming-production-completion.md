# Streaming Production Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process all 291 immutable input shards with SaT on Grid'5000 CUDA nodes, preserve verified remote checkpoints, produce the canonical three-artifact dataset, and publish and verify it on Hugging Face.

**Architecture:** A compute job lists shard keys from the pinned Hub revision, processes one shard at a time through the canonical single-shard pipeline, and atomically uploads each verified checkpoint to a staging branch in the existing output dataset. Finalization validates the shard-key namespace invariant, finalizes each shard independently, streams sorted partitions into one Parquet file with exact incremental statistics, publishes the three artifacts, and verifies the Hub readback.

**Tech Stack:** Python 3.12, PyArrow, Hugging Face Hub, Bash/OAR, PyTorch/wtpsplit SaT, pytest, Ruff, mypy.

---

### Task 1: Characterize and repair the streaming runtime

**Files:**
- Modify: `scripts/streaming/driver.py`
- Modify: `scripts/streaming/downloader.py`
- Modify: `scripts/streaming/offload.py`
- Test: `tests/unit/scripts/streaming/test_driver.py`
- Test: `tests/unit/scripts/streaming/test_offload.py`

- [ ] Add failing tests for full-shard enumeration, remote-first reuse, optional Wikivoyage probing, exact staging revision use, metadata size compatibility, and strict post-readback cleanup.
- [ ] Run the focused tests and confirm failures identify the current defects.
- [ ] Implement a real `stream-build` loop, exact remote validation, atomic checkpoint upload, and remote-first resume.
- [ ] Run focused tests and confirm green.

### Task 2: Repair Grid'5000 launchers

**Files:**
- Modify: `scripts/grid5000/submit_streaming_build.sh`
- Modify: `scripts/grid5000/run_streaming_build_job.sh`
- Modify: `scripts/grid5000/run_streaming_build.sh`
- Test: `tests/unit/scripts/test_submit_streaming_build_sh.py`
- Test: `tests/unit/scripts/test_run_streaming_build_job_sh.py`
- Test: `tests/unit/scripts/test_run_streaming_build_sh.py`

- [ ] Add failing executable tests proving the locked interpreter, online Hub checkpoint I/O, GPU preflight, exact argument propagation, and multi-shard driver invocation.
- [ ] Run them red.
- [ ] Implement the minimal safe launchers by following the committed working smoke/build launcher patterns.
- [ ] Run them green and validate each script with `bash -n`.

### Task 3: Implement bounded deterministic finalization

**Files:**
- Modify: `scripts/streaming/finalization.py`
- Modify: `src/osm_polygon_sentence_relevance/output/manifest.py`
- Add: `src/osm_polygon_sentence_relevance/output/streaming.py`
- Modify: `src/osm_polygon_sentence_relevance/output/validation.py`
- Test: `tests/unit/scripts/streaming/test_finalization.py`
- Add: `tests/unit/output/test_streaming_export.py`

- [ ] Add failing tests for crash-safe partitions, shard-prefix isolation, bounded Parquet writing, exact incremental statistics, canonical card/manifest rendering, and parity with the in-memory exporter.
- [ ] Run them red.
- [ ] Implement per-shard finalization plus ordered ParquetWriter output and SQLite-backed exact distinct statistics; never materialize the complete table.
- [ ] Extend validation with a bounded row-batch statistics path and verify parity.
- [ ] Run focused tests green.

### Task 4: Verify and publish implementation

**Files:**
- Modify: `docs/guides/grid5000.md`
- Modify: `CHANGELOG.md`
- Modify: `scripts/verify_distribution.py`
- Modify: distribution and documentation tests as required.

- [ ] Document the exact canary, resume, progress, finalization, and publication commands.
- [ ] Run Ruff format/check, mypy, full pytest with coverage, shell syntax checks, build, distribution verification, and `git diff --check`.
- [ ] Review the staged set to exclude `scripts/audit/` and `scripts/audit_upstream_correction.py`.
- [ ] Commit and push the implementation to `origin/main`.

### Task 5: Execute Grid'5000 production run

- [ ] Remove only obsolete project-generated Grid'5000 storage and prepare a clean pinned checkout, lean model cache, work/log roots, and secure Hub authentication.
- [ ] Run one small CUDA canary and verify GPU placement, checkpoint upload/readback, cleanup, and remote-first resume.
- [ ] Submit resumable production allocations until authoritative remote progress is 291/291.
- [ ] Run bounded finalization, validate all three artifacts, publish them to the existing output dataset `main`, and independently verify the published revision.
- [ ] Remove project-generated remote scratch only after the published readback succeeds.
