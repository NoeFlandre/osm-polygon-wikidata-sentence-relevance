# Grid'5000 Autonomous Pipeline Operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one Mac-side command that autonomously runs, resumes, validates, and publishes region-scoped or complete sentence-splitting and LLM-labeling workflows on Grid'5000.

**Architecture:** A typed Python state machine coordinates immutable input resolution, Grid'5000 site selection, managed remote storage, SSH, OAR allocations, existing compute payloads, checkpoints, finalization, and atomic Hugging Face publication. Durable state and live logs remain under `/Volumes/Seagate M3/projects/osm-polygon-wikidata-sentence-relevance`; existing model logic stays authoritative.

**Tech Stack:** Python 3.12, argparse, subprocess/OpenSSH, JSON/JSONL, PyArrow, Hugging Face Hub, OAR, pytest, mypy, ruff, existing Bash compute payloads.

---

## File structure

Create the package `src/osm_polygon_sentence_relevance/operator/`:

- `config.py`: enums, validated options, fixed local data root, run identity.
- `state.py`: atomic state snapshots and append-only events.
- `ssh.py`: typed SSH command/result boundary and reconnect-safe log reads.
- `sites.py`: site probes, resource requirements, deterministic selection.
- `storage.py`: pipeline-owned remote inventory and safe reclamation.
- `oar.py`: job submission, status parsing, exit classification.
- `token_budget.py`: prompt token sizing and GPU-safe llama.cpp runtime plan.
- `staging.py`: remote checkout/environment/artifact preparation.
- `workflows.py`: adapters for split, label, finalize, and publish commands.
- `controller.py`: resumable state transitions.
- `cli.py`: `run`, `status`, `logs`, `cancel`, and `cleanup`.
- `__init__.py`: deliberately small public exports.

Create focused tests under `tests/unit/operator/` and one fake-backend integration
test under `tests/integration/test_grid5000_operator.py`.

Modify:

- `pyproject.toml`: register `osm-polygon-grid5000`.
- `scripts/streaming/driver.py`: accept an exact shard selector.
- `scripts/grid5000/run_streaming_build.sh`: propagate exact shard selection.
- `scripts/grid5000/run_afghanistan_labeling.sh`: accept validated per-slot
  context instead of fixing it at 4096.
- labeling runtime and identity modules: support token-budget-derived context.
- distribution verifier, README, CLI reference, Grid'5000 guide, architecture,
  reproducibility guide, and changelog.

## Task 1: Public configuration and deterministic identity

**Files:**
- Create: `src/osm_polygon_sentence_relevance/operator/config.py`
- Create: `src/osm_polygon_sentence_relevance/operator/__init__.py`
- Test: `tests/unit/operator/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_region_all_identity_is_deterministic() -> None:
    first = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="all",
        source_commit="a" * 40,
    )
    second = OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage="all",
        source_commit="a" * 40,
    )
    assert first.run_id == second.run_id
    assert first.data_root == Path(
        "/Volumes/Seagate M3/projects/osm-polygon-wikidata-sentence-relevance"
    )


@pytest.mark.parametrize(
    ("scope", "region"),
    [("region", None), ("all", "afghanistan-latest")],
)
def test_scope_region_relationship_is_strict(scope: str, region: str | None) -> None:
    with pytest.raises(ValueError):
        OperatorConfig.build(
            scope=scope, region=region, stage="split", source_commit="a" * 40
        )
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest tests/unit/operator/test_config.py -q
```

Expected: import failure because the operator package does not exist.

- [ ] **Step 3: Implement immutable configuration**

Define `Scope`, `Stage`, `OperatorConfig`, `Grid5000Requirements`, and
`RunIdentity`. Validate 40-hex revisions, canonical shard names, repository IDs,
 positive integers, and stage/scope relationships. Derive `run_id` as the first
20 hex characters of SHA-256 over canonical sorted JSON. Fix the default data
root and output repository as constants; do not read them from ambient shell
variables.

- [ ] **Step 4: Run tests and verify GREEN**

Run the focused file, then:

```bash
uv run mypy src/osm_polygon_sentence_relevance/operator
```

- [ ] **Step 5: Commit**

```bash
git add src/osm_polygon_sentence_relevance/operator tests/unit/operator/test_config.py
git commit -m "Add autonomous operator configuration"
```

## Task 2: Durable local state and live event logs

**Files:**
- Create: `src/osm_polygon_sentence_relevance/operator/state.py`
- Create: `tests/unit/operator/test_state.py`

- [ ] **Step 1: Write RED tests for atomic state and restart**

Test that `StateStore.create(identity)` creates mode-0700 run directories,
mode-0600 `state.json` and `events.jsonl`, rejects a different identity, and
recovers the last valid state after a simulated temporary-file interruption.
Test that event records contain UTC time, state, message, and structured facts
without prompts, responses, or token-like fields.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest tests/unit/operator/test_state.py -q
```

- [ ] **Step 3: Implement minimal state storage**

Use `os.open`, `os.fchmod`, `fsync`, `os.replace`, and parent-directory
`fsync`. Expose:

```python
class StateStore:
    def load_or_create(self, identity: RunIdentity) -> RunState: ...
    def transition(self, expected: RunPhase, target: RunPhase, **facts: object) -> RunState: ...
    def append_event(self, level: str, message: str, **facts: object) -> None: ...
```

No SQLite or workflow framework is introduced.

- [ ] **Step 4: Verify GREEN and commit**

```bash
uv run pytest tests/unit/operator/test_state.py -q
git add src/osm_polygon_sentence_relevance/operator/state.py tests/unit/operator/test_state.py
git commit -m "Add durable operator run state"
```

## Task 3: SSH transport and terminal log streaming

**Files:**
- Create: `src/osm_polygon_sentence_relevance/operator/ssh.py`
- Create: `tests/unit/operator/test_ssh.py`

- [ ] **Step 1: Write RED transport tests**

Use an injected subprocess runner. Assert exact argv includes
`BatchMode=yes`, `ForwardAgent=no`, finite connect timeout, and one remote
`bash -lc` command. Test bounded retry only for connection failures, no retry
for remote exit 2, byte-offset log continuation, and redaction of keys named
`token`, `authorization`, `prompt`, and `response`.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest tests/unit/operator/test_ssh.py -q
```

- [ ] **Step 3: Implement OpenSSH boundary**

Expose `SshClient.run`, `SshClient.read_since`, and `SshError`. Use
`subprocess.run` with explicit argv, input, timeout, and captured text. Never
use `shell=True`, `eval`, agent forwarding, or password/token arguments.

- [ ] **Step 4: Verify GREEN and commit**

```bash
uv run pytest tests/unit/operator/test_ssh.py -q
git add src/osm_polygon_sentence_relevance/operator/ssh.py tests/unit/operator/test_ssh.py
git commit -m "Add bounded Grid5000 SSH transport"
```

## Task 4: Automatic site selection and managed storage reclamation

**Files:**
- Create: `src/osm_polygon_sentence_relevance/operator/sites.py`
- Create: `src/osm_polygon_sentence_relevance/operator/storage.py`
- Create: `tests/unit/operator/test_sites.py`
- Create: `tests/unit/operator/test_storage.py`

- [ ] **Step 1: Write RED site-selection tests**

Fixtures describe Nancy, Nantes, and Rennes probes. Require GPU memory,
CUDA capability, persistent free bytes, and expected queue delay. Assert the
selector chooses the compatible site with the earliest predicted start, uses
site name as a stable tiebreaker, and explains why incompatible sites were
rejected.

- [ ] **Step 2: Write RED cleanup tests**

Build a fake managed root containing active, completed, failed, cached, foreign,
symlinked, and protected entries. Assert only inventory-recorded completed or
failed pipeline-owned entries are eligible, oldest first. Assert canonical
paths outside the managed root, symlinks, wrong ownership, active checkpoints,
`.ssh`, and shell configuration can never be returned as deletion candidates.

- [ ] **Step 3: Verify RED**

```bash
uv run pytest tests/unit/operator/test_sites.py tests/unit/operator/test_storage.py -q
```

- [ ] **Step 4: Implement selector and cleanup planner**

Use typed `SiteProbe`, `SiteSelection`, `ManagedEntry`, and `CleanupPlan`.
Separate planning from mutation. `execute_cleanup` revalidates every path
immediately before removal and records reclaimed bytes in the event log.

- [ ] **Step 5: Verify GREEN and commit**

```bash
uv run pytest tests/unit/operator/test_sites.py tests/unit/operator/test_storage.py -q
git add src/osm_polygon_sentence_relevance/operator/sites.py \
  src/osm_polygon_sentence_relevance/operator/storage.py \
  tests/unit/operator/test_sites.py tests/unit/operator/test_storage.py
git commit -m "Add Grid5000 site and storage planning"
```

## Task 5: OAR job lifecycle and continuation

**Files:**
- Create: `src/osm_polygon_sentence_relevance/operator/oar.py`
- Create: `tests/unit/operator/test_oar.py`

- [ ] **Step 1: Write RED scheduler tests**

Cover accepted numeric job IDs, queued/running/terminated states, expected
walltime, non-zero payload error, scheduler cancellation, missing job, and one
submission per transition. Encode job `6807004` as a fixture with progress and
an expected continuation decision; encode context-overflow job `6808797` as a
deterministic failure that must not be resubmitted.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest tests/unit/operator/test_oar.py -q
```

- [ ] **Step 3: Implement typed OAR adapter**

Expose:

```python
class OarClient:
    def submit(self, request: SubmissionRequest) -> int: ...
    def status(self, job_id: int) -> JobStatus: ...
    def cancel(self, job_id: int) -> None: ...

def classify_exit(status: JobStatus, checkpoint: CheckpointFacts) -> ExitClass: ...
```

Only `EXPECTED_WALLTIME` and explicitly transient scheduler read failures are
automatic continuations. Application failures stop.

- [ ] **Step 4: Verify GREEN and commit**

```bash
uv run pytest tests/unit/operator/test_oar.py -q
git add src/osm_polygon_sentence_relevance/operator/oar.py tests/unit/operator/test_oar.py
git commit -m "Add resumable OAR lifecycle"
```

## Task 6: Exact region selection in the streaming splitter

**Files:**
- Modify: `scripts/streaming/driver.py`
- Modify: `scripts/grid5000/run_streaming_build.sh`
- Modify: `scripts/grid5000/run_streaming_build_job.sh`
- Modify: `scripts/grid5000/submit_streaming_build.sh`
- Create: `tests/unit/scripts/streaming/test_exact_shard_selection.py`
- Modify: existing streaming launcher tests

- [ ] **Step 1: Write RED exact-shard tests**

Assert `stream-build --shard-key afghanistan-latest` processes exactly that
remote key regardless of sort position, rejects a missing key before model
construction, and conflicts with `--max-shards`. Extend fake shell tests to
prove the value is serialized through submitter, wrapper, and payload.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest tests/unit/scripts/streaming/test_exact_shard_selection.py \
  tests/unit/scripts/test_streaming_launchers.py -q
```

- [ ] **Step 3: Implement minimal selector**

Add `shard_key: str | None` to streaming configuration and CLI. Filter the
sorted remote inventory only after validating membership. Propagate an empty
sentinel for `--scope all`; do not create a second driver.

- [ ] **Step 4: Verify GREEN and commit**

Run focused streaming tests and commit the four production files plus tests.

## Task 7: Token-budget preflight and adaptive llama.cpp context

**Files:**
- Create: `src/osm_polygon_sentence_relevance/operator/token_budget.py`
- Modify: `src/osm_polygon_sentence_relevance/labeling/runtime.py`
- Modify: `src/osm_polygon_sentence_relevance/labeling/contracts.py`
- Modify: `src/osm_polygon_sentence_relevance/labeling/cli.py`
- Modify: three Afghanistan labeling shell scripts
- Create: `tests/unit/operator/test_token_budget.py`
- Modify: labeling runtime and launcher tests

- [ ] **Step 1: Write the 7,265-token RED regression**

```python
def test_planner_never_places_7265_tokens_in_4096_slot() -> None:
    plan = plan_runtime(
        max_prompt_tokens=7265,
        response_tokens=512,
        gpu_memory_mb=40000,
        max_total_context=65536,
    )
    assert plan.per_slot_context >= 8192
    assert plan.parallel * plan.per_slot_context == plan.total_context
```

Also test deterministic next-supported-context rounding, reduced parallelism
when memory is constrained, refusal when no valid plan fits, and persisted
identity mismatch on resume.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest tests/unit/operator/test_token_budget.py \
  tests/unit/labeling/test_production_runtime.py -q
```

- [ ] **Step 3: Implement sizing and propagation**

The preflight uses the pinned tokenizer to call existing `build_messages` for
the selected table and measure the maximum encoded request. Supported slot
contexts are `(4096, 8192, 12288, 16384, 32768)`. Add response allowance and
choose the highest supported parallelism whose total context and measured
model memory fit the GPU budget. Propagate explicit per-slot context through
submitter, wrapper, payload, CLI, and `RunIdentity`.

- [ ] **Step 4: Verify GREEN and commit**

Run focused labeling tests, Bash syntax, mypy, then commit.

## Task 8: Remote staging and workflow adapters

**Files:**
- Create: `src/osm_polygon_sentence_relevance/operator/staging.py`
- Create: `src/osm_polygon_sentence_relevance/operator/workflows.py`
- Create: `tests/unit/operator/test_staging.py`
- Create: `tests/unit/operator/test_workflows.py`

- [ ] **Step 1: Write RED staging tests**

Test clean detached checkout at `origin/main`, locked `uv sync`, model and
tokenizer hash verification, resumable downloads, run-root containment, and
no large Mac-internal-disk path. Test that existing compatible artifacts are
reused and mismatches are preserved then replaced safely.

- [ ] **Step 2: Write RED workflow-command tests**

Assert exact immutable arguments for region split, all split, label resume,
finalize, and publish. `stage=all` must not invoke intermediate publication.
All remote commands are argv/quoted-data objects, not interpolated shell text.

- [ ] **Step 3: Verify RED**

```bash
uv run pytest tests/unit/operator/test_staging.py \
  tests/unit/operator/test_workflows.py -q
```

- [ ] **Step 4: Implement minimal adapters**

Reuse committed compute payloads. The Python layer produces typed command
descriptions, stages only missing immutable artifacts, and validates every
result. It contains no segmentation or labeling implementation.

- [ ] **Step 5: Verify GREEN and commit**

Run focused tests and commit.

## Task 9: Controller state machine

**Files:**
- Create: `src/osm_polygon_sentence_relevance/operator/controller.py`
- Create: `tests/unit/operator/test_controller.py`

- [ ] **Step 1: Write RED transition tests**

Use fake SSH, OAR, staging, workflows, and publisher ports. Cover:

- first run through site selection and submission;
- restart while queued and running without duplicate submission;
- live progress events;
- expected-walltime continuation after checkpoint validation;
- site failover before durable work;
- deterministic application failure stop;
- split-only, label-only, and split-then-label;
- complete validation before publication;
- immutable readback before `COMPLETE`.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest tests/unit/operator/test_controller.py -q
```

- [ ] **Step 3: Implement explicit transition table**

`Controller.advance()` performs at most one externally mutating transition.
`Controller.run()` loops with bounded polling until complete, stopped, or
failed. State compare-and-swap prevents duplicate submissions after restart.

- [ ] **Step 4: Verify GREEN and commit**

Run focused tests, mypy, and commit.

## Task 10: Public CLI and terminal renderer

**Files:**
- Create: `src/osm_polygon_sentence_relevance/operator/cli.py`
- Modify: `pyproject.toml`
- Create: `tests/unit/operator/test_cli.py`
- Create: `tests/integration/test_grid5000_operator.py`

- [ ] **Step 1: Write RED CLI tests**

Test exact `--help`, region relationship, stage choices, fixed external data
root, status without mutation, logs from byte offset, cancel preserving work,
cleanup preview/execution, and stable error messages.

- [ ] **Step 2: Write RED end-to-end fake backend**

One real subprocess invocation of:

```bash
osm-polygon-grid5000 run --scope region \
  --region afghanistan-latest --stage all
```

must progress through fake site selection, two split allocations, token
preflight, two label allocations, finalization, atomic fake publication,
readback, and complete state. Assert terminal output includes counters,
throughput, ETA, continuation, and commit identity.

- [ ] **Step 3: Verify RED**

Run the two new files and confirm missing entry point failures.

- [ ] **Step 4: Implement parser and renderer**

Use argparse and dependency injection. Emit human-readable terminal lines and
the same structured facts to JSONL. Handle SIGINT by stopping local monitoring
only; do not cancel the remote job unless `cancel` is explicitly invoked.

- [ ] **Step 5: Verify GREEN and commit**

Run focused tests, reinstall locked environment, verify installed `--help`,
then commit.

## Task 11: Factual deterministic publication card

**Files:**
- Modify: existing dataset-card/profile/manifest/validation modules
- Modify: labeling finalization and publication validation
- Create: `tests/unit/output/test_operator_publication_card.py`

- [ ] **Step 1: Write RED factual-card tests**

Build a small finalized Parquet fixture and assert README equality against a
fresh render. Change each data-derived count, scope, runtime, model identity,
example row, and plot input and assert the renderer changes accordingly.
Assert no numeric result appears from operator configuration alone. Require
Apache-2.0, producing GitHub commit URL, input revision, methods, limitations,
and selected scope.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest tests/unit/output/test_operator_publication_card.py -q
```

- [ ] **Step 3: Wire operator facts into existing renderers**

Pass validated run timing and scope into the existing deterministic profile.
Do not add a second card renderer. Extend publication validation to rerender
and compare README and plot hashes before upload and after immutable readback.

- [ ] **Step 4: Verify GREEN and commit**

Run all output/publication tests and commit.

## Task 12: Documentation, packaging, and repository acceptance

**Files:**
- Modify: `README.md`
- Modify: `docs/reference/cli.md`
- Modify: `docs/guides/grid5000.md`
- Modify: `docs/guides/reproducibility.md`
- Modify: `docs/architecture/overview.md`
- Modify: `CHANGELOG.md`
- Modify: `scripts/verify_distribution.py`
- Modify: distribution-verifier and documentation consistency tests

- [ ] **Step 1: Write RED documentation and distribution tests**

Require the installed operator entry point, exact public flags, external-volume
root, one-command examples, managed-cleanup boundary, live logs, continuation,
and publication contract. Require the wheel to contain the operator package
and the sdist to contain retained compute payloads.

- [ ] **Step 2: Verify RED**

Run documentation consistency and distribution verifier tests.

- [ ] **Step 3: Update public documentation**

Lead with the one-command workflow. Move low-level scripts to an advanced
troubleshooting section. Remove stale manual operational instructions that
contradict automatic orchestration.

- [ ] **Step 4: Run the complete acceptance gate**

```bash
uv sync --locked --extra hub --extra segmentation
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -q
uv build
uv run python scripts/verify_distribution.py dist/*.whl dist/*.tar.gz
bash -n scripts/grid5000/*.sh
git diff --check
```

Require coverage at or above 95%, a clean artifact audit, and no secrets,
models, datasets, caches, or logs in Git.

- [ ] **Step 5: Commit and push**

Commit documentation/packaging, push `main` normally, and require local
`HEAD == origin/main`.

## Task 13: Real Afghanistan operational acceptance

**Files:** No source changes unless acceptance reveals a reproducible defect.

- [ ] **Step 1: Preserve the current labeling run**

Inventory job history and validate the existing `13,952` labels under the
current run identity. Do not delete or mutate checkpoints.

- [ ] **Step 2: Run a non-publishing operator canary**

Use the one Mac command with Afghanistan, explicit canary mode available only
as an advanced acceptance flag, and a context fixture exceeding 4096 tokens.
Require automatic compatible-site selection, live terminal logs, CUDA,
checkpoint creation, and no publication.

- [ ] **Step 3: Prove automatic continuation**

Use a short acceptance walltime or controlled fake deadline to end the first
allocation after a valid checkpoint. Rerun/continue through the same operator
process and prove the next allocation reuses completed IDs.

- [ ] **Step 4: Validate evidence**

Archive under the external-volume run root:

- operator state and events;
- OAR job IDs and GPU preflight;
- checkpoint hashes and counts;
- context plan;
- live-log transcript;
- final local validation.

- [ ] **Step 5: Stop before unrelated production mutation**

Do not replace the currently published dataset merely to test orchestration.
Report the exact command users can now run for region/all and split/label/all.
