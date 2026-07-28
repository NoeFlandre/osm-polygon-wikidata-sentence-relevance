# Grid'5000 Autonomous Pipeline Operator Design

## Purpose

Provide one public command, launched from the operator's Mac, that runs the
sentence-splitting and LLM-labeling production workflows on Grid'5000 with
minimal human intervention. The command owns remote preparation, site
selection, OAR submission, progress reporting, checkpoint-aware continuation,
finalization, Hugging Face publication, independent readback validation, and
safe cleanup of pipeline-managed storage.

The implementation belongs exclusively to:

`/Users/noeflandre/osm-polygon-wikidata-sentence-relevance`

The only publication target is:

`NoeFlandre/osm-polygon-wikidata-sentence-relevance`

## Public command

The package exposes a Mac-side command named `osm-polygon-grid5000`.

Examples:

```bash
uv run osm-polygon-grid5000 run \
  --scope region \
  --region afghanistan-latest \
  --stage all
```

```bash
uv run osm-polygon-grid5000 run \
  --scope all \
  --stage split
```

The public choices are:

- `--scope region --region <canonical-shard>` processes one shard resolved
  from the immutable Hugging Face input revision.
- `--scope all` processes every discovered shard.
- `--stage split` produces and publishes the validated sentence dataset.
- `--stage label` labels an already validated sentence dataset and publishes
  the validated labeled dataset.
- `--stage all` splits and labels in one run. It does not publish an
  unnecessary intermediate sentence release; only the completed labeled
  dataset is published.

Safe automatic defaults are the primary interface. Narrow operational
overrides such as an explicit site, batch size, or llama.cpp parallelism may
be exposed when they are validated and identity-bound. They must not be
required for a normal run.

Companion commands are:

- `status`: report current durable state without mutation.
- `logs`: follow durable local and remote logs.
- `cancel`: stop an active OAR job while preserving checkpoints.
- `cleanup`: remove only pipeline-owned completed or failed remote runs.

## Architecture

The Mac-side controller is a Python state machine. Shell remains limited to
small compute-node payloads where direct process and signal control is useful.
The controller coordinates existing production splitting, labeling,
checkpoint, validation, and publication modules; it does not duplicate model
logic.

Each run has a deterministic identity derived from:

- requested scope and stage;
- immutable input revisions and file hashes;
- producing source commit;
- splitting model and revision;
- labeling model, tokenizer, quantization, and file hashes;
- prompt version;
- batch and runtime configuration;
- output repository.

Rerunning the same command loads the same state and resumes. A conflicting
identity fails closed rather than reusing incompatible checkpoints.

The state progression is:

```text
resolve inputs
→ select site
→ reclaim managed storage
→ prepare remote environment
→ submit allocation
→ monitor and stream logs
→ validate checkpoints
→ continue after walltime when required
→ finalize
→ validate publication directory
→ publish one atomic commit to main
→ independently download and validate
→ mark complete
```

## Module boundaries

The operator package is divided by responsibility:

- `operator/cli.py`: public parsing and terminal rendering.
- `operator/config.py`: validated configuration and run identity.
- `operator/state.py`: atomic local state and append-only events.
- `operator/ssh.py`: bounded SSH execution, reconnection, and log transport.
- `operator/sites.py`: Grid'5000 capability and availability selection.
- `operator/storage.py`: managed-path inventory and safe reclamation.
- `operator/oar.py`: OAR submission, monitoring, exit classification, and
  continuation.
- `operator/staging.py`: checkout, environment, input, model, and tokenizer
  preparation.
- `operator/split.py`: adapter over the existing streaming split workflow.
- `operator/label.py`: prompt sizing and adapter over the labeling workflow.
- `operator/publication.py`: validation, atomic publication, and readback.
- `operator/controller.py`: state transitions only.

Files remain focused enough to understand and test independently. Existing
low-level scripts may remain private payloads while they provide tested
behavior. Obsolete launchers are removed only after parity tests prove the
new controller covers their production contract.

## Local state and logs

All Mac-side data is stored under the single operator-managed root:

`/Volumes/Seagate M3/projects/osm-polygon-wikidata-sentence-relevance`

The controller never falls back to the Mac's internal disk for datasets,
models, publication artifacts, readback files, or run logs. It canonicalizes
the root, rejects symlinks and unexpected ownership, verifies free space
before transfer, and stops clearly when the volume is unavailable.

The Git repository contains only source and lightweight documentation. Each
run gets a directory below the external-volume root containing:

- atomic current state;
- immutable run identity;
- append-only JSONL events;
- a plain-text operator log;
- OAR job history and exit reasons;
- local copies of validation evidence;
- publication commit and readback evidence when complete.

Large downloads, model artifacts, finalized datasets, and independent Hub
readbacks also remain below this root. Temporary local files use an
allocation-specific subdirectory inside the same root and are atomically
renamed or safely removed; they do not use `/tmp` or the system temporary
directory.

The terminal displays live, concise progress:

```text
[site] selected nancy: A100 40 GB, managed storage ready
[split] afghanistan-latest: 18,304 / 54,462 sentences (33.6%)
[label] 13,952 / 54,462 rows (25.6%) — 0.834 rows/s — ETA 13h 29m
[oar] allocation ended at walltime; checkpoint batch-000108 verified
[oar] continuation submitted; reused 13,952 labels
[publish] local validation passed; uploading one atomic commit
[done] immutable Hugging Face readback validated
```

The controller follows non-interactive OAR jobs, tails remote logs, and reads
structured progress and timing files. It reconnects after transient SSH
failure. If the Mac process stops or sleeps, the same command resumes
monitoring from durable state. Credentials, prompts, and raw model responses
are never printed.

## Site selection

The controller queries available Grid'5000 sites before staging large files.
It selects a site that satisfies:

- required CUDA capability and GPU memory;
- enough managed persistent storage for durable inputs and checkpoints;
- allocation-local scratch for bounded transient files;
- scheduler availability compatible with the requested walltime.

Selection prefers the earliest schedulable compatible resource. An explicit
site override is permitted for diagnosis, but is not the normal workflow.

If a chosen site becomes unsuitable before durable work begins, the
controller may choose another site. Once identity-bound checkpoints exist,
site movement is allowed only when the checkpoint transport and readback
validation prove continuity.

## Managed storage

The controller owns one clearly named root under the Grid'5000 account. It
may automatically delete only:

- completed pipeline runs whose publication readback succeeded;
- failed pipeline runs with no unique reusable checkpoints;
- pipeline-created package environments;
- pipeline-created model, tokenizer, input, and package caches that can be
  reproduced from immutable identities;
- expired pipeline logs according to the documented retention rule.

It never deletes `.ssh`, shell configuration, authentication state, files
outside the managed root, or data not recorded in the pipeline inventory.
Deletion candidates are canonicalized, ownership-checked, symlink-rejected,
and recorded before removal. If managed cleanup cannot provide enough space,
the run stops safely rather than deleting unrelated files.

Large input data is streamed per shard into allocation-local scratch. A
verified remote checkpoint is durable before scratch eviction. The workflow
does not mirror the complete input dataset on a frontend.

## Splitting workflow

The controller resolves the current input branch to an immutable Hugging Face
revision and discovers the canonical shard set. It selects one requested
region or all regions.

For each shard it:

1. downloads the required source files into allocation-local scratch;
2. verifies immutable identity and hashes;
3. runs `sat-12l-sm` on explicit CUDA;
4. applies the existing conservative residual-boundary repair;
5. validates and atomically offloads the shard checkpoint;
6. verifies readback before evicting scratch.

Allocation continuation reuses verified shard checkpoints. Finalization is
deterministic for the selected scope and preserves the existing global
deduplication and sentence-context contracts.

## Labeling workflow

The labeling stage consumes the exact validated sentence artifact produced or
selected by the run. It retains the existing deterministic prompt, all OSM
tags, strict closed response schema, reason-label consistency, exact evidence
validation, bounded repair, atomic batch checkpoints, and complete-ID
finalization contract.

Before production submission, the controller builds the selected prompt
shapes with the pinned tokenizer and measures their token requirements. It
includes response allowance and server overhead, then chooses a supported
per-slot context and parallel-slot count that fit the selected GPU. Prompt
context is never silently truncated and OSM tags are never dropped.

The observed failure where a 7,265-token request reached a server configured
for 4,096 tokens becomes a preflight rejection rather than a runtime failure.
For that workload the plan must provide at least an 8,192-token slot or a
larger validated value. Reduced parallelism is preferred over invalid context.
The chosen values are persisted in the run identity, so continuation cannot
silently change them.

Validated batches are checkpointed atomically. Expected walltime termination
causes automatic continuation with the same identity and work directory.

## Failure policy

Failures are classified rather than retried indiscriminately:

- Expected walltime: verify the last checkpoint and submit a continuation.
- Transient SSH or scheduler read failure: bounded retry with backoff.
- Insufficient compatible GPU before staging: try another site.
- Insufficient managed storage: reclaim only eligible managed paths, then
  reevaluate.
- Preflight identity, context, schema, or resource failure: stop before
  inference.
- Deterministic application failure: preserve state and stop with the first
  causal error.
- Corrupt or mismatched checkpoint: fail closed and preserve forensic data.

Retry counts are bounded. The controller never loops indefinitely, switches to
CPU or MPS, publishes partial data, or silently relaxes validation.

## Finalization, dataset card, and publication

Publication targets only:

`https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance`

and only branch `main`.

The final publication directory is built from finalized data. The dataset card
is deterministic and factual: every count, percentage, distribution, timing,
scope statement, model identity, hash, plot, and example row is derived from
the finalized Parquet, run identity, and validated timing artifacts. No result
number is hand-entered.

The card is concise, professional, public-facing, Apache-2.0 licensed, and
explains:

- selected geographic scope;
- immutable input provenance;
- sentence-splitting method and model;
- normalization, boundary repair, context, and deduplication;
- labeling questions, prompt context, model, quantization, and runtime;
- model-generated-label limitations;
- factual distributions and runtime;
- GitHub source at the producing commit.

Publication requires:

1. complete scope and exact sentence-ID coverage;
2. schema, accounting, hash, card, plot, and example validation;
3. one atomic Hugging Face commit to `main`;
4. independent download from the returned immutable commit;
5. the same full validator passing on the readback bytes.

Only then is the run marked complete and eligible for managed cleanup.

## Testing and acceptance

All behavior is implemented with strict RED→GREEN TDD. Tests cover:

- command parsing and deterministic identity;
- atomic state and restart from every state;
- fake SSH and OAR transitions;
- live terminal logs, reconnection, and redaction;
- expected-walltime continuation;
- deterministic failure stopping;
- multi-site selection and failover;
- managed-storage cleanup and protected-path refusal;
- one-region and all-shard discovery;
- `split`, `label`, and `all` transitions;
- prompt token sizing, including a 7,265-token regression fixture;
- context/parallelism planning under GPU-memory constraints;
- checkpoint identity, corruption, and resume;
- factual deterministic card generation;
- atomic publication and immutable readback;
- a complete fake Grid'5000 run from one Mac command.

Repository acceptance includes formatting, lint, strict typing, the full test
suite and coverage gate, distribution build and verification, shell syntax,
and `git diff --check`.

Operational acceptance uses an Afghanistan non-publishing canary. It must prove
automatic site selection, remote preparation, live terminal progress,
explicit CUDA inference, checkpoint creation, interruption, automatic
continuation, checkpoint reuse, and final local validation. Publication is
tested with fakes until a separately scoped complete production run reaches
the real publication boundary.

## Non-goals

- A general Grid'5000 workflow engine.
- Arbitrary shell execution.
- Supporting direct local `.osm.pbf` input.
- CPU, MPS, or Mac model inference.
- Repository creation or token arguments.
- Deleting storage outside the pipeline-managed root.
- Reimplementing the existing segmentation, labeling, or publication logic.
