# Labeling

## Responsibility

The historical V1 lane assigns independent land-use/land-cover and
polygon-relevance labels with a pinned local LLM runtime. V2 is a separate
worldwide lane: deterministic area-bucket and H3 strata select the rows, then
one binary `place_relevance` label asks whether the target place is described
visually or geographically. The release stores the first-token yes/no scores
without generated explanations. The Afghanistan V1 artifact remains separate.

## Public entry points

The labeling CLI and operator use the runner, checkpoint, finalization,
validation, and publication APIs.

## Internal structure

Prompting, inference, repair, checkpointing, runtime validation, finalization,
and publication are isolated under `labeling/`.

`checkpoint_mirror.py` is an optional operational boundary: it queues each
validated local batch for one background Hugging Face staging upload without
making the runner depend on network availability.

`v2_input.py` is the V2-only input boundary. It reads the completed V1-schema
split output, fetches the pinned upstream polygon metadata one region at a time,
canonicalizes area buckets, and atomically writes the separate V2 sampling
input. V1 output files are never rewritten.

`v2_resumable_sampling.py` keeps the expensive V2 planning and candidate scans
in a persistent SQLite ledger. Finalized shards are verified and staged one at
a time, then reused by identity on the next allocation. A terminated job can
therefore resume without restarting completed shards or accepting an incomplete
output. Candidate retention commits after each Parquet batch, so an interruption
inside a large shard redoes at most that bounded batch.

## Invariants

Identity binds input, model, prompt, engine, and concurrency. Batches are
immutable and resumable. Publication requires complete validation.

## Tests

`tests/unit/labeling/` covers prompt fidelity, failures, resume, accounting,
publication, and generated cards.
