# Labeling

## Responsibility

Assigns land-use or land-cover and polygon-relevance labels with a pinned local
LLM runtime. V2 selection can bound a run with deterministic H3, language, and
OSM-primary-tag strata; the earlier Afghanistan V1 artifact remains separate.

## Public entry points

The labeling CLI and operator use the runner, checkpoint, finalization,
validation, and publication APIs.

## Internal structure

Prompting, inference, repair, checkpointing, runtime validation, finalization,
and publication are isolated under `labeling/`.

`checkpoint_mirror.py` is an optional operational boundary: it queues each
validated local batch for one background Hugging Face staging upload without
making the runner depend on network availability.

## Invariants

Identity binds input, model, prompt, engine, and concurrency. Batches are
immutable and resumable. Publication requires complete validation.

## Tests

`tests/unit/labeling/` covers prompt fidelity, failures, resume, accounting,
publication, and generated cards.
