# Labeling

## Responsibility

Assigns land-use or land-cover and polygon-relevance labels with a pinned local
LLM runtime.

## Public entry points

The labeling CLI and operator use the runner, checkpoint, finalization,
validation, and publication APIs.

## Internal structure

Prompting, inference, repair, checkpointing, runtime validation, finalization,
and publication are isolated under `labeling/`.

## Invariants

Identity binds input, model, prompt, engine, and concurrency. Batches are
immutable and resumable. Publication requires complete validation.

## Tests

`tests/unit/labeling/` covers prompt fidelity, failures, resume, accounting,
publication, and generated cards.
