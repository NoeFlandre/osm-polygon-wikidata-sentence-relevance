# Application

## Responsibility

Coordinates discovery, loading, joining, segmentation, checkpoint reuse,
finalization, and export.

## Public entry points

`run_pipeline` is the programmatic entry point; the build CLI exposes it as
`osm-polygon-sentence-relevance`.

## Internal structure

`application/` owns orchestration and CLI plumbing; `_checkpoint/` owns secure,
identity-bound checkpoint storage.

## Invariants

Input, output, and work paths cannot overlap. Shards use canonical order,
checkpoints validate before reuse, and output installation is atomic.

## Tests

`tests/unit/application/` covers orchestration, corruption, restart, and CLI
validation.
