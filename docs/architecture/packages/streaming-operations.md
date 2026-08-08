# Streaming operations

## Responsibility

Processes immutable input one shard at a time within bounded Grid'5000 storage.

## Public entry points

Drivers under `scripts/streaming/` are invoked by production launchers, not the
package API.

## Internal structure

Data-root safety, download, orchestration, remote offload, and finalization are
separate modules.

## Invariants

No full mirror is required. Independent files for the current shard are
downloaded with a bounded worker pool, while each file keeps the same
hash-verification and cleanup contract. Scratch is evicted only after
authoritative readback verifies hash and identity. Publication is a separate
validated step.

## Tests

`tests/unit/scripts/streaming/` covers ceilings, identity, sequential runs,
resume, offload, and cleanup.
