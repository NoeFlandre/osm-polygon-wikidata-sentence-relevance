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
validated step. A persistent, identity-bound ledger records those verified
readbacks, so later short allocations skip repeated Hub inspection while final
publication still validates every checkpoint independently. Within a shard,
the manifest remains authoritative after a hard stop: only one canonical
immediate-next orphan batch is discarded and recomputed; ambiguous entries
fail closed.

## Tests

`tests/unit/scripts/streaming/` covers ceilings, identity, sequential runs,
resume, offload, and cleanup.
