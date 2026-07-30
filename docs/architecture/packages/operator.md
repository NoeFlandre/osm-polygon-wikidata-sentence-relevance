# Operator

## Responsibility

Runs production from a Mac, selects policy-compliant Grid'5000 resources, and
resumes durable remote work without duplicating jobs.

## Public entry points

`osm-polygon-grid5000 run` starts a run; `osm-polygon-grid5000 resume`
reattaches by durable run ID.

## Internal structure

Configuration, state, SSH, scheduling, storage, relay, monitoring, and
completion are separate modules under `operator/`.

## Invariants

Inference requires an allocated CUDA node. State and job IDs are persisted
before monitoring. Cleanup cannot escape managed roots. Policy and quota checks
fail closed.

## Tests

`tests/unit/operator/` covers each module and orchestration seam.
