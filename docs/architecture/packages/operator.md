# Operator

## Responsibility

Runs production from a Mac, selects policy-compliant Grid'5000 resources, and
resumes durable remote work without duplicating jobs.

## Public entry points

`osm-polygon-grid5000 run` starts a run; `osm-polygon-grid5000 resume`
reattaches by durable run ID.

## Internal structure

Configuration, state, SSH, scheduling, storage, relay, monitoring, sampling
policy, and completion are separate modules under `operator/`. The stable
public `operator/config.py` facade delegates to `_config/`: `defaults.py` owns
immutable defaults, `enums.py` owns scope and stage values, `validation.py`
owns parsing and validation, and `models.py` owns the immutable configuration
and run-identity dataclasses. Dependencies flow from models to those leaf
modules, never back from validation into models.

## Invariants

Inference requires an allocated CUDA node. State and job IDs are persisted
before monitoring. Cleanup cannot escape managed roots. Policy and quota checks
fail closed.

## Tests

`tests/unit/operator/` covers each module and orchestration seam.
