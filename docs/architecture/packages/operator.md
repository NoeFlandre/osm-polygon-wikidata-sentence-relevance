# Operator

## Responsibility

Runs production from a Mac, selects policy-compliant Grid'5000 resources, and
resumes durable remote work without duplicating jobs.

## Public entry points

`osm-polygon-grid5000 run` starts a run; `osm-polygon-grid5000 resume`
reattaches by durable run ID.

## Internal structure

Configuration, preflight, state, recovery, SSH, scheduling, storage, relay,
monitoring, sampling policy, and completion are separate modules under
`operator/`. The stable
public `operator/config.py` facade delegates to `_config/`: `defaults.py` owns
immutable defaults, `enums.py` owns scope and stage values, `validation.py`
owns parsing and validation, and `models.py` owns the immutable configuration
and run-identity dataclasses. Dependencies flow from models to those leaf
modules, never back from validation into models.

`operator/preflight.py` owns the checks that must happen before a run can
mutate remote state: the local checkout must be a clean immutable commit, the
Hub input revision is resolved once, the remote home path is validated, and
Grid'5000 usage-policy checks fail closed. The CLI keeps compatibility aliases
for these functions while the implementation remains isolated and directly
tested.

`operator/recovery.py` contains the small state-machine decisions used by
resume: whether a recorded allocation may be reattached, whether a terminal
transition is valid for the current phase, and how recovery-attempt counters
advance. Remote inspection and submission stay in the CLI orchestration layer.

## Invariants

Inference requires an allocated CUDA node. State and job IDs are persisted
before monitoring. Cleanup cannot escape managed roots. Policy and quota checks
fail closed.

## Tests

`tests/unit/operator/` covers each module and orchestration seam.
