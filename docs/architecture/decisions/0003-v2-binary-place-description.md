# V2 binary place-description labeling

## Status

Accepted for implementation.

## Decision

V2 is rebuilt as a separate worldwide release lane. V1 Afghanistan artifacts,
schemas, prompts, checkpoints, and publication paths remain unchanged.

V2 labels one property only: whether a sentence describes the physical or
geographic character of its associated place. The permitted values are `yes`
and `no`. The model receives the previous sentence, target sentence, next
sentence, page title, and section title. Polygon metadata, language, source,
coordinates, tags, and identifiers are not prompt context.

The inference request generates one token and reads its first-token log-probability
values for the exact single-token answers `yes` and `no`. The output stores the
binary label, both scores, their margin, and a two-class relative probability.
JSON generation, explanations, evidence, uncertain labels, and repair calls
are not part of V2.

The standard non-MTP `ggml-org/Qwen3.6-27B-GGUF` Q4_K_M model is used. Its
revision and file hash are part of the run identity. A deterministic 128-row
benchmark chooses the safe batch and concurrency plan for the first allocation;
continuations reuse that plan.

Candidate selection validates the upstream `area_km2` and `area_bucket` values
against the fixed ranges: tiny `<0.1`, small `0.1–<1`, medium `1–<10`, and
large `10` square kilometres and above. Every large polygon enters the candidate
pool. Tiny, small, and medium polygons are ordered proportionally across H3
resolution-3 cells. The upstream `10-100km2` and `>100km2` labels both map to
`large`; there are no separate very-large or huge V2 categories. Polygons are
added until the pool can satisfy the requested sentence target, then sentences
receive a deterministic nested ranking. Larger targets extend the prior selection
without reshuffling it.

The operator preserves the V1 autonomous lifecycle: policy-aware site choice,
short allocations, durable per-batch checkpoints, asynchronous checkpoint
mirrors, automatic continuation, safe retry and site relocation, timing and
throughput reporting, and final publication only after validation. V2 output is
published below its existing namespace and never replaces `v1-afghanistan/`
artifacts.

## Evaluation

Every V2 run can export a deterministic 100-row manual-evaluation file with
the approved sentence context, model decision, score diagnostics, and blank
human-label and notes columns. This file is local evaluation material until
human labels are supplied.

## Compatibility and migration

The prompt version, output schema version, sampling version, model identity,
and tokenizer contract change together. Existing draft V2 checkpoints fail
identity validation and cannot be silently reused. V1 identity and files are
not migrated.
