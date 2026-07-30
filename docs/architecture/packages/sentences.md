# Sentences

## Responsibility

Normalizes text, infers sentence boundaries, repairs high-confidence residual
boundaries, and constructs deterministic rows.

## Public entry points

`SaTSentenceSegmenter`, `segment_joined_sections`, and
`finalize_sentence_dataset` are the supported surface.

## Internal structure

Preprocessing, SaT integration, device placement, segmentation, tables, and
finalization are separate modules under `sentences/`.

## Invariants

The model does not rewrite source text. Explicit accelerators never downgrade.
Context precedes exact deduplication and IDs are deterministic.

## Tests

`tests/unit/sentences/` covers multilingual boundaries, normalization, devices,
schemas, and finalization.
