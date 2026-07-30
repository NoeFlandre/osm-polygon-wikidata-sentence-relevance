# Publishing

## Responsibility

Publishes a validated export to an existing Hugging Face dataset.

## Public entry points

`publish_export_directory` performs one-commit publication.

## Internal structure

`publishing/` owns Hub adaptation and response validation; artifact generation
stays in `output/` and `labeling/`.

## Invariants

No token argument or repository creation exists. The validated artifact set is
one commit followed by immutable readback verification.

## Tests

`tests/unit/publishing/` uses injected Hub clients to cover commit construction
and failures without network access.
