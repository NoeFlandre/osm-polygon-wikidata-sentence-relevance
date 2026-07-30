# Output

## Responsibility

Builds, validates, documents, and atomically installs public dataset artifacts.

## Public entry points

`export_finalized_dataset`, `validate_export_directory`, and the card and
profile APIs are the supported surface.

## Internal structure

Export, manifests, profiles, cards, plots, and publication validation remain
separate under `output/`.

## Invariants

Statistics derive from Parquet. Hashes bind every artifact, cards reproduce
exactly, and temporary exports cannot replace valid output before validation.

## Tests

`tests/unit/output/` covers schemas, hashes, profiles, plots, cards, atomic
installation, and strict contents.
