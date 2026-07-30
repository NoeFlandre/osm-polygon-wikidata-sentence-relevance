# Operator Configuration Modularization

## Goal

Split the oversized operator configuration implementation into focused internal
modules without changing any public import, default, validation rule,
serialization, run identity, exception, or runtime behavior.

## Scope

The pass is limited to `osm_polygon_sentence_relevance.operator.config` and its
tests and package documentation. It does not change CLI flags, Grid'5000
behavior, persisted state, dataset schemas, model settings, remote jobs, or
publication.

## Architecture

`operator/config.py` remains the stable public module and becomes a thin,
explicit re-export facade.

Implementation moves into `operator/_config/`:

- `validation.py` owns primitive parsing, repository/model/revision/region
  validation, scope and stage normalization, and runtime-requirement
  normalization.
- `models.py` owns `Scope`, `Stage`, `Grid5000Requirements`, `RunIdentity`, and
  `OperatorConfig`, plus the immutable public defaults used to construct them.
- `__init__.py` explicitly re-exports the supported configuration surface.

The internal modules may import from each other in one direction only:
`models` depends on `validation`; `validation` never depends on `models`.

## Compatibility contract

The refactor must preserve:

- every symbol currently listed in `operator.config.__all__`;
- imports from both `operator.config` and `operator`;
- constructor and builder signatures;
- validation exception types and messages;
- stage-aware canonical identity fields;
- byte-identical `canonical_json` and identical 20-character `run_id`;
- reconstruction through `OperatorConfig.from_persisted`;
- frozen and slotted dataclass behavior;
- constants and default values.

No compatibility aliases beyond the existing public facade are introduced.
Private helpers are not public API.

## TDD boundary

Before production movement, add a structural compatibility test that fails
against the current monolith and requires:

- the `_config` package and its three modules;
- a facade containing only a docstring, explicit imports, and `__all__`;
- no production implementation classes or functions in `config.py`;
- every production Python file in this focused configuration boundary to stay
  below 500 physical lines.

Existing configuration tests remain the behavioral oracle. Add equivalence
tests only where they prove public exports, signatures, canonical identity, or
persisted reconstruction across the new boundary.

## Error handling

Validation remains eager and local. Persisted mappings continue to be
validated through the same construction path. The refactor does not catch,
translate, or broaden exceptions.

## Documentation

Update the operator package guide to name the new configuration modules and
their one-way dependency. No CLI or user guide change is required because
public behavior is unchanged.

## Acceptance

The pass is complete only when:

- the focused structural test is observed failing before implementation and
  passing afterward;
- all configuration and operator regression tests pass;
- the complete repository test suite meets its existing coverage gate;
- Ruff formatting and linting, `ty`, package build, distribution verification,
  shell syntax, and `git diff --check` pass;
- the staged diff contains only this refactor and documentation;
- no SSH, OAR, Hugging Face, inference, or publication operation occurs.
