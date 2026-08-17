![Afghanistan sentence relevance dataset overview](docs/assets/afghanistan-labeling-hero.png)

# OSM Polygon – Wikidata Sentence Relevance

[![CI](https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance/actions/workflows/ci.yml/badge.svg)](https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-MkDocs%20Material-526CFE)](https://noeflandre.github.io/osm-polygon-wikidata-sentence-relevance/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](.python-version)

Deterministic sentence extraction and relevance labeling for OSM polygons
joined to Wikipedia and Wikivoyage sections. The pipeline preserves source
metadata, uses immutable inputs and pinned runtimes, and publishes validated
Parquet releases with manifests and provenance.

## Public releases

| Release | Scope | Public data | Documentation |
| --- | --- | --- | --- |
| **V1 Afghanistan** | 54,462 labeled sentences, 161 polygons, 115 languages | [HF folder](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance/tree/main/v1-afghanistan) · [Trackio](https://huggingface.co/spaces/NoeFlandre/afghanistan-labeling-trackio) | [V1 guide](docs/releases/v1-afghanistan.md) · [HF card](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance/tree/main/v1-afghanistan/README.md) |
| **V2 worldwide** | 200,000 H3-stratified binary place-description labels | [HF folder](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance/tree/main/v2-worldwide) · [Trackio](https://huggingface.co/spaces/NoeFlandre/worldwide-stratified-labeling-trackio) | [V2 guide](docs/releases/v2-worldwide.md) · [HF card](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance/tree/main/v2-worldwide/README.md) |

The complete output dataset is
[NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance).
It is derived from the immutable upstream input dataset
[NoeFlandre/osm-polygon-wikidata-only](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only).
Its root README is only an index: release data lives in `v1-afghanistan/`
and `v2-worldwide/`; `.pipeline/checkpoints/<run-id>/` contains resumable
batch provenance and is not a release split.

## Quick start

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --locked --extra operator --extra hub --extra segmentation
```

Run the regional V1 workflow:

```bash
uv run osm-polygon-grid5000 run \
  --scope region --region afghanistan-latest --stage all
```

Run the worldwide V2 workflow:

```bash
uv run osm-polygon-grid5000 run --scope all --stage all
```

The operator resolves immutable input and source revisions, selects a
policy-compliant Grid'5000 GPU allocation, resumes durable checkpoints, and
publishes only after final validation. Use `--detach` for a durable local
supervisor. See the [Grid'5000 guide](docs/guides/grid5000.md) for the full
workflow and the [reproducibility guide](docs/guides/reproducibility.md) for
locked environments and identity checks.

## Documentation

- [Project documentation](https://noeflandre.github.io/osm-polygon-wikidata-sentence-relevance/)
- [Getting started](docs/guides/getting-started.md)
- [V1 Afghanistan release](docs/releases/v1-afghanistan.md)
- [V2 worldwide release](docs/releases/v2-worldwide.md)
- [Labeling and publication reference](docs/reference/labeling.md)
- [CLI reference](docs/reference/cli.md)
- [Data contract](docs/reference/data-contract.md)
- [Architecture overview](docs/architecture/overview.md)

## Repository layout

```text
src/osm_polygon_sentence_relevance/  Python package and compatibility facades
scripts/                              Grid'5000 and publication helpers
tests/                                Unit, integration, and compatibility tests
docs/                                 MkDocs source and release guides
```

## Development

```bash
uv sync --locked --all-extras --dev
just check
```

See [development](docs/guides/development.md) before changing contracts,
release paths, or publication behavior.

## Citation

Please cite the release you use and the reproducibility repository. The
machine-readable metadata is in [`citation.cff`](citation.cff).

```bibtex
@software{flandre2026osm,
  author = {Flandre, Noé},
  title = {OSM Polygon - Wikidata Sentence Relevance},
  year = {2026},
  url = {https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance}
}
```
