# V1 Afghanistan release

V1 is the complete Afghanistan labeling artifact, published in the
[`v1-afghanistan/` folder on Hugging Face](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance/tree/main/v1-afghanistan).
The folder is self-contained: its Parquet table, manifest, card, and plots are
published together.

## Release facts

| Metric | Value |
| --- | ---: |
| Labeled sentences | 54,462 |
| Unique polygons | 161 |
| Languages | 115 |
| Strong-positive yield (both labels yes) | 18.20% |

Each row contains two independent model annotations:

1. Does the sentence describe land use or land cover for the polygon?
2. Does the sentence describe the named target polygon itself?

The labels are model-generated annotations, not ground truth. Each decision
includes a closed-enumeration reason code and a short evidence excerpt. The
full data-derived card, provenance, model revision, prompt version, and
artifact hashes are in the [V1 HF card](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance/tree/main/v1-afghanistan/README.md).

## Reproduction and metrics

- [Labeling CLI reference](../reference/labeling.md)
- [Grid'5000 production workflow](../guides/grid5000.md)
- [Reproducibility and immutable inputs](../guides/reproducibility.md)
- [Public Afghanistan Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/afghanistan-labeling-trackio)
- [Afghanistan dataset presentation](https://noeflandre.github.io/osm-polygon-wikidata-sentence-relevance/presentations/afghanistan-dataset-overview/index.html)

V1 remains separate from the worldwide V2 contract. V2 has a different
sampling policy, one binary label, and its own folder and Trackio dashboard.
