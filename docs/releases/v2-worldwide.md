# V2 worldwide release

V2 is the worldwide, H3-stratified place-description release, published in
the [`v2-worldwide/` folder on Hugging Face](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance/tree/main/v2-worldwide).
The folder contains the final Parquet table, manifest, card, and deterministic
label/H3 plots.

## Release facts

| Metric | Value |
| --- | ---: |
| Labeled sentences | 200,000 |
| Unique polygons | 10,268 |
| Languages | 317 |
| Missing-coordinate rows | 0 |

Rows are selected with seed `sentence-relevance-v2` from H3 resolution-3
cells. Language and OSM primary tag remain metadata; they are not sampling
dimensions. Rows without latitude or longitude are excluded before allocation.
The selection is a deterministic nested prefix, so a larger target extends the
same run without reshuffling earlier rows.

## Label contract

`place_relevance=yes` means the target sentence describes the place in physical
or geographic terms: land cover, soil, vegetation, ecosystems, terrain,
geomorphology, visible structures, or physical setting. `no` covers chronology,
administration, people, events, economy, transport activity, navigation, links,
another place, or other non-physical facts.

The pinned `unsloth/Qwen3.6-27B-MTP-GGUF` runtime receives the page title,
section title, previous sentence, target sentence, and next sentence. It emits
one token with temperature `0`, seed `0`, and `max_tokens=1`. The annotation
decision uses the first-token log-probability margin (`yes_logprob -
no_logprob`), not the decoded token. The Parquet stores both log-probabilities,
the margin, and the relative two-class sigmoid score. The exact prompt and
immutable model identity are preserved in the [V2 HF card](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance/tree/main/v2-worldwide/README.md).

## Reproduction and metrics

- [Labeling and V2 reference](../reference/labeling.md)
- [Grid'5000 production workflow](../guides/grid5000.md)
- [V2 binary-label decision](../architecture/decisions/0003-v2-binary-place-description.md)
- [Public worldwide Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/worldwide-stratified-labeling-trackio)
- [Trackio snapshot dataset](https://huggingface.co/datasets/NoeFlandre/worldwide-stratified-labeling-trackio-data)

V2 never replaces V1 artifacts. Runtime checkpoint provenance remains under
`.pipeline/checkpoints/<run-id>/` and is not part of either public release
folder.
