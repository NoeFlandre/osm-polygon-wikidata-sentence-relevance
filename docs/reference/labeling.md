# Sentence labeling CLI

The focused entry point is:

```text
osm-polygon-label-sentences {probe,label,finalize,publish,track}
```

`probe` validates deterministic representative responses without writing
checkpoints. `label` requires immutable input, model, and source revisions, a
persistent work directory, the selected engine, and a positive batch size.
Each validated batch is checkpointed and the command reports completion,
elapsed time, and repair timing. A positive row limit selects a deterministic
canary. The operator's V1 regional command uses a zero sampling target for the
complete Afghanistan input; the V2 worldwide command supplies its positive
stratified target explicitly.

## Asynchronous checkpoint mirror

Grid'5000 payloads pass `--checkpoint-dataset-id` and a run-scoped
`--checkpoint-namespace checkpoints/<20-hex-run-id>` namespace. After each local
checkpoint and progress write, a single background worker commits that batch's
Parquet and metadata files to `.pipeline/checkpoints/<run-id>/` on the dataset's
`main` tree. This is not a public Git branch. The labeling loop never waits for Hub
network I/O. Failed uploads remain in `${work_dir}/.checkpoint-mirror/pending/`
and are retried when the same run resumes; a bounded final drain is attempted
when an allocation exits. The final V1 root or V2 `v2-worldwide/` publication is a
separate validated operation and is never performed by the mirror.

`finalize` refuses partial labels and creates the labeled Parquet, manifest,
and concise data-derived README. The historical V1 lane adds its two-label
plots, joint-label heatmap, polygon coverage funnel, normalized reason-code
charts, and selector-based slice table. The V2 lane adds one binary
`place_relevance` label, its two log-probabilities, derived margin and
two-class score, and a deterministic H3 sentence-distribution map.
`publish` revalidates that closed layout and uploads it to the
existing dataset in one Hub commit. No command accepts a token or creates a
repository.

## V2 stratified sampling and place-description labeling

The worldwide `all` label stage defaults to the separate V2 sampling contract: a target of
`200000` rows, seed `sentence-relevance-v2`, and H3 resolution `3`. Rows are
grouped only by the H3 cell containing their coordinates. A seeded weighted
allocation selects a deterministic proportional prefix of each H3-cell stratum
while preserving the input order in the output. Language and OSM primary tag
remain required metadata columns and are never quota dimensions. Every
large polygon (10 km² and above) enters the candidate pool; tiny, small, and
medium polygons are interleaved proportionally across occupied H3 cells. The
upstream `10-100km2` and `>100km2` labels both map to `large`. Each
selected sentence receives one binary `place_relevance` decision. `yes` means
the sentence describes the target place itself in visual or geographic terms,
such as terrain, landscape, land or water cover, soil, ecosystems, vegetation,
visible structures, or physical geographic setting. `no` means it does not make
that kind of place description. The model is asked for one token only. The
stored score evidence is the first-token log-probability returned for
`yes` and `no`, their difference, and its sigmoid two-class score. There are no
JSON explanations, reason codes, evidence excerpts, or `uncertain` labels in
V2. Rows with missing latitude or longitude are discarded before the candidate
pool is stratified. Missing language and tag values remain row metadata rather
than creating additional strata.

Increase `--sampling-target` on `run` or `resume RUN_ID` to extend the same sample. With the same immutable
input, seed, and H3 resolution, the smaller selection is always contained in
the larger one, so the next run adds rows instead of reshuffling the earlier
sample. The target is a mutable continuation budget, so the operator keeps the
same run identity and reuses validated checkpoints when it grows. It refuses a
smaller target once the run has been initialized. V2 targets are positive
integers. The V1 Afghanistan release and its
published Hub artifacts remain separate from this V2 sampling contract. V2
publication is mapped below `v2-worldwide/` on the same Hugging Face `main`
revision; the publisher refuses to replace the V1 root files.

When a larger V2 target is finalized locally, only a previously validated V2
output with the same immutable identity may be replaced. V1 outputs, mismatched
outputs, and symlinked output paths are rejected.

## Trackio

Install the optional dependency:

```bash
uv sync --locked --extra tracking
```

Then log exactly one static final run from a validated publication:

```bash
osm-polygon-label-sentences track \
  --output-dir /path/to/label-publication \
  --project afghanistan-labeling \
  --run-name final-afghanistan
```

The default Space follows the release lane in the manifest: V1 uses the public
[Afghanistan Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/afghanistan-labeling-trackio),
while V2 uses the separate
[worldwide dashboard](https://huggingface.co/spaces/NoeFlandre/worldwide-stratified-labeling-trackio).
Use `--space-id OWNER/SPACE` to override it, or call the Python API with
`space_id=None` for local-only storage. The command validates the manifest
analytics against a fresh computation from `sentences.parquet` before
initializing Trackio. It records one `step=0` containing:

For V2, that static run contains the one-label KPI cards, area-bucket table,
binary label distribution, and H3 coverage image. It does not emit the V1
joint-label, polygon-coverage, or reason-code metrics.

- KPI cards for total labeled sentences, unique polygons, unique languages,
  and strong-positive yield.
- An immutable joint-label table plus heatmap.
- The polygon coverage funnel table and plot.
- Normalized land-use and polygon reason-code tables and plot.
- A selector-based HTML table for language, source, and `osm_primary_tag`.
  Groups below 100 sentences are omitted; each shown group has both-yes rate,
uncertain rate, and sample size.

The [Afghanistan dataset presentation](https://noeflandre.github.io/osm-polygon-wikidata-sentence-relevance/presentations/afghanistan-dataset-overview/index.html)
provides a visual overview of the published table and labeling method.
