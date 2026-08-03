# Afghanistan labeling CLI

The focused entry point is:

```text
osm-polygon-label-sentences {probe,label,finalize,publish,track}
```

`probe` validates deterministic representative responses without writing
checkpoints. `label` requires immutable input, model, and source revisions, a
persistent work directory, the selected engine, and a positive batch size.
Each validated batch is checkpointed and the command reports completion,
elapsed time, and repair timing. A positive row limit selects a deterministic
canary; zero labels the complete input.

`finalize` refuses partial labels and creates the labeled Parquet, manifest,
concise data-derived README, legacy label plots, a joint-label heatmap, a
polygon coverage funnel, normalized reason-code charts, and a selector-based
slice table. `publish` revalidates that closed layout and uploads it to the
existing dataset in one Hub commit. No command accepts a token or creates a
repository.

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

By default the run is synced to the public
[Afghanistan Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/afghanistan-labeling-trackio).
Use `--space-id OWNER/SPACE` to choose another Space, or call the Python API
with `space_id=None` for local-only storage. The command validates the manifest
analytics against a fresh computation from `sentences.parquet` before
initializing Trackio. It records one `step=0` containing:

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
