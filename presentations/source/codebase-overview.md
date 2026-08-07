---
title: "How the sentence relevance pipeline works"
author: "Noé Flandre"
date: "August 2026"
fonts:
  heading: "Rubik"
  body: "Poppins"
footer:
  left: "OSM polygon sentence relevance"
  center: "Implementation overview"
  right: "{n}/{N}"
custom_css: |
  :root {
    --template-accent: #176b87;
    --template-accent-soft: #e7f3f5;
    --colloquium-progress-fill: var(--template-accent);
    --colloquium-link: #176b87;
    --colloquium-font-body: "Poppins", "Helvetica Neue", Arial, sans-serif;
    --colloquium-font-heading: "Rubik", "Helvetica Neue", Arial, sans-serif;
  }
  .slide--section-break { background: #176b87; color: #ffffff; }
  .slide--section-break h2 { color: #ffffff; }
  .colloquium-title-eyebrow { color: #176b87; letter-spacing: .14em; text-transform: uppercase; font-weight: 700; }
  .colloquium-title-rule { width: 180px; height: 6px; margin-top: 28px; background: #f2a65a; }
  .accent { color: #176b87; font-weight: 700; }
  .warm { color: #c86b25; font-weight: 700; }
  .small { color: #5b6472; font-size: .78em; line-height: 1.35; }
  .command { background: #eef4f5; border-left: 5px solid #176b87; padding: 18px 24px; font-family: "JetBrains Mono", Menlo, monospace; font-size: .78em; white-space: pre-wrap; }
  .state { color: #176b87; font-weight: 700; }
  .link-list a { color: #176b87; }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->
<!-- notes: [Sources] Project README; docs/architecture/overview.md; docs/reference/grid5000-operator.md. -->

# One command coordinates a reproducible remote pipeline

<div class="colloquium-title-eyebrow">Codebase overview</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Noé Flandre</p>
<p>Local control · remote CUDA · validated publication</p>
</div>

<p class="colloquium-title-note">The repository separates sentence construction, model labeling, remote operations, and public release so each boundary can be tested independently.</p>

---

<!-- layout: title-banner -->
<!-- notes: [Sources] README.md and docs/guides/grid5000.md. -->

# The Mac orchestrates; Grid’5000 performs inference; the Hub receives validated artifacts

<div class="colloquium-title-eyebrow">Execution model</div>
<div class="colloquium-title-rule"></div>

The local operator fixes the identity, checks policy and storage, selects a compatible site, monitors the job, and publishes only after readback validation. The remote worker never falls back to local inference.

---

<!-- columns: 25/25/25/25 -->
<!-- notes: [Sources] docs/architecture/overview.md, docs/architecture/packages/operator.md, and docs/guides/grid5000.md. -->

## Four boundaries keep the production path understandable

### Input

Immutable HF revision, six-table shard set, schema checks.

|||

### Sentences

Joins, SaT segmentation, normalization, boundary repair, deduplication.

|||

### Labels

Prompt, CUDA model runtime, strict parsing, repair, batch checkpoints.

|||

### Release

Manifest, data-derived card, plots, Trackio, one Hub commit.

---

<!-- rows: 34/66 -->
<!-- notes: [Sources] src/osm_polygon_sentence_relevance/ingestion, joins, sentences, and output packages; docs/architecture/packages/sentences.md. -->

## Sentence construction is deterministic before any model call

The pipeline reads the immutable source snapshot and builds the same sentence table from the same inputs.

===

1. Discover the six source tables for each shard.
2. Join polygons to Wikipedia and Wikivoyage sections with integrity checks.
3. Run the pinned SaT segmenter, then apply conservative residual-boundary repair.
4. Normalize text, remove exact duplicates, and derive stable sentence and content IDs.
5. Validate the schema, hashes, and row accounting before labeling.

<p class="small">The labeler receives a stable row. It never rewrites the sentence text.</p>

---

<!-- columns: 50/50 -->
<!-- notes: [Sources] src/osm_polygon_sentence_relevance/labeling/prompt.py, runtime.py, runner.py, validation.py, and docs/reference/labeling.md. -->

## V1 and V2 share a runtime but expose different label contracts

### V1 Afghanistan

Two independent decisions: land use or land cover, and relevance to the target polygon. The result includes reason codes and exact evidence.

### V2 worldwide

One decision: does the target sentence describe the place in visual or geographic terms? The result includes `place_relevance`, a reason, and exact evidence.

<p class="small">Both lanes use the target sentence, neighboring sentences, polygon metadata, language, section title, the OSM primary tag, and all OSM tags as context.</p>

---

<!-- columns: 45/55 -->
<!-- notes: [Sources] docs/architecture/packages/operator.md, docs/reference/grid5000-operator.md, and src/osm_polygon_sentence_relevance/operator/. -->

## Remote work is a resumable state machine, not a long shell session

<div class="state">created → prepared → submitted → running → checkpointed → validated → complete</div>

The durable state binds source commit, input revision, model, prompt, batch settings, and runtime configuration. A valid checkpoint is reused only when its identity and hashes still match.

|||

### The operator boundary

- SSH uses fixed arguments and bounded log reads.
- OAR jobs are submitted only after usage-policy and storage checks.
- Short allocations are expected; walltime stops preserve checkpoints.
- Ctrl-C stops local monitoring, not the remote allocation.
- Resume reattaches to evidence-backed jobs before submitting another one.

---

<!-- layout: section-break -->
<!-- title: center -->
<!-- notes: [Sources] docs/architecture/packages/operator.md and docs/guides/grid5000.md. -->

## Safety is part of the architecture

---

<!-- columns: 50/50 -->
<!-- notes: [Sources] docs/guides/grid5000.md, docs/reference/grid5000-operator.md, and src/osm_polygon_sentence_relevance/operator/. -->

## Every expensive or irreversible boundary fails closed

### Before remote work

- Clean source checkout and immutable revisions.
- GPU and CUDA preflight on the worker.
- Live usage-policy and quota checks.
- Managed storage cleanup only for eligible run directories.

### Before public release

- Complete checkpoint set and exact row accounting.
- Schema and content hash validation.
- Deterministic manifest, card, and PNG assets.
- Hub readback comparison after the commit.

---

<!-- columns: 50/50 -->
<!-- notes: [Sources] src/osm_polygon_sentence_relevance/labeling/finalization.py, publication.py, tracking.py, and docs/reference/labeling.md. -->

## Publication is a data contract, not a copy operation

1. Finalize validated checkpoints into `sentences.parquet`.
2. Compute all public metrics from that table.
3. Render the manifest, concise dataset card, and deterministic plots.
4. Upload the correct release prefix to the existing Hugging Face `main` tree.
5. Read back the committed files and verify bytes, schema, and card equality.

<p class="small">V1 stays at the dataset root. V2 uses <span class="accent">v2-worldwide/</span>. Checkpoint mirrors use a private run-scoped prefix and are never treated as a release.</p>

---

<!-- columns: 50/50 -->
<!-- notes: [Sources] pyproject.toml, justfile, .pre-commit-config.yaml, .github/workflows/, and CONTRIBUTING.md. -->

## The developer loop is intentionally small and repeatable

<div class="command">uv sync --locked --all-extras --dev
just check
pre-commit run --all-files</div>

The repository uses `uv` for environments, Ruff for format and lint, `ty` for type checking, pytest for behavior, pre-commit for local gates, Just for repeatable commands, and GitHub Actions for CI and documentation.

<p class="small">The package and docs remain public-facing. Operational scripts stay explicit and are tested as contracts.</p>

---

<!-- layout: title-banner -->
<!-- notes: [Sources] All links are public project endpoints. -->

# Follow the contracts from the public entry points

<div class="colloquium-title-eyebrow">Further reading</div>
<div class="colloquium-title-rule"></div>

[GitHub repository](https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance)  ·  [Architecture docs](https://noeflandre.github.io/osm-polygon-wikidata-sentence-relevance/architecture/overview/)  ·  [Operator guide](https://noeflandre.github.io/osm-polygon-wikidata-sentence-relevance/reference/grid5000-operator/)

[V1 Hugging Face release](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance)  ·  [V1 Trackio](https://huggingface.co/spaces/NoeFlandre/afghanistan-labeling-trackio)  ·  [V2 Trackio](https://huggingface.co/spaces/NoeFlandre/worldwide-stratified-labeling-trackio)

<p class="small">The codebase is the reproducibility record. The dataset card is the data-facing record.</p>
