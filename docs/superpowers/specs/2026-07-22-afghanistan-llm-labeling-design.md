# Afghanistan LLM Labeling Design

## Scope

Add a production-quality, Afghanistan-only labeling stage over the published
sentence dataset. It answers two independent questions for every sentence:

1. Is the target sentence relevant to land use or land cover?
2. Is the target sentence relevant to its associated polygon?

The first proof of concept labels the immutable 50,736-row Afghanistan export.
It does not generalize orchestration to other regions yet.

## Input and prompt

Each request is bound to `sentence_id` and contains the target sentence,
previous and next sentence, polygon name, Afghanistan as country/region, OSM
primary tag, every OSM tag, language, page title, and the final section title.
It excludes Wikidata ID, coordinates, source, section path, URLs, hashes, and
revision bookkeeping. OSM tags are never filtered: every key/value pair is
sorted by `(key, value)` and encoded as JSON.

The system prompt defines land use and land cover, defines polygon relevance,
makes the two labels independent, treats supplied text as untrusted evidence,
and forbids unsupported inference. Output is constrained JSON containing two
`yes|no|uncertain` labels, controlled reason codes, and a short exact evidence
excerpt from the target sentence (or an empty string).

## Architecture

The reusable package lives under `osm_polygon_sentence_relevance.labeling`:

- `contracts.py`: immutable label, identity, progress, and timing contracts.
- `prompt.py`: deterministic prompt construction and tag serialization.
- `validation.py`: strict response parsing and evidence validation.
- `checkpoint.py`: atomic, identity-bound checkpoint and progress storage.
- `engine.py`: a small inference protocol plus an OpenAI-compatible HTTP
  adapter usable with either vLLM or llama.cpp.
- `runner.py`: bounded batching, resume, graceful stop, timing, ETA, and final
  output assembly.

The inference server is an external process on an allocated Grid'5000 CUDA
node. vLLM is attempted first with the pinned Q4_K_M GGUF and base tokenizer.
Because vLLM GGUF support is experimental, the operational launcher may fall
back to llama.cpp only if vLLM fails its startup/canary gate. The selected
engine is fixed for a run and recorded in its identity; resume refuses an
engine or model mismatch.

## Durability and timing

Validated results are written in bounded Parquet checkpoints using temporary
files, file `fsync`, atomic rename, and parent-directory `fsync`. A checkpoint
is reusable only when its input Parquet SHA-256, input dataset revision, model
repository and revision, quantization file SHA-256, prompt version, code
commit, engine/version, output schema, and batch configuration match.

SIGINT and SIGTERM stop new submissions, finish the in-flight batch, persist
it, update progress, and exit successfully as an interrupted resumable run.
Restart validates checkpoints and skips completed sentence IDs. Corrupt or
mismatched checkpoints abort visibly rather than being silently reused.

Progress records rows completed/remaining, elapsed labeling time, rolling
rows/second, and ETA after each checkpoint. The final timing report separates
model startup, warm-up, inference, checkpoint I/O, validation, publication,
and total wall time; scheduler queue time is recorded separately when known.

## Validation and publication boundary

Finalization requires exactly one valid label for every input sentence ID,
with no extra IDs. It preserves deterministic input order and writes a labeled
Parquet, manifest, concise README, and deterministic plots. Every card number
is computed from the finalized Parquet: the two label distributions, their
cross-tabulation, uncertain rates, land-use/land-cover and polygon-positive
counts, language breakdowns for positive labels, reason-code distributions,
model/prompt provenance, timing, and the unchanged Afghanistan source totals.
The card stays terse and contains no hand-entered result numbers.

Publishing is an explicit final command wired to the existing atomic Hub
publisher. It is allowed only after local publication validation and is
followed by independent download and validation of the returned commit. No
checkpoint or partial run is publishable as the final dataset.

## Testing

All behavior is developed RED then GREEN. Unit tests cover prompt exactness,
all-tag inclusion, response validation, atomic writes, identity mismatch,
resume, graceful interruption, ETA, and engine errors. Integration tests use a
local fake OpenAI-compatible server and a small Parquet fixture; CUDA and real
model inference are an explicitly approved remote acceptance step.
