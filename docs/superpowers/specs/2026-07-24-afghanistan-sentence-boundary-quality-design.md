# Afghanistan Sentence-Boundary Quality Design

## Problem

The published Afghanistan dataset contains rows in which SaT 3L emitted
multiple obvious sentences as one segment. The supplied Arabic airport lead is
reproducible: `sat-3l-sm` returns one 370-character segment, while
normalization and finalization preserve that segment without merging it.

## Decision

Use `sat-12l-sm` as the production default. It is the strongest general
supervised-mixture SaT model and splits the reported Arabic regression into
the expected three sentences without threshold tuning.

Add a conservative post-model boundary repair for residual, high-confidence
boundaries. It keeps terminal punctuation on the preceding sentence and
splits only when both sides are substantive. Universal sentence marks
(`?`, `!`, Arabic question mark, CJK full stops, and Indic danda) are
recognized. A period is recognized only for Arabic-script languages when it
is followed by an Arabic-script clause; short abbreviation-like prefixes and
numeric continuations are not split.

This is intentionally not a general rule-based sentence tokenizer. SaT remains
authoritative, and the repair handles only obvious false negatives.

## Validation

RED-first tests cover the exact reported Arabic text, Arabic abbreviations and
decimals, CJK/Indic terminal marks, segment order, and batch isolation. A
corpus-level audit reports residual high-confidence boundaries and prevents
publication if any remain.

The Afghanistan rebuild uses the newest immutable upstream revision, a fresh
checkpoint identity, `sat-12l-sm` on Grid'5000 CUDA, and the existing
resumable streaming workflow. The final Parquet, manifest, README, and plots
are regenerated from data, validated locally, published atomically to
Hugging Face `main`, independently downloaded, and validated again.

## Public contract

The dataset card names the exact segmentation model and immutable revision,
describes the conservative residual-boundary repair, reports factual
sentence-length statistics from the final Parquet, and keeps the automated
segmentation limitation explicit. No hand-entered result counts are allowed.
