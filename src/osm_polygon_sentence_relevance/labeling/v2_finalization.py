"""Deterministic local finalization for the V2 binary score release."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .v2_analytics import build_v2_analytics
from .v2_checkpoint import V2CheckpointStore
from .v2_contracts import (
    V2_LOGIT_PROMPT_VERSION,
    V2_MODEL_FILE,
    V2_MODEL_REPO_ID,
    V2LogitRecord,
)
from .v2_manual_eval import write_v2_manual_eval
from .v2_sampling import select_v2_rows

V2_PUBLICATION_FILES: tuple[str, ...] = (
    "sentences.parquet",
    "manifest.json",
    "README.md",
    "assets/label_distribution.png",
    "assets/h3_sentence_distribution.png",
)
INPUT_DATASET_URL = (
    "https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only"
)
GITHUB_URL = "https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance"


@dataclass(frozen=True, slots=True)
class V2Publication:
    """Validated facts about one local V2 publication."""

    directory: Path
    row_count: int
    parquet_sha256: str
    files: tuple[Path, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plot_label_distribution(table: pa.Table, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the hub extra to render V2 assets") from exc
    counts = Counter(table["place_relevance"].to_pylist())
    fig, axis = plt.subplots(figsize=(7, 4), dpi=140)
    labels = ["yes", "no"]
    values = [counts.get(label, 0) for label in labels]
    bars = axis.bar(labels, values, color=("#2878b5", "#c44e52"))
    axis.set_ylabel("Sentences")
    axis.set_title("V2 place-description labels")
    axis.bar_label(bars, labels=[f"{value:,}" for value in values])
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": ""})
    plt.close(fig)


def _render_card(
    *,
    dataset_repo_id: str,
    table: pa.Table,
    parquet_bytes: int,
    identity: dict[str, Any],
    analytics: dict[str, Any],
    h3: dict[str, Any],
) -> str:
    counts = analytics["place_counts"]
    total = table.num_rows
    yes_rate = counts.get("yes", 0) / total if total else 0.0
    return f"""---
license: apache-2.0
language:
  - multilingual
tags:
  - geospatial
  - sentence-classification
  - openstreetmap
  - wikipedia
pretty_name: Worldwide place-description sentence labels (V2)
size_categories:
  - 100K<n<1M
dataset_info:
  config_name: default
  features:
    sentence_id: string
    place_relevance: string
    yes_logprob: float64
    no_logprob: float64
    logit_margin: float64
    two_class_probability: float64
  splits:
    - name: train
      num_bytes: {parquet_bytes}
      num_examples: {total}
---

# Worldwide place-description labels, V2

This is the separate V2 release. It contains **{total:,} labeled sentences** and keeps
the Afghanistan V1 release at the dataset root unchanged. The input sentences were
extracted from [{INPUT_DATASET_URL}]({INPUT_DATASET_URL}). The code and reproducibility
contracts are in [{GITHUB_URL}]({GITHUB_URL}).

## Method

The model receives only the page title, section title, previous sentence, target
sentence, and next sentence. It returns one token, `yes` or `no`, answering whether
the target describes the place in physical or geographic terms. This includes land
use or cover, soil, vegetation, ecosystems, terrain, visible structures, and the
place's physical setting. History, administration, people, events, economy,
transport activity, navigation, and unrelated places are `no`.

Inference uses `{V2_MODEL_REPO_ID}` (`{V2_MODEL_FILE}`), served by llama.cpp with
temperature 0 and one generated token. The first-token `yes` and `no` log-probabilities
are recorded; `logit_margin = yes_logprob - no_logprob`, and
`two_class_probability = sigmoid(logit_margin)`. This is a relative two-class score,
not a calibrated full-vocabulary probability.

## Sampling

All polygons in the `large` bucket (10 km² and above) enter the candidate pool.
Tiny (<0.1 km²), small (0.1 to <1 km²), and medium (1 to <10 km²) polygons are
ordered proportionally across occupied H3 resolution-3 cells. The upstream
`10-100km2` and `>100km2` labels both map to this `large` bucket. Sentences are
then ranked by the deterministic seed `{identity.get("sampling_seed", "")}` and
taken as a nested prefix up to the requested target. H3 resolution 3 is the
closest practical global unit to the requested roughly 100 km scale (average edge
about 69 km).

## Results

| Metric | Value |
| --- | ---: |
| Labeled sentences | {total:,} |
| Unique polygons | {analytics["unique_polygons"]:,} |
| Unique languages | {analytics["unique_languages"]:,} |
| Place-description yes | {counts.get("yes", 0):,} ({yes_rate:.1%}) |
| H3 cells represented | {h3["cell_count"]:,} |

![Label distribution](assets/label_distribution.png)

![Sentence distribution by H3 cell](assets/h3_sentence_distribution.png)

## Provenance

- Input revision: `{identity.get("input_dataset_revision", "")}`
- Model revision: `{identity.get("model_revision", "")}`
- Prompt version: `{V2_LOGIT_PROMPT_VERSION}`
- Source commit: `{identity.get("source_commit", "")}`
- License: Apache-2.0
"""


def finalize_v2_dataset(
    *,
    input_path: Path,
    store: V2CheckpointStore,
    output_dir: Path,
    dataset_repo_id: str,
) -> V2Publication:
    """Join validated V2 scores to the selected input and atomically stage output."""

    input_path = Path(input_path)
    if _sha256(input_path) != store.identity.input_sha256:
        raise ValueError("V2 input SHA-256 does not match run identity")
    source = pq.read_table(input_path)
    # A smoke lane deliberately labels ``row_limit`` rows while retaining the
    # full production sampling target in the shared run identity.  Finalize
    # the lane's actual prefix first; production has row_limit=0 and therefore
    # keeps the full target unchanged.
    target = int(
        store.identity.row_limit
        or store.identity.sampling_target
        or source.num_rows
    )
    selected = select_v2_rows(
        source,
        target=target,
        seed=str(store.identity.sampling_seed or "v2"),
    )
    records = store.load_all()
    by_id = {record.sentence_id: record for record in records}
    ids = selected["sentence_id"].to_pylist()
    if len(records) != selected.num_rows or set(by_id) != set(ids):
        raise ValueError(
            "V2 finalization requires exactly one score per selected sentence"
        )
    ordered = [by_id[value] for value in ids]
    table = selected
    additions = {
        "place_relevance": [record.place_relevance for record in ordered],
        "yes_logprob": [record.yes_logprob for record in ordered],
        "no_logprob": [record.no_logprob for record in ordered],
        "logit_margin": [record.logit_margin for record in ordered],
        "two_class_probability": [record.two_class_probability for record in ordered],
    }
    for name, values in additions.items():
        table = table.append_column(name, pa.array(values))
    analytics = build_v2_analytics(
        table, h3_resolution=int(store.identity.h3_resolution or 3)
    )
    h3_stats = {
        "resolution": int(store.identity.h3_resolution or 3),
        "cell_count": analytics.h3_cell_count,
        "sentence_count": table.num_rows - analytics.missing_coordinate_count,
        "missing_coordinate_count": analytics.missing_coordinate_count,
    }
    identity = store.identity.to_dict()
    manifest = {
        "schema_version": 2,
        "release_lane": "v2-worldwide",
        "dataset_repo_id": dataset_repo_id,
        "run_identity": identity,
        "statistics": {**analytics.to_dict(), "h3": h3_stats},
        "model": {"repo_id": V2_MODEL_REPO_ID, "file": V2_MODEL_FILE},
    }
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    os.chmod(staging, 0o700)
    try:
        assets = staging / "assets"
        assets.mkdir(mode=0o700)
        pq.write_table(table, staging / "sentences.parquet", compression="zstd")
        os.chmod(staging / "sentences.parquet", 0o600)
        _plot_label_distribution(table, assets / "label_distribution.png")
        from .analytics import render_h3_sentence_distribution

        render_h3_sentence_distribution(
            table,
            assets / "h3_sentence_distribution.png",
            resolution=int(store.identity.h3_resolution or 3),
            scope_label="Worldwide",
        )
        manifest["parquet_sha256"] = _sha256(staging / "sentences.parquet")
        manifest["artifact_sha256"] = {
            "sentences.parquet": manifest["parquet_sha256"],
            "assets/label_distribution.png": _sha256(assets / "label_distribution.png"),
            "assets/h3_sentence_distribution.png": _sha256(
                assets / "h3_sentence_distribution.png"
            ),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (staging / "README.md").write_text(
            _render_card(
                dataset_repo_id=dataset_repo_id,
                table=table,
                parquet_bytes=(staging / "sentences.parquet").stat().st_size,
                identity=identity,
                analytics=analytics.to_dict(),
                h3=h3_stats,
            )
        )
        for path in (
            staging / "manifest.json",
            staging / "README.md",
            *assets.iterdir(),
        ):
            os.chmod(path, 0o600)
        write_v2_manual_eval(
            selected,
            ordered,
            store.root / "manual_eval.jsonl",
            limit=min(100, selected.num_rows),
        )
        backup = output_dir.with_name(f".{output_dir.name}.backup")
        if output_dir.exists():
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(output_dir, backup)
        os.replace(staging, output_dir)
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_v2_publication(output_dir)


def validate_v2_publication(directory: Path) -> V2Publication:
    """Validate hashes, layout, identity, derived fields, and card equality."""

    directory = Path(directory)
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual != set(V2_PUBLICATION_FILES):
        raise ValueError("V2 publication file layout mismatch")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("release_lane") != "v2-worldwide":
        raise ValueError("V2 release lane mismatch")
    table = pq.read_table(directory / "sentences.parquet")
    if manifest.get("parquet_sha256") != _sha256(directory / "sentences.parquet"):
        raise ValueError("V2 Parquet hash mismatch")
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != {
        "sentences.parquet",
        "assets/label_distribution.png",
        "assets/h3_sentence_distribution.png",
    }:
        raise ValueError("V2 artifact hashes are incomplete")
    for name, digest in artifact_hashes.items():
        if digest != _sha256(directory / name):
            raise ValueError("V2 artifact hash mismatch")
    required = {
        "place_relevance",
        "yes_logprob",
        "no_logprob",
        "logit_margin",
        "two_class_probability",
    }
    if not required.issubset(table.column_names):
        raise ValueError("V2 output is missing score columns")
    for row in table.to_pylist():
        record = V2LogitRecord(
            sentence_id=row["sentence_id"],
            place_relevance=row["place_relevance"],
            yes_logprob=row["yes_logprob"],
            no_logprob=row["no_logprob"],
        )
        if (
            record.logit_margin != row["logit_margin"]
            or record.two_class_probability != row["two_class_probability"]
        ):
            raise ValueError("V2 derived score mismatch")
    identity = manifest["run_identity"]
    if identity.get("prompt_version") != V2_LOGIT_PROMPT_VERSION:
        raise ValueError("V2 prompt version mismatch")
    if identity.get("model_repo_id") != V2_MODEL_REPO_ID:
        raise ValueError("V2 model repository mismatch")
    if identity.get("model_file") != V2_MODEL_FILE:
        raise ValueError("V2 model file mismatch")
    analytics = build_v2_analytics(
        table, h3_resolution=int(identity.get("h3_resolution") or 3)
    )
    persisted = manifest["statistics"]
    if persisted != {**analytics.to_dict(), "h3": persisted.get("h3")}:
        raise ValueError("V2 analytics mismatch")
    if (
        persisted.get("place_counts") != analytics.place_counts
        or persisted.get("unique_polygons") != analytics.unique_polygons
    ):
        raise ValueError("V2 analytics mismatch")
    expected_card = _render_card(
        dataset_repo_id=str(manifest.get("dataset_repo_id", "")),
        table=table,
        parquet_bytes=(directory / "sentences.parquet").stat().st_size,
        identity=identity,
        analytics=analytics.to_dict(),
        h3=persisted["h3"],
    )
    if (directory / "README.md").read_text() != expected_card:
        raise ValueError("V2 README does not match derived facts")
    return V2Publication(
        directory=directory,
        row_count=table.num_rows,
        parquet_sha256=_sha256(directory / "sentences.parquet"),
        files=tuple(directory / name for name in V2_PUBLICATION_FILES),
    )


__all__ = [
    "V2_PUBLICATION_FILES",
    "V2Publication",
    "finalize_v2_dataset",
    "validate_v2_publication",
]
