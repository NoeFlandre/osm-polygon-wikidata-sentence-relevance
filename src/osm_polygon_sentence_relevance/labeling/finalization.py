"""Finalize complete labels into a factual publishable dataset."""

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

from .canary import select_canary_rows
from .checkpoint import CheckpointStore


class LabelFinalizationError(RuntimeError):
    """Raised when complete labeled output cannot be proven."""


@dataclass(frozen=True, slots=True)
class ValidatedLabeledPublication:
    """Validated facts used by publication."""

    directory: Path
    row_count: int
    parquet_sha256: str
    files: tuple[Path, ...]


_FILES = (
    "sentences.parquet",
    "manifest.json",
    "README.md",
    "assets/label_distribution.png",
    "assets/positive_languages.png",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_plots(table: pa.Table, assets: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional extra boundary
        raise LabelFinalizationError(
            "install the hub extra to render labeling plots"
        ) from exc
    assets.mkdir(mode=0o700)
    land = Counter(table["landuse_relevance"].to_pylist())
    polygon = Counter(table["polygon_relevance"].to_pylist())
    labels = ["yes", "no", "uncertain"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    for axis, title, counts in zip(
        axes, ("Land use / cover", "Polygon relevance"), (land, polygon), strict=True
    ):
        values = [counts.get(label, 0) for label in labels]
        bars = axis.bar(labels, values, color=["#2878B5", "#C44E52", "#999999"])
        axis.set_title(title)
        axis.set_ylabel("Sentences")
        axis.bar_label(bars, labels=[f"{value:,}" for value in values])
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Afghanistan relevance labels")
    fig.tight_layout()
    fig.savefig(assets / "label_distribution.png", metadata={"Software": ""})
    plt.close(fig)

    languages = table["language"].to_pylist()
    positives = table["landuse_relevance"].to_pylist()
    counts = Counter(
        language
        for language, value in zip(languages, positives, strict=True)
        if value == "yes"
    )
    top = counts.most_common(15)
    other = sum(counts.values()) - sum(value for _, value in top)
    if other:
        top.append(("Other", other))
    fig, axis = plt.subplots(figsize=(11, 7), dpi=120)
    names = [name for name, _ in reversed(top)]
    values = [value for _, value in reversed(top)]
    bars = axis.barh(names, values, color="#2878B5")
    axis.set_title("Languages among land-use / land-cover positive sentences")
    axis.set_xlabel("Sentences")
    axis.bar_label(bars, labels=[f"{value:,}" for value in values], padding=3)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(assets / "positive_languages.png", metadata={"Software": ""})
    plt.close(fig)
    for path in assets.iterdir():
        os.chmod(path, 0o600)


def _distribution(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _render_card(
    *,
    dataset_repo_id: str,
    row_count: int,
    stats: dict[str, Any],
    identity: dict[str, Any],
    timing: dict[str, Any],
) -> str:
    land = stats["landuse_relevance"]
    polygon = stats["polygon_relevance"]

    def value(counts: dict[str, int], key: str) -> str:
        count = counts.get(key, 0)
        return f"{count:,} ({count / row_count * 100:.2f}%)"

    scope = (
        f"This is a representative **{row_count:,}-row canary** selected "
        "deterministically for source and language coverage."
        if identity.get("row_limit", 0)
        else "This release labels the complete Afghanistan input."
    )

    return f"""---
license: apache-2.0
task_categories:
- text-classification
language:
- multilingual
pretty_name: Afghanistan polygon sentence relevance labels
configs:
- config_name: default
  data_files:
  - split: train
    path: sentences.parquet
---

# Afghanistan polygon sentence relevance labels

This proof of concept contains **{row_count:,} labeled sentences** from the Afghanistan-only sentence dataset. {scope} Each row independently records whether its target sentence is relevant to land use or land cover and whether it is relevant to its associated OSM polygon.

## Label summary

| Question | Yes | No | Uncertain |
|---|---:|---:|---:|
| Land use / land cover | {value(land, "yes")} | {value(land, "no")} | {value(land, "uncertain")} |
| Target polygon | {value(polygon, "yes")} | {value(polygon, "no")} | {value(polygon, "uncertain")} |

![Label distributions](https://huggingface.co/datasets/{dataset_repo_id}/resolve/main/assets/label_distribution.png)

![Positive-label languages](https://huggingface.co/datasets/{dataset_repo_id}/resolve/main/assets/positive_languages.png)

## Method

The labeler used `{identity["model_repo_id"]}` (`{identity["model_file"]}`), pinned at `{identity["model_revision"]}`, through `{identity["engine"]} {identity["engine_version"]}`. Prompt `{identity["prompt_version"]}` supplied the target and adjacent sentences, polygon name, country/region, language, page and section titles, the primary OSM tag, and every OSM tag. Structured output was validated before checkpointing. Labels are model-generated and should be audited before use as ground truth.

## Provenance and runtime

- Input dataset revision: `{identity["input_dataset_revision"]}`
- Input Parquet SHA-256: `{identity["input_sha256"]}`
- End-to-end labeling wall time: **{float(timing.get("total_wall_seconds", 0)):.2f} seconds**
- Model inference time: **{float(timing.get("inference_seconds", 0)):.2f} seconds**

The original sentence and polygon metadata are preserved. Added fields are `landuse_relevance`, `polygon_relevance`, `landuse_reason`, `polygon_reason`, and `label_evidence`. See `manifest.json` for exact counts, hashes, and run identity.
"""


def finalize_labeled_dataset(
    *, input_path: Path, store: CheckpointStore, output_dir: Path, dataset_repo_id: str
) -> ValidatedLabeledPublication:
    """Join complete checkpoints to input and build a factual publication."""

    input_path = Path(input_path)
    if _sha256(input_path) != store.identity.input_sha256:
        raise LabelFinalizationError(
            "input Parquet SHA-256 does not match run identity"
        )
    table = select_canary_rows(pq.read_table(input_path), store.identity.row_limit)
    regions = set(table["region"].to_pylist())
    if regions != {"afghanistan"}:
        raise LabelFinalizationError(
            "labeling finalization is restricted to Afghanistan"
        )
    records = store.load_all()
    by_id = {record.sentence_id: record for record in records}
    ids = table["sentence_id"].to_pylist()
    if len(records) != table.num_rows or set(by_id) != set(ids):
        raise LabelFinalizationError(
            "finalization requires exactly one label per input sentence"
        )
    ordered = [by_id[value] for value in ids]
    additions = {
        "landuse_relevance": [r.landuse_relevance.value for r in ordered],
        "polygon_relevance": [r.polygon_relevance.value for r in ordered],
        "landuse_reason": [r.landuse_reason for r in ordered],
        "polygon_reason": [r.polygon_reason for r in ordered],
        "label_evidence": [r.evidence for r in ordered],
    }
    for name, values in additions.items():
        table = table.append_column(name, pa.array(values, type=pa.string()))
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise LabelFinalizationError("output directory must not already exist")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        os.chmod(staging, 0o700)
        parquet_path = staging / "sentences.parquet"
        pq.write_table(table, parquet_path, compression="zstd")
        os.chmod(parquet_path, 0o600)
        stats: dict[str, Any] = {
            "row_count": table.num_rows,
            "landuse_relevance": _distribution(additions["landuse_relevance"]),
            "polygon_relevance": _distribution(additions["polygon_relevance"]),
            "landuse_reasons": _distribution(additions["landuse_reason"]),
            "polygon_reasons": _distribution(additions["polygon_reason"]),
            "joint_labels": _distribution(
                [
                    f"{land}|{polygon}"
                    for land, polygon in zip(
                        additions["landuse_relevance"],
                        additions["polygon_relevance"],
                        strict=True,
                    )
                ]
            ),
            "positive_languages": _distribution(
                [
                    language
                    for language, label in zip(
                        table["language"].to_pylist(),
                        additions["landuse_relevance"],
                        strict=True,
                    )
                    if label == "yes"
                ]
            ),
        }
        timing_path = store.root / "timing.json"
        timing = json.loads(timing_path.read_text()) if timing_path.is_file() else {}
        _render_plots(table, staging / "assets")
        artifact_sha256 = {
            name: _sha256(staging / name)
            for name in (
                "sentences.parquet",
                "assets/label_distribution.png",
                "assets/positive_languages.png",
            )
        }
        manifest = {
            "schema_version": 1,
            "dataset_repo_id": dataset_repo_id,
            "parquet_sha256": _sha256(parquet_path),
            "artifact_sha256": artifact_sha256,
            "statistics": stats,
            "run_identity": store.identity.to_dict(),
            "timing": timing,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (staging / "README.md").write_text(
            _render_card(
                dataset_repo_id=dataset_repo_id,
                row_count=table.num_rows,
                stats=stats,
                identity=store.identity.to_dict(),
                timing=timing,
            )
        )
        for name in ("manifest.json", "README.md"):
            os.chmod(staging / name, 0o600)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_labeled_publication(output_dir)


def validate_labeled_publication(directory: Path) -> ValidatedLabeledPublication:
    """Validate the closed publication layout and all factual identities."""

    directory = Path(directory)
    expected = {Path(name) for name in _FILES}
    actual = {
        path.relative_to(directory) for path in directory.rglob("*") if path.is_file()
    }
    if actual != expected:
        raise LabelFinalizationError("labeled publication file layout mismatch")
    manifest = json.loads((directory / "manifest.json").read_text())
    parquet = directory / "sentences.parquet"
    if manifest.get("parquet_sha256") != _sha256(parquet):
        raise LabelFinalizationError("labeled Parquet SHA-256 mismatch")
    artifact_sha256 = manifest.get("artifact_sha256")
    if not isinstance(artifact_sha256, dict):
        raise LabelFinalizationError("artifact SHA-256 manifest is missing")
    for name in (
        "sentences.parquet",
        "assets/label_distribution.png",
        "assets/positive_languages.png",
    ):
        if artifact_sha256.get(name) != _sha256(directory / name):
            raise LabelFinalizationError("artifact SHA-256 mismatch")
    table = pq.read_table(parquet)
    stats = manifest.get("statistics", {})
    if stats.get("row_count") != table.num_rows:
        raise LabelFinalizationError("labeled publication row count mismatch")
    for field in ("landuse_relevance", "polygon_relevance"):
        if stats.get(field) != _distribution(table[field].to_pylist()):
            raise LabelFinalizationError("labeled publication statistics mismatch")
    return ValidatedLabeledPublication(
        directory=directory,
        row_count=table.num_rows,
        parquet_sha256=_sha256(parquet),
        files=tuple(directory / name for name in _FILES),
    )


__all__ = [
    "LabelFinalizationError",
    "ValidatedLabeledPublication",
    "finalize_labeled_dataset",
    "validate_labeled_publication",
]
