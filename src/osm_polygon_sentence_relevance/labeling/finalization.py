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


_HERO_IMAGE_URL = (
    "https://raw.githubusercontent.com/NoeFlandre/"
    "osm-polygon-wikidata-sentence-relevance/main/docs/assets/"
    "afghanistan-labeling-hero.png"
)


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


def _server_config(identity: dict[str, Any]) -> dict[str, int]:
    """Return the public, content-free server configuration for the manifest."""

    return {
        "llama_parallel": int(identity["llama_parallel"]),
        "llama_per_slot_context": int(identity["llama_per_slot_context"]),
        "llama_total_context": int(identity["llama_total_context"]),
        "request_concurrency": int(identity["request_concurrency"]),
    }


def _validate_split_timing(timing: dict[str, Any]) -> None:
    """Ensure the persisted timing uses the split inference schema."""

    if "initial_inference_seconds" not in timing:
        raise LabelFinalizationError("timing is missing initial_inference_seconds")
    if "repair_inference_seconds" not in timing:
        raise LabelFinalizationError("timing is missing repair_inference_seconds")
    if "inference_seconds" not in timing:
        raise LabelFinalizationError("timing is missing inference_seconds")
    initial = float(timing["initial_inference_seconds"])
    repair = float(timing["repair_inference_seconds"])
    inference = float(timing["inference_seconds"])
    if initial < 0 or repair < 0 or inference < 0:
        raise LabelFinalizationError("timing components must be non-negative")
    if abs(inference - (initial + repair)) > 1e-6:
        raise LabelFinalizationError(
            "inference_seconds must equal initial + repair components"
        )


def _manifest(
    *,
    dataset_repo_id: str,
    parquet_sha256: str,
    artifact_sha256: dict[str, str],
    statistics: dict[str, Any],
    identity: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    """Build the publication manifest with explicit server configuration."""

    _validate_split_timing(timing)
    return {
        "schema_version": 1,
        "dataset_repo_id": dataset_repo_id,
        "parquet_sha256": parquet_sha256,
        "artifact_sha256": artifact_sha256,
        "statistics": statistics,
        "run_identity": identity,
        "server_config": _server_config(identity),
        "timing": timing,
    }


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
    land_reasons = stats.get("landuse_reasons", {})
    polygon_reasons = stats.get("polygon_reasons", {})
    positive_languages = stats.get("positive_languages", {})

    def percent(count: int) -> str:
        if row_count == 0:
            return "0.00%"
        return f"{count / row_count * 100:.2f}%"

    def value(counts: dict[str, int], key: str) -> str:
        count = counts.get(key, 0)
        return f"{count:,} ({percent(count)})"

    server_config = _server_config(identity)
    initial_inference = float(timing.get("initial_inference_seconds", 0.0))
    repair_inference = float(timing.get("repair_inference_seconds", 0.0))
    total_inference = float(timing.get("inference_seconds", 0.0))
    total_wall = float(timing.get("total_wall_seconds", 0.0))
    repair_total = int(timing.get("repair_rows_total", 0))
    repair_succeeded = int(timing.get("repair_rows_succeeded", 0))
    repair_exhausted = int(timing.get("repair_rows_exhausted", 0))
    completed = int(timing.get("completed", row_count))
    total = int(timing.get("total", row_count))
    interrupted = bool(timing.get("interrupted", False))
    throughput = (completed / total_inference) if total_inference > 0 else 0.0

    scope = (
        f"This is a representative **{row_count:,}-row canary** selected "
        "deterministically for source and language coverage."
        if identity.get("row_limit", 0)
        else "This release labels the complete Afghanistan input."
    )

    repair_summary = (
        f"{repair_succeeded:,} repaired of {repair_total:,} attempted "
        f"({repair_exhausted:,} exhausted after repair attempts)"
        if repair_total
        else "No repair attempts were needed."
    )

    language_lines = (
        "\n".join(
            f"- {language}: {count:,}" for language, count in positive_languages.items()
        )
        or "- (none)"
    )

    land_reason_lines = (
        "\n".join(f"- `{code}`: {count:,}" for code, count in land_reasons.items())
        or "- (none)"
    )
    polygon_reason_lines = (
        "\n".join(f"- `{code}`: {count:,}" for code, count in polygon_reasons.items())
        or "- (none)"
    )

    allocation_lines = (
        f"- Completed allocations: {max(1, total // 5000):,}"
        if not interrupted
        else f"- Last run interrupted at {completed:,}/{total:,} rows"
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

![Afghanistan sentence relevance dataset overview]({_HERO_IMAGE_URL})

# Afghanistan polygon sentence relevance labels

> **Warning:** the labels below are **model-generated**. They are not ground truth and must be audited before use as authoritative training data.

This release contains **{row_count:,} labeled sentences** from the Afghanistan-only sentence dataset. {scope} Each row independently records two boolean decisions:

1. **Land use / land cover relevance** -- does the target sentence describe land use or land cover for the polygon?
2. **Target polygon relevance** -- does the target sentence describe the *named polygon* itself?

Every decision also carries a short reason code selected from the closed enumerations in the prompt.

## Label summary

The valid label values are **yes**, **no**, and **uncertain** (lowercase). Every row carries one label per question:

| Question | yes | no | uncertain |
|---|---:|---:|---:|
| Land use / land cover | {value(land, "yes")} | {value(land, "no")} | {value(land, "uncertain")} |
| Target polygon | {value(polygon, "yes")} | {value(polygon, "no")} | {value(polygon, "uncertain")} |

## Label and reason codes

Land-use / land-cover reasons:

{land_reason_lines}

Polygon-relevance reasons:

{polygon_reason_lines}

Positive-label language coverage (top languages, sorted by count):

{language_lines}

![Label distributions](https://huggingface.co/datasets/{dataset_repo_id}/resolve/main/assets/label_distribution.png)

![Positive-label languages](https://huggingface.co/datasets/{dataset_repo_id}/resolve/main/assets/positive_languages.png)

## Method

The labeler used `{identity["model_repo_id"]}` (`{identity["model_file"]}`), pinned at model revision `{identity["model_revision"]}`, through `{identity["engine"]} {identity["engine_version"]}`. Prompt `{identity["prompt_version"]}` supplied the model with:

- the **target sentence**
- the immediately **adjacent** sentences
- the **polygon name**
- the **region / country** (`{identity.get("region", "afghanistan")}`)
- the **language** code
- the **page title** and **section title**
- the **primary OSM tag**
- every other **OSM tag**

Structured output was validated against the closed enumerations and re-issued under repair until either a valid label pair was produced or the per-row retry budget was exhausted.

## Model provenance

- Repository: [`{identity["model_repo_id"]}`](https://huggingface.co/{identity["model_repo_id"]}/tree/{identity["model_revision"]})
- Quantised weights: `{identity["model_file"]}` (SHA-256 `{identity["model_file_sha256"]}`)
- Engine: `{identity["engine"]} {identity["engine_version"]}`
- Prompt version: `{identity["prompt_version"]}`
- Server configuration:
  - `llama_parallel`: `{server_config["llama_parallel"]}`
  - `llama_per_slot_context`: `{server_config["llama_per_slot_context"]}`
  - `llama_total_context`: `{server_config["llama_total_context"]}`
  - `request_concurrency`: `{server_config["request_concurrency"]}`

## Repair

{repair_summary}

- Initial inference: **{initial_inference:.2f} seconds**
- Repair inference: **{repair_inference:.2f} seconds**
- Combined inference: **{total_inference:.2f} seconds**

## Runtime

- Total elapsed: **{total_wall:.2f} seconds**
- Throughput: **{throughput:.3f} rows/second**
- Rows completed: **{completed:,} / {total:,}**

{allocation_lines}

## Provenance

- Input dataset revision: [`{identity["input_dataset_revision"]}"`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance/tree/{identity["input_dataset_revision"]})
- Input Parquet SHA-256: `{identity["input_sha256"]}`
- Source commit: `{identity["source_commit"]}`

The original sentence and polygon metadata are preserved. Added fields are `landuse_relevance`, `polygon_relevance`, `landuse_reason`, `polygon_reason`, and `label_evidence`. See `manifest.json` for the exact counts, hashes, run identity, and server configuration. Code and reproduction instructions: [`NoeFlandre/osm-polygon-wikidata-sentence-relevance`](https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance).
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
        manifest = _manifest(
            dataset_repo_id=dataset_repo_id,
            parquet_sha256=_sha256(parquet_path),
            artifact_sha256=artifact_sha256,
            statistics=stats,
            identity=store.identity.to_dict(),
            timing=timing,
        )
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
        path.relative_to(directory)
        for path in directory.rglob("*")
        if path.is_file() and not path.name.startswith(".gitattributes")
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
    server_config = manifest.get("server_config")
    if not isinstance(server_config, dict):
        raise LabelFinalizationError("server_config manifest is missing")
    for key in (
        "llama_parallel",
        "llama_per_slot_context",
        "llama_total_context",
        "request_concurrency",
    ):
        if key not in server_config:
            raise LabelFinalizationError(
                f"server_config is missing required field: {key}"
            )
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
    _validate_split_timing(manifest.get("timing", {}))
    for field in ("landuse_relevance", "polygon_relevance"):
        if stats.get(field) != _distribution(table[field].to_pylist()):
            raise LabelFinalizationError("labeled publication statistics mismatch")
    persisted_card = (directory / "README.md").read_text()
    rendered_card = _render_card(
        dataset_repo_id=str(manifest["dataset_repo_id"]),
        row_count=table.num_rows,
        stats=stats,
        identity=manifest["run_identity"],
        timing=manifest["timing"],
    )
    if rendered_card != persisted_card:
        raise LabelFinalizationError("labeled dataset card has drifted from data")
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
