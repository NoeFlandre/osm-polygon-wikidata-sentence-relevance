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

from .analytics import (
    H3_MAP_ASSET_NAME,
    build_label_analytics,
    h3_sentence_distribution,
    render_analytics_assets,
    render_h3_sentence_distribution,
)
from .checkpoint import CheckpointStore
from .contracts import LabelRecord
from .releases import (
    V1_TRACKIO_SPACE_ID,
    ReleaseLane,
    release_lane,
    release_prefix,
    trackio_space_url,
)
from .sampling import select_label_rows


class LabelFinalizationError(RuntimeError):
    """Raised when complete labeled output cannot be proven."""


_HERO_IMAGE_URL = (
    "https://raw.githubusercontent.com/NoeFlandre/"
    "osm-polygon-wikidata-sentence-relevance/main/docs/assets/"
    "afghanistan-labeling-hero.png"
)
# Backwards-compatible name for callers that log the historical V1 release.
TRACKIO_SPACE_ID = V1_TRACKIO_SPACE_ID
TRACKIO_SPACE_URL = trackio_space_url(ReleaseLane.V1_AFGHANISTAN)
INPUT_DATASET_URL = (
    "https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only"
)
GITHUB_REPO_URL = (
    "https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance"
)
PRESENTATION_URL = (
    "https://noeflandre.github.io/osm-polygon-wikidata-sentence-relevance/"
    "presentations/afghanistan-dataset-overview/index.html"
)


@dataclass(frozen=True, slots=True)
class ValidatedLabeledPublication:
    """Validated facts used by publication."""

    directory: Path
    row_count: int
    parquet_sha256: str
    files: tuple[Path, ...]


_BASE_FILES = (
    "sentences.parquet",
    "manifest.json",
    "README.md",
    "assets/label_distribution.png",
    "assets/positive_languages.png",
    "assets/joint_label_heatmap.png",
    "assets/polygon_coverage_funnel.png",
    "assets/reason_code_distribution.png",
)
_V2_FILES = (*_BASE_FILES, f"assets/{H3_MAP_ASSET_NAME}")
_ANALYTICS_ASSETS = _BASE_FILES[5:]
_ANALYTICS_COLUMNS = (
    "polygon_id",
    "language",
    "source",
    "osm_primary_tag",
    "landuse_relevance",
    "polygon_relevance",
    "landuse_reason",
    "polygon_reason",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publication_revision(identity: dict[str, Any]) -> str:
    """Return the single Hugging Face revision used for public releases."""

    del identity
    return "main"


def _scope_label(table: pa.Table) -> str:
    """Derive a concise scope label from the finalized rows."""

    if "region" not in table.column_names:
        return "Dataset"
    values = {
        str(value).strip()
        for value in table["region"].to_pylist()
        if value is not None and str(value).strip()
    }
    if len(values) != 1:
        return "Global" if len(values) > 1 else "Dataset"
    value = next(iter(values)).removesuffix("-latest")
    return value.replace("_", " ").replace("-", " ").title() or "Dataset"


def _identity_scope_label(identity: dict[str, Any]) -> str:
    """Return the legacy scope used when rendering a card without its table."""

    if release_lane(identity) is ReleaseLane.V2_WORLDWIDE:
        return "Worldwide"

    region = identity.get("region")
    if isinstance(region, str) and region.strip():
        value = region.strip().removesuffix("-latest")
        return value.replace("_", " ").replace("-", " ").title()
    return "Afghanistan"


def _release_files(identity: dict[str, Any]) -> tuple[str, ...]:
    """Return the local file layout required by this release lane."""

    return (
        _V2_FILES if release_lane(identity) is ReleaseLane.V2_WORLDWIDE else _BASE_FILES
    )


def _render_plots(
    table: pa.Table, assets: Path, *, scope_label: str = "Afghanistan"
) -> None:
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
    fig.suptitle(f"{scope_label} relevance labels")
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
    axis.set_title(
        f"Languages among {scope_label.lower()} land-use / land-cover positive sentences"
    )
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


def _analytics_table(table: pa.Table) -> pa.Table:
    """Supply null dimensions for legacy fixtures lacking optional metadata.

    Production labeling always carries these columns. Keeping null columns for
    older checkpoint fixtures preserves the publication contract without
    inventing polygon, source, or tag values.
    """

    for name in _ANALYTICS_COLUMNS:
        if name not in table.column_names:
            table = table.append_column(
                name, pa.nulls(table.num_rows, type=pa.string())
            )
    return table


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
    publication_revision: str = "main",
) -> dict[str, Any]:
    """Build the publication manifest with explicit server configuration."""

    _validate_split_timing(timing)
    return {
        "schema_version": 1,
        "dataset_repo_id": dataset_repo_id,
        "publication_revision": publication_revision,
        "release_lane": release_lane(identity).value,
        "release_prefix": release_prefix(release_lane(identity)),
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
    scope_label: str | None = None,
    publication_revision: str | None = None,
) -> str:
    land = stats["landuse_relevance"]
    polygon = stats["polygon_relevance"]
    analytics = stats["analytics"]

    def percent(count: int) -> str:
        if row_count == 0:
            return "0.00%"
        return f"{count / row_count * 100:.2f}%"

    def value(counts: dict[str, int], key: str) -> str:
        count = counts.get(key, 0)
        return f"{count:,} ({percent(count)})"

    server_config = _server_config(identity)

    scope_label = scope_label or _identity_scope_label(identity)
    publication_revision = publication_revision or _publication_revision(identity)
    lane = release_lane(identity)
    is_v1_afghanistan = lane is ReleaseLane.V1_AFGHANISTAN
    remote_prefix = release_prefix(lane)
    remote_data_path = f"{remote_prefix}/sentences.parquet"
    if identity.get("row_limit", 0):
        scope = (
            f"This is a representative **{row_count:,}-row canary** selected "
            "deterministically for source and language coverage."
        )
    elif identity.get("sampling_version") is not None and identity.get(
        "sampling_target", 0
    ):
        scope = (
            f"This release contains a deterministic stratified sample of "
            f"**{row_count:,} rows** from the {scope_label} input."
        )
    else:
        scope = f"This release labels the complete {scope_label} input."
    asset_base = f"https://huggingface.co/datasets/{dataset_repo_id}/resolve/main/"
    asset_base += f"{remote_prefix}/assets"
    pretty_name = f"{scope_label} polygon sentence relevance labels"
    hero = (
        f"![{scope_label} sentence relevance dataset overview]({_HERO_IMAGE_URL})\n\n"
        if is_v1_afghanistan
        else ""
    )
    release_trackio_url = trackio_space_url(lane)
    release_links = (
        "## Release lane\n\n"
        "This is the V1 Afghanistan release, published under\n"
        "`v1-afghanistan/`. The worldwide stratified release is published\n"
        "separately under `v2-worldwide/` on the same HF `main` revision.\n\n"
        "## Trackio\n\n"
        "The metrics and plots for this release are logged in the\n"
        f"[public Trackio dashboard]({release_trackio_url}).\n\n"
        f"For a visual introduction, see the [{scope_label} dataset presentation]({PRESENTATION_URL}).\n\n"
        if is_v1_afghanistan
        else (
            "## Release lane\n\n"
            "This is the V2 worldwide stratified release. Its files live under\n"
            "`v2-worldwide/` on the same HF `main` revision as the preserved V1\n"
            "Afghanistan files.\n\n"
            "## Trackio\n\n"
            "Batch progress and final metrics are logged in the separate\n"
            f"[worldwide Trackio dashboard]({release_trackio_url}).\n\n"
        )
    )
    slice_note = (
        "The public Trackio dashboard provides an interactive slice table for language, "
        "source, and `osm_primary_tag`. It shows both-yes rate, uncertain rate, and "
        "sample size; groups smaller than 100 sentences are omitted."
        if is_v1_afghanistan
        else "The separate worldwide Trackio dashboard provides an interactive slice "
        "table for language, source, and `osm_primary_tag`; groups smaller than 100 "
        "sentences are omitted."
    )

    def rate(value: float) -> str:
        return f"{value * 100:.2f}%"

    joint_lines = "\n".join(
        f"| {land} | {polygon} | {analytics['joint_counts'][f'{land}|{polygon}']:,} | "
        f"{rate(analytics['joint_percentages'][f'{land}|{polygon}'])} |"
        for land in ("yes", "no", "uncertain")
        for polygon in ("yes", "no", "uncertain")
    )
    funnel = analytics["coverage_funnel"]
    slice_count = len(analytics.get("slices", []))

    h3_note = ""
    if lane is ReleaseLane.V2_WORLDWIDE:
        h3_stats = stats.get("h3_sentence_distribution", {})
        h3_note = (
            "\n### Geographic distribution\n\n"
            "The map shows every labeled sentence assigned to its H3 cell. "
            f"Resolution `{h3_stats.get('resolution', identity.get('h3_resolution'))}`; "
            f"{h3_stats.get('cell_count', 0):,} occupied cells and "
            f"{h3_stats.get('missing_coordinate_count', 0):,} rows without coordinates.\n\n"
            f"![Worldwide labeled-sentence distribution by H3 cell]({asset_base}/{H3_MAP_ASSET_NAME})\n"
        )

    return f"""---
license: apache-2.0
task_categories:
- text-classification
language:
- multilingual
pretty_name: {pretty_name}
configs:
- config_name: v1-afghanistan
  data_files:
  - split: train
    path: {remote_data_path}
---

{hero}# {pretty_name}

> **Warning:** the labels below are **model-generated**. They are not ground truth and must be audited before use as authoritative training data.

This release contains **{row_count:,} labeled sentences** extracted from the
[NoeFlandre/osm-polygon-wikidata-only dataset]({INPUT_DATASET_URL}). {scope} Each
row independently records two boolean decisions:

Code and reproduction instructions are in the [GitHub repository]({GITHUB_REPO_URL}).

1. **Land use / land cover relevance** -- does the target sentence describe land use or land cover for the polygon?
2. **Target polygon relevance** -- does the target sentence describe the *named polygon* itself?

Every decision also carries a short reason code selected from the closed enumerations in the prompt.

## Dataset metrics

| Metric | Value |
|---|---:|
| Labeled sentences | {analytics["total_labeled_sentences"]:,} |
| Unique polygons | {analytics["unique_polygons"]:,} |
| Unique languages | {analytics["unique_languages"]:,} |
| Strong-positive yield (both labels yes) | {rate(analytics["strong_positive_yield"])} |

## Sentence preparation

The sentences were extracted from the linked upstream dataset after its article
and polygon joins. Each source section was consumed once by the multilingual
`wtpsplit` SaT model (`sat-12l-sm`). SaT proposes boundaries; it does not rewrite
the text. A conservative repair then separates only high-confidence punctuation
boundaries that remain inside a model segment. It keeps terminal punctuation
with the preceding sentence and avoids splitting abbreviations, initials,
lowercase or numeric continuations, and URL query strings. For Arabic-tagged
rows, a period boundary also requires the following clause to begin in Arabic
script.

Each segment is trimmed and normalized deterministically: Unicode NFC is
applied, selected zero-width and control characters are handled, whitespace is
collapsed, and leading MediaWiki edit markers are removed. Case, punctuation,
accents, ZWNJ, and ZWJ are preserved. Empty normalized segments are dropped.
Adjacent context is assigned before deduplication, then exact duplicates are
collapsed by `(polygon_id, language, sentence_text_normalized)` using a stable
canonical occurrence. Publication scans the final table with the same
high-confidence boundary predicate and refuses any residual embedded boundary.

## Sentence labeling

The labeler used `{identity["model_repo_id"]}` (`{identity["model_file"]}`), pinned at model revision `{identity["model_revision"]}`, through `{identity["engine"]} {identity["engine_version"]}`. Prompt `{identity["prompt_version"]}` supplied the model with:

- the **target sentence**
- the immediately **adjacent** sentences
- the **polygon name**
- the **region / country** (`{scope_label}`)
- the **language** code
- the **page title** and **section title**
- the **primary OSM tag**
- every other **OSM tag**

Structured output was validated against the closed enumerations and re-issued under repair until either a valid label pair was produced or the per-row retry budget was exhausted.

## Label summary

The valid label values are **yes**, **no**, and **uncertain** (lowercase). Every row carries one label per question:

| Question | yes | no | uncertain |
|---|---:|---:|---:|
| Land use / land cover | {value(land, "yes")} | {value(land, "no")} | {value(land, "uncertain")} |
| Target polygon | {value(polygon, "yes")} | {value(polygon, "no")} | {value(polygon, "uncertain")} |

{release_links}### Joint labels

Counts and percentages use all labeled sentences.

| Land use / land cover | Polygon relevance | Count | Share |
|---|---|---:|---:|
{joint_lines}

### Polygon coverage

The funnel counts unique polygons with at least one sentence meeting each condition.

| Stage | Polygons |
|---|---:|
| All polygons | {funnel["all_polygons"]:,} |
| At least one polygon-relevant sentence | {funnel["polygon_relevant_polygons"]:,} |
| At least one land-use / land-cover-relevant sentence | {funnel["landuse_relevant_polygons"]:,} |
| At least one sentence with both labels yes | {funnel["both_yes_polygons"]:,} |

### Reason-code distribution

The bars show the normalized share of all labeled sentences for each reason code.

![Reason-code distribution]({asset_base}/reason_code_distribution.png)

{slice_note} The release contains {slice_count:,} qualifying groups.

## Language coverage

![Positive-label languages]({asset_base}/positive_languages.png)

{h3_note}
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

## Provenance

- Input dataset revision: [`{identity["input_dataset_revision"]}`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance/tree/{identity["input_dataset_revision"]})
- Input Parquet SHA-256: `{identity["input_sha256"]}`
- Source commit: `{identity["source_commit"]}`

The original sentence and polygon metadata are preserved. Added fields are `landuse_relevance`, `polygon_relevance`, `landuse_reason`, `polygon_reason`, and `label_evidence`. See `manifest.json` for the exact counts, hashes, run identity, and server configuration. Code and reproduction instructions: [`NoeFlandre/osm-polygon-wikidata-sentence-relevance`](https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance).
"""


def _validate_replaceable_v2_output(
    directory: Path, current_identity: dict[str, Any]
) -> None:
    """Allow replacement only for a validated earlier V2 target."""

    if directory.is_symlink() or not directory.is_dir():
        raise LabelFinalizationError(
            "existing output must be a regular V2 publication directory"
        )
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelFinalizationError("existing V2 output manifest is invalid") from exc
    previous_identity = manifest.get("run_identity")
    if not isinstance(previous_identity, dict):
        raise LabelFinalizationError("existing V2 output identity is missing")
    if release_lane(previous_identity) is not ReleaseLane.V2_WORLDWIDE:
        raise LabelFinalizationError("existing output is not a V2 publication")
    previous_immutable = dict(previous_identity)
    current_immutable = dict(current_identity)
    previous_target = previous_immutable.pop("sampling_target", None)
    current_target = current_immutable.pop("sampling_target", None)
    if previous_immutable != current_immutable:
        raise LabelFinalizationError("existing V2 output identity does not match")
    if (
        isinstance(previous_target, bool)
        or not isinstance(previous_target, int)
        or isinstance(current_target, bool)
        or not isinstance(current_target, int)
        or current_target <= previous_target
    ):
        raise LabelFinalizationError("existing V2 output target is not expandable")
    try:
        validate_labeled_publication(directory)
    except LabelFinalizationError as exc:
        raise LabelFinalizationError("existing V2 output failed validation") from exc


def finalize_labeled_dataset(
    *, input_path: Path, store: CheckpointStore, output_dir: Path, dataset_repo_id: str
) -> ValidatedLabeledPublication:
    """Join complete checkpoints to input and build a factual publication."""

    input_path = Path(input_path)
    if _sha256(input_path) != store.identity.input_sha256:
        raise LabelFinalizationError(
            "input Parquet SHA-256 does not match run identity"
        )
    table = select_label_rows(
        pq.read_table(input_path),
        row_limit=store.identity.row_limit,
        sampling_target=store.identity.sampling_target,
        sampling_seed=store.identity.sampling_seed,
        h3_resolution=store.identity.h3_resolution,
    )
    records = store.load_all()
    by_id = {record.sentence_id: record for record in records}
    ids = table["sentence_id"].to_pylist()
    if len(records) != table.num_rows or set(by_id) != set(ids):
        raise LabelFinalizationError(
            "finalization requires exactly one label per input sentence"
        )
    ordered = [by_id[value] for value in ids]
    identity = store.identity.to_dict()
    lane = release_lane(identity)
    if len(ordered) != len(records) or not all(
        isinstance(record, LabelRecord) for record in ordered
    ):
        raise LabelFinalizationError("publication requires two-question labels")
    additions = {
        "landuse_relevance": [record.landuse_relevance.value for record in ordered],
        "polygon_relevance": [record.polygon_relevance.value for record in ordered],
        "landuse_reason": [record.landuse_reason for record in ordered],
        "polygon_reason": [record.polygon_reason for record in ordered],
        "label_evidence": [record.evidence for record in ordered],
    }
    for name, values in additions.items():
        table = table.append_column(name, pa.array(values, type=pa.string()))
    output_dir = Path(output_dir)
    regions = (
        {
            str(value).strip().removesuffix("-latest")
            for value in table["region"].to_pylist()
            if value is not None and str(value).strip()
        }
        if "region" in table.column_names
        else set()
    )
    if lane is ReleaseLane.V2_WORLDWIDE:
        if len(regions) < 2 or regions == {"afghanistan"}:
            raise LabelFinalizationError(
                "worldwide V2 publication requires multiple input regions"
            )
    elif regions and regions != {"afghanistan"}:
        raise LabelFinalizationError(
            "V1 Afghanistan publication cannot contain other regions"
        )
    # ``Path.exists()`` is false for a dangling symlink. Treat any symlink at
    # the output path as existing so replacement validation fails closed.
    previous_output = output_dir.exists() or output_dir.is_symlink()
    if previous_output:
        if release_lane(identity) is not ReleaseLane.V2_WORLDWIDE:
            raise LabelFinalizationError(
                "existing output may only be replaced by a V2 continuation"
            )
        _validate_replaceable_v2_output(output_dir, identity)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        os.chmod(staging, 0o700)
        parquet_path = staging / "sentences.parquet"
        pq.write_table(table, parquet_path, compression="zstd")
        os.chmod(parquet_path, 0o600)
        stats = {
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
        label_analytics = build_label_analytics(_analytics_table(table))
        stats["analytics"] = label_analytics.to_dict()
        timing_path = store.root / "timing.json"
        timing = json.loads(timing_path.read_text()) if timing_path.is_file() else {}
        scope_label = _scope_label(table)
        publication_revision = _publication_revision(identity)
        _render_plots(table, staging / "assets", scope_label=scope_label)
        render_analytics_assets(label_analytics, staging / "assets")
        if lane is ReleaseLane.V2_WORLDWIDE:
            stats["h3_sentence_distribution"] = render_h3_sentence_distribution(
                table,
                staging / "assets" / H3_MAP_ASSET_NAME,
                resolution=int(identity.get("h3_resolution") or 0),
                scope_label=scope_label,
            )
        artifact_names = (
            "assets/label_distribution.png",
            "assets/positive_languages.png",
            *_ANALYTICS_ASSETS,
        )
        artifact_names = ("sentences.parquet", *artifact_names)
        if lane is ReleaseLane.V2_WORLDWIDE:
            artifact_names += (f"assets/{H3_MAP_ASSET_NAME}",)
        artifact_sha256 = {name: _sha256(staging / name) for name in artifact_names}
        manifest = _manifest(
            dataset_repo_id=dataset_repo_id,
            parquet_sha256=_sha256(parquet_path),
            artifact_sha256=artifact_sha256,
            statistics=stats,
            identity=identity,
            timing=timing,
            publication_revision=publication_revision,
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (staging / "README.md").write_text(
            _render_card(
                dataset_repo_id=dataset_repo_id,
                row_count=table.num_rows,
                stats=stats,
                identity=identity,
                timing=timing,
                scope_label=scope_label,
                publication_revision=publication_revision,
            )
        )
        for name in ("manifest.json", "README.md"):
            os.chmod(staging / name, 0o600)
        backup: Path | None = None
        if previous_output:
            descriptor, backup_name = tempfile.mkstemp(
                prefix=f".{output_dir.name}.backup-", dir=output_dir.parent
            )
            os.close(descriptor)
            backup = Path(backup_name)
            backup.unlink()
            os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except BaseException:
            if backup is not None and backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_labeled_publication(output_dir)


def validate_labeled_publication(directory: Path) -> ValidatedLabeledPublication:
    """Validate the closed publication layout and all factual identities."""

    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelFinalizationError("labeled publication manifest is invalid") from exc
    identity = manifest.get("run_identity")
    if not isinstance(identity, dict):
        raise LabelFinalizationError("labeled publication identity is missing")
    lane = release_lane(identity)
    if manifest.get("release_lane") != lane.value:
        raise LabelFinalizationError("labeled publication release lane mismatch")
    if manifest.get("release_prefix") != release_prefix(lane):
        raise LabelFinalizationError("labeled publication release prefix mismatch")
    expected = {Path(name) for name in _release_files(identity)}
    actual = {
        path.relative_to(directory)
        for path in directory.rglob("*")
        if path.is_file() and not path.name.startswith(".gitattributes")
    }
    if actual != expected:
        raise LabelFinalizationError("labeled publication file layout mismatch")
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
    artifact_names = (
        "sentences.parquet",
        "assets/label_distribution.png",
        "assets/positive_languages.png",
        *_ANALYTICS_ASSETS,
        *((f"assets/{H3_MAP_ASSET_NAME}",) if lane is ReleaseLane.V2_WORLDWIDE else ()),
    )
    for name in artifact_names:
        if artifact_sha256.get(name) != _sha256(directory / name):
            raise LabelFinalizationError("artifact SHA-256 mismatch")
    table = pq.read_table(parquet)
    stats = manifest.get("statistics", {})
    if stats.get("row_count") != table.num_rows:
        raise LabelFinalizationError("labeled publication row count mismatch")
    _validate_split_timing(manifest.get("timing", {}))
    expected_revision = _publication_revision(identity)
    if manifest.get("publication_revision", expected_revision) != expected_revision:
        raise LabelFinalizationError(
            "labeled publication revision does not match identity"
        )
    for field in ("landuse_relevance", "polygon_relevance"):
        if stats.get(field) != _distribution(table[field].to_pylist()):
            raise LabelFinalizationError("labeled publication statistics mismatch")
    expected_analytics = build_label_analytics(_analytics_table(table)).to_dict()
    if stats.get("analytics") != expected_analytics:
        raise LabelFinalizationError("labeled publication analytics mismatch")
    if lane is ReleaseLane.V2_WORLDWIDE:
        expected_h3 = h3_sentence_distribution(
            table, resolution=int(identity.get("h3_resolution") or 0)
        )
        if stats.get("h3_sentence_distribution") != {
            "resolution": int(identity.get("h3_resolution") or 0),
            "cell_count": len(expected_h3[0]),
            "sentence_count": sum(expected_h3[0].values()),
            "missing_coordinate_count": expected_h3[1],
            "cells": expected_h3[0],
        }:
            raise LabelFinalizationError("labeled publication H3 statistics mismatch")
    persisted_card = (directory / "README.md").read_text()
    rendered_card = _render_card(
        dataset_repo_id=str(manifest["dataset_repo_id"]),
        row_count=table.num_rows,
        stats=stats,
        identity=identity,
        timing=manifest["timing"],
        scope_label=_scope_label(table),
        publication_revision=expected_revision,
    )
    if rendered_card != persisted_card:
        raise LabelFinalizationError("labeled dataset card has drifted from data")
    return ValidatedLabeledPublication(
        directory=directory,
        row_count=table.num_rows,
        parquet_sha256=_sha256(parquet),
        files=tuple(directory / name for name in _release_files(identity)),
    )


__all__ = [
    "LabelFinalizationError",
    "ValidatedLabeledPublication",
    "finalize_labeled_dataset",
    "validate_labeled_publication",
]
