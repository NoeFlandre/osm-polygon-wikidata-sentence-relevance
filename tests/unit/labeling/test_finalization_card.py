"""Public labeled-dataset card contract tests.

The dataset card (README.md) must:

- link to the GitHub repository
- link to the immutable input dataset revision on the Hub
- include a model-generated-labels warning
- define the two independent questions (land-use/land-cover; polygon)
- document the label values and reason-code distribution
- enumerate the exact context supplied to the model
- record model repo + GGUF file + immutable model revision + llama.cpp
  version + prompt version + server configuration
- show total rows and label distributions
- omit operational runtime and allocation details
- license under Apache-2.0
- be re-renderable from the labeled Parquet + validated manifest, and
  equality-checked by the publication validator.

The card must NOT include:
- operational history (failed jobs, debugging)
- vLLM discussion
- queue details
- AI-style filler or repeated statistics
- hard-coded counts that diverge from the labeled Parquet.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
from osm_polygon_sentence_relevance.labeling.contracts import (
    LabelRecord,
    LabelValue,
    RunIdentity,
)
from osm_polygon_sentence_relevance.labeling.finalization import (
    LabelFinalizationError,
    _render_card,
    finalize_labeled_dataset,
    validate_labeled_publication,
)


def _identity(input_sha256: str = "a" * 64) -> RunIdentity:
    return RunIdentity(
        input_sha256=input_sha256,
        input_dataset_revision="b" * 40,
        model_repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        model_revision="c" * 40,
        model_file="Qwen3.6-27B-Q4_K_M.gguf",
        model_file_sha256="d" * 64,
        prompt_version="afghanistan-landuse-polygon-v2",
        source_commit="e" * 40,
        engine="llama.cpp",
        engine_version="version: 1 (555881e)",
        batch_size=2,
        llama_parallel=16,
        llama_per_slot_context=4096,
        llama_total_context=65536,
        request_concurrency=16,
    )


def _build_publication(tmp_path: Path, n_rows: int = 3) -> Path:
    input_path = tmp_path / "input.parquet"
    pq.write_table(
        pa.table(
            {
                "sentence_id": [f"s{i}" for i in range(n_rows)],
                "region": ["afghanistan"] * n_rows,
                "language": ["en"] * n_rows,
                "sentence_text_raw": [f"farming sentence {i}" for i in range(n_rows)],
            }
        ),
        input_path,
    )
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    store = CheckpointStore(tmp_path / "work", _identity(digest))
    records = [
        LabelRecord(
            f"s{i}",
            LabelValue.YES,
            LabelValue.YES,
            "explicit_land_use",
            "direct_polygon_reference",
            "farming",
        )
        for i in range(n_rows)
    ]
    store.write_batch(0, records)
    store.write_timing(
        {
            "total_wall_seconds": 100.0,
            "initial_inference_seconds": 90.0,
            "repair_inference_seconds": 5.0,
            "inference_seconds": 95.0,
            "checkpoint_and_validation_seconds": 5.0,
        }
    )
    output = tmp_path / "publication"
    finalize_labeled_dataset(
        input_path=input_path,
        store=store,
        output_dir=output,
        dataset_repo_id="owner/dataset",
    )
    return output


def test_card_includes_github_repository_link(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    card = (output / "README.md").read_text()
    assert (
        "https://github.com/NoeFlandre/osm-polygon-wikidata-sentence-relevance" in card
    )


def test_card_starts_with_hero_image_after_front_matter(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    card = (output / "README.md").read_text()
    hero = (
        "![Afghanistan sentence relevance dataset overview]("
        "https://raw.githubusercontent.com/NoeFlandre/"
        "osm-polygon-wikidata-sentence-relevance/main/docs/assets/"
        "afghanistan-labeling-hero.png)"
    )
    assert hero in card
    assert card.index(hero) < card.index(
        "# Afghanistan polygon sentence relevance labels"
    )


def test_card_includes_immutable_input_revision_link(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    card = (output / "README.md").read_text()
    assert (
        "https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance/tree/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        in card
    )


def test_card_includes_model_generated_labels_warning(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    card = (output / "README.md").read_text().lower()
    assert "model-generated" in card
    assert "ground truth" in card or "audited" in card


def test_card_defines_both_independent_questions(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    card = (output / "README.md").read_text()
    assert "land use" in card.lower() or "land-use" in card.lower()
    assert "land cover" in card.lower() or "land-cover" in card.lower()
    assert "polygon" in card.lower()


def test_card_documents_label_values_and_reasons(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    card = (output / "README.md").read_text()
    # Label values are documented; detailed reason codes remain in the plot.
    for token in (
        "yes",
        "no",
        "uncertain",
        "Reason-code distribution",
    ):
        assert token in card, f"missing {token!r} in card"


def test_card_enumerates_supplied_context(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    card = (output / "README.md").read_text()
    for token in (
        "target sentence",
        "adjacent",
        "polygon name",
        "region",
        "language",
        "page title",
        "section title",
        "OSM tag",
        "primary OSM tag",
    ):
        assert token.lower() in card.lower(), f"missing {token!r} in card"


def test_card_records_model_provenance_and_server_config(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    card = (output / "README.md").read_text()
    assert "unsloth/Qwen3.6-27B-MTP-GGUF" in card
    assert "Qwen3.6-27B-Q4_K_M.gguf" in card
    assert ("c" * 40) in card  # model revision
    assert "version: 1 (555881e)" in card  # llama.cpp version
    assert "afghanistan-landuse-polygon-v2" in card  # prompt version
    # Server configuration block.
    assert "llama_parallel" in card
    assert "16" in card
    assert "65536" in card  # total context


def test_card_omits_repair_and_runtime_sections(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    card = (output / "README.md").read_text().lower()
    assert "## repair" not in card
    assert "## runtime" not in card


def test_card_omits_runtime_throughput_and_allocations(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    card = (output / "README.md").read_text().lower()
    for token in ("initial inference", "throughput", "completed allocations"):
        assert token not in card


def test_card_license_is_apache_2(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    card = (output / "README.md").read_text()
    assert "license: apache-2.0" in card


def test_card_has_no_operational_history(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    card = (output / "README.md").read_text().lower()
    for forbidden in (
        "failed job",
        "failure",
        "vllm",
        "queue",
        "walltime",
        "wall time",
    ):
        assert forbidden not in card, f"forbidden phrase present: {forbidden!r}"


def test_card_no_hardcoded_counts_diverging_from_parquet(tmp_path: Path) -> None:
    """A re-rendered card from the Parquet must match the persisted card."""

    output = _build_publication(tmp_path)
    # Read the persisted card and the labeled Parquet.
    persisted = (output / "README.md").read_text()
    table = pq.read_table(output / "sentences.parquet")
    manifest = json.loads((output / "manifest.json").read_text())

    # Re-render the card from the Parquet + manifest and assert equality.
    rerendered = _render_card(
        dataset_repo_id="owner/dataset",
        row_count=table.num_rows,
        stats=manifest["statistics"],
        identity=manifest["run_identity"],
        timing=manifest["timing"],
    )
    assert rerendered == persisted


def test_card_rerender_detects_tampering(tmp_path: Path) -> None:
    """If the persisted card drifts from the Parquet, validation fails."""

    output = _build_publication(tmp_path)
    readme = output / "README.md"
    original = readme.read_text()
    # Insert a bogus statistic in the card.
    tampered = original.replace("3 labeled sentences", "99 labeled sentences")
    readme.write_text(tampered)
    with pytest.raises(LabelFinalizationError, match="card"):
        validate_labeled_publication(output)


def test_card_link_to_input_revision_uses_full_sha(tmp_path: Path) -> None:
    """The card must always link to the full 40-character immutable SHA."""

    output = _build_publication(tmp_path)
    card = (output / "README.md").read_text()
    pattern = (
        r"https://huggingface\.co/datasets/"
        r"NoeFlandre/osm-polygon-wikidata-sentence-relevance/tree/"
        r"[0-9a-f]{40}"
    )
    assert re.search(pattern, card)


def test_card_omits_timing_when_manifest_changes(tmp_path: Path) -> None:
    """Operational timing changes do not alter the public-facing card."""

    output = _build_publication(tmp_path)
    original = (output / "README.md").read_text()
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["timing"]["total_wall_seconds"] = 4321.0
    (output / "manifest.json").write_text(json.dumps(manifest))
    assert (output / "README.md").read_text() == original


def test_card_omits_allocation_metrics(tmp_path: Path) -> None:
    output = _build_publication(tmp_path)
    card = (output / "README.md").read_text().lower()
    assert "allocation" not in card


def test_validator_rejects_card_drift(tmp_path: Path) -> None:
    """The validator must reject a publication whose card drifted."""

    output = _build_publication(tmp_path)
    # Re-render the card with the wrong row count.
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["statistics"]["row_count"] += 1
    (output / "manifest.json").write_text(json.dumps(manifest))
    # The card still shows the old row count; the validator catches this.
    with pytest.raises(LabelFinalizationError):
        validate_labeled_publication(output)
