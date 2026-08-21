"""Dynamic-field contracts for the V2 dataset card renderer."""

from __future__ import annotations

import pyarrow as pa

from osm_polygon_sentence_relevance.labeling.v2_finalization import _render_card


def _inputs() -> tuple[pa.Table, dict[str, object], dict[str, object], dict[str, object]]:
    table = pa.table({"place_relevance": ["yes", "yes", *(["no"] * 8)]})
    identity = {
        "input_dataset_revision": "input-revision",
        "model_revision": "model-revision",
        "source_commit": "source-commit",
        "sampling_seed": "sample-seed",
        "sampling_version": "sample-version",
        "h3_resolution": 5,
    }
    analytics = {
        "place_counts": {"yes": 2, "no": 8},
        "unique_polygons": 3,
        "unique_languages": 4,
    }
    h3 = {"cell_count": 6, "missing_coordinate_count": 7}
    return table, identity, analytics, h3


def test_render_card_includes_exact_dynamic_identity_statistics_and_rates() -> None:
    table, identity, analytics, h3 = _inputs()

    card = _render_card(
        dataset_repo_id="owner/dataset",
        table=table,
        parquet_bytes=1234,
        identity=identity,
        analytics=analytics,
        h3=h3,
        parquet_sha256="parquet-digest",
    )

    assert "num_examples: 10" in card
    assert "Input dataset revision: [`input-revision`]" in card
    assert "model-revision" in card
    assert "source-commit" in card
    assert "`parquet-digest`" in card
    assert "missing_coordinate_count: 7" in card
    assert "ranked with seed `sample-seed`" in card
    assert "sampling version is `sample-version`" in card
    assert "H3 resolution is `5`" in card
    assert "- Input sampling seed: `sample-seed`" in card
    assert "| `place_relevance=yes` | 2 (20.00%) |" in card
    assert "| `place_relevance=no` | 8 (80.00%) |" in card
    assert "H3 cells represented | 6" in card


def test_render_card_uses_safe_defaults_for_missing_digest_and_coordinates() -> None:
    table, identity, analytics, h3 = _inputs()
    h3 = {"cell_count": 1}

    card = _render_card(
        dataset_repo_id="owner/dataset",
        table=table,
        parquet_bytes=1,
        identity=identity,
        analytics=analytics,
        h3=h3,
    )

    assert "`recorded in manifest.json`" in card
    assert "missing_coordinate_count: 0" in card


def test_render_card_uses_empty_identity_and_zero_count_defaults() -> None:
    card = _render_card(
        dataset_repo_id="owner/dataset",
        table=pa.table({"place_relevance": []}),
        parquet_bytes=0,
        identity={},
        analytics={
            "place_counts": {},
            "unique_polygons": 0,
            "unique_languages": 0,
        },
        h3={"cell_count": 0},
    )

    assert "- model revision ``;" in card
    assert "- Input dataset revision: [``]" in card
    assert "- Input sampling seed: ``" in card
    assert "ranked with seed ``" in card
    assert "- Source commit used for the run: [``]" in card
    assert "sampling version is ``; H3 resolution is `3`." in card
    assert "| `place_relevance=yes` | 0 (0.00%) |" in card
    assert "| `place_relevance=no` | 0 (0.00%) |" in card
