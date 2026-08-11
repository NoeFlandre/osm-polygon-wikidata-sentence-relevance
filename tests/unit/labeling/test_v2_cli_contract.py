from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_sentence_relevance.labeling.cli import _identity, _load_input
from osm_polygon_sentence_relevance.labeling.runtime import build_runtime_plan
from osm_polygon_sentence_relevance.labeling.v2_contracts import (
    V2_LOGIT_PROMPT_VERSION,
    V2_MODEL_FILE,
    V2_MODEL_REPO_ID,
)


def test_v2_identity_uses_standard_model_and_logit_prompt(tmp_path: Path) -> None:
    path = tmp_path / "input.parquet"
    path.write_bytes(b"input")
    args = Namespace(
        row_limit=0,
        request_concurrency=None,
        release_lane="v2-worldwide",
        sampling_target=200_000,
        sampling_seed="seed",
        h3_resolution=3,
        input_dataset_revision="b" * 40,
        model_revision="c" * 40,
        model_file_sha256="d" * 64,
        source_commit="e" * 40,
        engine="llama.cpp",
        engine_version="1",
        batch_size=128,
    )
    identity = _identity(
        args, path, build_runtime_plan(parallel=16, per_slot_context=4096)
    )
    assert identity.model_repo_id == V2_MODEL_REPO_ID
    assert identity.model_file == V2_MODEL_FILE
    assert identity.prompt_version == V2_LOGIT_PROMPT_VERSION
    assert identity.sampling_version == "v2-area-h3-logit"


def test_v2_input_requires_language_and_primary_tag_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.parquet"
    pq.write_table(
        pa.table(
            {
                "sentence_id": ["s1"],
                "sentence_text_raw": ["A valley."],
                "previous_sentence": [None],
                "next_sentence": [None],
                "page_title": ["Valley"],
                "section_path": [["Geography"]],
                "area_km2": [1.0],
                "area_bucket": ["medium"],
                "polygon_id": ["p1"],
                "lat": [45.0],
                "lon": [2.0],
            }
        ),
        path,
    )

    with pytest.raises(ValueError, match="language|osm_primary_tag"):
        _load_input(path, v2=True)
