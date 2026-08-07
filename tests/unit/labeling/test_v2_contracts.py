from __future__ import annotations

from osm_polygon_sentence_relevance.labeling.v2_contracts import (
    V2_LOGIT_PROMPT_VERSION,
    V2_MODEL_FILE,
    V2_MODEL_FILE_SHA256,
    V2_MODEL_REPO_ID,
    V2_MODEL_REVISION,
    V2LogitRecord,
)


def test_v2_identity_is_pinned_to_standard_non_mtp_model() -> None:
    assert V2_MODEL_REPO_ID == "ggml-org/Qwen3.6-27B-GGUF"
    assert V2_MODEL_FILE == "Qwen3.6-27B-Q4_K_M.gguf"
    assert V2_MODEL_REVISION == "4c8d89a3b10d66695ded02bacee44f9dcf64848b"
    assert V2_MODEL_FILE_SHA256 == (
        "65b753ea835627f7b511143c6ceb976525c7f21f5df8c664bc0a9c23d1c49921"
    )
    assert V2_LOGIT_PROMPT_VERSION == "worldwide-place-description-logit-v2"


def test_v2_logit_record_derives_margin_and_relative_probability() -> None:
    record = V2LogitRecord(
        sentence_id="s1",
        place_relevance="yes",
        yes_logprob=-0.2,
        no_logprob=-1.2,
    )

    assert record.logit_margin == 1.0
    assert record.two_class_probability == 1 / (1 + 2.718281828459045**-1)


def test_v2_logit_record_rejects_non_binary_labels_and_non_finite_scores() -> None:
    import math

    import pytest

    with pytest.raises(ValueError, match="place_relevance"):
        V2LogitRecord(
            sentence_id="s1",
            place_relevance="uncertain",
            yes_logprob=-1.0,
            no_logprob=-1.0,
        )
    with pytest.raises(ValueError, match="finite"):
        V2LogitRecord(
            sentence_id="s1",
            place_relevance="no",
            yes_logprob=math.nan,
            no_logprob=-1.0,
        )


def test_v2_logit_record_rejects_empty_ids_and_boolean_scores() -> None:
    import pytest

    with pytest.raises(ValueError, match="sentence_id"):
        V2LogitRecord("", "yes", -1.0, -1.0)
    with pytest.raises(ValueError, match="yes_logprob"):
        V2LogitRecord("s1", "yes", True, -1.0)
    with pytest.raises(ValueError, match="no_logprob"):
        V2LogitRecord("s1", "yes", -1.0, False)


def test_v2_logit_record_uses_negative_margin_probability_branch() -> None:
    record = V2LogitRecord("s1", "no", -2.0, -1.0)
    assert record.logit_margin == -1.0
    assert record.two_class_probability < 0.5
