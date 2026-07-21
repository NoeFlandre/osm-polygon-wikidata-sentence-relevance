"""Contracts for the maintained Grid'5000 production guide."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs" / "guides" / "grid5000.md"


def _text() -> str:
    return DOC.read_text(encoding="utf-8")


def test_guide_contains_only_supported_frontend_entrypoints() -> None:
    text = _text()
    assert "submit_streaming_build.sh" in text
    assert "submit_streaming_finalization.sh" in text
    for obsolete in (
        "submit_gpu_build.sh",
        "run_gpu_build.sh",
        "submit_gpu_smoke.sh",
        "run_gpu_smoke.sh",
    ):
        assert obsolete not in text


def test_guide_has_exactly_two_canonical_submission_commands() -> None:
    commands = [
        line.strip()
        for line in _text().splitlines()
        if line.strip().startswith('"${REPO_ROOT}/scripts/grid5000/submit_')
    ]
    assert commands == [
        '"${REPO_ROOT}/scripts/grid5000/submit_streaming_build.sh" \\',
        '"${REPO_ROOT}/scripts/grid5000/submit_streaming_finalization.sh" \\',
    ]


def test_guide_requires_remote_cuda_for_sentence_inference() -> None:
    text = _text().casefold()
    assert "allocated cuda job" in text
    assert "--device cuda" in text
    assert "never run inference on a frontend or local mac" in text


def test_guide_documents_bounded_storage_and_resume() -> None:
    text = _text()
    assert "one shard" in text
    assert "verified remote checkpoints" in text
    assert "MAX_SHARDS=0" in text
    assert "same run ID" in text


def test_guide_uses_placeholders_without_personal_paths() -> None:
    text = _text()
    for personal in ("nflandre", "/Users/", "/Volumes/", "/srv/storage/"):
        assert personal not in text


def test_guide_documents_validation_before_publication() -> None:
    text = _text()
    assert "validate_export_directory" in text
    assert "Publication is a separate remote mutation" in text
    assert "independently download it and validate the readback" in text
