"""Contract tests for the extracted resume orchestration boundary."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator import resume


def test_resume_orchestration_exposes_resume_run() -> None:
    assert callable(resume.resume_run)


def test_resume_orchestration_rejects_invalid_run_id_before_side_effects() -> None:
    with pytest.raises(RuntimeError, match="twenty lowercase"):
        resume.resume_run("invalid", SimpleNamespace(), object())  # type: ignore[arg-type]


def test_resume_orchestration_rejects_missing_state_without_services(
    tmp_path: Path,
) -> None:
    services = SimpleNamespace(data_root=tmp_path)

    with pytest.raises(RuntimeError, match="run state does not exist"):
        resume.resume_run("a" * 20, SimpleNamespace(), services)  # type: ignore[arg-type]


def test_resume_orchestration_rejects_non_object_identity(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "runs" / ("a" * 20) / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"run_identity": []}), encoding="utf-8")
    services = SimpleNamespace(data_root=tmp_path)

    with pytest.raises(RuntimeError, match="identity is not an object"):
        resume.resume_run("a" * 20, SimpleNamespace(), services)  # type: ignore[arg-type]


def test_resume_orchestration_rejects_identity_that_changes_run_id(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "runs" / ("a" * 20) / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"run_identity": {}}), encoding="utf-8")
    services = SimpleNamespace(
        data_root=tmp_path,
        config_type=SimpleNamespace(
            from_persisted=lambda _identity: SimpleNamespace(
                run_id="b" * 20,
                stage="label",
                source_commit="a" * 40,
            )
        ),
    )

    with pytest.raises(RuntimeError, match="does not reproduce"):
        resume.resume_run("a" * 20, SimpleNamespace(), services)  # type: ignore[arg-type]
