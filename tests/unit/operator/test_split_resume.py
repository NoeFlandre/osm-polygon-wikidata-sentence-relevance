"""RED-to-GREEN contracts for split-stage recovery evidence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator.oar import ExitClass, JobState, JobStatus
from osm_polygon_sentence_relevance.operator.split_resume import (
    SplitResumeError,
    SplitResumeInspection,
    classify_split_terminal,
    inspect_split_resume,
    split_failure_reason,
)


def _status(message: str = "") -> JobStatus:
    return JobStatus(42, JobState.ERROR, exit_code=256, message=message)


def _inspection(completed: int, total: int = 375) -> SplitResumeInspection:
    return SplitResumeInspection(
        exit_code=None,
        checkpoint_count=completed,
        total_shards=total,
        identity_matches=True,
    )


def test_graceful_split_deadline_is_resumable() -> None:
    assert (
        classify_split_terminal(_status(), _inspection(23), exit_code=130)
        is ExitClass.CONTINUE
    )


def test_missing_split_exit_file_is_resumable_when_partial() -> None:
    assert (
        classify_split_terminal(_status(), _inspection(23), exit_code=None)
        is ExitClass.CONTINUE
    )


def test_complete_split_inventory_is_complete() -> None:
    assert (
        classify_split_terminal(_status(), _inspection(375), exit_code=130)
        is ExitClass.COMPLETE
    )


def test_nonzero_split_payload_failure_is_not_retried() -> None:
    result = classify_split_terminal(_status(), _inspection(23), exit_code=1)
    assert result is ExitClass.FAILED
    assert split_failure_reason(_inspection(23), exit_code=1) == "nonzero-exit"


def test_missing_split_checkpoints_fail_safely() -> None:
    inspection = _inspection(0)
    assert (
        classify_split_terminal(_status(), inspection, exit_code=130)
        is ExitClass.FAILED
    )
    assert split_failure_reason(inspection, exit_code=130) == "no-durable-work"


def test_split_identity_mismatch_fails_safely() -> None:
    inspection = SplitResumeInspection(
        exit_code=None,
        checkpoint_count=23,
        total_shards=375,
        identity_matches=False,
    )
    assert (
        classify_split_terminal(_status(), inspection, exit_code=130)
        is ExitClass.FAILED
    )
    assert split_failure_reason(inspection, exit_code=130) == "identity-mismatch"


def test_cancelled_split_is_not_resubmitted() -> None:
    assert (
        classify_split_terminal(
            _status("allocation cancelled by operator"),
            _inspection(0),
            exit_code=None,
        )
        is ExitClass.CANCELLED
    )


def test_incomplete_split_inventory_has_stable_failure_reason() -> None:
    assert split_failure_reason(_inspection(23), exit_code=0) == (
        "checkpoint-progress-invalid"
    )


def test_split_inspection_marks_partial_inventory() -> None:
    assert _inspection(23).strictly_partial
    assert not _inspection(0).strictly_partial
    assert not _inspection(375).strictly_partial


def test_inspect_split_resume_uses_hub_checkpoint_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSsh:
        def run(self, _command: str) -> SimpleNamespace:
            return SimpleNamespace(stdout="130\n")

    monkeypatch.setattr(
        "scripts.streaming.offload.discover_run",
        lambda **kwargs: [object()] * 23,
    )
    monkeypatch.setattr(
        "scripts.streaming.driver.list_remote_shard_keys",
        lambda **kwargs: [f"shard-{index}" for index in range(375)],
    )
    result = inspect_split_resume(
        ssh=FakeSsh(),
        repo_id="owner/output",
        input_repo_id="owner/input",
        input_revision="a" * 40,
        run_id="b" * 20,
        source_commit="c" * 40,
        pipeline_version="0.1.0",
        model_name="sat-12l-sm",
        batch_size=128,
        staging_revision="checkpoints/" + "b" * 20,
        exit_file="/run/logs/42/build.exit_code",
        cache_dir=tmp_path,
        hub_api=object(),
    )
    assert result.exit_code == 130
    assert result.checkpoint_count == 23
    assert result.total_shards == 375
    assert result.identity_matches


def test_inspect_split_resume_accepts_missing_exit_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSsh:
        def run(self, _command: str) -> SimpleNamespace:
            return SimpleNamespace(stdout="")

    monkeypatch.setattr(
        "scripts.streaming.offload.discover_run",
        lambda **kwargs: [object()],
    )
    monkeypatch.setattr(
        "scripts.streaming.driver.list_remote_shard_keys",
        lambda **kwargs: ["one"],
    )
    result = inspect_split_resume(
        ssh=FakeSsh(),
        repo_id="owner/output",
        input_repo_id="owner/input",
        input_revision="a" * 40,
        run_id="b" * 20,
        source_commit="c" * 40,
        pipeline_version="0.1.0",
        model_name="sat-12l-sm",
        batch_size=128,
        staging_revision="checkpoints/" + "b" * 20,
        exit_file="/run/logs/42/build.exit_code",
        cache_dir=tmp_path,
        hub_api=object(),
    )
    assert result.exit_code is None
    assert result.checkpoint_count == result.total_shards == 1


def test_inspect_split_resume_wraps_hub_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSsh:
        def run(self, _command: str) -> SimpleNamespace:
            return SimpleNamespace(stdout="130")

    monkeypatch.setattr(
        "scripts.streaming.offload.discover_run",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    with pytest.raises(SplitResumeError, match="validate split"):
        inspect_split_resume(
            ssh=FakeSsh(),
            repo_id="owner/output",
            input_repo_id="owner/input",
            input_revision="a" * 40,
            run_id="b" * 20,
            source_commit="c" * 40,
            pipeline_version="0.1.0",
            model_name="sat-12l-sm",
            batch_size=128,
            staging_revision="checkpoints/" + "b" * 20,
            exit_file="/run/logs/42/build.exit_code",
            cache_dir=tmp_path,
            hub_api=object(),
        )


def test_inspect_split_resume_preserves_split_resume_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSsh:
        def run(self, _command: str) -> SimpleNamespace:
            return SimpleNamespace(stdout="130")

    original = SplitResumeError("already classified")
    monkeypatch.setattr(
        "scripts.streaming.offload.discover_run",
        lambda **kwargs: (_ for _ in ()).throw(original),
    )
    with pytest.raises(SplitResumeError, match="already classified") as raised:
        inspect_split_resume(
            ssh=FakeSsh(),
            repo_id="owner/output",
            input_repo_id="owner/input",
            input_revision="a" * 40,
            run_id="b" * 20,
            source_commit="c" * 40,
            pipeline_version="0.1.0",
            model_name="sat-12l-sm",
            batch_size=128,
            staging_revision="checkpoints/" + "b" * 20,
            exit_file="/run/logs/42/build.exit_code",
            cache_dir=tmp_path,
            hub_api=object(),
        )
    assert raised.value is original


def test_inspect_split_resume_constructs_default_hub_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSsh:
        def run(self, _command: str) -> SimpleNamespace:
            return SimpleNamespace(stdout="130")

    client = object()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: client)
    seen: list[object] = []
    monkeypatch.setattr(
        "scripts.streaming.offload.discover_run",
        lambda **kwargs: seen.append(kwargs["hub_api"]) or [object()],
    )
    monkeypatch.setattr(
        "scripts.streaming.driver.list_remote_shard_keys",
        lambda **kwargs: ["one"],
    )
    result = inspect_split_resume(
        ssh=FakeSsh(),
        repo_id="owner/output",
        input_repo_id="owner/input",
        input_revision="a" * 40,
        run_id="b" * 20,
        source_commit="c" * 40,
        pipeline_version="0.1.0",
        model_name="sat-12l-sm",
        batch_size=128,
        staging_revision="checkpoints/" + "b" * 20,
        exit_file="/run/logs/42/build.exit_code",
        cache_dir=tmp_path,
    )
    assert seen == [client]
    assert result.checkpoint_count == 1


def test_inspect_split_resume_rejects_malformed_exit_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSsh:
        def run(self, _command: str) -> SimpleNamespace:
            return SimpleNamespace(stdout="not-an-int")

    with pytest.raises(SplitResumeError, match="exit file"):
        inspect_split_resume(
            ssh=FakeSsh(),
            repo_id="owner/output",
            input_repo_id="owner/input",
            input_revision="a" * 40,
            run_id="b" * 20,
            source_commit="c" * 40,
            pipeline_version="0.1.0",
            model_name="sat-12l-sm",
            batch_size=128,
            staging_revision="checkpoints/" + "b" * 20,
            exit_file="/run/logs/42/build.exit_code",
            cache_dir=tmp_path,
            hub_api=object(),
        )
