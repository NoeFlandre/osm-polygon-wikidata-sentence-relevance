"""Direct tests for terminal inspection against real production artifacts.

Every test in this module is a RED→GREEN contract for
``recorded_job.inspect_remote_resume`` and ``recorded_job.classify_terminal``.
Fixtures are created through the authoritative ``CheckpointStore`` so the
inspector is exercised against the exact layout and metadata the production
labeling pipeline writes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as _Result
from typing import Any, Final

import pytest

from osm_polygon_sentence_relevance.labeling.checkpoint import CheckpointStore
from osm_polygon_sentence_relevance.labeling.contracts import (
    LabelRecord,
    LabelValue,
    RunIdentity,
)
from osm_polygon_sentence_relevance.operator import recorded_job
from osm_polygon_sentence_relevance.operator.oar import ExitClass, JobState, JobStatus
from osm_polygon_sentence_relevance.operator.ssh import LogChunk

#: Path layout used by CheckpointStore: progress.json + checkpoints/batch-NN.*
_LABEL_WORK: Final[str] = "/home/u/osm-polygon-operator/run/label-work"
_LABEL_OUTPUT: Final[str] = "/home/u/osm-polygon-operator/run/label-output"
_LOGS: Final[str] = "/home/u/osm-polygon-operator/run/logs"


def _identity() -> RunIdentity:
    return RunIdentity(
        input_sha256="a" * 64,
        input_dataset_revision="b" * 40,
        model_repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        model_revision="5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace",
        model_file="Qwen3.6-27B-Q4_K_M.gguf",
        model_file_sha256="c" * 64,
        prompt_version="v1",
        source_commit="d" * 40,
        engine="llama.cpp",
        engine_version="b1234",
        batch_size=128,
        row_limit=0,
        llama_parallel=8,
        llama_per_slot_context=8192,
        llama_total_context=65536,
        request_concurrency=8,
    )


def _records(start: int, count: int) -> list[LabelRecord]:
    return [
        LabelRecord(
            sentence_id=f"s{start + i:08d}",
            landuse_relevance=LabelValue.YES,
            polygon_relevance=LabelValue.YES,
            landuse_reason="x",
            polygon_reason="y",
            evidence="z",
        )
        for i in range(count)
    ]


@dataclass
class _FakeSsh:
    """Minimal SSH fake that maps remote paths to local filesystem reads.

    The inspector must use the exact real production layout: progress.json
    under ``label_work`` (not under ``checkpoints/``) and paired
    ``checkpoints/batch-NNNNNN.parquet``/``.json``. These are read off the
    real fixture directory written by ``CheckpointStore``.
    """

    fixture_root: Path
    read_files: dict[str, str]
    manifests: dict[str, bool]
    sha_overrides: dict[str, str] | None = None

    def _actual_parquet_sha(self, remote_path: str) -> str:
        # Map remote /home/u/.../label-work/checkpoints/batch-NN.parquet to
        # the real bytes on disk.
        local = Path(remote_path.replace(_LABEL_WORK, str(self.fixture_root)))
        return hashlib.sha256(local.read_bytes()).hexdigest()

    def run(self, command: str) -> LogChunk:  # type: ignore[override]
        stripped = command.strip()
        if stripped.startswith("if test -f") and "manifest.json" in stripped:
            for path, present in self.manifests.items():
                if path in command and present:
                    return LogChunk(text="yes", next_offset=0, eof=True)
            return LogChunk(text="no", next_offset=0, eof=True)
        if "sha256sum" in stripped:
            # parse the trailing path out of the command
            tokens = stripped.split()
            remote_path = tokens[-1]
            if self.sha_overrides and remote_path in self.sha_overrides:
                digest = self.sha_overrides[remote_path]
            else:
                digest = self._actual_parquet_sha(remote_path)
            return LogChunk(
                text=f"{digest}  {remote_path}\n",
                next_offset=0,
                eof=True,
            )
        if stripped.startswith("find") and "checkpoints" in stripped:
            names = sorted(
                f"f\t{p.name}"
                for p in self.fixture_root.glob("checkpoints/batch-*")
                if p.is_file()
            )
            return LogChunk(text="\n".join(names) + "\n", next_offset=0, eof=True)
        if stripped.startswith("ls -1"):
            return LogChunk(text="", next_offset=0, eof=True)
        return LogChunk(text="", next_offset=0, eof=True)

    def read_since(self, path: str, offset: int) -> LogChunk:
        # map both /home/u/... and the fixture root
        if path in self.read_files:
            return LogChunk(text=self.read_files[path], next_offset=0, eof=True)
        local = Path(path.replace(_LABEL_WORK, str(self.fixture_root)))
        if local.is_file():
            return LogChunk(text=local.read_text(), next_offset=0, eof=True)
        raise recorded_job.ResumeError(f"missing {path}")


def _build_fixture(
    tmp_path: Path,
    *,
    completed: int,
    total: int,
    batch_size: int = 4,
    include_timing: bool = False,
    write_manifest: bool = True,
) -> Mapping[str, str]:
    identity = _identity()
    store = CheckpointStore(tmp_path, identity)
    written = 0
    index = 0
    while written < completed:
        remaining = min(batch_size, completed - written)
        store.write_batch(index, _records(written, remaining))
        written += remaining
        index += 1
    store.write_progress(
        completed=completed,
        total=total,
        elapsed_seconds=10.0,
    )
    if include_timing:
        store.write_timing({"started_at": 1.0, "finished_at": 2.0})
    if write_manifest:
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "identity": identity.to_dict(),
                    "row_count": completed,
                }
            )
        )
    files = {}
    for path in tmp_path.rglob("*"):
        if path.is_file():
            rel = str(path.relative_to(tmp_path))
            if path.suffix in {".json", ".txt"}:
                files[rel] = path.read_text()
    return files


def _exit_file(content: str) -> str:
    return f"{_LOGS}/2895249/labeling.exit_code"


def test_real_manifest_with_complete_progress_is_complete(
    tmp_path: Path,
) -> None:
    files = _build_fixture(tmp_path, completed=8, total=8)
    ssh = _FakeSsh(
        fixture_root=tmp_path,
        read_files={
            f"{_LABEL_WORK}/progress.json": files["progress.json"],
            _exit_file("0"): "0\n",
        },
        manifests={f"{_LABEL_OUTPUT}/manifest.json": True},
    )
    inspection = recorded_job.inspect_remote_resume(
        ssh,
        label_work_root=_LABEL_WORK,
        label_output_root=_LABEL_OUTPUT,
        expected_identity=_identity().to_dict(),
        exit_file=_exit_file("0"),
    )
    assert inspection.manifest_present is True
    assert inspection.exit_code == 0
    assert inspection.progress.completed == 8
    assert inspection.progress.strictly_partial is False
    classification = recorded_job.classify_terminal(
        JobStatus(2895249, JobState.TERMINATED), inspection
    )
    assert classification is ExitClass.COMPLETE


def test_partial_progress_with_valid_batches_is_resumable(
    tmp_path: Path,
) -> None:
    files = _build_fixture(tmp_path, completed=4, total=16, write_manifest=False)
    ssh = _FakeSsh(
        fixture_root=tmp_path,
        read_files={
            f"{_LABEL_WORK}/progress.json": files["progress.json"],
            _exit_file("0"): "0\n",
        },
        manifests={f"{_LABEL_OUTPUT}/manifest.json": False},
    )
    inspection = recorded_job.inspect_remote_resume(
        ssh,
        label_work_root=_LABEL_WORK,
        label_output_root=_LABEL_OUTPUT,
        expected_identity=_identity().to_dict(),
        exit_file=_exit_file("0"),
    )
    assert inspection.exit_code == 0
    assert inspection.manifest_present is False
    assert inspection.progress.strictly_partial is True
    assert inspection.checkpoint_pairs >= 1
    assert inspection.checkpoint_parquet_shas_match is True
    assert inspection.identity_matches is True
    classification = recorded_job.classify_terminal(
        JobStatus(2895249, JobState.TERMINATED), inspection
    )
    assert classification is ExitClass.CONTINUE


def test_missing_manifest_no_valid_checkpoints_is_failed(tmp_path: Path) -> None:
    files = _build_fixture(
        tmp_path, completed=0, total=16, batch_size=0, write_manifest=False
    )
    # no batches at all -> no parquet files
    ssh = _FakeSsh(
        fixture_root=tmp_path,
        read_files={
            f"{_LABEL_WORK}/progress.json": files["progress.json"],
            _exit_file("0"): "0\n",
        },
        manifests={f"{_LABEL_OUTPUT}/manifest.json": False},
    )
    inspection = recorded_job.inspect_remote_resume(
        ssh,
        label_work_root=_LABEL_WORK,
        label_output_root=_LABEL_OUTPUT,
        expected_identity=_identity().to_dict(),
        exit_file=_exit_file("0"),
    )
    classification = recorded_job.classify_terminal(
        JobStatus(2895249, JobState.TERMINATED), inspection
    )
    assert classification is ExitClass.FAILED


def test_sha_mismatch_is_failed(tmp_path: Path) -> None:
    files = _build_fixture(tmp_path, completed=4, total=16, write_manifest=False)
    parquet = next(tmp_path.glob("checkpoints/batch-*.parquet"))
    remote_pq_path = f"{_LABEL_WORK}/checkpoints/{parquet.name}"
    # Force the remote sha256sum to return a wrong digest.
    ssh = _FakeSsh(
        fixture_root=tmp_path,
        read_files={
            f"{_LABEL_WORK}/progress.json": files["progress.json"],
            _exit_file("0"): "0\n",
        },
        manifests={f"{_LABEL_OUTPUT}/manifest.json": False},
        sha_overrides={remote_pq_path: "0" * 64},
    )
    inspection = recorded_job.inspect_remote_resume(
        ssh,
        label_work_root=_LABEL_WORK,
        label_output_root=_LABEL_OUTPUT,
        expected_identity=_identity().to_dict(),
        exit_file=_exit_file("0"),
    )
    classification = recorded_job.classify_terminal(
        JobStatus(2895249, JobState.TERMINATED), inspection
    )
    assert classification is ExitClass.FAILED


def test_nonzero_exit_is_failed(tmp_path: Path) -> None:
    files = _build_fixture(tmp_path, completed=4, total=16, write_manifest=False)
    ssh = _FakeSsh(
        fixture_root=tmp_path,
        read_files={
            f"{_LABEL_WORK}/progress.json": files["progress.json"],
            _exit_file("512"): "512\n",
        },
        manifests={f"{_LABEL_OUTPUT}/manifest.json": False},
    )
    inspection = recorded_job.inspect_remote_resume(
        ssh,
        label_work_root=_LABEL_WORK,
        label_output_root=_LABEL_OUTPUT,
        expected_identity=_identity().to_dict(),
        exit_file=_exit_file("512"),
    )
    classification = recorded_job.classify_terminal(
        JobStatus(2895249, JobState.TERMINATED), inspection
    )
    assert classification is ExitClass.FAILED


def test_progress_json_read_from_label_work_root_not_checkpoints_dir(
    tmp_path: Path,
) -> None:
    """Inspect must read ${label_work}/progress.json, not checkpoints/progress.json.

    This is the production layout CheckpointStore writes.
    """

    files = _build_fixture(tmp_path, completed=4, total=8, write_manifest=False)

    # The fake _FakeSsh.read_since will raise ResumeError if it sees the wrong
    # path; assert the inspector queries the real layout.
    ssh = _FakeSsh(
        fixture_root=tmp_path,
        read_files={
            f"{_LABEL_WORK}/progress.json": files["progress.json"],
            _exit_file("0"): "0\n",
        },
        manifests={f"{_LABEL_OUTPUT}/manifest.json": False},
    )
    recorded_job.inspect_remote_resume(
        ssh,
        label_work_root=_LABEL_WORK,
        label_output_root=_LABEL_OUTPUT,
        expected_identity=_identity().to_dict(),
        exit_file=_exit_file("0"),
    )


def test_missing_job_with_complete_evidence_is_complete(
    tmp_path: Path,
) -> None:
    """A job purged from OAR but with full durable evidence still completes."""

    files = _build_fixture(tmp_path, completed=8, total=8)
    ssh = _FakeSsh(
        fixture_root=tmp_path,
        read_files={
            f"{_LABEL_WORK}/progress.json": files["progress.json"],
            _exit_file("0"): "0\n",
        },
        manifests={f"{_LABEL_OUTPUT}/manifest.json": True},
    )
    inspection = recorded_job.inspect_remote_resume(
        ssh,
        label_work_root=_LABEL_WORK,
        label_output_root=_LABEL_OUTPUT,
        expected_identity=_identity().to_dict(),
        exit_file=_exit_file("0"),
    )
    classification = recorded_job.classify_terminal(
        JobStatus(2895249, JobState.MISSING), inspection
    )
    assert classification is ExitClass.COMPLETE


def test_missing_job_with_partial_evidence_is_resumable(
    tmp_path: Path,
) -> None:
    files = _build_fixture(tmp_path, completed=4, total=16, write_manifest=False)
    ssh = _FakeSsh(
        fixture_root=tmp_path,
        read_files={
            f"{_LABEL_WORK}/progress.json": files["progress.json"],
            _exit_file("0"): "0\n",
        },
        manifests={f"{_LABEL_OUTPUT}/manifest.json": False},
    )
    inspection = recorded_job.inspect_remote_resume(
        ssh,
        label_work_root=_LABEL_WORK,
        label_output_root=_LABEL_OUTPUT,
        expected_identity=_identity().to_dict(),
        exit_file=_exit_file("0"),
    )
    classification = recorded_job.classify_terminal(
        JobStatus(2895249, JobState.MISSING), inspection
    )
    assert classification is ExitClass.CONTINUE


def test_missing_job_with_no_evidence_fails_safely(tmp_path: Path) -> None:
    """MISSING with no progress/checkpoints must fail with ResumeError."""

    files = _build_fixture(
        tmp_path, completed=0, total=16, batch_size=0, write_manifest=False
    )
    ssh = _FakeSsh(
        fixture_root=tmp_path,
        read_files={
            f"{_LABEL_WORK}/progress.json": files["progress.json"],
            _exit_file("0"): "0\n",
        },
        manifests={f"{_LABEL_OUTPUT}/manifest.json": False},
    )
    inspection = recorded_job.inspect_remote_resume(
        ssh,
        label_work_root=_LABEL_WORK,
        label_output_root=_LABEL_OUTPUT,
        expected_identity=_identity().to_dict(),
        exit_file=_exit_file("0"),
    )
    with pytest.raises(recorded_job.ResumeError):
        recorded_job.classify_terminal(JobStatus(2895249, JobState.MISSING), inspection)


# ------------------------------------------------------------------
# Validation helpers and edge cases
# ------------------------------------------------------------------


def test_is_terminal_handles_all_live_states() -> None:
    """TERMINATED / ERROR / MISSING are terminal; everything else is live."""

    from osm_polygon_sentence_relevance.operator.oar import JobState
    from osm_polygon_sentence_relevance.operator.recorded_job import is_terminal

    assert is_terminal(JobState.TERMINATED) is True
    assert is_terminal(JobState.ERROR) is True
    assert is_terminal(JobState.MISSING) is True
    assert is_terminal(JobState.QUEUED) is False
    assert is_terminal(JobState.RUNNING) is False


def test_parse_batch_index_rejects_unrelated_names() -> None:
    """Names that are not ``batch-NNNNNN.{parquet,json}`` return None."""

    from osm_polygon_sentence_relevance.operator.recorded_job import _parse_batch_index

    assert _parse_batch_index("batch-000000.parquet") == 0
    assert _parse_batch_index("batch-123456.json") == 123456
    assert _parse_batch_index("plain-file.parquet") is None
    assert _parse_batch_index("batch-12.parquet") is None  # too short
    assert _parse_batch_index("batch-abcdef.parquet") is None  # not digits
    assert _parse_batch_index("batch-000000.txt") is None  # wrong ext
    assert _parse_batch_index("not-batch.parquet") is None


def test_read_progress_payload_rejects_invalid_json_and_non_mapping(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.ssh import LogChunk

    class _Bad:
        def read_since(self, _path: str, _o: int) -> Any:
            return LogChunk(text="not json", next_offset=0, eof=True)

    with pytest.raises(recorded_job.ResumeError, match="not valid JSON"):
        recorded_job._read_progress_payload(_Bad(), str(tmp_path))

    class _List:
        def read_since(self, _path: str, _o: int) -> Any:
            return LogChunk(text="[]", next_offset=0, eof=True)

    with pytest.raises(recorded_job.ResumeError, match="not a mapping"):
        recorded_job._read_progress_payload(_List(), str(tmp_path))

    class _Blank:
        def read_since(self, _path: str, _o: int) -> Any:
            return LogChunk(text="", next_offset=0, eof=True)

    assert recorded_job._read_progress_payload(_Blank(), str(tmp_path)) is None


def test_read_remote_bytes_digest_rejects_malformed_output() -> None:
    from osm_polygon_sentence_relevance.operator import recorded_job

    class _Empty:
        def run(self, _cmd: str) -> Any:
            return _Result(stdout="")

    with pytest.raises(recorded_job.ResumeError, match="no output"):
        recorded_job._read_remote_bytes_digest(_Empty(), "/home/u/p.parquet")

    class _Garbage:
        def run(self, _cmd: str) -> Any:
            return _Result(stdout="not-a-sha")

    with pytest.raises(recorded_job.ResumeError, match="malformed"):
        recorded_job._read_remote_bytes_digest(_Garbage(), "/home/u/p.parquet")


def test_enumerate_remote_checkpoint_files_rejects_unexpected_entries() -> None:
    from osm_polygon_sentence_relevance.operator import recorded_job

    class _Lister:
        def run(self, _cmd: str) -> Any:
            return _Result(stdout="f\tweird-entry.parquet\n")

    with pytest.raises(recorded_job.ResumeError, match="unexpected"):
        recorded_job._enumerate_remote_checkpoint_files(_Lister(), "/home/u/lw")


def test_inspect_remote_resume_rejects_inconsistent_counters(tmp_path: Path) -> None:
    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.ssh import LogChunk

    payload = json.dumps({"completed": 12, "total": 5, "identity": {}})

    class _Ssh:
        def run(self, _cmd: str) -> Any:
            t = _cmd.strip()
            if t.startswith("test -f") and "manifest" in t:
                return LogChunk(text="no", next_offset=0, eof=True)
            if "find" in t and "checkpoints" in t:
                return LogChunk(text="", next_offset=0, eof=True)
            return LogChunk(text="", next_offset=0, eof=True)

        def read_since(self, path: str, _o: int) -> Any:
            if path.endswith("progress.json"):
                return LogChunk(text=payload, next_offset=0, eof=True)
            return LogChunk(text="0", next_offset=0, eof=True)

    with pytest.raises(recorded_job.ResumeError, match="counters are inconsistent"):
        recorded_job.inspect_remote_resume(
            _Ssh(),
            label_work_root=str(tmp_path / "lw"),
            label_output_root=str(tmp_path / "out"),
            expected_identity={},
            exit_file=str(tmp_path / "x.exit"),
        )


def test_inspect_remote_resume_treats_absent_progress_as_failed_not_corrupt(
    tmp_path: Path,
) -> None:
    """A pre-checkpoint payload failure has no progress file and is not corruption."""

    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.oar import (
        ExitClass,
        JobState,
        JobStatus,
    )
    from osm_polygon_sentence_relevance.operator.ssh import LogChunk

    class _Ssh:
        def run(self, command: str) -> Any:
            if "manifest.json" in command:
                return LogChunk(text="no", next_offset=0, eof=True)
            if "find" in command and "checkpoints" in command:
                return LogChunk(text="", next_offset=0, eof=True)
            return LogChunk(text="", next_offset=0, eof=True)

        def read_since(self, _path: str, _offset: int) -> Any:
            return LogChunk(text="", next_offset=0, eof=True)

    inspection = recorded_job.inspect_remote_resume(
        _Ssh(),
        label_work_root=str(tmp_path / "label-work"),
        label_output_root=str(tmp_path / "label-output"),
        expected_identity={},
        exit_file=str(tmp_path / "labeling.exit_code"),
    )

    assert inspection.progress == recorded_job.ProgressFacts(
        completed=0,
        total=0,
        identity_matches=False,
    )
    assert (
        recorded_job.classify_terminal(
            JobStatus(2895249, JobState.TERMINATED, exit_code=256),
            inspection,
        )
        is ExitClass.FAILED
    )


def test_classify_terminal_insufficient_evidence_yields_failed(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.oar import (
        ExitClass,
        JobState,
        JobStatus,
    )

    inspection = recorded_job.ResumeInspection(
        exit_code=0,
        manifest_present=False,
        progress=recorded_job.ProgressFacts(
            completed=0, total=10, identity_matches=True
        ),
        checkpoint_pairs=0,
        checkpoint_parquet_shas_match=True,
        identity_matches=True,
    )
    status = JobStatus(42, JobState.TERMINATED)
    assert recorded_job.classify_terminal(status, inspection) is ExitClass.FAILED


def test_classify_terminal_failed_for_malformed_digest() -> None:
    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.oar import (
        ExitClass,
        JobState,
        JobStatus,
    )

    inspection = recorded_job.ResumeInspection(
        exit_code=None,
        manifest_present=False,
        progress=recorded_job.ProgressFacts(
            completed=2, total=10, identity_matches=True
        ),
        checkpoint_pairs=2,
        checkpoint_parquet_shas_match=True,
        identity_matches=True,
    )
    status = JobStatus(42, JobState.TERMINATED)
    assert recorded_job.classify_terminal(status, inspection) is ExitClass.FAILED


def test_classify_terminal_failed_for_identity_mismatch() -> None:
    """Identity mismatch produces FAILED."""

    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.oar import (
        ExitClass,
        JobState,
        JobStatus,
    )

    inspection = recorded_job.ResumeInspection(
        exit_code=0,
        manifest_present=True,
        progress=recorded_job.ProgressFacts(
            completed=10, total=10, identity_matches=False
        ),
        checkpoint_pairs=10,
        checkpoint_parquet_shas_match=True,
        identity_matches=False,
    )
    status = JobStatus(42, JobState.TERMINATED)
    assert recorded_job.classify_terminal(status, inspection) is ExitClass.FAILED


def test_classify_terminal_failed_for_nonzero_exit() -> None:
    """Non-zero exit_code (without it being None) produces FAILED."""

    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.oar import (
        ExitClass,
        JobState,
        JobStatus,
    )

    inspection = recorded_job.ResumeInspection(
        exit_code=137,
        manifest_present=False,
        progress=recorded_job.ProgressFacts(
            completed=4, total=10, identity_matches=True
        ),
        checkpoint_pairs=4,
        checkpoint_parquet_shas_match=True,
        identity_matches=True,
    )
    status = JobStatus(42, JobState.TERMINATED)
    assert recorded_job.classify_terminal(status, inspection) is ExitClass.FAILED


def test_inspect_remote_resume_rejects_invalid_exit_file(tmp_path: Path) -> None:
    """A non-integer exit file payload raises ResumeError."""

    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.ssh import LogChunk

    payload = json.dumps({"completed": 4, "total": 10, "identity": {}})

    class _Ssh:
        def run(self, _cmd: str) -> Any:
            t = _cmd.strip()
            if "manifest" in t:
                return LogChunk(text="no", next_offset=0, eof=True)
            if "checkpoints" in t and "find" in t:
                return LogChunk(text="", next_offset=0, eof=True)
            return LogChunk(text="", next_offset=0, eof=True)

        def read_since(self, path: str, _o: int) -> Any:
            if path.endswith("progress.json"):
                return LogChunk(text=payload, next_offset=0, eof=True)
            if path.endswith("exit"):
                return LogChunk(text="not-a-number", next_offset=0, eof=True)
            return LogChunk(text="0", next_offset=0, eof=True)

    with pytest.raises(recorded_job.ResumeError, match="exit file is not an integer"):
        recorded_job.inspect_remote_resume(
            _Ssh(),
            label_work_root=str(tmp_path / "lw"),
            label_output_root=str(tmp_path / "out"),
            expected_identity={},
            exit_file=str(tmp_path / "x.exit"),
        )


def test_progress_payload_invalid_counters_and_missing_total(
    tmp_path: Path,
) -> None:
    """Reading progress.json that is not a mapping raises ResumeError."""

    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.ssh import LogChunk

    class _Read:
        def __init__(self, body: str) -> None:
            self._body = body

        def read_since(self, _p: str, _o: int) -> Any:
            return LogChunk(text=self._body, next_offset=0, eof=True)

    with pytest.raises(recorded_job.ResumeError, match="not valid JSON"):
        recorded_job._read_progress_payload(
            _Read('{"completed": 1, "total":'), str(tmp_path)
        )


def test_inspect_remote_resume_progress_invalid_triggers_resume_error(
    tmp_path: Path,
) -> None:
    """An invalid completed field in ``progress.json`` raises."""

    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.ssh import LogChunk

    payload = json.dumps({"completed": "x", "total": 10, "identity": {}})

    class _Ssh:
        def run(self, _cmd: str) -> Any:
            t = _cmd.strip()
            if "manifest" in t:
                return LogChunk(text="no", next_offset=0, eof=True)
            return LogChunk(text="", next_offset=0, eof=True)

        def read_since(self, path: str, _o: int) -> Any:
            if path.endswith("progress.json"):
                return LogChunk(text=payload, next_offset=0, eof=True)
            return LogChunk(text="0", next_offset=0, eof=True)

    with pytest.raises(recorded_job.ResumeError, match="completed is invalid"):
        recorded_job.inspect_remote_resume(
            _Ssh(),
            label_work_root=str(tmp_path / "lw"),
            label_output_root=str(tmp_path / "out"),
            expected_identity={},
            exit_file=str(tmp_path / "x.exit"),
        )


def test_inspect_remote_resume_progress_missing_total_triggers_resume_error(
    tmp_path: Path,
) -> None:
    """A ``progress.json`` with neither total nor remaining raises."""

    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.ssh import LogChunk

    payload = json.dumps({"completed": 4, "identity": {}})

    class _Ssh:
        def run(self, _cmd: str) -> Any:
            t = _cmd.strip()
            if "manifest" in t:
                return LogChunk(text="no", next_offset=0, eof=True)
            return LogChunk(text="", next_offset=0, eof=True)

        def read_since(self, path: str, _o: int) -> Any:
            if path.endswith("progress.json"):
                return LogChunk(text=payload, next_offset=0, eof=True)
            return LogChunk(text="0", next_offset=0, eof=True)

    with pytest.raises(recorded_job.ResumeError, match="missing total/remaining"):
        recorded_job.inspect_remote_resume(
            _Ssh(),
            label_work_root=str(tmp_path / "lw"),
            label_output_root=str(tmp_path / "out"),
            expected_identity={},
            exit_file=str(tmp_path / "x.exit"),
        )


def test_inspect_remote_resume_rejects_invalid_metadata_json(
    tmp_path: Path,
) -> None:
    """A checkpoint metadata file that is not valid JSON raises."""

    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.ssh import LogChunk

    payload = json.dumps(
        {
            "completed": 2,
            "total": 10,
            "identity": {},
        }
    )
    expected_identity = {"batch_size": 128}

    class _Ssh:
        def run(self, cmd: str) -> Any:
            t = cmd.strip()
            if "manifest" in t:
                return LogChunk(text="no", next_offset=0, eof=True)
            if "find" in t and "checkpoints" in t:
                return LogChunk(
                    text="f\tbatch-000000.parquet\nf\tbatch-000000.json\n",
                    next_offset=0,
                    eof=True,
                )
            if "sha256sum" in t:
                return LogChunk(
                    text="0" * 64
                    + "  /home/u/label-work/checkpoints/batch-000000.parquet\n",
                    next_offset=0,
                    eof=True,
                )
            return LogChunk(text="", next_offset=0, eof=True)

        def read_since(self, path: str, _o: int) -> Any:
            if path.endswith("progress.json"):
                return LogChunk(text=payload, next_offset=0, eof=True)
            if path.endswith("batch-000000.json"):
                return LogChunk(text="not-json", next_offset=0, eof=True)
            return LogChunk(text="0", next_offset=0, eof=True)

    with pytest.raises(
        recorded_job.ResumeError, match="checkpoint metadata is not valid JSON"
    ):
        recorded_job.inspect_remote_resume(
            _Ssh(),
            label_work_root="/home/u/label-work",
            label_output_root="/home/u/label-output",
            expected_identity=expected_identity,
            exit_file="/home/u/logs/x.exit",
        )


def test_inspect_remote_resume_rejects_metadata_non_mapping(
    tmp_path: Path,
) -> None:
    """A checkpoint metadata file that is not a mapping raises."""

    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.ssh import LogChunk

    payload = json.dumps(
        {
            "completed": 2,
            "total": 10,
            "identity": {},
        }
    )

    class _Ssh:
        def run(self, cmd: str) -> Any:
            t = cmd.strip()
            if "manifest" in t:
                return LogChunk(text="no", next_offset=0, eof=True)
            if "find" in t and "checkpoints" in t:
                return LogChunk(
                    text="f\tbatch-000000.parquet\nf\tbatch-000000.json\n",
                    next_offset=0,
                    eof=True,
                )
            if "sha256sum" in t:
                return LogChunk(
                    text="0" * 64
                    + "  /home/u/label-work/checkpoints/batch-000000.parquet\n",
                    next_offset=0,
                    eof=True,
                )
            return LogChunk(text="", next_offset=0, eof=True)

        def read_since(self, path: str, _o: int) -> Any:
            if path.endswith("progress.json"):
                return LogChunk(text=payload, next_offset=0, eof=True)
            if path.endswith("batch-000000.json"):
                return LogChunk(text="[]", next_offset=0, eof=True)
            return LogChunk(text="0", next_offset=0, eof=True)

    with pytest.raises(
        recorded_job.ResumeError, match="checkpoint metadata is not a mapping"
    ):
        recorded_job.inspect_remote_resume(
            _Ssh(),
            label_work_root="/home/u/label-work",
            label_output_root="/home/u/label-output",
            expected_identity={},
            exit_file="/home/u/logs/x.exit",
        )


def test_inspect_remote_resume_records_missing_parquet_sha(tmp_path: Path) -> None:
    """A checkpoint metadata file without ``parquet_sha256`` stops SHA checks."""

    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.ssh import LogChunk

    payload = json.dumps(
        {
            "completed": 2,
            "total": 10,
            "identity": {"k": "v"},
        }
    )

    class _Ssh:
        def run(self, cmd: str) -> Any:
            t = cmd.strip()
            if "manifest" in t:
                return LogChunk(text="no", next_offset=0, eof=True)
            if "find" in t and "checkpoints" in t:
                return LogChunk(
                    text="f\tbatch-000000.parquet\nf\tbatch-000000.json\n",
                    next_offset=0,
                    eof=True,
                )
            if "sha256sum" in t:
                return LogChunk(
                    text="0" * 64
                    + "  /home/u/label-work/checkpoints/batch-000000.parquet\n",
                    next_offset=0,
                    eof=True,
                )
            return LogChunk(text="", next_offset=0, eof=True)

        def read_since(self, path: str, _o: int) -> Any:
            if path.endswith("progress.json"):
                return LogChunk(text=payload, next_offset=0, eof=True)
            if path.endswith("batch-000000.json"):
                return LogChunk(
                    text=json.dumps({"identity": {"k": "v"}, "no_sha": True}),
                    next_offset=0,
                    eof=True,
                )
            return LogChunk(text="0", next_offset=0, eof=True)

    inspection = recorded_job.inspect_remote_resume(
        _Ssh(),
        label_work_root="/home/u/label-work",
        label_output_root="/home/u/label-output",
        expected_identity={"k": "v"},
        exit_file="/home/u/logs/x.exit",
    )
    # Without parquet_sha256 in the metadata, ``checkpoint_parquet_shas_match``
    # must be False so the orchestrator cannot use this as evidence of
    # resumability.
    assert inspection.checkpoint_parquet_shas_match is False


def test_inspect_remote_resume_rejects_unsafe_label_work_path(tmp_path: Path) -> None:
    from osm_polygon_sentence_relevance.operator import recorded_job
    from osm_polygon_sentence_relevance.operator.relay import RelayError

    class _Ssh:
        def run(self, _cmd: str) -> Any:
            return _Result(stdout="")

        def read_since(self, _p: str, _o: int) -> Any:
            return _Result(stdout="0")

    with pytest.raises(RelayError):
        recorded_job.inspect_remote_resume(
            _Ssh(),
            label_work_root="/unsafe path/with spaces",
            label_output_root=str(tmp_path),
            expected_identity={},
            exit_file=str(tmp_path / "x.exit"),
        )
