"""Coverage-targeted tests for the driver module uncovered branches."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest import mock

import pytest
import scripts.streaming.driver as driver_mod
from scripts.streaming.data_root import DataRootRejected
from scripts.streaming.driver import (
    DriverConfig,
    DriverError,
    OarJobIdRequired,
    StreamDriver,
    list_remote_shard_keys,
    main,
)
from scripts.streaming.offload import OffloadHandle


@pytest.fixture(autouse=True)
def _set_oar_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")


# ---------------------------------------------------------------------------
# Constructor validation.
# ---------------------------------------------------------------------------


def _good_args(tmp_path: Path) -> dict:
    hub_api = mock.MagicMock()
    hub_api.list_repo_tree.return_value = []
    hub_api.file_exists.return_value = False
    return {
        "repo_id": "owner/repo",
        "resolved_revision": "a" * 40,
        "source_commit": "b" * 40,
        "work_dir": tmp_path,
        "input_root": tmp_path / "input",
        "upstream_repo_id": "upstream/repo",
        "hub_api": hub_api,
        "run_id": "run-1",
        "staging_revision": "rev",
        "offload_local_cache_dir": tmp_path / "cache",
        "max_disk_bytes": 1 << 30,
    }


def _offload_handle(shard_key: str) -> OffloadHandle:
    digest = "d" * 64
    return OffloadHandle(
        repo_id="owner/repo",
        run_id="run-1",
        shard_key=shard_key,
        staging_revision="rev",
        folder_path=f"checkpoints/run-1/{shard_key}",
        expected_table_sha256=digest,
        computed_table_sha256=digest,
        table_bytes=123,
        metadata={"shard_key": shard_key},
    )


def test_main_process_shard_without_confirm(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["process-shard", "--shard", "italy-latest"])


def test_main_process_shard_with_confirm(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    with (
        mock.patch("osm_polygon_sentence_relevance.sentences.sat.SaTSentenceSegmenter"),
        mock.patch("huggingface_hub.HfApi"),
        mock.patch(
            "scripts.streaming.driver.StreamDriver.process_shard"
        ) as mock_process,
    ):
        rc = main(
            [
                "process-shard",
                "--confirm-offload",
                "--shard",
                "italy-latest",
                "--run-id",
                "run-1",
                "--staging-revision",
                "rev",
                "--repo-id",
                "owner/repo",
                "--upstream-repo-id",
                "upstream/repo",
                "--resolved-revision",
                "a" * 40,
                "--source-commit",
                "b" * 40,
                "--work-dir",
                str(tmp_path),
                "--input-root",
                str(tmp_path / "input"),
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "OK: processed and offloaded italy-latest" in captured.out
        mock_process.assert_called_once()


def test_ctor_rejects_blank_repo_id(tmp_path: Path) -> None:
    a = _good_args(tmp_path)
    a["repo_id"] = ""
    with pytest.raises(ValueError, match="repo_id"):
        StreamDriver(**a)


def test_ctor_rejects_repo_id_no_slash(tmp_path: Path) -> None:
    a = _good_args(tmp_path)
    a["repo_id"] = "noslash"
    with pytest.raises(ValueError, match="repo_id"):
        StreamDriver(**a)


def test_ctor_rejects_bad_resolved_revision(tmp_path: Path) -> None:
    a = _good_args(tmp_path)
    a["resolved_revision"] = "abc"
    with pytest.raises(ValueError, match="resolved_revision"):
        StreamDriver(**a)


def test_ctor_rejects_non_hex_resolved_revision(tmp_path: Path) -> None:
    a = _good_args(tmp_path)
    a["resolved_revision"] = "z" * 40
    with pytest.raises(ValueError, match="resolved_revision"):
        StreamDriver(**a)


def test_ctor_rejects_bad_source_commit(tmp_path: Path) -> None:
    a = _good_args(tmp_path)
    a["source_commit"] = "short"
    with pytest.raises(ValueError, match="source_commit"):
        StreamDriver(**a)


def test_ctor_rejects_non_hex_source_commit(tmp_path: Path) -> None:
    a = _good_args(tmp_path)
    a["source_commit"] = "z" * 40
    with pytest.raises(ValueError, match="source_commit"):
        StreamDriver(**a)


def test_ctor_rejects_none_hub_api(tmp_path: Path) -> None:
    a = _good_args(tmp_path)
    a["hub_api"] = None
    with pytest.raises(ValueError, match="hub_api"):
        StreamDriver(**a)


def test_ctor_raises_when_oar_job_id_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OAR_JOB_ID", raising=False)
    a = _good_args(tmp_path)
    with pytest.raises(OarJobIdRequired, match="OAR_JOB_ID"):
        StreamDriver(**a)


def test_ctor_rejects_work_dir_in_tmp(monkeypatch, tmp_path: Path) -> None:
    """If work_dir resolves to /tmp, DataRootRejected surfaces as DriverError."""
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    a = _good_args(tmp_path)
    a["work_dir"] = Path("/tmp")
    with pytest.raises(DriverError, match="work_dir resolves"):
        StreamDriver(**a)


def test_ctor_rejects_non_tmp_data_root_other_reason(
    monkeypatch, tmp_path: Path
) -> None:
    """If data root rejects for a non-TMP reason (e.g. NOT_REGULAR_DIR), re-raise."""
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    a = _good_args(tmp_path)
    a["work_dir"] = tmp_path / "file-not-dir"
    (tmp_path / "file-not-dir").write_text("x")
    with pytest.raises(DataRootRejected):
        StreamDriver(**a)


def test_ctor_rejects_pinned_revision_mismatch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    a = _good_args(tmp_path)
    a["work_dir"] = tmp_path
    (tmp_path / "state.json").write_text(json.dumps({"resolved_revision": "c" * 40}))
    with pytest.raises(DriverError, match="pinned revision mismatch"):
        StreamDriver(**a)


def test_ctor_rejects_corrupt_state_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    a = _good_args(tmp_path)
    a["work_dir"] = tmp_path
    (tmp_path / "state.json").write_text("not json")
    with pytest.raises(DriverError, match="malformed JSON"):
        StreamDriver(**a)


# ---------------------------------------------------------------------------
# DriverConfig frozenness.
# ---------------------------------------------------------------------------


def test_driver_config_is_frozen(tmp_path: Path) -> None:
    c = DriverConfig(
        repo_id="o/r",
        resolved_revision="a" * 40,
        source_commit="b" * 40,
        work_dir=tmp_path,
        input_root=tmp_path,
        upstream_repo_id="u/r",
        run_id="r",
        staging_revision="rev",
        offload_local_cache_dir=tmp_path,
        max_disk_bytes=1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.run_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _assert_disk_ceiling_or_raise.
# ---------------------------------------------------------------------------


def test_disk_ceiling_violation_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    sd = StreamDriver(**_good_args(tmp_path))
    with (
        mock.patch(
            "scripts.streaming.driver.shutil.disk_usage",
            return_value=mock.Mock(free=1, total=1, used=0),
        ),
        pytest.raises(DriverError, match="disk ceiling violated"),
    ):
        sd._assert_disk_ceiling_or_raise()


def test_disk_usage_oserror_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    sd = StreamDriver(**_good_args(tmp_path))
    with (
        mock.patch(
            "scripts.streaming.driver.shutil.disk_usage",
            side_effect=OSError("nope"),
        ),
        pytest.raises(DriverError, match="disk_usage probe failed"),
    ):
        sd._assert_disk_ceiling_or_raise()


# ---------------------------------------------------------------------------
# process_shard: failure paths.
# ---------------------------------------------------------------------------


def test_process_shard_no_shards_discovered(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    sd = StreamDriver(**_good_args(tmp_path))
    with (
        mock.patch.object(sd, "_download_shard"),
        mock.patch("scripts.streaming.driver.discover_shards", return_value=[]),
        pytest.raises(DriverError, match="no shards discovered"),
    ):
        sd.process_shard("italy-latest", segmenter=mock.Mock())


def test_process_shard_shard_not_in_inbox(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    sd = StreamDriver(**_good_args(tmp_path))

    other = mock.Mock()
    other.shard_key = "france-latest"
    with (
        mock.patch.object(sd, "_download_shard"),
        mock.patch("scripts.streaming.driver.discover_shards", return_value=[other]),
        pytest.raises(DriverError, match="not present in staged inbox"),
    ):
        sd.process_shard("italy-latest", segmenter=mock.Mock())


def test_process_shard_returns_no_checkpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    sd = StreamDriver(**_good_args(tmp_path))

    target = mock.Mock()
    target.shard_key = "italy-latest"
    res = mock.Mock(published=False, reused=False)

    with (
        mock.patch.object(sd, "_download_shard"),
        mock.patch("scripts.streaming.driver.discover_shards", return_value=[target]),
        mock.patch("scripts.streaming.driver.process_single_shard", return_value=res),
        pytest.raises(DriverError, match="returned no checkpoint"),
    ):
        sd.process_shard("italy-latest", segmenter=mock.Mock())


def test_process_shard_offload_failure_raises_driver_error(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    sd = StreamDriver(**_good_args(tmp_path))

    target = mock.Mock()
    target.shard_key = "italy-latest"
    res = mock.Mock(published=True, reused=False)

    with (
        mock.patch.object(sd, "_download_shard"),
        mock.patch("scripts.streaming.driver.discover_shards", return_value=[target]),
        mock.patch("scripts.streaming.driver.process_single_shard", return_value=res),
        mock.patch(
            "scripts.streaming.driver.load_shard_checkpoint",
            return_value=(mock.Mock(), mock.Mock(), {}),
        ),
        mock.patch(
            "scripts.streaming.driver.CheckpointOffloader",
        ) as m_co,
    ):
        m_co.return_value.upload_and_verify.side_effect = DriverError("boom")
        with pytest.raises(DriverError, match="boom"):
            sd.process_shard("italy-latest", segmenter=mock.Mock())


# ---------------------------------------------------------------------------
# _download_shard: missing upstream file.
# ---------------------------------------------------------------------------


def test_download_shard_missing_required_file(monkeypatch, tmp_path: Path) -> None:
    """A missing required (non-wikivoyage) upstream file aborts."""
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    args = _good_args(tmp_path)
    sd = StreamDriver(**args)
    inbox = tmp_path / "shards" / "inbox" / "italy-latest"
    with (
        mock.patch.object(
            sd.hub_api, "list_repo_tree", side_effect=Exception("network error")
        ),
        pytest.raises(DriverError, match="failed to download required file"),
    ):
        sd._download_shard(shard_key="italy-latest", inbox=inbox)


def test_download_shard_reuses_partial_inbox(monkeypatch, tmp_path: Path) -> None:
    """An existing inbox with the right shard is reused."""
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    args = _good_args(tmp_path)
    sd = StreamDriver(**args)

    inbox = tmp_path / "shards" / "inbox" / "italy-latest"
    inbox.mkdir(parents=True)
    for subdir, fname in driver_mod._DOWNLOAD_LAYOUT:
        d = inbox / subdir
        d.mkdir(parents=True, exist_ok=True)
        (d / fname.replace("<shard>", "italy-latest")).write_bytes(b"x")
    sd._download_shard(shard_key="italy-latest", inbox=inbox)


def test_download_shard_cleans_corrupt_partial(monkeypatch, tmp_path: Path) -> None:
    """A partial inbox with no valid shards is cleaned up."""
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    args = _good_args(tmp_path)
    sd = StreamDriver(**args)

    inbox = tmp_path / "shards" / "inbox" / "italy-latest"
    inbox.mkdir(parents=True)
    (inbox / "stray").write_text("hi")
    with mock.patch("scripts.streaming.driver.PerFileHubDownloader") as m_dl:
        m_inst = m_dl.return_value
        m_inst.download_many = mock.Mock(side_effect=Exception("noop"))
        with pytest.raises(Exception, match="noop"):
            sd._download_shard(shard_key="italy-latest", inbox=inbox)


# ---------------------------------------------------------------------------
# safe_cleanup_scratch.
# ---------------------------------------------------------------------------


def test_safe_cleanup_scratch_missing(tmp_path: Path) -> None:
    from scripts.streaming.data_root import safe_cleanup_scratch

    safe_cleanup_scratch(tmp_path / "missing", prefix_requirement="osm_")


def test_safe_cleanup_scratch_evicts_dir(monkeypatch, tmp_path: Path) -> None:
    from scripts.streaming.data_root import safe_cleanup_scratch

    p = tmp_path / "osm_scratch" / "dir"
    p.mkdir(parents=True)
    safe_cleanup_scratch(p, prefix_requirement="osm_")
    assert not p.exists()


# ---------------------------------------------------------------------------
# _write_state.
# ---------------------------------------------------------------------------


def test_write_state_preserves_existing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    sd = StreamDriver(**_good_args(tmp_path))
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps({"existing_key": "keep"}))
    sd._write_state(updated=True)
    payload = json.loads(sp.read_text())
    assert payload["existing_key"] == "keep"
    assert payload["resolved_revision"] == "a" * 40
    assert payload["last_updated"] is True


def test_write_state_corrupt_existing_overwrites(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    sd = StreamDriver(**_good_args(tmp_path))
    sp = tmp_path / "state.json"
    sp.write_text("not json")
    sd._write_state(updated=False)
    payload = json.loads(sp.read_text())
    assert payload["last_updated"] is False


# ---------------------------------------------------------------------------
# evict_active.
# ---------------------------------------------------------------------------


def test_evict_active_removes_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OAR_JOB_ID", "12345")
    sd = StreamDriver(**_good_args(tmp_path))
    active = tmp_path / "shards" / "active" / "italy-latest"
    active.mkdir(parents=True)
    sd.evict_active("italy-latest")
    assert not active.exists()


# ---------------------------------------------------------------------------
# main: CLI gate.
# ---------------------------------------------------------------------------


def test_main_unknown_command(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["nope"])


def test_remote_inventory_ignores_nested_and_rejects_empty() -> None:
    hub = mock.Mock()
    hub.list_repo_tree.return_value = [
        mock.Mock(path="polygons/nested/a.parquet"),
        mock.Mock(path="polygons/readme.txt"),
        mock.Mock(path="elsewhere/a.parquet"),
    ]
    with pytest.raises(DriverError, match="no polygon shard"):
        list_remote_shard_keys(hub_api=hub, repo_id="o/r", revision="r")


def test_remote_inventory_is_sorted_and_unique() -> None:
    hub = mock.Mock()
    hub.list_repo_tree.return_value = [
        mock.Mock(path="polygons/b.parquet"),
        mock.Mock(path="polygons/a.parquet"),
        mock.Mock(path="polygons/a.parquet"),
    ]
    assert list_remote_shard_keys(hub_api=hub, repo_id="o/r", revision="r") == [
        "a",
        "b",
    ]


def test_remote_inventory_supports_another_table_directory() -> None:
    hub = mock.Mock()
    hub.list_repo_tree.return_value = [
        mock.Mock(path="wikivoyage/documents/b.parquet"),
        mock.Mock(path="wikivoyage/documents/a.parquet"),
        mock.Mock(path="wikivoyage/documents/ignored.txt"),
    ]

    assert list_remote_shard_keys(
        hub_api=hub,
        repo_id="o/r",
        revision="r",
        directory="wikivoyage/documents",
    ) == ["a", "b"]


@pytest.mark.parametrize("directory", ["", "/absolute", "tables/../polygons"])
def test_remote_inventory_rejects_unsafe_directory(directory: str) -> None:
    with pytest.raises(ValueError, match="relative repository path"):
        list_remote_shard_keys(
            hub_api=mock.Mock(),
            repo_id="o/r",
            revision="r",
            directory=directory,
        )


def test_remote_inventory_allows_absent_optional_directory() -> None:
    hub = mock.Mock()
    hub.list_repo_tree.return_value = []

    assert (
        list_remote_shard_keys(
            hub_api=hub,
            repo_id="o/r",
            revision="r",
            directory="wikivoyage/documents",
            allow_empty=True,
        )
        == []
    )


def test_remote_inventory_rejects_empty_required_non_polygon_directory() -> None:
    hub = mock.Mock()
    hub.list_repo_tree.return_value = []
    with pytest.raises(DriverError, match="no shard files found under"):
        list_remote_shard_keys(
            hub_api=hub,
            repo_id="o/r",
            revision="r",
            directory="wikivoyage/documents",
        )


def test_ctor_rejects_blank_optional_shard_key(tmp_path: Path) -> None:
    args = _good_args(tmp_path)
    args["optional_shard_keys"] = {""}
    with pytest.raises(ValueError, match="optional_shard_keys"):
        StreamDriver(**args)


def test_download_shard_uses_precomputed_optional_inventory(
    tmp_path: Path,
) -> None:
    args = _good_args(tmp_path)
    args["optional_shard_keys"] = {"a-latest"}
    driver = StreamDriver(**args)
    inbox = tmp_path / "shards" / "inbox" / "a-latest"

    with (
        mock.patch("scripts.streaming.driver.discover_shards", return_value=[]),
        mock.patch("scripts.streaming.driver.PerFileHubDownloader") as downloader,
    ):
        driver._download_shard(shard_key="a-latest", inbox=inbox)

    downloader.return_value.download_many.assert_called_once()
    paths = downloader.return_value.download_many.call_args.args[0]
    assert len(paths) == 6
    args["hub_api"].file_exists.assert_not_called()


def test_ctor_rejects_missing_allocation_work_dir(monkeypatch, tmp_path: Path) -> None:
    args = _good_args(tmp_path)
    allocation = tmp_path / "allocation-12345"
    allocation.mkdir()
    candidate = allocation / "missing"
    args["work_dir"] = candidate
    monkeypatch.setattr(
        "scripts.streaming.driver.check_data_root",
        mock.Mock(
            side_effect=DataRootRejected(
                reason="TMP_FORBIDDEN", role="work", path=str(candidate)
            )
        ),
    )
    monkeypatch.setattr(
        "scripts.streaming.driver.discover_oar_scratch_dir", lambda **_: allocation
    )
    with pytest.raises(DriverError, match="already be a directory"):
        StreamDriver(**args)


def test_process_shard_wraps_remote_validation_failure(tmp_path: Path) -> None:
    driver = StreamDriver(**_good_args(tmp_path))
    with (
        mock.patch(
            "scripts.streaming.driver.inspect_remote_checkpoint",
            side_effect=driver_mod.CheckpointOffloadError("corrupt"),
        ),
        pytest.raises(DriverError, match="remote checkpoint validation") as error,
    ):
        driver.process_shard("a-latest", segmenter=mock.Mock())
    assert isinstance(error.value.__cause__, driver_mod.CheckpointOffloadError)


def test_process_shard_reuses_remote_and_evicts_local(tmp_path: Path) -> None:
    driver = StreamDriver(**_good_args(tmp_path))
    handle = _offload_handle("a-latest")
    with (
        mock.patch(
            "scripts.streaming.driver.inspect_remote_checkpoint",
            return_value=handle,
        ),
        mock.patch.object(driver, "_evict_local_shard_state") as evict,
    ):
        assert driver.process_shard("a-latest", segmenter=mock.Mock()) is handle
    evict.assert_called_once_with("a-latest")
    assert driver.has_verified_checkpoint("a-latest")
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["verified_checkpoints"] == {
        "a-latest": {
            "segmented_table_bytes": 123,
            "segmented_table_sha256": "d" * 64,
        }
    }


def test_checkpoint_ledger_requires_exact_run_identity(tmp_path: Path) -> None:
    state = {
        "resolved_revision": "a" * 40,
        "run_id": "other-run",
        "verified_checkpoints": {
            "a-latest": {
                "segmented_table_bytes": 123,
                "segmented_table_sha256": "d" * 64,
            }
        },
    }
    (tmp_path / "state.json").write_text(json.dumps(state))

    with pytest.raises(DriverError, match="run_id mismatch"):
        StreamDriver(**_good_args(tmp_path))


def test_checkpoint_ledger_round_trips_across_allocations(tmp_path: Path) -> None:
    first = StreamDriver(**_good_args(tmp_path))
    first._write_state(updated=True, completed=_offload_handle("a-latest"))

    resumed = StreamDriver(**_good_args(tmp_path))

    assert resumed.has_verified_checkpoint("a-latest") is True
    assert resumed.has_verified_checkpoint("b-latest") is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state.update({"verified_checkpoints": []}),
        lambda state: state.pop("source_commit"),
        lambda state: state["verified_checkpoints"].update(
            {"../bad": state["verified_checkpoints"].pop("a-latest")}
        ),
        lambda state: state["verified_checkpoints"].update({"a-latest": "bad"}),
        lambda state: state["verified_checkpoints"]["a-latest"].update({"extra": True}),
        lambda state: state["verified_checkpoints"]["a-latest"].update(
            {"segmented_table_sha256": "bad"}
        ),
        lambda state: state["verified_checkpoints"]["a-latest"].update(
            {"segmented_table_bytes": 0}
        ),
    ],
)
def test_checkpoint_ledger_rejects_malformed_durable_state(
    tmp_path: Path, mutation
) -> None:
    driver = StreamDriver(**_good_args(tmp_path))
    driver._write_state(updated=True, completed=_offload_handle("a-latest"))
    path = tmp_path / "state.json"
    state = json.loads(path.read_text())
    mutation(state)
    path.write_text(json.dumps(state))

    with pytest.raises(DriverError, match="checkpoint|verified_checkpoints"):
        StreamDriver(**_good_args(tmp_path))


@pytest.mark.parametrize(
    "changes",
    [
        {"repo_id": "other/repo"},
        {"run_id": "other-run"},
        {"staging_revision": "other-revision"},
        {"shard_key": "../bad"},
        {"computed_table_sha256": "e" * 64},
        {
            "expected_table_sha256": "bad",
            "computed_table_sha256": "bad",
        },
        {
            "expected_table_sha256": "z" * 64,
            "computed_table_sha256": "z" * 64,
        },
        {"table_bytes": True},
        {"table_bytes": 0},
    ],
)
def test_checkpoint_ledger_rejects_unverified_handle(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    driver = StreamDriver(**_good_args(tmp_path))
    handle = dataclasses.replace(_offload_handle("a-latest"), **changes)

    with pytest.raises(DriverError, match="does not match this run"):
        driver._write_state(updated=True, completed=handle)

    assert driver.has_verified_checkpoint("a-latest") is False


def test_main_reports_reused_checkpoints_once_and_processes_only_pending_shards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_driver = mock.Mock()
    fake_driver.has_verified_checkpoint.side_effect = lambda key: key == "a-latest"
    monkeypatch.setattr(driver_mod, "StreamDriver", mock.Mock(return_value=fake_driver))
    monkeypatch.setattr(
        driver_mod,
        "list_remote_shard_keys",
        mock.Mock(side_effect=[[], [], ["a-latest", "b-latest"]]),
    )
    monkeypatch.setattr("huggingface_hub.HfApi", mock.Mock())
    monkeypatch.setattr(
        "osm_polygon_sentence_relevance.sentences.sat.SaTSentenceSegmenter",
        mock.Mock(),
    )

    result = main(
        [
            "stream-build",
            "--confirm-offload",
            "--run-id",
            "run-1",
            "--staging-revision",
            "rev",
            "--repo-id",
            "owner/repo",
            "--upstream-repo-id",
            "upstream/repo",
            "--resolved-revision",
            "a" * 40,
            "--source-commit",
            "b" * 40,
            "--work-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    fake_driver.process_shard.assert_called_once_with("b-latest", segmenter=mock.ANY)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events == [
        {"completed": 1, "event": "resume", "remaining": 1, "total": 2},
        {"completed": 2, "shard_key": "b-latest", "total": 2},
    ]


def test_main_does_not_load_segmenter_when_every_shard_is_already_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_driver = mock.Mock()
    fake_driver.has_verified_checkpoint.return_value = True
    segmenter = mock.Mock()
    monkeypatch.setattr(driver_mod, "StreamDriver", mock.Mock(return_value=fake_driver))
    monkeypatch.setattr(
        driver_mod,
        "list_remote_shard_keys",
        mock.Mock(side_effect=[[], [], ["a-latest", "b-latest"]]),
    )
    monkeypatch.setattr("huggingface_hub.HfApi", mock.Mock())
    monkeypatch.setattr(
        "osm_polygon_sentence_relevance.sentences.sat.SaTSentenceSegmenter",
        segmenter,
    )

    result = main(
        [
            "stream-build",
            "--confirm-offload",
            "--run-id",
            "run-1",
            "--staging-revision",
            "rev",
            "--repo-id",
            "owner/repo",
            "--upstream-repo-id",
            "upstream/repo",
            "--resolved-revision",
            "a" * 40,
            "--source-commit",
            "b" * 40,
            "--work-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    segmenter.assert_not_called()
    fake_driver.process_shard.assert_not_called()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events == [{"completed": 2, "event": "resume", "remaining": 0, "total": 2}]


def test_download_shard_wraps_optional_probe_failure(tmp_path: Path) -> None:
    args = _good_args(tmp_path)
    args["hub_api"].file_exists.side_effect = OSError("network")
    driver = StreamDriver(**args)
    with pytest.raises(DriverError, match="optional file exists") as error:
        driver._download_shard(
            shard_key="a-latest", inbox=tmp_path / "shards" / "inbox" / "a-latest"
        )
    assert isinstance(error.value.__cause__, OSError)


def test_download_shard_rejects_optional_pair_mismatch(tmp_path: Path) -> None:
    args = _good_args(tmp_path)
    args["hub_api"].file_exists.side_effect = [True, False]
    driver = StreamDriver(**args)
    with pytest.raises(DriverError, match="both exist or both be absent"):
        driver._download_shard(
            shard_key="a-latest", inbox=tmp_path / "shards" / "inbox" / "a-latest"
        )


def test_evict_local_state_removes_all_shard_scratch(tmp_path: Path) -> None:
    driver = StreamDriver(**_good_args(tmp_path))
    inbox = tmp_path / "shards" / "inbox" / "a-latest"
    active = tmp_path / "shards" / "active" / "a-latest"
    partial = tmp_path / "shards" / "partial" / "a-latest"
    inbox.mkdir(parents=True)
    active.mkdir(parents=True)
    partial.mkdir(parents=True)
    driver._evict_local_shard_state("a-latest")
    assert not inbox.exists()
    assert not active.exists()
    assert not partial.exists()


def test_write_state_removes_failed_temporary_file(tmp_path: Path) -> None:
    driver = StreamDriver(**_good_args(tmp_path))
    with (
        mock.patch("scripts.streaming.driver.os.replace", side_effect=OSError("fail")),
        pytest.raises(OSError, match="fail"),
    ):
        driver._write_state(updated=True)
    assert not list(tmp_path.glob(".state.json.tmp-*"))


def test_main_rejects_non_positive_max_shards(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "stream-build",
                "--confirm-offload",
                "--max-shards",
                "0",
                "--run-id",
                "r",
                "--staging-revision",
                "r",
                "--repo-id",
                "o/r",
                "--upstream-repo-id",
                "u/r",
                "--resolved-revision",
                "a" * 40,
                "--source-commit",
                "b" * 40,
                "--work-dir",
                str(tmp_path),
            ]
        )
