"""Contracts for lossless cross-site split resume bundles."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from scripts.streaming import resume_bundle as bundle_module
from scripts.streaming.resume_bundle import (
    ResumeBundleError,
    create_resume_bundle,
    merge_resume_bundle,
    valid_streaming_shard_key,
    validate_resume_bundle,
    validate_streaming_state,
)

from osm_polygon_sentence_relevance.application._checkpoint.inventory import (
    SourceFileEntry,
)
from osm_polygon_sentence_relevance.application._checkpoint.partial import (
    append_partial_batch,
    create_partial_state,
    load_partial_state,
)
from osm_polygon_sentence_relevance.contracts.schemas import (
    SEGMENTED_SENTENCES_SCHEMA,
)
from osm_polygon_sentence_relevance.sentences.segmentation import SegmentationReport
from osm_polygon_sentence_relevance.sentences.table import SegmentedBatch

IDENTITY: dict[str, str | int] = {
    "repo_id": "owner/output",
    "resolved_revision": "a" * 40,
    "source_commit": "b" * 40,
    "run_id": "c" * 20,
    "staging_revision": "checkpoints/" + "c" * 20,
    "pipeline_version": "0.1.0",
    "model_name": "sat-12l-sm",
    "batch_size": 128,
}
SHARD = "romania-latest"
SOURCE_FILES = [SourceFileEntry(f"polygons/{SHARD}.parquet", 10, "d" * 64)]
INPUT_ROOT = Path("/home/nflandre/osm-polygon-operator/run/work/shards/inbox")


def _report() -> SegmentationReport:
    return SegmentationReport(
        input_section_occurrence_count=128,
        emitted_segment_count=128,
        retained_sentence_occurrence_count=128,
        dropped_empty_raw_count=0,
        dropped_empty_normalized_count=0,
        wikipedia_sentence_occurrence_count=128,
        wikivoyage_sentence_occurrence_count=0,
    )


def _batch(start: int, end: int) -> SegmentedBatch:
    return SegmentedBatch(
        start_index=start,
        end_index=end,
        table=SEGMENTED_SENTENCES_SCHEMA.empty_table(),
        report=_report(),
    )


def _write_state(
    work: Path,
    *shards: str,
    identity: dict[str, str | int] = IDENTITY,
) -> None:
    work.mkdir(parents=True, mode=0o700, exist_ok=True)
    payload: dict[str, object] = {
        **identity,
        "schema_version": 1,
        "last_updated": True,
        "verified_checkpoints": {
            shard: {
                "segmented_table_sha256": str(index + 1) * 64,
                "segmented_table_bytes": index + 1,
            }
            for index, shard in enumerate(shards)
        },
    }
    (work / "state.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.chmod(work / "state.json", 0o600)


def _write_partial(work: Path, *, batches: int = 1) -> None:
    state = create_partial_state(
        work,
        shard_key=SHARD,
        source_commit=str(IDENTITY["source_commit"]),
        input_dataset_revision=str(IDENTITY["resolved_revision"]),
        pipeline_version=str(IDENTITY["pipeline_version"]),
        model_name=str(IDENTITY["model_name"]),
        batch_size=int(IDENTITY["batch_size"]),
        input_root=INPUT_ROOT,
        source_files=SOURCE_FILES,
        total_sections=512,
    )
    for index in range(batches):
        state = append_partial_batch(state, _batch(index * 128, (index + 1) * 128))


def test_bundle_round_trip_preserves_ledger_and_partial_progress(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-work"
    _write_state(source, "afghanistan-latest", "albania-latest")
    _write_partial(source, batches=2)

    bundle = create_resume_bundle(source, tmp_path / "bundle", IDENTITY)
    validated = validate_resume_bundle(bundle.root, IDENTITY)
    destination = tmp_path / "destination-work"
    result = merge_resume_bundle(destination, validated.root, IDENTITY)

    assert result.imported
    assert result.completed_shards == 2
    assert result.partial_shard == SHARD
    state = json.loads((destination / "state.json").read_text(encoding="utf-8"))
    assert tuple(state["verified_checkpoints"]) == (
        "afghanistan-latest",
        "albania-latest",
    )
    partial = load_partial_state(
        destination,
        shard_key=SHARD,
        source_commit=str(IDENTITY["source_commit"]),
        input_dataset_revision=str(IDENTITY["resolved_revision"]),
        pipeline_version=str(IDENTITY["pipeline_version"]),
        model_name=str(IDENTITY["model_name"]),
        batch_size=int(IDENTITY["batch_size"]),
        input_root=INPUT_ROOT,
        source_files=SOURCE_FILES,
        total_sections=512,
    )
    assert partial is not None
    assert partial.next_section_index == 256


def test_merge_unions_non_conflicting_completed_ledgers(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_state(source, "albania-latest")
    _write_state(destination, "afghanistan-latest")
    bundle = create_resume_bundle(source, tmp_path / "bundle", IDENTITY)

    result = merge_resume_bundle(destination, bundle.root, IDENTITY)

    assert result.completed_shards == 2
    state = json.loads((destination / "state.json").read_text(encoding="utf-8"))
    assert set(state["verified_checkpoints"]) == {
        "afghanistan-latest",
        "albania-latest",
    }


def test_merge_keeps_the_more_advanced_matching_partial(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_state(source)
    _write_state(destination)
    _write_partial(source, batches=2)
    _write_partial(destination, batches=1)
    bundle = create_resume_bundle(source, tmp_path / "bundle", IDENTITY)

    result = merge_resume_bundle(destination, bundle.root, IDENTITY)

    assert result.partial_shard == SHARD
    progress = json.loads(
        (destination / "shards" / "partial" / SHARD / "progress.json").read_text()
    )
    assert progress["next_section_index"] == 256


def test_bundle_rejects_identity_drift_before_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_state(source)
    wrong = {**IDENTITY, "source_commit": "e" * 40}

    with pytest.raises(ResumeBundleError, match="source_commit"):
        create_resume_bundle(source, tmp_path / "bundle", wrong)
    assert not (tmp_path / "bundle").exists()


def test_bundle_rejects_multiple_partial_shards(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_state(source)
    _write_partial(source)
    other = source / "shards" / "partial" / "zambia-latest"
    other.mkdir(mode=0o700)

    with pytest.raises(ResumeBundleError, match="at most one partial shard"):
        create_resume_bundle(source, tmp_path / "bundle", IDENTITY)


def test_bundle_rejects_corrupt_partial_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_state(source)
    _write_partial(source)
    batch = next((source / "shards" / "partial" / SHARD).glob("batch-*.parquet"))
    batch.write_bytes(b"corrupt")
    os.chmod(batch, 0o600)

    with pytest.raises(ResumeBundleError, match="partial"):
        create_resume_bundle(source, tmp_path / "bundle", IDENTITY)


def test_merge_rejects_conflicting_checkpoint_descriptor(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_state(source, "afghanistan-latest")
    _write_state(destination, "afghanistan-latest")
    destination_state = json.loads((destination / "state.json").read_text())
    destination_state["verified_checkpoints"]["afghanistan-latest"][
        "segmented_table_sha256"
    ] = "f" * 64
    (destination / "state.json").write_text(json.dumps(destination_state))
    os.chmod(destination / "state.json", 0o600)
    bundle = create_resume_bundle(source, tmp_path / "bundle", IDENTITY)

    with pytest.raises(ResumeBundleError, match="conflicting checkpoint"):
        merge_resume_bundle(destination, bundle.root, IDENTITY)


def test_validate_rejects_manifest_hash_tampering(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_state(source)
    bundle = create_resume_bundle(source, tmp_path / "bundle", IDENTITY)
    (bundle.root / "state.json").write_text("{}", encoding="utf-8")
    os.chmod(bundle.root / "state.json", 0o600)

    with pytest.raises(ResumeBundleError, match="hash mismatch"):
        validate_resume_bundle(bundle.root, IDENTITY)


@pytest.mark.parametrize(
    "value",
    [None, "", "Upper", "has space", "slash/name", "dollar$", True],
)
def test_shard_key_validator_rejects_unsafe_values(value: object) -> None:
    assert not valid_streaming_shard_key(value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: [], "JSON object"),
        (lambda payload: {**payload, "source_commit": "e" * 40}, "source_commit"),
        (lambda payload: {**payload, "schema_version": 2}, "schema version"),
        (lambda payload: {**payload, "verified_checkpoints": []}, "must be an object"),
        (
            lambda payload: {**payload, "verified_checkpoints": {"BAD": {}}},
            "ledger is malformed",
        ),
        (
            lambda payload: {
                **payload,
                "verified_checkpoints": {"safe-latest": []},
            },
            "ledger is malformed",
        ),
        (
            lambda payload: {
                **payload,
                "verified_checkpoints": {"safe-latest": {"wrong": 1}},
            },
            "descriptor is malformed",
        ),
        (
            lambda payload: {
                **payload,
                "verified_checkpoints": {
                    "safe-latest": {
                        "segmented_table_sha256": "x" * 64,
                        "segmented_table_bytes": 1,
                    }
                },
            },
            "descriptor is malformed",
        ),
        (
            lambda payload: {
                **payload,
                "verified_checkpoints": {
                    "safe-latest": {
                        "segmented_table_sha256": "a" * 64,
                        "segmented_table_bytes": True,
                    }
                },
            },
            "descriptor is malformed",
        ),
    ],
)
def test_streaming_state_validation_rejects_malformed_contracts(
    mutation: object, message: str
) -> None:
    payload: object = {
        **IDENTITY,
        "schema_version": 1,
        "verified_checkpoints": {},
    }
    assert callable(mutation)
    with pytest.raises(ResumeBundleError, match=message):
        validate_streaming_state(mutation(payload), IDENTITY)  # type: ignore[operator]


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        [{}],
        [{"path": "/absolute", "size": 1, "sha256": "a" * 64}],
        [{"path": "../up", "size": 1, "sha256": "a" * 64}],
        [{"path": "ok", "size": -1, "sha256": "a" * 64}],
        [{"path": "ok", "size": 1, "sha256": "bad"}],
    ],
)
def test_source_manifest_parser_rejects_invalid_entries(value: object) -> None:
    with pytest.raises(ResumeBundleError, match="source manifest"):
        bundle_module._source_entries(value)


def test_partial_root_rejects_symlink_and_invalid_key(tmp_path: Path) -> None:
    work = tmp_path / "work"
    partial = work / "shards" / "partial"
    partial.mkdir(parents=True)
    target = tmp_path / "target"
    target.mkdir()
    (partial / "link").symlink_to(target)
    with pytest.raises(ResumeBundleError, match="unsafe entry"):
        bundle_module._partial_directories(work)
    (partial / "link").unlink()
    (partial / "INVALID").mkdir()
    with pytest.raises(ResumeBundleError, match="invalid shard key"):
        bundle_module._partial_directories(work)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_root", "relative", "input_root"),
        ("total_sections", True, "total_sections"),
        ("source_files", [], "source manifest"),
    ],
)
def test_partial_metadata_rejects_invalid_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    work = tmp_path / "work"
    _write_state(work)
    _write_partial(work)
    progress = work / "shards" / "partial" / SHARD / "progress.json"
    payload = json.loads(progress.read_text())
    payload[field] = value
    progress.write_text(json.dumps(payload))
    os.chmod(progress, 0o600)
    with pytest.raises(ResumeBundleError, match=message):
        create_resume_bundle(work, tmp_path / "bundle", IDENTITY)


def test_bundle_rejects_existing_destination_and_completed_partial(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    _write_state(work)
    destination = tmp_path / "bundle"
    destination.mkdir()
    with pytest.raises(ResumeBundleError, match="already exists"):
        create_resume_bundle(work, destination, IDENTITY)

    other = tmp_path / "other"
    _write_state(other, SHARD)
    _write_partial(other)
    with pytest.raises(ResumeBundleError, match="already present"):
        create_resume_bundle(other, tmp_path / "other-bundle", IDENTITY)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: {**payload, "schema_version": 9}, "schema version"),
        (lambda payload: {**payload, "identity": {}}, "identity"),
        (lambda payload: {**payload, "files": []}, "file inventory"),
        (
            lambda payload: {**payload, "files": {"../state.json": "a" * 64}},
            "file inventory",
        ),
        (lambda payload: {**payload, "snapshot_id": "0" * 20}, "snapshot identity"),
    ],
)
def test_bundle_manifest_validation_rejects_invalid_contract(
    tmp_path: Path, mutate: object, message: str
) -> None:
    work = tmp_path / "work"
    _write_state(work)
    bundle = create_resume_bundle(work, tmp_path / "bundle", IDENTITY)
    manifest = bundle.root / "inventory.json"
    payload = json.loads(manifest.read_text())
    assert callable(mutate)
    manifest.write_text(json.dumps(mutate(payload)))  # type: ignore[operator]
    os.chmod(manifest, 0o600)
    with pytest.raises(ResumeBundleError, match=message):
        validate_resume_bundle(bundle.root, IDENTITY)


def test_bundle_validation_rejects_extra_file(tmp_path: Path) -> None:
    work = tmp_path / "work"
    _write_state(work)
    bundle = create_resume_bundle(work, tmp_path / "bundle", IDENTITY)
    extra = bundle.root / "extra"
    extra.write_text("x")
    os.chmod(extra, 0o600)
    with pytest.raises(ResumeBundleError, match="unexpected or missing"):
        validate_resume_bundle(bundle.root, IDENTITY)


def test_merge_keeps_more_advanced_destination_partial(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_state(source)
    _write_state(destination)
    _write_partial(source, batches=1)
    _write_partial(destination, batches=2)
    bundle = create_resume_bundle(source, tmp_path / "bundle", IDENTITY)

    merge_resume_bundle(destination, bundle.root, IDENTITY)

    progress = json.loads(
        (destination / "shards" / "partial" / SHARD / "progress.json").read_text()
    )
    assert progress["next_section_index"] == 256


def test_merge_rejects_different_active_partial_shards(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_state(source)
    _write_state(destination)
    _write_partial(source)
    _write_partial(destination)
    source_partial = source / "shards" / "partial" / SHARD
    destination_partial = destination / "shards" / "partial" / SHARD
    imported = bundle_module._validate_partial(source, IDENTITY)
    local = bundle_module._validate_partial(destination, IDENTITY)
    assert imported is not None
    assert local is not None
    divergent = replace(imported, shard_key="zambia-latest", directory=source_partial)
    with pytest.raises(ResumeBundleError, match="divergent partial"):
        bundle_module._merge_partial(
            destination,
            imported_partial=divergent,
            destination_partial=replace(local, directory=destination_partial),
            merged_ledger={},
        )


def test_merge_discards_partial_for_completed_shard(tmp_path: Path) -> None:
    destination = tmp_path / "destination"
    _write_state(destination)
    _write_partial(destination)
    local = bundle_module._validate_partial(destination, IDENTITY)
    assert local is not None
    assert (
        bundle_module._merge_partial(
            destination,
            imported_partial=None,
            destination_partial=local,
            merged_ledger={SHARD: object()},
        )
        is None
    )
    assert not local.directory.exists()


def test_filesystem_guards_reject_missing_symlink_and_bad_mode(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ResumeBundleError, match="directory is inaccessible"):
        bundle_module._ensure_directory(missing)
    regular = tmp_path / "regular"
    regular.write_text("x")
    with pytest.raises(ResumeBundleError, match="not a real directory"):
        bundle_module._ensure_directory(regular)
    with pytest.raises(ResumeBundleError, match="file is inaccessible"):
        bundle_module._ensure_regular(missing, 0o600)
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ResumeBundleError, match="not regular"):
        bundle_module._ensure_regular(directory, 0o600)
    os.chmod(regular, 0o644)
    with pytest.raises(ResumeBundleError, match="unsafe mode"):
        bundle_module._ensure_regular(regular, 0o600)


def test_json_loader_rejects_malformed_and_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text("{")
    os.chmod(path, 0o600)
    with pytest.raises(ResumeBundleError, match="malformed"):
        bundle_module._load_json_mapping(path, "payload")
    path.write_text("[]")
    os.chmod(path, 0o600)
    with pytest.raises(ResumeBundleError, match="JSON object"):
        bundle_module._load_json_mapping(path, "payload")


def test_create_cleans_temporary_bundle_after_copy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    _write_state(work)
    monkeypatch.setattr(
        bundle_module.shutil,
        "copyfile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("copy failed")),
    )
    with pytest.raises(OSError, match="copy failed"):
        create_resume_bundle(work, tmp_path / "bundle", IDENTITY)
    assert not any(tmp_path.glob(".bundle.*"))


def test_merge_ignores_imported_partial_already_completed_at_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_state(source)
    _write_partial(source)
    _write_state(destination, SHARD)
    bundle = create_resume_bundle(source, tmp_path / "bundle", IDENTITY)

    result = merge_resume_bundle(destination, bundle.root, IDENTITY)

    assert result.partial_shard is None
    assert not (destination / "shards" / "partial" / SHARD).exists()


def test_merge_rejects_divergent_matching_partial_history(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_state(source)
    _write_state(destination)
    _write_partial(source)
    _write_partial(destination)
    imported = bundle_module._validate_partial(source, IDENTITY)
    local = bundle_module._validate_partial(destination, IDENTITY)
    assert imported is not None
    assert local is not None
    divergent_batch = replace(imported.batches[0], sha256="f" * 64)
    divergent = replace(imported, batches=(divergent_batch,))
    with pytest.raises(ResumeBundleError, match="histories diverge"):
        bundle_module._merge_partial(
            destination,
            imported_partial=divergent,
            destination_partial=local,
            merged_ledger={},
        )
