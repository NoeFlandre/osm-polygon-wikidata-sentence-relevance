"""Strict TDD coverage for durable local operator state and event logs."""

from __future__ import annotations

import json
import math
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from osm_polygon_sentence_relevance.operator import OperatorConfig
from osm_polygon_sentence_relevance.operator.state import (
    EVENT_SCHEMA_VERSION,
    EVENTS_FILENAME,
    RUN_MODE,
    STATE_FILENAME,
    STATE_SCHEMA_VERSION,
    RunPhase,
    RunState,
    StateError,
    StateIdentityMismatch,
    StateSecurityError,
    StateStore,
    StateTransitionError,
    _coerce_facts,
    _ensure_no_symlink_ancestors,
    _open_secure_no_follow,
    _parse_json_timestamp,
    _parse_utc_timestamp,
    _read_event_lines,
    _read_file_bytes,
    _read_text_file,
    _remove_temporary_if_regular,
    _validate_directory,
    _validate_json_value,
)


class _Clock:
    """Deterministic injectable clock."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self) -> str:
        timestamp = f"2026-07-28T{len(self.calls):02d}:00:00+00:00"
        self.calls.append(timestamp)
        return timestamp


def _identity(*, stage: str = "split") -> object:
    return OperatorConfig.build(
        scope="region",
        region="afghanistan-latest",
        stage=stage,
        source_commit="a" * 40,
        input_revision="b" * 40,
    ).run_identity


def _run_dir(root: Path, identity: object) -> Path:
    return root / "runs" / identity.run_id


def _write_state_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _state_payload(run_id: str, run_identity: object) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "run_id": run_id,
        "run_identity": json.loads(run_identity.canonical_json),
        "phase": RunPhase.CREATED.value,
        "sequence": 0,
        "timestamp": "2026-07-28T00:00:00+00:00",
        "facts": {},
    }


def _events(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_load_or_create_creates_run_directory_and_state(tmp_path: Path) -> None:
    identity = _identity()
    clock = _Clock()
    store = StateStore(data_root=tmp_path, clock=clock)

    run_state = store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)

    assert run_state.phase == RunPhase.CREATED
    assert run_state.sequence == 0
    assert (run_dir / "state.json").exists()
    assert oct(run_dir.stat().st_mode & 0o777) == "0o700"


def test_load_or_create_uses_injected_clock_once(tmp_path: Path) -> None:
    identity = _identity()
    clock = _Clock()
    store = StateStore(data_root=tmp_path, clock=clock)

    first = store.load_or_create(identity)
    second = store.load_or_create(identity)

    assert first.timestamp == "2026-07-28T00:00:00+00:00"
    assert len(clock.calls) == 1
    assert second == first


def test_existing_matching_identity_resumes_without_bytes_change(
    tmp_path: Path,
) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    run = store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)
    state_file = run_dir / "state.json"
    original = state_file.read_bytes()

    resumed = store.load_or_create(identity)

    assert resumed == run
    assert state_file.read_bytes() == original


def test_state_identity_mismatch_rejects_persisted_run_identity_exactly(
    tmp_path: Path,
) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    state_path = _run_dir(tmp_path, identity) / "state.json"
    payload = _state_payload(identity.run_id, identity)
    payload["run_identity"]["region"] = "iran-latest"
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateIdentityMismatch):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_transition_increments_sequence_and_updates_phase(tmp_path: Path) -> None:
    identity = _identity(stage="label")
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    transitioned = store.transition(
        expected=RunPhase.CREATED,
        target=RunPhase.INPUTS_RESOLVED,
        facts={"input_revision": "a" * 40},
    )

    assert transitioned.sequence == 1
    assert transitioned.phase == RunPhase.INPUTS_RESOLVED
    assert transitioned.facts["input_revision"] == "a" * 40


def test_wrong_expected_phase_preserves_exact_state_bytes(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    state_path = _run_dir(tmp_path, identity) / "state.json"
    before = state_path.read_bytes()

    with pytest.raises(StateTransitionError):
        store.transition(
            expected=RunPhase.CHECKPOINTED,
            target=RunPhase.VALIDATED,
            facts={"input_revision": "a" * 40},
        )

    assert state_path.read_bytes() == before


def test_temporary_state_file_is_not_created_when_rejected(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)
    (run_dir / ".state.json.tmp").write_text("pre", encoding="utf-8")

    with pytest.raises(StateSecurityError):
        store.transition(
            expected=RunPhase.CREATED,
            target=RunPhase.INPUTS_RESOLVED,
            facts={"input_revision": "a" * 40},
        )


def test_state_event_file_mode_and_directory_mode(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    store.transition(
        expected=RunPhase.CREATED,
        target=RunPhase.INPUTS_RESOLVED,
        facts={"input_revision": "a" * 40},
    )
    store.append_event(
        level="info", message="immutable", facts={"input_revision": "a" * 40}
    )

    run_dir = _run_dir(tmp_path, identity)
    assert oct(run_dir.stat().st_mode & 0o777) == "0o700"
    assert oct((run_dir / "state.json").stat().st_mode & 0o777) == "0o600"
    assert oct((run_dir / "events.jsonl").stat().st_mode & 0o777) == "0o600"


def test_symlink_data_root_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "root"
    real_root.mkdir()
    root_link = tmp_path / "root-link"
    root_link.symlink_to(real_root)

    with pytest.raises(StateSecurityError):
        StateStore(data_root=root_link).load_or_create(_identity())


def test_symlink_run_directory_rejected_before_state_read(tmp_path: Path) -> None:
    identity = _identity()
    run_root = _run_dir(tmp_path, identity)
    run_root.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    run_root.symlink_to(outside)

    with pytest.raises(StateSecurityError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_operations_reject_run_directory_symlink(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    run_dir = _run_dir(tmp_path, identity)
    backup = tmp_path / "run-dir-backup"
    backup.mkdir()
    shutil.rmtree(run_dir)
    run_dir.symlink_to(backup)

    with pytest.raises(StateSecurityError):
        store.load()
    with pytest.raises(StateSecurityError):
        store.transition(expected=RunPhase.CREATED, target=RunPhase.INPUTS_RESOLVED)
    with pytest.raises(StateSecurityError):
        store.append_event(level="info", message="ignored")


def test_existing_run_directory_with_insecure_mode_rejected(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    run_dir.mkdir(parents=True)
    os.chmod(run_dir, 0o777)

    payload = _state_payload(identity.run_id, identity)
    _write_state_file(run_dir / "state.json", payload)

    with pytest.raises(StateSecurityError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_operations_reject_run_directory_mode_and_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    run_dir = _run_dir(tmp_path, identity)
    os.chmod(run_dir, 0o755)

    with pytest.raises(StateSecurityError):
        store.transition(expected=RunPhase.CREATED, target=RunPhase.INPUTS_RESOLVED)
    with pytest.raises(StateSecurityError):
        store.append_event(level="info", message="ignored")
    with pytest.raises(StateSecurityError):
        store.load()


def test_operations_reject_wrong_owner_after_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not hasattr(os, "getuid"):
        pytest.skip("Owner enforcement unavailable")

    import osm_polygon_sentence_relevance.operator.state as state_module

    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    monkeypatch.setattr(state_module.os, "getuid", lambda: 99_999)

    with pytest.raises(StateSecurityError):
        store.load()
    with pytest.raises(StateSecurityError):
        store.transition(expected=RunPhase.CREATED, target=RunPhase.INPUTS_RESOLVED)
    with pytest.raises(StateSecurityError):
        store.append_event(level="info", message="ignored")


def test_wrong_owner_root_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not hasattr(os, "getuid"):
        pytest.skip("Owner enforcement unavailable")
    monkeypatch.setattr(
        "osm_polygon_sentence_relevance.operator.state.os.getuid", lambda: 99_999
    )

    with pytest.raises(StateSecurityError):
        StateStore(data_root=tmp_path).load_or_create(_identity())


def test_malformed_state_json_is_rejected(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_missing_fields_are_rejected(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps({"schema_version": STATE_SCHEMA_VERSION}), encoding="utf-8"
    )

    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_unsupported_schema_version_is_rejected(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    run_dir.mkdir(parents=True)
    payload = _state_payload(identity.run_id, identity)
    payload["schema_version"] = 999
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_events_append_in_order_and_survive_reload(tmp_path: Path) -> None:
    identity = _identity()
    clock = _Clock()
    store = StateStore(data_root=tmp_path, clock=clock)
    store.load_or_create(identity)

    for index in range(3):
        store.append_event(
            level="info", message=f"event-{index}", facts={"checkpoint": index}
        )

    store.transition(
        expected=RunPhase.CREATED,
        target=RunPhase.INPUTS_RESOLVED,
        facts={"input_revision": "a" * 40},
    )
    store.append_event(level="info", message="post-transition", facts={"checkpoint": 3})

    events = _events(_run_dir(tmp_path, identity) / "events.jsonl")
    assert [event["message"] for event in events] == [
        "event-0",
        "event-1",
        "event-2",
        "post-transition",
    ]
    assert [event["event_sequence"] for event in events] == [0, 1, 2, 3]

    reloaded = StateStore(data_root=tmp_path).load_or_create(identity)
    assert reloaded.phase == RunPhase.INPUTS_RESOLVED


def test_events_use_append_timestamp_and_monotonic_sequence(tmp_path: Path) -> None:
    identity = _identity()
    clock = _Clock()
    store = StateStore(data_root=tmp_path, clock=clock)
    store.load_or_create(identity)

    store.append_event(level="info", message="a")
    store.append_event(level="info", message="b")

    events = _events(_run_dir(tmp_path, identity) / "events.jsonl")
    assert [event["timestamp"] for event in events] == [
        "2026-07-28T01:00:00+00:00",
        "2026-07-28T02:00:00+00:00",
    ]
    assert [event["event_sequence"] for event in events] == [0, 1]
    assert len(clock.calls) == 3


def test_events_are_compact_json_objects_per_line(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    store.append_event(level="info", message="compact", facts={"checkpoint": 1})

    raw = (
        (_run_dir(tmp_path, identity) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(raw) == 1
    assert json.loads(raw[0]) == json.loads(raw[0])
    assert json.dumps(json.loads(raw[0]), separators=(",", ":")) == raw[0]


def test_recursive_unsafe_facts_keys_rejected_and_not_written(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    with pytest.raises(StateSecurityError):
        store.append_event(
            level="info", message="unsafe", facts={"meta": {"response": {"v": 1}}}
        )

    assert not (_run_dir(tmp_path, identity) / "events.jsonl").exists()


def test_facts_with_non_json_object_rejected(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    with pytest.raises(StateError):
        store.transition(
            expected=RunPhase.CREATED,
            target=RunPhase.INPUTS_RESOLVED,
            facts={"value": object()},
        )


def test_nan_and_infinity_rejected_in_facts(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    with pytest.raises(StateError):
        store.append_event(level="info", message="bad", facts={"v": math.nan})

    with pytest.raises(StateError):
        store.append_event(level="info", message="bad", facts={"v": math.inf})


def test_injected_clock_is_deterministic_for_events(tmp_path: Path) -> None:
    clock = _Clock()
    store = StateStore(data_root=tmp_path, clock=clock)
    store.load_or_create(_identity())
    store.transition(
        expected=RunPhase.CREATED,
        target=RunPhase.INPUTS_RESOLVED,
        facts={"input_revision": "a" * 40},
    )
    store.append_event(level="info", message="timed")

    events = _events(_run_dir(tmp_path, _identity()) / "events.jsonl")
    assert events[-1]["timestamp"] == "2026-07-28T02:00:00+00:00"


def test_state_operations_do_not_write_outside_root(tmp_path: Path) -> None:
    opened: list[Path] = []
    real_os_open = os.open

    def tracked_open(path: os.PathLike[str] | str, flags: int, *args: object) -> int:
        opened.append(Path(path).resolve())
        return real_os_open(path, flags, *args)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "osm_polygon_sentence_relevance.operator.state.os.open", tracked_open
    )
    try:
        store = StateStore(data_root=tmp_path)
        store.load_or_create(_identity())
        store.transition(
            expected=RunPhase.CREATED,
            target=RunPhase.INPUTS_RESOLVED,
            facts={"input_revision": "a" * 40},
        )
        store.append_event(level="info", message="done")
    finally:
        monkeypatch.undo()

    assert opened
    assert all(path.is_relative_to(tmp_path.resolve()) for path in opened)


def test_public_exports_are_deliberate() -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    expected = {
        "RunPhase",
        "RunState",
        "StateStore",
        "StateError",
        "StateIdentityMismatch",
        "StateTransitionError",
        "StateSecurityError",
        "STATE_SCHEMA_VERSION",
    }
    assert expected.issubset(set(state_module.__all__))


def test_run_id_must_be_20_lowercase_hex_in_persisted_state(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    payload = _state_payload(identity.run_id, identity)
    payload["run_id"] = "A" * 20
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)

    payload["run_id"] = "abcdef"
    (run_dir / "state.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_facts_null_in_persisted_state_rejected(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    payload = _state_payload(identity.run_id, identity)
    payload["facts"] = None
    _write_state_file(run_dir / "state.json", payload)

    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_bool_schema_version_and_sequence_rejected(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    payload = _state_payload(identity.run_id, identity)
    payload["schema_version"] = True
    payload["sequence"] = False
    _write_state_file(run_dir / "state.json", payload)

    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_naive_or_non_utc_timestamp_is_rejected(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    payload = _state_payload(identity.run_id, identity)

    payload["timestamp"] = "2026-07-28T00:00:00"
    _write_state_file(run_dir / "state.json", payload)
    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)

    payload["timestamp"] = "2026-07-28T00:00:00+01:00"
    _write_state_file(run_dir / "state.json", payload)
    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)

    payload["timestamp"] = "not-a-time"
    _write_state_file(run_dir / "state.json", payload)
    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_state_with_unexpected_top_level_fields_is_rejected(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    payload = _state_payload(identity.run_id, identity)
    payload["extra"] = "bad"
    _write_state_file(run_dir / "state.json", payload)

    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_state_with_extra_run_identity_field_is_rejected(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    payload = _state_payload(identity.run_id, identity)
    payload["run_identity"]["run_id"] = identity.run_id
    _write_state_file(run_dir / "state.json", payload)

    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_existing_state_symlink_rejected(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    run_dir.mkdir(parents=True)
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    (run_dir / "state.json").symlink_to(target)

    with pytest.raises(StateSecurityError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_existing_state_non_regular_file_rejected(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    run_dir.mkdir(parents=True)
    state_dir = run_dir / "state.json"
    state_dir.mkdir()

    with pytest.raises(StateSecurityError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_existing_state_mode_rejected(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    run_dir.mkdir(parents=True)
    _write_state_file(run_dir / "state.json", _state_payload(identity.run_id, identity))
    os.chmod(run_dir / "state.json", 0o644)

    with pytest.raises(StateSecurityError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_existing_events_symlink_rejected_and_target_preserved(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)

    target = tmp_path / "events-target.jsonl"
    target.write_text('{"pinned":true}', encoding="utf-8")
    events_file = run_dir / "events.jsonl"
    events_file.symlink_to(target)

    with pytest.raises(StateSecurityError):
        store.append_event(level="info", message="fail")

    assert target.read_text(encoding="utf-8") == '{"pinned":true}'


def test_existing_events_non_regular_rejected(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)
    events_dir = run_dir / "events.jsonl"
    events_dir.mkdir()

    with pytest.raises(StateSecurityError):
        store.append_event(level="info", message="fail")


def test_existing_events_wrong_mode_rejected(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    events = _run_dir(tmp_path, identity) / "events.jsonl"
    events.write_text("{}\n", encoding="utf-8")
    os.chmod(events, 0o644)

    with pytest.raises(StateSecurityError):
        store.append_event(level="info", message="fail")


def test_existing_lock_symlink_rejected(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    run_dir = _run_dir(tmp_path, identity)
    link = run_dir / ".state.lock"
    if link.exists():
        link.unlink()
    link.symlink_to(tmp_path / "other")

    with pytest.raises(StateSecurityError):
        store.transition(
            expected=RunPhase.CREATED,
            target=RunPhase.INPUTS_RESOLVED,
            facts={"input_revision": "a" * 40},
        )


def test_existing_lock_wrong_mode_rejected(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    run_dir = _run_dir(tmp_path, identity)
    lock = run_dir / ".state.lock"
    lock.write_text("", encoding="utf-8")
    os.chmod(lock, 0o644)

    with pytest.raises(StateSecurityError):
        store.transition(
            expected=RunPhase.CREATED,
            target=RunPhase.INPUTS_RESOLVED,
            facts={"input_revision": "a" * 40},
        )


def test_stale_store_cannot_overwrite_transition(tmp_path: Path) -> None:
    identity = _identity()
    first = StateStore(data_root=tmp_path)
    second = StateStore(data_root=tmp_path)

    first.load_or_create(identity)
    second.load_or_create(identity)

    first.transition(
        expected=RunPhase.CREATED,
        target=RunPhase.INPUTS_RESOLVED,
        facts={"input_revision": "a" * 40},
    )

    with pytest.raises(StateTransitionError):
        second.transition(
            expected=RunPhase.CREATED,
            target=RunPhase.QUEUED,
            facts={"input_revision": "a" * 40},
        )


def test_two_stores_append_unique_event_sequence(tmp_path: Path) -> None:
    identity = _identity()
    store_a = StateStore(data_root=tmp_path)
    store_b = StateStore(data_root=tmp_path)
    store_a.load_or_create(identity)
    store_b.load_or_create(identity)

    def _append_many(store: StateStore, prefix: str) -> None:
        for i in range(5):
            store.append_event(level="info", message=f"{prefix}-{i}", facts={"n": i})

    t1 = threading.Thread(target=_append_many, args=(store_a, "a"))
    t2 = threading.Thread(target=_append_many, args=(store_b, "b"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    events = _events(_run_dir(tmp_path, identity) / "events.jsonl")
    sequences = [event["event_sequence"] for event in events]
    assert sequences == list(range(len(sequences)))
    assert len(set(sequences)) == len(sequences)


def test_malformed_jsonl_prevents_append_and_preserves_bytes(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)
    (run_dir / "events.jsonl").write_text('{"good": true}\n{not-json', encoding="utf-8")
    prior = (run_dir / "events.jsonl").read_text(encoding="utf-8")

    with pytest.raises(StateError):
        store.append_event(level="info", message="x")

    assert (run_dir / "events.jsonl").read_text(encoding="utf-8") == prior


def test_truncated_jsonl_prevents_append_and_preserves_bytes(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)
    (run_dir / "events.jsonl").write_text(
        '{"event_sequence": 0, "timestamp": "2026-07-28T00:00:00+00:00", "level": "info", "phase": "created", "message": "ok", "facts": {}}',
        encoding="utf-8",
    )
    prior = (run_dir / "events.jsonl").read_text(encoding="utf-8")

    with pytest.raises(StateError):
        store.append_event(level="info", message="next")

    assert (run_dir / "events.jsonl").read_text(encoding="utf-8") == prior


def test_tmp_state_file_symlink_is_not_overwritten(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)
    target = tmp_path / "tmp-target"
    target.write_text("original", encoding="utf-8")
    tmp_path_link = run_dir / ".state.json.tmp"
    tmp_path_link.symlink_to(target)

    with pytest.raises(StateSecurityError):
        store.transition(
            expected=RunPhase.CREATED,
            target=RunPhase.INPUTS_RESOLVED,
            facts={"input_revision": "a" * 40},
        )

    assert target.read_text(encoding="utf-8") == "original"


def test_failed_event_append_preserves_existing_bytes(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    event_path = run_dir / "events.jsonl"
    event_path.write_text(
        '{"event_sequence":0, "schema_version":1, "timestamp":"2026-07-28T00:00:00+00:00", "level":"info", "phase":"created", "message":"bad", "facts":{}}',
        encoding="utf-8",
    )
    prior = event_path.read_text(encoding="utf-8")

    with pytest.raises(StateError):
        store.append_event(level="info", message="next")

    assert event_path.read_text(encoding="utf-8") == prior


def test_lock_release_after_transition_failure(tmp_path: Path) -> None:
    identity = _identity()
    store_a = StateStore(data_root=tmp_path)
    store_b = StateStore(data_root=tmp_path)
    store_a.load_or_create(identity)
    store_b.load_or_create(identity)

    original = store_a._write_state

    def _failing_write(*args: Any, **kwargs: Any) -> None:
        raise StateError("forced")

    store_a._write_state = _failing_write

    def _run() -> None:
        with pytest.raises(StateError):
            store_a.transition(
                expected=RunPhase.CREATED,
                target=RunPhase.INPUTS_RESOLVED,
                facts={"input_revision": "a" * 40},
            )

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()

    # If the lock were not released on failure, this would hang/fail.
    store_b.append_event(level="info", message="can-proceed")

    store_a._write_state = original


def test_events_lock_release_after_failure(tmp_path: Path) -> None:
    identity = _identity()
    store_a = StateStore(data_root=tmp_path)
    store_b = StateStore(data_root=tmp_path)
    store_a.load_or_create(identity)
    store_b.load_or_create(identity)

    run_dir = _run_dir(tmp_path, identity)
    event_path = run_dir / "events.jsonl"
    event_path.write_text("{not-json", encoding="utf-8")

    def _run() -> None:
        with pytest.raises(StateError):
            store_a.append_event(level="info", message="retry")

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()

    # Next append should still reach event path and fail deterministically on same invalid file,
    # not deadlock on an unreleased lock.
    with pytest.raises(StateError):
        store_b.append_event(level="info", message="retry-2")


def test_load_without_active_run_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(StateError, match="no active run"):
        StateStore(data_root=tmp_path).load()


def test_noop_transition_allows_missing_facts(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    transitioned = store.transition(
        expected=RunPhase.CREATED, target=RunPhase.INPUTS_RESOLVED
    )
    assert transitioned.facts == {}


def test_append_event_empty_level_or_message_rejected(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    with pytest.raises(StateError, match="level must"):
        store.append_event(level="   ", message="ok")
    with pytest.raises(StateError, match="message must"):
        store.append_event(level="info", message="   ")


def test_mutating_caller_facts_does_not_change_persisted_state(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    facts = {"nested": {"items": [1, 2, 3]}}
    state = store.transition(
        expected=RunPhase.CREATED,
        target=RunPhase.INPUTS_RESOLVED,
        facts=facts,
    )
    facts["nested"]["items"].append(4)

    reloaded = store.load()
    assert state == reloaded
    assert reloaded.facts == {
        "nested": {"items": [1, 2, 3]},
    }


def test_data_root_must_exist_and_be_directory(tmp_path: Path) -> None:
    identity = _identity()
    with pytest.raises(StateSecurityError):
        StateStore(data_root=tmp_path / "missing").load_or_create(identity)

    file_root = tmp_path / "not-a-dir"
    file_root.write_text("x", encoding="utf-8")
    with pytest.raises(StateSecurityError):
        StateStore(data_root=file_root).load_or_create(identity)


def test_parse_state_run_id_length_is_rejected(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    run_dir.mkdir(parents=True)
    payload = _state_payload(identity.run_id, identity)
    payload["run_id"] = "x" * 10
    _write_state_file(run_dir / "state.json", payload)

    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_run_identity_field_is_json_safe_and_finite(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    state = store.load_or_create(identity)
    assert state.run_identity == json.loads(identity.canonical_json)


def test_load_or_create_with_invalid_run_id_rejects_identity_regex(
    tmp_path: Path,
) -> None:
    identity = _identity()
    object.__setattr__(identity, "run_id", "Z" * 20)
    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_parse_state_rejects_non_string_phase_and_invalid_phase(tmp_path: Path) -> None:
    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    run_dir.mkdir(parents=True)
    payload = _state_payload(identity.run_id, identity)
    payload["phase"] = 2
    _write_state_file(run_dir / "state.json", payload)
    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)

    payload["phase"] = "bad"
    _write_state_file(run_dir / "state.json", payload)
    with pytest.raises(StateError):
        StateStore(data_root=tmp_path).load_or_create(identity)


def test_validate_json_scalars_and_finite_float() -> None:
    assert _validate_json_value(True) is True
    assert _validate_json_value(None) is None
    assert _validate_json_value(1.5) == 1.5


def test_validate_json_rejects_non_string_mapping_key() -> None:
    with pytest.raises(StateError, match="string keys"):
        _validate_json_value({1: "value"})


def test_coerce_facts_rejects_non_mapping() -> None:
    with pytest.raises(StateError, match="facts must be a mapping"):
        _coerce_facts(object())


def test_parse_json_timestamp_rejects_non_string_and_accepts_z_suffix() -> None:
    with pytest.raises(StateError, match="state timestamp must be a string"):
        _parse_json_timestamp(123)

    assert _parse_json_timestamp("2026-07-28T00:00:00Z") == "2026-07-28T00:00:00Z"


def test_open_secure_no_follow_fallback_is_exercised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    original_open = state_module.os.open
    calls = {"count": 0}

    def mock_open(
        path: os.PathLike[str] | str, flags: int, mode: int | None = None
    ) -> int:
        calls["count"] += 1
        if calls["count"] == 1:
            raise AttributeError("forced nofollow fallback path")
        if mode is None:
            return original_open(path, flags)
        return original_open(path, flags, mode)

    monkeypatch.setattr(state_module.os, "open", mock_open)
    target = tmp_path / "fallback.txt"

    fd = _open_secure_no_follow(target, os.O_CREAT | os.O_RDWR, 0o600)
    assert fd >= 0
    os.close(fd)


def test_validate_directory_rejects_symlink_missing_and_mode_mismatch(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(StateSecurityError):
        _validate_directory(link)

    with pytest.raises(StateSecurityError):
        _validate_directory(tmp_path / "missing")

    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(StateSecurityError):
        _validate_directory(file_path)

    dir_path = tmp_path / "mode"
    dir_path.mkdir()
    os.chmod(dir_path, 0o755)
    with pytest.raises(StateSecurityError):
        _validate_directory(dir_path, mode=0o700, label="run directory mode")


def test_read_text_file_rejects_invalid_json_and_non_object(tmp_path: Path) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{broken", encoding="utf-8")
    os.chmod(invalid_json, 0o600)
    with pytest.raises(StateError, match="invalid JSON payload"):
        _read_text_file(invalid_json)

    list_payload = tmp_path / "list.json"
    list_payload.write_text('["bad"]', encoding="utf-8")
    os.chmod(list_payload, 0o600)
    with pytest.raises(StateError, match="payload must be a JSON object"):
        _read_text_file(list_payload)


def test_read_event_lines_rejects_malformed_and_invalid_sequence(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "events-malformed.jsonl"
    malformed.write_text("not-json\n", encoding="utf-8")
    os.chmod(malformed, 0o600)
    with pytest.raises(StateError, match="malformed event log"):
        _read_event_lines(malformed)

    no_newline = tmp_path / "events-no-newline.jsonl"
    no_newline.write_text(
        '{"event_sequence":0, "schema_version":1, "timestamp":"2026-07-28T00:00:00+00:00", "level":"info", "phase":"created", "message":"ok", "facts":{}}',
        encoding="utf-8",
    )
    os.chmod(no_newline, 0o600)
    with pytest.raises(StateError, match="missing trailing newline"):
        _read_event_lines(no_newline)

    blank = tmp_path / "events-blank.jsonl"
    blank.write_text("   \n", encoding="utf-8")
    os.chmod(blank, 0o600)
    with pytest.raises(StateError, match="malformed event log line"):
        _read_event_lines(blank)

    bad_entry = tmp_path / "events-bad-entry.jsonl"
    bad_entry.write_text("[]\n", encoding="utf-8")
    os.chmod(bad_entry, 0o600)
    with pytest.raises(StateError, match="malformed event log entry"):
        _read_event_lines(bad_entry)

    bad_sequence = tmp_path / "events-bad-sequence.jsonl"
    bad_sequence.write_text(
        '{"event_sequence":"0", "schema_version":1, "timestamp":"2026-07-28T00:00:00+00:00", '
        '"level":"info", "phase":"created", "message":"ok", "facts":{}}\n',
        encoding="utf-8",
    )
    os.chmod(bad_sequence, 0o600)
    with pytest.raises(
        StateError, match="event sequence must be a non-negative integer"
    ):
        _read_event_lines(bad_sequence)


def test_read_event_lines_rejects_bool_schema_and_sequence_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events-bool.jsonl"
    payload = {
        "schema_version": True,
        "event_sequence": False,
        "timestamp": "2026-07-28T00:00:00+00:00",
        "level": "info",
        "phase": RunPhase.CREATED.value,
        "message": "ok",
        "facts": {},
    }
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    os.chmod(path, 0o600)
    with pytest.raises(StateError, match="event schema version unsupported"):
        _read_event_lines(path)

    payload["schema_version"] = EVENT_SCHEMA_VERSION
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    with pytest.raises(
        StateError, match="event sequence must be a non-negative integer"
    ):
        _read_event_lines(path)


def test_transition_without_active_run_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(StateError, match="no active run"):
        StateStore(data_root=tmp_path).transition(
            expected=RunPhase.CREATED,
            target=RunPhase.INPUTS_RESOLVED,
        )


def test_load_rejects_state_with_mismatched_run_id(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)

    payload = _state_payload(identity.run_id, identity)
    payload["run_id"] = "0" * 20
    _write_state_file(run_dir / "state.json", payload)

    with pytest.raises(StateIdentityMismatch):
        store.load()


def test_load_or_create_runs_parent_validation_and_missing_state_errors(
    tmp_path: Path,
) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    run_dir = _run_dir(tmp_path, identity)
    bad_run_dir = tmp_path / "other" / identity.run_id
    with pytest.raises(StateSecurityError):
        store._validate_run_directory(bad_run_dir, tmp_path / "runs")

    run_dir.mkdir(parents=True)
    with pytest.raises(StateError):
        store._load_state(run_dir, identity.run_id)


def test_parse_state_validates_optional_contract_boundaries(tmp_path: Path) -> None:
    identity = _identity()
    good_payload = _state_payload(identity.run_id, identity)

    store = StateStore(data_root=tmp_path)

    store._active_identity = json.loads(identity.canonical_json)
    parsed = store._parse_state(good_payload, identity.run_id)
    assert parsed.run_id == identity.run_id

    missing = {"schema_version": STATE_SCHEMA_VERSION, "run_id": identity.run_id}
    with pytest.raises(StateError, match="missing required state fields"):
        store._parse_state(missing, identity.run_id)

    extra = dict(good_payload)
    extra["extra"] = True
    with pytest.raises(StateError, match="unexpected state fields"):
        store._parse_state(extra, identity.run_id)

    bad_schema = dict(good_payload)
    bad_schema["schema_version"] = 999
    with pytest.raises(StateError, match="unsupported state schema version"):
        store._parse_state(bad_schema, identity.run_id)

    bad_run_id = dict(good_payload)
    bad_run_id["run_id"] = "x" * 20
    with pytest.raises(StateError, match="state run_id must be"):
        store._parse_state(bad_run_id, identity.run_id)

    mismatched_run_id = dict(good_payload)
    with pytest.raises(StateIdentityMismatch, match="does not match path"):
        store._parse_state(mismatched_run_id, "1" * 20)

    bad_phase = dict(good_payload)
    bad_phase["phase"] = 2
    with pytest.raises(StateError, match="state phase must be a string"):
        store._parse_state(bad_phase, identity.run_id)

    bad_phase_value = dict(good_payload)
    bad_phase_value["phase"] = "bad"
    with pytest.raises(StateError, match="unsupported state phase"):
        store._parse_state(bad_phase_value, identity.run_id)

    bad_sequence = dict(good_payload)
    bad_sequence["sequence"] = -1
    with pytest.raises(
        StateError, match="state sequence must be a non-negative integer"
    ):
        store._parse_state(bad_sequence, identity.run_id)

    no_identity = dict(good_payload)
    no_identity["run_identity"] = []
    with pytest.raises(StateError, match="state identity must be a JSON object"):
        store._parse_state(no_identity, identity.run_id)

    null_facts = dict(good_payload)
    null_facts["facts"] = None
    with pytest.raises(StateError, match="state facts may not be null"):
        store._parse_state(null_facts, identity.run_id)


def test_private_accessors_and_load_state_identity_binding(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    state = store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)

    assert store._state_path(identity.run_id).name == "state.json"
    assert store._event_path(identity.run_id).name == "events.jsonl"
    assert store._lock_path(identity.run_id).name == ".state.lock"

    store._active_identity = None
    loaded = store._load_state(run_dir, identity.run_id)
    assert loaded == state
    assert loaded.run_id == identity.run_id


def test_write_state_handles_existing_tmp_and_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    identity = _identity()
    store = StateStore(data_root=tmp_path)
    state = store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)

    tmp = run_dir / ".state.json.tmp"
    tmp.mkdir()
    with pytest.raises(StateError):
        store._write_state(run_dir, state)

    tmp.rmdir()

    def bad_replace(_src: object, _dst: object) -> None:
        raise OSError("blocked")

    monkeypatch.setattr(state_module.os, "replace", bad_replace)
    with pytest.raises(StateError):
        store._write_state(run_dir, state)


def test_append_event_internal_error_path_and_newline_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)

    events = run_dir / "events.jsonl"
    events.write_text(
        '{"event_sequence":0, "schema_version":1, "timestamp":"2026-07-28T00:00:00+00:00", "level":"info", "phase":"created", "message":"first", "facts":{}}\n',
        encoding="utf-8",
    )
    os.chmod(events, 0o600)

    payload = {
        "schema_version": 1,
        "event_sequence": 1,
        "timestamp": "2026-07-28T00:00:00+00:00",
        "level": "info",
        "phase": "created",
        "message": "second",
        "facts": {},
    }
    store._append_event(run_dir, payload)

    def failing_fsync(_fd: int) -> None:
        raise OSError("blocked")

    monkeypatch.setattr(state_module.os, "fsync", failing_fsync)
    with pytest.raises(StateError):
        store._append_event(run_dir, payload)


def test_run_lock_handles_flock_unlock_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)
    lock_ctx = state_module._RunLockContext(store, run_dir)
    lock_ctx.__enter__()

    orig_flock = state_module.fcntl.flock

    def failing_flock(fd: int, operation: int) -> None:
        if operation == state_module.fcntl.LOCK_UN:
            raise OSError("unlock failure")
        return orig_flock(fd, operation)

    try:
        monkeypatch.setattr(state_module.fcntl, "flock", failing_flock)
        with pytest.raises(OSError, match="unlock failure"):
            lock_ctx.__exit__(None, None, None)
    finally:
        monkeypatch.setattr(state_module.fcntl, "flock", orig_flock)


def test_ensure_no_symlink_ancestors_handles_relative_and_ancestor_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    _ensure_no_symlink_ancestors(Path("root/run-id"))

    alias = tmp_path / "alias-root"
    alias.symlink_to(root)
    with pytest.raises(StateSecurityError):
        _ensure_no_symlink_ancestors(alias / "run-id")


def test_validate_directory_mode_labelless_path_mismatch(tmp_path: Path) -> None:
    mode_dir = tmp_path / "mode"
    mode_dir.mkdir()
    os.chmod(mode_dir, 0o700)
    with pytest.raises(
        StateSecurityError, match="filesystem entry must have expected permissions"
    ):
        _validate_directory(mode_dir, mode=0o600)


def test_read_file_bytes_rejects_missing_and_dangling_symlink(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(StateError, match="durable entry is missing"):
        _read_file_bytes(missing)

    link = tmp_path / "dangling.json"
    link.symlink_to(tmp_path / "nowhere")
    with pytest.raises(StateError, match="filesystem entry must not be a symlink"):
        _read_file_bytes(link)


def test_remove_temporary_if_regular_cleans_only_regular_files(tmp_path: Path) -> None:
    regular = tmp_path / "regular"
    regular.write_text("keep", encoding="utf-8")
    _remove_temporary_if_regular(regular)
    assert not regular.exists()

    target = tmp_path / "target"
    target.write_text("keep", encoding="utf-8")
    link = tmp_path / "target-link"
    link.symlink_to(target)
    _remove_temporary_if_regular(link)
    assert target.read_text(encoding="utf-8") == "keep"


def test_read_event_lines_rejects_missing_or_invalid_records(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"

    payload = {
        "schema_version": 1,
        "event_sequence": 0,
        "timestamp": "2026-07-28T00:00:00+00:00",
        "level": "info",
        "phase": "created",
        "message": "ok",
        "facts": {},
    }
    missing = dict(payload)
    missing.pop("schema_version")
    path.write_text(f"{json.dumps(missing)}\n", encoding="utf-8")
    os.chmod(path, 0o600)
    with pytest.raises(StateError, match="missing required fields"):
        _read_event_lines(path)

    extra = dict(payload)
    extra["unexpected"] = True
    path.write_text(f"{json.dumps(extra)}\n", encoding="utf-8")
    with pytest.raises(StateError, match="unexpected fields"):
        _read_event_lines(path)

    payload["schema_version"] = True
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    with pytest.raises(StateError, match="event schema version unsupported"):
        _read_event_lines(path)

    payload["schema_version"] = 1
    payload["event_sequence"] = True
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    with pytest.raises(
        StateError, match="event sequence must be a non-negative integer"
    ):
        _read_event_lines(path)

    payload["event_sequence"] = -1
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    with pytest.raises(
        StateError, match="event sequence must be a non-negative integer"
    ):
        _read_event_lines(path)

    payload["event_sequence"] = 1
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    with pytest.raises(StateError, match="event sequence must match line order"):
        _read_event_lines(path)

    payload["event_sequence"] = 0
    payload["timestamp"] = "2026-07-28T00:00:00"
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    with pytest.raises(StateError, match="must be an UTC timestamp"):
        _read_event_lines(path)

    payload["timestamp"] = "2026-07-28T00:00:00+01:00"
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    with pytest.raises(StateError, match="must be an UTC timestamp"):
        _read_event_lines(path)

    payload["timestamp"] = "2026-07-28T00:00:00+00:00"
    payload["phase"] = "bad"
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    with pytest.raises(StateError, match="event phase unsupported"):
        _read_event_lines(path)

    payload["phase"] = "created"
    payload["facts"] = {"nested": {"raw_response": "reject"}}
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    with pytest.raises(StateError, match="unsafe factual key rejected"):
        _read_event_lines(path)


def test_read_event_lines_rejects_zero_based_sequence_violations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"event_sequence":0, "schema_version":1, "timestamp":"2026-07-28T00:00:00+00:00", "level":"info", "phase":"created", "message":"first", "facts":{}}\n'
        '{"event_sequence":2, "schema_version":1, "timestamp":"2026-07-28T00:01:00+00:00", "level":"info", "phase":"created", "message":"second", "facts":{}}\n',
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    with pytest.raises(StateError, match="event sequence must match line order"):
        _read_event_lines(path)


def test_parse_utc_timestamp_rejects_invalid_and_non_utc_values() -> None:
    with pytest.raises(StateError, match="UTC"):
        _parse_utc_timestamp("2026-07-28T00:00:00")
    with pytest.raises(StateError, match="UTC"):
        _parse_utc_timestamp("2026-07-28T00:00:00+01:00")


def test_dangling_entry_checks_for_state_event_and_tmp_files(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)

    state_json = run_dir / "state.json"
    state_json.unlink()
    state_json.symlink_to(tmp_path / "missing-state")
    with pytest.raises(StateSecurityError):
        store.load_or_create(identity)

    state_json.unlink()
    _write_state_file(state_json, _state_payload(identity.run_id, identity))
    os.chmod(state_json, 0o600)
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)

    events = run_dir / "events.jsonl"
    if events.exists():
        events.unlink()
    events.symlink_to(tmp_path / "missing-events")
    with pytest.raises(StateSecurityError):
        store.append_event(level="info", message="fail")

    state_tmp = run_dir / ".state.json.tmp"
    if state_tmp.exists():
        state_tmp.unlink()
    state_tmp.symlink_to(tmp_path / "missing-state-tmp")
    with pytest.raises(StateSecurityError):
        store.transition(expected=RunPhase.CREATED, target=RunPhase.INPUTS_RESOLVED)

    event_tmp = run_dir / ".events.jsonl.tmp"
    if event_tmp.exists():
        event_tmp.unlink()
    event_tmp.symlink_to(tmp_path / "missing-events-tmp")
    with pytest.raises(StateSecurityError):
        store.append_event(level="info", message="fail")


def test_mkdir_with_mode_rejects_when_permissions_cannot_be_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    original_stat = state_module.os.stat

    def fake_stat(path: object) -> object:
        result = original_stat(path)
        return type(
            "_Stat",
            (),
            {
                "st_mode": (result.st_mode & ~0o777) | 0o644,
                "st_uid": result.st_uid,
            },
        )()

    monkeypatch.setattr(state_module.os, "stat", fake_stat)
    with pytest.raises(StateSecurityError, match="expected permissions"):
        state_module._mkdir_with_mode(tmp_path / "bad-mode", RUN_MODE)


def test_read_file_bytes_rejects_non_regular_entry_and_closes_descriptor(
    tmp_path: Path,
) -> None:
    target = tmp_path / "directory"
    target.mkdir()

    with pytest.raises(StateError, match="must be a regular file"):
        _read_file_bytes(target)


def test_read_event_lines_rejects_non_regular_file_and_closes_descriptor(
    tmp_path: Path,
) -> None:
    target = tmp_path / "events-directory"
    target.mkdir()

    with pytest.raises(StateError, match="must be a regular file"):
        _read_event_lines(target)


def test_read_event_lines_empty_file_returns_no_events(tmp_path: Path) -> None:
    path = tmp_path / "events-empty.jsonl"
    path.write_text("", encoding="utf-8")
    os.chmod(path, 0o600)
    assert _read_event_lines(path) == []


def test_read_event_lines_rejects_non_string_phase(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    payload = {
        "schema_version": 1,
        "event_sequence": 0,
        "timestamp": "2026-07-28T00:00:00+00:00",
        "level": "info",
        "phase": 2,
        "message": "ok",
        "facts": {},
    }
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    os.chmod(path, 0o600)

    with pytest.raises(StateError, match="event phase must be a string"):
        _read_event_lines(path)


def test_validate_run_directory_rejects_missing_internal_runs_root(
    tmp_path: Path,
) -> None:
    store = StateStore(data_root=tmp_path)
    run_dir = tmp_path / "runs" / ("a" * 20)
    with pytest.raises(StateError, match="internal runs root missing"):
        store._validate_run_directory(run_dir)


def test_validate_run_directory_rejects_lexical_parent_escape(tmp_path: Path) -> None:
    store = StateStore(data_root=tmp_path)
    run_dir = tmp_path / "other" / ("a" * 20)
    store._runs_root = tmp_path / "runs"
    (tmp_path / "other").mkdir()
    os.chmod((tmp_path / "other"), 0o700)
    run_dir.mkdir()
    os.chmod(run_dir, 0o700)
    with pytest.raises(StateSecurityError, match="run directory escapes runs root"):
        store._validate_run_directory(run_dir)


def test_validate_run_directory_rejects_physical_parent_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    store = StateStore(data_root=tmp_path)
    runs_root = tmp_path / "runs"
    runs_root.mkdir(mode=RUN_MODE)
    store._runs_root = runs_root
    run_dir = runs_root / ("b" * 20)
    run_dir.mkdir()
    os.chmod(run_dir, RUN_MODE)

    calls = {"count": 0}

    def fake_realpath(path: str) -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            return "/var/real-runs"
        return "/var/alternate-runs"

    monkeypatch.setattr(state_module.os.path, "realpath", fake_realpath)
    with pytest.raises(StateSecurityError, match="run directory escapes runs root"):
        store._validate_run_directory(run_dir)
    assert calls["count"] == 2


def test_append_event_without_active_run_is_rejected() -> None:
    with pytest.raises(StateError, match="no active run"):
        StateStore(data_root=Path("/tmp")).append_event(level="info", message="x")


def test_load_or_create_detects_state_identity_after_parsing(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    original = store._load_state
    modified = json.loads(identity.canonical_json)
    modified["region"] = "iran-latest"

    def mismatched_state(_run_dir: Path, _run_id: str) -> RunState:
        return RunState(
            schema_version=STATE_SCHEMA_VERSION,
            run_id=identity.run_id,
            run_identity=modified,
            phase=RunPhase.CREATED,
            sequence=0,
            timestamp="2026-07-28T00:00:00+00:00",
            facts={},
        )

    store._load_state = mismatched_state
    try:
        with pytest.raises(StateIdentityMismatch, match="state identity mismatch"):
            store.load_or_create(identity)
    finally:
        store._load_state = original


def test_load_or_create_rejects_divergent_loaded_run_id(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    def mismatched_state(_run_dir: Path, _run_id: str) -> Any:
        return type(
            "RunState",
            (),
            {
                "run_identity": json.loads(identity.canonical_json),
                "run_id": "0" * 20,
                "phase": RunPhase.CREATED,
                "sequence": 0,
                "timestamp": "2026-07-28T00:00:00+00:00",
                "facts": {},
            },
        )()

    store._load_state = mismatched_state
    with pytest.raises(StateIdentityMismatch, match="state run id mismatch"):
        store.load_or_create(identity)


def test_load_rejects_state_with_mismatched_run_id_after_parse(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    original = store._load_state

    def mismatched(_run_dir: Path, run_id: str) -> RunState:
        loaded = original(_run_dir, run_id)
        return RunState(
            schema_version=loaded.schema_version,
            run_id="0" * 20,
            run_identity=loaded.run_identity,
            phase=loaded.phase,
            sequence=loaded.sequence,
            timestamp=loaded.timestamp,
            facts=loaded.facts,
        )

    store._load_state = mismatched
    with pytest.raises(StateIdentityMismatch, match="state run id mismatch"):
        store.load()


def test_load_or_create_loads_events_with_invalid_level_or_message(
    tmp_path: Path,
) -> None:
    path = _run_dir(tmp_path, _identity()) / EVENTS_FILENAME
    payload = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_sequence": 0,
        "timestamp": "2026-07-28T00:00:00+00:00",
        "level": " ",
        "phase": RunPhase.CREATED.value,
        "message": "ok",
        "facts": {},
    }
    path.parent.mkdir(parents=True)
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    os.chmod(path, 0o600)
    with pytest.raises(StateError, match="event level must be a non-empty string"):
        _read_event_lines(path)

    payload["level"] = "info"
    payload["message"] = ""
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    with pytest.raises(StateError, match="event message must be a non-empty string"):
        _read_event_lines(path)


def test_read_event_lines_rejects_null_or_non_mapping_facts(tmp_path: Path) -> None:
    path = _run_dir(tmp_path, _identity()) / EVENTS_FILENAME
    path.parent.mkdir(parents=True)

    payload = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_sequence": 0,
        "timestamp": "2026-07-28T00:00:00+00:00",
        "level": "info",
        "phase": RunPhase.CREATED.value,
        "message": "ok",
        "facts": None,
    }
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    os.chmod(path, 0o600)
    with pytest.raises(StateError, match="event facts may not be null"):
        _read_event_lines(path)

    payload["facts"] = []
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    with pytest.raises(StateError, match="event facts must be a mapping"):
        _read_event_lines(path)


def test_write_state_validates_and_cleans_tmp_file_on_open_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    identity = _identity()
    store = StateStore(data_root=tmp_path)
    state = store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)
    run_state = RunState(
        schema_version=STATE_SCHEMA_VERSION,
        run_id=identity.run_id,
        run_identity=state.run_identity,
        phase=RunPhase.INPUTS_RESOLVED,
        sequence=1,
        timestamp="2026-07-28T01:00:00+00:00",
        facts={},
    )
    prior = (run_dir / STATE_FILENAME).read_bytes()

    def fail_regular_file(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected")

    monkeypatch.setattr(state_module, "_validate_regular_file", fail_regular_file)
    with pytest.raises(StateError, match="failed to write state"):
        store._write_state(run_dir, run_state)
    assert (run_dir / STATE_FILENAME).read_bytes() == prior


def test_append_event_rejects_existing_tmp_files(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)
    (run_dir / ".events.jsonl.tmp").write_text("leftover", encoding="utf-8")
    os.chmod(run_dir / ".events.jsonl.tmp", 0o600)

    with pytest.raises(StateError, match="temporary event file already exists"):
        store.append_event(level="info", message="x")


def test_append_event_tmp_path_existing_directory_is_rejected(tmp_path: Path) -> None:
    identity = _identity()
    store = StateStore(data_root=tmp_path)
    run_dir = _run_dir(tmp_path, identity)
    store.load_or_create(identity)
    event_tmp = run_dir / ".events.jsonl.tmp"
    event_tmp.mkdir()
    with pytest.raises(StateError, match="invalid temporary event file path"):
        store.append_event(level="info", message="x")


def test_append_event_rejects_tmp_fdopen_failure_and_preserves_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    identity = _identity()
    run_dir = _run_dir(tmp_path, identity)
    store = StateStore(data_root=tmp_path)
    stored = store.load_or_create(identity)
    event_path = run_dir / EVENTS_FILENAME
    prior = event_path.read_bytes() if event_path.exists() else b""
    original_load_state = store._load_state
    store._load_state = lambda *_args, **_kwargs: stored

    original_fdopen = state_module.os.fdopen

    def failing_fdopen(*args: object, **kwargs: object) -> Any:
        raise OSError("blocked fdopen")

    monkeypatch.setattr(state_module.os, "fdopen", failing_fdopen)
    with pytest.raises(StateError, match="failed to append event"):
        store.append_event(level="info", message="x")
    monkeypatch.setattr(state_module.os, "fdopen", original_fdopen)
    store._load_state = original_load_state

    if prior:
        assert event_path.read_bytes() == prior
    else:
        assert not event_path.exists()


def test_remove_temporary_if_regular_retries_cleanup_race(tmp_path: Path) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    temporary = tmp_path / "temporary"
    temporary.write_text("keep", encoding="utf-8")
    os.chmod(temporary, 0o600)
    original_unlink = state_module.Path.unlink

    calls = {"count": 0}

    def fake_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self == temporary and calls["count"] == 0:
            calls["count"] += 1
            raise FileNotFoundError
        return original_unlink(self, *args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(state_module.Path, "unlink", fake_unlink)
    state_module._remove_temporary_if_regular(temporary)
    assert calls["count"] == 1
    monkeypatch.setattr(state_module.Path, "unlink", original_unlink)


def test_run_lock_exit_raises_single_unlock_failure_without_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)

    lock_ctx = state_module._RunLockContext(store, run_dir)
    lock_ctx.__enter__()
    original_flock = state_module.fcntl.flock

    def failing_flock(fd: int, operation: int) -> None:
        if operation == state_module.fcntl.LOCK_UN:
            raise OSError("unlock failed")
        return original_flock(fd, operation)

    monkeypatch.setattr(state_module.fcntl, "flock", failing_flock)
    with pytest.raises(OSError, match="unlock failed"):
        lock_ctx.__exit__(None, None, None)
    monkeypatch.setattr(state_module.fcntl, "flock", original_flock)


def test_validate_lock_parent_rejects_missing_internal_runs_root(
    tmp_path: Path,
) -> None:
    store = StateStore(data_root=tmp_path)
    with pytest.raises(StateError, match="internal runs root missing"):
        store._validate_lock_parent(tmp_path / "runs" / ("a" * 20))


def test_validate_lock_parent_rejects_parent_escape(tmp_path: Path) -> None:
    store = StateStore(data_root=tmp_path)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    store._runs_root = runs_root

    with pytest.raises(StateSecurityError, match="run directory escapes runs root"):
        store._validate_lock_parent(runs_root / "other" / ("a" * 20))


def test_validate_lock_parent_rejects_physical_parent_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    store = StateStore(data_root=tmp_path)
    runs_root = tmp_path / "runs"
    runs_root.mkdir(mode=RUN_MODE)
    store._runs_root = runs_root
    run_dir = runs_root / ("a" * 20)
    run_dir.mkdir()
    os.chmod(run_dir, RUN_MODE)

    calls = {"count": 0}

    def fake_realpath(path: str) -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            return "/var/real-runs"
        return "/var/alternate-runs"

    monkeypatch.setattr(state_module.os.path, "realpath", fake_realpath)
    with pytest.raises(StateSecurityError, match="run directory escapes runs root"):
        store._validate_lock_parent(run_dir)
    assert calls["count"] == 2


def test_lock_context_rejects_invalid_lock_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)

    run_dir = _run_dir(tmp_path, identity)
    monkeypatch.setattr(state_module, "_open_secure_no_follow", lambda *args: None)
    with (
        pytest.raises(StateError, match="cannot open state lock"),
        store._run_lock(run_dir),
    ):
        pass


def test_append_event_partial_write_is_rejected_and_prefix_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)
    event_path = run_dir / "events.jsonl"
    event_path.write_text(
        '{"event_sequence":0, "schema_version":1, "timestamp":"2026-07-28T00:00:00+00:00", "level":"info", "phase":"created", "message":"first", "facts":{}}\n',
        encoding="utf-8",
    )
    os.chmod(event_path, 0o600)
    prior = event_path.read_bytes()
    tmp_path_obj = run_dir / ".events.jsonl.tmp"

    def short_write(_fd: int, payload: bytes) -> int:
        return 0

    monkeypatch.setattr(state_module.os, "write", short_write)
    with pytest.raises(StateError, match="partial file write failed"):
        store.append_event(level="info", message="second")

    assert event_path.read_bytes() == prior
    assert not tmp_path_obj.exists()


def test_append_event_replace_and_parent_fsync_failures_preserve_tmp_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)
    event_path = run_dir / "events.jsonl"
    event_path.write_text(
        '{"event_sequence":0, "schema_version":1, "timestamp":"2026-07-28T00:00:00+00:00", "level":"info", "phase":"created", "message":"first", "facts":{}}\n',
        encoding="utf-8",
    )
    os.chmod(event_path, 0o600)
    tmp_path_obj = run_dir / ".events.jsonl.tmp"

    def bad_replace(_src: object, _dst: object) -> None:
        raise OSError("blocked")

    original_replace = state_module.os.replace
    monkeypatch.setattr(state_module.os, "replace", bad_replace)
    with pytest.raises(StateError, match="failed to append event"):
        store.append_event(level="info", message="second")
    assert not tmp_path_obj.exists()
    monkeypatch.setattr(state_module.os, "replace", original_replace)

    def bad_fsync_parent(_path: Path) -> None:
        raise OSError("blocked")

    monkeypatch.setattr(state_module, "_fsync_parent", bad_fsync_parent)
    with pytest.raises(StateError, match="failed to append event"):
        store.append_event(level="info", message="third")
    assert not tmp_path_obj.exists()


def test_lock_release_preserves_primary_failure_and_surfaces_unlock_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)

    orig_flock = state_module.fcntl.flock

    def failing_flock(fd: int, operation: int) -> None:
        if operation == state_module.fcntl.LOCK_UN:
            raise OSError("unlock failed")
        return orig_flock(fd, operation)

    monkeypatch.setattr(state_module.fcntl, "flock", failing_flock)
    with (
        pytest.raises(ValueError, match="primary failure"),
        state_module._RunLockContext(store, run_dir),
    ):
        raise ValueError("primary failure")

    monkeypatch.setattr(state_module.fcntl, "flock", orig_flock)


def test_lock_release_multiple_failures_raise_exception_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osm_polygon_sentence_relevance.operator.state as state_module

    identity = _identity()
    store = StateStore(data_root=tmp_path)
    store.load_or_create(identity)
    run_dir = _run_dir(tmp_path, identity)

    orig_flock = state_module.fcntl.flock
    orig_close = state_module.os.close

    def failing_flock(_fd: int, operation: int) -> None:
        if operation == state_module.fcntl.LOCK_UN:
            raise OSError("unlock failed")
        return orig_flock(_fd, operation)

    def failing_close(_fd: int) -> None:
        raise OSError("close failed")

    lock_ctx = state_module._RunLockContext(store, run_dir)
    lock_ctx.__enter__()

    monkeypatch.setattr(state_module.fcntl, "flock", failing_flock)
    monkeypatch.setattr(state_module.os, "close", failing_close)
    with pytest.raises(ExceptionGroup):
        lock_ctx.__exit__(None, None, None)

    monkeypatch.setattr(state_module.fcntl, "flock", orig_flock)
    monkeypatch.setattr(state_module.os, "close", orig_close)
