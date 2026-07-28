"""Durable local operator state and append-only event logs."""

from __future__ import annotations

import contextlib
import copy
import fcntl
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from json import JSONDecodeError
from pathlib import Path
from typing import Final, cast

from osm_polygon_sentence_relevance.operator.config import DATA_ROOT, RunIdentity

STATE_FILENAME: Final[str] = "state.json"
EVENTS_FILENAME: Final[str] = "events.jsonl"
LOCK_FILENAME: Final[str] = ".state.lock"
STATE_TMP_FILENAME: Final[str] = ".state.json.tmp"
EVENT_TMP_FILENAME: Final[str] = ".events.jsonl.tmp"
STATE_SCHEMA_VERSION: Final[int] = 1
EVENT_SCHEMA_VERSION: Final[int] = 1
RUN_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{20}$")
STATE_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "run_id",
        "run_identity",
        "phase",
        "sequence",
        "timestamp",
        "facts",
    }
)
EVENT_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "event_sequence",
        "timestamp",
        "level",
        "phase",
        "message",
        "facts",
    }
)
UNSAFE_FACT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "token",
        "access_token",
        "authorization",
        "password",
        "secret",
        "prompt",
        "response",
        "raw_response",
    }
)

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

RUN_MODE: Final[int] = 0o700
FILE_MODE: Final[int] = 0o600
NOFOLLOW: Final[int] = getattr(os, "O_NOFOLLOW", 0)


class RunPhase(StrEnum):
    """Persistent state for each production phase."""

    CREATED = "created"
    INPUTS_RESOLVED = "inputs_resolved"
    SITE_SELECTED = "site_selected"
    STORAGE_READY = "storage_ready"
    REMOTE_PREPARED = "remote_prepared"
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    CHECKPOINTED = "checkpointed"
    FINALIZING = "finalizing"
    VALIDATED = "validated"
    PUBLISHING = "publishing"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StateError(RuntimeError):
    """Base class for all operator state storage errors."""


class StateIdentityMismatch(StateError):
    """Existing durable state does not match the provided run identity."""


class StateTransitionError(StateError):
    """State transition failed due to unexpected current phase."""


class StateSecurityError(StateError):
    """Filesystem safety or ownership policy was violated."""


@dataclass(frozen=True, slots=True)
class RunState:
    """Serializable state payload for one run."""

    schema_version: int
    run_id: str
    run_identity: dict[str, JSONValue]
    phase: RunPhase
    sequence: int
    timestamp: str
    facts: dict[str, JSONValue]


def _clock_now() -> str:
    return datetime.now(UTC).isoformat()


def _is_float_finite(value: float) -> bool:
    return math.isfinite(value)


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _is_regular_file(stat_result: os.stat_result) -> bool:
    return stat.S_ISREG(stat_result.st_mode)


def _validate_json_value(value: object) -> JSONValue:
    if type(value) is str:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not _is_float_finite(value):
            raise StateError("facts must contain finite JSON values")
        return value
    if type(value) is bool:
        return value
    if value is None:
        return value
    if isinstance(value, list):
        return [_validate_json_value(item) for item in value]
    if isinstance(value, dict):
        sanitized: dict[str, JSONValue] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise StateError("facts must contain string keys")
            if key.casefold() in UNSAFE_FACT_KEYS:
                raise StateSecurityError(f"unsafe factual key rejected: {key}")
            sanitized[key] = _validate_json_value(child)
        return sanitized
    raise StateError("facts must be JSON-safe scalar/list/mapping values")


def _coerce_facts(facts: Mapping[str, object] | None) -> dict[str, JSONValue]:
    if facts is None:
        return {}
    if not isinstance(facts, Mapping):
        raise StateError("facts must be a mapping")
    return _validate_json_value(dict(facts))  # type: ignore[return-value]


def _parse_utc_timestamp(value: object, label: str = "timestamp") -> str:
    if not isinstance(value, str):
        raise StateError(f"{label} must be a string")

    normalized = value
    if value.endswith("Z"):
        normalized = value[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:  # pragma: no cover - defensive formatting branch
        raise StateError(f"{label} must be an ISO-8601 UTC value") from exc

    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise StateError(f"{label} must be an UTC timestamp")

    return value


def _open_secure_no_follow(path: Path, flags: int, mode: int | None = None) -> int:
    if path.is_symlink():
        raise StateSecurityError("filesystem entry must not be a symlink")
    if mode is None:
        mode = FILE_MODE

    nofollow_flags = flags | NOFOLLOW
    try:
        return os.open(path, nofollow_flags, mode)
    except AttributeError:
        return os.open(path, flags, mode)


def _ensure_no_symlink_ancestors(path: Path) -> None:
    current = path
    if not current.is_absolute():
        current = Path.cwd() / current

    for ancestor in (current, *current.parents):
        if ancestor.is_symlink():
            raise StateSecurityError("filesystem path may not include symlinks")


def _validate_owner(stat_result: os.stat_result) -> None:
    if hasattr(os, "getuid") and stat_result.st_uid != os.getuid():
        raise StateSecurityError("filesystem owner check failed")


def _fsync_fd(file_descriptor: int) -> None:
    os.fsync(file_descriptor)


def _fsync_parent(path: Path) -> None:
    parent_fd = _open_secure_no_follow(path.parent, os.O_RDONLY)
    try:
        _fsync_fd(parent_fd)
    finally:
        os.close(parent_fd)


def _validate_regular_file(path: Path, fd: int, *, mode: int | None = None) -> None:
    stat_result = os.fstat(fd)
    if not _is_regular_file(stat_result):
        raise StateSecurityError("filesystem entry must be a regular file")
    if mode is not None and (stat_result.st_mode & 0o777) != mode:
        raise StateSecurityError("filesystem entry must have strict mode")
    _validate_owner(stat_result)


def _validate_directory(
    path: Path,
    *,
    mode: int | None = None,
    label: str | None = None,
) -> None:
    if path.is_symlink():
        raise StateSecurityError("filesystem entry must not be a symlink")
    if not _path_exists(path) or not path.is_dir():
        raise StateSecurityError("filesystem entry must be a directory")
    stat_result = os.stat(path)
    if mode is not None and (stat_result.st_mode & 0o777) != mode:
        if label is None:
            raise StateSecurityError("filesystem entry must have expected permissions")
        raise StateSecurityError(
            f"{label} must have mode {oct(mode)}; found {oct(stat_result.st_mode & 0o777)}"
        )
    _validate_owner(stat_result)


def _mkdir_with_mode(path: Path, mode: int) -> None:
    path.mkdir(mode=mode, parents=True, exist_ok=False)
    os.chmod(path, mode)
    stat_result = os.stat(path)
    if (stat_result.st_mode & 0o777) != mode:
        raise StateSecurityError("filesystem entry must have expected permissions")
    _validate_owner(stat_result)


def _parse_json_timestamp(value: object) -> str:
    """Backward-compatible helper used by existing callers."""
    return _parse_utc_timestamp(value, label="state timestamp")


def _read_file_bytes(path: Path, *, mode: int | None = None) -> bytes:
    if not _path_exists(path):
        raise StateError("durable entry is missing")
    if path.is_symlink():
        raise StateSecurityError("filesystem entry must not be a symlink")

    fd = _open_secure_no_follow(path, os.O_RDONLY)
    try:
        _validate_regular_file(path, fd, mode=mode)
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            return handle.read()
    finally:
        if fd != -1:
            os.close(fd)


def _read_text_file(path: Path) -> dict[str, object]:
    raw = _read_file_bytes(path, mode=FILE_MODE)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except JSONDecodeError as exc:
        raise StateError("invalid JSON payload") from exc
    if not isinstance(payload, dict):
        raise StateError("payload must be a JSON object")
    return payload


def _read_event_lines(path: Path) -> list[dict[str, object]]:
    if not _path_exists(path):
        return []

    if path.is_symlink():
        raise StateSecurityError("filesystem entry must not be a symlink")

    fd = _open_secure_no_follow(path, os.O_RDONLY)
    try:
        _validate_regular_file(path, fd, mode=FILE_MODE)
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            raw_text = handle.read()
    finally:
        if fd != -1:
            os.close(fd)

    if not raw_text:
        return []
    if not raw_text.endswith("\n"):
        raise StateError("malformed event log: missing trailing newline")

    events: list[dict[str, object]] = []
    for index, line in enumerate(raw_text.splitlines()):
        if not line.strip():
            raise StateError(f"malformed event log line {index + 1}")

        try:
            parsed = json.loads(line)
        except JSONDecodeError as exc:
            raise StateError("malformed event log") from exc

        if not isinstance(parsed, dict):
            raise StateError("malformed event log entry")

        if set(parsed) != EVENT_REQUIRED_KEYS:
            missing = EVENT_REQUIRED_KEYS.difference(parsed)
            unexpected = set(parsed).difference(EVENT_REQUIRED_KEYS)
            if missing:
                raise StateError(
                    f"event log entry missing required fields: {', '.join(sorted(missing))}"
                )
            raise StateError(
                f"event log entry contains unexpected fields: {', '.join(sorted(unexpected))}"
            )

        schema_version = parsed["schema_version"]
        if type(schema_version) is not int or schema_version != EVENT_SCHEMA_VERSION:
            raise StateError("event schema version unsupported")

        event_sequence = parsed["event_sequence"]
        if type(event_sequence) is not int or event_sequence < 0:
            raise StateError("event sequence must be a non-negative integer")
        if event_sequence != index:
            raise StateError("event sequence must match line order")

        timestamp = _parse_utc_timestamp(parsed["timestamp"], "event timestamp")

        level = parsed["level"]
        if not isinstance(level, str) or not level.strip():
            raise StateError("event level must be a non-empty string")

        message = parsed["message"]
        if not isinstance(message, str) or not message.strip():
            raise StateError("event message must be a non-empty string")

        phase_value = parsed["phase"]
        if not isinstance(phase_value, str):
            raise StateError("event phase must be a string")
        try:
            phase = RunPhase(phase_value)
        except ValueError as exc:
            raise StateError(f"event phase unsupported: {phase_value}") from exc

        facts = parsed["facts"]
        if facts is None:
            raise StateError("event facts may not be null")
        if not isinstance(facts, Mapping):
            raise StateError("event facts must be a mapping")
        facts = _validate_json_value(dict(facts))

        validated = {
            "schema_version": schema_version,
            "event_sequence": event_sequence,
            "timestamp": timestamp,
            "level": level,
            "phase": phase,
            "message": message,
            "facts": facts,
        }
        events.append(validated)

    return events


def _next_event_sequence(existing: list[dict[str, object]]) -> int:
    if not existing:
        return 0
    return cast(int, existing[-1]["event_sequence"]) + 1


def _write_full(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise StateError("partial file write failed")
        offset += written


def _remove_temporary_if_regular(path: Path) -> None:
    if not _path_exists(path):
        return
    if path.is_symlink():
        return
    if _is_regular_file(os.stat(path)):
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


class StateStore:
    """Durable state + append-only event log for a single run."""

    def __init__(
        self,
        data_root: Path | str = DATA_ROOT,
        clock: Callable[[], str] = _clock_now,
    ) -> None:
        self._data_root = Path(data_root)
        self._clock = clock
        self._run_id: str | None = None
        self._active_identity: dict[str, JSONValue] | None = None
        self._runs_root: Path | None = None

    def load_or_create(self, identity: RunIdentity) -> RunState:
        root = self._validate_root()
        runs_root = root / "runs"
        if _path_exists(runs_root):
            _validate_directory(runs_root, mode=RUN_MODE, label="runs directory mode")
        else:
            _mkdir_with_mode(runs_root, RUN_MODE)

        self._runs_root = runs_root
        run_id = self._validate_run_id(identity.run_id)

        run_dir = runs_root / run_id
        self._run_id = run_id
        self._active_identity = json.loads(identity.canonical_json)

        if _path_exists(run_dir):
            self._validate_run_directory(run_dir)
        else:
            _mkdir_with_mode(run_dir, RUN_MODE)
            self._validate_run_directory(run_dir)

        with self._run_lock(run_dir):
            state_path = self._state_path(run_dir)
            if _path_exists(state_path):
                state = self._load_state(run_dir, run_id)
                if state.run_identity != self._active_identity:
                    raise StateIdentityMismatch("state identity mismatch")
                if state.run_id != run_id:
                    raise StateIdentityMismatch("state run id mismatch")
                return state

            state = RunState(
                schema_version=STATE_SCHEMA_VERSION,
                run_id=identity.run_id,
                run_identity=copy.deepcopy(self._active_identity),
                phase=RunPhase.CREATED,
                sequence=0,
                timestamp=self._clock(),
                facts={},
            )
            self._write_state(run_dir, state)
            return state

    def transition(
        self,
        *,
        expected: RunPhase,
        target: RunPhase,
        facts: Mapping[str, object] | None = None,
    ) -> RunState:
        if self._run_id is None or self._active_identity is None:
            raise StateError("no active run; call load_or_create first")

        run_dir = self._run_dir(self._run_id)
        with self._run_lock(run_dir):
            state = self._load_state(run_dir, self._run_id)
            if state.phase != expected:
                raise StateTransitionError(
                    f"expected phase {expected.value}, found {state.phase.value}"
                )

            normalized_facts = _coerce_facts(facts)
            merged_facts = dict(state.facts)
            merged_facts.update(copy.deepcopy(normalized_facts))

            next_state = RunState(
                schema_version=STATE_SCHEMA_VERSION,
                run_id=state.run_id,
                run_identity=copy.deepcopy(state.run_identity),
                phase=target,
                sequence=state.sequence + 1,
                timestamp=self._clock(),
                facts=merged_facts,
            )
            self._write_state(run_dir, next_state)
            return next_state

    def append_event(
        self,
        *,
        level: str,
        message: str,
        facts: Mapping[str, object] | None = None,
    ) -> None:
        if not level.strip():
            raise StateError("event level must be a non-empty string")
        if not message.strip():
            raise StateError("event message must be a non-empty string")

        if self._run_id is None or self._active_identity is None:
            raise StateError("no active run; call load_or_create first")

        run_dir = self._run_dir(self._run_id)
        with self._run_lock(run_dir):
            state = self._load_state(run_dir, self._run_id)

            normalized_facts = _coerce_facts(facts)
            timestamp = self._clock()
            events = _read_event_lines(run_dir / EVENTS_FILENAME)
            event_sequence = _next_event_sequence(events)

            payload = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event_sequence": event_sequence,
                "timestamp": timestamp,
                "level": level,
                "phase": state.phase.value,
                "message": message,
                "facts": copy.deepcopy(normalized_facts),
            }
            self._append_event(run_dir, payload)

    def load(self) -> RunState:
        if self._run_id is None:
            raise StateError("no active run; call load_or_create first")
        run_dir = self._run_dir(self._run_id)
        with self._run_lock(run_dir):
            state = self._load_state(run_dir, self._run_id)
            if state.run_id != self._run_id:
                raise StateIdentityMismatch("state run id mismatch")
            return state

    def _run_dir(self, run_id: str) -> Path:
        return self._data_root / "runs" / run_id

    def _run_dir_path(self, run: Path | str) -> Path:
        if isinstance(run, str):
            return self._run_dir(run)
        return run

    def _state_path(self, run_dir: Path | str) -> Path:
        return self._run_dir_path(run_dir) / STATE_FILENAME

    def _event_path(self, run_dir: Path | str) -> Path:
        return self._run_dir_path(run_dir) / EVENTS_FILENAME

    def _event_tmp_path(self, run_dir: Path | str) -> Path:
        return self._run_dir_path(run_dir) / EVENT_TMP_FILENAME

    def _lock_path(self, run_dir: Path | str) -> Path:
        return self._run_dir_path(run_dir) / LOCK_FILENAME

    def _state_tmp_path(self, run_dir: Path | str) -> Path:
        return self._run_dir_path(run_dir) / STATE_TMP_FILENAME

    def _validate_root(self) -> Path:
        path = self._data_root
        if not _path_exists(path) or not path.is_dir() or path.is_symlink():
            raise StateSecurityError("data root must be an existing real directory")
        _ensure_no_symlink_ancestors(path)
        _validate_directory(path, mode=None)
        return path

    def _validate_run_directory(
        self,
        run_dir: Path,
        runs_root: Path | None = None,
    ) -> None:
        if runs_root is None:
            runs_root = self._runs_root
        if runs_root is None:
            raise StateError("internal runs root missing")

        _ensure_no_symlink_ancestors(run_dir)
        if not _path_exists(run_dir) or not run_dir.is_dir():
            raise StateSecurityError("run directory missing")

        _validate_directory(run_dir, mode=RUN_MODE, label="run directory mode")

        run_parent = Path(os.path.realpath(str(run_dir.parent)))
        expected_parent = Path(os.path.realpath(str(runs_root)))
        if run_parent != expected_parent:
            raise StateSecurityError("run directory escapes runs root")

    def _validate_run_id(self, value: str) -> str:
        if not isinstance(value, str) or not RUN_ID_PATTERN.fullmatch(value):
            raise StateError("run id must be 20 lowercase hexadecimal characters")
        return value

    def _validate_lock_parent(self, run_dir: Path) -> None:
        if self._runs_root is None:
            raise StateError("internal runs root missing")

        if run_dir.parent != self._runs_root:
            raise StateSecurityError("run directory escapes runs root")

        _ensure_no_symlink_ancestors(run_dir.parent)
        if not _path_exists(run_dir.parent) or not run_dir.parent.is_dir():
            raise StateSecurityError("runs directory missing")
        _validate_directory(run_dir.parent, mode=RUN_MODE, label="runs directory mode")

        parent_real = Path(os.path.realpath(str(run_dir.parent)))
        expected_real = Path(os.path.realpath(str(self._runs_root)))
        if parent_real != expected_real:
            raise StateSecurityError("run directory escapes runs root")

    def _run_lock(self, run_dir: Path) -> _RunLockContext:
        return _RunLockContext(self, run_dir)

    def _load_state(self, run_dir: Path, run_id: str) -> RunState:
        path = run_dir / STATE_FILENAME
        if not _path_exists(path):
            raise StateError("state file is missing")
        payload = _read_text_file(path)
        return self._parse_state(payload, run_id)

    def _parse_state(self, payload: Mapping[str, object], run_id: str) -> RunState:
        payload_keys = set(payload)
        if payload_keys != STATE_REQUIRED_KEYS:
            missing = STATE_REQUIRED_KEYS.difference(payload_keys)
            extra = payload_keys.difference(STATE_REQUIRED_KEYS)
            if missing:
                raise StateError(
                    f"missing required state fields: {', '.join(sorted(missing))}"
                )
            raise StateError(f"unexpected state fields: {', '.join(sorted(extra))}")

        schema_version = payload["schema_version"]
        if type(schema_version) is not int or schema_version != STATE_SCHEMA_VERSION:
            raise StateError("unsupported state schema version")

        payload_run_id = payload["run_id"]
        if not isinstance(payload_run_id, str) or not RUN_ID_PATTERN.fullmatch(
            payload_run_id
        ):
            raise StateError("state run_id must be 20 lowercase hexadecimal characters")
        if payload_run_id != run_id:
            raise StateIdentityMismatch("state run id does not match path")

        phase_value = payload["phase"]
        if not isinstance(phase_value, str):
            raise StateError("state phase must be a string")
        try:
            phase = RunPhase(phase_value)
        except ValueError as exc:
            raise StateError(f"unsupported state phase: {phase_value}") from exc

        sequence = payload["sequence"]
        if type(sequence) is not int or sequence < 0:
            raise StateError("state sequence must be a non-negative integer")

        timestamp = _parse_utc_timestamp(payload["timestamp"], "state timestamp")

        run_identity_obj = payload["run_identity"]
        if not isinstance(run_identity_obj, dict):
            raise StateError("state identity must be a JSON object")
        run_identity = _coerce_facts(cast(Mapping[str, object], run_identity_obj))

        expected_identity = self._active_identity
        if expected_identity is None:
            expected_identity = dict(run_identity)
            self._active_identity = expected_identity
        elif run_identity != expected_identity:
            raise StateIdentityMismatch("state identity mismatch")

        facts_raw = payload["facts"]
        if facts_raw is None:
            raise StateError("state facts may not be null")
        facts = _coerce_facts(cast(Mapping[str, object], facts_raw))

        return RunState(
            schema_version=schema_version,
            run_id=payload_run_id,
            run_identity=run_identity,
            phase=phase,
            sequence=sequence,
            timestamp=timestamp,
            facts=facts,
        )

    def _write_state(self, run_dir: Path, state: RunState) -> None:
        target = self._state_path(run_dir)
        tmp_path = self._state_tmp_path(run_dir)

        if _path_exists(tmp_path):
            if tmp_path.is_symlink():
                raise StateSecurityError("invalid temporary state file path")
            if not tmp_path.is_file() or tmp_path.is_symlink():
                raise StateSecurityError("invalid temporary state file path")
            raise StateSecurityError("temporary state file already exists")

        payload = {
            "schema_version": state.schema_version,
            "run_id": state.run_id,
            "run_identity": state.run_identity,
            "phase": state.phase.value,
            "sequence": state.sequence,
            "timestamp": state.timestamp,
            "facts": state.facts,
        }
        text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

        fd: int | None = None
        try:
            fd = _open_secure_no_follow(
                tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE
            )
            os.fchmod(fd, FILE_MODE)
            _validate_regular_file(tmp_path, fd, mode=FILE_MODE)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = None
                handle.write(text)
                handle.flush()
                _fsync_fd(handle.fileno())

            os.replace(tmp_path, target)
            _fsync_parent(target)
        except OSError as exc:
            raise StateError("failed to write state") from exc
        finally:
            if fd is not None:
                os.close(fd)
            _remove_temporary_if_regular(tmp_path)

    def _append_event(self, run_dir: Path, payload: dict[str, object]) -> None:
        path = self._event_path(run_dir)
        tmp_path = self._event_tmp_path(run_dir)

        if _path_exists(path):
            _ = _read_event_lines(path)

        if _path_exists(tmp_path):
            if tmp_path.is_symlink():
                raise StateSecurityError("invalid temporary event file path")
            if not tmp_path.is_file() or tmp_path.is_symlink():
                raise StateSecurityError("invalid temporary event file path")
            raise StateSecurityError("temporary event file already exists")

        previous_bytes = b""
        if _path_exists(path):
            previous_bytes = _read_file_bytes(path, mode=FILE_MODE)

        line = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        next_bytes = previous_bytes + line.encode("utf-8") + b"\n"

        fd: int | None = None
        try:
            fd = _open_secure_no_follow(
                tmp_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                FILE_MODE,
            )
            os.fchmod(fd, FILE_MODE)
            with os.fdopen(fd, "wb") as handle:
                fd = None
                _write_full(handle.fileno(), next_bytes)
                handle.flush()
                _fsync_fd(handle.fileno())

            os.replace(tmp_path, path)
            _fsync_parent(path)
        except OSError as exc:
            raise StateError("failed to append event") from exc
        finally:
            if fd is not None:
                os.close(fd)
            _remove_temporary_if_regular(tmp_path)


class _RunLockContext:
    """Per-run advisory exclusive lock."""

    def __init__(self, store: StateStore, run_dir: Path) -> None:
        self._store = store
        self._run_dir = run_dir
        self._fd: int | None = None

    def __enter__(self) -> _RunLockContext:
        self._store._validate_root()
        self._store._validate_lock_parent(self._run_dir)
        self._store._validate_run_directory(self._run_dir)

        lock_path = self._store._lock_path(self._run_dir)
        while True:
            try:
                self._fd = _open_secure_no_follow(
                    lock_path,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL,
                    FILE_MODE,
                )
                break
            except FileExistsError:
                self._fd = _open_secure_no_follow(lock_path, os.O_RDWR, FILE_MODE)
                break

        if self._fd is None:
            raise StateSecurityError("cannot open state lock")

        try:
            _validate_regular_file(lock_path, self._fd, mode=FILE_MODE)
            os.fchmod(self._fd, FILE_MODE)
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            return self
        except Exception:
            os.close(self._fd)
            self._fd = None
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        failures: list[Exception] = []
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except Exception as error:
                failures.append(error)

            try:
                os.close(self._fd)
            except Exception as error:
                failures.append(error)
            finally:
                self._fd = None

        if not failures:
            return None

        if exc is None:
            if len(failures) == 1:
                raise failures[0]
            raise ExceptionGroup("state lock release failed", failures)

        for failure in failures:
            exc.add_note(str(failure))
        return None


__all__ = [
    "STATE_SCHEMA_VERSION",
    "RunPhase",
    "RunState",
    "StateStore",
    "StateError",
    "StateIdentityMismatch",
    "StateTransitionError",
    "StateSecurityError",
]
