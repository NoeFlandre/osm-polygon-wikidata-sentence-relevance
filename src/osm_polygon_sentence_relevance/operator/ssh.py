"""OpenSSH transport boundary for command execution and resumable log reads."""

from __future__ import annotations

import base64
import math
import re
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from subprocess import TimeoutExpired
from typing import Final, Protocol


class _Runner(Protocol):
    def __call__(
        self,
        args: Sequence[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class SshResult:
    """Result of a successful ssh attempt after configured retries."""

    stdout: str
    stderr: str
    returncode: int
    attempts: int


@dataclass(frozen=True, slots=True)
class LogChunk:
    """A chunk read from a remote log.

    text: decoded UTF-8 content from the requested byte window.
    next_offset: byte offset for the next request.
    eof: whether the current end of file was reached.
    reset: ``True`` when the remote log appears to have truncated.
    """

    text: str
    next_offset: int
    eof: bool
    reset: bool = False


class SshError(RuntimeError):
    """Stable base class for transport, protocol and remote failures."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        returncode: int | None,
        attempts: int,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.returncode = returncode
        self.attempts = attempts


class SshConnectionError(SshError):
    """Connection-class failures that do not indicate remote-command failure."""


class SshRemoteError(SshError):
    """Remote command completed but failed logically or with exit status."""


class SshTimeoutError(SshError):
    """Local subprocess timeout while executing ssh."""


class SshProtocolError(SshError):
    """Malformed protocol payload or invalid log metadata."""


SSH_COMMAND: Final[str] = "bash"
SLEEPER: Callable[[float], None] = time.sleep
_READ_MAX_BYTES: Final[int] = 1_048_576
_READ_DEFAULT_BYTES: Final[int] = 65_536
_SAFE_FACT_KEYS: Final[frozenset[str]] = frozenset(
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

_TARGET_META: Final[frozenset[str]] = frozenset(
    {
        ";",
        "&",
        "|",
        "$",
        "`",
        "\\",
        '"',
        "'",
        "<",
        ">",
        "!",
        "#",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "*",
        "?",
        "~",
    }
)
_REMOTE_PATH_META: Final[frozenset[str]] = frozenset(
    {
        "\\",
        "`",
        "$",
        "\n",
        "\r",
        "\t",
        "\x00",
        ";",
        "|",
        "&",
    }
)
_HOST_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*$"
)
_USER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")

_REMOTE_READ_SCRIPT: Final[str] = r"""set -euo pipefail
path=$1
offset=$2
max_bytes=$3

if [ ! -e "$path" ]; then
  printf 'status=missing\n'
  exit 0
fi

if [ -L "$path" ]; then
  printf 'status=symlink\n'
  exit 0
fi

if [ ! -f "$path" ]; then
  printf 'status=non_regular\n'
  exit 0
fi

size=$(stat -c %s "$path")

if [ "$offset" -gt "$size" ]; then
  printf 'status=truncated\n'
  exit 0
fi

if [ "$offset" -eq "$size" ]; then
  printf 'status=ok\n'
  printf 'next_offset=%s\n' "$offset"
  printf 'eof=true\n'
  printf 'payload=\n'
  printf 'bytes=0\n'
  exit 0
fi

read_bytes=$max_bytes
if [ $((offset + read_bytes)) -gt "$size" ]; then
  read_bytes=$((size - offset))
fi

payload=$(dd if="$path" bs=1 skip="$offset" count="$read_bytes" status=none | base64 -w0)
next_offset=$((offset + read_bytes))
if [ "$next_offset" -ge "$size" ]; then
  eof=true
else
  eof=false
fi

printf 'status=ok\n'
printf 'next_offset=%s\n' "$next_offset"
printf 'eof=%s\n' "$eof"
printf 'payload=%s\n' "$payload"
printf 'bytes=%s\n' "$read_bytes"
"""


def _validate_target(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("target must be a string")

    if not value.strip() or value != value.strip():
        raise ValueError("target contains surrounding whitespace")

    if any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in value):
        raise ValueError("target contains control characters")

    if value.startswith("-"):
        raise ValueError("target cannot start with a leading dash")

    if any(ch in _TARGET_META for ch in value):
        raise ValueError("target contains unsupported shell metacharacters")

    if value.count("@") > 1:
        raise ValueError("target must be host or user@host")

    if "/" in value:
        raise ValueError("target cannot include path separators")

    user: str | None
    host: str
    if "@" in value:
        user, host = value.split("@", maxsplit=1)
        if not user:
            raise ValueError("target user cannot be empty")
        if not host:
            raise ValueError("target host cannot be empty")
        if not _USER_RE.fullmatch(user):
            raise ValueError("target user has an invalid format")
    else:
        user = None
        host = value

    if not _HOST_RE.fullmatch(host):
        raise ValueError("target host has an invalid format")

    return f"{user}@{host}" if user else host


def _coerce_positive_int(value: object, field: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")

    if isinstance(value, int):
        if value < 0 if allow_zero else value <= 0:
            raise ValueError(
                f"{field} must be {'non-negative' if allow_zero else 'positive'} integer"
            )
        return value

    if isinstance(value, str):
        if not value:
            raise ValueError(f"{field} must be an integer")
        if not value.isdigit():
            raise ValueError(f"{field} must be an integer")
        if value.startswith("0") and value != "0":
            raise ValueError(f"{field} must not use leading zeros")
        parsed = int(value)
        if parsed < 0 if allow_zero else parsed <= 0:
            raise ValueError(
                f"{field} must be {'non-negative' if allow_zero else 'positive'} integer"
            )
        return parsed

    raise ValueError(f"{field} must be an integer")


def _coerce_positive_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite number")
    if not isinstance(value, int | float):
        raise ValueError(f"{field} must be finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return parsed


def _validate_remote_path(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("remote path must be a string")

    if value != value.strip():
        raise ValueError("remote path contains surrounding whitespace")

    if not value.startswith("/"):
        raise ValueError("remote path must be absolute")

    if value == "/":
        raise ValueError("remote path must point to a file path")

    if any(ch in _REMOTE_PATH_META for ch in value):
        raise ValueError("remote path contains unsupported characters")

    if any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in value):
        raise ValueError("remote path contains control characters")

    parts = [part for part in value.split("/") if part]
    if ".." in parts:
        raise ValueError("remote path must not contain traversal")
    if any(part == "." for part in parts):
        raise ValueError("remote path must not contain traversal")

    return value


def _decode_text(value: object, field: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SshProtocolError(
                f"{field} is not valid UTF-8",
                category="protocol",
                returncode=None,
                attempts=0,
            ) from exc
    raise SshProtocolError(
        f"{field} is not textual",
        category="protocol",
        returncode=None,
        attempts=0,
    )


def _coerce_result_text(result: object, field: str) -> str:
    if not hasattr(result, field):
        raise SshProtocolError(
            "runner response is missing required output",
            category="protocol",
            returncode=None,
            attempts=0,
        )
    return _decode_text(getattr(result, field), field)


def _coerce_result_rc(result: object) -> int:
    if not hasattr(result, "returncode"):
        raise SshProtocolError(
            "runner response is missing return code",
            category="protocol",
            returncode=None,
            attempts=0,
        )
    value = result.returncode
    if not isinstance(value, int):
        raise SshProtocolError(
            "runner response return code must be an integer",
            category="protocol",
            returncode=None,
            attempts=0,
        )
    return value


def _decode_utf8_with_boundary(raw: bytes) -> tuple[str, int]:
    if not raw:
        return "", 0

    for drop in range(0, 4):
        candidate = raw[:-drop] if drop else raw
        try:
            return candidate.decode("utf-8"), len(candidate)
        except UnicodeDecodeError as exc:
            if exc.reason != "unexpected end of data" or exc.end != len(candidate):
                raise SshProtocolError(
                    "remote log payload is not UTF-8",
                    category="protocol",
                    returncode=None,
                    attempts=0,
                ) from exc
            if drop >= 3:
                raise SshProtocolError(
                    "remote log payload UTF-8 boundary could not be resolved",
                    category="protocol",
                    returncode=None,
                    attempts=0,
                ) from exc
            continue

    raise SshProtocolError(
        "remote log payload could not be decoded",
        category="protocol",
        returncode=None,
        attempts=0,
    )


def _parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"invalid boolean payload: {value}")


def _parse_payload(raw: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in raw.splitlines():
        if not line:
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise SshProtocolError(
                "invalid log protocol line",
                category="protocol",
                returncode=None,
                attempts=0,
            )
        parsed[key] = value
    return parsed


def _clear_sensitive_facts(
    payload: Mapping[str, object] | list[object] | object,
) -> dict[str, object] | list[object] | object:
    if isinstance(payload, dict):
        sanitized: dict[str, object] = {}
        for key, value in payload.items():
            if not isinstance(key, str):
                raise TypeError("facts keys must be strings")
            if key.casefold() in _SAFE_FACT_KEYS:
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = _clear_sensitive_facts(value)
        return sanitized

    if isinstance(payload, list):
        return [_clear_sensitive_facts(item) for item in payload]

    if isinstance(payload, (str, int, float, bool, type(None))):
        if isinstance(payload, float) and not math.isfinite(payload):
            raise TypeError("facts must not include NaN or infinity")
        return payload

    raise TypeError("facts must be JSON-safe values")


def _is_auth_or_host_failure(stderr: str) -> bool:
    lowered = stderr.lower()
    return (
        "permission denied" in lowered
        or "host key verification failed" in lowered
        or "could not resolve" in lowered
        or "connection timed out" in lowered
    )


class SshClient:
    """OpenSSH subprocess boundary with deterministic retry and read helpers."""

    def __init__(
        self,
        *,
        target: str,
        runner: _Runner | None = None,
        attempts: int = 3,
        connect_timeout: int = 30,
        server_alive_interval: int = 10,
        server_alive_count_max: int = 3,
        backoff: tuple[float, ...] = (2.0, 5.0),
        sleeper: Callable[[float], None] | None = None,
        command_timeout: float = 120.0,
        read_max_bytes: int = _READ_MAX_BYTES,
    ) -> None:
        self.target = _validate_target(target)
        self._attempts = _coerce_positive_int(attempts, "attempts")
        self._connect_timeout = _coerce_positive_int(connect_timeout, "connect_timeout")
        self._server_alive_interval = _coerce_positive_int(
            server_alive_interval,
            "server_alive_interval",
        )
        self._server_alive_count_max = _coerce_positive_int(
            server_alive_count_max,
            "server_alive_count_max",
        )
        self._command_timeout = _coerce_positive_float(
            command_timeout, "command_timeout"
        )
        if read_max_bytes > _READ_MAX_BYTES:
            raise ValueError("read_max_bytes exceeds configured hard maximum")
        self._read_max_bytes = _coerce_positive_int(read_max_bytes, "read_max_bytes")
        if not backoff:
            raise ValueError("backoff must provide at least one delay")
        self._backoff = tuple(
            _coerce_positive_float(delay, "backoff") for delay in backoff
        )
        if sleeper is None:
            sleeper = SLEEPER
        self._sleeper = sleeper
        self._runner: _Runner = runner if runner is not None else _default_runner

    def run(self, command: str) -> SshResult:
        if not isinstance(command, str):
            raise ValueError("command must be a string")
        if command != command.strip():
            raise ValueError("command may not contain surrounding whitespace")

        args = self._build_run_args(command)
        return self._run_with_retries(args)

    def read_since(
        self,
        remote_path: str,
        offset: int,
        max_bytes: int | None = None,
    ) -> LogChunk:
        validated_offset = _coerce_positive_int(offset, "offset", allow_zero=True)
        path = _validate_remote_path(remote_path)
        if max_bytes is None:
            request_max = _READ_DEFAULT_BYTES
        else:
            request_max = _coerce_positive_int(max_bytes, "max_bytes")
            if request_max > self._read_max_bytes:
                raise ValueError("max_bytes exceeds configured hard maximum")

        request_max = min(request_max, self._read_max_bytes)

        args = self._build_read_args(path, validated_offset, request_max)
        result = self._run_with_retries(args)

        parsed = _parse_payload(result.stdout)
        status = parsed.get("status")
        if status == "missing":
            return LogChunk(
                text="", next_offset=validated_offset, eof=False, reset=False
            )

        if status == "truncated":
            return LogChunk(text="", next_offset=0, eof=False, reset=True)

        if status == "symlink":
            raise SshProtocolError(
                "remote path is not allowed to be a symlink",
                category="protocol",
                returncode=result.returncode,
                attempts=result.attempts,
            )

        if status == "non_regular":
            raise SshProtocolError(
                "remote path is not a regular file",
                category="protocol",
                returncode=result.returncode,
                attempts=result.attempts,
            )

        if status != "ok":
            raise SshProtocolError(
                "remote log protocol returned unsupported status",
                category="protocol",
                returncode=result.returncode,
                attempts=result.attempts,
            )

        remote_next = int(parsed["next_offset"])
        eof = _parse_bool(parsed["eof"])
        encoded = parsed.get("payload", "")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise SshProtocolError(
                "remote log payload is not base64",
                category="protocol",
                returncode=result.returncode,
                attempts=result.attempts,
            ) from exc

        text, consumed = _decode_utf8_with_boundary(raw)
        rewind = len(raw) - consumed
        next_offset = remote_next - rewind
        if next_offset < validated_offset:
            raise SshProtocolError(
                "remote log offset regression",
                category="protocol",
                returncode=result.returncode,
                attempts=result.attempts,
            )
        return LogChunk(text=text, next_offset=next_offset, eof=eof, reset=False)

    def _run_with_retries(self, args: list[str]) -> SshResult:
        for attempt in range(1, self._attempts + 1):
            try:
                process = self._runner(
                    args,
                    capture_output=True,
                    text=False,
                    timeout=self._command_timeout,
                )
            except TimeoutExpired as exc:
                if attempt >= self._attempts:
                    raise SshTimeoutError(
                        "ssh command timeout",
                        category="timeout",
                        returncode=None,
                        attempts=attempt,
                    ) from exc
                self._sleeper(self._backoff[(attempt - 1) % len(self._backoff)])
                continue
            except Exception as exc:
                raise SshConnectionError(
                    "ssh transport runner failed",
                    category="connection",
                    returncode=None,
                    attempts=attempt,
                ) from exc

            returncode = _coerce_result_rc(process)
            stdout = _coerce_result_text(process, "stdout")
            stderr = _coerce_result_text(process, "stderr")

            if returncode == 0:
                return SshResult(
                    stdout=stdout,
                    stderr=stderr,
                    returncode=returncode,
                    attempts=attempt,
                )

            if returncode == 255:
                if _is_auth_or_host_failure(stderr):
                    raise SshConnectionError(
                        "ssh authentication or host-key failure",
                        category="connection",
                        returncode=returncode,
                        attempts=attempt,
                    )
                if attempt >= self._attempts:
                    raise SshConnectionError(
                        "ssh transport failure",
                        category="connection",
                        returncode=returncode,
                        attempts=attempt,
                    )
                self._sleeper(self._backoff[(attempt - 1) % len(self._backoff)])
                continue

            raise SshRemoteError(
                "ssh remote command returned non-zero status",
                category="remote",
                returncode=returncode,
                attempts=attempt,
            )

        # Defensive fallback; normally unreachable due loop exits above.
        raise SshConnectionError(
            "ssh command exhausted retries",
            category="connection",
            returncode=None,
            attempts=self._attempts,
        )

    def _build_run_args(self, command: str) -> list[str]:
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            f"ConnectTimeout={self._connect_timeout}",
            "-o",
            f"ServerAliveInterval={self._server_alive_interval}",
            "-o",
            f"ServerAliveCountMax={self._server_alive_count_max}",
            "--",
            self.target,
            SSH_COMMAND,
            "-lc",
            shlex.quote(command),
        ]

    def _build_read_args(self, path: str, offset: int, max_bytes: int) -> list[str]:
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            f"ConnectTimeout={self._connect_timeout}",
            "-o",
            f"ServerAliveInterval={self._server_alive_interval}",
            "-o",
            f"ServerAliveCountMax={self._server_alive_count_max}",
            "--",
            self.target,
            SSH_COMMAND,
            "-lc",
            shlex.quote(_REMOTE_READ_SCRIPT),
            "operator-read",
            shlex.quote(path),
            str(offset),
            str(max_bytes),
        ]


def _default_runner(
    args: Sequence[str],
    *,
    capture_output: bool,
    text: bool,
    timeout: float,
) -> object:
    return subprocess.run(
        list(args),
        capture_output=capture_output,
        text=text,
        check=False,
        timeout=timeout,
    )


__all__ = [
    "SshResult",
    "LogChunk",
    "SshError",
    "SshConnectionError",
    "SshRemoteError",
    "SshTimeoutError",
    "SshProtocolError",
    "SshClient",
    "_clear_sensitive_facts",
]
