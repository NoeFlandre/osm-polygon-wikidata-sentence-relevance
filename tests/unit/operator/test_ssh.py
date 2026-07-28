"""Strict TDD coverage for bounded OpenSSH transport and resumable log reads."""

from __future__ import annotations

import base64
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from osm_polygon_sentence_relevance.operator.ssh import (
    _READ_MAX_BYTES,
    SLEEPER,
    LogChunk,
    SshClient,
    SshConnectionError,
    SshProtocolError,
    SshRemoteError,
    SshResult,
    SshTimeoutError,
    _clear_sensitive_facts,
    _coerce_positive_float,
    _coerce_positive_int,
    _coerce_result_rc,
    _coerce_result_text,
    _decode_text,
    _decode_utf8_with_boundary,
    _default_runner,
    _is_auth_or_host_failure,
    _parse_bool,
    _parse_payload,
    _validate_remote_path,
    _validate_target,
)


@dataclass
class _FakeCompleted:
    """Typed subprocess-like result returned by the fake SSH runner."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


class FakeRunner:
    """Deterministic runner that replays queued responses."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, args: list[str], **kwargs: object) -> _FakeCompleted:
        self.calls.append((args, dict(kwargs)))
        if not self.responses:
            raise AssertionError("runner invoked after all responses were consumed")

        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, _FakeCompleted)
        return response


class FakeSleeper:
    """Collect exact backoff delays without sleeping."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def __call__(self, value: float) -> None:
        self.values.append(value)


def _build_client(
    *,
    target: str = "nancy",
    responses: list[object] | None = None,
    attempts: int = 3,
    connect_timeout: int = 30,
    server_alive_interval: int = 10,
    server_alive_count_max: int = 4,
    backoff: tuple[float, ...] = (2.0, 5.0, 8.0),
    max_read_bytes: int = 32768,
    sleeper: FakeSleeper | Callable[[float], None] | None = None,
    command_timeout: int = 90,
) -> tuple[SshClient, FakeRunner, FakeSleeper]:
    runner = FakeRunner(responses or [])
    fake_sleeper = sleeper if isinstance(sleeper, FakeSleeper) else FakeSleeper()

    client = SshClient(
        target=target,
        runner=runner,
        attempts=attempts,
        connect_timeout=connect_timeout,
        server_alive_interval=server_alive_interval,
        server_alive_count_max=server_alive_count_max,
        backoff=backoff,
        read_max_bytes=max_read_bytes,
        sleeper=fake_sleeper,
        command_timeout=command_timeout,
    )
    return client, runner, fake_sleeper


def _status_payload(status: str, **fields: object) -> str:
    lines = [f"status={status}"]
    for key, value in fields.items():
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def _payload_text(text: str, *, next_offset: int, eof: bool) -> str:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return _status_payload(
        "ok",
        next_offset=next_offset,
        eof=str(eof).lower(),
        payload=encoded,
        bytes=0,
    )


def _args_for_call(client: SshClient, runner: FakeRunner) -> list[str]:
    return runner.calls[0][0]


def _kwargs_for_call(client: SshClient, runner: FakeRunner) -> dict[str, object]:
    return runner.calls[0][1]


def test_run_uses_expected_ssh_argv() -> None:
    client, runner, _ = _build_client(
        responses=[_FakeCompleted(0, "ok", "")],
    )
    result = client.run("echo hello")

    args = _args_for_call(client, runner)
    assert args[:4] == ["ssh", "-o", "BatchMode=yes", "-o"]
    assert "-o" in args
    assert "ForwardAgent=no" in args
    assert "-o" in args
    assert "ClearAllForwardings=yes" in args
    idx = args.index("--")
    assert args[idx + 1] == "nancy"
    assert args[idx + 2] == "bash"
    assert args[idx + 3] == "-lc"
    assert args[idx + 4] == "echo hello"
    assert args[-1] == "echo hello"
    assert result.attempts == 1


def test_batch_mode_enabled() -> None:
    client, runner, _ = _build_client(responses=[_FakeCompleted(0, "", "")])
    client.run("echo ready")

    args = _args_for_call(client, runner)
    assert "BatchMode=yes" in args


def test_forward_agent_disabled() -> None:
    client, runner, _ = _build_client(responses=[_FakeCompleted(0, "", "")])
    client.run("echo done")

    args = _args_for_call(client, runner)
    assert "ForwardAgent=no" in args


def test_clear_all_forwardings_enabled() -> None:
    client, runner, _ = _build_client(responses=[_FakeCompleted(0, "", "")])
    client.run("echo done")

    args = _args_for_call(client, runner)
    assert "ClearAllForwardings=yes" in args


def test_finite_connect_timeout_and_alive_settings() -> None:
    client, runner, _ = _build_client(
        connect_timeout=17,
        server_alive_interval=11,
        server_alive_count_max=3,
        responses=[_FakeCompleted(0, "", "")],
    )

    client.run("true")
    args = _args_for_call(client, runner)
    assert "-o" in args
    assert "ConnectTimeout=17" in args
    assert "ServerAliveInterval=11" in args
    assert "ServerAliveCountMax=3" in args


def test_dash_dash_injects_clean_target_positioning() -> None:
    client, runner, _ = _build_client(responses=[_FakeCompleted(0, "", "")])
    client.run("echo")

    args = _args_for_call(client, runner)
    assert "--" in args
    assert args.index("--") < len(args) - 2


def test_exactly_one_fixed_bootstrap_command_is_used() -> None:
    client, runner, _ = _build_client(responses=[_FakeCompleted(0, "", "")])
    client.run("echo")

    args = _args_for_call(client, runner)
    idx = args.index("-lc")
    assert args[idx + 1] == "echo"


def test_shell_true_is_never_used() -> None:
    client, runner, _ = _build_client(responses=[_FakeCompleted(0, "", "")])
    client.run("echo")

    assert _kwargs_for_call(client, runner).get("shell") is None


def test_no_real_subprocess_runs_with_fake_runner() -> None:
    _, runner, _ = _build_client(responses=[_FakeCompleted(0, "", "")])
    assert len(runner.calls) == 0


def test_target_nancy_accepted() -> None:
    client, runner, _ = _build_client(
        responses=[_FakeCompleted(0, "", "")], target="nancy"
    )
    client.run("echo")
    assert (
        _args_for_call(client, runner)[_args_for_call(client, runner).index("--") + 1]
        == "nancy"
    )


def test_target_user_at_nancy_accepted() -> None:
    client, runner, _ = _build_client(
        responses=[_FakeCompleted(0, "", "")],
        target="ops@nancy",
    )
    client.run("echo")
    assert (
        _args_for_call(client, runner)[_args_for_call(client, runner).index("--") + 1]
        == "ops@nancy"
    )


def test_leading_dash_target_rejected() -> None:
    with pytest.raises(ValueError, match=r"leading dash"):
        SshClient(target="-nancy")


def test_whitespace_target_rejected() -> None:
    with pytest.raises(ValueError, match=r"whitespace|control"):
        SshClient(target="na ncy")


def test_meta_or_control_target_rejected() -> None:
    for target in [
        "nancy;rm -rf /",
        "na\tncy",
        "na\x00ncy",
        "na\ncy",
        "na/",
        "na:22",
        "user@",
    ]:
        with pytest.raises(ValueError, match=r"target"):
            SshClient(target=target)


def test_run_success_preserves_stdout_stderr_returncode() -> None:
    client, runner, _ = _build_client(
        responses=[_FakeCompleted(0, "stdout", "stderr")],
    )

    result = client.run("echo hello")

    assert isinstance(result, SshResult)
    assert result.stdout == "stdout"
    assert result.stderr == "stderr"
    assert result.returncode == 0
    assert result.attempts == 1


def test_command_argument_is_required_and_tight() -> None:
    client, _, _ = _build_client(responses=[_FakeCompleted(0, "ok", "")])

    with pytest.raises(ValueError, match="command must be a string"):
        client.run(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="surrounding whitespace"):
        client.run(" echo hello")


def test_transport_failure_retries_to_exact_bound() -> None:
    client, runner, _ = _build_client(
        attempts=4,
        responses=[
            _FakeCompleted(255),
            _FakeCompleted(255),
            _FakeCompleted(255),
            _FakeCompleted(255),
        ],
    )

    with pytest.raises(SshConnectionError) as exc:
        client.run("echo")

    assert exc.value.attempts == 4
    assert len(runner.calls) == 4
    assert isinstance(exc.value, SshConnectionError)


def test_timeout_failure_retries_to_exact_bound() -> None:
    client, runner, _ = _build_client(
        attempts=2,
        responses=[
            subprocess.TimeoutExpired(["ssh"], timeout=10),
            subprocess.TimeoutExpired(["ssh"], timeout=10),
        ],
    )

    with pytest.raises(SshTimeoutError) as exc:
        client.run("echo")

    assert exc.value.attempts == 2
    assert len(runner.calls) == 2


def test_retry_backoff_sequence_is_deterministic() -> None:
    sleeper = FakeSleeper()
    client, runner, fake = _build_client(
        attempts=3,
        backoff=(0.5, 1.5, 2.5),
        responses=[
            _FakeCompleted(255),
            _FakeCompleted(255),
            _FakeCompleted(0, "ok", ""),
        ],
        sleeper=sleeper,
    )

    client.run("echo")

    assert fake.values == [0.5, 1.5]
    assert len(runner.calls) == 3


def test_remote_exit_two_is_not_retried() -> None:
    client, runner, _ = _build_client(
        attempts=3,
        responses=[_FakeCompleted(2, "", "command failed")],
    )

    with pytest.raises(SshRemoteError):
        client.run("echo")

    assert len(runner.calls) == 1


def test_auth_failure_is_not_retried() -> None:
    client, runner, _ = _build_client(
        attempts=3,
        responses=[
            _FakeCompleted(
                255,
                "",
                "Permission denied (publickey,password).",
            ),
        ],
    )

    with pytest.raises(SshConnectionError) as exc:
        client.run("echo")

    assert len(runner.calls) == 1
    assert exc.value.attempts == 1


def test_exhausted_transport_failure_is_safe_error() -> None:
    client, runner, _ = _build_client(
        attempts=2,
        responses=[_FakeCompleted(255), _FakeCompleted(255)],
    )

    with pytest.raises(SshConnectionError) as exc:
        client.run("echo")

    assert exc.value.attempts == 2
    assert "ssh transport failure" in str(exc.value)


def test_exhausted_timeout_failure_is_safe_error() -> None:
    client, runner, _ = _build_client(
        attempts=2,
        responses=[
            subprocess.TimeoutExpired(["ssh"], timeout=10),
            subprocess.TimeoutExpired(["ssh"], timeout=10),
        ],
    )

    with pytest.raises(SshTimeoutError) as exc:
        client.run("echo")

    assert exc.value.attempts == 2
    assert "ssh command timeout" in str(exc.value)


def test_public_errors_do_not_leak_injected_secret() -> None:
    secret = "TOKEN_XYZ"
    client, runner, _ = _build_client(
        responses=[
            _FakeCompleted(2, "", f"error from remote using {secret}"),
        ],
    )

    with pytest.raises(SshRemoteError) as exc:
        client.run(f"echo {secret}")

    assert secret not in str(exc.value)


def test_recursive_redaction_removes_sensitive_fields() -> None:
    payload: dict[str, Any] = {
        "token": "hidden-token",
        "ok": True,
        "nested": {
            "Access_Token": "nested-token",
            "prompt": "ask",
            "details": {
                "secret": "deep",
                "value": "keep",
            },
        },
    }
    redacted = _clear_sensitive_facts(payload)

    assert redacted == {
        "token": "<redacted>",
        "ok": True,
        "nested": {
            "Access_Token": "<redacted>",
            "prompt": "<redacted>",
            "details": {
                "secret": "<redacted>",
                "value": "keep",
            },
        },
    }


def test_read_since_rejects_negative_offset() -> None:
    client, _, _ = _build_client(responses=[])

    with pytest.raises(ValueError, match=r"offset"):
        client.read_since("/tmp/run.log", -1)


def test_read_since_rejects_zero_or_oversized_max_bytes() -> None:
    client, _, _ = _build_client(responses=[], max_read_bytes=1024)

    with pytest.raises(ValueError, match=r"max_bytes"):
        client.read_since("/tmp/run.log", 0, max_bytes=0)

    with pytest.raises(ValueError, match=r"max_bytes"):
        client.read_since("/tmp/run.log", 0, max_bytes=2048)


def test_read_since_rejects_relative_or_traversal_path() -> None:
    client, _, _ = _build_client(responses=[])

    for path in [
        "logs/run.log",
        "./run.log",
        "../run.log",
        "/tmp/../run.log",
        "/run/./log",
    ]:
        with pytest.raises(ValueError, match=r"remote path"):
            client.read_since(path, 0)


def test_read_since_rejects_newline_control_and_meta_chars() -> None:
    client, _, _ = _build_client(responses=[])

    for path in [
        "/tmp/bad\nlog",
        "/tmp/bad\x00log",
        "/tmp/bad;rm.log",
        "/tmp/bad|x",
        "/tmp/$HOME/log",
    ]:
        with pytest.raises(ValueError, match=r"remote path"):
            client.read_since(path, 0)


def test_missing_log_is_not_fatal() -> None:
    client, runner, _ = _build_client(
        responses=[_FakeCompleted(0, _status_payload("missing"))],
    )
    chunk = client.read_since("/tmp/missing.log", 12)

    assert isinstance(chunk, LogChunk)
    assert chunk.text == ""
    assert chunk.next_offset == 12
    assert chunk.eof is False
    assert chunk.reset is False
    assert runner.calls[0][0][-3] == "/tmp/missing.log"


def test_empty_log_returns_offset_and_eof() -> None:
    client, _, _ = _build_client(
        responses=[
            _FakeCompleted(
                0, _status_payload("ok", next_offset=0, eof=True, payload="", bytes=0)
            )
        ],
    )
    chunk = client.read_since("/tmp/empty.log", 0)

    assert chunk.text == ""
    assert chunk.next_offset == 0
    assert chunk.eof is True


def test_growing_ascii_log_advances_offset() -> None:
    client, _, _ = _build_client(
        responses=[
            _FakeCompleted(
                0,
                _payload_text("first", next_offset=5, eof=False),
            ),
        ]
    )
    chunk = client.read_since("/tmp/out.log", 0, max_bytes=16)

    assert chunk.text == "first"
    assert chunk.next_offset == 5
    assert chunk.eof is False


def test_utf8_multibyte_boundary_is_not_lost_or_duplicated() -> None:
    full = "汉字"
    bytes_payload = full.encode("utf-8")
    first_part = bytes_payload[:4]
    second_part = bytes_payload[3:]

    client, runner, _ = _build_client(
        responses=[
            _FakeCompleted(
                0,
                _status_payload(
                    "ok",
                    next_offset=4,
                    eof=False,
                    payload=base64.b64encode(first_part).decode("ascii"),
                    bytes=len(first_part),
                ),
            ),
            _FakeCompleted(
                0,
                _status_payload(
                    "ok",
                    next_offset=6,
                    eof=True,
                    payload=base64.b64encode(second_part).decode("ascii"),
                    bytes=len(second_part),
                ),
            ),
        ]
    )

    first = client.read_since("/tmp/utf8.log", 0, max_bytes=4)
    second = client.read_since("/tmp/utf8.log", first.next_offset, max_bytes=4)

    assert first.text == full[:1]
    assert second.text == full[1:]
    assert first.next_offset == 3
    assert first.next_offset + len(second.text.encode("utf-8")) >= len(
        full.encode("utf-8")
    )


def test_max_bytes_is_strictly_enforced_by_validator() -> None:
    client, _, _ = _build_client(max_read_bytes=4)

    with pytest.raises(ValueError, match=r"max_bytes"):
        client.read_since("/tmp/run.log", 0, max_bytes=128)


def test_truncation_replacement_is_explicit() -> None:
    client, _, _ = _build_client(
        responses=[
            _FakeCompleted(0, _status_payload("truncated", next_offset=0, eof=False))
        ],
    )

    chunk = client.read_since("/tmp/run.log", 128)

    assert chunk.reset is True
    assert chunk.next_offset == 0


def test_symlink_file_is_rejected_explicitly() -> None:
    client, _, _ = _build_client(
        responses=[
            _FakeCompleted(0, _status_payload("symlink", next_offset=0, eof=False))
        ],
    )

    with pytest.raises(SshProtocolError):
        client.read_since("/tmp/run.log", 0)


def test_non_regular_file_is_rejected_explicitly() -> None:
    client, _, _ = _build_client(
        responses=[
            _FakeCompleted(0, _status_payload("non_regular", next_offset=0, eof=False))
        ],
    )

    with pytest.raises(SshProtocolError):
        client.read_since("/tmp/run.log", 0)


def test_permission_denied_remote_file_is_explicit_failure() -> None:
    client, _, _ = _build_client(
        responses=[_FakeCompleted(13, "", "Permission denied")],
    )

    with pytest.raises(SshRemoteError):
        client.read_since("/tmp/run.log", 0)


def test_log_read_remote_app_failure_is_not_retried() -> None:
    client, runner, _ = _build_client(
        attempts=3,
        responses=[_FakeCompleted(2, "", "boom")],
    )

    with pytest.raises(SshRemoteError):
        client.read_since("/tmp/run.log", 0)

    assert len(runner.calls) == 1


def test_read_since_timeout_is_used_for_local_subprocess_call() -> None:
    client, runner, _ = _build_client(
        responses=[_FakeCompleted(0, _status_payload("missing"))],
        command_timeout=55,
    )

    client.read_since("/tmp/run.log", 0)

    _, kwargs = _args_for_call(client, runner), _kwargs_for_call(client, runner)

    assert "timeout" in kwargs
    assert kwargs["timeout"] == 55


def test_mutable_metadata_cannot_mutate_returned_redacted_result() -> None:
    fact: dict[str, Any] = {"secret": "hidden", "ok": True}
    facts_copy = fact.copy()
    redacted = _clear_sensitive_facts(fact)
    facts_copy["secret"] = "changed"
    fact["ok"] = False

    assert redacted == {"secret": "<redacted>", "ok": True}


def test_validate_target_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="target must be a string"):
        _validate_target(123)


def test_validate_target_rejects_control_characters() -> None:
    with pytest.raises(ValueError, match="control"):
        _validate_target("na\ncy")


def test_validate_target_rejects_unsupported_metacharacters() -> None:
    with pytest.raises(ValueError, match="metachar"):
        _validate_target("nancy;rm")


def test_validate_target_rejects_multiple_user_host_separators() -> None:
    with pytest.raises(ValueError, match="host or user@host"):
        _validate_target("a@b@c")


def test_validate_target_rejects_empty_user_or_host() -> None:
    with pytest.raises(ValueError, match="user cannot be empty"):
        _validate_target("@nancy")
    with pytest.raises(ValueError, match="host cannot be empty"):
        _validate_target("nancy@")


def test_validate_target_rejects_invalid_user_and_host() -> None:
    with pytest.raises(ValueError, match="invalid format"):
        _validate_target(".bad@nancy")
    with pytest.raises(ValueError, match="host has an invalid format"):
        _validate_target("nancy@bad_host")
    with pytest.raises(ValueError, match="cannot include path separators"):
        _validate_target("nancy/host")


def test_validate_target_accepts_valid_user_at_host() -> None:
    assert _validate_target("ops@nancy") == "ops@nancy"


def test_validate_target_rejects_surrounding_whitespace() -> None:
    with pytest.raises(ValueError, match="surrounding whitespace"):
        _validate_target(" nancy")


def test_coerce_positive_int_rejects_non_integer_inputs() -> None:
    with pytest.raises(ValueError, match="integer"):
        _coerce_positive_int(True, "attempts")
    with pytest.raises(ValueError, match="integer"):
        _coerce_positive_int("", "attempts")
    with pytest.raises(ValueError, match="must be an integer"):
        _coerce_positive_int("x", "attempts")
    with pytest.raises(ValueError, match="must be an integer"):
        _coerce_positive_int([], "attempts")
    with pytest.raises(ValueError, match="must be"):
        _coerce_positive_int(0, "attempts")


def test_coerce_positive_int_rejects_string_zero_and_parses_valid_string() -> None:
    with pytest.raises(ValueError, match="positive"):
        _coerce_positive_int("0", "attempts")
    assert _coerce_positive_int("7", "attempts") == 7


def test_coerce_positive_int_rejects_non_integer_text() -> None:
    with pytest.raises(ValueError, match="integer"):
        _coerce_positive_int("abc", "attempts")


def test_coerce_positive_int_rejects_leading_zeros() -> None:
    with pytest.raises(ValueError, match="must not use leading zeros"):
        _coerce_positive_int("07", "attempts")


def test_coerce_positive_int_allows_zero_with_flag() -> None:
    assert _coerce_positive_int(0, "offset", allow_zero=True) == 0


def test_coerce_positive_float_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="finite number"):
        _coerce_positive_float(False, "command_timeout")
    with pytest.raises(ValueError, match="finite number"):
        _coerce_positive_float("a", "command_timeout")
    with pytest.raises(ValueError, match="must be a finite"):
        _coerce_positive_float(float("nan"), "command_timeout")
    with pytest.raises(ValueError, match="must be a finite"):
        _coerce_positive_float(float("inf"), "command_timeout")


def test_validate_remote_path_rejects_whitespace_and_control_bytes() -> None:
    with pytest.raises(ValueError, match="surrounding whitespace"):
        _validate_remote_path(" /tmp/run.log")
    with pytest.raises(ValueError, match="control"):
        _validate_remote_path("/tmp/run\x1f.log")


def test_validate_remote_path_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="string"):
        _validate_remote_path(123)
    with pytest.raises(ValueError, match="absolute"):
        _validate_remote_path("logs/run.log")
    with pytest.raises(ValueError, match="file path"):
        _validate_remote_path("/")
    with pytest.raises(ValueError, match="traversal"):
        _validate_remote_path("/tmp/../run.log")


def test_validate_remote_path_rejects_relative_components_and_meta_chars() -> None:
    with pytest.raises(ValueError, match="traversal"):
        _validate_remote_path("/tmp/./run.log")
    with pytest.raises(ValueError, match="unsupported"):
        _validate_remote_path("/tmp/abc;def")


def test_decode_text_rejects_non_utf8_bytes() -> None:
    with pytest.raises(SshProtocolError, match="not valid UTF-8"):
        _decode_text(b"\xff", "stdout")


def test_coerce_result_text_requires_output_field() -> None:
    class Missing:
        returncode = 0

    with pytest.raises(SshProtocolError, match="missing required output"):
        _coerce_result_text(Missing(), "stdout")


def test_coerce_result_rc_requires_return_code() -> None:
    class Missing:
        stdout = ""

    with pytest.raises(SshProtocolError, match="missing return code"):
        _coerce_result_rc(Missing())


def test_coerce_result_rc_requires_integer_return_code() -> None:
    with pytest.raises(SshProtocolError, match="must be an integer"):
        _coerce_result_rc(_FakeCompleted("x", "", ""))


def test_coerce_result_text_rejects_nontext_output() -> None:
    class Bad:
        returncode = 0
        stdout = object()
        stderr = ""

    with pytest.raises(SshProtocolError, match="is not textual"):
        _coerce_result_text(Bad(), "stdout")


def test_decode_utf8_with_boundary_handles_empty_and_invalid_inputs() -> None:
    assert _decode_utf8_with_boundary(b"") == ("", 0)
    with pytest.raises(SshProtocolError, match="not UTF-8"):
        _decode_utf8_with_boundary(b"\xff")


def test_parse_bool_is_case_insensitive_and_rejects_bad_values() -> None:
    assert _parse_bool("TRUE")
    assert not _parse_bool("False")
    with pytest.raises(ValueError, match="invalid boolean"):
        _parse_bool("maybe")


def test_parse_payload_rejects_invalid_line() -> None:
    with pytest.raises(SshProtocolError, match="invalid log protocol line"):
        _parse_payload("badline")


def test_parse_payload_ignores_blank_lines() -> None:
    payload = _parse_payload("\nstatus=ok\n\neof=true\n")

    assert payload == {"status": "ok", "eof": "true"}


def test_default_runner_executes_subprocess_without_shell() -> None:
    output = _default_runner(
        ["/bin/echo", "hello"],
        capture_output=True,
        text=False,
        timeout=1.0,
    )

    assert output.returncode == 0
    assert b"hello" in output.stdout


def test_clear_sensitive_facts_rejects_non_string_keys_and_bad_objects() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        _clear_sensitive_facts({1: "x"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="JSON-safe"):
        _clear_sensitive_facts(object())  # type: ignore[arg-type]


def test_clear_sensitive_facts_supports_lists_and_nonfinite_numbers() -> None:
    sanitized = _clear_sensitive_facts(
        [{"token": "x", "values": [1, {"secret": "y"}]}, 1, 2.0]
    )

    assert sanitized == [
        {"token": "<redacted>", "values": [1, {"secret": "<redacted>"}]},
        1,
        2.0,
    ]

    with pytest.raises(TypeError, match="not include NaN"):
        _clear_sensitive_facts([float("nan")])


def test_auth_or_host_failure_classifier() -> None:
    assert _is_auth_or_host_failure("Permission denied")
    assert _is_auth_or_host_failure("Host key verification failed")
    assert _is_auth_or_host_failure("Could not resolve host foo")
    assert _is_auth_or_host_failure("Connection timed out while connecting")
    assert not _is_auth_or_host_failure("some other error")


def test_constructor_validates_backoff_and_defaults() -> None:
    with pytest.raises(ValueError, match="backoff must provide at least one delay"):
        SshClient(target="nancy", backoff=())

    client = SshClient(target="nancy")
    assert client._sleeper is SLEEPER
    with pytest.raises(ValueError, match="exceeds configured hard maximum"):
        SshClient(target="nancy", read_max_bytes=_READ_MAX_BYTES + 1)


def test_read_since_rejects_invalid_protocol_statuses() -> None:
    client, _, _ = _build_client(
        responses=[
            _FakeCompleted(
                0,
                _status_payload(
                    "ok",
                    next_offset="x",
                    eof="false",
                    payload=base64.b64encode(b"x").decode("ascii"),
                    bytes=1,
                ),
            ),
        ],
    )

    with pytest.raises(ValueError, match="invalid literal"):
        client.read_since("/tmp/run.log", 0)


def test_run_retries_non_transport_runner_errors() -> None:
    client, _, _ = _build_client(
        responses=[ValueError("boom")],
    )

    with pytest.raises(SshConnectionError, match="transport runner failed"):
        client._run_with_retries(["ssh", "nancy", "true"])


def test_offset_regression_is_protocol_error() -> None:
    bad = _status_payload(
        "ok",
        next_offset=0,
        eof="false",
        payload=base64.b64encode(b"x").decode("ascii"),
    )
    client, _, _ = _build_client(responses=[_FakeCompleted(0, bad)])
    with pytest.raises(SshProtocolError, match="offset regression"):
        client.read_since("/tmp/run.log", 10, max_bytes=1)


def test_read_since_rejects_invalid_base64_payload() -> None:
    client, _, _ = _build_client(
        responses=[
            _FakeCompleted(
                0, _status_payload("ok", next_offset=1, eof=False, payload="%", bytes=1)
            )
        ],
    )

    with pytest.raises(SshProtocolError, match="payload is not base64"):
        client.read_since("/tmp/run.log", 0)


def test_decode_text_non_utf8_bytes_is_protocol_error() -> None:
    with pytest.raises(SshProtocolError, match="not valid UTF-8"):
        _decode_text(b"\xff", "stdout")


def test_read_since_rejects_unknown_protocol_status() -> None:
    client, _, _ = _build_client(
        responses=[_FakeCompleted(0, _status_payload("weird"))],
    )
    with pytest.raises(SshProtocolError, match="unsupported status"):
        client.read_since("/tmp/run.log", 0, max_bytes=1)


def test_run_with_retries_exhausted_without_attempts_is_connection_error() -> None:
    client, _, _ = _build_client(responses=[_FakeCompleted(0, "ok")])
    client._attempts = 0

    with pytest.raises(SshConnectionError, match="ssh command exhausted retries"):
        client._run_with_retries(["ssh", "nancy", "true"])
