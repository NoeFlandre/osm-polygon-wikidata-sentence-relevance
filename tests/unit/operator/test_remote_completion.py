"""Unit tests for remote completion evidence, publication results, and status marking."""

from __future__ import annotations

import json
import shlex
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator import remote_completion
from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout


class _FakeSsh:
    def __init__(self, output: str = "") -> None:
        self.output = output
        self.commands: list[str] = []

    def run(self, command: str) -> SimpleNamespace:
        self.commands.append(command)
        return SimpleNamespace(stdout=self.output)


def test_remote_exit_code_reads_exact_path_and_parses_integer() -> None:
    ssh = _FakeSsh("0\n")
    layout = RemoteLayout(PurePosixPath("/r"))

    code = remote_completion.remote_exit_code(ssh, layout, 42, "build.exit_code")  # type: ignore[arg-type]

    assert code == 0
    assert len(ssh.commands) == 1
    assert (
        ssh.commands[0]
        == "test -f /r/logs/42/build.exit_code && cat /r/logs/42/build.exit_code"
    )


@pytest.mark.parametrize(
    ("exit_text", "expected_code"), [("0\n", 0), ("1\n", 1), ("130\n", 130)]
)
def test_remote_exit_code_valid_integers(exit_text: str, expected_code: int) -> None:
    ssh = _FakeSsh(exit_text)
    layout = RemoteLayout(PurePosixPath("/r"))

    assert remote_completion.remote_exit_code(ssh, layout, 1, "exit") == expected_code  # type: ignore[arg-type]


def test_remote_exit_code_with_text_attribute() -> None:
    class ResultWithText:
        text = "0\n"

    class SshWithText:
        def run(self, _cmd: str) -> ResultWithText:
            return ResultWithText()

    layout = RemoteLayout(PurePosixPath("/r"))
    assert remote_completion.remote_exit_code(SshWithText(), layout, 1, "exit") == 0  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_text", ["bad\n", "", "   ", "12.3", "none"])
def test_remote_exit_code_invalid_content_raises_runtime_error(
    invalid_text: str,
) -> None:
    ssh = _FakeSsh(invalid_text)
    layout = RemoteLayout(PurePosixPath("/r"))

    with pytest.raises(
        RuntimeError, match="remote payload exit status is invalid"
    ) as exc_info:
        remote_completion.remote_exit_code(ssh, layout, 1, "exit")  # type: ignore[arg-type]

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_assert_remote_exit_zero_passes_and_fails() -> None:
    layout = RemoteLayout(PurePosixPath("/r"))

    remote_completion.assert_remote_exit_zero(_FakeSsh("0\n"), layout, 1, "exit")  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="remote payload returned non-zero status"):
        remote_completion.assert_remote_exit_zero(_FakeSsh("2\n"), layout, 1, "exit")  # type: ignore[arg-type]


def test_publish_split_constructs_exact_quoted_command_and_returns_commit() -> None:
    ssh = _FakeSsh("abcdef123456\n")
    layout = RemoteLayout(PurePosixPath("/r"))
    hostile_output_dir = PurePosixPath("/out dir; rm -rf /; $(whoami); `id`; *")
    hostile_dataset_id = "owner/dataset; $(touch /tmp/bad); `echo secret`; *"

    commit = remote_completion.publish_split(
        ssh,  # type: ignore[arg-type]
        layout,
        hostile_output_dir,
        hostile_dataset_id,
    )

    assert commit == "abcdef123456"
    assert len(ssh.commands) == 1
    raw_command = ssh.commands[0]
    tokens = shlex.split(raw_command)
    assert len(tokens) == 3
    assert tokens[0] == "/r/repo/.venv/bin/python"
    assert tokens[1] == "-c"

    python_code = tokens[2]
    assert repr(str(hostile_output_dir)) in python_code
    assert repr(hostile_dataset_id) in python_code
    assert (
        "from osm_polygon_sentence_relevance.publishing import publish_export_directory"
        in python_code
    )
    assert "publish_export_directory" in python_code
    assert "target_revision='main'" in python_code


@pytest.mark.parametrize("short_output", ["123456", "abc", "", "   "])
def test_publish_split_short_output_raises_runtime_error(short_output: str) -> None:
    ssh = _FakeSsh(short_output)
    layout = RemoteLayout(PurePosixPath("/r"))

    with pytest.raises(
        RuntimeError, match="Hugging Face publication did not return a commit"
    ):
        remote_completion.publish_split(
            ssh, layout, PurePosixPath("/out"), "owner/data"
        )  # type: ignore[arg-type]


def test_publish_label_constructs_exact_quoted_command_and_returns_commit() -> None:
    ssh = _FakeSsh("c" * 40 + "\n")
    layout = RemoteLayout(PurePosixPath("/r"))
    output_dir = PurePosixPath("/label output; $(touch bad)")
    dataset_id = "owner/labels; `echo bad`"

    commit = remote_completion.publish_label(
        ssh,  # type: ignore[arg-type]
        layout,
        output_dir,
        dataset_id,
    )

    assert commit == "c" * 40
    tokens = shlex.split(ssh.commands[0])
    assert tokens[:2] == ["/r/repo/.venv/bin/python", "-c"]
    assert repr(str(output_dir)) in tokens[2]
    assert repr(dataset_id) in tokens[2]
    assert (
        "from osm_polygon_sentence_relevance.labeling.publication import "
        "publish_labeled_dataset"
    ) in tokens[2]
    assert "target_revision='main'" in tokens[2]


def test_label_publication_commit_selects_latest_valid_record() -> None:
    older_commit = "a" * 40
    latest_commit = "b" * 40
    stdout = (
        "noise line\n"
        "not json\n"
        '{"other": 1}\n'
        f'{{"commit_id": "{older_commit}"}}\n'
        '{"commit_id": "malformed_short"}\n'
        "even more noise\n"
        f'{{"commit_id": "{latest_commit}"}}\n'
        "trailing noise line\n"
    )
    ssh = _FakeSsh(stdout)
    layout = RemoteLayout(PurePosixPath("/r"))

    commit = remote_completion.label_publication_commit(ssh, layout, 99)  # type: ignore[arg-type]

    assert commit == latest_commit
    assert (
        ssh.commands[0]
        == "test -f /r/logs/99/labeling.stdout.log && cat /r/logs/99/labeling.stdout.log"
    )


@pytest.mark.parametrize(
    "invalid_payload",
    [
        '{"commit_id":"' + "A" * 40 + '"}',  # Uppercase
        '{"commit_id":"' + "a" * 39 + '"}',  # Short
        '{"commit_id":"' + "g" * 40 + '"}',  # Non-hex
        '{"commit_id":" ' + "a" * 40 + ' "}',  # Padded
        '{"commit_id":12345}',  # Non-string
        "not json at all",  # Non-JSON
        "[]",  # Non-mapping
        '{"commit_id":null}',  # Null
    ],
)
def test_label_publication_commit_rejects_invalid_commits(invalid_payload: str) -> None:
    ssh = _FakeSsh(invalid_payload)
    layout = RemoteLayout(PurePosixPath("/r"))

    with pytest.raises(
        RuntimeError, match="label publication did not report an immutable Hub commit"
    ):
        remote_completion.label_publication_commit(ssh, layout, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("status", ["active", "complete", "failed"])
def test_mark_remote_status_writes_exact_command(status: str) -> None:
    ssh = _FakeSsh()
    layout = RemoteLayout(PurePosixPath("/r"))

    remote_completion.mark_remote_status(ssh, layout, status)  # type: ignore[arg-type]

    assert len(ssh.commands) == 1
    expected_json = json.dumps(
        {"schema_version": 1, "status": status},
        sort_keys=True,
        separators=(",", ":"),
    )
    marker_path = str(layout.root / ".operator-managed.json")
    expected_command = (
        "printf '%s\\n' "
        + shlex.quote(expected_json)
        + f" > {shlex.quote(marker_path)} && chmod 0600 {shlex.quote(marker_path)}"
    )
    assert ssh.commands[0] == expected_command


def test_mark_remote_status_invalid_status_raises_without_ssh_call() -> None:
    ssh = _FakeSsh()
    layout = RemoteLayout(PurePosixPath("/r"))

    with pytest.raises(ValueError, match="invalid managed status"):
        remote_completion.mark_remote_status(ssh, layout, "unknown")  # type: ignore[arg-type]

    assert len(ssh.commands) == 0
