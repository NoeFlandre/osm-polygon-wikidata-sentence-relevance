"""Unit tests for the low-level SSH/SCP transport and remote listing mechanics."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator.relay_transport import (
    RelayError,
    RemoteEntry,
    RemoteTransfer,
    list_remote_checkpoints,
    list_remote_dir,
    validate_safe_remote_path,
)


def test_validate_safe_remote_path_accepts_valid_absolute_paths() -> None:
    """Safe absolute remote paths are accepted and returned unchanged."""

    assert (
        validate_safe_remote_path("/home/u/label-work/checkpoints/batch-000001.parquet")
        == "/home/u/label-work/checkpoints/batch-000001.parquet"
    )


def test_validate_safe_remote_path_rejects_empty_and_non_string() -> None:
    """Empty strings and non-string inputs are rejected."""

    with pytest.raises(RelayError, match="non-empty string"):
        validate_safe_remote_path("")
    with pytest.raises(RelayError, match="non-empty string"):
        validate_safe_remote_path(None)  # type: ignore[arg-type]


def test_validate_safe_remote_path_rejects_surrounding_whitespace() -> None:
    """Paths with leading or trailing whitespace are rejected."""

    with pytest.raises(RelayError, match="surrounding whitespace"):
        validate_safe_remote_path(" /home/u/work")
    with pytest.raises(RelayError, match="surrounding whitespace"):
        validate_safe_remote_path("/home/u/work ")


def test_validate_safe_remote_path_blocks_all_unsafe_characters() -> None:
    """Every documented unsafe path character is rejected."""

    unsafe_chars = [
        " ",
        "\t",
        "\n",
        "\r",
        "\x00",
        '"',
        "'",
        "$",
        "`",
        ";",
        "&",
        "|",
        "<",
        ">",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "#",
        "?",
        ":",
        "!",
        "*",
        "\\",
        "~",
    ]
    for ch in unsafe_chars:
        path = f"/home/u/dir{ch}file"
        with pytest.raises(RelayError, match="unsafe characters"):
            validate_safe_remote_path(path)


def test_validate_safe_remote_path_rejects_traversal() -> None:
    """Dot and dot-dot path traversal components are refused."""

    with pytest.raises(RelayError, match="path traversal refused"):
        validate_safe_remote_path("/home/u/../etc/passwd")
    with pytest.raises(RelayError, match="path traversal refused"):
        validate_safe_remote_path("../escape")


def test_remote_transfer_validates_ssh_target() -> None:
    """SSH target must be a non-empty string without control characters."""

    empty_transfer = RemoteTransfer("")
    with pytest.raises(RelayError, match="ssh target must be a non-empty string"):
        empty_transfer.fetch("/home/u/file", Path("/tmp/out"))

    ctrl_transfer = RemoteTransfer("sophia\nmalicious")
    with pytest.raises(RelayError, match="ssh target contains control characters"):
        ctrl_transfer.fetch("/home/u/file", Path("/tmp/out"))


def test_remote_transfer_fetch_exact_argv_and_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fetch invokes scp with exact argv, shell=False, timeout, and sets 0600 mode."""

    executed_calls: list[dict[str, object]] = []

    def mock_run(
        argv: list[str],
        check: bool = False,
        shell: bool = True,
        timeout: int | None = None,
    ) -> SimpleNamespace:
        executed_calls.append(
            {"argv": argv, "check": check, "shell": shell, "timeout": timeout}
        )
        # Create the temporary file that scp would write to
        tmp_target = argv[-1]
        Path(tmp_target).write_text("data")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    transfer = RemoteTransfer("sophia")
    local_dst = tmp_path / "downloaded.txt"
    transfer.fetch("/home/u/work/data.txt", local_dst)

    assert len(executed_calls) == 1
    call = executed_calls[0]
    assert call["shell"] is False
    assert call["timeout"] == 120
    assert call["check"] is True
    argv = call["argv"]
    assert isinstance(argv, list)
    assert argv[:4] == ["scp", "-B", "-q", "-p"]
    assert argv[4] == "sophia:/home/u/work/data.txt"

    assert local_dst.exists()
    assert local_dst.read_text() == "data"
    assert (local_dst.stat().st_mode & 0o777) == 0o600


def test_remote_transfer_fetch_traversal_refused(tmp_path: Path) -> None:
    """Remote path traversal is refused in fetch."""

    transfer = RemoteTransfer("sophia")
    with pytest.raises(RelayError, match="traversal refused"):
        transfer.fetch("../escape", tmp_path / "out")


def test_remote_transfer_fetch_traversal_backup_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit traversal check inside fetch fires if validator is bypassed."""

    monkeypatch.setattr(
        "osm_polygon_sentence_relevance.operator.relay_transport.validate_safe_remote_path",
        lambda p: p,
    )
    transfer = RemoteTransfer("sophia")
    with pytest.raises(RelayError, match="remote traversal refused"):
        transfer.fetch("/home/u/../escape", tmp_path / "out")


def test_remote_transfer_fetch_cleanup_on_process_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Temp file is unlinked on CalledProcessError."""

    def mock_run(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ["scp"])

    monkeypatch.setattr(subprocess, "run", mock_run)

    transfer = RemoteTransfer("sophia")
    local_dst = tmp_path / "out.txt"
    with pytest.raises(RelayError, match="scp fetch failed for /home/u/work/data.txt"):
        transfer.fetch("/home/u/work/data.txt", local_dst)

    # Temporary files should be cleaned up
    assert list(tmp_path.glob(".*")) == []


def test_remote_transfer_fetch_cleanup_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Temp file is unlinked on TimeoutExpired."""

    def mock_run(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(["scp"], 120)

    monkeypatch.setattr(subprocess, "run", mock_run)

    transfer = RemoteTransfer("sophia")
    local_dst = tmp_path / "out.txt"
    with pytest.raises(RelayError, match="scp fetch failed for /home/u/work/data.txt"):
        transfer.fetch("/home/u/work/data.txt", local_dst)

    assert list(tmp_path.glob(".*")) == []


def test_remote_transfer_fetch_symlink_destination_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fetch refuses symlink target after fetch."""

    def mock_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        tmp_target = Path(argv[-1])
        tmp_target.write_text("data")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    real_target = tmp_path / "real_file"
    real_target.write_text("real")
    symlink_dst = tmp_path / "symlink_dst"
    symlink_dst.symlink_to(real_target)

    transfer = RemoteTransfer("sophia")

    def mock_replace(src: Path | str, dst: Path | str) -> None:
        Path(dst).unlink(missing_ok=True)
        os.symlink(real_target, dst)

    monkeypatch.setattr(os, "replace", mock_replace)

    with pytest.raises(RelayError, match="refusing to follow a symlink"):
        transfer.fetch("/home/u/work/data.txt", symlink_dst)


def test_remote_transfer_push_validates_source_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Push checks local source is regular file and path is safe."""

    transfer = RemoteTransfer("sophia")
    missing_file = tmp_path / "missing.txt"
    with pytest.raises(RelayError, match="refusing to push non-regular file"):
        transfer.push(missing_file, "/home/u/work/remote.txt")

    dir_path = tmp_path / "dir"
    dir_path.mkdir()
    with pytest.raises(RelayError, match="refusing to push non-regular file"):
        transfer.push(dir_path, "/home/u/work/remote.txt")

    file_path = tmp_path / "real.txt"
    file_path.write_text("hello")
    symlink_path = tmp_path / "link.txt"
    symlink_path.symlink_to(file_path)
    with pytest.raises(RelayError, match="refusing to push non-regular file"):
        transfer.push(symlink_path, "/home/u/work/remote.txt")


def test_remote_transfer_push_exact_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Push invokes scp with exact argv, shell=False, and timeout."""

    executed_calls: list[dict[str, object]] = []

    def mock_run(
        argv: list[str],
        check: bool = False,
        shell: bool = True,
        timeout: int | None = None,
    ) -> SimpleNamespace:
        executed_calls.append(
            {"argv": argv, "check": check, "shell": shell, "timeout": timeout}
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    file_path = tmp_path / "local.txt"
    file_path.write_text("content")

    transfer = RemoteTransfer("sophia")
    transfer.push(file_path, "/home/u/work/remote.txt")

    assert len(executed_calls) == 1
    call = executed_calls[0]
    assert call["shell"] is False
    assert call["timeout"] == 120
    assert call["check"] is True
    assert call["argv"] == [
        "scp",
        "-B",
        "-q",
        "-p",
        str(file_path),
        "sophia:/home/u/work/remote.txt",
    ]


def test_remote_transfer_ssh_mkdir_0700_exact_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ssh_mkdir_0700 uses install -d -m 0700 with exact argv."""

    executed_calls: list[dict[str, object]] = []

    def mock_run(
        argv: list[str],
        check: bool = False,
        shell: bool = True,
        timeout: int | None = None,
    ) -> SimpleNamespace:
        executed_calls.append(
            {"argv": argv, "check": check, "shell": shell, "timeout": timeout}
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    transfer = RemoteTransfer("sophia")
    transfer.ssh_mkdir_0700("/home/u/work/dir")

    assert len(executed_calls) == 1
    call = executed_calls[0]
    assert call["shell"] is False
    assert call["timeout"] == 60
    assert call["check"] is True
    assert call["argv"] == [
        "ssh",
        "sophia",
        "install -d -m 0700 /home/u/work/dir",
    ]


def test_remote_transfer_ssh_chmod_exact_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ssh_chmod uses chmod -R <mode:o> with exact argv."""

    executed_calls: list[dict[str, object]] = []

    def mock_run(
        argv: list[str],
        check: bool = False,
        shell: bool = True,
        timeout: int | None = None,
    ) -> SimpleNamespace:
        executed_calls.append(
            {"argv": argv, "check": check, "shell": shell, "timeout": timeout}
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    transfer = RemoteTransfer("sophia")
    transfer.ssh_chmod("/home/u/work/dir", 0o700)

    assert len(executed_calls) == 1
    call = executed_calls[0]
    assert call["shell"] is False
    assert call["timeout"] == 60
    assert call["check"] is True
    assert call["argv"] == [
        "ssh",
        "sophia",
        "chmod -R 700 /home/u/work/dir",
    ]


def test_remote_transfer_ssh_atomic_rename_exact_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ssh_atomic_rename uses exact shell-script payload via ssh."""

    executed_calls: list[dict[str, object]] = []

    def mock_run(
        argv: list[str],
        check: bool = False,
        shell: bool = True,
        timeout: int | None = None,
    ) -> SimpleNamespace:
        executed_calls.append(
            {"argv": argv, "check": check, "shell": shell, "timeout": timeout}
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    transfer = RemoteTransfer("sophia")
    transfer.ssh_atomic_rename("/home/u/work/staging", "/home/u/work/final")

    assert len(executed_calls) == 1
    call = executed_calls[0]
    assert call["shell"] is False
    assert call["timeout"] == 60
    assert call["check"] is True
    assert call["argv"] == [
        "ssh",
        "sophia",
        "if [ -e /home/u/work/final ]; then rmdir -- /home/u/work/final || exit 1; fi; mv -- /home/u/work/staging /home/u/work/final",
    ]


def test_list_remote_dir_exact_argv_and_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """list_remote_dir issues exact find argv and parses f/d/l/other entries omitting blank and dot entries."""

    executed_calls: list[dict[str, object]] = []

    stdout_data = "f\tfile1.parquet\nd\tsubdir\nl\tlink1\np\tfifo1\nf\t.\nd\t..\n\n"

    def mock_run(
        argv: list[str],
        check: bool = False,
        shell: bool = True,
        capture_output: bool = False,
        text: bool = False,
        timeout: int | None = None,
    ) -> SimpleNamespace:
        executed_calls.append(
            {
                "argv": argv,
                "check": check,
                "shell": shell,
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
            }
        )
        return SimpleNamespace(returncode=0, stdout=stdout_data)

    monkeypatch.setattr(subprocess, "run", mock_run)

    entries = list_remote_dir("sophia", "/home/u/work")

    assert len(executed_calls) == 1
    call = executed_calls[0]
    assert call["shell"] is False
    assert call["check"] is True
    assert call["timeout"] == 60
    assert call["argv"] == [
        "ssh",
        "sophia",
        "find",
        "/home/u/work",
        "-mindepth",
        "1",
        "-maxdepth",
        "1",
        "-printf",
        "%y\t%f\n",
    ]

    assert entries == [
        RemoteEntry(name="file1.parquet", kind="file"),
        RemoteEntry(name="subdir", kind="dir"),
        RemoteEntry(name="link1", kind="symlink"),
        RemoteEntry(name="fifo1", kind="other"),
    ]


def test_list_remote_dir_traversal_backup_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit traversal check inside list_remote_dir fires if validator is bypassed."""

    monkeypatch.setattr(
        "osm_polygon_sentence_relevance.operator.relay_transport.validate_safe_remote_path",
        lambda p: p,
    )
    with pytest.raises(RelayError, match="refusing to traverse remote checkpoint root"):
        list_remote_dir("sophia", "/home/u/../escape")


def test_list_remote_checkpoints_path_composition_and_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """list_remote_checkpoints appends /checkpoints and parses entries."""

    executed_calls: list[dict[str, object]] = []

    stdout_data = "f\tbatch-000000.parquet\nf\tbatch-000000.json\np\tfifo\n"

    def mock_run(
        argv: list[str],
        check: bool = True,
        shell: bool = True,
        capture_output: bool = False,
        text: bool = False,
        timeout: int | None = None,
    ) -> SimpleNamespace:
        executed_calls.append(
            {
                "argv": argv,
                "check": check,
                "shell": shell,
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
            }
        )
        return SimpleNamespace(returncode=0, stdout=stdout_data)

    monkeypatch.setattr(subprocess, "run", mock_run)

    entries = list_remote_checkpoints("sophia", "/home/u/work/")

    assert len(executed_calls) == 1
    call = executed_calls[0]
    assert call["shell"] is False
    assert call["check"] is False
    assert call["timeout"] == 60
    assert call["argv"] == [
        "ssh",
        "sophia",
        "find",
        "/home/u/work/checkpoints",
        "-mindepth",
        "1",
        "-maxdepth",
        "1",
        "-printf",
        "%y\t%f\n",
    ]

    assert entries == [
        RemoteEntry(name="batch-000000.parquet", kind="file"),
        RemoteEntry(name="batch-000000.json", kind="file"),
        RemoteEntry(name="fifo", kind="other"),
    ]


def test_list_remote_checkpoints_missing_directory_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the checkpoints directory is absent (returncode != 0), empty list is returned."""

    def mock_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    entries = list_remote_checkpoints("sophia", "/home/u/work")
    assert entries == []


def test_list_remote_checkpoints_subprocess_failure_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CalledProcessError in subprocess.run is wrapped in RelayError."""

    def mock_run(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(255, ["ssh"])

    monkeypatch.setattr(subprocess, "run", mock_run)

    with pytest.raises(RelayError, match="listing remote checkpoints failed"):
        list_remote_checkpoints("sophia", "/home/u/work")
