"""Low-level SSH/SCP transport and remote-directory listing mechanics.

This module owns low-level transport operations (SCP fetch/push, SSH mkdir/chmod/rename)
and safe remote filesystem listing for the operator relay pipeline.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

RELAY_DIR_MODE: int = 0o700
FILE_MODE: int = 0o600


class RelayError(RuntimeError):
    """A relay operation was unsafe and must abort."""


#: Characters that must never appear in a remote or local path the relay
#: composes into a shell command.
_UNSAFE_PATH_CHARS: frozenset[str] = frozenset(
    {
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
    }
)


def validate_safe_remote_path(path: str) -> str:
    """Refuse paths containing characters that could break a shell command.

    The relay invokes ``ssh``/``scp`` via subprocess argv lists; this
    validator is the guard for any path that ends up formatted into a remote
    ``printf`` or command argument.
    """

    if not isinstance(path, str) or not path:
        raise RelayError("path must be a non-empty string")
    if path != path.strip():
        raise RelayError(f"path has surrounding whitespace: {path!r}")
    if any(ch in _UNSAFE_PATH_CHARS for ch in path):
        raise RelayError(f"path contains unsafe characters: {path!r}")
    parts = Path(path).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise RelayError(f"path traversal refused: {path!r}")
    return path


@dataclass(frozen=True, slots=True)
class RemoteTransfer:
    """Byte-accurate transfer of one file between the Mac and a frontend.

    The real implementation shells out to OpenSSH ``scp`` with explicit,
    validated, non-shell-interpolated arguments. The transfer refuses paths
    that contain shell metacharacters, traverses no symlinks, and writes to
    a freshly-created destination with mode ``0o600``.
    """

    ssh_target: str

    def fetch(self, remote_path: str, local_path: Path) -> None:
        """Retrieve ``remote_path`` to ``local_path`` (overwriting)."""

        validate_safe_remote_path(remote_path)
        if ".." in Path(remote_path).parts:
            raise RelayError(f"remote traversal refused: {remote_path!r}")
        if not isinstance(self.ssh_target, str) or not self.ssh_target:
            raise RelayError("ssh target must be a non-empty string")
        if "\n" in self.ssh_target or "\x00" in self.ssh_target:
            raise RelayError("ssh target contains control characters")
        local_path.parent.mkdir(parents=True, exist_ok=True, mode=RELAY_DIR_MODE)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{local_path.name}.", suffix=".tmp", dir=local_path.parent
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            subprocess.run(
                [
                    "scp",
                    "-B",
                    "-q",
                    "-p",
                    f"{self.ssh_target}:{remote_path}",
                    str(tmp_path),
                ],
                check=True,
                shell=False,
                timeout=120,
            )
            os.chmod(tmp_path, FILE_MODE)
            os.replace(tmp_path, local_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            tmp_path.unlink(missing_ok=True)
            raise RelayError(f"scp fetch failed for {remote_path}") from exc
        if local_path.is_symlink():
            local_path.unlink(missing_ok=True)
            raise RelayError(f"refusing to follow a symlink at {local_path}")

    def push(self, local_path: Path, remote_path: str) -> None:
        """Send ``local_path`` to ``remote_path`` (overwriting)."""

        if not local_path.is_file() or local_path.is_symlink():
            raise RelayError(f"refusing to push non-regular file: {local_path}")
        validate_safe_remote_path(remote_path)
        subprocess.run(
            [
                "scp",
                "-B",
                "-q",
                "-p",
                str(local_path),
                f"{self.ssh_target}:{remote_path}",
            ],
            check=True,
            shell=False,
            timeout=120,
        )

    def ssh_mkdir_0700(self, remote_path: str) -> None:
        """Create ``remote_path`` on the destination with mode ``0700``.

        Uses ``install -d -m 0700`` so the mode is set atomically.
        """

        validate_safe_remote_path(remote_path)
        subprocess.run(
            [
                "ssh",
                self.ssh_target,
                f"install -d -m 0700 {remote_path}",
            ],
            check=True,
            shell=False,
            timeout=60,
        )

    def ssh_chmod(self, remote_path: str, mode: int) -> None:
        """Recursively chmod a remote path."""

        validate_safe_remote_path(remote_path)
        subprocess.run(
            [
                "ssh",
                self.ssh_target,
                f"chmod -R {mode:o} {remote_path}",
            ],
            check=True,
            shell=False,
            timeout=60,
        )

    def ssh_atomic_rename(self, src: str, dst: str) -> None:
        """Atomic rename on the destination. Refuses non-empty ``dst``."""

        validate_safe_remote_path(src)
        validate_safe_remote_path(dst)
        subprocess.run(
            [
                "ssh",
                self.ssh_target,
                f"if [ -e {dst} ]; then rmdir -- {dst} || exit 1; fi; "
                f"mv -- {src} {dst}",
            ],
            check=True,
            shell=False,
            timeout=60,
        )


@dataclass(frozen=True, slots=True)
class RemoteEntry:
    """One entry inside a remote directory."""

    name: str
    kind: str  # "file" | "dir" | "symlink" | "other"


def _parse_find_output(stdout: str) -> list[RemoteEntry]:
    """Parse ``find -printf "%y\\t%f\\n"`` output into RemoteEntry instances."""

    entries: list[RemoteEntry] = []
    for raw in stdout.splitlines():
        kind, _, name = raw.partition("\t")
        name = name.strip()
        if not name or name in {".", ".."}:
            continue
        kind_token = kind.strip()
        if kind_token == "f":
            entries.append(RemoteEntry(name, "file"))
        elif kind_token == "d":
            entries.append(RemoteEntry(name, "dir"))
        elif kind_token == "l":
            entries.append(RemoteEntry(name, "symlink"))
        else:
            entries.append(RemoteEntry(name, "other"))
    return entries


def list_remote_dir(ssh_target: str, remote_dir: str) -> list[RemoteEntry]:
    """Read a remote directory's non-recursive entries safely."""

    validate_safe_remote_path(remote_dir)
    if ".." in Path(remote_dir).parts:
        raise RelayError("refusing to traverse remote checkpoint root")
    proc = subprocess.run(
        [
            "ssh",
            ssh_target,
            "find",
            remote_dir,
            "-mindepth",
            "1",
            "-maxdepth",
            "1",
            "-printf",
            "%y\t%f\n",
        ],
        check=True,
        shell=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return _parse_find_output(proc.stdout)


def list_remote_checkpoints(ssh_target: str, remote_root: str) -> list[RemoteEntry]:
    """List the ``checkpoints/`` subdirectory of a remote label-work root.

    Returns an empty list when the directory is absent (fresh root with no
    batches yet). Refuses symlinks, unexpected entries, and traversal.
    """

    validate_safe_remote_path(remote_root)
    remote_dir = f"{remote_root.rstrip('/')}/checkpoints"
    try:
        proc = subprocess.run(
            [
                "ssh",
                ssh_target,
                "find",
                remote_dir,
                "-mindepth",
                "1",
                "-maxdepth",
                "1",
                "-printf",
                "%y\t%f\n",
            ],
            check=False,
            shell=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.CalledProcessError as exc:
        raise RelayError(f"listing remote checkpoints failed: {exc}") from exc
    if proc.returncode != 0:
        # The checkpoints dir is missing: that's a fresh root, return empty.
        return []
    return _parse_find_output(proc.stdout)


__all__ = [
    "FILE_MODE",
    "RELAY_DIR_MODE",
    "RelayError",
    "RemoteEntry",
    "RemoteTransfer",
    "list_remote_checkpoints",
    "list_remote_dir",
    "validate_safe_remote_path",
]
