"""Strict parsing of Grid'5000 home quota facts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from osm_polygon_sentence_relevance.operator.ssh import SshClient

#: Read-only home-quota acquisition. The nested timeout keeps a broken or
#: unavailable quota service from blocking a resume controller indefinitely.
#: ``set +e`` keeps rc=1 (over quota) from aborting the command; rc>1 still
#: surfaces as a transport failure.
_HOME_QUOTA_COMMAND: Final[str] = (
    "set +e; quota_output=$(timeout 15s quota 2>&1); quota_rc=$?; set -e; "
    'if [ "$quota_rc" -gt 1 ]; then exit "$quota_rc"; fi; '
    "printf '%s\\n' \"$quota_output\""
)

_QUOTA_ROW = re.compile(
    r"^\s*(?P<used>[0-9]+)\*?\s+"
    r"(?P<soft>[0-9]+)\s+"
    r"(?P<hard>[0-9]+)(?:\s|$)"
)


class QuotaError(RuntimeError):
    """Grid'5000 did not expose a usable home quota."""


@dataclass(frozen=True, slots=True)
class QuotaUsage:
    """Byte-denominated home quota facts."""

    used_bytes: int
    soft_limit_bytes: int
    hard_limit_bytes: int

    @property
    def soft_headroom_bytes(self) -> int:
        """Return writable headroom that stays within the soft quota."""

        return max(0, self.soft_limit_bytes - self.used_bytes)

    @property
    def soft_limit_exceeded(self) -> bool:
        """Return whether current usage is already above the soft limit."""

        return self.used_bytes > self.soft_limit_bytes


def parse_quota_output(output: str) -> QuotaUsage:
    """Parse the first quota data row emitted by Grid'5000's ``quota``."""

    for line in output.splitlines():
        match = _QUOTA_ROW.match(line)
        if match is None:
            continue
        used_kib = int(match.group("used"))
        soft_kib = int(match.group("soft"))
        hard_kib = int(match.group("hard"))
        if soft_kib <= 0 or hard_kib < soft_kib:
            raise QuotaError("home quota limits are invalid")
        return QuotaUsage(
            used_bytes=used_kib * 1024,
            soft_limit_bytes=soft_kib * 1024,
            hard_limit_bytes=hard_kib * 1024,
        )
    raise QuotaError("home quota output has no usable data row")


def read_home_quota(ssh: SshClient) -> QuotaUsage:
    """Read the current home quota over SSH, tolerating over-quota rc=1."""

    return parse_quota_output(ssh.run(_HOME_QUOTA_COMMAND).stdout)


__all__ = ["QuotaError", "QuotaUsage", "parse_quota_output", "read_home_quota"]
