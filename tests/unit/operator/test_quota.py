"""Remote Grid'5000 home-quota parsing contracts."""

from __future__ import annotations

import pytest

from osm_polygon_sentence_relevance.operator.quota import (
    _HOME_QUOTA_COMMAND,
    QuotaError,
    parse_quota_output,
)


def test_home_quota_command_has_a_bounded_remote_timeout() -> None:
    assert "quota_output=$(timeout -k 1s 5s quota 2>&1)" in _HOME_QUOTA_COMMAND
    assert 'timeout -k 1s 10s du -sk -- "$HOME"' in _HOME_QUOTA_COMMAND
    assert "25000000 100000000" in _HOME_QUOTA_COMMAND
    assert 'if [ "$quota_rc" -gt 1 ]; then exit "$quota_rc"; fi;' in (
        _HOME_QUOTA_COMMAND
    )


def test_parse_quota_output_uses_soft_limit_headroom() -> None:
    output = """\
Disk quotas for user user (uid 1):
     Filesystem  blocks   quota   limit   grace   files quota limit grace
nfs:/export/home
                37262328* 25000000 100000000 5days 124118 0 10000000
"""
    quota = parse_quota_output(output)
    assert quota.used_bytes == 37_262_328 * 1024
    assert quota.soft_limit_bytes == 25_000_000 * 1024
    assert quota.hard_limit_bytes == 100_000_000 * 1024
    assert quota.soft_headroom_bytes == 0
    assert quota.soft_limit_exceeded


def test_parse_quota_output_reports_positive_headroom() -> None:
    quota = parse_quota_output(" 1000 25000000 100000000 7days 3 0 100\n")
    assert quota.soft_headroom_bytes == 24_999_000 * 1024
    assert not quota.soft_limit_exceeded


@pytest.mark.parametrize(
    "output",
    [
        "",
        "Filesystem blocks quota limit\n",
        " 1000 0 100000\n",
        " 1000 25000000 20000000\n",
        " 1000M 25000000 100000000\n",
    ],
)
def test_parse_quota_output_rejects_missing_or_invalid_limits(output: str) -> None:
    with pytest.raises(QuotaError):
        parse_quota_output(output)
