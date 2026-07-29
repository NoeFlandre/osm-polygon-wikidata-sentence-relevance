"""Contracts for Grid'5000 site discovery: default sites and remote probing.

These tests pin the public behaviour of :mod:`site_discovery`:

* the immutable, deterministic default site list;
* factual (read-only) probing of one frontend into a :class:`SiteProbe`;
* that transport, parsing and validation failures collapse to the same
  unreachable probe shape without leaking remote details.

Queue depth is observed for diagnostics only and never drives compatibility
or an ETA forecast.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from osm_polygon_sentence_relevance.operator.site_discovery import (
    DEFAULT_SITES,
    probe_site,
)
from osm_polygon_sentence_relevance.operator.sites import SiteProbe
from osm_polygon_sentence_relevance.operator.ssh import SshError

_IDLE_NODE = (
    '{"state":"Alive","jobs":0,"gpu_mem":80000,"gpu_compute_capability_major":8}\n'
)


def _install_probe_ssh(
    monkeypatch: pytest.MonkeyPatch,
    *,
    availability: str = "",
    managed: str = "1000\n0\n0\n",
    quota: str = "0 25000000 100000000\n",
    queue: str = "0",
    raise_on: str | None = None,
) -> list[str]:
    """Replace ``SshClient`` with a command-dispatching fake.

    Returns the recorded command list so a test may assert on the exact
    remote script that probing interpolated.
    """

    commands: list[str] = []

    class _FakeSsh:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, command: str) -> SimpleNamespace:
            commands.append(command)
            if raise_on is not None and raise_on in command:
                raise SshError(
                    "offline", category="transport", returncode=255, attempts=1
                )
            if "oarnodes -J" in command:
                return SimpleNamespace(stdout=availability)
            if "quota" in command:
                return SimpleNamespace(stdout=quota)
            if "oarstat" in command:
                return SimpleNamespace(stdout=queue)
            return SimpleNamespace(stdout=managed)

    monkeypatch.setattr(
        "osm_polygon_sentence_relevance.operator.site_discovery.SshClient", _FakeSsh
    )
    return commands


def test_default_sites_are_in_deterministic_order() -> None:
    assert DEFAULT_SITES == (
        "bordeaux",
        "grenoble",
        "lille",
        "louvain",
        "luxembourg",
        "lyon",
        "nancy",
        "nantes",
        "rennes",
        "sophia",
        "strasbourg",
        "toulouse",
    )


def test_probe_constructs_reachable_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = _install_probe_ssh(
        monkeypatch,
        availability=_IDLE_NODE,
        managed="1000\n0\n0\n",
        queue="3\n",
    )
    probe = probe_site("nancy", "a" * 20)
    assert probe == SiteProbe(
        "nancy",
        "nancy",
        True,
        80_000,
        (8, 0),
        1_024_000,
        3,
        True,
        False,
        False,
    )
    assert any("oarnodes -J" in command for command in commands)


def test_probe_without_run_id_omits_managed_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = _install_probe_ssh(
        monkeypatch, availability=_IDLE_NODE, managed="1000\n0\n0\n"
    )
    probe = probe_site("nancy")
    assert probe.has_managed_run is False
    assert probe.label_runtime_ready is False
    managed_commands = [command for command in commands if "df -Pk" in command]
    assert managed_commands
    assert "osm-polygon-operator" not in managed_commands[0]


def test_probe_with_valid_run_id_checks_managed_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "a" * 20
    commands = _install_probe_ssh(
        monkeypatch,
        availability=_IDLE_NODE,
        managed="1000\n1\n1\n",
    )
    probe = probe_site("nancy", run_id)
    assert probe.has_managed_run is True
    assert probe.label_runtime_ready is True
    managed_commands = [command for command in commands if "df -Pk" in command]
    assert managed_commands
    assert f"osm-polygon-operator/{run_id}" in managed_commands[0]


def test_probe_rejects_malformed_run_id() -> None:
    with pytest.raises(ValueError, match="twenty"):
        probe_site("nancy", "bad")


def test_probe_is_unreachable_on_malformed_three_line_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_ssh(monkeypatch, availability=_IDLE_NODE, managed="1000\n0\n")
    assert probe_site("nancy").reachable is False


def test_probe_is_unreachable_on_invalid_managed_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_ssh(monkeypatch, availability=_IDLE_NODE, managed="1000\n2\n0\n")
    assert probe_site("nancy").reachable is False


def test_probe_reports_no_gpu_on_empty_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_ssh(monkeypatch, availability="", managed="1000\n0\n0\n")
    probe = probe_site("nancy")
    assert probe.reachable is True
    assert probe.gpu_memory_mb == 0
    assert probe.cuda_capability is None
    assert probe.idle_compatible is False


def test_probe_selects_peak_compatible_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    availability = (
        '{"state":"Alive","jobs":0,"gpu_mem":80000,'
        '"gpu_compute_capability_major":8}\n'
        '{"state":"Alive","jobs":2,"gpu_mem":160000,'
        '"gpu_compute_capability_major":9}\n'
    )
    _install_probe_ssh(monkeypatch, availability=availability, managed="1000\n0\n0\n")
    probe = probe_site("nancy")
    assert probe.gpu_memory_mb == 160_000
    assert probe.cuda_capability == (9, 0)


def test_persistent_free_is_capped_by_soft_quota_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_ssh(
        monkeypatch,
        availability=_IDLE_NODE,
        managed="5000000\n0\n0\n",
        quota="24000000 25000000 100000000\n",
    )
    assert probe_site("nancy").persistent_free_bytes == 1_000_000 * 1024


def test_idle_compatible_reflects_factual_oar_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    busy = (
        '{"state":"Alive","jobs":1,"gpu_mem":80000,"gpu_compute_capability_major":8}\n'
    )
    _install_probe_ssh(monkeypatch, availability=busy, managed="1000\n0\n0\n")
    assert probe_site("nancy").idle_compatible is False
    _install_probe_ssh(monkeypatch, availability=_IDLE_NODE, managed="1000\n0\n0\n")
    assert probe_site("nancy").idle_compatible is True


def test_probe_detects_managed_run_without_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_ssh(monkeypatch, availability=_IDLE_NODE, managed="1000\n1\n0\n")
    probe = probe_site("nancy", "a" * 20)
    assert probe.has_managed_run is True
    assert probe.label_runtime_ready is False


def test_probe_detects_runtime_ready_without_managed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_ssh(monkeypatch, availability=_IDLE_NODE, managed="1000\n0\n1\n")
    probe = probe_site("nancy", "a" * 20)
    assert probe.has_managed_run is False
    assert probe.label_runtime_ready is True


def test_queue_depth_reports_waiting_and_held_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_ssh(
        monkeypatch, availability=_IDLE_NODE, managed="1000\n0\n0\n", queue="7\n"
    )
    assert probe_site("nancy").queued_jobs == 7


def test_queue_depth_returns_zero_on_ssh_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_ssh(
        monkeypatch,
        availability=_IDLE_NODE,
        managed="1000\n0\n0\n",
        raise_on="oarstat",
    )
    probe = probe_site("nancy")
    assert probe.reachable is True
    assert probe.queued_jobs == 0


def test_queue_depth_returns_zero_on_malformed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_ssh(
        monkeypatch,
        availability=_IDLE_NODE,
        managed="1000\n0\n0\n",
        queue="not-a-number\n",
    )
    probe = probe_site("nancy")
    assert probe.reachable is True
    assert probe.queued_jobs == 0


def test_transport_failure_produces_unreachable_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_ssh(monkeypatch, availability=_IDLE_NODE, raise_on="oarnodes -J")
    probe = probe_site("nancy")
    assert probe.reachable is False
    assert probe.gpu_memory_mb == 0
    assert probe.persistent_free_bytes == 0


def test_malformed_availability_json_produces_unreachable_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe_ssh(
        monkeypatch, availability="{not valid json", managed="1000\n0\n0\n"
    )
    assert probe_site("nancy").reachable is False


def test_public_failure_does_not_leak_remote_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "AKIA-DO-NOT-LEAK-TOKEN"
    _install_probe_ssh(
        monkeypatch,
        availability='{"state":"Alive","leak":' + secret + " broken",
        managed="1000\n0\n0\n",
    )
    probe = probe_site("nancy")
    assert probe.reachable is False
    assert secret not in repr(probe)
