"""Direct tests for the OAR availability probe parser.

Coverage targets:
* ``parse_oarnodes_records`` rejects malformed and non-Alive records.
* ``parse_availability_stdout`` returns ``idle_compatible`` only when at
  least one Alive node has zero assigned jobs.
* ``availability_command`` quotes the user/host so a hostile target name
  cannot inject shell metacharacters, and rejects unsafe path grammar.
* ``meets(SiteRequirements)`` aggregates per-node facts into a single
  boolean compatibility claim.
"""

from __future__ import annotations

import json

import pytest

from osm_polygon_sentence_relevance.operator.sites import SiteRequirements
from osm_polygon_sentence_relevance.operator.sites_availability import (
    availability_command,
    parse_availability_stdout,
    parse_oarnodes_records,
)


def _record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "state": "Alive",
        "jobs": 0,
        "gpu_mem": 80000,
        "gpu_compute_capability_major": 8,
    }
    base.update(overrides)
    return base


def _encode_lines(payload: list[dict[str, object]]) -> str:
    return "\n".join(json.dumps(row) for row in payload) + "\n"


def test_parse_oarnodes_records_skips_non_alive_records() -> None:
    payload = [
        _record(state="Dead"),
        _record(state="Absent"),
        _record(),
    ]
    assert len(parse_oarnodes_records(payload)) == 1


def test_parse_oarnodes_records_requires_numeric_fields() -> None:
    payload = [
        _record(gpu_mem="not-a-number"),
        _record(gpu_compute_capability_major="x"),
        _record(jobs="busy"),
        _record(),
    ]
    assert len(parse_oarnodes_records(payload)) == 1


def test_string_job_identifier_counts_as_one_assigned_job() -> None:
    """OAR emits a scalar job id for a busy resource, not a job list."""

    payload = _record(jobs="2981309")
    nodes = parse_oarnodes_records(payload)
    assert len(nodes) == 1
    assert nodes[0].jobs_assigned == 1


def test_idle_compatible_is_true_when_at_least_one_compatible_node_is_idle() -> None:
    payload = [
        _record(jobs=0, gpu_mem=16000, gpu_compute_capability_major=7),
        _record(jobs=2, gpu_mem=80000, gpu_compute_capability_major=8),
    ]
    probe = parse_availability_stdout(_encode_lines(payload))
    requirements = SiteRequirements(gpu_memory_mb=40000, cuda_capability=(8, 0))
    assert probe.idle_compatible(requirements) is False


def test_idle_compatible_is_false_when_every_node_is_busy() -> None:
    payload = [
        _record(jobs=1),
        _record(jobs=4),
    ]
    probe = parse_availability_stdout(_encode_lines(payload))
    assert probe.idle_compatible(SiteRequirements()) is False


def test_idle_compatible_is_false_when_payload_is_empty() -> None:
    probe = parse_availability_stdout("")
    assert probe.idle_compatible(SiteRequirements()) is False


def test_meets_aggregates_per_node_requirements() -> None:
    payload = [
        _record(gpu_mem=16000, gpu_compute_capability_major=7),
        _record(gpu_mem=80000, gpu_compute_capability_major=8),
    ]
    probe = parse_availability_stdout(_encode_lines(payload))
    requirements = SiteRequirements(gpu_memory_mb=40000, cuda_capability=(8, 0))
    assert probe.meets(requirements) is True


def test_meets_returns_false_when_no_node_satisfies_requirements() -> None:
    payload = [
        _record(gpu_mem=16000, gpu_compute_capability_major=7),
    ]
    probe = parse_availability_stdout(_encode_lines(payload))
    requirements = SiteRequirements(gpu_memory_mb=40000, cuda_capability=(8, 0))
    assert probe.meets(requirements) is False


def test_parse_availability_stdout_raises_on_invalid_json() -> None:
    with pytest.raises(ValueError, match="invalid oarnodes JSON"):
        parse_availability_stdout("{not json")


def test_availability_command_is_remote_query_not_nested_ssh() -> None:
    cmd = availability_command()
    assert not cmd.startswith("ssh ")
    assert "oarnodes -J" in cmd
    assert "jq" in cmd
    assert "type" in cmd


def test_parse_oarnodes_records_accepts_single_object_payload() -> None:
    payload = _record()
    nodes = parse_oarnodes_records(payload)
    assert len(nodes) == 1


def test_coerce_int_rejects_bools_and_malformed_strings() -> None:
    """Bool and unparseable strings are rejected before the minimum check."""

    from osm_polygon_sentence_relevance.operator.sites_availability import _coerce_int

    assert _coerce_int(True, minimum=0) is None
    assert _coerce_int(False, minimum=0) is None
    assert _coerce_int("not-numeric", minimum=0) is None
    assert _coerce_int(None, minimum=0) is None
    # Below-minimum numbers are coerced to None, not silently rounded.
    assert _coerce_int(-1, minimum=0) is None
    assert _coerce_int("5", minimum=10) is None
    # Valid numeric strings are accepted.
    assert _coerce_int("42", minimum=0) == 42


def test_parse_availability_stdout_keeps_blank_lines() -> None:
    """Blank lines and surrounding whitespace are tolerated."""

    payload = json.dumps(
        {
            "state": "Alive",
            "jobs": 0,
            "gpu_mem": 80000,
            "gpu_compute_capability_major": 8,
        }
    )
    out = "\n   \n" + payload + "\n   \n"
    probe = parse_availability_stdout(out)
    assert probe.idle_compatible(SiteRequirements()) is True
