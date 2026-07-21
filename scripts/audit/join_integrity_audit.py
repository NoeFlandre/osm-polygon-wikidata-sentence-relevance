"""Read-only join-integrity audit across all processable shards (Phase 9M-C).

Runs the existing join path (no model loading, no GPU, no inference,
no publishing) across every shard returned by ``discover_shards``,
records every failure with full diagnostics, and emits a structured
JSON report plus a human-readable summary on stdout.

For each failing shard, the offending table/key/source rows are
materialised so the operator can decide whether the defect belongs
in the upstream dataset or in the join contract.

Usage::

    python -m scripts.audit.join_integrity_audit \
        --input-root <INPUT_ROOT> \
        --report <REPORT.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa

from osm_polygon_sentence_relevance.contracts.errors import JoinIntegrityError
from osm_polygon_sentence_relevance.ingestion.discovery import discover_shards
from osm_polygon_sentence_relevance.joins import (
    join_wikipedia_sections,
    join_wikivoyage_sections,
)


@dataclass
class ShardFailure:
    shard_key: str
    exception_type: str
    table: str | None
    column: str | None
    message: str
    sample_offending_values: list[Any] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_key": self.shard_key,
            "exception_type": self.exception_type,
            "table": self.table,
            "column": self.column,
            "message": self.message,
            "sample_offending_values": self.sample_offending_values,
            "detail": self.detail,
        }


def _read_table(path: Path) -> pa.Table:
    return pa.parquet.read_table(str(path))


def _missing_qid_count(shard) -> dict[str, int] | None:
    """For each wikivoyage-failing shard, count the actual number of
    wikivoyage_documents rows whose wikidata QID is missing from the
    polygons table, plus how many distinct missing QIDs there are.
    """
    if shard.wikivoyage_documents is None or shard.wikivoyage_sections is None:
        return None
    try:
        polygons = _read_table(shard.polygons)
        wv_docs = _read_table(shard.wikivoyage_documents)
    except Exception:
        return None
    poly_qids = set(polygons.column("wikidata").to_pylist())
    wv_qids = wv_docs.column("wikidata").to_pylist()
    missing = [q for q in wv_qids if q not in poly_qids]
    return {
        "wikivoyage_documents_row_count": len(wv_qids),
        "missing_wikidata_rows": len(missing),
        "missing_wikidata_distinct": len(set(missing)),
    }


def _audit_one(shard) -> tuple[bool, str | None, ShardFailure | None]:
    """Run both join paths for one shard. Returns (ok, source, failure).

    source is "wikipedia" or "wikivoyage" when the failure belongs to
    that join; None if both joins succeeded or a non-join error occurred.
    """
    polygons = _read_table(shard.polygons)
    polygon_articles = _read_table(shard.polygon_articles)
    wp_docs = _read_table(shard.wikipedia_documents)
    wp_sections = _read_table(shard.wikipedia_sections)

    try:
        join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)
    except JoinIntegrityError as exc:
        return (
            False,
            "wikipedia",
            ShardFailure(
                shard_key=shard.shard_key,
                exception_type=type(exc).__name__,
                table=exc.table_name,
                column=exc.key,
                message=str(exc),
                sample_offending_values=list(exc.sample or []),
            ),
        )
    except Exception as exc:
        return (
            False,
            "wikipedia",
            ShardFailure(
                shard_key=shard.shard_key,
                exception_type=type(exc).__name__,
                table=None,
                column=None,
                message=str(exc) or repr(exc),
            ),
        )

    if shard.wikivoyage_documents is not None and shard.wikivoyage_sections is not None:
        wv_docs = _read_table(shard.wikivoyage_documents)
        wv_sections = _read_table(shard.wikivoyage_sections)
        try:
            join_wikivoyage_sections(polygons, wv_docs, wv_sections)
        except JoinIntegrityError as exc:
            return (
                False,
                "wikivoyage",
                ShardFailure(
                    shard_key=shard.shard_key,
                    exception_type=type(exc).__name__,
                    table=exc.table_name,
                    column=exc.key,
                    message=str(exc),
                    sample_offending_values=list(exc.sample or []),
                ),
            )
        except Exception as exc:
            return (
                False,
                "wikivoyage",
                ShardFailure(
                    shard_key=shard.shard_key,
                    exception_type=type(exc).__name__,
                    table=None,
                    column=None,
                    message=str(exc) or repr(exc),
                ),
            )

    return True, None, None


def _trace_italy_mismatch(
    shards, target_shard: str = "italy-latest"
) -> dict[str, Any] | None:
    """Trace the offending row through polygons + polygon_articles for italy-latest.

    Returns the violation record, or None if the target shard is absent
    or has no integrity violation.
    """
    target = None
    for s in shards:
        if s.shard_key == target_shard:
            target = s
            break
    if target is None:
        return None

    polygons = _read_table(target.polygons)
    polygon_articles = _read_table(target.polygon_articles)

    poly_wd = dict(
        zip(
            polygons.column("polygon_id").to_pylist(),
            polygons.column("wikidata").to_pylist(),
            strict=True,
        )
    )

    wd_pa = polygon_articles.column("wikidata").to_pylist()
    pid_pa = polygon_articles.column("polygon_id").to_pylist()

    mismatched_rows: list[dict[str, Any]] = []
    for polygon_id, wd in zip(pid_pa, wd_pa, strict=True):
        expected = poly_wd.get(polygon_id)
        if expected != wd:
            mismatched_rows.append(
                {
                    "polygon_id": polygon_id,
                    "polygon_articles_wikidata": wd,
                    "polygons_wikidata": expected,
                    "upstream_record_at_fault": (
                        "polygon_articles" if (expected and wd) else "unknown"
                    ),
                }
            )

    polygons_only = sorted(set(polygons.column("polygon_id").to_pylist()) - set(pid_pa))
    pa_only = sorted(set(pid_pa) - set(polygons.column("polygon_id").to_pylist()))

    return {
        "shard_key": target_shard,
        "mismatch_count": len(mismatched_rows),
        "mismatched_rows_first_25": mismatched_rows[:25],
        "mismatched_rows_total": len(mismatched_rows),
        "polygons_only_polygon_ids_first_25": polygons_only[:25],
        "polygons_only_polygon_ids_total": len(polygons_only),
        "polygon_articles_only_polygon_ids_first_25": pa_only[:25],
        "polygon_articles_only_polygon_ids_total": len(pa_only),
        "polygons_row_count": polygons.num_rows,
        "polygon_articles_row_count": polygon_articles.num_rows,
        "polygons_unique_polygon_ids": len(
            set(polygons.column("polygon_id").to_pylist())
        ),
        "polygon_articles_unique_polygon_ids": len(set(pid_pa)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--target-shard",
        default="italy-latest",
        help="Shard to deep-trace when wikidata mismatches are found (default: italy-latest)",
    )
    args = parser.parse_args()

    started = time.monotonic()
    shards = discover_shards(args.input_root)
    discover_elapsed = time.monotonic() - started

    summary: dict[str, Any] = {
        "input_root": str(args.input_root),
        "total_shards": len(shards),
        "discover_elapsed_seconds": round(discover_elapsed, 3),
        "passed": 0,
        "failed": 0,
        "failures": [],
        "exception_type_counts": {},
        "table_counts": {},
        "column_counts": {},
    }
    failures: list[ShardFailure] = []
    exception_type_counter: Counter[str] = Counter()
    table_counter: Counter[str] = Counter()
    column_counter: Counter[str] = Counter()

    audit_started = time.monotonic()
    for shard in shards:
        ok, _source, failure = _audit_one(shard)
        if ok:
            summary["passed"] += 1
            continue
        summary["failed"] += 1
        assert failure is not None
        if failure.table == "wikivoyage_documents":
            failure.detail = _missing_qid_count(shard) or {}
        failures.append(failure)
        exception_type_counter[failure.exception_type] += 1
        if failure.table is not None:
            table_counter[failure.table] += 1
        if failure.column is not None:
            column_counter[failure.column] += 1

    summary["audit_elapsed_seconds"] = round(time.monotonic() - audit_started, 3)
    summary["failures"] = [f.to_dict() for f in failures]
    summary["exception_type_counts"] = dict(exception_type_counter)
    summary["table_counts"] = dict(table_counter)
    summary["column_counts"] = dict(column_counter)

    summary["italy_trace"] = _trace_italy_mismatch(
        shards, target_shard=args.target_shard
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))

    print(f"Total shards: {summary['total_shards']}")
    print(f"Passed:       {summary['passed']}")
    print(f"Failed:       {summary['failed']}")
    print(f"Discover:     {summary['discover_elapsed_seconds']}s")
    print(f"Audit:        {summary['audit_elapsed_seconds']}s")
    print()
    print(f"Exception type counts: {dict(exception_type_counter)}")
    print(f"Table counts:          {dict(table_counter)}")
    print(f"Column counts:         {dict(column_counter)}")
    print()
    if failures:
        print("Failing shards:")
        for f in failures:
            print(
                f"  - {f.shard_key}: {f.exception_type} ({f.table}.{f.column}): {f.message}"
            )
    print()
    print(f"Report: {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
