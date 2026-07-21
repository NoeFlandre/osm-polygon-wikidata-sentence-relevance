"""Read-only 291-shard join integrity audit against the corrected upstream snapshot.

This is a one-off verification harness used during the Phase 9N
publication. It does not touch any production source code; it only
imports the public join APIs from the pinned sister project
(``osm-polygon-wikidata-sentence-relevance`` at the exact revision
recorded in the run) and exercises them against the staged candidate
data root.

For each shard discovered by
:func:`osm_polygon_sentence_relevance.ingestion.discovery.discover_shards`,
it runs the Wikipedia and Wikivoyage join paths. No model is loaded,
no inference is run, no Hub API is called.

Usage::

    python scripts/audit_upstream_correction.py \\
        --input-root <CANDIDATE_PROCESSED_DIR> \\
        --report <REPORT.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
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
    source: str
    exception_type: str
    message: str
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_key": self.shard_key,
            "source": self.source,
            "exception_type": self.exception_type,
            "message": self.message,
            "detail": self.detail,
        }


def _read(path: Path) -> pa.Table:
    return pa.parquet.read_table(str(path))


def _audit(shard) -> tuple[bool, list[ShardFailure]]:
    failures: list[ShardFailure] = []
    try:
        polygons = _read(shard.polygons)
        polygon_articles = _read(shard.polygon_articles)
        wp_docs = _read(shard.wikipedia_documents)
        wp_sections = _read(shard.wikipedia_sections)
        join_wikipedia_sections(polygons, polygon_articles, wp_docs, wp_sections)
    except JoinIntegrityError as exc:
        failures.append(
            ShardFailure(
                shard_key=shard.shard_key,
                source="wikipedia",
                exception_type=type(exc).__name__,
                message=str(exc),
                detail={
                    "table_name": exc.table_name,
                    "key": exc.key,
                    "sample": list(exc.sample or []),
                },
            )
        )
        return False, failures
    except Exception as exc:
        failures.append(
            ShardFailure(
                shard_key=shard.shard_key,
                source="wikipedia",
                exception_type=type(exc).__name__,
                message=str(exc) or repr(exc),
                detail={},
            )
        )
        return False, failures

    if shard.wikivoyage_documents is not None and shard.wikivoyage_sections is not None:
        try:
            wv_docs = _read(shard.wikivoyage_documents)
            wv_sections = _read(shard.wikivoyage_sections)
            join_wikivoyage_sections(polygons, wv_docs, wv_sections)
        except JoinIntegrityError as exc:
            failures.append(
                ShardFailure(
                    shard_key=shard.shard_key,
                    source="wikivoyage",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    detail={
                        "table_name": exc.table_name,
                        "key": exc.key,
                        "sample": list(exc.sample or []),
                    },
                )
            )
            return False, failures
        except Exception as exc:
            failures.append(
                ShardFailure(
                    shard_key=shard.shard_key,
                    source="wikivoyage",
                    exception_type=type(exc).__name__,
                    message=str(exc) or repr(exc),
                    detail={},
                )
            )
            return False, failures

    return True, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    t0 = time.monotonic()
    shards = discover_shards(args.input_root)
    discover_elapsed = time.monotonic() - t0

    failures: list[ShardFailure] = []
    audit_started = time.monotonic()
    for shard in shards:
        ok, fs = _audit(shard)
        if not ok:
            failures.extend(fs)
    audit_elapsed = time.monotonic() - audit_started

    summary = {
        "input_root": str(args.input_root),
        "total_shards": len(shards),
        "passed": len(shards) - len({f.shard_key for f in failures}),
        "failed": len({f.shard_key for f in failures}),
        "discover_elapsed_seconds": round(discover_elapsed, 3),
        "audit_elapsed_seconds": round(audit_elapsed, 3),
        "failures": [f.to_dict() for f in failures],
        "exception_type_counts": dict(Counter(f.exception_type for f in failures)),
        "source_counts": dict(Counter(f.source for f in failures)),
        "table_counts": dict(
            Counter(
                f.detail.get("table_name", "<unknown>")
                for f in failures
                if f.detail.get("table_name")
            )
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))

    print(f"Total shards: {summary['total_shards']}")
    print(f"Passed:       {summary['passed']}")
    print(f"Failed:       {summary['failed']}")
    print(f"Discover:     {summary['discover_elapsed_seconds']}s")
    print(f"Audit:        {summary['audit_elapsed_seconds']}s")
    if failures:
        print()
        print("Failures:")
        for f in failures:
            print(f"  - {f.shard_key} ({f.source}): {f.exception_type}: {f.message}")
    print()
    print(f"Report: {args.report}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
