# Grid'5000 operator reference

The installed Mac-side entry point is:

```text
osm-polygon-grid5000 run --scope {region,all}
    [--region REGION-LATEST] --stage {split,label,all}
```

For `--scope region`, `--region` is required and must be a canonical shard
such as `afghanistan-latest`; it is forbidden with `--scope all`. The operator
uses only the configured external-volume state root. It selects a compatible
site, streams remote logs, preserves checkpoints across allocation boundaries,
and publishes only a complete, validated production run. `--row-limit` is an
advanced non-publishing canary control: zero means the complete selected scope.

```text
osm-polygon-grid5000 status RUN_ID
osm-polygon-grid5000 resume RUN_ID
osm-polygon-grid5000 cleanup --site SITE
```

`status` is local and read-only. `resume` reconstructs the immutable historical
run identity from the external-volume state, reattaches a recorded live OAR job
without submitting a duplicate, and continues only from validated checkpoint
pairs after a terminal allocation. When a compatible site changes, checkpoints
are validated, relayed through the external volume, independently read back,
and installed before one new bounded allocation is submitted.

`cleanup` previews pipeline-owned completed or failed remote runs unless
`--execute` is supplied. Ctrl-C during `run` or `resume` stops local monitoring
but leaves OAR work and checkpoints intact; rerun the exact `resume RUN_ID`
command printed by the operator.

## Earliest policy-compliant start

When a fresh split submission, a regional label submission, or `resume` finds
a queued job whose OAR start forecast is more than ten minutes away, it checks
every configured site for a compatible GPU that is factually idle now. A site
is eligible only when the required immutable runtime is staged there and the
live usage-policy and home-quota checks pass. Worldwide `all` runs optimize
the split allocation first; labeling then stays with the site holding the
validated split output.

The queued job remains the fallback. Replacement trials request a 20-minute
allocation, which is easier for OAR to backfill than the normal 55-minute
allocation. If OAR already forecasts a start more than ten minutes away, the
operator cancels that trial immediately and checks the next site. When OAR
provides no forecast, the operator observes the trial for at most two minutes.
Only after a trial is confirmed `Running` does it adopt that job ID and cancel
the old queued reservation. During the trial it also watches the fallback; if
the fallback starts first, the trial is cancelled.

The trial is tagged `day` or `night` using Europe/Paris time and its complete
20-minute walltime. Near the 09:00 and 19:00 weekday boundaries, the operator
selects the next window when the job cannot fit entirely in the current one.
Inside the allocation, inference receives ten minutes, followed by five
minutes for graceful checkpointing and five minutes of scheduler margin. The
same immutable run identity and checkpoint directory are reused by every
allocation.

This uses Grid'5000's documented exception for immediately available jobs of
at most one hour. It does not infer an ETA from queue depth and does not submit
speculative jobs to several sites. Ctrl-C leaves both job IDs in durable state;
running the same `resume RUN_ID` command recovers the trial decision without a
duplicate submission.
