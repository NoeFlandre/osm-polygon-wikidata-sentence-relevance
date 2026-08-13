# Grid'5000 operator reference

The installed Mac-side entry point is:

```text
osm-polygon-grid5000 run --scope {region,all}
    [--region REGION-LATEST] --stage {split,label,all} [--detach]
```

For `--scope region`, `--region` is required and must be a canonical shard
such as `afghanistan-latest`; it is forbidden with `--scope all`. The operator
uses only the configured external-volume state root. It selects a compatible
site, streams remote logs, preserves checkpoints across allocation boundaries,
and publishes only a complete, validated production run. `--row-limit` is an
advanced smoke control. Regional workflows stop after a positive, non-publishing
row limit. A worldwide V2 `all` workflow instead validates and preserves that
isolated smoke, then automatically continues its separate production lane with
`row_limit=0`. Zero starts V2 production directly.

```text
osm-polygon-grid5000 status RUN_ID
osm-polygon-grid5000 resume RUN_ID [--detach]
osm-polygon-grid5000 cleanup --site SITE
```

`status` is local and read-only. `resume` reconstructs the immutable historical
run identity from the external-volume state, reattaches a recorded live OAR job
without submitting a duplicate, and continues only from validated checkpoint
pairs after a terminal allocation. When a compatible site changes, checkpoints
are validated, relayed through the external volume, independently read back,
and installed before one new bounded allocation is submitted. Sentence
splitting relays only its verified completed-shard ledger and, when present,
the single active partial shard; immutable input Parquet files are downloaded
again on the destination site. Labeling continues to relay its validated batch
checkpoint pairs. A live finalization job is reattached and monitored before
any new-submission policy, quota, checkout, or token preflight is attempted;
those checks run only when a replacement allocation is actually needed.

`cleanup` previews pipeline-owned completed or failed remote runs unless
`--execute` is supplied. Ctrl-C during `run` or `resume` stops local monitoring
but leaves OAR work and checkpoints intact; rerun the exact `resume RUN_ID`
command printed by the operator.

For unattended production, add `--detach` to `run` or `resume`. The command
starts one local supervisor and returns immediately. The supervisor uses a
deterministic `tmux` session when available, or a detached child process as a
fallback. It writes an append-only console log below the external data root,
reattaches validated partial runs, and stops only at a complete (or explicitly
split-only) durable state. A second invocation with the same run identity is
refused while its supervisor session is active. The start message prints the
exact session and log path.

## Earliest policy-compliant start

When a fresh submission or `resume` finds a queued job whose OAR start forecast
is more than ten minutes away, it checks every configured site for a compatible
GPU that is factually idle now. It repeats that live scan every five minutes
until the fallback starts, its forecast moves inside ten minutes, or a trial
starts elsewhere. A site is eligible only when its required immutable runtime
is staged and the live usage-policy and home-quota checks pass.

The queued job remains the fallback. Replacement trials request a 15-minute
allocation, which is easier for OAR to backfill than the normal allocation.
If OAR already forecasts a start more than ten minutes away, the
operator cancels that trial immediately and checks the next site. When OAR
provides no forecast, the operator observes the trial for at most two minutes.
Only after a trial is confirmed `Running` does it adopt that job ID and cancel
the old queued reservation. During the trial it also watches the fallback; if
the fallback starts first, the trial is cancelled.

The trial is tagged `day` or `night` using Europe/Paris time and its complete
15-minute walltime. Near the 09:00 and 19:00 weekday boundaries, the operator
selects the next window when the job cannot fit entirely in the current one.
The same immutable run identity and checkpoint directory are reused by every
allocation. Split trials derive their processing deadline from the requested
walltime, reserving one scheduler minute and either one or four minutes for
graceful interruption depending on whether the compute environment is reused.

This uses Grid'5000's documented exception for immediately available jobs of
at most one hour. It does not infer an ETA from queue depth and never retains
more than one fallback plus one trial. Ctrl-C leaves both job IDs in durable state;
running the same `resume RUN_ID` command recovers the trial decision without a
duplicate submission.
