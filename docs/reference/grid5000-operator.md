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
