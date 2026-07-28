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
osm-polygon-grid5000 cleanup --site SITE
```

`status` is local and read-only. `cleanup` previews pipeline-owned completed or
failed remote runs unless `--execute` is supplied. Ctrl-C during `run` stops
local monitoring but leaves OAR work and checkpoints intact.
