# Earliest Policy-Compliant Grid'5000 Start Design

## Goal

When a resumable labeling allocation is queued far in the future, the operator
should look for an earlier, immediately runnable allocation across compatible
Grid'5000 sites without losing the existing reservation, duplicating inference,
or violating the Grid'5000 usage policy.

## Safety model

- A recorded queued allocation remains the fallback until a replacement is
  confirmed `Running`.
- The operator never submits replacements to several sites concurrently.
- Only a compatible, currently idle GPU can be tried as an immediate
  replacement.
- The replacement walltime is at most 55 minutes and uses OAR's `day` policy
  type during a weekday daytime window. This fits Grid'5000's documented
  exception for jobs of at most one hour that start within ten minutes.
- The live `usagepolicycheck` and storage quota preflights must pass at the
  candidate site before any submission.
- If the candidate does not become `Running` within ten minutes, it is
  cancelled and the original reservation remains intact.
- Once the candidate is confirmed `Running`, the original queued job is
  cancelled and the replacement becomes the sole durable active job.
- Job IDs and the replacement lifecycle are written atomically to local durable
  state before monitoring so Ctrl+C and `resume RUN_ID` cannot duplicate work.
- A remote execution lock remains the final protection against simultaneous
  inference if scheduler state changes at a boundary.

## Selection

The operator reads the recorded job's OAR forecast. It only optimizes a queued
job whose forecast is more than ten minutes away. It probes all configured
sites and filters them by:

1. SSH reachability;
2. GPU memory and CUDA capability;
3. persistent storage headroom;
4. a currently idle compatible GPU reported by `oarnodes`;
5. a successful live usage-policy check;
6. absence of another live managed allocation for the same run.

Candidates are ordered by reuse of an already prepared managed run, then site
name. Preparation and immutable asset staging happen before submission. The
operator tries candidates sequentially, but never retains more than one trial.

## State and recovery

Durable state records:

- `fallback_site` and `fallback_job_id`;
- `replacement_site` and `replacement_job_id`;
- `replacement_status`;
- the final active `site` and `job_id`.

On resume:

- a live trial is inspected first;
- if it is running, the fallback is cancelled and the trial is adopted;
- if it is merely queued and still within the ten-minute trial window, the
  operator reattaches;
- if the window expired, the trial is cancelled and the fallback is restored;
- if the fallback has already started, no replacement is attempted.

## User-visible output

The terminal reports:

- the fallback forecast and walltime;
- every candidate rejected and the factual reason;
- the selected trial site;
- the ten-minute immediate-start deadline;
- whether the fallback was retained or replaced;
- the exact active job after the decision.

The operator never claims a site is earlier unless its replacement job is
actually `Running`.

## Tests

RED-first unit and lifecycle tests cover:

- no optimization for running, near-start, missing-forecast, night, or weekend
  jobs;
- deterministic idle-compatible candidate selection;
- policy/storage/preparation failures preserving the fallback;
- one replacement submission at a time;
- timeout cancellation preserving the fallback;
- running replacement adoption followed by fallback cancellation;
- Ctrl+C recovery during each durable replacement state;
- no duplicate inference and no cancellation of a running fallback.

