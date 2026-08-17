# Streaming performance benchmark

This on-demand benchmark compares standard and streaming semantic-operator
paths without changing Git state. The maintained matrix uses the real fenic
queue, scheduler, rate-limit gate, and bounded `semantic.join` implementation
with a simulated provider. It therefore runs end to end with zero provider
calls. A matrix may explicitly select provider execution, which requires a
separate acknowledgement and a positive, bounded cost estimate.

## Plan and run

Plan against an explicit clean checkout and ref:

```text
uv run python benchmarks/streaming/run_matrix.py plan \
  --matrix benchmarks/streaming/matrices/streaming-v1.json \
  --checkout /path/to/fenic --expect-ref <branch-or-sha> \
  --output /path/to/new-receipt-directory
```

Run the plan on the same host:

```text
uv run python benchmarks/streaming/run_matrix.py run \
  --plan /path/to/new-receipt-directory/plan.json \
  --max-cost-usd 15
```

Add `--approve-provider-spend` only for a matrix whose executable scenario has
`execution_mode: provider`. Planning resolves the commit and records the host,
schema and harness hashes, expanded cells, request estimate, pricing, and cap.
The runner re-derives those values from the matrix before execution. It refuses
a dirty or moved checkout, an altered plan, missing acknowledgement, or any
actual-plus-reserved-plus-remaining amount above the lower of the matrix and
command-line caps.

Every output directory is single use. A durable reservation is written before
each provider-backed subprocess starts. If that subprocess times out or fails
without usable metrics, its conservative reservation remains charged in the
run state rather than disappearing. The child accepts only a matrix-derived
cell authenticated by its parent runner; it is not a standalone provider-call
interface.

## Matrix contract

`matrix.schema.json` bounds rows, physical requests, token estimates, batch
sizes, repetitions, timeouts, and cost. Pricing must be positive. Per-scenario
input and output token declarations are the inputs to the conservative cost
projection.

The maintained `streaming-v1.json` executes a bounded join scenario against
today's fenic APIs. Map and predicate operators and two- and three-step chain
shapes remain reserved with execution disabled. They can be enabled only when
their runtime path and engagement evidence are implemented.

A verdict compares arms interleaved within one run on one host. To compare two
commits, supply `--baseline-checkout` and `--baseline-ref` to the same plan so
both checkouts participate in that interleaving. Receipt aggregation rejects
mixed run IDs and plan IDs, so a cross-run or historical comparison cannot
produce `PASS` or `FAIL`.

## Evidence and verdicts

The output includes the immutable plan, execution specs, cell receipts, logs,
durable run state, JSON/CSV/Markdown summaries, and a SHA-256 manifest. Each
receipt records the tested commit, arm, workload, wall time, peak RSS, result
hash and count, request metrics, path-engagement evidence, lifecycle
availability, and cumulative spend.

Request and result counts are hard gates. The result count must equal the
matrix-derived expected count, and standard and streaming arms must produce
the same result hash. Engagement evidence must also prove that the two arms
used different execution paths.

Lifecycle fields are independently available. Event counts, queue depth,
rate-limit events, and idle-gap measurements are present only when the tested
checkout exposes each measurement. Missing means null with a reason, never an
inferred zero. Without rate-limit-event availability the timing verdict is
`REGIME_UNVERIFIED`; wall time alone is not a saturation claim.

For three or more repetitions, a streaming median more than 20% slower is a
failure only when its median-absolute-deviation band does not overlap the
standard band. An overlapping regression is `INCONCLUSIVE`. Cache-heavy timing
is observational, while parity and request counts remain hard-gated. Naturally
observed rate limiting places a cell outside the intended regime.

Credentials use the normal provider configuration path. The benchmark never
reads, records, or prints credential values.
