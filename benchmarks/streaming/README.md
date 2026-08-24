# Provider-free streaming performance benchmark

This benchmark compares standard and streaming semantic-operator paths with
zero provider calls. The maintained matrix runs bounded `semantic.join`
through fenic's real queue, scheduler, rate-limit gate, lifecycle events, and
`SimulatedCompletionsClient`. It has no provider execution mode, credential
path, pricing model, spend acknowledgement, or cost ledger.

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
  --plan /path/to/new-receipt-directory/plan.json
```

Planning records the commit, host, schema and harness hashes, expanded cells,
and deterministic arm order. Running re-derives those values before execution.
It refuses a dirty or moved checkout, an altered plan, or an output directory
that has already started a run. The tool never changes Git state.

A verdict compares standard and streaming arms interleaved within one run on
one host. To compare two commits, add `--baseline-checkout` and
`--baseline-ref` to the same plan. This adds an independent within-commit
standard-versus-streaming comparison for the baseline checkout; it does not
compute a timing verdict between commits. Aggregation rejects a missing or
mixed run ID or plan ID, so historical receipts cannot produce a verdict.

## Matrix contract

`matrix.schema.json` defines the complete input contract. `jsonschema` is a
required development dependency, and validation fails if it is unavailable.
The schema bounds rows, physical requests, pair blocks, token budgets, batch
sizes, repetitions, latency, and timeouts.

The maintained matrix executes one bounded-join shape. It produces four pair
blocks and eight token-budget blocks. Every token-budget block is larger than
the streaming watermark. The adapter's own geometry guard rejects any matrix
that does not bind the window, span multiple pair blocks, and split on the
token budget. Map, predicate, and chain shapes remain reserved with execution
disabled.

## Evidence and verdicts

The output contains the immutable plan, start marker, cell receipts, logs,
JSON/CSV/Markdown summaries, and a SHA-256 manifest. Each receipt records the
tested commit, arm, workload geometry, wall time, peak RSS, a hash of the actual
join rows, result and request counts, path-engagement evidence, and lifecycle
availability.

Result content, result count, request count, workload geometry, lifecycle
settlement, and path engagement are hard gates. Any hard-gate failure produces
a `FAIL` summary. The run exits nonzero for every verdict other than `PASS` or
the explicitly timing-only `OBSERVATIONAL` state.

For three or more repetitions, a streaming median at most 20% slower passes,
including when the spread bands overlap. A regression above 20% fails when the
median-absolute-deviation bands are separate. An above-threshold result with
overlapping bands is `INCONCLUSIVE`. Cache-heavy non-join timing is
`OBSERVATIONAL`; joins never become observational merely because left inputs
repeat. Naturally observed rate limiting produces `OUTSIDE_REGIME`.

Lifecycle fields are independently available. A missing event count, queue
depth, rate-limit count, or idle-gap measurement stays null with a reason. It
never becomes an inferred zero. If rate-limit events cannot be measured, the
timing result is `REGIME_UNVERIFIED`; wall time alone is not a saturation
claim.
