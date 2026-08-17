# Streaming performance gate

This is an on-demand benchmark gate for the standard and streaming semantic
operator paths. It is intentionally not a CI check and never changes Git
state. Provider execution is always a separate, explicitly acknowledged step.

## Two stages

Plan against an explicit checkout and ref:

```text
uv run python benchmarks/streaming/run_matrix.py plan \
  --matrix benchmarks/streaming/matrices/streaming-v1.json \
  --checkout /path/to/fenic --expect-ref <branch-or-sha> \
  --output /path/to/receipts
```

Then, from the same host and checkout, run only after reviewing the estimate:

```text
uv run python benchmarks/streaming/run_matrix.py run \
  --plan /path/to/receipts/plan.json \
  --approve-provider-spend --max-cost-usd 15
```

`plan` resolves and records the checkout commit, dirty state, host metadata,
expanded cells, conservative physical-request estimate, pricing inputs, and
cap. It makes no provider call. `run` refuses a dirty checkout, changed HEAD,
missing spend acknowledgement, or a projected cost above the supplied cap.
Each cell runs in a fresh process and receipts are flushed before the next
cell. A comparison verdict is valid only when standard and streaming arms are
interleaved in one orchestrator invocation on one host. Comparing two commits
uses `--baseline-checkout` and `--baseline-ref` in the same plan; the runner
interleaves both checkouts' arms in that session. Historical or cross-run
comparisons are observational and can never produce a gate failure.

## Matrix and reserved chains

`matrix.schema.json` is the versioned contract. It supports one-step map and
predicate operators and two-/three-step chains with `barriered`,
`unfused_unbarriered`, or `fused_unbarriered` execution shapes. The maintained
`streaming-v1.json` preserves the original 72-cell map/predicate matrix. Its
single-operator scenarios are executable. Two- and three-step chain examples
reserve the schema Pipeline Fusion will need, but remain provider-disabled.

## Receipts and verdicts

The output directory contains `plan.json`, `cells/<cell-id>.json`,
`summary.json`, `summary.csv`, `summary.md`, and a SHA-256 `manifest.json`. A cell receipt records the
tested commit, scenario/shape, arm, seed, wall time, throughput, peak RSS,
result hash/count, `LMMetrics`, lifecycle availability, and
cumulative actual spend. Raw receipts are immutable evidence; reruns use a
new output directory.

`LMMetrics`, wall time, throughput, RSS, result parity, and request counts are
populated today. Lifecycle event counts, queue depth, rate-limit events, and
idle-gap measurements are populated only when the tested checkout exposes the
request-lifecycle collector. Otherwise each field is null with an availability
reason, never zero or inferred. A summary without that collector marks timing
as `REGIME_UNVERIFIED`; wall time is not used as a saturation claim.

Unique-input cells fail only when the candidate median is more than 20% slower
and its median-absolute-deviation band does not overlap the baseline band.
An above-threshold overlap is `INCONCLUSIVE` and should be rerun with seven
interleaved repetitions. Cache-heavy timing is `OBSERVATIONAL`, but output
parity and physical request counts remain hard checks. Correctness failures
are hard failures. Observed rate limits classify a cell outside the intended
regime. When rate-limit events cannot be observed, the timing regime remains
unverified even though output and request-count gates still apply.

Credentials are loaded by the normal provider configuration path. The harness
records only credential presence and never reads or prints credential values.

This benchmark is not a CI job. A future provider-free mode may reuse an
optional label-triggered job mechanism, but it must use a dedicated benchmark
label rather than overloading a full-Python-matrix label. That workflow
mechanism is not assumed to exist until its own change lands.
