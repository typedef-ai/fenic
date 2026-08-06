# Mixed-complexity semantic workload benchmark — design freeze

**Status:** IMPLEMENTATION CHECKPOINT — focused proof gate and required 24-row
pilot passed; the approved 96/192 full matrix is stopped by its own 60-second
real-clock condition before a larger arm is dispatched. This is not an
implementation-review freeze.
**Branch:** `herd/fenic-exec-engine-mixed-complexity-workload`, stacked on the
TD-3385 directed-closure head `7295e02`.

## Purpose and boundary

The accepted validation matrix used uniform synthetic rows. This node tests the
engine stack under a production-shaped, multi-step workload: typedef's observed
12-pass semantic-pipeline dependency envelope, mixed per-row output complexity,
real ModelClient scheduling/rate-limit behavior, and explicit materialization
arms. It extends—not replaces—`tests/_inference/rate_limit_harness/harness.py`.

All runs use `SimulatedCompletionsClient` / `SimulatedServerLimiter`, real
`time.time()` plus real `asyncio.sleep`, and no API key, Session provider
validation, `.env` read, or network call. Estimated and actual provider spend:
**$0.00**. The provider-free selector rule at
`.context/PROVIDER-FREE-TEST-SELECTION.md` applies: the new exact harness test
nodes are allowlisted only after their bodies prove no provider boundary.

This is a benchmark and measurement node. It does not relax typedef's existing
failure/resume/progress semantics, does not claim current typedef has an
unmaterialized N-op chain, and does not broaden B1 beyond its already-reviewed
map-to-extract matcher.

## Grounded workload specification

The canonical read-only typedef survey found these dependencies in
`ingest/static_loaders/semantic/pipeline/dependencies.py`; today's executor runs
each pass as `pass_instance.execute(df).cache(); df.count()` for lazy-error
visibility. The harness preserves that 12-step envelope and its uneven output
cost rather than pretending all rows/passes are identical.

| Step | Typedef analogue | Operator | Per-row completion distribution | Input / max output token class |
| --- | --- | --- | --- | --- |
| 01 | relation analysis | extract | constant 72 | medium / 160 |
| 02 | column analysis | extract | lognormal(4.4, 0.30) | wide / 256 |
| 03 | join analysis | map | constant 56 | medium / 160 |
| 04 | filter analysis | map | lognormal(3.6, 0.25) | wide / 128 |
| 05 | grouping analysis | map | regime shift 40 -> 120 | wide / 192 |
| 06 | time analysis | map | lognormal(4.2, 0.35) | medium / 224 |
| 07 | window analysis | map | constant 80 | medium / 192 |
| 08 | output shape | map | lognormal(4.0, 0.25) | wide / 192 |
| 09 | audit analysis | map | regime shift 48 -> 180 | medium / 256 |
| 10 | business semantics | map | lognormal(4.5, 0.35) | wide / 288 |
| 10a | grain humanization | map | constant 96 | medium / 224 |
| 11 | analysis summary | map | lognormal(5.5, 0.45) | narrow / 768 |

Each step gets a stable derived seed (`base_seed + step ordinal`) and its draws
are precomputed by the existing `constant`, `lognormal`, or `regime_shift`
helpers before dispatch. The simulator must encode `(step_id, row_id)` in each
request, select that step's precomputed result, and emit a deterministic map
string or valid extract JSON according to the real operator's response format.
Thus retry/dispatch order cannot alter logical data or actual token use.

The initial evidence matrix is 96 and 192 rows, three fixed base seeds, client
TPM 150,000 and matching server TPM for the no-overshoot lane, plus an
intentionally modest 0.90x server-TPM lane that naturally exercises simulated
429/backoff. Request latency is 1 ms real sleep. Pilot first at 24 rows; stop a
full evidence run if it exceeds 60 seconds or produces an unexpected semantic
result. The implementation records raw draws, scenario knobs, trace-derived
counts, per-step receipts, and wall samples under
`.context/validation/mixed-complexity-workload/`.

## Arm contract

All arms run the same inputs, seeds, step matrix, simulated server, logical
outputs, and ModelClient. They differ only in checkpointing/fusion mechanics.

| Arm | Execution shape | Question answered |
| --- | --- | --- |
| `barriered` | Faithful typedef baseline: execute each of the 12 step-shaped operators, cache/checkpoint, then force count before the next step. | What does today's cache-plus-count failure/progress boundary cost under mixed complexity? |
| `unbarriered_unfused` | Same dependency-only plan without per-step cache/count. The eligible overlay is converted with B1 intentionally disabled at the test seam, so it uses ordinary map then extract physical nodes. | What gap remains when barriers disappear but B1 does not pipeline the eligible handoff? |
| `unbarriered_fused` | Same unbarriered plan; only the eligible overlay is converted through the current `FusedMapExtractExec` / `MapExtract` path. | What incremental gain comes from B1's existing bounded pipeline once a pass gap exists? |

Typedef's present 12 passes contain extracts followed by maps, not a current
unmaterialized map-to-extract pair. To avoid inventing a production claim, the
arms add one clearly labeled **eligible-chain overlay** after the grain-shaped
step: a map produces deterministic narrative text and its immediate extract
produces the summary-shaped struct. It has the same row inputs/draws in every
arm. The barriered arm checkpoints both overlay sides; the unfused arm uses the
same logical expressions with the test-only converter fusion seam disabled; the
fused arm must assert `FusedMapExtractExec`. Report its result separately from
the 12-step faithful baseline. This is the B1 opportunity measurement, not an
assertion that typedef already has that chain.

## Implementation plan and measurement contract

1. Add a workload-specific simulator beside the existing rate-limit harness,
   reusing `RateLimitScenario`, `SimulatedServerLimiter`, real ModelClient
   queue/scheduler/backoff/settlement, and distribution helpers. Do not alter
   the existing single-batch harness semantics. Its trace adds step ID but keeps
   success-only token accounting and logical-completion accounting unchanged.
2. Build provider-free direct operator and local physical-plan fixtures around a
   `LanguageModel` backed by that simulator. Swap the simulated client before
   any semantic execution; assert its trace is nonempty and no default provider
   client method is invoked. Map responses are deterministic strings and
   extracts deterministic schema-valid JSON. The three arms must have identical
   final row IDs and structs, per-step logical completion totals, and actual
   output token draws.
3. Attach P0's `set_request_lifecycle_collector(..., execution_id=...)` to the
   shared simulator. Compute `compute_idle_gap_metrics` over the whole run and
   per operation/step. Preserve P0 semantics exactly: queue delay and
   rate-limited time are separate; retry backoff remains in flight and is not
   recast as execution idle.
4. Emit JSON receipts containing wall time, logical rows/sec, per-step wall,
   lifecycle idle totals/p50/p95/non-rate-limited remainder, queue/rate-limited
   time, total attempts, simulated 429s/retries/backoffs, actual/reserved output
   tokens, reservation efficiency, achieved output TPM, and adaptive-estimator
   reservation snapshots by step. The report may state causal conclusions only
   where arm deltas and traces support them.

Focused proof tests will first establish: deterministic step draws are
dispatch-order invariant; the simulator's real ModelClient trace exercises
queue/settle and natural 429/retry; barriered output equals each unbarriered
arm; the converter selects fused only in the fused arm; and P0's rate-limited
idle exclusion remains true across a multi-step trace. The full real-clock
matrix is an explicit benchmark command, not a default unit-test requirement.

## Decision rules and successor

Report three separate outcomes, never a blended score:

1. **Barrier relaxation:** barriered versus unbarriered-unfused, qualified as a
   typedef product/failure/resume/progress decision—not a fenic defect.
2. **B1 fusion:** unbarriered-unfused versus unbarriered-fused for the labeled
   eligible overlay only; quantify idle handoff and wall, not raw throughput
   alone.
3. **Adaptive rate limiting:** matching-server versus modest-overshoot lanes;
   characterize 429/retry, reservation efficiency, and estimator adaptation
   without inducing a real provider limit.

The next Captain-authorized node after this node's full implementation/review
cycle is **TD-3383**, already assigned in Linear by Herd Command. Its design
must use this barrier-arm evidence and typedef TD-4372 to identify safe
collect/show/count pushdown boundaries (semantic/model operators, cache,
metrics, joins, aggregates, sorts, and SQL), then implement only clear
low-risk cases. It is a separate design-first, local-only node; no TD-3383 work
starts now.

## Frozen design review request — DISPOSED

The design review was approved by Herd Command. Implementation was therefore
authorized; the active HOLD is solely the measured 60-second matrix bound
recorded below, not the prior design gate.

## Herd design disposition and implementation checkpoint

Herd Command approved the design unchanged. Its load-bearing requirement is
preserved in code: all three arms must have identical final row IDs/structs,
per-step logical totals, and actual output-token draws; a failure raises rather
than loosening the fixture. The default gate contains only focused proof tests;
the pilot and matrix remain explicit opt-in real-clock commands. Every durable
receipt stamps `mixed-workload-harness-v1`, `mixed-workload-v1`, all scenario
knobs, and the derived seed per step.

Focused local-only gate: **4 passed, 2 skipped**. It proves stable step/row
draws, all-arm parity, B1 selection only in the fused arm, natural simulated
server-429/retry behavior, query-level LMMetrics capture, and P0's
rate-limited-idle exclusion over a multi-step lifecycle. The exact-node
provider-free selector entries are in
`.context/PROVIDER-FREE-TEST-SELECTION.md`; every semantic action swaps the
fixture model client for `WorkloadSimulatedCompletionsClient` before dispatch.

The required 24-row matching-server pilot passed and wrote its raw receipt to
`.context/validation/mixed-complexity-workload/pilot/mixed-workload-v1-matching-pilot-24-seed101.json`.
It executed 336 logical completions per arm (24 rows × 14 step/overlay
operators), selected fusion only in the fused arm, and observed no server 429:

| Arm | Wall s | Actual output tokens | Reserved output tokens | P0 idle / queue ms |
| --- | ---: | ---: | ---: | ---: |
| barriered | 0.3295 | 30,696 | 55,488 | 153.760 / 703.306 |
| unbarriered-unfused | 0.0615 | 30,696 | 75,604 | 1.721 / 3,687.259 |
| unbarriered-fused | 0.0640 | 30,696 | 75,609 | 4.746 / 3,681.361 |

The `base_seed=101` pilot's actual simulated server total is 51,816 tokens
(30,696 completion + 21,120 input). Deterministic preflight expansion gives
207,435 at 96 rows and 415,002 at 192 rows. At the approved 150,000 TPM,
192 rows exceed the initially full bucket by 265,002 tokens. Even at a perfect
2,500-token/s refill and zero other work, that **single arm** has a 106.0 s
lower bound, already over the approved 60 s full-evidence stop condition.
The full 12-receipt matrix therefore has not been started: dispatching it would
violate the frozen design rather than measure it honestly.

## Matrix Amendment — APPROVED BY HERD COMMAND

Herd Command accepted the hold and clarified that the prior 60-second bound was
a runaway guard, not a scientific cutoff: the deterministic TPM-refill wait is
the governor-bound signal the experiment is intended to measure. The full
matrix is now authorized with these non-negotiable controls:

1. Each arm has a **300-second** real-clock wall stop.
2. A no-progress watchdog stops an arm after **60 consecutive seconds without
   a lifecycle `settled` event**. This is the hang detector; normal TPM refill
   wait is not treated as a failure while settlements continue.
3. The complete evidence run has a **45-minute** wall budget. If observed
   timing projects the 12 receipts past it, omit both 192-row lanes for the
   third seed only and write an explicit matrix-reduction receipt—never silently
   truncate.
4. Keep the approved 96/192 rows, 150,000 client TPM, and matching/0.90x-server
   lanes otherwise unchanged. In particular, do not increase TPM to hide the
   refill-bound regime.

Every receipt will include its deterministic preflight actual-token expansion,
the initial client/server bucket capacities, excess token math, and controls.
For the pilot seed, this is 207,435 at 96 rows and 415,002 at 192 rows against
the 150,000-token client bucket; other seeds retain their own exact deterministic
draw total. Proceed through the matrix, preserve partial evidence on any guard,
then freeze for implementation review. TD-3383 remains blocked until that review
cycle completes.

## Matrix recovery note — before final lane

The first matrix execution completed eleven durable receipts in 2,289.326s of
active benchmark wall. Its reduction check was mistakenly evaluated once before
third-seed matching and again before third-seed overshoot. Matching was therefore
already written when the second check forecast 2,992.550s and emitted a manifest
claiming both third-seed lanes were dropped. The receipts make the inconsistency
visible; nothing was deleted or concealed.

Before another simulated call, the runner was corrected to evaluate reduction
only once, before either third-seed lane. The only missing lane is
192-row/seed-303/modest-overshoot. The deterministic active-wall budget leaves
410.674s; the first two overshoot scenarios were 386.655s and 387.232s, so the
bounded completion is authorized by the same 45-minute envelope. A dedicated
resume receipt will stamp this recovery and update the manifest; no prior arm is
rerun, no TPM/row count changes, and no provider call is involved.
