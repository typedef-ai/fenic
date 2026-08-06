## Execution-engine experiment anchor — BASE STACK LAND; TD-3334/TD-3384 LAND; TD-3385 FROZEN

This reviewer-less draft is the single branches-only anchor for the completed
`fenic-exec-engine-breakdown-v1` experiment. It is not a review request; no
additional PRs were created for the stack or validation epilogue.

**Base stack: COMPLETE.** Every ratified implementation node below is pushed and
`LAND`. The Captain-authorized real-provider validation Amendment A is
**COMPLETE and report-review accepted** after the hardened full matrix passed
normalized semantic parity. TD-3334 is an evidence-backed no-op after its $0
metrics pre-check and production survey. TD-3384's local eager-no-op audit is
LAND with three behavior-preserving fast paths and a directed test/docs closure.
TD-3385 is frozen as a benchmark-backed no-op recommendation: current NumPy
embedding math is both faster on the measured shapes and the safer numerical
match.

## Frozen stack DAG

```text
P0  herd/fenic-exec-engine-p0-foundation @ 834f5c08d  LAND
 |
 `-- P1a herd/fenic-exec-engine-p1a-sim-join @ c8d797a  LAND
     |
     `-- P1b herd/fenic-exec-engine-p1b-reduce-residual @ c8d797a  LAND (no-op)
         |
         `-- B0 herd/fenic-exec-engine-b0-model-stream @ c7967d092  LAND
             |
             `-- P1c herd/fenic-exec-engine-p1c-semantic-join @ 743e09b  LAND
                 |
                 `-- B0+1c herd/fenic-exec-engine-b0-1c-stream-adapter @ 574a743  LAND
                     |
                     `-- B1 herd/fenic-exec-engine-b1-two-op-fusion @ bd7d89b  LAND
                         |
                         `-- Validation herd/fenic-exec-engine-validation @ 9098734  REPORT ACCEPTED
                             |
                             `-- TD-3334 herd/fenic-exec-engine-td3334-n-op-fusion @ 54ea844  ACCEPTED NO-OP
                                 |
                                 `-- TD-3384 herd/fenic-exec-engine-td3384-eager-noop-audit @ f0fcc14  LAND (directed test/docs closure)
                                     |
                                     `-- TD-3385 herd/fenic-exec-engine-td3385-polars-embedding-benchmark  FROZEN (benchmark-backed no-op; Herd review requested)
```

| Node | Final SHA | Review receipt | Lane receipt |
| --- | --- | --- | --- |
| P0 foundation | `834f5c08d` | `.context/td-review/p0-foundation-review-2026-08-05.md` | `.context/outbox/p0-foundation-lane-record-2026-08-05.md` |
| P1a sim-join | `c8d797a` | `.context/td-review/p1a-sim-join-c8d797a.md` | `.context/outbox/p1a-sim-join-lane-record-2026-08-05.md` |
| P1b reduce residual | `c8d797a` | `.context/td-review/p1b-reduce-residual-c8d797a.md` | `.context/outbox/p1b-reduce-residual-lane-record-2026-08-05.md` |
| B0 model stream | `c7967d092` | `.context/td-review/b0-model-stream-c7967d0-review-2026-08-06.md` | `.context/outbox/b0-model-stream-lane-record-2026-08-05.md` |
| P1c semantic.join | `743e09b` | `.context/td-review/p1c-semantic-join-review-2026-08-06-delta.md` | `.context/outbox/p1c-semantic-join-lane-record-2026-08-06.md` |
| B0+1c adapter | `574a743` | `.context/td-review/b0-1c-stream-adapter-receipt-2026-08-05.md` | `.context/outbox/b0-1c-stream-adapter-lane-record-2026-08-06.md` |
| B1 fusion | `bd7d89b` | `.context/td-review/b1-two-op-fusion-review-2026-08-05.md` | `.context/outbox/b1-two-op-fusion-lane-record-2026-08-06.md` |
| Validation epilogue | `9098734` | Herd Command report-review **ACCEPTED** | `.context/outbox/fenic-exec-engine-validation-report-2026-08-06.md` |
| TD-3334 N-op fusion | `54ea844` | Herd Command **ACCEPTED** no-op | `.context/outbox/td3334-n-op-fusion-lane-record-2026-08-06.md` |
| TD-3384 eager no-op audit | `f0fcc14` | Herd Command **LAND**; directed test/docs closure, no new round | `.context/td-review/td3384-eager-noop-review-2026-08-06.md` |
| TD-3385 embedding math benchmark | current branch head | Herd Command review requested | `.context/outbox/td3385-polars-embedding-benchmark-lane-record-2026-08-06.md` |

## Final evidence

- Base-stack spend: **$0.00 actual**. Original validation's stopped arm retains
  a `$0.041574` upper bound. Amendment A completed its full matrix with
  serialized LMMetrics reporting `$0.000000` despite non-empty lifecycle
  evidence; that telemetry is not a billing receipt. Conservative cumulative
  planning bound: **$0.785305** against the $50 cap.
- P0 provides deterministic lifecycle accounting and the serial map→extract
  reference fixture. P1a bounds sim-join left search slices. P1b is a reviewed
  no-op. B0 bounds row-local request/future/raw-response blocks. P1c bounds
  semantic.join predicate blocks by 1,024 pairs and 32,768 rendered tokens.
  B0+1c routes Predicate through B0. B1 pipelines eligible map→extract blocks.
- B0/B1 do **not** claim to bound final postprocessed Series or final result
  materialization. B1 additionally documents that an earlier extract block may
  have dispatched before a later map block fails, while the query still aborts.
- TD-3334's deterministic fake/local pre-check proves `iter_completions` and
  legacy `get_completions` both accumulate LMMetrics; B0 has no iterator-path
  metrics blind spot. The canonical typedef sync survey found semantic
  operations separated by deliberate `cache()`/`count()` barriers, with no
  unmaterialized production N-op chain. B1 therefore covers production today.
  This is not a permanent typedef constraint: TD-4372 and TD-3383 are reducing
  pass gaps. The first resulting production multi-op chain reactivates TD-3334,
  using the validation matrix as its sizing prior.
- TD-3384 proves and removes three recurrent eager no-ops: ingestion's
  all-column `select`, same-order union alignment, and direct-column identity
  projection. It preserves coercion, differing-order union, aliases/computed
  projections, schemas, and empty frames; negative guard-boundary tests cover
  aliases, casts, regex selectors, and reorderings. Source/sink/restore
  candidates remain deliberately unchanged. Future freeze records must be
  committed before dispatching their pinned Herd review.
- TD-3385 measured actual native-Polars array alternatives at 384 and 1,536
  dimensions. They are 20% to 541% slower than the NumPy paths; native dot also
  changes reduction results beyond the predeclared tight parity tolerance. Its
  no-code recommendation is frozen pending Herd Command review.

The current Captain-facing validation report is
`.context/outbox/fenic-exec-engine-validation-report-2026-08-06.md`.
