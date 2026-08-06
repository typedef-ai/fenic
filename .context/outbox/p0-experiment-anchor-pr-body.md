## Execution-engine experiment anchor — BASE STACK LAND; validation Amendment A complete

This reviewer-less draft is the single branches-only anchor for the completed
`fenic-exec-engine-breakdown-v1` experiment. It is not a review request; no
additional PRs were created for the stack or validation epilogue.

**Base stack: COMPLETE.** Every ratified implementation node below is pushed and
`LAND`. The Captain-authorized real-provider validation Amendment A is
**COMPLETE and frozen for report review** after the hardened full matrix passed
normalized semantic parity.

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
                         `-- Validation herd/fenic-exec-engine-validation @ 2281a8f  AMENDMENT A COMPLETE (report review)
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
| Validation epilogue | `2281a8f` | pending Herd Command **report** review | `.context/outbox/fenic-exec-engine-validation-report-2026-08-06.md` |

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

The current Captain-facing validation report is
`.context/outbox/fenic-exec-engine-validation-report-2026-08-06.md`.
