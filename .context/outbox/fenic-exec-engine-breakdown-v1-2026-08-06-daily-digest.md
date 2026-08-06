# Execution-engine experiment daily digest — validation Amendment A complete — 2026-08-06

```text
2026-08-06 / fenic-exec-engine-breakdown-v1 / BASE STACK COMPLETE; VALIDATION AMENDMENT A COMPLETE
receipts: every ratified branch is pushed and LAND. Final B1 receipt chain is 2ee449b (Herd LAND-WITH-FOLLOWUPS) -> bd7d89b (directed test/docs closure, no separate review round). Final fake/local gate: 45 passed; ruff and diff checks clean.
memory: all bounds are working-set contracts, not blanket RSS-win claims. B0 does not bound final operator output; B1 does not bound final projection materialization. B1’s deterministic idle comparison is 30 ns serial -> 0 ns fused handoff.
correctness: P1c’s DO-NOT-LAND caught real cross-side lineage corruption and silent one-pair token-budget acceptance; delta LAND confirmed both fixes. B1 documents and tests its deliberate prior-extract-dispatch-before-later-map-abort divergence.
idle: P0 metric semantics remain authoritative: queue/rate-limit separate; retry backoff excluded. B1 now asserts its full lifecycle event sequence.
validation: Herd accepted the original honest HOLD and authorized one Amendment A. The hardened harness persisted all seven synthetic raw/normalized comparisons, LMMetrics, and lifecycle event receipts before asserting. Full matrix parity passed: fusion wall time was 55.65% faster at 64 rows and 46.54% faster at 160, but 2.58% slower at 320; this is a conditional signal, not a universal throughput claim. Join 16×16 returned 64/64 expected survivors in 7,614.994 ms at 304,168,960 B peak RSS.
idle: P0 metric treatment stayed intact. There were 134 lifecycle `rate_limited` transitions, but zero retried/failed events and zero excluded rate-limited idle ns; no provider 429/backoff claim is made. Fusion idle gap improved at 64/160 and regressed at 320, matching the mixed wall result.
spend: original stopped arm remains a $0.041574 conservative upper bound. Amendment A's serialized LMMetrics report $0.000000 tokens/requests/cost despite non-empty lifecycle events, a telemetry limitation not a free-provider claim. Cumulative conservative planning bound $0.785305, below $50.
next gate: `.context/outbox/fenic-exec-engine-validation-report-2026-08-06.md` is frozen for Herd Command **report review**. TD-3334 generalized N-op fusion is next by Captain order, ahead of TD-3384/TD-3385, but does not start until this gate resolves.
```

The current Captain-facing report is
`.context/outbox/fenic-exec-engine-validation-report-2026-08-06.md`.
