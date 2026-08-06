# Execution-engine experiment daily digest — TD-3334 survey frozen — 2026-08-06

```text
2026-08-06 / fenic-exec-engine-breakdown-v1 / BASE STACK COMPLETE; VALIDATION REPORT ACCEPTED; TD-3334 SURVEY FROZEN
receipts: every ratified branch is pushed and LAND. Final B1 receipt chain is 2ee449b (Herd LAND-WITH-FOLLOWUPS) -> bd7d89b (directed test/docs closure, no separate review round). Final fake/local gate: 45 passed; ruff and diff checks clean.
memory: all bounds are working-set contracts, not blanket RSS-win claims. B0 does not bound final operator output; B1 does not bound final projection materialization. B1’s deterministic idle comparison is 30 ns serial -> 0 ns fused handoff.
correctness: P1c’s DO-NOT-LAND caught real cross-side lineage corruption and silent one-pair token-budget acceptance; delta LAND confirmed both fixes. B1 documents and tests its deliberate prior-extract-dispatch-before-later-map-abort divergence.
idle: P0 metric semantics remain authoritative: queue/rate-limit separate; retry backoff excluded. B1 now asserts its full lifecycle event sequence.
validation: Herd accepted the original honest HOLD, authorized one Amendment A, and accepted the final report. The hardened harness persisted all seven synthetic raw/normalized comparisons, LMMetrics, and lifecycle event receipts before asserting. Full matrix parity passed: fusion wall time was 55.65% faster at 64 rows and 46.54% faster at 160, but 2.58% slower at 320; this is a conditional signal, not a universal throughput claim. Join 16×16 returned 64/64 expected survivors in 7,614.994 ms at 304,168,960 B peak RSS.
idle: P0 metric treatment stayed intact. There were 134 lifecycle `rate_limited` transitions, but zero retried/failed events and zero excluded rate-limited idle ns; no provider 429/backoff claim is made. Fusion idle gap improved at 64/160 and regressed at 320, matching the mixed wall result.
spend: original stopped arm remains a $0.041574 conservative upper bound. Amendment A's serialized LMMetrics report $0.000000 tokens/requests/cost despite non-empty lifecycle events, a telemetry limitation not a free-provider claim. Cumulative conservative planning bound $0.785305, below $50.
TD-3334 pre-check: the $0 fake/local regression test passed (2 passed): B0's `iter_completions` path and legacy `get_completions` both accumulate deterministic LMMetrics, while a distinct unused client remains zero. No iterator-path metrics fix card is warranted; the live zero metrics are retained as a provider/telemetry limitation.
TD-3334 survey: the canonical typedef data-intelligence sync paths contain a 12-pass logical sequence, but every pass is immediately `cache()`d and `count()`ed. Each physical plan contains at most one semantic operation; other surveyed production uses likewise materialize after one. No generalized N-op code is justified: B1 already covers the relevant two-op shape, and removing barriers would change failure/resume/cache/progress semantics. The 320-row governor/TPM hypothesis remains a hypothesis, not a conclusion.
next gate: `.context/outbox/td3334-n-op-fusion-lane-record-2026-08-06.md` is frozen for Herd Command review. TD-3384 and TD-3385 remain HOLD pending disposition.
```

The current Captain-facing report is
`.context/outbox/fenic-exec-engine-validation-report-2026-08-06.md`.
