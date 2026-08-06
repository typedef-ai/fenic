# Execution-engine experiment closing report — extended charter complete

**Mission:** `fenic-exec-engine-breakdown-v1`
**Status:** COMPLETE — every admitted and subsequently extended-charter node is
LAND/no-op, pushed, and recorded below. The experiment is now on stand-down for
the Captain's return review.
**Headline:** the full experiment closed in one day rather than its planned
week-scale envelope. The speed came from narrow, review-gated slices and
deterministic local evidence—not from relaxing correctness, spend, or review
discipline.

## Final stack map

```text
P0 @834f5c08d  lifecycle foundation + evidence harness
 └─ P1a @c8d797a  sim-join left-slice bound
    └─ P1b @c8d797a  reviewed no-op for reduce
       └─ B0 @c7967d092  row-local ModelClient iterator
          └─ P1c @743e09b  bounded semantic.join blocks
             └─ B0+1c @574a743  Predicate-on-B0 adapter
               └─ B1 @bd7d89b  bounded map→extract fusion
                  └─ Validation @9098734  report accepted
                     └─ TD-3334 @54ea844  accepted no-op
                        └─ TD-3384 @f0fcc14 + 9e0ce56  landed eager-no-op paths
                           └─ TD-3385 @72323f7 + directed closure  landed no-op
                              └─ Mixed workload @b2de73a  LAND-WITH-FOLLOWUPS
                                 └─ TD-3383 @0b66160  accepted no-op
```

| Node | Final outcome | Review history / receipt |
| --- | --- | --- |
| P0 | Lifecycle events/idle metrics, sorted sim-join golden, toolchain pin, baseline harness | LAND-WITH-FOLLOWUPS; three Minors closed in `834f5c08d`. `.context/td-review/p0-foundation-review-2026-08-05.md` |
| P1a | Lance table once; 1,024-left-row slices for search/explode/unnest; final N×k result explicitly remains an output boundary | LAND. `.context/td-review/p1a-sim-join-c8d797a.md` |
| P1b | No code: current reduce is serial and short-document RSS evidence could not establish a material residual operator signal | LAND as no-op with amended evidence ledger. `.context/td-review/p1b-reduce-residual-c8d797a.md` |
| B0 | 100-request `iter_batch_requests`, `iter_completions`, row-local map/extract/classify stream opt-in, preserved list API for reduce | Initial REVIEW_INFEASIBLE was superseded by LAND after restored Claude→Codex review. `.context/td-review/b0-model-stream-c7967d0-review-2026-08-05.md`, `.context/td-review/b0-model-stream-c7967d0-review-2026-08-06.md` |
| P1c | 1,024-pair resident cap plus 32,768 rendered-token budget; block-nested semantic.join | DO-NOT-LAND at `01a0bf008`, then delta LAND at `743e09b`. `.context/td-review/p1c-semantic-join-review-2026-08-05.md`, `.context/td-review/p1c-semantic-join-review-2026-08-06-delta.md` |
| B0+1c | Predicate uses B0 iterator inside P1c blocks; direct transpiler predicate proof added | LAND-WITH-FOLLOWUPS at `94a9ec3`; one test-only Minor closed at `574a743`. `.context/td-review/b0-1c-stream-adapter-receipt-2026-08-05.md` |
| B1 | Eligible string map→extract chains pipeline B0 blocks; initial `2ee449b`, final test/doc delta `bd7d89b` | LAND-WITH-FOLLOWUPS, no Majors; two Minors closed directly by Herd direction. `.context/td-review/b1-two-op-fusion-review-2026-08-05.md` |
| Validation | Captain-authorized real-provider matrix with Amendment A hardening | Report ACCEPTED. `.context/outbox/fenic-exec-engine-validation-report-2026-08-06.md` |
| TD-3334 | Fake metrics pre-check and typedef production-shape survey | Accepted no-op: deliberate current per-pass barriers leave no unmaterialized N-op production chain. `.context/outbox/td3334-n-op-fusion-lane-record-2026-08-06.md` |
| TD-3384 | Three behavior-preserving eager no-op fast paths | LAND; directed guard/doc closure at `9e0ce56`. `.context/td-review/td3384-eager-noop-review-2026-08-06.md` |
| TD-3385 | Native-Polars embedding math benchmark | LAND-WITH-FOLLOWUPS as no-op; directed closure commits replayable harness/raw receipts and provider-free test rule. `.context/outbox/td3385-polars-embedding-benchmark-lane-record-2026-08-06.md` |
| Mixed workload | Real-clock, $0, 12-step mixed-complexity simulated workload across barriered/unfused/fused arms | LAND-WITH-FOLLOWUPS at `b2de73a`; the direct closure strengthened step-content parity, resume provenance, fusion assertions, and synthetic-key hermeticity. `.context/td-review/mixed-workload-review-2026-08-06.md` |
| TD-3383 | Local collect/show/count pushdown boundary audit and narrow `show(n)` candidate | Accepted no-op at `0b66160`: generic pushdown is unsafe; the candidate visibly changed Polars first/last sampling to a first-ten prefix. `.context/outbox/td3383-collect-show-count-pushdown-lane-record-2026-08-06.md` |

The complete SHA/receipt matrix is the final anchor body:
`.context/outbox/p0-experiment-anchor-pr-body.md`.

## Evidence and spend

- **Total direct actual receipt: $0.00; conservative planning bound:
  $0.785305.** The Captain-authorized validation ran under the $50 backstop.
  Its original stopped arm retains a `$0.041574` upper bound; Amendment A
  serialized LMMetrics reported zero despite lifecycle activity, a telemetry
  limitation rather than a free-provider claim. Every other node is local/fake.
- P0 deterministic serial map→extract baseline: one 30 ns gross idle gap,
  p50/p95 30 ns, queue delay 20 ns; rate-limit wait is separately reported and
  retry backoff is excluded from idle/queue attribution.
- B1 deterministic fused fixture: one zero-width interval / 0 ns gross idle,
  10 ns queue delay, with the full queued→dispatched→settled sequence now
  asserted. This is a scheduler-handoff proof, not a live-provider latency
  claim.
- The final B1 fake/local gate was **45 passed**, with ruff and diff checks
  clean. Its 64-row isolated-child RSS receipt was 332,349,440 B versus P0's
  320,765,952 B; this is intentionally **not** a memory-improvement claim.
- Other reported RSS values are contract/evidence receipts, not wins inferred
  from noisy child-process peaks or incomparable matrix sizes. Final output
  widths are kept separate from transient working-set claims.

## Final not-needed ledger

| Candidate | Final disposition |
| --- | --- |
| Audit-branch code cherry-pick | Not needed: the audit branch contained roadmap documentation, not implementation. |
| `semantic.reduce` semaphore / 1b packer | Not needed for tested short synthetic documents. Current reduce is serial; a count-only harness does not settle large-document retention. Revisit only with a realistic large-document signal. |
| Sim-join temp-dir cleanup redo | Not needed: current `TemporaryDirectory` behavior was retained. |
| Sim-join output projection redesign | Deferred product/API question: named vectors remain an output contract; P1a only bounds transient search work. |
| Polars allocator metric | Not available through the project’s stable interface. Peak child RSS remains the accepted metric with its noise qualification. |
| Real-provider validation | Not needed for this mission: fake/local tests exercised all intended contract boundaries at $0.00. |
| P1c response cache/memoization decision | Deliberately deferred: bounded blocks do not erase the pre-existing lineage re-execution/cost risk. |
| TD-3334 generalized N-op fusion | Not needed for typedef production today: `cache()`/`count()` separates every pass. TD-4372/TD-3383 pass-gap removal makes the first unmaterialized multi-op chain the concrete revisit trigger. |
| TD-3385 native Polars embedding replacement | Not needed: every candidate was slower, and native dot changed numerical reduction results beyond tight parity. |
| TD-3383 local action pushdown | Not needed in v0: even a cache-free direct-column `show(10)` prefix changes incumbent first/last display sampling. `count` without full materialization requires a separately designed action-mode physical API. |

## Premise kills and qualifications

1. M0 was not “80% done”: the provider idle-gap metric was absent, so the
   foundation was 75% with smoke-grade evidence until P0 instrumented it.
2. The roadmap’s reduce concurrency premise was false on current main:
   `ThreadPoolExecutor(min(10, groups))` was gone; no semaphore should be
   reintroduced.
3. The audit branch was documentation-only and stale relative to current
   sim-join extras/cleanup behavior; re-derivation beat cherry-picking.
4. B0 bounds request/future/raw-response working state, **not**
   `BaseOperator.execute`’s final postprocessed result list. That qualification
   remains true after B1; B1 also leaves final projection materialization out
   of its bounded claim.
5. B1 intentionally changes failure-side-effect timing: if a later map block
   fails, already-dispatched earlier extract blocks may have run. The query
   still propagates its established `ExecutionError`; final `bd7d89b` now
   characterizes and documents this divergence.
6. TD-3383's “safest” candidate was not safe: `show(10)` is a formatted sample
   of the fully materialized result, including tail rows. A source prefix limit
   changes visible output before any performance claim can matter.

## Review history: what the gates caught

- **P0:** LAND-WITH-FOLLOWUPS. The three Minors drove zero-cost lifecycle
  cleanup, repo lint fixes, and explicit retry-backoff accounting limitations.
- **P1a:** LAND; review confirmed B=1 observes real slices, preserves top-k,
  invalid-vector, cleanup, and output visibility contracts.
- **P1b:** LAND with amended framing. Review rejected overclaiming a
  short-document RSS experiment as a bound for large documents, preserving the
  no-op decision instead of adding speculative machinery.
- **B0:** first review direction failed twice with empty Claude output and was
  correctly recorded REVIEW_INFEASIBLE rather than self-ratified. A later
  independent Claude→Codex review LANDed B0, confirmed bounded prefetch and
  accepted the documented block-scoped dedup divergence; it surfaced the
  final-materialization qualification.
- **P1c:** the critical DO-NOT-LAND caught a live, silent backward-lineage
  corruption when `_right_uuid` existed on the left (and the symmetric case),
  plus silent acceptance of a one-pair-over-token-budget prompt. The delta
  fixed both, added the cross-direction tests, a loud `ValueError`, a
  defense-in-depth cap assertion, and explicit empty/single-row coverage; the
  delta reviewer re-reproduced the old corruption and confirmed it now raises.
- **B0+1c:** LAND-WITH-FOLLOWUPS caught the missing direct
  transpiler-built `semantic.predicate` proof. `574a743` makes
  `get_completions` fail immediately and observes B0’s real `[100, 1]` batches.
- **B1:** LAND-WITH-FOLLOWUPS caught the undocumented partial-dispatch-before-
  abort consequence and an incomplete event-order assertion. `bd7d89b` closes
  both with a fake-client characterization, one-line disclosure, and P0-style
  lifecycle sequence assertion. Herd Command directed no extra review round.
- **Mixed workload:** LAND-WITH-FOLLOWUPS caught overstated 12-step content
  parity, weak partial-third-seed provenance, a missing matrix-level fusion
  assertion, and an environment-dependent test fixture. `b2de73a` adds the
  retained-column content proof, manifest validation, `used_fusion` contract,
  and construction-only synthetic-key fixture; no matrix rerun was needed.
- **TD-3383:** the approved design then accepted no-op review treated exact
  `show(10)` output parity as a hard gate. It caught that a source prefix is not
  equivalent to Polars' first/last sample of the fully executed result, so the
  prototype was removed and the boundary-design note became the deliverable.

## Process incidents and lessons

- An externally terminated session was recovered from the ticket, ratified
  plan, lane records, and receipts with no work loss. The outbox is therefore
  validated as a durable continuity mechanism, not ceremony.
- P0’s first review began while the worktree was still changing. The reviewer
  detected drift, waited for two stable quiescence windows, then reviewed the
  stable snapshot. Future gates should explicitly verify quiescence before
  dispatch.
- Review-path confusion occurred in the B1 target context: the reviewer’s
  globally-installed independent-review skill made it self-label its findings
  “solo.” Driver-side nonce/thread/provenance checks established the actual
  Claude→Codex review was independent; this was a tooling-context artifact,
  not a reason to weaken the gate.
- Timeout truncations were handled as preflight failures, not counted reviews:
  B0’s first wrapper used a two-minute tool timeout despite a longer inner
  timeout; B1 had three similarly truncated invocation attempts before the
  driver raised the wrapper limit. The later complete responses alone were
  counted. Review commands need both inner and outer timeout budgets set.
- Placeholder-key over-broad test selectors hit pre-existing 401 paths in P1c,
  TD-3384, and TD-3385. They incurred no charge and were excluded rather than
  misreported as fake/local validation. The committed provider-free
  allowlist/exclusion rule is now part of spend discipline.

## Extended-charter completion

All extended-charter `herd/` branches are pushed. The anchor PR #358 has the
final reviewer-less stack map. The original admitted charter closed with TD-3385;
the Captain then admitted the mixed-workload bridge and TD-3383 pushdown audit.

That workload node now has a complete 12-receipt, $0 simulated matrix at
`.context/validation/mixed-complexity-workload/matrix/`, LAND-WITH-FOLLOWUPS
after Herd's independent review and its directly authorized test/docs closure.
At 96/192 rows and matching/0.90x-server TPM, the matrix verified row IDs,
overlay structs, per-step logical totals, and actual output-token draws; the
new focused retained-column proof verifies baseline step 01–11 content equality
before final projection. Every receipt recorded—and `run_matrix` now
asserts—`[false, false, true]` `used_fusion`, so the no-material-B1-overlay
gain (fused/unfused wall delta at most 0.09%) is an engaged, governor-bound
null rather than an inactive comparator. No 300s wall or 60s no-settlement guard
fired. The canonical 192-row 415,002-token expansion against a 150,000-token
bucket produced 107.5–113.1s matching arms, supporting the governor-bound
hypothesis and lane-dependent barrier effects. A documented matrix control-flow
defect briefly created an inconsistent third-seed reduction manifest; the
evidence was kept, the runner was fixed, and one bounded final simulator lane
brought active wall to 2,674.807s (25.193s under budget). The old path had
already overwritten the historical reduction object, so it is not invented
retroactively; the reusable resume path now retains it as `superseded_reduction`.
TD-3383's design was approved and its evidence-backed no-op was independently
accepted. The design note is the primary deliverable: its fail-closed classifier
marks cache, semantic/model, join, aggregate, sort, SQL, and sink paths as hard
boundaries; generic `collect`/`count` shortcuts are rejected. The one narrow
cache-free direct-column `show(10)` candidate failed exact output parity because
incumbent Polars formatting samples first/last rows of the full result, while a
source limit shows only rows 0–9. The prototype was removed without retry.
Its evidence is at
`.context/validation/td3383-pushdown/show-prefix-parity-failure-2026-08-06.md`.

The deferred next question is precise rather than speculative: count without
full materialization needs an action-mode physical-execution contract that owns
per-operator count semantics, QueryMetrics/LMMetrics, cache-write/checkpoint
behavior, error timing, and source/SQL/sink capability. It is not a hidden
optimization behind the current `count()` API.

## Final stand-down

The extended charter is complete: validation probe, TD-3334 no-op, TD-3384 fast
paths, TD-3385 no-op, mixed-workload matrix, and TD-3383 no-op. Total actual
provider spend remains **$0.00** in the local extensions (with the validation
report's conservative $0.785305 planning bound unchanged). No further nodes,
commits, or reviews are authorized; the experiment holds for Captain return
review.
