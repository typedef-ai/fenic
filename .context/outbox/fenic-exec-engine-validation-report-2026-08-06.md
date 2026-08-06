# Real-provider validation epilogue — HOLD report

**Lane:** `herd/fenic-exec-engine-validation` stacked at B1 final `bd7d89bc8301387f36652a7be2ea7ad66f3edefd`  
**Status:** Amendment A COMPLETE — normalized semantic parity passed across the full matrix; final report review required before TD-3334.
**Design and receipt:** `.context/outbox/fenic-exec-engine-validation-lane-record-2026-08-06.md`

## Outcome

The original probe halted correctly on its first unfused arm, but Herd Command authorized one bounded Amendment A after accepting that HOLD. Amendment A persisted value-level, synthetic evidence before every parity assertion, tightened the category constraint, and completed the full matrix with no normalized semantic divergence.

The result establishes a live fusion signal at 64 and 160 rows, but it is not monotonic at 320 rows. The valid conclusion is therefore conditional: fusion can materially reduce end-to-end wall time on these small/mid synthetic workloads; it is not a universal live-provider throughput win.

## Probe design and spend control

The design was written before any keyed call. It used only `.env` credentials (mode `600`; values never read or printed), a fresh temporary session/database per arm to prevent response-cache reuse, and `gpt-4.1-nano` at conservative `250 RPM / 50,000 TPM`. The direct design estimate was `$0.634061`; its 10× uncertainty reserve was `$6.340610`, below the `$40` go/no-go ceiling and `$50` hard backstop.

The failing arm completed 64 map and 64 extract requests before its post-result parity assertion. It raised before serializing the in-memory Fenic `LMMetrics`, so an exact actual-cost receipt was not retained. This is a harness deficiency, not evidence of zero spend. A conservative upper bound using 1,200 input and the observed 512-token output reservation for each of the 128 completed requests is **$0.041574**. No other live arm ran.

The actual extracted category values were likewise not serialized before the count-only assertion raised. `.context/outbox/fenic-exec-engine-validation-divergence-evidence-2026-08-06.md` records this second harness deficiency. It is evidence that divergence occurred, not a value-level comparison; the report makes no stronger claim.

| Run | Estimate | Actual cost receipt | Result |
|---|---:|---:|---|
| Unfused map→extract, 64 rows | `$0.035021` under the frozen 384-output estimate | unavailable (safety stop before metric serialization); conservative upper bound `$0.041574` | **HOLD**: 64 rows returned, semantic parity failed |
| Fused map→extract, 64/160/320 rows | `$0.297677` combined frozen estimate | `$0.000000` — not run | blocked by first-arm HOLD |
| semantic.join, 16×16 | `$0.038707` frozen estimate | `$0.000000` — not run | blocked by first-arm HOLD |

The `$0.000000` entries mean no request was issued for those arms, not that the first arm was free. Aggregate exact live spend is therefore **unavailable**; aggregate conservative post-run upper bound is **$0.041574**, far below `$50`.

## Amendment A complete matrix

Every Amendment A arm wrote its synthetic expected/observed values, normalized comparison, LMMetrics, lifecycle summary, and raw lifecycle events before it could assert. The seven durable receipts are under `.context/validation/amendment-a-evidence/`; for example, `unfused-64.json` records `ALPHA/BETA/GAMMA/DELTA` exactly for record IDs 0–3 and has zero mismatches.

| Rows | Wall unfused → fused | Wall change | Non-rate-limited idle unfused → fused | Queue-delay sum unfused → fused | Result |
|---:|---:|---:|---:|---:|---|
| 64 | 7,767.452 → 3,445.045 ms | **−55.65%** | 7.254 → 2.069 ms (**−71.48%**) | 387.249 → 908.952 ms | parity pass |
| 160 | 10,684.165 → 5,711.820 ms | **−46.54%** | 10.483 → 7.994 ms (**−23.74%**) | 10,336.276 → 9,976.039 ms | parity pass |
| 320 | 22,416.596 → 22,995.301 ms | **+2.58%** | 18.737 → 21.780 ms (**+16.24%**) | 286,008.091 → 13,466.541 ms | parity pass; do not claim gain |

The P0 collector excludes identified rate-limited portions from the non-rate-limited idle value. It emitted 134 `rate_limited` transitions across the matrix, but each arm reported `total_rate_limited_ns: 0`; there were zero `retried` and zero `failed` events. This is governor lifecycle telemetry, not evidence of a provider 429/backoff, and no rate limiting was induced.

The bounded `semantic.join` 16×16 arm returned all 64 expected survivors, took 7,614.994 ms, and recorded a process peak RSS of 304,168,960 bytes. Its pair cap remained 1,024. This is a single live bounded-behaviour receipt, not a before/after memory comparison.

### Cost receipt caveat

Amendment A's per-arm serialized LMMetrics each report `0` requests, tokens, and dollars despite non-empty queued/dispatched/settled lifecycle evidence. Record those values as the requested **reported LMMetrics actuals** (`$0.000000` total), but they are not a reliable billing receipt and must not be read as proof that the provider was free. The original arm's conservative upper bound remains `$0.041574`; the Amendment A direct reserve was `$0.743731`, for a conservative cumulative planning bound of **$0.785305**, well below `$50`. No attempt was made to diagnose or rerun this telemetry limitation.

## Requested measurements

| Goal | Live result | Interpretation |
|---|---|---|
| Fusion gain: unfused versus fused wall-clock, P0 idle gap, tokens, cost | Measured in Amendment A | Strong at 64/160, mixed at 320; token/cost LMMetrics are zero-valued telemetry and non-authoritative. |
| Bounded semantic.join: 16×16 memory, wall-clock, survivor parity | 64/64 expected survivors; 304,168,960 B peak RSS; 7,614.994 ms | Bounded live receipt only; no baseline memory comparison. |
| Natural 429/backoff observation | 134 governor `rate_limited` transitions; no retry/failure; zero excluded rate-limited idle ns | Observational only; no provider 429 claim. |

## Caveats retained from the landed stack

- B0 bounds raw request/future/raw-response working sets, not final `BaseOperator.execute` postprocessed-output materialization.
- B1 similarly does not claim to bound final projection/result materialization.
- In an eligible fused chain, an earlier extract block can dispatch before a later map block fails; the query still aborts. This was documented and tested in the B1 closeout.
- The real-provider result is a prompt/model parity finding, not a diagnosis of those execution bounds.

## Charter handoff

TD-3334 generalized N-op fusion is now next, ahead of TD-3384 and TD-3385, but it does not start until Herd Command accepts this amended validation report. This report is frozen for the final **report review**, not a code review.
