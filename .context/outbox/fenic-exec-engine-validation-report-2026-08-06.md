# Real-provider validation epilogue — HOLD report

**Lane:** `herd/fenic-exec-engine-validation` stacked at B1 final `bd7d89bc8301387f36652a7be2ea7ad66f3edefd`  
**Status:** HOLD — safety stop on real/fake semantic divergence; report review required before the charter proceeds to TD-3384.  
**Design and receipt:** `.context/outbox/fenic-exec-engine-validation-lane-record-2026-08-06.md`

## Outcome

The Captain-authorized real-provider probe did **not** establish a live fusion gain. The first, unfused 64-row map→extract baseline returned 64 rows but did not reproduce the deliberately literal fake-client `record_id → category` expectation. That is precisely the predeclared semantic-divergence stop condition. The run halted before the fused comparator, larger matrix sizes, and the bounded semantic.join arm. No retry, diagnosis, or rate-limit induction was performed.

This is not an implementation regression claim. It is an honest statement that the real-provider task/prompt behaviour differed from the synthetic fake path, so treating the requested wall-clock or idle-gap difference as an execution engine gain would be invalid.

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

## Requested measurements

| Goal | Live result | Interpretation |
|---|---|---|
| Fusion gain: unfused versus fused wall-clock, P0 idle gap, tokens, cost | Not measured | Comparator was not permitted after semantic divergence. No performance claim. |
| Bounded semantic.join: 16×16 memory, wall-clock, survivor parity | Not run | The code's reviewed bound remains intact, but this report contributes no live-provider validation. |
| Natural 429/backoff observation | None visible before the stop | Lifecycle events were not serialized; do not infer a positive absence or calculate excluded idle time. |

## Caveats retained from the landed stack

- B0 bounds raw request/future/raw-response working sets, not final `BaseOperator.execute` postprocessed-output materialization.
- B1 similarly does not claim to bound final projection/result materialization.
- In an eligible fused chain, an earlier extract block can dispatch before a later map block fails; the query still aborts. This was documented and tested in the B1 closeout.
- The real-provider result is a prompt/model parity finding, not a diagnosis of those execution bounds.

## Charter handoff

TD-3384 then TD-3385 remain acknowledged as the next local-only, `$0` nodes, but they do not start until Herd Command accepts this validation HOLD/report outcome. This report is now frozen for that final **report review**, not a code review.
