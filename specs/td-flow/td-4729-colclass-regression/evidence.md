# TD-4729: the reported column-classification streaming regression is provider drift

**Verdict.** The +12.6% streaming regression that the TD-4768 Mattermost
rerun reported for column classification (300 requests, window 100,
structured outputs, gpt-5.4-nano low) is not caused by this branch's
admission-watermark streaming design. No client-side mechanism exists at
that shape. The delta is provider service-time drift, which three
interleaved repetitions cannot average out. This document is the distilled
evidence; the raw outputs live in the td-evidence drop cited at the end.

**Branch under test:** `brandon/exec-engine-streaming-foundation-pr2`,
production source at `7ec5d26d39044665c40ffae77103f91a71cfc7d6` (this
mission adds tests and docs only; `git diff 7ec5d26..HEAD -- src/` is
empty).

## 1. The charge's three leads, resolved

**Cap saturation (refuted).** The claim that the pending cap
`min(1000, 3L)` = 300 equals the stage's request count assumed a
look-ahead basis of 100. The basis is `max(batch_size, rpm)`
(`src/fenic/_inference/model_client.py`), and typedef configures its
OpenAI clients at `rpm=15_000`, so the caps were 1,000 pending / 50,000
completed. The TD-4768 raw cells prove no cap ever bound: every arm B cell
recorded exactly 1,800 lifecycle events = 300×3 lifecycle + 300×3
streaming-stage events, with zero `window_advance`, `slot_wait`, or
`completed_cap_blocked` events — the decomposition only works if all 300
requests were admitted in the iterator's initial upfront loop.

**The 300-versus-290 request count (resolved, not a retry).** Solving both
arms' event totals jointly (A: 3Q+2R+L = 900; B: 6Q+2R+L = 1,800;
provider calls Q+R = 300) forces Q=300 queued, R=0 retries, L=0
rate-limit flags. The extra 10 requests over the offline recount are the
per-model `classify_query_metadata_batch` call embedded at the end of
`classify_columns_batch_narrowed` (typedef
`column_classification.py:3242`): one request per target table, 10
tables. Confirmed provider-free by capturing the live prepare loop's
chunk rows (exactly 290) against the committed cache, and again at full
scale by this mission's probe (300 queued, batches of 290 + 10, zero
retries). Filed as TD-4966. TD-4768 freeze comparability limit 1 is
retired.

**Structured outputs (null).** The real narrowed-contract schema ran
through every provider-free round and the real-provider probe below; it
adds no arm-asymmetric cost.

## 2. Provider-free: the client machinery measures clean

A harness drives the real Map operator → LanguageModel → ModelClient with
a simulated provider at the exact shape: 290 requests, the narrowed
schema as structured output, 8,070-token prompts, 447-token JSON
responses, rpm 15,000 / tpm 30,000,000, deterministic per-index latencies
identical across arms (the provider term cancels), fresh process per
cell, arms interleaved, collector attached and detached.

| Round                   | Environment                                                                            | Median wall A | Median wall B |    B−A |
| ----------------------- | -------------------------------------------------------------------------------------- | ------------: | ------------: | -----: |
| Baseline                | fenic venv, python 3.11                                                                |      14.275 s |      14.260 s | −0.11% |
| Baseline, collector off | same                                                                                   |      14.259 s |      14.292 s | +0.23% |
| Enriched                | typedef's env (python 3.12.12), 100-connection pool waves, 2 ms/request event-loop CPU |      18.178 s |      18.117 s | −0.33% |
| Enriched, collector off | same                                                                                   |      18.156 s |      18.108 s | −0.26% |

Every streaming-only cost measured directly: window admission 6–9 ms
total, response drain ~1 ms, per-response postprocess ~17 ms, the
duplicate request-fingerprint computation under 50 ms, stage-event
emission indistinguishable from off. The entire asymmetric surface sums
to tens of milliseconds against the reported 2,000 ms gap.

## 3. Real provider: the regression does not replicate

Six full-scale cells (three interleaved pairs, matching the TD-4768
design) on the same committed Mattermost cache through the TD-4768
harness lineage, with one addition: the complete per-request lifecycle
event stream is retained. All cells cache-clean (0.5–0.9% cached input
tokens, under the ruled 1% bar) with zero retries.

| Pair | Arm A wall s | Arm B wall s |    B−A |
| ---- | -----------: | -----------: | -----: |
| 1    |       14.737 |       13.699 | −1.038 |
| 2    |       14.063 |       13.969 | −0.094 |
| 3    |       14.897 |       22.388 | +7.491 |

Median walls: A 14.737 s, B 13.969 s (B −5.2%). The paired deltas have
mixed signs; there is no consistent arm effect.

Pair 3 is the drift mechanism caught on camera. B3's event stream shows
its requests dispatched normally (pacing p50 925 ms, admission 7 ms,
drain 0.4 ms, zero retries) and then two requests sat in provider service
for 15.1 s and 18.4 s — against an 11.5 s service maximum in the A3 cell
minutes earlier — while the cell's service median also drifted from
~6.0 s to 7.1 s. Wall time at this shape is bounded by the slowest
requests of two sequential collects, so single-cell walls inherit
provider tails at the multi-second scale. The TD-4768 record's 1.7–2.2 s
deltas sit well inside this observed envelope, and its three same-sign
pairs (probability 1/8 under drift, with its two B cells run back to
back) are unremarkable.

The event streams also decompose every wall identically in both arms:
290-request collect span, a ~0.2 s seam, the embedded 10-request
query-metadata collect (1.5–1.7 s, no arm pattern), and a sub-15 ms tail.
The same drift attribution covers the freeze record's query-metadata
stage (+37.5% on a 10-request cell it itself flags as the noisiest).

## 4. The permanent guard

`tests/_inference/test_streaming_parity_column_classification_shape.py`
pins the parity conclusion provider-free: batch and streaming arms at the
shape's client-side profile must stay at wall-clock parity under
deterministic identical latencies, with ordered, correctly parsed
structured outputs. A negative control forces the streaming caps to
(1, 1) and asserts the harness detects the resulting admission
serialization as an order-of-magnitude wall collapse. Mutation check:
applying the serialized caps to the parity test's streaming arm fails the
assertion at 4.506 s versus a 0.189 s batch wall.

## 5. The matrix wins are intact

This mission changes no production source, so the td-4925 matrix results
apply to this branch's code by content identity. Two guards on top of the
identity argument:

- The td-4925 provider-free W=32/W=100 grading cells run a bounded-join
  scenario that requires descendant-branch code (`Join(pair_block_size=…)`)
  and never measured #364 alone — attempting them against this branch
  fails at workload construction, which itself re-confirms the identity
  boundary.
- The two real-provider matrix cells nearest the regressing shape (map,
  W=100, 200-unique and 1,000-unique) were re-run against this branch at
  the freeze head, three interleaved reps per arm, all runs cache-clean
  with zero rate limiting:

  | Cell                     | Standard walls s | Streaming walls s  | Median delta |
  | ------------------------ | ---------------- | ------------------ | -----------: |
  | map, 200 unique, W=100   | 2.74, 5.30, 5.44 | 2.59, 2.65, 2.72   |       −50.0% |
  | map, 1,000 unique, W=100 | 9.38, 9.82, 9.96 | 8.85, 11.89, 12.16 |       +21.0% |

  The 200-unique win held and grew. The 1,000-unique cell is the drift
  envelope again, now with receipts on both sides of it: the streaming
  arm's own submission-stage total swung from 5.2 s (rep 1, which beat
  every standard rep) to 7.5–7.6 s (reps 2–3) — rep-to-rep variance in
  per-request client submission cost under machine load, on identical
  code — while the 200-unique _standard_ walls swung 2× (2.7 s to 5.4 s)
  from provider drift in the same session, and both arms' absolute walls
  sit ~50% above the td-4925 campaign's for the same cells. A 3-rep
  sample inside that envelope cannot overturn the identity argument, and
  reading it as a new regression would repeat exactly the 3-rep
  overreach this investigation corrected. td-4925's accepted grade for
  this cell (−0.6%) stands on the unchanged code.

## 6. Raw evidence

td-evidence drop:
`td-4729-colclass-regression-v1/investigation-2026-08-24/` on branch
`brandon/td-4729-colclass-regression-v1` — probe cell records with full
event streams, provider-free round results, the matrix-win re-check
receipts, and the harness scripts that regenerate all of it. Each file is
sha256-listed in the drop's `manifest.json` with the generating repo and
commit.

Decision records for this mission (protocol, uncommitted by design) are in
the mission worktree's `.context/outbox/`: the STOP 1 diagnosis plan, the
Step 0 resolution, the Step 1 falsification, the STOP 2 attribution, and
the freeze record.
