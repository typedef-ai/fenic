# TD-3385 NumPy vs native-Polars embedding math — lane record

**Status:** LAND-WITH-FOLLOWUPS; directed reproducibility and test-selection
closure in this record.
**Branch:** `herd/fenic-exec-engine-td3385-polars-embedding-benchmark`, stacked
on TD-3384's directed-closure head `9e0ce56`.

## Scope and spend

This is a local-only Polars/NumPy benchmark. Estimated and actual provider spend
are **$0.00**; no `.env` value was read. The fixture uses seeded synthetic
`Float32` arrays, not embeddings or provider completions. Polars 1.43.1 is the
project environment. A separate isolated compatibility probe verified the
candidate `arr.eval`, `arr.sum`, fixed-array scalar arithmetic, and null/zero
behavior on Polars 1.35.1. `1.35.0` itself is unavailable because its runtime
wheel has been yanked; no product compatibility claim depends on that yanked
patch release.

## Candidate expressions and parity boundary

The native candidates are real expressions, not a straw man:

- normalize: `array / sqrt(array.arr.eval(element * element).arr.sum())`;
- similarity: fixed-array multiplication/subtraction followed by `arr.sum()`
  (and norms for cosine), for both column-to-column and literal query vectors;
- average: `arr.to_struct()`, one scalar `mean()` per dimension, then
  `concat_arr`, guarded by non-null vector count.

Small explicit `Float32` fixtures covered ordinary vectors, null vectors, and
zero vectors. Normalize preserved `Array(Float32, D)`, nulls, and all-NaN zero
vectors. Dot/cosine/L2 preserved `Float32`, null propagation, and cosine NaN
for a zero denominator. The average candidate preserved `Array(Float32, D)`,
ignored null vectors, and returned a null vector for an all-null group.

That behavioral agreement is insufficient to replace the existing paths:
on a 32,768 x 384 dense fixture, native dot changes floating-point reduction
order. Pairwise dot differs at 30,104/32,768 rows, with maximum absolute delta
`5.7220458984375e-05`, and fails the benchmark's `rtol=atol=2e-6` numerical
parity guard. The vector-query dot candidate also fails that guard. This is a
compatibility blocker even before performance is considered.

## Micro-measurement

All timings are medians of execution-only runs; fixture construction is excluded.
The 384-dimensional matrix used 32,768 rows, three warmups, and nine timed
rounds. Its average case used 128 groups of 256 vectors. The 1,536-dimensional
matrix used 8,192 rows, three warmups, and seven timed rounds. Values compare
the current NumPy `map_batches` logic with the candidate native expression.
Negative gain means native Polars is slower.

| Shape | Operation | NumPy median | Native median | Native gain | Parity result |
| --- | --- | ---: | ---: | ---: | --- |
| 32,768 x 384 | normalize | 8.063 ms | 15.671 ms | -94.4% | pass |
| 32,768 x 384 | column dot | 3.241 ms | 9.032 ms | -178.6% | fail (reduction order) |
| 32,768 x 384 | column cosine | 8.437 ms | 9.329 ms | -10.6% | pass |
| 32,768 x 384 | column L2 | 4.239 ms | 14.351 ms | -238.5% | pass |
| 32,768 x 384 | query dot | 3.683 ms | 11.127 ms | -202.1% | fail (reduction order) |
| 32,768 x 384 | query cosine | 6.236 ms | 19.037 ms | -205.3% | pass |
| 32,768 x 384 | query L2 | 4.719 ms | 28.078 ms | -494.9% | pass |
| 32,768 x 384; 128 x 256 groups | embedding average | 344.498 ms | 1,653.003 ms | -379.8% | pass |
| 8,192 x 1,536 | normalize | 8.560 ms | 19.519 ms | -128.0% | pass on 384 boundary matrix |
| 8,192 x 1,536 | column dot | 3.196 ms | 12.215 ms | -282.2% | not a replacement |
| 8,192 x 1,536 | column cosine | 7.830 ms | 12.700 ms | -62.2% | not a replacement |
| 8,192 x 1,536 | column L2 | 4.356 ms | 23.010 ms | -428.2% | not a replacement |
| 8,192 x 1,536 | query dot | 3.470 ms | 14.799 ms | -326.5% | not a replacement |
| 8,192 x 1,536 | query cosine | 5.954 ms | 25.849 ms | -334.1% | not a replacement |
| 8,192 x 1,536 | query L2 | 4.380 ms | 29.934 ms | -583.4% | not a replacement |

## Recommendation — no code change

Retain the NumPy implementations of `embedding.normalize`,
`embedding.compute_similarity`, and embedding `avg`. No candidate is a clear
parity-preserving win. Normalize and all similarity candidates are slower on
both representative dimensions; dot also changes numerical results beyond the
predeclared tight guard. Native average requires O(dimensions) expression
construction (`to_struct` plus one aggregate per component) and is 4.80x slower
at 384 dimensions. It is explicitly not an optimization for high-dimensional
embedding models.

Revisit only with a materially different Polars array kernel that both preserves
the existing numerical contract and wins on a documented production-shaped
fixture. This is a charter-authorized benchmark-backed no-op, not evidence that
native Polars array expressions are generally unsuitable.

## Verification and process incident

The provider-free production behavior gate passed: **18 passed** — normalize,
all three similarity metrics in both forms, null/zero/dtype/validation cases,
and embedding-average including null vectors. It used only construction-local
test model aliases; no semantic embedding expression was executed.

An initial over-broad selector accidentally included three existing tests that
call the real embedding provider. The known placeholder key received HTTP 401
responses before any successful provider response; no `.env` value was read,
no valid credential was used, and no charge was incurred. The command was
stopped, excluded from this receipt, and no provider-capable test was rerun.
This is a test-selection incident and reinforces TD-3384's earlier lesson.

## Reproducibility closure

The original freeze contained conclusions but not replayable timed receipts.
Herd Command directed this closure before marking the admitted charter complete.
The exact seeded local harness is now committed at
`.context/validation/td3385-evidence/benchmark_embedding_math.py`; it writes
the raw per-round arrays in `benchmark-results.json` and the independent
dot-reduction counterexample in `parity-boundaries.json`. The receipt records
Polars 1.43.1, NumPy 2.4.6, three warmups, each timed round, fixed fixture
shapes/seeds, and the `2e-6` tolerance. Running its documented command rewrites
only those local evidence files and makes no network or provider call.

The reproduction yielded the medians above. Its pair-dot receipt again reports
30,104 differing 384-D rows and maximum absolute delta
`5.7220458984375e-05`; query-dot independently reports 29,998 differing rows
at the same maximum delta. `uvx ruff check` passes for the committed harness.

The accompanying `.context/PROVIDER-FREE-TEST-SELECTION.md` makes the standing
fix explicit: local-only nodes use inspected, exact pytest node IDs only and
exclude the known provider-capable metrics, embedding, reader, sim-join, and
semantic-join targets. A placeholder key is construction-only, never proof of
provider-free execution.

## Herd disposition and charter state

Herd Command **LAND-WITH-FOLLOWUPS**ed the no-op recommendation at `72323f7`:
the dot reduction-order divergence blocks replacement by itself and the uniform
slower measurements make it moot. Herd independently verified TD-3384's
directed closure at `9e0ce56`. This one directed evidence/docs commit closes the
two follow-ups; Herd verifies it directly and requires no new review round.

The original admitted charter is complete: real-provider validation/report,
TD-3334's evidence-backed no-op, TD-3384's three fast paths, and TD-3385's
benchmark-backed no-op are all recorded. The former stand-down is superseded by
the Captain's subsequent multi-step workload-benchmark extension; that new
node begins only after this closure is committed and pushed.
