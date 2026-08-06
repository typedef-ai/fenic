# TD-3385 NumPy vs native-Polars embedding math — lane record

**Status:** frozen recommendation: no implementation; awaiting Herd Command
review.
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
| 32,768 x 384 | normalize | 7.363 ms | 13.064 ms | -77.4% | pass |
| 32,768 x 384 | column dot | 2.881 ms | 7.713 ms | -167.7% | fail (reduction order) |
| 32,768 x 384 | column cosine | 7.775 ms | 9.334 ms | -20.1% | pass |
| 32,768 x 384 | column L2 | 3.813 ms | 15.905 ms | -317.1% | pass |
| 32,768 x 384 | query dot | 3.270 ms | 9.814 ms | -200.2% | fail (reduction order) |
| 32,768 x 384 | query cosine | 5.737 ms | 17.025 ms | -196.8% | pass |
| 32,768 x 384 | query L2 | 4.505 ms | 24.670 ms | -447.6% | pass |
| 32,768 x 384; 128 x 256 groups | embedding average | 324.816 ms | 1,527.120 ms | -370.1% | pass |
| 8,192 x 1,536 | normalize | 7.476 ms | 15.802 ms | -111.4% | pass on 384 boundary matrix |
| 8,192 x 1,536 | column dot | 2.859 ms | 10.249 ms | -258.4% | not a replacement |
| 8,192 x 1,536 | column cosine | 7.467 ms | 12.057 ms | -61.5% | not a replacement |
| 8,192 x 1,536 | column L2 | 3.726 ms | 17.881 ms | -379.9% | not a replacement |
| 8,192 x 1,536 | query dot | 3.141 ms | 12.476 ms | -297.2% | not a replacement |
| 8,192 x 1,536 | query cosine | 5.519 ms | 22.701 ms | -311.3% | not a replacement |
| 8,192 x 1,536 | query L2 | 4.166 ms | 26.725 ms | -541.5% | not a replacement |

## Recommendation — no code change

Retain the NumPy implementations of `embedding.normalize`,
`embedding.compute_similarity`, and embedding `avg`. No candidate is a clear
parity-preserving win. Normalize and all similarity candidates are slower on
both representative dimensions; dot also changes numerical results beyond the
predeclared tight guard. Native average requires O(dimensions) expression
construction (`to_struct` plus one aggregate per component) and is 4.70x slower
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

`git diff --check` is clean. There are no source or test changes in this node,
so no linter result is represented as a code-change gate.

## Frozen review request — FROM HERD COMMAND

The local-only benchmark matrix, parity boundary, and no-op recommendation are
frozen. **Request review FROM HERD COMMAND** for TD-3385 at the current branch
head; scope is this evidence record and the corresponding anchor/daily-digest
status update. **HOLD:** do not start a new node or alter this branch after the
pinned review request unless Herd Command issues a disposition.
