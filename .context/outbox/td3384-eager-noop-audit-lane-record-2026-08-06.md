# TD-3384 eager no-op physical-operator audit — lane record

**Status:** implementation evidence frozen for Herd Command review.  
**Branch:** `herd/fenic-exec-engine-td3384-eager-noop-audit`, stacked on the
accepted TD-3334 head `54ea84404b9655a587c235470253bae36edfa3f6`.

## Scope and spend

This node used local Polars data only. Estimated provider spend is **$0.00**;
no credential was read. The stopped placeholder-key incident below produced no
successful provider response, but is not an actual billing receipt. The audit
rule was applied literally: prove a recurrent no-op case and measure it before
adding a fast path. Schema, column order, non-identity, and empty-frame
behavior are covered by the new focused test file.

## Audit results

The micro-measurement uses Polars `1.43.1`, a 4,096-row × 512-column `Int64`
DataFrame (16,777,216 estimated bytes), five warmups, and 40 median-timed
rounds. It measures execution only, not fixture construction.

| Candidate | Recurrent no-op proof | Before median | After median | Decision |
| --- | --- | ---: | ---: | --- |
| `apply_ingestion_coercions` unconditional all-column `select` | `InMemorySourceExec`, `FileSourceExec`, DuckDB table/cache reads, and SQL results all call it. Ordinary scalar source frames require no dtype rewrite. | 0.960354 ms | 0.350979 ms | LAND: return the original DataFrame when no target dtype changes. |
| `UnionExec` unconditional right-side alignment | Logical `Union` requires matching names/types but allows differing order; ordinary same-order source unions therefore need no alignment. | 0.869271 ms | 0.267583 ms | LAND: preserve the right DataFrame when `right.columns == left.columns`; retain `select(left.columns)` for differing order. |
| `ProjectionExec` direct-column identity `select` | A public direct-column projection lowers to ordered `pl.col(name)` expressions matching every child column. | 0.431937 ms | 0.137688 ms | LAND: return the child DataFrame only for every-column, same-order direct column expressions. Aliases and computed expressions still select. |
| Source/sink paths outside ingestion | Sources perform file/DuckDB I/O or already flow through the ingestion candidate. Sinks must check/write external state and return their defined empty result. | n/a | n/a | Not needed: no behavior-preserving eager DataFrame fast path. |
| Join normalize/restore | Only an input already named a reserved normalization alias makes rename/restore structurally idle. On the same wide frame, that pair costs 0.034583 ms median. | 0.034583 ms | n/a | Not needed: too small and not a generic source/sink path; retain the explicit alias contract. |

The before/after figures are local micro-measurements, not a general workload
throughput claim. They establish that the three selected operations have a
material eager-all-column cost on representative wide frames.

## Behavior contract

`tests/_backends/local/physical_plan/test_eager_noop_fast_paths.py` proves:

1. scalar and empty ingestion frames retain exact identity/schema/data, while
   `Array` normalization still creates the required `List` result;
2. same-order union keeps the right DataFrame, while different-order union still
   restores left-side order and values, including an empty-schema-compatible
   case; and
3. identity projections retain exact identity for populated and empty frames,
   while an aliased/computed column still evaluates through Polars.

The focused physical/transpiler/ingestion gate passed **28 tests**. `uvx ruff`
and `git diff --check` passed.

## Process incident — broad test selection stopped

One attempted broader command included
`tests/_backends/local/test_metrics.py::test_semantic_metrics`, which is not a
fake-injected test. It tried the known synthetic placeholder key and received
authentication `401` failures from OpenAI before any successful response. No
`.env` value was read and the command was stopped; no further provider-capable
tests will run for this local-only node. This is a test-selection mistake, not
a product failure or a cost receipt.

## Frozen review request — FROM HERD COMMAND

The audit evidence and the three behavior-preserving fast paths are frozen.
**Request review FROM HERD COMMAND** for TD-3384 at the current branch head.
Review scope: no-op guards, schema/order/empty/non-identity preservation,
micro-measurement accounting, and the explicit no-op dispositions for
source/sink/restore paths. **HOLD:** do not start TD-3385 until disposition.
