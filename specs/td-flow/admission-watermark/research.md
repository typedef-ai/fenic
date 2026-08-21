---
workflow_id: admission-watermark
phase: research
research_stage: findings_ready
track: engineering
size_class: high-risk
status: approved
portability_level: 3
recommended_track: engineering
source_inputs:
  - originating task/ticket (not reproduced here)
  - src/fenic/_inference/model_client.py
  - src/fenic/_inference/language_model.py
  - src/fenic/_backends/local/semantic_operators/
  - src/fenic/_backends/local/async_udf_stream.py
  - src/fenic/_inference/request_lifecycle.py
  - tests/_inference/test_model_client_cache_behavior.py
  - tools/benchmark_semantic_operator_memory.py
  - 202a24b (derived scope: frozen instrumentation source)
  - e274de4 (derived scope: provider-free benchmark source)
last_updated: 2026-08-21
---

# Research: Semantic request admission

**Date:** 2026-08-21 · **Status:** approved · **Stage:** findings_ready ·
**Branch/commit:** `brandon/exec-engine-streaming-foundation-pr2@0660d6f40b640b83895210b12461739e4127254e`

## Research Questions

A query plan for the Research findings below. You do not need to answer these
questions. They guide the research toward how the system works today.

1. In `src/fenic/_backends/local/async_udf_stream.py`, how are asynchronous
   requests admitted, tracked, and released as results become available?
2. In the local semantic operators, how do operator inputs become request batches,
   and where do those batches enter the asynchronous execution path?
3. How does the current execution path preserve result order, and which data
   structures retain pending inputs, requests, and completed results?
4. Which test suites exercise streaming and non-streaming semantic execution,
   including failure, ordering, and rate-limit behavior?
5. What existing instrumentation records execution-stage time, and where can it
   distinguish admission, dispatch, ordered-result waiting, and draining?
6. How do the repository's memory tests measure the retained working set during
   semantic operator execution?
7. How do the benchmark tools construct and compare representative semantic join
   workloads without relying on an external provider?
8. Which downstream consumers of the semantic execution path require results in
   submission order, where is ordered emission a contract rather than an
   implementation artifact, and where could order be restored later instead?

## Decision Log

- **[approved]** The questions cover the local asynchronous executor, semantic
  operators, tests, instrumentation, memory checks, benchmark tooling, and the
  downstream ordering contract.
- **[observed]** The semantic completion stream currently emits an ordered
  response sequence without a request index at the operator boundary.
- **[observed]** The frozen stage-timing instrumentation is not present on this
  branch; its source is an earlier commit that must be treated as a porting
  input rather than current behavior.
- **[applied]** A document review distinguished the observed internal positional
  dependency from a separately documented public ordering guarantee.

## Inputs

**Used:** this document's approved questions; the checked-out semantic
execution, local-operator, lifecycle, test, and memory-harness sources; the
frozen stage-instrumentation source at `202a24b`; and the provider-free
benchmark source at `e274de4`.

**Deliberately excluded:** the originating task body. The findings below answer
the approved questions from repository sources instead of restating task
requirements.

## Findings

### 1. Two bounded asynchronous paths exist, with different ordering state

`AsyncUDFSyncStream` is the local-transpiler path, not the semantic completion
iterator: the transpiler imports and constructs it, while semantic completions
enter `ModelClient.iter_batch_requests` through `LanguageModel.iter_completions`
(`src/fenic/_backends/local/transpiler/expr_converter.py:21,479`; `src/fenic/_inference/language_model.py:114-132`).

The async-UDF stream keeps pending tasks and completed results in dictionaries
keyed by input index. It yields only its next expected index, limits pending
tasks and completed results separately, and switches from waiting for any task
to waiting for that expected task when the results buffer is full
(`src/fenic/_backends/local/async_udf_stream.py:14-45,64-70,95-139`).

The semantic completion iterator uses a different bounded state shape:
`unique_futures` maps request keys to futures and `pending` is a FIFO deque of
future/key pairs. Its watermark is captured as
`max(batch_size, rate_limit_strategy.rpm)` at iterator construction
(`src/fenic/_inference/model_client.py:541-559,564-569`).

### 2. Row-local semantic operators enter the model-client iterator lazily

The default request sender chunks an iterable only for non-language-model
senders. `CompletionOnlyRequestSender` instead forwards the iterable to
`LanguageModel.iter_completions`, which lazily builds a completion request for
each message and calls the model client iterator
(`src/fenic/_backends/local/semantic_operators/base.py:37-53,108-126`; `src/fenic/_inference/language_model.py:96-132`).

Row-local operators are opt-in: `BaseOperator.stream_requests` defaults to
`False`. When enabled, it turns each returned response into a one-item
postprocess result and appends it to one final `pl.Series`; inputs are rendered
lazily in input order (`src/fenic/_backends/local/semantic_operators/base.py:128-194`).
The opt-in default and the Map iterator path are covered directly in the model
client tests (`tests/_inference/test_model_client_cache_behavior.py:800-831`).

### 3. The semantic iterator couples admission, ordered consumption, and release

After filling `pending` to the captured watermark, the iterator removes the
leftmost future and calls `result()` on it. Only after that wait completes does
it remove the corresponding dedup entry, refill the deque to the watermark, and
yield that response (`src/fenic/_inference/model_client.py:593-612`). Thus the
same FIFO queue both bounds the live request/future/dedup working set and
selects the next future waited on and emitted.

The current tests make both properties observable: they expect completion order
to match input order, verify a successor is admitted before a slow peer settles,
and constrain active requests and the live dedup map to the admission watermark
(`tests/_inference/test_model_client_cache_behavior.py:425-485,548-585,643-672`).
They also preserve the public error wrapping boundary and cache behavior after a
request leaves the live window (`tests/_inference/test_model_client_cache_behavior.py:487-513,721-770,773-797`).

### 4. Current lifecycle events observe provider transitions, not iterator stages

The lifecycle collector has event types for queued, rate-limited, dispatched,
settled, retried, and failed requests. Its idle-gap calculation reconstructs
queue delay and rate-limited portions from those transitions
(`src/fenic/_inference/request_lifecycle.py:14-42,68-156`). The iterator tests
assert indexed queued and settled lifecycle events for one semantic operation
(`tests/_inference/test_model_client_cache_behavior.py:516-545`).

The checked-out branch has no named streaming-stage events. The frozen
instrumentation source adds optional collector-gated timing records around
window admission, request dispatch, window advance, ordered slot wait, and
response drain; it surrounds the FIFO `Future.result()` call with the slot-wait
measurement (`202a24b:src/fenic/_inference/model_client.py:608-724`). This is
an external frozen source, not a claim about code currently on this branch.

### 5. Ordered output is a current downstream positional dependency

`BaseOperator.execute` appends stream responses in iterator order and returns a
plain `pl.Series`; the stream interface carries no request index or separate
reordering hook (`src/fenic/_backends/local/semantic_operators/base.py:158-178`).
The direct row-local consumers therefore use positional alignment: Map is
tested against input-order results, Extract tests two output positions, and
Predicate tests a 101-row boolean series
(`tests/_inference/test_model_client_cache_behavior.py:807-831`; `tests/_backends/local/functions/test_semantic_extract.py:326-355`; `tests/_backends/local/functions/test_semantic_predicate.py:382-440`).

Semantic join is the clearest downstream dependency. It executes a `Predicate`
over the cross-join's rendered-input series, then attaches the returned series
as a column to that same pair DataFrame before filtering. That operation requires
the result positions to align with the submitted pair positions
(`src/fenic/_backends/local/semantic_operators/join.py:45-58,87-102`).

`semantic.reduce` remains on the list-shaped completion API because it is an
aggregation operator, so it is outside this row-local streaming dependency
(`src/fenic/_inference/language_model.py:96-104`). Current code contains no
later component that accepts out-of-order completion records plus indexes and
restores their input order; restoration would require a boundary not present in
the current row-local interface (`src/fenic/_backends/local/semantic_operators/base.py:158-178`).

### 6. Peak RSS is the existing retained-working-set measurement

The memory harness launches each case in an isolated child process and reports
`resource.getrusage(RUSAGE_SELF).ru_maxrss` as its authoritative peak-memory
signal; it reports Polars allocation as unavailable rather than substituting a
different allocator metric (`tools/benchmark_semantic_operator_memory.py:1-27,71-95`).
The test harness invokes the semantic-join case with network calls disabled and
checks the machine-readable RSS payload (`tests/test_benchmark_semantic_operator_memory.py:8-40`).

### 7. The separate regression benchmark is provider-free but depends on this branch

The regression benchmark executes only a simulated semantic-join scenario. It
constructs a deterministic workload, counts calls to the list and iterator
model-client paths, measures the `join.execute()` interval, and records result,
RSS, and admission-high-water evidence (`e274de4:benchmarks/streaming/run_case.py:32-146`).
Its adapter import is `benchmarks.semantic_join_stream_adapter`, which is absent
from `origin/main`; this is why the benchmark cannot currently be detached onto
main without the streaming branch (`e274de4:benchmarks/streaming/run_case.py:39-46`).

## Open Questions

None.

## Handoff

**Next step (paste into a fresh tab):**

> Use the `td-design` skill. Research is approved at
> `specs/td-flow/admission-watermark/research.md` (track: engineering, size:
> high-risk). Design-stage triggers held: multiple viable approaches and a
> cross-cutting ordering-and-memory contract. Choose the approach, weigh
> alternatives, and fix the cross-cutting contracts before the structure. Use
> this research doc's Findings — the `## Research Questions` section is
> provenance, not a spec.
> If this is Codex: I explicitly permit optional subagent use for this phase
> where the skill allows it.

**Approved decisions:** preserve positional result alignment at the row-local
operator boundary; restore order at that emission edge; model the semantic
iterator after the existing indexed async-UDF shape unless a divergence is
explicitly justified; retain a bounded working set with separate admission and
completed-result caps; map the existing ordered-wait timing signal to the
new wait-for-next-expected measurement.
**Open questions (carried forward):** None.
**Non-goals / out of scope:** changing aggregation operators or removing the
row-local streaming opt-in.
**Evidence summary:** the current model-client iterator couples FIFO admission,
waiting, and emission (`src/fenic/_inference/model_client.py:593-612`); row-local
operators produce positional series (`src/fenic/_backends/local/semantic_operators/base.py:158-178`);
semantic join consumes that series positionally
(`src/fenic/_backends/local/semantic_operators/join.py:87-102`); the async-UDF
stream already separates pending and completed indexed state
(`src/fenic/_backends/local/async_udf_stream.py:64-139`).
**Known weak assumptions:** the frozen stage instrumentation is not on this
branch and must be ported without changing what its ordered-wait measurement
means.
**Next artifact:** `specs/td-flow/admission-watermark/design.md`.
**Rollback if:** the design cannot keep the bounded working set while preserving
positional emission — ROLLBACK to the Research stage.
