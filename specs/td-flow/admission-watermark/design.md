---
workflow_id: admission-watermark
phase: design
track: engineering
size_class: high-risk
status: approved
portability_level: 3
source_inputs:
  - specs/td-flow/admission-watermark/research.md
  - src/fenic/_inference/model_client.py
  - src/fenic/_backends/local/async_udf_stream.py
  - 202a24b (frozen stage-instrumentation source)
last_updated: 2026-08-21
---

# Design: Decoupled semantic admission and ordered emission

The semantic iterator currently uses one FIFO deque to bound live work, select
the next future to wait on, and determine response emission. The design separates
those responsibilities while preserving the row-local positional result interface.
It follows the repository's existing indexed async-UDF stream rather than adding a
third asynchronous execution pattern.

## Chosen Approach

Use indexed pending and completed slots inside one retained-slot budget in the
semantic model-client iterator.

Each input receives a monotonically increasing submission index before it enters
the provider queue. The iterator tracks live, not-yet-transferred slots in
`pending[index]`; a pending future can be incomplete or already settled when the
completed buffer is full. It tracks transferred out-of-order responses in
`completed[index]`. Admission transfers settled slots from `pending` into
`completed`, then submits more work while the shared budget permits it. Emission drains
only the contiguous sequence beginning at `next_index_to_emit`; this is the
single ordering boundary, immediately before the row-local operator appends to
its output series.

The captured look-ahead basis `L = max(batch_size, rpm)` is the total retained
slot budget. Pending and completed slots share this budget. The iterator admits
new requests only when `len(pending) + len(completed) < L`. This matches the
public window contract for every L, including values above 1,000.

This is the smallest defensible slice: it changes only the semantic iterator's
internal state machine. The streaming opt-in, completion API, positional series
output, queue/rate-limit integration, cache, and error boundary remain intact.

## Alternatives Considered

- **Emit completions in settlement order.** This would remove ordered waiting,
  but it breaks the positional alignment that row-local operators and semantic
  join use. It is rejected because the existing operator boundary has no indexed
  result representation.

- **Use independent pending and completed caps.** This can reduce ordered-head
  stalls, but it lets retained state exceed the documented L window. It is
  rejected because callers cannot rely on the public memory bound.

- **Admit all work and reorder after completion.** This maximizes look-ahead but
  retains an input-sized future/result set. It is rejected because it discards
  the bounded-working-set property required by streaming.

- **Introduce a new public watermark parameter.** Separate caller tuning could
  be useful later, but it changes the API and adds configuration semantics before
  the existing policy has been measured with decoupled state. It is deferred.

## Contracts & Seams

### Positional emission contract

The iterator emits responses in submission-index order. `BaseOperator` continues
to receive a plain ordered iterator and build a positional `pl.Series`; semantic
join therefore continues to attach predicate results to join pairs by position.
No out-of-order response record crosses this boundary.

### Bounded-state contract

At every observable point, `len(pending) + len(completed) <= L`. A settled
future left in `pending` still counts against L. A request key remains in the
live dedup map until its final retained response emits, so the dedup map is also
bounded by L. Peak RSS remains the process-level validation of that bound.

### Completion and backpressure seam

The iterator transfers settled pending slots into `completed` without changing
the shared retained count. It drains the contiguous ordered prefix before
refilling the newly free slots. A blocked early index can therefore hold the
window at L while later completed responses wait for ordered emission.

### Error, cache, and dedup seam

Provider failures remain normalized at the existing iterator boundary. Streaming
submissions are tagged with their live slot index and bypass only the global
thread-error entries owned by that stream; their futures carry provider failures
until the emission edge. A worker error with no owned live-slot tag retains
immediate handling. The stream-owned error entry is cleared when the generator
exits, whether by exhaustion or failure. A slot failure is therefore observed when
its submission index reaches the emission edge, so earlier responses retain their
positional behavior.

Each bounded admission group uses one batch cache read. Cache hits, writes, and
live deduplication retain their existing result semantics. The live dedup map
tracks a retained-slot count per request fingerprint. It removes a key only
after the last pending or completed slot for that fingerprint emits.

### Instrumentation seam

The stage instrumentation covers admission, dispatch, advance, and response
drain. The iterator no longer reports a separate completed-cap-blocked interval
because pending and completed slots share one public budget.

### Test seams

`None` requests receive an index and consume one retained slot. They preserve
their input position and perform no provider dispatch. An empty input terminates
without a wait cycle.

Deterministic completion clients must independently control settlement order and
release of the next expected index. Tests observe pending/completed/dedup high
waters, ordered output, successor admission after an out-of-order completion,
high waters, ordered output, failure normalization at the emission edge, and
cache behavior after a slot emits. They cover L below and above 1,000,
interleaved `None` requests, an empty request iterator, and one batch cache read
for a duplicate-only initial window.
The grading evidence requires both end-to-end wall-time parity against standard
execution and the cap-saturation measurement. A reduced ordered-wait share alone
does not pass if completed-cap backpressure leaves a residual wall-time gap.

## Scope Boundary

The design covers the semantic model-client iterator, its stage instrumentation,
the row-local streaming tests, and the benchmark/memory evidence needed to grade
the change. It does not change `AsyncUDFSyncStream`, aggregation operators, the
public streaming opt-in, provider interfaces, or add a public concurrency option.

## Later

- Evaluate a caller-visible admission watermark only after the shared-budget
  behavior has benchmark evidence.
- Consider sharing a private indexed-stream helper only if the two existing
  implementations converge beyond their current separate execution domains.

## Open Questions

None.

## Decision Log

- **[applied]** Use indexed pending and completed-result buffers, following the
  established async-UDF state shape.
- **[applied]** Restore positional order at the row-local emission edge rather
  than exposing settlement order to operators.
- **[applied]** Bound pending and completed slots together by
  `L = max(batch_size, rpm)`.
- **[applied]** Defer indexed provider failures to the emission edge while
  retaining immediate handling for unowned fatal worker failures and cleanup for
  stream-owned error entries.
- **[applied]** Keep a request fingerprint live until its final retained slot
  emits, so ordered buffering does not break duplicate suppression.
- **[applied]** Batch cache reads for each bounded admission group.
- **[applied]** Hold scope to iterator internals, instrumentation, tests, and
  evidence; do not add a public tuning option or a shared executor abstraction.
- **[rejected]** Completion-order emission, a larger FIFO window, and unbounded
  admit-all reordering.
- **[deferred]** A public admission-watermark setting pending measured evidence.

## Handoff

**Next step:** Build the implementation structure from this design and the
research findings, retaining the iterator-only scope.

**Approved decisions:** indexed live slots; one shared retained-slot budget;
positional order restored at emission; per-slot error and dedup ownership;
batched cache reads.
**Open questions (carried forward):** None.
**Non-goals / out of scope:** a public tuning parameter, a shared executor
abstraction, aggregation operators, provider interfaces, and child branches.
**Evidence summary:** the current iterator couples FIFO waiting and admission;
the local async-UDF stream already demonstrates independent indexed state.
**Known weak assumptions:** the shared budget must satisfy the parity and
peak-RSS checks under the target workload.
**Next artifact:** `specs/td-flow/admission-watermark/structure.md`.
**Rollback if:** the first implementation slice cannot preserve positional output,
bounded state, and indexed error behavior together.
