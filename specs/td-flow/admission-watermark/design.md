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

Use indexed slots with independent pending and completed-result caps inside the
semantic model-client iterator.

Each input receives a monotonically increasing submission index before it enters
the provider queue. The iterator tracks live, not-yet-transferred slots in
`pending[index]`; a pending future can be incomplete or already settled when the
completed buffer is full. It tracks transferred out-of-order responses in
`completed[index]`. Admission transfers settled slots from `pending` into
`completed`, then submits more work while both caps permit it. Emission drains
only the contiguous sequence beginning at `next_index_to_emit`; this is the
single ordering boundary, immediately before the row-local operator appends to
its output series.

For v0, the captured look-ahead basis `L = max(batch_size, rpm)` remains an
existing compatibility input, but it is not an admission limit. The iterator uses
the async-UDF ratios and hard limits: `pending_admission_cap = min(1_000, 3L)`
and `completed_result_cap = min(50_000, 10L)`. The pending admission cap bounds
new submissions; the completed-result cap independently bounds reordered
responses. This preserves the existing public argument and rate-limit policy
while removing semantic batch size as the equal-sized ordered-slot gate. The
retained execution state is bounded by the two caps rather than one FIFO window.

This is the smallest defensible slice: it changes only the semantic iterator's
internal state machine. The streaming opt-in, completion API, positional series
output, queue/rate-limit integration, cache, and error boundary remain intact.

## Alternatives Considered

- **Emit completions in settlement order.** This would remove ordered waiting,
  but it breaks the positional alignment that row-local operators and semantic
  join use. It is rejected because the existing operator boundary has no indexed
  result representation.

- **Increase the fixed FIFO window.** A larger window can reduce stalls in a
  particular workload, but it keeps admission coupled to the next ordered slot
  and increases retained state without a separate completed-result cap. It is
  rejected because it does not remove the serialization mechanism.

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

At every observable point, `len(pending) <= pending_admission_cap` and
`len(completed) <= completed_result_cap`. A settled future left in `pending` because the
completed buffer is full still counts against `pending_admission_cap`; it is not hidden
state. A request key remains in the live dedup map until its response is emitted,
so the dedup map is bounded by the retained slot set as well. The v0 bound is at
most `min(1_000, 3L) + min(50_000, 10L)` retained slots plus fixed iterator
overhead; peak RSS remains the process-level validation of that bound.

### Completion and backpressure seam

When the completed buffer has room, the iterator waits for any pending future and
transfers no more than the remaining completed-buffer capacity into `completed`.
Additional done futures remain indexed in `pending` and count against its cap.
It then refills admission without requiring the earliest unresolved index to
settle. When the completed buffer is full and the next expected index is absent,
it waits specifically for that index; if that future is still in `pending`, its
response is consumed directly into the ordered drain so the completed cap never
temporarily overflows. The iterator drains the contiguous ordered prefix before
further admission. This is the intentional convergence with
`AsyncUDFSyncStream`'s independent pending/result buffers and next-index emission
rule, with capacity-limited transfer required by the semantic iterator's strict
buffer accounting.

### Error, cache, and dedup seam

Provider failures remain normalized at the existing iterator boundary. Streaming
submissions are tagged with their live slot index and bypass only the global
thread-error entries owned by that stream; their futures carry provider failures
until the emission edge. A worker error with no owned live-slot tag retains
immediate handling. The stream-owned error entry is cleared when the generator
exits, whether by exhaustion or failure. A slot failure is therefore observed when
its submission index reaches the emission edge, so earlier responses retain their
positional behavior.

Cache lookup/write behavior remains unchanged. The live dedup map additionally
tracks a retained-slot count per request fingerprint. It removes a key only after
the last pending or completed slot for that fingerprint has emitted; an original,
a later duplicate, and an intervening slow slot therefore still share one live
provider future when caching is disabled.

### Instrumentation seam

The frozen stage instrumentation is ported with the same event coverage for
admission, dispatch, advance, response drain, and ordered waiting. The historical
`slot_wait` comparison field is retained, but it measures the new
wait-for-next-expected interval: time spent blocked because the emission edge
cannot advance. Completion collection that is not waiting for the expected index
is not attributed to that field. A separate completed-cap-blocked counter and
duration identify the subset where a full completed buffer prevents further
admission. This preserves before/after comparability while making cap saturation
observable rather than merely moving serialization to a larger buffer.

### Test seams

`None` requests receive an index and enter `completed` as an already-settled
response; they consume completed-result capacity, preserve their input position,
and perform no provider dispatch. An empty input terminates without a wait cycle.

Deterministic completion clients must independently control settlement order and
release of the next expected index. Tests observe pending/completed/dedup high
waters, ordered output, successor admission after an out-of-order completion,
failure normalization at the emission edge, and cache behavior after a slot is
emitted. They also cover simultaneous settlement when only one completed-buffer
slot remains, interleaved `None` requests, and an empty request iterator. A
blocked earliest index must also drive completed-cap saturation and done pending
slots, proving that admission reaches the two-cap bound before it pauses. That
case records admission progress, ordered-wait time, and completed-cap-blocked
time. A later-index failure behind a blocked earliest index must allow successor
admission and then raise only at its emission edge. An original/duplicate/slow
intervening/third-duplicate case must make one provider request with caching
disabled. A deliberately re-serialized variant must report the ordered-wait stage
so the instrumentation can demonstrate sensitivity to the mechanism it grades.

The grading evidence requires both end-to-end wall-time parity against standard
execution and the cap-saturation measurement. A reduced ordered-wait share alone
does not pass if completed-cap backpressure leaves a residual wall-time gap.

## Scope Boundary

The design covers the semantic model-client iterator, its stage instrumentation,
the row-local streaming tests, and the benchmark/memory evidence needed to grade
the change. It does not change `AsyncUDFSyncStream`, aggregation operators, the
public streaming opt-in, provider interfaces, or add a public concurrency option.

## Later

- Evaluate a caller-visible admission watermark only after the internal two-cap
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
- **[applied]** Use the async-UDF pending and completed-buffer ratios and hard
  limits, so admission is not capped by semantic batch size while retained state
  remains bounded.
- **[applied]** Limit each collection transfer to completed-buffer capacity and
  count surplus done futures in pending state.
- **[applied]** Defer indexed provider failures to the emission edge while
  retaining immediate handling for unowned fatal worker failures and cleanup for
  stream-owned error entries.
- **[applied]** Keep a request fingerprint live until its final retained slot
  emits, so ordered buffering does not break duplicate suppression.
- **[applied]** Grade completed-cap saturation separately from ordered-wait time
  and require end-to-end parity, not a stage-share improvement alone.
- **[applied]** Preserve the historical ordered-wait metric name while mapping it
  to wait-for-next-expected behavior.
- **[applied]** Hold scope to iterator internals, instrumentation, tests, and
  evidence; do not add a public tuning option or a shared executor abstraction.
- **[rejected]** Completion-order emission, a larger FIFO window, and unbounded
  admit-all reordering.
- **[deferred]** A public admission-watermark setting pending measured evidence.

## Handoff

**Next step:** Build the implementation structure from this design and the
research findings, retaining the iterator-only scope.

**Approved decisions:** indexed live slots; independent pending-admission and
completed-result caps; positional order restored at emission; per-slot error and
dedup ownership; comparable ordered-wait instrumentation plus cap-saturation
evidence.
**Open questions (carried forward):** None.
**Non-goals / out of scope:** a public tuning parameter, a shared executor
abstraction, aggregation operators, provider interfaces, and child branches.
**Evidence summary:** the current iterator couples FIFO waiting and admission;
the local async-UDF stream already demonstrates independent indexed state.
**Known weak assumptions:** the selected cap ratios must satisfy the parity and
peak-RSS checks under the target workload rather than merely reduce ordered-wait
share.
**Next artifact:** `specs/td-flow/admission-watermark/structure.md`.
**Rollback if:** the first implementation slice cannot preserve positional output,
bounded state, and indexed error behavior together.
