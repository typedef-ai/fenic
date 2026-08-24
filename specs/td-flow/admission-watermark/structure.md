---
workflow_id: admission-watermark
phase: structure
track: engineering
size_class: high-risk
status: approved
portability_level: 3
source_inputs:
  - specs/td-flow/admission-watermark/research.md
  - specs/td-flow/admission-watermark/design.md
last_updated: 2026-08-21
---

<!-- The td-structure template repeats the same File Changes / Validation
     headings under every phase by design. -->
<!-- markdownlint-disable MD024 -->

# Structure: Bounded indexed semantic streaming

This outline implements the approved indexed-slot iterator without widening the
public API or touching child branches. Each phase is an end-to-end behavior slice:
the iterator state, the externally visible ordered series, and deterministic tests
move together.

## Desired End State

- Semantic streaming admits and collects work independently of positional output.
- Row-local operators and semantic join continue to receive results in submission
  order.
- Pending, completed-result, and dedup state have observable bounded high waters.
- Indexed provider failures preserve ordered error delivery without leaking global
  thread errors into later admission.
- The stage record retains comparable ordered-wait evidence and reports completed
  cap saturation separately.
- The provider-free regression harness, peak-RSS harness, and allowed provider
  samples grade the implementation against standard execution.

## Implementation Overview

- [x] Phase 1: Indexed bounded iterator and positional output
- [x] Phase 2: Indexed error and duplicate ownership
- [x] Phase 3: Stage instrumentation and deterministic saturation evidence
- [ ] Phase 4: Benchmark and memory grading

---

## ✅ Phase 1: Indexed bounded iterator and positional output

Replace the FIFO pending deque in the semantic iterator with indexed pending and
completed state. This slice proves ordered output and independent completion
collection without changing the public stream entry point.

### File Changes

- **`src/fenic/_inference/model_client.py`**: replace the FIFO iterator state
  with indexed live slots; introduce private cap derivation from the captured
  look-ahead basis; collect any settled slot into a capacity-limited completed
  buffer; drain only contiguous indices at the emission edge.
- **`tests/_inference/test_model_client_cache_behavior.py`**: add deterministic
  fixtures and assertions for out-of-order settlement, successor admission,
  ordered output, `None` requests, empty input, and pending/completed high-water
  limits.

### Validation

#### Automated Verification

- [x] `uv run pytest tests/_inference/test_model_client_cache_behavior.py`
- [x] `trunk check src/fenic/_inference/model_client.py tests/_inference/test_model_client_cache_behavior.py`

#### Manual Verification

None needed — deterministic future-control tests cover the runtime state machine.

---

## ✅ Phase 2: Indexed error and duplicate ownership

Make the new state machine safe under asynchronous failure and repeated requests.
This phase is independently verifiable by forcing a later failure and duplicate
responses behind a blocked earliest slot.

### File Changes

- **`src/fenic/_inference/model_client.py`**: add private stream ownership for
  queue-item thread errors, defer only owned slot failures to indexed emission,
  clean ownership state on iterator exit, and retain fingerprint entries until
  their final pending/completed slot emits.
- **`tests/_inference/test_model_client_cache_behavior.py`**: add blocked-earlier
  / failed-later / admitted-successor coverage and a cache-disabled
  original-duplicate-slow-third-duplicate case.

### Validation

#### Automated Verification

- [x] `uv run pytest tests/_inference/test_model_client_cache_behavior.py`
- [x] `trunk check src/fenic/_inference/model_client.py tests/_inference/test_model_client_cache_behavior.py`

#### Manual Verification

None needed — the controlled completion clients make failure timing observable.

---

## ✅ Phase 3: Stage instrumentation and deterministic saturation evidence

Port the frozen streaming stage timing to the indexed state machine. Preserve the
historical ordered-wait comparison while exposing the cap-saturation subset that
can still pause admission.

### File Changes

- **`src/fenic/_inference/request_lifecycle.py`**: add the private streaming
  stage event/data shape and aggregation helpers required by the benchmark receipt.
- **`src/fenic/_inference/model_client.py`**: emit admission, dispatch, advance,
  response-drain, and wait-for-next-expected timing; publish that final interval
  through the existing ordered-wait comparison field and a separate completed-cap
  saturation measurement.
- **`tests/_inference/test_model_client_cache_behavior.py`**: drive a saturated
  completed buffer plus done pending slots and assert both timing signals; include
  a deliberately FIFO-re-serialized variant that restores ordered-wait evidence.

### Validation

#### Automated Verification

- [x] `uv run pytest tests/_inference/test_model_client_cache_behavior.py`
- [x] `trunk check src/fenic/_inference/model_client.py src/fenic/_inference/request_lifecycle.py tests/_inference/test_model_client_cache_behavior.py`

#### Manual Verification

None needed — deterministic timestamps and the negative variant exercise the
measurement contract.

---

## Phase 4: Benchmark and memory grading

Run the retained provider-free benchmark and memory harness against the complete
iterator. Use the stage receipt to distinguish any residual positional-emission
cost from avoidable admission serialization. Provider runs remain opt-in and
subject to the stated spend gate.

### File Changes

- **`tools/benchmark_semantic_operator_memory.py`** and
  **`tests/test_benchmark_semantic_operator_memory.py`**: run the existing
  isolated peak-RSS comparison for the streaming workload and preserve
  provider-free operation if the receipt needs a branch-local assertion.
- **Inherited regression harness:** run the provider-free regression scenario
  from the descendant benchmark PR only after its stack manager has restacked it
  on this branch. This branch does not modify `benchmarks/streaming/` or
  `benchmarks/semantic_join_stream_adapter.py`; those files remain owned by the
  descendant deliverable.
- **`specs/td-flow/admission-watermark/structure.md`**: record benchmark tables,
  pre-run provider estimate, actual spend, and the final old/new branch SHA mapping
  at freeze.

### Validation

#### Automated Verification

- [ ] `uv run pytest tests/_inference/test_model_client_cache_behavior.py tests/test_benchmark_semantic_operator_memory.py`
- [ ] Run the descendant provider-free harness after restacking; do not modify
      its branch to make it runnable here.
- [ ] `uv run python tools/benchmark_semantic_operator_memory.py --cases semantic_join --rows 2 --label smoke --json`
- [ ] `trunk check tools/benchmark_semantic_operator_memory.py tests/test_benchmark_semantic_operator_memory.py`

#### Manual Verification

- [ ] Before any provider run, record the cost estimate; stop for direction if it
      projects above the hard cap.

## Open Questions

None.

## Decision Log

- **[applied]** Use four vertical behavior slices: iterator ordering, ownership
  semantics, timing evidence, then grading.
- **[applied]** Keep implementation to iterator internals and its existing tests
  until the grading phase requires benchmark receipt changes.
- **[applied]** The approval to advance permits these slices without another
  planning pause, but does not waive cost, mismatch, or verification stops.
- **[deferred]** Any public admission-control parameter until measured evidence
  justifies it.

## Handoff

**Next step:** Implement the checked phases in order. Record completed automated
verification beside each phase and stop if the design contracts cannot coexist.
