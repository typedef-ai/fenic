# Adaptive Output-Token Estimation & Settlement Design Specification

**Version:** 1.1
**Status:** Implemented (see PR for commit range)
**Last Updated:** 2026-06-09

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [Implementation](#implementation)
6. [Integration](#integration)
7. [Observability](#observability)
8. [Scope & Non-Goals](#scope--non-goals)
9. [Risks & Mitigations](#risks--mitigations)
10. [Pre-existing Issues (Noted, Not Fixed)](#pre-existing-issues-noted-not-fixed)
11. [Testing](#testing)

---

## Overview

### Purpose

Fenic's client-side rate limiter (`RPM` + `TPM` token buckets) reserves **output**
tokens at a worst-case ceiling and never reconciles that reservation against what
a request actually used. The result is chronic over-reservation of the TPM bucket,
so throughput sits far below the provider's true limit.

Concretely, for every request the limiter reserves:

```text
output_tokens = max_completion_tokens + reasoning_budget
```

where `reasoning_budget` is a static per-effort constant — OpenAI `2048/4096/8192/16384`
(`openai_profile_manager.py:56-68`), Anthropic the full `thinking_token_budget`
(`anthropic_batch_chat_completions_client.py:379`), Gemini `1.5 × thinking_budget`
(`gemini_native_chat_completions_client.py:388-395`), OpenRouter `0.15–0.60 × model max output`
(`openrouter_batch_chat_completions_client.py:304-315`). The same number is applied to
**every row in a batch** and is also used as the API's `max_tokens` hard cap, so the two
concerns are entangled. Actual usage (`ResponseUsage`: prompt/completion/thinking/cached)
is captured on every response (`openai_chat_completions_core.py:166-186`) but flows only to
`LMMetrics` and the cache — **never back to the estimator or the bucket**
(`model_client.py:735-783`).

This design closes that loop with two complementary mechanisms and one structural change.

### Key Design Decisions

1. **Settlement is the deterministic backbone.** When a response returns, refund
   `reserved − actual` to the TPM bucket. This recovers over-reservation slack from
   completion #1, independent of any learned estimate. It is the part that reliably
   reclaims throughput.

2. **Adaptive estimation is the opportunistic accelerator.** Maintain a rolling
   distribution of _actual_ output tokens per `(profile, max_completion_tokens)` key and
   reserve a high quantile (p95, or p99 for reasoning profiles) × a safety margin, instead
   of the static ceiling. This shrinks the up-front reservation, which matters most while
   requests are in flight before their actuals land.

3. **Decouple the reservation from the API cap.** The provider still receives a generous
   `max_tokens` (`_get_max_output_token_request_limit` is **untouched**), so nothing
   truncates. Only the _rate-limiter reservation_ drops. This is the structural unlock that
   makes (2) safe.

4. **Default-on with a single conservative dial.** Ship enabled by default with a
   `safety_margin` knob (default `1.15`). The reservation is always clamped `≤` the old
   static ceiling, so we only ever reserve _less_, never more. Disabling turns off adaptive
   estimation (reservations use the static ceiling) but settlement remains always-on.

5. **Smallest seam.** No scheduler changes. Configuration lives at the shared
   `SemanticConfig` level (mirroring `llm_response_cache`) and threads through the model
   registry — no per-model-config duplication. Scoped to completions only.

### Success Metrics

- **Higher TPM utilization** on large batches (reserved-vs-actual output ratio approaches
  the safety margin instead of the old ceiling), surfaced via new `LMMetrics` fields.
- **No regression** for small batches or when disabled (static-ceiling reservation, settlement still runs).
- **Bounded 429 risk:** background 429 rate stays low; occasional misses are absorbed by
  the existing retry queue + backoff.

---

## Quick Start

Adaptive estimation is **on by default** — no configuration required:

```python
from fenic.api.session.config import (
    OpenAILanguageModel,
    SemanticConfig,
    SessionConfig,
)

config = SessionConfig(
    app_name="my_batch_job",
    semantic=SemanticConfig(
        language_models={
            "default": OpenAILanguageModel(model_name="gpt-4o-mini", rpm=500, tpm=200_000)
        },
        default_language_model="default",
        # adaptive_token_estimation defaults to enabled, safety_margin=1.15
    ),
)
```

Tune the safety margin (higher = more conservative reservations, lower 429 risk, lower
throughput):

```python
from fenic.api.session.config import AdaptiveTokenEstimationConfig

semantic = SemanticConfig(
    language_models={...},
    default_language_model="default",
    adaptive_token_estimation=AdaptiveTokenEstimationConfig(safety_margin=1.30),
)
```

Disable adaptive estimation (reservations use the static ceiling; settlement is still always-on):

```python
adaptive_token_estimation=AdaptiveTokenEstimationConfig(enabled=False)
```

---

## Architecture

### Two mechanisms, one structural change

```text
                          ┌─────────────────────────── ModelClient (event loop) ──────────────────────────┐
Producer thread           │                                                                                │
─────────────────         │  _process_queue (dispatch)            _handle_response (settle + observe)       │
estimate_tokens_for_req   │  ┌──────────────────────┐             ┌──────────────────────────────────────┐ │
  output = estimator      │  │ check_and_consume    │  success →  │ usage = response.usage                │ │
    .reserve(key) ────────┼─▶│   (debit reserved)   │ ──────────▶ │ estimator.observe(key, actual_out)    │ │
  (reads rolling stats)   │  └──────────────────────┘             │ strategy.settle(reserved, actual)     │ │
        ▲                 │                                       │ metrics.reserved/actual += ...        │ │
        └─────────────────┼───────────────────────────────────────┴───────────────────────────────────────┘
          rolling stats updated under lock (written here, read on producer thread)
```

- **Decouple:** `estimate_tokens_for_request`'s output term reads the estimator; the API
  cap path (`_get_max_output_token_request_limit`) is unchanged.
- **Estimate:** `OutputTokenEstimator` returns a clamped high-quantile reservation, or the
  static ceiling during cold-start / when disabled.
- **Settle:** on success, the bucket is corrected by `reserved − actual`.

### Why within-batch learning works (and its honest limit)

`request_queue` has `maxsize=100` (`model_client.py:150`), and `_submit_batch_requests`
blocks on `enqueue_future.result()` → `request_queue.put(...)` (`model_client.py:622-626`)
whenever the queue is full. So the producer can only run a bounded distance ahead of the
**consumer**. For batches `≫ 100`, later rows are estimated after earlier rows have been
dispatched and (for fast-enough models) completed, so their estimates reflect observed
usage.

**Honest limit (peer-reviewed):** the queue slot frees at _dispatch_, not at _completion_.
At batch start the producer can race ~100+ rows ahead before any response returns, so the
first cohort always uses the cold-start fallback. This is acceptable: **settlement still
corrects the bucket from completion #1**, so the core over-reservation problem is fixed even
when the estimator is cold. Adaptive estimation is a bonus on top, not a prerequisite.

### Estimator key

`key = (profile_hash, max_completion_tokens)`.

`max_completion_tokens` is already on `FenicCompletionsRequest` (`types.py:67`), so it is a
**free** proxy for operator shape — `semantic.map` (`max_completion_tokens=512`) and
`semantic.extract` (`1024`) under the same profile no longer pool into one distribution, and
we never thread `operation_name` (which is not on the request) into the estimator. `profile_hash`
comes from the existing `ProfileHashMixin`. Model identity is implicit (one estimator per
client).

> Pooling safety: even if two shapes did share a key, the p95 of the union ≥ the p95 of
> either subset, so pooling only ever _over_-reserves the smaller shape — never less safe.

---

## Configuration

### `AdaptiveTokenEstimationConfig`

New Pydantic config in `src/fenic/api/session/config.py`, exported from
`fenic.api.session` and `fenic.api` alongside `LLMResponseCacheConfig`.

```python
class AdaptiveTokenEstimationConfig(BaseModel):
    """Tunes adaptive output-token reservation for rate limiting.

    Output-token reservations are learned from observed usage and clamped to the
    request's max_completion_tokens ceiling, then corrected after each response
    (settlement). Enabled by default. Setting enabled=False disables adaptive
    estimation (reservations use the static ceiling); settlement is always-on
    regardless and cannot increase 429 risk.
    """
    enabled: bool = True
    safety_margin: float = Field(
        default=1.15, ge=1.0, le=4.0,
        description=(
            "Multiplier on the modeled output-token reservation. Higher reserves "
            "more (safer, lower throughput); lower reserves less (higher throughput, "
            "higher 429 risk)."
        ),
    )
```

Only two public knobs. Internal constants — sample window (`256`), warm-up
`min_samples` (`30`), quantiles (`0.95` / `0.99` for reasoning) — live in the estimator and
are not exposed, to keep the surface easy to use.

### `SemanticConfig` integration

Add one field next to `llm_response_cache` (`config.py:1122`):

```python
adaptive_token_estimation: Optional[AdaptiveTokenEstimationConfig] = None
```

`None` means **enabled with defaults** (note: opposite of `llm_response_cache`, where `None`
disables — adaptive estimation is default-on).

### Resolution

Mirror the cache resolution (`config.py:1694-1711`). Add
`ResolvedAdaptiveTokenEstimationConfig` to `_resolved_session_config.py` (next to
`ResolvedCacheConfig`) and set it on `ResolvedSemanticConfig` (`:157`), always populated
(default-constructed when the user config is `None`):

```python
ate_cfg = self.semantic.adaptive_token_estimation if self.semantic else None
resolved_ate = ResolvedAdaptiveTokenEstimationConfig(
    enabled=ate_cfg.enabled if ate_cfg else True,
    safety_margin=ate_cfg.safety_margin if ate_cfg else 1.15,
)
```

---

## Implementation

### `OutputTokenEstimator`

New module `src/fenic/_inference/output_token_estimator.py`. One instance per
`ModelClient`. Thread-safe: **written** on the asyncio event loop (`_handle_response`),
**read** on the producer thread (`estimate_tokens_for_request`).

```python
class OutputTokenEstimator:
    def __init__(self, *, enabled: bool, safety_margin: float,
                 min_samples: int = 30, window: int = 256):
        self._enabled = enabled
        self._safety_margin = safety_margin
        self._min_samples = min_samples
        self._window = window
        self._samples: dict[Hashable, deque[int]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._lock = threading.Lock()

    def reserve(self, key, *, static_ceiling: int, reasoning: bool) -> int:
        """Output-token reservation. Always <= static_ceiling (== the API cap)."""
        if not self._enabled:
            return static_ceiling
        with self._lock:
            samples = self._samples.get(key)
            if samples is None or len(samples) < self._min_samples:
                return static_ceiling            # cold-start fallback
            ordered = sorted(samples)
        q = 0.99 if reasoning else 0.95
        modeled = _quantile(ordered, q) * self._safety_margin
        return max(1, min(static_ceiling, math.ceil(modeled)))

    def observe(self, key, actual_output_tokens: int) -> None:
        with self._lock:
            self._samples[key].append(actual_output_tokens)
```

- `static_ceiling` is each provider's existing computed value
  (`max_completion_tokens + reasoning_budget`), which equals the API cap — so the clamp
  guarantees we never reserve more than today.
- `reasoning = static_reasoning_budget > 0` (provider-agnostic; reuses values already
  computed in the per-provider output estimate).

### `RateLimitStrategy.settle`

Add to the base class as a concrete **no-op** (so `AdaptiveBackoffRateLimitStrategy`
inherits it — OpenRouter does no token accounting), overridden by the token strategies.
Runs **event-loop-only**, same thread as `check_and_consume_rate_limit`; no extra lock.

```python
def settle(self, reserved: TokenEstimate, actual: TokenEstimate) -> None:
    """Correct bucket(s) by (reserved - actual). Event-loop thread only."""
    # base: no-op
```

`UnifiedTokenRateLimitStrategy`:

```python
def settle(self, reserved, actual):
    now = time.time()
    delta = reserved.total_tokens - actual.total_tokens   # >0 = over-reserved → refund
    cap = self.unified_tokens_bucket._get_available_capacity(now)
    self.unified_tokens_bucket._set_capacity(
        min(self.tpm, max(0, cap + delta)), now
    )
```

`SeparatedTokenRateLimitStrategy` (Anthropic) — input and output deltas independently:

```python
def settle(self, reserved, actual):
    now = time.time()
    in_delta = reserved.input_tokens - actual.input_tokens
    out_delta = reserved.output_tokens - actual.output_tokens
    in_cap = self.input_tokens_bucket._get_available_capacity(now)
    out_cap = self.output_tokens_bucket._get_available_capacity(now)
    self.input_tokens_bucket._set_capacity(min(self.input_tpm, max(0, in_cap + in_delta)), now)
    self.output_tokens_bucket._set_capacity(min(self.output_tpm, max(0, out_cap + out_delta)), now)
```

`actual` is built from `ResponseUsage`:
`TokenEstimate(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens + usage.thinking_tokens)`.
Reconciling input against `usage.prompt_tokens` is a free bonus — it corrects the static
`1.05×` Anthropic fudge and credits prompt-cache hits.

---

## Integration

### Per-provider output estimate (the decoupling)

Each completion client's output-estimate method already computes the static ceiling. Route
it through the shared estimator — a one-line wrap per client. OpenAI example
(`openai_batch_chat_completions_client.py:135-146`):

```python
def _estimate_output_tokens(self, request) -> int:
    static_ceiling = (request.max_completion_tokens or 0) + reasoning_tokens   # unchanged
    if request.max_completion_tokens is None and request.messages.user_file:
        static_ceiling += self.token_counter.count_file_output_tokens(messages=request.messages)
    return self._output_estimator.reserve(
        self._estimator_key(request),
        static_ceiling=static_ceiling,
        reasoning=reasoning_tokens > 0,
    )
```

`_get_max_output_token_request_limit` / `get_max_output_token_request_limit` are **not
touched** — the API still receives the generous cap.

`_estimator_key(request)` → `(self.get_profile_hash_for_request(request), request.max_completion_tokens)`.

### `_handle_response` hook (settle + observe + metrics)

In the success branch of `_handle_response` (`model_client.py:764-783`), after the existing
cache write, only for completion responses that carried a reservation (dedup hits reserved
nothing and are skipped):

```python
if (
    isinstance(maybe_response, FenicCompletionsResponse)
    and maybe_response.usage is not None
    and queue_item.estimated_tokens.total_tokens > 0
):
    usage = maybe_response.usage
    actual = TokenEstimate(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens + usage.thinking_tokens,
    )
    self._output_estimator.observe(self._estimator_key(queue_item.request), actual.output_tokens)
    self.rate_limit_strategy.settle(queue_item.estimated_tokens, actual)
    self._record_reservation_metrics(queue_item.estimated_tokens, actual)
```

The `isinstance` check scopes everything to completions; embeddings never reach it.

### Registry wiring

`SessionModelRegistry` already receives the `ResolvedSemanticConfig` and threads `cache=`
into each client (`model_registry.py:65,318,341,363,372`). Pass the resolved adaptive config
the same way; each `ModelClient` constructs its own `OutputTokenEstimator` from `enabled` +
`safety_margin`. Embedding clients ignore it.

---

## Observability

Adaptive systems must be observable to be trusted and tuned. Add to `LMMetrics`
(`metrics.py:13-46`):

```python
num_reserved_output_tokens: int = 0   # output tokens debited from the TPM bucket
# (num_output_tokens already tracks ACTUAL output)
```

Update `__add__` and the human-readable summary strings (`QueryMetrics.__str__` and
`get_execution_plan_details`). `to_dict()` is intentionally left unchanged to avoid a
metrics-table schema migration; the value stays accessible via the dataclass field and the
summaries. Reservation efficiency
= `num_output_tokens / num_reserved_output_tokens` (→ 1 means tight; the old static behavior
produces a small ratio). `_record_reservation_metrics` increments
`num_reserved_output_tokens` by `reserved.output_tokens` so users can watch the gap shrink
and validate settlement.

> Implementation note: each provider owns its live `LMMetrics` (e.g.
> `OpenAIChatCompletionsCore._metrics`), and `ModelClient.get_metrics()` returns that live
> instance. `_record_reservation_metrics` increments the counter on `self.get_metrics()`, so
> the reserved count aggregates alongside the existing actual-usage counters with no separate
> accounting path.

---

## Scope & Non-Goals

**In scope (v1):** adaptive output estimation + settlement + reservation/cap decoupling for
the four completion providers; shared config; observability counter.

**Explicitly out of scope:**

- Provider rate-limit **header** auto-discovery (the "Plan B" reframe) — layers on later as
  another input to the same limiter, no rework required.
- 429-specific refunds (refunding a failed attempt when the error class implies no tokens
  were spent). v1 keeps the conservative "assume spent on any failure."
- Per-row input-aware output scaling (output ≈ f(input length)).
- Cross-run persistence of learned distributions.
- Embeddings (no output tokens).
- Any `_process_queue` / scheduler change.

---

## Risks & Mitigations

| Risk                                                                                                                                                                     | Mitigation                                                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reserving less ⇒ more concurrent dispatch ⇒ genuinely higher 429 risk. Settlement does **not** prevent under-reservation 429s (tokens already spent before settle runs). | High quantile (p95/p99) × margin (default 1.15), clamped ≤ old ceiling. Existing retry queue + backoff absorb the residual. `safety_margin` and `enabled=False` are user escape hatches.                                                 |
| Reasoning models have correlated, bursty thinking spikes that a pooled p95 may under-cover.                                                                              | Use p99 for reasoning-enabled profiles; reservation still clamped ≤ the full thinking budget ceiling; settlement corrects after.                                                                                                         |
| Concurrency: estimator read (producer thread) vs write (event loop).                                                                                                     | Dedicated `threading.Lock` in the estimator. `settle()` is event-loop-only and documented as such.                                                                                                                                       |
| Behavior change for existing users (default-on).                                                                                                                         | Reservation is always ≤ prior ceiling (never more aggressive than today on a per-request basis beyond the intended throughput gain); conservative default margin; one-line disable.                                                      |
| Small batches (< `min_samples`) never warm up.                                                                                                                           | Documented: they transparently use the static ceiling (no regression, no surprise); settlement still applies.                                                                                                                            |
| During an active 429 backoff, in-flight successes that complete after `backoff()` zeroed the bucket still `settle()` and re-inject capacity the backoff meant to drain.  | Measured by the harness at ~0.4% of the bucket per event when the estimator is warm; only occurs when the client is over-provisioned vs. the true limit. Deferred — tracked in TD-3375 (backoff-generation guard sketch included there). |

---

## Pre-existing Issues (Noted, Not Fixed)

Surfaced during peer review; **out of scope** for this change but recorded:

1. **Retry re-consumes the frozen estimate.** A `QueueItem` re-enqueued on
   timeout/transient (`model_client.py:719-724,746-758`) re-debits the bucket on each
   dispatch but `settle()` fires once on eventual success — N reservations, 1 settlement.
   Not a permanent leak (buckets refill), but suppresses throughput during retry storms.
2. **`_get_queued_requests` retry priority is only on-average** — it races both queues and
   can return one item from each per poll (`model_client.py:810-831`).
3. **Mutex inconsistency** — `Unified`/`Separated.check_and_consume_rate_limit` don't take
   `self.mutex` while `AdaptiveBackoff` does. `settle()` adopts the same event-loop-affinity
   assumption; if the threading model ever changes, all three need consistent locking.

(Codex also flagged an `AttributeError` in `_cancel_in_flight_requests` referencing
`output_tokens_bucket`; **verified false** — no such reference exists in `model_client.py`.)

---

## Testing

- `tests/_inference/test_output_token_estimator.py` — unit: cold-start returns the static
  ceiling; warms up after `min_samples`; reserves the configured quantile × margin; clamps
  to the ceiling; reasoning uses p99; thread-safe under concurrent observe/reserve;
  `enabled=False` always returns the ceiling.
- `tests/_inference/test_rate_limit_settlement.py` — unit on `Unified`/`Separated.settle`:
  over-reservation refunds (clamped to max), under-reservation debits (clamped to 0),
  `AdaptiveBackoff.settle` is a no-op.
- `tests/_inference/test_model_client_estimation.py` — orchestration with lightweight fakes:
  actuals feed the estimator and the bucket; later-batch reservations shrink; dedup hits and
  embeddings are skipped; failed attempts are not settled.
- `tests/_inference/test_adaptive_estimation_config.py` — config validation (`safety_margin`
  bounds, default-on semantics, `None` → enabled) and resolution.
- Extend an existing semantic smoke test to assert `num_reserved_output_tokens` is populated
  (`> 0`). (Aggregate `reserved ≥ actual` holds for typical distributions but is not
  guaranteed under heavy tail-row clustering, so don't assert it as an invariant.)
- Regression: with `enabled=False`, reservations use the static ceiling (no adaptive estimation);
  settlement still runs (always-on) and refunds over-reservation to the bucket.
- `tests/_inference/test_provider_routing_coverage.py` — hermetic Anthropic/Gemini/OpenRouter
  routing (learned estimate drops, API cap unchanged, disabled stays at ceiling) and
  thinking-token settlement (`actual = completion + thinking`).
- `tests/_inference/rate_limit_harness/` + `test_rate_limit_performance.py` — hermetic
  performance harness: a `SimulatedCompletionsClient` drives the **real**
  queue/backoff/settle path against a simulated server-side limiter with configurable output
  distributions; counts logical completions (never retried attempts) and emits a per-event
  trace (dispatch/success/429/backoff/settle).

---

## Measured Results (performance harness)

From `test_rate_limit_performance.py` (n=840 rows, ceiling 200, actual output ~80, real clock):

| Mode                                     | Wall-clock | Reservation efficiency |
| ---------------------------------------- | ---------- | ---------------------- |
| Pre-feature (ceiling reserve, no settle) | 3.14s      | 0.40                   |
| Settlement only (`enabled=False`)        | 0.14s      | 0.40                   |
| Estimation + settlement (`enabled=True`) | 0.13s      | **0.79**               |

- **Settlement alone is ~23× throughput** on this workload; adaptive estimation roughly
  doubles reservation efficiency on top. This is why settlement is always-on.
- Configured ≤ true provider limit ⇒ **0 server-side 429s**; over-provisioned configs
  correctly engage the 429 → backoff → retry path and still drain the batch.
- Regime-shift workloads (estimator-key pooling under-reserves after a tiny→large output
  shift) leaked **0 server-side 429s** — per-success settlement closes the overshoot window
  before the server is hit.
- Settle-after-backoff re-injection (see Risks) measured at ~12 tokens/event ≈ 0.4% of the
  bucket when warm — basis for deferring the backoff-generation guard (TD-3375).

### Live-API confirmation (`benchmarks/rate_limit_stress.py`)

Real OpenAI run (gpt-4.1-nano, tpm=20k, naive `max_output_tokens=2048`, 40 rows/phase of a
varied-output color-extraction task averaging ~12 actual output tokens/row):

| Mode                    | Wall-clock | Reserved out tokens | Efficiency |
| ----------------------- | ---------- | ------------------- | ---------- |
| Pre-feature             | 207.6s     | 81,920              | 0.006      |
| Settlement only         | 3.2s       | 81,920              | 0.006      |
| Adaptive (post-warm-up) | 1.2s       | 1,160               | 0.409      |

**174.6× end-to-end**, identical results and cost, zero provider 429s. Settlement delivers
the bulk (65×); adaptive estimation tightens reservations ~70× and adds another ~2.7× of
wall-clock by admitting the whole batch in one burst.

---

## Summary

A small, default-on, reversible change that closes fenic's open estimate→actual loop:

- ✅ **Settlement** corrects the TPM bucket from the first completion (deterministic).
- ✅ **Adaptive estimation** shrinks up-front reservations using observed usage (opportunistic).
- ✅ **Decoupled** from the API cap, so reservations drop with zero truncation risk.
- ✅ **One shared config**, default-on, single `safety_margin` dial.
- ✅ **No scheduler surgery**, completions only, observable via `LMMetrics`.
