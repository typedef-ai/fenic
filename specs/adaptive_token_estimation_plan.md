# Adaptive Output-Token Estimation & Settlement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close fenic's open estimate→actual loop so the rate limiter reserves a learned (not worst-case) number of output tokens and refunds over-reservation after each response, reclaiming throughput without raising 429 risk.

**Architecture:** Three coordinated pieces, all behind one default-on config. (1) A per-client `OutputTokenEstimator` learns the distribution of _actual_ output tokens and returns a high-quantile reservation clamped to the existing static ceiling. (2) Each provider's output estimate routes through it, while the API `max_tokens` cap path is left untouched (decoupling). (3) `RateLimitStrategy.settle()` refunds `reserved − actual` to the token bucket(s) on each successful response. No scheduler changes; completions only.

**Tech Stack:** Python 3, Pydantic v2 (config), `uv`/`pytest` (tests), `dataclasses`, `threading`, `collections.deque`. Package manager is `uv` (run tests with `uv run pytest ...`).

**Design spec:** `specs/adaptive_token_estimation_design.md`

---

## File Structure

**Create:**

- `src/fenic/_inference/output_token_estimator.py` — the learning estimator (one responsibility: model output-token reservations from observed usage).
- `tests/_inference/test_output_token_estimator.py` (Task 1)
- `tests/_inference/test_rate_limit_settlement.py` (Task 2)
- `tests/_inference/test_metrics_reserved.py` (Task 3)
- `tests/_inference/test_adaptive_estimation_config.py` (Task 4)
- `tests/_inference/test_model_client_estimation.py` (Task 5)
- `tests/_inference/test_adaptive_estimation_wiring.py` (Task 6)
- `tests/_inference/test_provider_output_routing.py` (Task 7)
- `tests/_inference/test_adaptive_estimation_regression.py` (Task 8)

**Modify:**

- `src/fenic/_inference/rate_limit_strategy.py` — add `settle()` (base no-op + Unified + Separated overrides).
- `src/fenic/core/metrics.py` — add `LMMetrics.num_reserved_output_tokens` (+ `__add__`, display).
- `src/fenic/api/session/config.py` — add `AdaptiveTokenEstimationConfig`, `SemanticConfig` field, resolution in `_to_resolved_config()`.
- `src/fenic/core/_resolved_session_config.py` — add `ResolvedAdaptiveTokenEstimationConfig`, field on `ResolvedSemanticConfig`.
- `src/fenic/api/session/__init__.py`, `src/fenic/api/__init__.py`, `src/fenic/__init__.py` — export the new config.
- `src/fenic/_inference/model_client.py` — base wiring (estimator, key, reservation helper, reconcile hook).
- `src/fenic/_backends/local/model_registry.py` — thread resolved config to clients.
- `src/fenic/_inference/openai/openai_batch_chat_completions_client.py`, `.../anthropic/anthropic_batch_chat_completions_client.py`, `.../google/gemini_native_chat_completions_client.py`, `.../openrouter/openrouter_batch_chat_completions_client.py` — constructor param + route output estimate through the estimator.

---

### Task 1: `OutputTokenEstimator`

**Files:**

- Create: `src/fenic/_inference/output_token_estimator.py`
- Test: `tests/_inference/test_output_token_estimator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/_inference/test_output_token_estimator.py
import threading

from fenic._inference.output_token_estimator import OutputTokenEstimator


def _estimator(**kw):
    # small min_samples keeps tests fast
    return OutputTokenEstimator(enabled=True, safety_margin=1.0, min_samples=5, window=256, **kw)


def test_cold_start_returns_static_ceiling():
    est = _estimator()
    # fewer than min_samples observations -> fall back to the static ceiling
    for _ in range(4):
        est.observe(("p", 512), 10)
    assert est.reserve(("p", 512), static_ceiling=8704, reasoning=False) == 8704


def test_warm_reserves_below_ceiling():
    est = _estimator()
    for _ in range(50):
        est.observe(("p", 512), 100)
    # p95 of constant 100 * margin 1.0 == 100, clamped below the 8704 ceiling
    assert est.reserve(("p", 512), static_ceiling=8704, reasoning=False) == 100


def test_reserve_clamps_to_static_ceiling():
    est = _estimator()
    for _ in range(50):
        est.observe(("p", 512), 100000)  # huge actuals
    # learned value would exceed the ceiling, so it is clamped down
    assert est.reserve(("p", 512), static_ceiling=512, reasoning=False) == 512


def test_safety_margin_applied():
    est = OutputTokenEstimator(enabled=True, safety_margin=1.5, min_samples=5, window=256)
    for _ in range(50):
        est.observe(("p", 512), 100)
    assert est.reserve(("p", 512), static_ceiling=8704, reasoning=False) == 150


def test_reasoning_uses_higher_quantile():
    est = _estimator()
    # right-skewed: mostly small, a few large -> p99 > p95
    for _ in range(99):
        est.observe(("p", 512), 100)
    est.observe(("p", 512), 5000)
    p95 = est.reserve(("p", 512), static_ceiling=10000, reasoning=False)
    p99 = est.reserve(("p", 512), static_ceiling=10000, reasoning=True)
    assert p99 > p95


def test_disabled_always_returns_ceiling():
    est = OutputTokenEstimator(enabled=False, safety_margin=1.0, min_samples=5)
    for _ in range(50):
        est.observe(("p", 512), 100)
    assert est.reserve(("p", 512), static_ceiling=8704, reasoning=False) == 8704


def test_keys_are_isolated():
    est = _estimator()
    for _ in range(50):
        est.observe(("p", 512), 100)
    # a different key has no samples -> ceiling
    assert est.reserve(("p", 1024), static_ceiling=4096, reasoning=False) == 4096


def test_concurrent_observe_and_reserve_is_safe():
    est = _estimator()

    def writer():
        for _ in range(2000):
            est.observe(("p", 512), 100)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(2000):
        est.reserve(("p", 512), static_ceiling=8704, reasoning=False)
    for t in threads:
        t.join()
    assert est.reserve(("p", 512), static_ceiling=8704, reasoning=False) == 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_inference/test_output_token_estimator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fenic._inference.output_token_estimator'`

- [ ] **Step 3: Write the implementation**

```python
# src/fenic/_inference/output_token_estimator.py
"""Adaptive output-token reservation estimator.

Learns the distribution of ACTUAL output tokens (completion + thinking) per
(profile_hash, max_completion_tokens) key and returns a high-quantile reservation
for the rate limiter, always clamped to the caller-supplied static ceiling (which
equals the provider's max_tokens cap). Thread-safe: written on the asyncio event
loop, read on the producer thread.
"""

import math
import threading
from collections import defaultdict, deque
from typing import Hashable, Optional


def _quantile(sorted_vals: list[int], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted, non-empty list."""
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[int(pos)])
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


class OutputTokenEstimator:
    """Per-client learned output-token reservation."""

    def __init__(
        self,
        *,
        enabled: bool,
        safety_margin: float,
        min_samples: int = 30,
        window: int = 256,
    ):
        self._enabled = enabled
        self._safety_margin = safety_margin
        self._min_samples = min_samples
        self._window = window
        self._samples: dict[Hashable, deque] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._lock = threading.Lock()

    def reserve(self, key: Hashable, *, static_ceiling: int, reasoning: bool) -> int:
        """Output-token reservation. Always in [1, static_ceiling]."""
        if not self._enabled:
            return static_ceiling
        with self._lock:
            samples: Optional[deque] = self._samples.get(key)
            if samples is None or len(samples) < self._min_samples:
                return static_ceiling
            ordered = sorted(samples)
        q = 0.99 if reasoning else 0.95
        modeled = _quantile(ordered, q) * self._safety_margin
        return max(1, min(static_ceiling, math.ceil(modeled)))

    def observe(self, key: Hashable, actual_output_tokens: int) -> None:
        """Record an actual output-token count for `key`."""
        if not self._enabled:
            return
        with self._lock:
            self._samples[key].append(actual_output_tokens)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_inference/test_output_token_estimator.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/fenic/_inference/output_token_estimator.py tests/_inference/test_output_token_estimator.py
git commit -m "feat(inference): add adaptive OutputTokenEstimator"
```

---

### Task 2: `RateLimitStrategy.settle()`

**Files:**

- Modify: `src/fenic/_inference/rate_limit_strategy.py`
- Test: `tests/_inference/test_rate_limit_settlement.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/_inference/test_rate_limit_settlement.py
import time

from fenic._inference.rate_limit_strategy import (
    AdaptiveBackoffRateLimitStrategy,
    SeparatedTokenRateLimitStrategy,
    TokenEstimate,
    UnifiedTokenRateLimitStrategy,
)


def test_unified_settle_refunds_over_reservation():
    s = UnifiedTokenRateLimitStrategy(rpm=100, tpm=1000)
    reserved = TokenEstimate(input_tokens=200, output_tokens=300)  # total 500
    assert s.check_and_consume_rate_limit(reserved)               # bucket: 1000 -> 500
    actual = TokenEstimate(input_tokens=200, output_tokens=50)     # total 250
    s.settle(reserved, actual)                                     # refund 250 -> ~750
    avail = s.unified_tokens_bucket._get_available_capacity(time.time())
    assert 745 <= avail <= 755


def test_unified_settle_debits_under_reservation():
    s = UnifiedTokenRateLimitStrategy(rpm=100, tpm=1000)
    reserved = TokenEstimate(input_tokens=100, output_tokens=100)  # total 200
    assert s.check_and_consume_rate_limit(reserved)               # bucket: 1000 -> 800
    actual = TokenEstimate(input_tokens=100, output_tokens=400)    # total 500 (used MORE)
    s.settle(reserved, actual)                                     # extra debit 300 -> ~500
    avail = s.unified_tokens_bucket._get_available_capacity(time.time())
    assert 495 <= avail <= 505


def test_unified_settle_clamps_to_capacity():
    s = UnifiedTokenRateLimitStrategy(rpm=100, tpm=1000)
    reserved = TokenEstimate(input_tokens=0, output_tokens=10)
    # never consumed; bucket is full at 1000; a refund must not exceed tpm
    s.settle(reserved, TokenEstimate(input_tokens=0, output_tokens=0))
    avail = s.unified_tokens_bucket._get_available_capacity(time.time())
    assert avail == 1000


def test_separated_settle_refunds_each_bucket():
    s = SeparatedTokenRateLimitStrategy(rpm=100, input_tpm=1000, output_tpm=1000)
    reserved = TokenEstimate(input_tokens=300, output_tokens=400)
    assert s.check_and_consume_rate_limit(reserved)  # in: 700, out: 600
    actual = TokenEstimate(input_tokens=250, output_tokens=50)
    s.settle(reserved, actual)  # refund in 50 -> 750, out 350 -> 950
    now = time.time()
    assert 745 <= s.input_tokens_bucket._get_available_capacity(now) <= 755
    assert 945 <= s.output_tokens_bucket._get_available_capacity(now) <= 955


def test_adaptive_settle_is_noop():
    s = AdaptiveBackoffRateLimitStrategy(rpm=100)
    # no token accounting; must not raise
    s.settle(TokenEstimate(input_tokens=10, output_tokens=10),
             TokenEstimate(input_tokens=5, output_tokens=5))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_inference/test_rate_limit_settlement.py -v`
Expected: FAIL with `AttributeError: 'UnifiedTokenRateLimitStrategy' object has no attribute 'settle'`

- [ ] **Step 3: Add the base no-op `settle()` to `RateLimitStrategy`**

In `src/fenic/_inference/rate_limit_strategy.py`, inside `class RateLimitStrategy(ABC)`, after `context_tokens_per_minute` (around line 109), add a concrete method (NOT abstract, so `AdaptiveBackoffRateLimitStrategy` inherits the no-op):

```python
    def settle(self, reserved: "TokenEstimate", actual: "TokenEstimate") -> None:
        """Correct reserved capacity against actual usage after a response.

        Refunds (reserved - actual) when over-reserved, debits further when
        under-reserved. Base implementation is a no-op for strategies that do not
        track tokens (e.g. AdaptiveBackoff). MUST be called on the asyncio event
        loop thread only (same thread as check_and_consume_rate_limit), so it needs
        no additional locking.
        """
        return None
```

- [ ] **Step 4: Override in `UnifiedTokenRateLimitStrategy`**

Add this method to `UnifiedTokenRateLimitStrategy` (after `_check_max_rate_limits`, around line 304):

```python
    def settle(self, reserved: TokenEstimate, actual: TokenEstimate) -> None:
        now = time.time()
        delta = reserved.total_tokens - actual.total_tokens
        available = self.unified_tokens_bucket._get_available_capacity(now)
        self.unified_tokens_bucket._set_capacity(
            min(self.tpm, max(0, available + delta)), now
        )
```

- [ ] **Step 5: Override in `SeparatedTokenRateLimitStrategy`**

Add this method to `SeparatedTokenRateLimitStrategy` (after `_check_max_rate_limits`, around line 386):

```python
    def settle(self, reserved: TokenEstimate, actual: TokenEstimate) -> None:
        now = time.time()
        in_delta = reserved.input_tokens - actual.input_tokens
        out_delta = reserved.output_tokens - actual.output_tokens
        in_avail = self.input_tokens_bucket._get_available_capacity(now)
        out_avail = self.output_tokens_bucket._get_available_capacity(now)
        self.input_tokens_bucket._set_capacity(
            min(self.input_tpm, max(0, in_avail + in_delta)), now
        )
        self.output_tokens_bucket._set_capacity(
            min(self.output_tpm, max(0, out_avail + out_delta)), now
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/_inference/test_rate_limit_settlement.py -v`
Expected: PASS (5 passed)

- [ ] **Step 7: Commit**

```bash
git add src/fenic/_inference/rate_limit_strategy.py tests/_inference/test_rate_limit_settlement.py
git commit -m "feat(inference): add settle() reconciliation to rate limit strategies"
```

---

### Task 3: `LMMetrics.num_reserved_output_tokens`

**Files:**

- Modify: `src/fenic/core/metrics.py:13-46` (LMMetrics), display strings.
- Test: add to `tests/_inference/test_rate_limit_settlement.py` (or a metrics test if one exists).

> Note: intentionally **not** added to `QueryMetrics.to_dict()` to avoid a metrics-table schema migration. The value stays accessible via the dataclass field and the human-readable summaries.

- [ ] **Step 1: Write the failing test**

```python
# tests/_inference/test_metrics_reserved.py
from fenic.core.metrics import LMMetrics


def test_reserved_field_defaults_zero_and_adds():
    a = LMMetrics(num_output_tokens=10, num_reserved_output_tokens=100)
    b = LMMetrics(num_output_tokens=5, num_reserved_output_tokens=40)
    c = a + b
    assert c.num_output_tokens == 15
    assert c.num_reserved_output_tokens == 140


def test_reserved_field_default_is_zero():
    assert LMMetrics().num_reserved_output_tokens == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/_inference/test_metrics_reserved.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'num_reserved_output_tokens'`

- [ ] **Step 3: Add the field and include it in `__add__`**

In `src/fenic/core/metrics.py`, add the field to `LMMetrics` (after `num_requests`, line 28):

```python
    num_requests: int = 0
    num_reserved_output_tokens: int = 0
```

Update `LMMetrics.__add__` (line 39) to carry it:

```python
        return LMMetrics(
            num_uncached_input_tokens=self.num_uncached_input_tokens + other.num_uncached_input_tokens,
            num_cached_input_tokens=self.num_cached_input_tokens + other.num_cached_input_tokens,
            num_output_tokens=self.num_output_tokens + other.num_output_tokens,
            cost=self.cost + other.cost,
            num_requests=self.num_requests + other.num_requests,
            num_reserved_output_tokens=self.num_reserved_output_tokens + other.num_reserved_output_tokens,
        )
```

- [ ] **Step 4: Surface it in the human-readable summaries**

In `QueryMetrics.__str__` (line 229), change the LM tokens line to:

```python
            f"Language Model Tokens: {self.total_lm_metrics.num_uncached_input_tokens:,} input tokens, {self.total_lm_metrics.num_cached_input_tokens:,} cached input tokens, {self.total_lm_metrics.num_output_tokens:,} output tokens ({self.total_lm_metrics.num_reserved_output_tokens:,} reserved)\n"
```

In `get_execution_plan_details` (line 197), change the LM usage line to:

```python
                        f"{indent_str}  Language Model Usage: {op.lm_metrics.num_uncached_input_tokens:,} input tokens, {op.lm_metrics.num_cached_input_tokens:,} cached input tokens, {op.lm_metrics.num_output_tokens:,} output tokens ({op.lm_metrics.num_reserved_output_tokens:,} reserved)",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/_inference/test_metrics_reserved.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/fenic/core/metrics.py tests/_inference/test_metrics_reserved.py
git commit -m "feat(metrics): track reserved output tokens for reservation efficiency"
```

---

### Task 4: Configuration (`AdaptiveTokenEstimationConfig`)

**Files:**

- Modify: `src/fenic/core/_resolved_session_config.py` (add resolved dataclass + field on `ResolvedSemanticConfig`).
- Modify: `src/fenic/api/session/config.py` (add config class, `SemanticConfig` field, resolution in `_to_resolved_config`).
- Modify: `src/fenic/api/session/__init__.py`, `src/fenic/api/__init__.py`, `src/fenic/__init__.py` (exports).
- Test: `tests/_inference/test_adaptive_estimation_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/_inference/test_adaptive_estimation_config.py
import pytest
from pydantic import ValidationError

from fenic.api.session.config import (
    AdaptiveTokenEstimationConfig,
    OpenAILanguageModel,
    SemanticConfig,
    SessionConfig,
)


def test_defaults_enabled_with_margin():
    cfg = AdaptiveTokenEstimationConfig()
    assert cfg.enabled is True
    assert cfg.safety_margin == 1.15


def test_safety_margin_bounds():
    with pytest.raises(ValidationError):
        AdaptiveTokenEstimationConfig(safety_margin=0.5)
    with pytest.raises(ValidationError):
        AdaptiveTokenEstimationConfig(safety_margin=10.0)


def _session(adaptive=None):
    return SessionConfig(
        app_name="t",
        semantic=SemanticConfig(
            language_models={"m": OpenAILanguageModel(model_name="gpt-4o-mini", rpm=100, tpm=1000)},
            default_language_model="m",
            adaptive_token_estimation=adaptive,
        ),
    )


def test_resolution_defaults_to_enabled_when_absent():
    resolved = _session(None)._to_resolved_config()
    ate = resolved.semantic.adaptive_token_estimation
    assert ate.enabled is True
    assert ate.safety_margin == 1.15


def test_resolution_respects_disabled():
    resolved = _session(AdaptiveTokenEstimationConfig(enabled=False))._to_resolved_config()
    assert resolved.semantic.adaptive_token_estimation.enabled is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_inference/test_adaptive_estimation_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'AdaptiveTokenEstimationConfig'`

- [ ] **Step 3: Add the resolved dataclass**

In `src/fenic/core/_resolved_session_config.py`, after `class ResolvedCacheConfig` (line 178+), add:

```python
@dataclass
class ResolvedAdaptiveTokenEstimationConfig:
    enabled: bool = True
    safety_margin: float = 1.15
```

Then add a field to `ResolvedSemanticConfig` (line 153-157):

```python
@dataclass
class ResolvedSemanticConfig:
    language_models: Optional[ResolvedLanguageModelConfig] = None
    embedding_models: Optional[ResolvedEmbeddingModelConfig] = None
    llm_response_cache: Optional[ResolvedCacheConfig] = None
    adaptive_token_estimation: ResolvedAdaptiveTokenEstimationConfig = field(
        default_factory=ResolvedAdaptiveTokenEstimationConfig
    )
```

Add `field` to the dataclasses import at the top of the file:

```python
from dataclasses import dataclass, field
```

- [ ] **Step 4: Add the public config class**

In `src/fenic/api/session/config.py`, add the config class near `LLMResponseCacheConfig` (around line 1361). Confirm `Field` is already imported from pydantic at the top of the file (it is, used throughout).

```python
class AdaptiveTokenEstimationConfig(BaseModel):
    """Tunes adaptive output-token reservation for rate limiting.

    Output-token reservations are learned from observed usage and clamped to the
    request's max_completion_tokens ceiling, then corrected after each response
    (settlement). Enabled by default; disabling reverts to static worst-case
    reservation with no settlement.
    """

    enabled: bool = True
    safety_margin: float = Field(
        default=1.15,
        ge=1.0,
        le=4.0,
        description=(
            "Multiplier on the modeled output-token reservation. Higher reserves "
            "more (safer, lower throughput); lower reserves less (higher throughput, "
            "higher 429 risk)."
        ),
    )
```

- [ ] **Step 5: Add the field to `SemanticConfig`**

In `src/fenic/api/session/config.py`, add to `SemanticConfig` next to `llm_response_cache` (line 1122):

```python
    llm_response_cache: Optional[LLMResponseCacheConfig] = None
    adaptive_token_estimation: Optional[AdaptiveTokenEstimationConfig] = None
```

- [ ] **Step 6: Resolve it in `_to_resolved_config`**

In `_to_resolved_config()` (config.py:1556), in the block that builds `resolved_semantic` (around line 1694-1711), add the resolution and import. First import the resolved class — add it to the existing `from fenic.core._resolved_session_config import (...)` import group at the top of config.py:

```python
    ResolvedAdaptiveTokenEstimationConfig,
```

Then, just before `resolved_semantic = ResolvedSemanticConfig(...)` (line 1707), add:

```python
        ate_cfg = self.semantic.adaptive_token_estimation if self.semantic else None
        resolved_ate = ResolvedAdaptiveTokenEstimationConfig(
            enabled=ate_cfg.enabled if ate_cfg else True,
            safety_margin=ate_cfg.safety_margin if ate_cfg else 1.15,
        )
```

And pass it into the constructor:

```python
        resolved_semantic = ResolvedSemanticConfig(
            language_models=language_models,
            embedding_models=embedding_models,
            llm_response_cache=resolved_cache,
            adaptive_token_estimation=resolved_ate,
        )
```

- [ ] **Step 7: Export the public class**

Mirror `LLMResponseCacheConfig` in all three export files (add the import and the `__all__` entry next to it):

- `src/fenic/api/session/__init__.py` (cache at lines 12, 38)
- `src/fenic/api/__init__.py` (cache at lines 67, 153)
- `src/fenic/__init__.py` (cache at lines 19, 134)

In each, add `AdaptiveTokenEstimationConfig` to the import from `fenic.api.session.config` and to `__all__`.

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/_inference/test_adaptive_estimation_config.py -v`
Expected: PASS (4 passed)

- [ ] **Step 9: Commit**

```bash
git add src/fenic/core/_resolved_session_config.py src/fenic/api/session/config.py src/fenic/api/session/__init__.py src/fenic/api/__init__.py src/fenic/__init__.py tests/_inference/test_adaptive_estimation_config.py
git commit -m "feat(config): add AdaptiveTokenEstimationConfig (default-on)"
```

---

### Task 5: Base `ModelClient` wiring

**Files:**

- Modify: `src/fenic/_inference/model_client.py` (`__init__`, `_estimator_key`, `_adaptive_output_reservation`, `_reconcile_completion`, `_handle_response` hook).
- Test: `tests/_inference/test_model_client_estimation.py`

This task adds the machinery but does NOT yet change any provider's estimate (Task 7) or registry wiring (Task 6), so behavior is unchanged until those land.

- [ ] **Step 1: Write the failing test (fake client exercises the base methods)**

```python
# tests/_inference/test_model_client_estimation.py
from typing import Union

from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.rate_limit_strategy import (
    TokenEstimate,
    UnifiedTokenRateLimitStrategy,
)
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
    ResponseUsage,
)
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core._resolved_session_config import ResolvedAdaptiveTokenEstimationConfig
from fenic.core.metrics import LMMetrics


class _StubProviderClass:
    _base_url = None


class _FakeCompletionsClient(ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]):
    def __init__(self, strategy, adaptive_estimation=None):
        super().__init__(
            model="gpt-4o-mini",
            model_provider=ModelProvider.OPENAI,
            model_provider_class=_StubProviderClass(),
            rate_limit_strategy=strategy,
            token_counter=TiktokenTokenCounter(model_name="gpt-4o-mini"),
            adaptive_estimation=adaptive_estimation,
        )
        self._metrics = LMMetrics()

    async def make_single_request(self, request):
        return None

    def estimate_tokens_for_request(self, request) -> TokenEstimate:
        return TokenEstimate(input_tokens=10, output_tokens=10)

    def get_metrics(self) -> LMMetrics:
        return self._metrics

    def reset_metrics(self):
        self._metrics = LMMetrics()

    def _get_max_output_token_request_limit(self, request):
        return request.max_completion_tokens


def _request(max_tokens=512):
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="s", examples=[], user="u"),
        max_completion_tokens=max_tokens,
        top_logprobs=None,
        structured_output=None,
        temperature=0.0,
    )


def _client(enabled=True, margin=1.0):
    strategy = UnifiedTokenRateLimitStrategy(rpm=1000, tpm=1_000_000)
    cfg = ResolvedAdaptiveTokenEstimationConfig(enabled=enabled, safety_margin=margin)
    return _FakeCompletionsClient(strategy, adaptive_estimation=cfg)


def test_estimator_key_uses_profile_and_max_tokens():
    client = _client()
    try:
        key = client._estimator_key(_request(512))
        assert key == (client.get_profile_hash_for_request(_request(512)), 512)
    finally:
        client.shutdown()


def test_adaptive_reservation_cold_then_warm():
    client = _client(enabled=True, margin=1.0)
    try:
        req = _request(512)
        # cold-start: returns the static ceiling
        assert client._adaptive_output_reservation(req, static_ceiling=8704, reasoning=False) == 8704
        # feed actuals via the reconcile path, enough to warm up (min_samples=30 default)
        for _ in range(40):
            client._reconcile_completion(
                req,
                TokenEstimate(input_tokens=10, output_tokens=8704),
                ResponseUsage(prompt_tokens=10, completion_tokens=100, total_tokens=110),
            )
        assert client._adaptive_output_reservation(req, static_ceiling=8704, reasoning=False) == 100
    finally:
        client.shutdown()


def test_reconcile_settles_bucket_and_records_metrics():
    client = _client()
    try:
        req = _request(512)
        reserved = TokenEstimate(input_tokens=10, output_tokens=8704)
        client.rate_limit_strategy.check_and_consume_rate_limit(reserved)
        before = client.rate_limit_strategy.unified_tokens_bucket._get_available_capacity(
            __import__("time").time()
        )
        client._reconcile_completion(
            req,
            reserved,
            ResponseUsage(prompt_tokens=10, completion_tokens=50, total_tokens=60),
        )
        after = client.rate_limit_strategy.unified_tokens_bucket._get_available_capacity(
            __import__("time").time()
        )
        assert after > before  # over-reservation refunded
        assert client.get_metrics().num_reserved_output_tokens == 8704
    finally:
        client.shutdown()


def test_reconcile_skips_when_nothing_reserved():
    client = _client()
    try:
        req = _request(512)
        client._reconcile_completion(
            req,
            TokenEstimate(input_tokens=0, output_tokens=0),  # dedup hit reserved nothing
            ResponseUsage(prompt_tokens=10, completion_tokens=50, total_tokens=60),
        )
        assert client.get_metrics().num_reserved_output_tokens == 0
    finally:
        client.shutdown()


def test_disabled_reservation_returns_ceiling_even_after_observations():
    client = _client(enabled=False)
    try:
        req = _request(512)
        for _ in range(40):
            client._reconcile_completion(
                req,
                TokenEstimate(input_tokens=10, output_tokens=8704),
                ResponseUsage(prompt_tokens=10, completion_tokens=100, total_tokens=110),
            )
        assert client._adaptive_output_reservation(req, static_ceiling=8704, reasoning=False) == 8704
    finally:
        client.shutdown()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/_inference/test_model_client_estimation.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'adaptive_estimation'`

- [ ] **Step 3: Add the constructor param and estimator construction**

In `src/fenic/_inference/model_client.py`, add the import near the other `_inference` imports (after line 38):

```python
from fenic._inference.output_token_estimator import OutputTokenEstimator
```

Add the parameter to `ModelClient.__init__` (after `cache` at line 125):

```python
        cache: Optional["LLMResponseCache"] = None,
        adaptive_estimation: Optional["ResolvedAdaptiveTokenEstimationConfig"] = None,
    ):
```

Add the import for the type (top of file, under TYPE_CHECKING is fine, but it is used at runtime for defaults so import directly):

```python
from fenic.core._resolved_session_config import ResolvedAdaptiveTokenEstimationConfig
```

Inside `__init__`, after `self.cache = cache` (line 148), construct the estimator:

```python
        _ate = adaptive_estimation or ResolvedAdaptiveTokenEstimationConfig()
        self._output_estimator = OutputTokenEstimator(
            enabled=_ate.enabled,
            safety_margin=_ate.safety_margin,
        )
```

- [ ] **Step 4: Add the key, reservation, and reconcile helpers**

In `src/fenic/_inference/model_client.py`, add these methods to `ModelClient` (place them near `_count_auxiliary_input_tokens`, around line 293):

```python
    def _estimator_key(self, request: RequestT):
        """Key for the output-token estimator: (profile_hash, max_completion_tokens)."""
        max_tokens = getattr(request, "max_completion_tokens", None)
        return (self.get_profile_hash_for_request(request), max_tokens)

    def _adaptive_output_reservation(
        self, request: RequestT, *, static_ceiling: int, reasoning: bool
    ) -> int:
        """Learned output-token reservation, clamped to the static ceiling."""
        return self._output_estimator.reserve(
            self._estimator_key(request),
            static_ceiling=static_ceiling,
            reasoning=reasoning,
        )

    def _reconcile_completion(self, request, reserved, usage) -> None:
        """Feed actuals to the estimator and settle the bucket after a response.

        Event-loop thread only (called from _handle_response). Skips dedup hits
        that reserved nothing.
        """
        if reserved.total_tokens <= 0:
            return
        actual = TokenEstimate(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens + usage.thinking_tokens,
        )
        self._output_estimator.observe(self._estimator_key(request), actual.output_tokens)
        self.rate_limit_strategy.settle(reserved, actual)
        self.get_metrics().num_reserved_output_tokens += reserved.output_tokens
```

- [ ] **Step 5: Call it from `_handle_response`**

In `_handle_response` (model_client.py:735-783), in the `else` (success) branch, right after the existing cache-write block and before `# Set result` (around line 781), add:

```python
            # Reconcile reserved vs actual tokens (adaptive estimation + settlement)
            if (
                isinstance(maybe_response, FenicCompletionsResponse)
                and maybe_response.usage is not None
            ):
                self._reconcile_completion(
                    queue_item.request,
                    queue_item.estimated_tokens,
                    maybe_response.usage,
                )
```

`FenicCompletionsResponse` is already imported in `model_client.py` (line 39-43 import group). `TokenEstimate` is already imported (line 31-34).

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/_inference/test_model_client_estimation.py -v`
Expected: PASS (5 passed)

- [ ] **Step 7: Commit**

```bash
git add src/fenic/_inference/model_client.py tests/_inference/test_model_client_estimation.py
git commit -m "feat(inference): wire estimator + settlement into ModelClient reconcile hook"
```

---

### Task 6: Thread the resolved config through the registry and provider constructors

**Files:**

- Modify: `src/fenic/_backends/local/model_registry.py` (`__init__`, `_initialize_language_model`, 4 client constructions).
- Modify: the 4 completion client constructors to accept + forward `adaptive_estimation`.

No behavior change yet (providers still compute the static ceiling); this just makes each client's estimator reflect the user's config.

- [ ] **Step 1: Add `adaptive_estimation` to each completion client constructor**

For each of the four files, add the parameter after `cache` and forward it to `super().__init__(...)`.

`src/fenic/_inference/openai/openai_batch_chat_completions_client.py` — add param after `cache` (line 46) and forward (in the `super().__init__` call, line 65-74):

```python
        cache: Optional["LLMResponseCache"] = None,
        base_url: Optional[str] = None,
        adaptive_estimation=None,
    ):
```

and in `super().__init__(...)`:

```python
            cache=cache,
            adaptive_estimation=adaptive_estimation,
        )
```

`src/fenic/_inference/anthropic/anthropic_batch_chat_completions_client.py` — add param after `base_url` (line 81) and forward (super call line 96-107):

```python
        base_url: Optional[str] = None,
        adaptive_estimation=None,
    ):
```

and in `super().__init__(...)`:

```python
            cache=cache,
            adaptive_estimation=adaptive_estimation,
        )
```

`src/fenic/_inference/google/gemini_native_chat_completions_client.py` — add param after `cache` (line 78) and forward (super call ends line 103):

```python
        cache: Optional["LLMResponseCache"] = None,
        adaptive_estimation=None,
    ):
```

and in `super().__init__(...)`:

```python
            cache=cache,
            adaptive_estimation=adaptive_estimation,
        )
```

`src/fenic/_inference/openrouter/openrouter_batch_chat_completions_client.py` — add param after `cache` (line 68) and forward (super call ends line 92):

```python
        cache: Optional[LLMResponseCache] = None,
        adaptive_estimation=None,
    ):
```

and in `super().__init__(...)`:

```python
            cache=cache,
            adaptive_estimation=adaptive_estimation,
        )
```

- [ ] **Step 2: Thread the config through the registry**

In `src/fenic/_backends/local/model_registry.py`, change `_initialize_language_model` to accept the config (line 291-292):

```python
    def _initialize_language_model(
        self, model_config: ResolvedModelConfig, cache=None, adaptive_estimation=None
    ) -> LanguageModel:
```

In `__init__`, read it from `config` and pass it down (line 76-77):

```python
            adaptive_estimation = config.adaptive_token_estimation
            for alias, model_config in language_model_config.model_configs.items():
                model = self._initialize_language_model(model_config, cache, adaptive_estimation)
```

Add `adaptive_estimation=adaptive_estimation` to all four completion-client constructions (lines 313-320, 336-343, 357-364, 367-373). For example, the OpenAI one becomes:

```python
                client = OpenAIBatchChatCompletionsClient(
                    model=model_config.model_name,
                    rate_limit_strategy=rate_limit_strategy,
                    profiles=model_config.profiles,
                    default_profile_name=model_config.default_profile,
                    cache=cache,
                    base_url=model_config.base_url,
                    adaptive_estimation=adaptive_estimation,
                )
```

Apply the same `adaptive_estimation=adaptive_estimation` addition to the `AnthropicBatchCompletionsClient`, `GeminiNativeChatCompletionsClient`, and `OpenRouterBatchChatCompletionsClient` constructions.

- [ ] **Step 3: Write a wiring test**

```python
# tests/_inference/test_adaptive_estimation_wiring.py
from fenic.api.session.config import (
    AdaptiveTokenEstimationConfig,
    OpenAILanguageModel,
    SemanticConfig,
    SessionConfig,
)
from fenic._backends.local.model_registry import SessionModelRegistry


def _registry(adaptive, monkeypatch):
    # SessionModelRegistry.__init__ validates provider API keys with a LIVE network
    # call (model_registry.py:100-105). A dummy env var still 401s, so stub it out.
    monkeypatch.setattr(
        "fenic._backends.local.model_registry._validate_provider_api_keys",
        lambda providers: None,
    )
    semantic = SemanticConfig(
        language_models={"m": OpenAILanguageModel(model_name="gpt-4o-mini", rpm=100, tpm=100000)},
        default_language_model="m",
        adaptive_token_estimation=adaptive,
    )
    resolved = SessionConfig(app_name="t", semantic=semantic)._to_resolved_config()
    return SessionModelRegistry(resolved.semantic)


def test_client_estimator_reflects_config(monkeypatch):
    reg = _registry(AdaptiveTokenEstimationConfig(safety_margin=1.5), monkeypatch)
    try:
        client = reg.get_language_model().client
        assert client._output_estimator._enabled is True
        assert client._output_estimator._safety_margin == 1.5
    finally:
        reg.shutdown_models()


def test_client_estimator_disabled(monkeypatch):
    reg = _registry(AdaptiveTokenEstimationConfig(enabled=False), monkeypatch)
    try:
        client = reg.get_language_model().client
        assert client._output_estimator._enabled is False
    finally:
        reg.shutdown_models()
```

> The `monkeypatch.setattr` above is **required** — without it the registry's `__init__` makes a real `client.models.list()` call and fails with a 401 before the estimator is ever constructed. Cleanup uses `reg.shutdown_models()` (model_registry.py:201), the actual method name.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/_inference/test_adaptive_estimation_wiring.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/fenic/_backends/local/model_registry.py src/fenic/_inference/openai/openai_batch_chat_completions_client.py src/fenic/_inference/anthropic/anthropic_batch_chat_completions_client.py src/fenic/_inference/google/gemini_native_chat_completions_client.py src/fenic/_inference/openrouter/openrouter_batch_chat_completions_client.py tests/_inference/test_adaptive_estimation_wiring.py
git commit -m "feat(inference): thread adaptive estimation config to all completion clients"
```

---

### Task 7: Route each provider's output estimate through the estimator (the decoupling)

**Files:**

- Modify the output-estimate method in each of the four completion clients. The API-cap methods (`_get_max_output_token_request_limit` / `get_max_output_token_request_limit`) are **NOT** touched.
- Test: `tests/_inference/test_provider_output_routing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/_inference/test_provider_output_routing.py
from fenic._inference.openai.openai_batch_chat_completions_client import (
    OpenAIBatchChatCompletionsClient,
)
from fenic._inference.rate_limit_strategy import UnifiedTokenRateLimitStrategy, TokenEstimate
from fenic._inference.types import FenicCompletionsRequest, LMRequestMessages, ResponseUsage
from fenic.core._resolved_session_config import ResolvedAdaptiveTokenEstimationConfig


def _req(max_tokens=512):
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="s", examples=[], user="u"),
        max_completion_tokens=max_tokens,
        top_logprobs=None,
        structured_output=None,
        temperature=0.0,
    )


def _openai_client(margin=1.0):
    return OpenAIBatchChatCompletionsClient(
        model="gpt-4o-mini",
        rate_limit_strategy=UnifiedTokenRateLimitStrategy(rpm=1000, tpm=1_000_000),
        adaptive_estimation=ResolvedAdaptiveTokenEstimationConfig(enabled=True, safety_margin=margin),
    )


def test_openai_output_estimate_drops_after_learning():
    client = _openai_client(margin=1.0)
    try:
        req = _req(512)
        ceiling = client.estimate_tokens_for_request(req).output_tokens  # cold = static ceiling
        for _ in range(40):
            client._reconcile_completion(
                req,
                TokenEstimate(input_tokens=10, output_tokens=ceiling),
                ResponseUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            )
        learned = client.estimate_tokens_for_request(req).output_tokens
        assert learned < ceiling
        assert learned == 20  # p95 of constant 20 * margin 1.0
        # the API cap is unchanged (still the generous ceiling)
        assert client._get_max_output_token_request_limit(req) == 512
    finally:
        client.shutdown()
```

> `gpt-4o-mini` is non-reasoning, so its `expected_additional_reasoning_tokens` is 0 and the static ceiling equals `max_completion_tokens` (512). The API-cap assertion confirms the decoupling.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/_inference/test_provider_output_routing.py -v`
Expected: FAIL — `learned == 512` (still the static ceiling, because routing isn't wired yet).

- [ ] **Step 3: Route OpenAI's `_estimate_output_tokens`**

In `src/fenic/_inference/openai/openai_batch_chat_completions_client.py`, replace `_estimate_output_tokens` (lines 135-146) with:

```python
    def _estimate_output_tokens(self, request: FenicCompletionsRequest) -> int:
        """Estimate the number of output tokens for a request."""
        base_tokens = request.max_completion_tokens or 0
        if request.max_completion_tokens is None and request.messages.user_file:
            base_tokens += self.token_counter.count_file_output_tokens(
                messages=request.messages
            )
        profile_config = self._profile_manager.get_profile_by_name(request.model_profile)
        reasoning_tokens = profile_config.expected_additional_reasoning_tokens
        static_ceiling = base_tokens + reasoning_tokens
        return self._adaptive_output_reservation(
            request, static_ceiling=static_ceiling, reasoning=reasoning_tokens > 0
        )
```

- [ ] **Step 4: Route Anthropic's output estimate**

In `src/fenic/_inference/anthropic/anthropic_batch_chat_completions_client.py`, replace `estimate_tokens_for_request` (lines 364-381) with:

```python
    def estimate_tokens_for_request(self, request: FenicCompletionsRequest):
        """Estimate the number of tokens for a request."""
        input_tokens = self.count_tokens(request.messages)
        input_tokens += self._count_auxiliary_input_tokens(request)

        thinking_budget = self._profile_manager.get_profile_by_name(
            request.model_profile
        ).thinking_token_budget
        static_ceiling = request.max_completion_tokens + thinking_budget
        output_tokens = self._adaptive_output_reservation(
            request, static_ceiling=static_ceiling, reasoning=thinking_budget > 0
        )
        return TokenEstimate(input_tokens=input_tokens, output_tokens=output_tokens)
```

(`_get_max_output_token_request_limit` at lines 328-344 is unchanged.)

- [ ] **Step 5: Route Gemini's `_estimate_output_tokens`**

In `src/fenic/_inference/google/gemini_native_chat_completions_client.py`, replace `_estimate_output_tokens` (lines 362-372) with:

```python
    def _estimate_output_tokens(self, request: FenicCompletionsRequest) -> int:
        """Estimate the number of output tokens for a request."""
        estimated_output_tokens = request.max_completion_tokens or 0
        if request.max_completion_tokens is None and request.messages.user_file:
            estimated_output_tokens = self.token_counter.count_file_output_tokens(
                request.messages
            )
        reasoning_tokens = self._get_expected_additional_reasoning_tokens(request)
        static_ceiling = estimated_output_tokens + reasoning_tokens
        return self._adaptive_output_reservation(
            request, static_ceiling=static_ceiling, reasoning=reasoning_tokens > 0
        )
```

- [ ] **Step 6: Route OpenRouter's `_estimate_output_tokens`**

In `src/fenic/_inference/openrouter/openrouter_batch_chat_completions_client.py`, replace `_estimate_output_tokens` (lines 267-273) with:

```python
    def _estimate_output_tokens(self, request: FenicCompletionsRequest) -> int:
        """Estimate the number of output tokens for a request."""
        base_tokens = request.max_completion_tokens or 0
        if request.max_completion_tokens is None and request.messages.user_file:
            base_tokens += self.token_counter.count_file_output_tokens(messages=request.messages)
        reasoning_tokens = self._get_expected_additional_reasoning_tokens(request)
        static_ceiling = base_tokens + reasoning_tokens
        return self._adaptive_output_reservation(
            request, static_ceiling=static_ceiling, reasoning=reasoning_tokens > 0
        )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/_inference/test_provider_output_routing.py -v`
Expected: PASS (1 passed)

- [ ] **Step 8: Commit**

```bash
git add src/fenic/_inference/openai/openai_batch_chat_completions_client.py src/fenic/_inference/anthropic/anthropic_batch_chat_completions_client.py src/fenic/_inference/google/gemini_native_chat_completions_client.py src/fenic/_inference/openrouter/openrouter_batch_chat_completions_client.py tests/_inference/test_provider_output_routing.py
git commit -m "feat(inference): route provider output estimates through adaptive estimator"
```

---

### Task 8: Regression + full-suite verification

**Files:**

- Test: `tests/_inference/test_adaptive_estimation_regression.py`

- [ ] **Step 1: Write the disabled-path regression test**

```python
# tests/_inference/test_adaptive_estimation_regression.py
from fenic._inference.openai.openai_batch_chat_completions_client import (
    OpenAIBatchChatCompletionsClient,
)
from fenic._inference.rate_limit_strategy import UnifiedTokenRateLimitStrategy, TokenEstimate
from fenic._inference.types import FenicCompletionsRequest, LMRequestMessages, ResponseUsage
from fenic.core._resolved_session_config import ResolvedAdaptiveTokenEstimationConfig


def _req():
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="s", examples=[], user="u"),
        max_completion_tokens=512,
        top_logprobs=None,
        structured_output=None,
        temperature=0.0,
    )


def test_disabled_matches_static_ceiling_after_observations():
    client = OpenAIBatchChatCompletionsClient(
        model="gpt-4o-mini",
        rate_limit_strategy=UnifiedTokenRateLimitStrategy(rpm=1000, tpm=1_000_000),
        adaptive_estimation=ResolvedAdaptiveTokenEstimationConfig(enabled=False),
    )
    try:
        req = _req()
        baseline = client.estimate_tokens_for_request(req).output_tokens
        for _ in range(40):
            client._reconcile_completion(
                req,
                TokenEstimate(input_tokens=10, output_tokens=baseline),
                ResponseUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        # disabled -> estimate never moves from the static ceiling
        assert client.estimate_tokens_for_request(req).output_tokens == baseline == 512
    finally:
        client.shutdown()
```

- [ ] **Step 2: Run the regression test**

Run: `uv run pytest tests/_inference/test_adaptive_estimation_regression.py -v`
Expected: PASS (1 passed)

- [ ] **Step 3: Run the full local suite + lint**

Run:

```bash
uv run pytest tests/_inference -v
uv run ruff check src/fenic/_inference src/fenic/core/metrics.py src/fenic/api/session/config.py src/fenic/core/_resolved_session_config.py
```

Expected: all inference tests pass; ruff reports no errors. Fix any failures before continuing.

- [ ] **Step 4: Run the broader local test target**

Run: `just test-local`
Expected: PASS (no regressions). If a metrics snapshot/golden test fails because of the new `(N reserved)` summary text, update that golden to match.

- [ ] **Step 5: Commit**

```bash
git add tests/_inference/test_adaptive_estimation_regression.py
git commit -m "test(inference): regression-guard disabled adaptive estimation path"
```

---

## Self-Review

**Spec coverage:**

- Settlement (deterministic backbone) → Task 2 (`settle()`), Task 5 (reconcile hook).
- Adaptive estimation (accelerator) → Task 1 (estimator), Task 7 (provider routing).
- Decouple reservation from API cap → Task 7 (cap methods untouched; routing test asserts cap unchanged).
- Default-on, single `safety_margin` dial → Task 4 (config), Task 6 (wiring).
- Shared config home (cache pattern) → Task 4 + Task 6.
- Estimator key `(profile_hash, max_completion_tokens)` → Task 5 (`_estimator_key`).
- p95/p99 + clamp + cold-start → Task 1.
- Observability (`num_reserved_output_tokens`) → Task 3, recorded in Task 5.
- Completions-only scope → Task 5 (`isinstance` gate), embeddings constructors untouched.
- No scheduler changes → confirmed: `_process_queue` is never modified.

**Placeholder scan:** none — every step has concrete code/commands.

**Type consistency:** `OutputTokenEstimator.reserve(key, static_ceiling, reasoning)` / `.observe(key, n)`, `RateLimitStrategy.settle(reserved, actual)`, `ModelClient._estimator_key` / `_adaptive_output_reservation(request, static_ceiling, reasoning)` / `_reconcile_completion(request, reserved, usage)`, `ResolvedAdaptiveTokenEstimationConfig(enabled, safety_margin)` are used consistently across Tasks 1–8.

**Known follow-ups (out of scope, from the spec):** provider rate-limit header discovery; 429-specific refunds; per-row input-aware scaling; the pre-existing retry double-consume and `_get_queued_requests` retry-priority behaviors.
