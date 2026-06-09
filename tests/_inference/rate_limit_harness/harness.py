"""Hermetic, real-clock rate-limit performance harness for fenic ModelClient.

This harness exercises the *real* `ModelClient` queue/scheduler, rate-limit gate,
backoff, settlement, and adaptive output-token estimation, but never touches a
provider API. A `SimulatedCompletionsClient` subclasses `ModelClient` directly
(no API key) and a `SimulatedServerLimiter` models the provider's true
server-side TPM/RPM ceiling using the same `RateLimitBucket` math the client
uses, so we can measure when the client overshoots and gets 429'd.

Key invariants (see task spec / model_client.py):
  * `make_single_request` RETURNS a `TransientException` (never raises) to engage
    the retry/backoff path through `_handle_response`.
  * Throughput counts LOGICAL completions and success-only tokens. Retried
    attempts are NOT counted as throughput.
  * REAL clock everywhere. The rate-limit buckets read `time.time()`, and
    request latency is a real `asyncio.sleep`. We never monkeypatch time, which
    would split the wall clock from the event loop's monotonic clock.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from fenic._inference.model_client import ModelClient, TransientException
from fenic._inference.rate_limit_strategy import (
    RateLimitBucket,
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
    """Minimal stand-in for a ModelProviderClass (only `_base_url` is read)."""

    _base_url = None


# ---------------------------------------------------------------------------
# Output distributions
# ---------------------------------------------------------------------------
#
# Each helper returns a *callable* `(n_rows, seed) -> list[int]` that produces a
# fully precomputed, deterministic list of per-row output-token draws. Drawing
# everything up front (keyed by row index, not dispatch order) means each row's
# actual output is stable regardless of dispatch/retry interleaving.

OutputSpec = Callable[[int, int], List[int]]


def constant(v: int) -> OutputSpec:
    """Every row emits exactly `v` output tokens."""

    def _draw(n_rows: int, seed: int) -> List[int]:
        return [int(v)] * n_rows

    return _draw


def lognormal(mean: float, sigma: float) -> OutputSpec:
    """Lognormal output sizes (heavy right tail), seeded and deterministic.

    `mean` is the mean of the underlying normal (i.e. exp(mean) is the median of
    the draw); `sigma` is its standard deviation.
    """

    def _draw(n_rows: int, seed: int) -> List[int]:
        rng = random.Random(seed)
        return [max(1, int(rng.lognormvariate(mean, sigma))) for _ in range(n_rows)]

    return _draw


def regime_shift(small: int, large: int) -> OutputSpec:
    """First half of rows emit `small` tokens, second half emit `large`.

    Models a workload whose output size distribution shifts mid-stream — the
    estimator learns a small reservation early then is systematically under
    water once the large regime begins.
    """

    def _draw(n_rows: int, seed: int) -> List[int]:
        half = n_rows // 2
        return [int(small)] * half + [int(large)] * (n_rows - half)

    return _draw


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass
class RateLimitScenario:
    """Declarative description of one rate-limit performance run.

    Attributes:
        rpm/tpm: limits the CLIENT is configured with.
        true_rpm/true_tpm: limits the simulated server actually enforces.
        n_rows: number of unique requests in the batch.
        static_ceiling: provider max_tokens cap / disabled-reservation amount.
        input_tokens: per-request input token count (fixed).
        output_spec: callable `(n_rows, seed) -> list[int]` of per-row outputs.
        seed: RNG seed for the output draws.
        latency_s: per-request simulated latency (real asyncio.sleep).
        safety_margin: adaptive estimator safety multiplier.
        enabled: adaptive output-token estimation on/off.
        settlement_enabled: when False, settle() is replaced by a no-op to
            simulate PRE-FEATURE behavior (ceiling reservation, never refunded).
    """

    rpm: int
    tpm: int
    true_rpm: int
    true_tpm: int
    n_rows: int
    static_ceiling: int
    input_tokens: int
    output_spec: OutputSpec
    seed: int = 1234
    latency_s: float = 0.002
    safety_margin: float = 1.15
    enabled: bool = True
    settlement_enabled: bool = True

    # Precomputed at build time so each row's output is deterministic.
    output_draws: List[int] = field(init=False)

    def __post_init__(self) -> None:
        self.output_draws = self.output_spec(self.n_rows, self.seed)


# ---------------------------------------------------------------------------
# Simulated server-side limiter
# ---------------------------------------------------------------------------


class SimulatedServerLimiter:
    """Models the PROVIDER's true server-side rate limit.

    Reuses the real `RateLimitBucket` refill math. A request is admitted only
    when both the request bucket (>=1) and the token bucket (>= total_tokens)
    have capacity; otherwise it is rejected, which the client sees as a 429.
    """

    def __init__(self, true_rpm: int, true_tpm: int):
        self._requests = RateLimitBucket(max_capacity=true_rpm)
        self._tokens = RateLimitBucket(max_capacity=true_tpm)

    def try_consume(self, now: float, total_tokens: int) -> bool:
        """Admit (and consume) iff both buckets have capacity. Else 429."""
        avail_req = self._requests._get_available_capacity(now)
        avail_tok = self._tokens._get_available_capacity(now)
        if avail_req >= 1 and avail_tok >= total_tokens:
            self._requests._set_capacity(avail_req - 1, now)
            self._tokens._set_capacity(avail_tok - total_tokens, now)
            return True
        return False


# ---------------------------------------------------------------------------
# Simulated completions client
# ---------------------------------------------------------------------------


class SimulatedCompletionsClient(
    ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    """A ModelClient that simulates a provider with a true server-side limit.

    Exercises the real queue, rate-limit gate, backoff, settlement, and adaptive
    estimation. `make_single_request` returns a TransientException on a simulated
    429 (never raises) so the retry/backoff path engages.
    """

    def __init__(self, scenario: RateLimitScenario):
        strategy = UnifiedTokenRateLimitStrategy(rpm=scenario.rpm, tpm=scenario.tpm)
        super().__init__(
            model="sim",
            model_provider=ModelProvider.OPENAI,
            model_provider_class=_StubProviderClass(),
            rate_limit_strategy=strategy,
            token_counter=TiktokenTokenCounter(model_name="gpt-4o-mini"),
            adaptive_estimation=ResolvedAdaptiveTokenEstimationConfig(
                enabled=scenario.enabled, safety_margin=scenario.safety_margin
            ),
        )
        self._metrics = LMMetrics()
        self.scenario = scenario
        self.output_draws = scenario.output_draws
        self.server = SimulatedServerLimiter(
            true_rpm=scenario.true_rpm, true_tpm=scenario.true_tpm
        )
        # Event trace: list of tuples whose first element is the event name.
        self.trace: List[tuple] = []

        # Instrument settle() and backoff() by wrapping the real strategy methods.
        original_settle = strategy.settle
        original_backoff = strategy.backoff

        if scenario.settlement_enabled:

            def _settle(reserved: TokenEstimate, actual: TokenEstimate) -> None:
                self.trace.append(
                    (
                        "settle",
                        time.time(),
                        reserved.output_tokens,
                        actual.output_tokens,
                    )
                )
                return original_settle(reserved, actual)

            strategy.settle = _settle  # type: ignore[method-assign]
        else:
            # Pre-feature: never refund reserved capacity.
            def _settle_noop(reserved: TokenEstimate, actual: TokenEstimate) -> None:
                self.trace.append(("settle_noop", time.time()))

            strategy.settle = _settle_noop  # type: ignore[method-assign]

        def _backoff(curr_time: float) -> int:
            self.trace.append(("backoff", time.time()))
            return original_backoff(curr_time)

        strategy.backoff = _backoff  # type: ignore[method-assign]

    # -- token estimation -------------------------------------------------

    def estimate_tokens_for_request(
        self, request: FenicCompletionsRequest
    ) -> TokenEstimate:
        input_tokens = self.scenario.input_tokens
        output = self._adaptive_output_reservation(
            request, static_ceiling=self.scenario.static_ceiling, reasoning=False
        )
        return TokenEstimate(input_tokens=input_tokens, output_tokens=output)

    def _get_max_output_token_request_limit(
        self, request: FenicCompletionsRequest
    ) -> int:
        return self.scenario.static_ceiling

    # -- the simulated provider call -------------------------------------

    @staticmethod
    def _row_index(request: FenicCompletionsRequest) -> int:
        """Parse the row index encoded as `row-{i}` in the user content."""
        user = request.messages.user or ""
        return int(user.split("row-", 1)[1])

    async def make_single_request(self, request: FenicCompletionsRequest):
        i = self._row_index(request)
        actual_out = self.output_draws[i]
        total = self.scenario.input_tokens + actual_out
        self.trace.append(("dispatch", time.time(), i))
        await asyncio.sleep(self.scenario.latency_s)
        if not self.server.try_consume(time.time(), total):
            self.trace.append(("server_429", time.time(), i))
            return TransientException(Exception("simulated 429"))
        self._metrics.num_output_tokens += actual_out
        self._metrics.num_uncached_input_tokens += self.scenario.input_tokens
        self._metrics.num_requests += 1
        self.trace.append(("success", time.time(), i))
        return FenicCompletionsResponse(
            completion="ok",
            logprobs=None,
            usage=ResponseUsage(
                prompt_tokens=self.scenario.input_tokens,
                completion_tokens=actual_out,
                total_tokens=total,
                thinking_tokens=0,
            ),
        )

    # -- metrics / profile plumbing --------------------------------------

    def get_metrics(self) -> LMMetrics:
        return self._metrics

    def reset_metrics(self):
        self._metrics = LMMetrics()

    def get_profile_hash(self, profile_name: Optional[str]) -> Optional[str]:
        return None


# ---------------------------------------------------------------------------
# Report + runner
# ---------------------------------------------------------------------------


@dataclass
class RateLimitReport:
    """Measured outcome of one scenario run.

    All token counts are SUCCESS-ONLY; completion counts are LOGICAL (one per
    unique row, regardless of how many attempts it took).
    """

    wall: float
    logical_completions: int
    n_rows: int
    actual_output_tokens: int
    reserved_output_tokens: int
    total_attempts: int
    server_429: int
    retries: int
    reservation_efficiency: float
    achieved_output_tpm: float

    def __str__(self) -> str:
        return (
            "RateLimitReport(\n"
            f"  wall={self.wall:.3f}s\n"
            f"  logical_completions={self.logical_completions}/{self.n_rows}\n"
            f"  actual_output_tokens={self.actual_output_tokens}\n"
            f"  reserved_output_tokens={self.reserved_output_tokens}\n"
            f"  total_attempts={self.total_attempts}\n"
            f"  server_429={self.server_429}\n"
            f"  retries={self.retries}\n"
            f"  reservation_efficiency={self.reservation_efficiency:.3f}\n"
            f"  achieved_output_tpm={self.achieved_output_tpm:.0f}\n"
            ")"
        )


def _build_requests(scenario: RateLimitScenario) -> List[FenicCompletionsRequest]:
    """One unique request per row; user content encodes the row index."""
    return [
        FenicCompletionsRequest(
            messages=LMRequestMessages(system="s", examples=[], user=f"row-{i}"),
            max_completion_tokens=scenario.static_ceiling,
            top_logprobs=None,
            structured_output=None,
            temperature=0.0,
        )
        for i in range(scenario.n_rows)
    ]


def run_scenario(
    scenario: RateLimitScenario,
) -> Tuple[RateLimitReport, List[tuple]]:
    """Run one scenario end-to-end and return (report, trace).

    Builds the simulated client, submits a batch of unique requests through the
    real `make_batch_requests` path, and derives throughput/efficiency metrics
    from success-only token counts and logical completion counts.
    """
    client = SimulatedCompletionsClient(scenario)
    try:
        requests = _build_requests(scenario)
        submit_start = time.time()
        client.make_batch_requests(requests, "perf")
        wall = time.time() - submit_start

        m = client.get_metrics()
        # `num_requests` is incremented only on success in make_single_request.
        logical_completions = m.num_requests
        actual_output_tokens = m.num_output_tokens
        reserved_output_tokens = m.num_reserved_output_tokens

        total_attempts = sum(1 for e in client.trace if e[0] == "dispatch")
        server_429 = sum(1 for e in client.trace if e[0] == "server_429")
        retries = total_attempts - logical_completions

        reservation_efficiency = actual_output_tokens / max(1, reserved_output_tokens)
        achieved_output_tpm = 60.0 * actual_output_tokens / wall if wall > 0 else 0.0

        report = RateLimitReport(
            wall=wall,
            logical_completions=logical_completions,
            n_rows=scenario.n_rows,
            actual_output_tokens=actual_output_tokens,
            reserved_output_tokens=reserved_output_tokens,
            total_attempts=total_attempts,
            server_429=server_429,
            retries=retries,
            reservation_efficiency=reservation_efficiency,
            achieved_output_tpm=achieved_output_tpm,
        )
        return report, list(client.trace)
    finally:
        client.shutdown()
