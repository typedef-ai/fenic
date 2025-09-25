import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from fenic._constants import MINUTE_IN_SECONDS
from fenic.core.error import ExecutionError, InternalError, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class TokenEstimate:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = field(init=False)

    def __post_init__(self):
        self.total_tokens = self.input_tokens + self.output_tokens

    def __str__(self):
        return f"Input Tokens: {self.input_tokens}, Output Tokens: {self.output_tokens}, Total Tokens: {self.total_tokens}"

    def __add__(self, other):
        return TokenEstimate(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class RateLimitBucket:
    """Manages a token bucket for rate limiting."""
    def __init__(self, max_capacity: int):
        self.max_capacity = max_capacity
        self.current_capacity_ = max_capacity
        self.last_update_time_ = time.time()

    def _get_available_capacity(self, curr_time: float) -> int:
        """Calculates the available capacity based on the elapsed time and refill rate."""
        time_since_last_check = curr_time - self.last_update_time_
        replenished_amount = math.floor(
            self.max_capacity * time_since_last_check / MINUTE_IN_SECONDS
        )
        return min(
            replenished_amount + self.current_capacity_,
            self.max_capacity,
        )

    def _set_capacity(self, capacity: int, curr_time: float):
        """Updates the current capacity and last update time."""
        self.current_capacity_ = capacity
        self.last_update_time_ = curr_time


class RateLimitStrategy(ABC):
    """Base class for implementing rate limiting strategies for language model requests.

    This abstract class defines the interface for rate limiting strategies that control
    both request rate (RPM) and token usage rate (TPM) for language model API calls.
    Subclasses must implement specific token rate limiting strategies.

    Attributes:
        rpm: Requests per minute limit. Must be greater than 0.
        requests_bucket: Token bucket for tracking and limiting request rate.
    """
    def __init__(self, rpm: int):
        if rpm <= 0:
            raise ValidationError("rpm must be greater than 0")
        self.rpm = rpm
        self.requests_bucket = RateLimitBucket(max_capacity=self.rpm)
        self.mutex = threading.Lock()

    @abstractmethod
    def backoff(self, curr_time: float) -> int:
        """Backoff the request/token rate limit bucket."""
        pass

    @abstractmethod
    def check_and_consume_rate_limit(self, token_estimate: TokenEstimate) -> bool:
        """Checks if there is enough capacity in both token and request rate limit buckets.

        If there is sufficient capacity, this method will consume the required tokens
        and request quota. This is an abstract method that must be implemented by subclasses.

        Args:
            token_estimate: A TokenEstimate object containing the estimated input, output,
                          and total tokens for the request.

        Returns:
            bool: True if there was enough capacity and it was consumed, False otherwise.

        Raises:
            ConfigurationError: If there is insufficient capacity to handle the request.
        """
        pass


    @abstractmethod
    def context_tokens_per_minute(self) -> int:
        """Returns the total token rate limit per minute for this strategy.

        This is an abstract method that must be implemented by subclasses to specify
        their token rate limiting behavior.

        Returns:
            int: The total number of tokens allowed per minute.
        """
        pass


class AdaptiveBackoffRateLimitStrategy(RateLimitStrategy):
    """Adaptive RPM limiter with multiplicative backoff and optional additive increase.

    On backoff, reduces RPM by a multiplier and clears burst capacity. When no
    provider RPM hint is active and no cooldown is in effect, increases RPM
    additively after a configurable number of consecutive successful requests,
    up to a configurable maximum. Provider hints (RPM limit and retry-at time)
    clamp RPM and gate scheduling until the cooldown expires. Uses a single
    requests bucket and does not perform token accounting.

    Attributes:
        rpm: The starting RPM.
        min_rpm: The minimum RPM.
        backoff_multiplier: The multiplier to use when backoff is called.
        max_rpm: The maximum RPM.
        additive_increment: The amount to increment the RPM by after a certain number of consecutive successes.
        increase_after_successes: The number of consecutive successes required to increment the RPM.
    """

    def __init__(
        self,
        rpm: int = 25_000,
        min_rpm: int = 50,
        backoff_multiplier: float = 0.75,
        *,
        max_rpm: int | None = None,
        additive_increment: int = 50,
        increase_after_successes: int = 30,
    ):
        super().__init__(rpm=rpm)
        self._min_rpm = max(1, min_rpm)
        if not (0 < backoff_multiplier < 1):
            raise InternalError("backoff multiplier must be between 0 and 1")
        self._backoff_multiplier = backoff_multiplier
        # Cap upward growth; default to the starting rpm
        self._max_rpm = max(rpm, self._min_rpm) if max_rpm is None else max(max_rpm, self._min_rpm)
        # Additive increase controls (only when no provider hint is present)
        self._additive_increment = max(1, additive_increment)
        self._increase_after_successes = max(1, increase_after_successes)
        self._consecutive_successes = 0
        self._rpm_hint: int | None = None
        self._cooldown_until: float = 0.0
        self._on_cooldown = False

    def register_rate_limit_hint(
        self, rpm_hint: int | None, retry_at_epoch_seconds: float | None
    ) -> None:
        """Register provider-processed hints: rpm limit and absolute retry time.

        Args:
            rpm_hint: Max requests per minute allowed by provider (if known)
            retry_at_epoch_seconds: Unix epoch seconds we should not send before (if known)
        """
        with self.mutex:
            if isinstance(rpm_hint, int) and rpm_hint > 0:
                self._rpm_hint = rpm_hint
                self.rpm = min(self.rpm, rpm_hint)
                self.requests_bucket = RateLimitBucket(max_capacity=self.rpm)
            if (
                isinstance(retry_at_epoch_seconds, (int, float))
                and retry_at_epoch_seconds > 0
            ):
                self._cooldown_until = float(retry_at_epoch_seconds)
                if not self._on_cooldown:
                    self._on_cooldown = True
                    logger.warning(
                        f"Provider is throttling requests. Pausing for {retry_at_epoch_seconds - time.time():.2f}s before resuming at the provider specified limit of {rpm_hint} requests per minute."
                    )
            else:
                logger.warning(
                        f"Provider is throttling requests. Resetting RPM to {rpm_hint} requests per minute as specified by the provider."
                    )

    def backoff(self, curr_time: float) -> int:
        """Backoff the request rate limit bucket."""
        with self.mutex:
            # Reduce rpm multiplicatively; clamp by hint and min
            new_rpm = self._rpm_hint if self._rpm_hint else max(self._min_rpm, int(self.rpm * self._backoff_multiplier))
            if new_rpm != self.rpm:
                logger.debug(
                    f"AdaptiveBackoff: reducing rpm multiplicatively from {self.rpm} to {new_rpm} after backoff"
                )
                self.rpm = new_rpm
                # Replace bucket: drop burst capacity
                self.requests_bucket = RateLimitBucket(max_capacity=self.rpm)
            # Zero capacity to yield scheduling immediately after sleep completes
            self.requests_bucket._set_capacity(0, curr_time)
            # Reset growth tracking after backoff
            self._consecutive_successes = 0
        return 0

    def check_and_consume_rate_limit(self, token_estimate: TokenEstimate) -> bool:
        now = time.time()
        # Cooldown gate: do not allow any requests until reset time
        if now < self._cooldown_until:
            return False
        available_requests = self.requests_bucket._get_available_capacity(now)
        if available_requests >= 1:
            self._on_cooldown = False
            self.requests_bucket._set_capacity(available_requests - 1, now)
            # Track successful scheduling and consider additive growth
            self._record_success_and_maybe_grow(now)
            return True
        return False

    def context_tokens_per_minute(self) -> int:
        # Not used; return large sentinel
        return 1_000_000_000

    def __str__(self):
        return f"AdaptiveBackoffRateLimitStrategy(rpm={self.rpm}, min_rpm={self._min_rpm}, backoff_multiplier={self._backoff_multiplier})"

    # Internal helpers
    def _record_success_and_maybe_grow(self, now: float) -> None:
        with self.mutex:
            # Do not grow if provider supplied an explicit rpm hint
            if self._rpm_hint is not None or self._on_cooldown:
                self._consecutive_successes = 0
                return
            self._consecutive_successes += 1
            if (
                self._consecutive_successes >= self._increase_after_successes
                and self.rpm < self._max_rpm
            ):
                new_rpm = min(self._max_rpm, self.rpm + self._additive_increment)
                if new_rpm != self.rpm:
                    # Preserve current available capacity proportionally when resizing
                    available = self.requests_bucket._get_available_capacity(now)
                    self.rpm = new_rpm
                    new_bucket = RateLimitBucket(max_capacity=self.rpm)
                    # Clamp carried capacity to new max
                    new_bucket._set_capacity(min(available, self.rpm), now)
                    self.requests_bucket = new_bucket
                    logger.debug(
                        f"AdaptiveBackoff: increasing rpm additively to {self.rpm} after {self._consecutive_successes} consecutive successes"
                    )
                self._consecutive_successes = 0


class UnifiedTokenRateLimitStrategy(RateLimitStrategy):
    """Rate limiting strategy that uses a single token bucket for both input and output tokens.

    This strategy enforces both a request rate limit (RPM) and a unified token rate limit (TPM)
    where input and output tokens share the same quota.

    Attributes:
        tpm: Total tokens per minute limit. Must be greater than 0.
        unified_tokens_bucket: Token bucket for tracking and limiting total token usage.
    """
    def __init__(self, rpm: int, tpm: int):
        super().__init__(rpm)
        self.tpm = tpm
        self.unified_tokens_bucket = RateLimitBucket(max_capacity=self.tpm)

    def backoff(self, curr_time: float) -> int:
        """Backoff the request/token rate limit bucket."""
        # Eliminate burst capacity, in case of rate limit errors
        self.unified_tokens_bucket._set_capacity(0, curr_time)
        self.requests_bucket._set_capacity(0, curr_time)

    def check_and_consume_rate_limit(self, token_estimate: TokenEstimate) -> bool:
        """Checks and consumes rate limits for both requests and total tokens.

        This implementation uses a single token bucket for both input and output tokens,
        enforcing the total token limit across all token types.

        Args:
            token_estimate: A TokenEstimate object containing the estimated input, output,
                          and total tokens for the request.

        Returns:
            bool: True if there was enough capacity and it was consumed, False otherwise.
        """
        now = time.time()
        self._check_max_rate_limits(token_estimate)
        available_tokens = self.unified_tokens_bucket._get_available_capacity(now)
        available_requests = self.requests_bucket._get_available_capacity(now)
        has_request_capacity = available_requests >= 1
        has_token_capacity = available_tokens >= token_estimate.total_tokens
        has_capacity = has_request_capacity and has_token_capacity
        if has_capacity:
            available_tokens -= token_estimate.total_tokens
            available_requests -= 1
            self.unified_tokens_bucket._set_capacity(available_tokens, now)
            self.requests_bucket._set_capacity(available_requests, now)

        return has_capacity

    def _check_max_rate_limits(self, token_estimate: TokenEstimate):
        """Checks if the strategy is configured with enough capacity to handle the request.
        """
        if self.tpm < token_estimate.total_tokens:
            raise ExecutionError(f"Insufficient capacity to handle the request. TPM limit is {self.tpm} but request requires an estimated {token_estimate.total_tokens} tokens.  Please configure the model with more capacity.")

    def context_tokens_per_minute(self) -> int:
        """Returns the total token rate limit per minute.

        Returns:
            int: The total number of tokens allowed per minute (tpm).
        """
        return self.tpm

    def __str__(self):
        """Returns a string representation of the rate limit strategy.

        Returns:
            str: A string showing the RPM and TPM limits.
        """
        return f"UnifiedTokenRateLimitStrategy(rpm={self.rpm}, tpm={self.tpm})"


class SeparatedTokenRateLimitStrategy(RateLimitStrategy):
    """Rate limiting strategy that uses separate token buckets for input and output tokens.

    This strategy enforces both a request rate limit (RPM) and separate token rate limits
    for input (input_tpm) and output (output_tpm) tokens.

    Attributes:
        input_tpm: Input tokens per minute limit. Must be greater than 0.
        output_tpm: Output tokens per minute limit. Must be greater than 0.
        input_tokens_bucket: Token bucket for tracking and limiting input token usage.
        output_tokens_bucket: Token bucket for tracking and limiting output token usage.
    """
    def __init__(self, rpm: int, input_tpm: int, output_tpm: int):
        super().__init__(rpm)
        self.input_tpm = input_tpm
        self.output_tpm = output_tpm
        self.input_tokens_bucket = RateLimitBucket(max_capacity=self.input_tpm)
        self.output_tokens_bucket = RateLimitBucket(max_capacity=self.output_tpm)

    def backoff(self, curr_time: float) -> int:
        """Backoff the request/token rate limit bucket."""
        # Eliminate burst capacity, in case of rate limit errors
        self.input_tokens_bucket._set_capacity(0, curr_time)
        self.output_tokens_bucket._set_capacity(0, curr_time)
        self.requests_bucket._set_capacity(0, curr_time)

    def check_and_consume_rate_limit(self, token_estimate: TokenEstimate) -> bool:
        """Checks and consumes rate limits for requests, input tokens, and output tokens.

        This implementation uses separate token buckets for input and output tokens,
        enforcing separate limits for each token type.

        Args:
            token_estimate: A TokenEstimate object containing the estimated input, output,
                          and total tokens for the request.

        Returns:
            bool: True if there was enough capacity and it was consumed, False otherwise.
        """
        now = time.time()
        self._check_max_rate_limits(token_estimate)
        available_input_tokens = self.input_tokens_bucket._get_available_capacity(now)
        available_requests = self.requests_bucket._get_available_capacity(now)
        available_output_tokens = self.output_tokens_bucket._get_available_capacity(now)
        has_output_token_capacity = available_output_tokens >= token_estimate.output_tokens
        has_request_capacity = available_requests >= 1
        has_token_capacity = available_input_tokens >= token_estimate.input_tokens
        has_capacity = has_request_capacity and has_token_capacity and has_output_token_capacity
        if has_capacity:
            available_input_tokens -= token_estimate.input_tokens
            available_output_tokens -= token_estimate.output_tokens
            available_requests -= 1
            self.input_tokens_bucket._set_capacity(available_input_tokens, now)
            self.output_tokens_bucket._set_capacity(available_output_tokens, now)
            self.requests_bucket._set_capacity(available_requests, now)
        return has_capacity

    def _check_max_rate_limits(self, token_estimate: TokenEstimate):
        """Checks if the strategy is configured with enough capacity to handle the request.
        """
        if self.input_tpm < token_estimate.input_tokens:
            raise ExecutionError(f"Insufficient capacity to handle the request. Input TPM limit is {self.input_tpm} but request requires an estimated {token_estimate.input_tokens} input tokens.  Please configure the model with more capacity.")
        if self.output_tpm < token_estimate.output_tokens:
            raise ExecutionError(f"Insufficient capacity to handle the request. Output TPM limit is {self.output_tpm} but request requires an estimated {token_estimate.output_tokens} output tokens.  Please configure the model with more capacity.")

    def context_tokens_per_minute(self) -> int:
        """Returns the total token rate limit per minute.

        Returns:
            int: The sum of input and output tokens allowed per minute.
        """
        return self.input_tpm + self.output_tpm

    def __str__(self):
        """Returns a string representation of the rate limit strategy.

        Returns:
            str: A string showing the RPM, input TPM, and output TPM limits.
        """
        return f"SeparatedTokenRateLimitStrategy(rpm={self.rpm}, input_tpm={self.input_tpm}, output_tpm={self.output_tpm})"
