import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from fenic._constants import MINUTE_IN_SECONDS


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
            raise ValueError("rpm must be greater than 0")
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


