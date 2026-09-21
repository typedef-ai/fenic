"""Request-local retry limits for the shared model scheduler."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Union

import pytest

from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.rate_limit_strategy import RateLimitStrategy, TokenEstimate
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
)
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core.error import ExecutionError
from fenic.core.metrics import LMMetrics


class _Provider:
    _base_url = None


class _RateLimit(RateLimitStrategy):
    def __init__(self):
        super().__init__(rpm=1_000)

    def backoff(self, curr_time: float) -> int:
        return 0

    def check_and_consume_rate_limit(self, token_estimate: TokenEstimate) -> bool:
        return True

    def context_tokens_per_minute(self) -> int:
        return 1_000_000


class _Counter:
    def count_tokens(self, _messages, ignore_file: bool = False) -> int:
        return 0

    def count_file_input_tokens(self, _messages) -> int:
        return 0

    def count_file_output_tokens(self, _messages) -> int:
        return 0


Outcome = Literal["timeout", "transient", "fatal", "success"]


class _RetryClient(ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]):
    def __init__(self, outcomes: list[Outcome], max_backoffs: int):
        super().__init__(
            model="retry-test",
            model_provider=ModelProvider.OPENAI,
            model_provider_class=_Provider(),
            rate_limit_strategy=_RateLimit(),
            token_counter=_Counter(),
            max_backoffs=max_backoffs,
            initial_backoff_seconds=0,
        )
        self.outcomes = outcomes
        self.calls = 0
        self.cancelled_attempts = 0
        self.started = asyncio.Event()
        self._metrics = LMMetrics()

    async def make_single_request(
        self, _request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        self.calls += 1
        self.started.set()
        outcome = self.outcomes.pop(0) if self.outcomes else "fatal"
        if outcome == "timeout":
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled_attempts += 1
                raise
        if outcome == "transient":
            return TransientException(RuntimeError("retryable"))
        if outcome == "fatal":
            return FatalException(RuntimeError("fatal"))
        return FenicCompletionsResponse(completion="ok", logprobs=None, usage=None)

    def estimate_tokens_for_request(self, _request) -> TokenEstimate:
        return TokenEstimate(input_tokens=1, output_tokens=1)

    def get_metrics(self) -> LMMetrics:
        return self._metrics

    def reset_metrics(self):
        self._metrics = LMMetrics()

    def _get_max_output_token_request_limit(self, _request) -> int:
        return 1


def _request() -> FenicCompletionsRequest:
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="s", examples=[], user="u"),
        max_completion_tokens=1,
        top_logprobs=None,
        structured_output=None,
        temperature=0,
    )


@pytest.mark.parametrize("max_backoffs", [0, 1, 2])
def test_timeouts_have_a_request_local_send_ceiling(max_backoffs):
    client = _RetryClient(
        ["timeout"] * (max_backoffs + 1) + ["fatal"], max_backoffs
    )
    try:
        with pytest.raises(ExecutionError, match="maximum number of retries"):
            client.make_batch_requests([_request()], "timeout", request_timeout=0.01)
        assert client.calls == max_backoffs + 1
        assert client.cancelled_attempts == client.calls
    finally:
        client.shutdown()


@pytest.mark.parametrize("max_backoffs", [0, 1, 2])
def test_transient_and_timeout_attempts_share_one_ceiling(max_backoffs):
    client = _RetryClient(["transient", "timeout", "transient", "fatal"], max_backoffs)
    try:
        with pytest.raises(ExecutionError, match="maximum number of retries"):
            client.make_batch_requests([_request()], "mixed", request_timeout=0.01)
        assert client.calls == max_backoffs + 1
    finally:
        client.shutdown()


def test_last_permitted_attempt_can_succeed_after_elapsed_backoff():
    client = _RetryClient(["transient", "success"], max_backoffs=1)
    try:
        assert client.make_batch_requests([_request()], "success") == [
            FenicCompletionsResponse(completion="ok", logprobs=None, usage=None)
        ]
        assert client.calls == 2
        assert client.num_backoffs == 0
    finally:
        client.shutdown()


def test_shutdown_cancels_an_attempt_without_requeueing():
    client = _RetryClient(["timeout", "success"], max_backoffs=2)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                client.make_batch_requests, [_request()], "shutdown", request_timeout=60
            )
            asyncio.run_coroutine_threadsafe(client.started.wait(), client._event_loop).result(1)
            client.shutdown()
            with pytest.raises(asyncio.CancelledError):
                future.result(1)
        assert client.calls == 1
    finally:
        if not client.shutdown_event.is_set():
            client.shutdown()
