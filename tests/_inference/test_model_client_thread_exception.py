"""Batch isolation for exceptions surfaced from the event loop.

`_register_thread_exception` records a failure against the submitting batch so a long
batch fails fast instead of waiting on every future. That record must not outlive
the batch that produced it: a later batch on the same thread would otherwise
re-raise a stale exception belonging to an earlier, unrelated operation.
"""

import asyncio
import threading
from typing import List, Optional, Union

import pytest

from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    QueueItem,
    TransientException,
)
from fenic._inference.rate_limit_strategy import RateLimitStrategy, TokenEstimate
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
)
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core._inference.model_provider import ModelProviderClass
from fenic.core.error import ExecutionError
from fenic.core.metrics import LMMetrics


class DummyProvider(ModelProviderClass):
    @property
    def name(self) -> str:
        return "dummy"

    def create_client(self):
        return object()

    def create_aio_client(self):
        return object()

    async def validate_api_key(self) -> None:
        return


class DummyRateLimitStrategy(RateLimitStrategy):
    def __init__(self):
        super().__init__(rpm=100)

    def backoff(self, curr_time: float) -> int:
        return 0

    def check_and_consume_rate_limit(self, token_estimate: TokenEstimate) -> bool:
        return True

    def context_tokens_per_minute(self) -> int:
        return 60_000


class DummyTokenCounter:
    def count_tokens(self, messages, ignore_file: bool = False) -> int:
        return 0

    def count_file_input_tokens(self, messages) -> int:
        return 0

    def count_file_output_tokens(self, messages) -> int:
        return 0


class FlakyCompletionClient(
    ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    """Raises an unhandled error on the first request, then succeeds."""

    def __init__(self, failures: int = 1):
        super().__init__(
            model="dummy-completion",
            model_provider=ModelProvider.OPENAI,
            model_provider_class=DummyProvider(),
            rate_limit_strategy=DummyRateLimitStrategy(),
            token_counter=DummyTokenCounter(),
        )
        self._metrics = LMMetrics()
        self.remaining_failures = failures
        self.call_count = 0

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        self.call_count += 1
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            # Not a Transient/Fatal wrapper: an unhandled error, the shape a
            # response that fails structured-output validation takes.
            raise ValueError("first batch failure")
        return FenicCompletionsResponse(
            completion=f"response-for-{request.messages.user}",
            logprobs=None,
            usage=None,
        )

    def estimate_tokens_for_request(
        self, request: FenicCompletionsRequest
    ) -> TokenEstimate:
        return TokenEstimate(input_tokens=1, output_tokens=1)

    def get_metrics(self) -> LMMetrics:
        return self._metrics

    def reset_metrics(self):
        self._metrics = LMMetrics()

    def _get_max_output_token_request_limit(
        self, request: FenicCompletionsRequest
    ) -> int:
        return 8

    def _get_max_output_tokens_estimate(self, request: FenicCompletionsRequest) -> int:
        return 8


def _request(user: str) -> FenicCompletionsRequest:
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="s", examples=[], user=user),
        max_completion_tokens=8,
        top_logprobs=None,
        structured_output=None,
        temperature=None,
    )


def _batch(
    client: FlakyCompletionClient, user: str
) -> List[Optional[FenicCompletionsResponse]]:
    return client.make_batch_requests([_request(user)], operation_name="test")


def test_failed_batch_does_not_fail_the_next_batch():
    client = FlakyCompletionClient(failures=1)
    try:
        with pytest.raises(ExecutionError, match="first batch failure"):
            _batch(client, "first")
        assert client.thread_exceptions == {}
        assert client.active_batches == set()

        # The endpoint is healthy again, so this batch must report its own outcome.
        responses = _batch(client, "second")
        assert responses[0].completion == "response-for-second"

        responses = _batch(client, "third")
        assert responses[0].completion == "response-for-third"
    finally:
        client.shutdown()


def test_thread_exception_is_delivered_only_once():
    client = FlakyCompletionClient(failures=0)
    try:
        batch_id = "failed-batch"
        client.thread_exceptions[(threading.get_ident(), batch_id)] = ValueError(
            "stale"
        )

        with pytest.raises(ValueError, match="stale"):
            client._maybe_raise_thread_exception(batch_id)

        # Delivered once: a second call is a no-op rather than a repeat raise.
        client._maybe_raise_thread_exception(batch_id)
        assert client.thread_exceptions == {}
    finally:
        client.shutdown()


@pytest.mark.parametrize("same_thread", [True, False])
def test_thread_exception_is_scoped_to_its_thread_and_batch(same_thread):
    client = FlakyCompletionClient(failures=0)
    try:
        current_thread = threading.get_ident()
        foreign_key = (
            current_thread if same_thread else current_thread + 1,
            "old-batch" if same_thread else "new-batch",
        )
        foreign_exception = ValueError("foreign failure")
        client.thread_exceptions[foreign_key] = foreign_exception

        client._maybe_raise_thread_exception("new-batch")
        assert client.thread_exceptions == {foreign_key: foreign_exception}
    finally:
        client.shutdown()


class LateSiblingCompletionClient(FlakyCompletionClient):
    """Lets an old sibling fail only after the next batch starts."""

    def __init__(self):
        super().__init__(failures=0)
        self.old_sibling_started = asyncio.Event()
        self.new_batch_started = asyncio.Event()
        self.late_failure_registered = asyncio.Event()
        self.caller_threads = []

    async def make_single_request(self, request: FenicCompletionsRequest):
        if request.messages.user == "old-fast":
            await asyncio.wait_for(self.old_sibling_started.wait(), timeout=5)
            raise ValueError("first batch failure")
        if request.messages.user == "old-slow":
            self.old_sibling_started.set()
            await asyncio.wait_for(self.new_batch_started.wait(), timeout=5)
            raise ValueError("late sibling failure")
        if request.messages.user == "new-first":
            self.new_batch_started.set()
        return await super().make_single_request(request)

    async def _enqueue_request(self, queue_item: QueueItem):
        await super()._enqueue_request(queue_item)
        if queue_item.request.messages.user == "new-first":
            # Hold the caller before its next submission-time exception check.
            await asyncio.wait_for(self.late_failure_registered.wait(), timeout=5)

    def _register_thread_exception(self, queue_item: QueueItem, exception: Exception):
        super()._register_thread_exception(queue_item, exception)
        if queue_item.request.messages.user == "old-slow":
            self.late_failure_registered.set()

    def _submit_batch_requests(
        self, requests, batch_id, operation_name, request_timeout
    ):
        self.caller_threads.append(threading.get_ident())
        return super()._submit_batch_requests(
            requests, batch_id, operation_name, request_timeout
        )


def test_late_sibling_failure_does_not_fail_the_next_batch(caplog):
    client = LateSiblingCompletionClient()
    try:
        with pytest.raises(ExecutionError, match="first batch failure"):
            client.make_batch_requests(
                [_request("old-fast"), _request("old-slow")],
                operation_name="old-batch",
                request_timeout=5,
            )

        responses = client.make_batch_requests(
            [_request("new-first"), _request("new-second")],
            operation_name="new-batch",
            request_timeout=5,
        )

        assert client.old_sibling_started.is_set()
        assert client.new_batch_started.is_set()
        assert client.late_failure_registered.is_set()
        assert client.caller_threads == [threading.get_ident()] * 2
        assert [response.completion for response in responses] == [
            "response-for-new-first",
            "response-for-new-second",
        ]
        assert client.thread_exceptions == {}
        assert client.active_batches == set()
    finally:
        client.shutdown()
    assert not any(
        "Task exception was never retrieved" in record.getMessage()
        for record in caplog.records
    )
