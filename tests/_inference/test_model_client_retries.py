"""Request-local retry limits for the shared model scheduler."""

import asyncio
import multiprocessing
import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Literal, Union

import pytest

from fenic._backends.local.async_utils import EventLoopManager
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
    ResponseUsage,
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


Outcome = Literal["timeout", "transient", "fatal", "success", "success_with_usage"]


class _RetryClient(ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]):
    def __init__(
        self,
        outcomes: list[Outcome] | dict[str, list[Outcome]],
        max_backoffs: int,
        cache=None,
        close_timeout_seconds: float = 10,
    ):
        super().__init__(
            model="retry-test",
            model_provider=ModelProvider.OPENAI,
            model_provider_class=_Provider(),
            rate_limit_strategy=_RateLimit(),
            token_counter=_Counter(),
            max_backoffs=max_backoffs,
            initial_backoff_seconds=0,
            cache=cache,
            _provider_close_timeout_seconds=close_timeout_seconds,
        )
        self.outcomes = outcomes
        self.calls = 0
        self.calls_by_payload: dict[str, int] = defaultdict(int)
        self.cancelled_attempts = 0
        self.cancelled_attempts_by_payload: dict[str, int] = defaultdict(int)
        self.reconciled_usage: list[ResponseUsage] = []
        self.started = asyncio.Event()
        self.shutdown_events: list[str] = []
        self.close_error = False
        self._metrics = LMMetrics()

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        payload = request.messages.user or ""
        self.calls += 1
        self.calls_by_payload[payload] += 1
        self.started.set()
        if isinstance(self.outcomes, dict):
            outcomes = self.outcomes[payload]
            outcome = outcomes.pop(0) if outcomes else "fatal"
        else:
            outcome = self.outcomes.pop(0) if self.outcomes else "fatal"
        if outcome == "timeout":
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled_attempts += 1
                self.cancelled_attempts_by_payload[payload] += 1
                self.shutdown_events.append("cancelled")
                raise
        if outcome == "transient":
            return TransientException(RuntimeError("retryable"))
        if outcome == "fatal":
            return FatalException(RuntimeError("fatal"))
        if outcome == "success_with_usage":
            return FenicCompletionsResponse(
                completion="ok",
                logprobs=None,
                usage=ResponseUsage(
                    prompt_tokens=2,
                    completion_tokens=3,
                    total_tokens=5,
                ),
            )
        return FenicCompletionsResponse(completion="ok", logprobs=None, usage=None)

    def estimate_tokens_for_request(self, _request) -> TokenEstimate:
        return TokenEstimate(input_tokens=1, output_tokens=1)

    def get_metrics(self) -> LMMetrics:
        return self._metrics

    def reset_metrics(self):
        self._metrics = LMMetrics()

    def _get_max_output_token_request_limit(self, _request) -> int:
        return 1

    def _reconcile_completion(self, _request, _estimated_tokens, usage):
        self.reconciled_usage.append(usage)

    async def _close_provider(self):
        self.shutdown_events.append("close")
        if self.close_error:
            raise RuntimeError("close failed")


class _CacheProbe:
    def __init__(self):
        self.writes = []

    def get_batch(self, _cache_keys):
        return {}

    def set(self, *args):
        self.writes.append(args)


def _run_hanging_close_shutdown(close_timeout_seconds, entered, released):
    client = _RetryClient(
        ["success"], max_backoffs=1, close_timeout_seconds=close_timeout_seconds
    )
    original_release = EventLoopManager.release_loop

    async def hanging_close():
        entered.set()
        await asyncio.Event().wait()

    def release_loop(manager):
        released.set()
        original_release(manager)

    client._close_provider = hanging_close
    EventLoopManager.release_loop = release_loop
    client.shutdown()


def _request(payload="u") -> FenicCompletionsRequest:
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="s", examples=[], user=payload),
        max_completion_tokens=1,
        top_logprobs=None,
        structured_output=None,
        temperature=0,
    )


def _queue_item(request, future=None, attempts_started=0):
    return QueueItem(
        thread_id=0,
        request=request,
        future=future or Future(),
        estimated_tokens=TokenEstimate(input_tokens=1, output_tokens=1),
        batch_id="test",
        request_timeout=0.01,
        attempts_started=attempts_started,
    )


def _run_on_client_loop(client, coroutine):
    return asyncio.run_coroutine_threadsafe(coroutine, client._event_loop).result(1)


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


def test_neighbors_cannot_extend_each_others_request_local_allowance():
    client = _RetryClient(
        {
            "timeout": ["timeout", "timeout", "fatal"],
            "transient": ["transient", "transient", "fatal"],
        },
        max_backoffs=1,
    )
    try:
        futures, _, _ = client._submit_batch_requests(
            [_request("timeout"), _request("transient")],
            "neighbors",
            request_timeout=0.01,
        )
        for future in futures:
            exception = future.exception(timeout=1)
            assert exception is not None
            assert "maximum number of retries" in str(exception)
        assert client.calls_by_payload == {"timeout": 2, "transient": 2}
        assert client.cancelled_attempts_by_payload["timeout"] == 2
    finally:
        client.shutdown()


def test_terminal_retry_failure_propagates_to_deduplicated_waiters_without_caching():
    cache = _CacheProbe()
    client = _RetryClient(["transient", "fatal"], max_backoffs=1, cache=cache)
    try:
        with pytest.raises(ExecutionError, match="fatal"):
            client.make_batch_requests([_request("duplicate"), _request("duplicate")], "dedup")
        assert client.calls_by_payload == {"duplicate": 2}
        assert cache.writes == []
    finally:
        client.shutdown()


def test_success_on_last_attempt_settles_only_returned_usage():
    cache = _CacheProbe()
    client = _RetryClient(["transient", "success_with_usage"], max_backoffs=1, cache=cache)
    try:
        response = client.make_batch_requests([_request("usage")], "usage")[0]
        assert response.usage == ResponseUsage(
            prompt_tokens=2, completion_tokens=3, total_tokens=5
        )
        assert client.calls_by_payload == {"usage": 2}
        assert client.reconciled_usage == [response.usage]
        assert len(cache.writes) == 1
    finally:
        client.shutdown()


def test_cancelled_and_terminal_items_never_dispatch_or_requeue():
    client = _RetryClient(["success"], max_backoffs=1)
    try:
        cancelled = Future()
        cancelled.cancel()
        terminal = Future()
        terminal.set_result("already complete")
        _run_on_client_loop(client, client._process_single_request(_queue_item(_request(), cancelled)))
        _run_on_client_loop(client, client._process_single_request(_queue_item(_request(), terminal)))
        _run_on_client_loop(
            client,
            client._handle_response(
                _queue_item(_request(), cancelled, attempts_started=1),
                TransientException(RuntimeError("retryable")),
            ),
        )
        assert client.calls == 0
        assert client.retry_queue.empty()
        assert cancelled.done()
        assert terminal.done()
    finally:
        client.shutdown()


def test_shutdown_before_dispatch_fails_the_pending_future_without_requeueing():
    client = _RetryClient(["success"], max_backoffs=1)
    queue_item = _queue_item(_request("shutdown"))

    async def process_after_shutdown():
        client.shutdown_event.set()
        await client._process_single_request(queue_item)

    try:
        _run_on_client_loop(client, process_after_shutdown())
        assert queue_item.future.done()
        exception = queue_item.future.exception(timeout=1)
        assert exception is not None
        assert "shut down" in str(exception)
        assert client.calls == 0
        assert client.retry_queue.empty()
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


def test_shutdown_closes_provider_after_cancellation_and_before_loop_release(
    monkeypatch,
):
    client = _RetryClient(["timeout", "success"], max_backoffs=2)
    original_release = EventLoopManager.release_loop

    def release_loop(manager):
        client.shutdown_events.append("release")
        original_release(manager)

    monkeypatch.setattr(EventLoopManager, "release_loop", release_loop)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                client.make_batch_requests,
                [_request()],
                "shutdown-order",
                request_timeout=60,
            )
            asyncio.run_coroutine_threadsafe(
                client.started.wait(), client._event_loop
            ).result(1)
            client.shutdown()
            with pytest.raises(asyncio.CancelledError):
                future.result(1)
        assert client.calls == 1
        assert client.shutdown_events == ["cancelled", "close", "release"]
    finally:
        if not client.shutdown_event.is_set():
            client.shutdown()


def test_provider_close_failure_still_releases_the_shared_loop(monkeypatch, caplog):
    client = _RetryClient(["success"], max_backoffs=1)
    client.close_error = True
    original_release = EventLoopManager.release_loop

    def release_loop(manager):
        client.shutdown_events.append("release")
        original_release(manager)

    monkeypatch.setattr(EventLoopManager, "release_loop", release_loop)
    try:
        client.shutdown()
        assert client.shutdown_events == ["close", "release"]
        assert (
            "Could not close provider resources for model retry-test during shutdown "
            "(RuntimeError)"
            in caplog.text
        )
        assert "close failed" not in caplog.text
    finally:
        if not client.shutdown_event.is_set():
            client.shutdown()


def test_provider_close_timeout_releases_loop_and_cancels_close_task(
    monkeypatch, caplog
):
    client = _RetryClient(
        ["success"], max_backoffs=1, close_timeout_seconds=0.01
    )
    entered_close = threading.Event()
    cancelled_close = threading.Event()
    original_release = EventLoopManager.release_loop

    async def hanging_close():
        entered_close.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_close.set()
            raise

    def release_loop(manager):
        client.shutdown_events.append("release")
        original_release(manager)

    monkeypatch.setattr(client, "_close_provider", hanging_close)
    monkeypatch.setattr(EventLoopManager, "release_loop", release_loop)
    try:
        assert client._provider_close_timeout_seconds == 0.01
        client.shutdown()
        assert entered_close.is_set()
        assert cancelled_close.is_set()
        assert client.calls == 0
        assert client.retry_queue.empty()
        assert client.shutdown_events == ["release"]
        assert (
            "Could not close provider resources for model retry-test during shutdown "
            "(TimeoutError)"
            in caplog.text
        )
        assert "hanging_close" not in caplog.text
    finally:
        if not client.shutdown_event.is_set():
            client.shutdown()


def test_provider_close_timeout_defaults_to_ten_seconds():
    client = _RetryClient(["success"], max_backoffs=1)
    try:
        assert client._provider_close_timeout_seconds == 10
    finally:
        client.shutdown()


def test_hanging_close_process_exits_after_short_deadline():
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    released = context.Event()
    process = context.Process(
        target=_run_hanging_close_shutdown,
        args=(0.01, entered, released),
    )
    process.start()
    try:
        assert entered.wait(5), "provider close did not start"
        process.join(1)
        assert process.exitcode == 0
        assert released.is_set()
    finally:
        if process.is_alive():
            process.kill()
            process.join()


def test_provider_close_timeout_cancels_task_while_shared_loop_stays_owned():
    client = _RetryClient(
        ["success"], max_backoffs=1, close_timeout_seconds=0.01
    )
    peer = _RetryClient(["success"], max_backoffs=1)
    cancelled_close = threading.Event()

    async def hanging_close():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_close.set()
            raise

    client._close_provider = hanging_close
    try:
        client.shutdown()
        assert cancelled_close.wait(1)
        assert EventLoopManager().loop is peer._event_loop
        assert peer._event_loop.is_running()
    finally:
        if not client.shutdown_event.is_set():
            client.shutdown()
        if not peer.shutdown_event.is_set():
            peer.shutdown()
