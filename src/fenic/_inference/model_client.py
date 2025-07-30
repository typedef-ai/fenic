import asyncio
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from pydantic import BaseModel
from tqdm import tqdm

from fenic._constants import MILLISECOND_IN_SECONDS, MINUTE_IN_SECONDS
from fenic._inference.rate_limit_strategy import (
    RateLimitStrategy,
    TokenEstimate,
)
from fenic._inference.token_counter import TiktokenTokenCounter, Tokenizable
from fenic._inference.types import LMRequestMessages
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core.metrics import LMMetrics

# Type variables
RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")
# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ResponseUsage:
    """Token usage information from API response."""
    prompt_tokens: int
    completion_tokens: int  # Actual completion tokens (non-thinking)
    total_tokens: int
    cached_tokens: int = 0
    thinking_tokens: int = 0  # Separate thinking token count


@dataclass
class FenicCompletionsResponse:
    completion: str
    logprobs: Optional[List[ChatCompletionTokenLogprob]]
    usage: Optional[ResponseUsage] = None


@dataclass
class FenicCompletionsRequest:
    messages: LMRequestMessages
    max_completion_tokens: int
    top_logprobs: Optional[int]
    structured_output: Optional[type[BaseModel]]
    temperature: float
    model_profile: Optional[str] = None


# Exception classes
@dataclass
class TransientException:
    """Represents an exception that might be resolved with a retry."""

    exception: Exception


@dataclass
class FatalException:
    """Represents an exception that is unlikely to be resolved with retries."""

    exception: Exception

@dataclass
class QueueItem(Generic[RequestT]):
    """Represents an item in the request queue."""

    thread_id: int
    request: RequestT
    future: Future
    estimated_tokens: TokenEstimate
    batch_id: str


class ModelClient(Generic[RequestT, ResponseT], ABC):
    """Base client for interacting with language models.

    This abstract base class provides a robust framework for interacting with language models,
    handling rate limiting, request queuing, retries, and deduplication. It manages concurrent
    requests efficiently using an asynchronous event loop and implements token-based rate limiting.

    Type Parameters:
        RequestT: The type of request objects this client handles
        ResponseT: The type of response objects this client returns

    Attributes:
        model (str): The name or identifier of the model
        model_provider (ModelProvider): The provider of the model (e.g., OPENAI, ANTHROPIC)
        rate_limit_strategy (RateLimitStrategy): Strategy for rate limiting requests
        token_counter (TiktokenTokenCounter): Counter for estimating token usage
    """

    def __init__(
            self,
            model: str,
            model_provider: ModelProvider,
            rate_limit_strategy: RateLimitStrategy,
            token_counter: TiktokenTokenCounter,
            queue_size: int = 100,
            initial_backoff_seconds: float = 1,
            backoff_factor: float = 2,
            max_backoffs: int = 10,
    ):
        """Initialize the ModelClient with configuration for model interaction.

        Args:
            model: The name or identifier of the model
            model_provider: The model provider (OPENAI, ANTHROPIC)
            alias: The Model Client's alias, for logging purposes
            rate_limit_strategy: Strategy for rate limiting requests
            token_counter: Implementation for predicting input token counts
            queue_size: Maximum size of the request queue (default: 100)
            initial_backoff_seconds: Initial delay for exponential backoff (default: 1)
            backoff_factor: Factor by which backoff time increases (default: 2)
            max_backoffs: Maximum number of retry attempts (default: 10)
        """
        self.model = model
        self.model_provider = model_provider
        self.rate_limit_strategy = rate_limit_strategy
        self.context_tokens_per_minute = rate_limit_strategy.context_tokens_per_minute()
        self.token_counter = token_counter
        # Async queues
        self.request_queue = asyncio.Queue(maxsize=queue_size)
        self.retry_queue = asyncio.Queue()  # No size limit to avoid deadlocking
        self.pending_requests: List[QueueItem[RequestT]] = (
            []
        )  # requests waiting to be processed
        self.inflight_requests: Set[asyncio.Task] = set()
        self.shutdown_event = asyncio.Event()

        # Backoff handling
        self.initial_backoff_seconds: float = initial_backoff_seconds
        self.backoff_factor: float = backoff_factor
        self.max_backoffs: int = max_backoffs
        self.last_transient_exception_time: float = 0
        self.num_backoffs: int = 0


        # Thread-specific exception tracking
        self.thread_exceptions: Dict[int, Exception] = {}
        self.thread_exceptions_lock = threading.Lock()

        # Register with the event loop manager
        ModelClientManager().register_client(self)

        logger.info(
            f"Initialized client for model {model} with rate limit strategy {self.rate_limit_strategy}"
        )

    @abstractmethod
    async def make_single_request(
            self, request: RequestT
    ) -> Union[None, ResponseT, TransientException, FatalException]:
        """Make a single API call to the language model.

        This method must be implemented by subclasses to handle the actual API communication
        with the language model provider.

        Args:
            request: The request data to send to the model

        Returns:
            Union[None, ResponseT, TransientException, FatalException]: The API response,
            None if the request was empty, or an exception wrapper indicating either a
            transient error (can be retried) or a fatal error (should not be retried)
        """
        pass

    @abstractmethod
    def estimate_tokens_for_request(self, request: RequestT) -> TokenEstimate:
        """Estimate the token usage for a given request.

        This method must be implemented by subclasses to accurately predict token usage
        for both input and output tokens.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate: Object containing estimated input and output tokens
        """
        pass

    def count_tokens(self, messages: Tokenizable) -> int:
        """Count the number of tokens in a tokenizable object.

        Args:
            messages: The tokenizable object to count tokens for

        Returns:
            int: The number of tokens in the object
        """
        return self.token_counter.count_tokens(messages)

    @abstractmethod
    def get_request_key(self, request: RequestT) -> Any:
        """Generate a unique key for request deduplication.

        This method must be implemented by subclasses to provide a hashable key that
        uniquely identifies a request for deduplication purposes.

        Args:
            request: The request to generate a key for

        Returns:
            Any: A hashable value that uniquely identifies this request
        """
        pass

    @abstractmethod
    def get_metrics(self) -> LMMetrics:
        """Get the current metrics for this model client.

        Returns:
            LMMetrics: The current metrics for this client
        """
        pass

    @abstractmethod
    def reset_metrics(self):
        """Reset all metrics for this model client to their initial values."""
        pass

    def _count_auxiliary_input_tokens(self, request: RequestT) -> int:
        """Count extra input tokens for structured output, tools, etc. Override as needed."""
        if hasattr(request, 'structured_output') and request.structured_output:
            return self._estimate_structured_output_overhead(request.structured_output)
        return 0

    def _estimate_structured_output_overhead(self, response_format: type[BaseModel]) -> int:
        """Default structured output token estimation. Override for provider-specific logic."""

        schema_str = json.dumps(response_format.model_json_schema(), separators=(',', ':'))
        return self.count_tokens(schema_str)


    @abstractmethod
    def _get_max_output_tokens(self, request: RequestT) -> int:
        """Get conservative output token estimate. Override in subclasses for provider-specific logic."""
        pass

    #
    # Public methods (called from user threads)
    #
    def shutdown(self):
        """Shut down the model client and clean up resources.

        This method:
        1. Cancels all pending and in-flight requests
        2. Unregisters the client from the ModelClientManager
        3. Cleans up all associated resources
        4. Ensures all threads are properly notified of the shutdown
        """
        exception = Exception(f"Model client for {self.model} has been shut down")

        ModelClientManager().event_loop.call_soon_threadsafe(self.shutdown_event.set)

        if self.pending_requests:
            for queue_item in self.pending_requests:
                self._register_thread_exception(queue_item, exception)
            self.pending_requests = []

        while not self.request_queue.empty():
            try:
                queue_item = self.request_queue.get_nowait()
                self._register_thread_exception(queue_item, exception)
            except asyncio.QueueEmpty:
                break

        while not self.retry_queue.empty():
            try:
                queue_item = self.retry_queue.get_nowait()
                self._register_thread_exception(queue_item, exception)
            except asyncio.QueueEmpty:
                break

        cancel_future = asyncio.run_coroutine_threadsafe(
            self._cancel_in_flight_requests(), ModelClientManager().event_loop
        )
        cancel_future.result()

        ModelClientManager().unregister_client(self)

    def make_batch_requests(
            self,
            requests: List[Optional[RequestT]],
            operation_name: str,
    ) -> List[ResponseT]:
        """Submit and process a batch of requests asynchronously.

        This method handles the submission and processing of multiple requests in parallel,
        with automatic deduplication and rate limiting. It provides progress tracking
        and handles empty requests appropriately.

        Args:
            requests: List of requests to process. None entries are handled as empty responses
            operation_name: Name for logging purposes to identify the operation

        Returns:
            List[ResponseT]: List of responses in the same order as the input requests
        """
        batch_id = str(uuid.uuid4())
        logger.info(
            f"Creating batch {batch_id} with {len(requests)} requests for {operation_name} using (model: {self.model})"
        )
        return self._make_batch_requests(requests, operation_name, batch_id)

    #
    # Producer methods (run on the user thread)
    #
    def _get_or_create_request_future(
            self, unique_futures: Dict[Any, Future], request: RequestT
    ) -> tuple[Future, TokenEstimate | None]:
        """Retrieves an existing future for a duplicate request or creates a new one.

        Args:
            unique_futures: A dictionary mapping request keys to their futures.
            request: The current request being processed.

        Returns:
            A tuple of the future for the request and the estimated number of tokens (0 for duplicates).
        """
        key = self.get_request_key(request)

        # Return existing future for duplicate requests
        if key in unique_futures:
            new_future = Future()
            existing_future = unique_futures[key]

            # Copy result from original future to new future
            def _copy_future_result(input_future: Future, output_future: Future):
                if input_future.cancelled():
                    output_future.cancel()
                elif input_future.exception() is not None:
                    output_future.set_exception(input_future.exception())
                else:
                    output_future.set_result(input_future.result())

            # If original future already done, copy result immediately
            if existing_future.done():
                _copy_future_result(existing_future, new_future)
            else:
                # Otherwise add callback to copy result when ready
                existing_future.add_done_callback(
                    lambda input_future, output_future=new_future: _copy_future_result(
                        input_future, output_future
                    )
                )

            return new_future, None  # No tokens for duplicate requests

        # If it's a new request, create a future and estimate its token cost
        new_future = Future()
        unique_futures[key] = new_future
        token_estimate = self.estimate_tokens_for_request(request)
        return new_future, token_estimate

    def _maybe_raise_thread_exception(self):
        """Surface exceptions from event loop to calling thread immediately."""
        current_thread_id = threading.get_ident()
        with self.thread_exceptions_lock:
            if current_thread_id in self.thread_exceptions:
                raise self.thread_exceptions[current_thread_id]

    def _calculate_backoff_time(self, backoff_iteration: int) -> float:
        """Calculates the backoff duration using exponential backoff with a maximum limit.

        Args:
            backoff_iteration: The current backoff iteration.

        Returns:
            The backoff time in seconds.
        """
        backoff = self.initial_backoff_seconds * (
                self.backoff_factor ** backoff_iteration
        )
        return min(backoff, MINUTE_IN_SECONDS)

    def _check_and_consume_rate_limit(self, token_amount: TokenEstimate) -> bool:
        """Checks if there is enough capacity in both the token and request rate limit buckets,
        and consumes the capacity if so.

        Args:
            token_amount: A TokenEstimate object containing the estimated input, output, and total tokens.

        Returns:
            True if there was enough capacity and it was consumed, False otherwise.
        """
        return self.rate_limit_strategy.check_and_consume_rate_limit(token_amount)

    async def _enqueue_request(self, queue_item: QueueItem[RequestT]):
        """Enqueue a request to be processed.

        Args:
            queue_item: The queue item to enqueue.
        """
        await self.request_queue.put(queue_item)



    def _make_batch_requests(self,
                             requests: List[Optional[RequestT]],
                             operation_name: str,
                             batch_id: Optional[str] = None) -> List[ResponseT]:
        """Standard batch processing without sampling (used by both sampling and non-sampling flows)."""
        if batch_id is None:
            batch_id = str(uuid.uuid4())

        logger.info(
            f"Processing batch {batch_id} with {len(requests)} requests for {operation_name} using (model: {self.model})"
        )

        # Submit requests and get futures
        request_futures, num_unique_requests, total_token_estimate = self._submit_batch_requests(
            requests, batch_id
        )

        logger.info(
            f"Batch {batch_id}: Submitted {num_unique_requests} unique requests with {total_token_estimate}"
        )

        # Wait for responses
        responses = self._collect_batch_responses(request_futures, batch_id)

        logger.info(
            f"Batch {batch_id}: Completed with {len(responses)} responses from {self.model}"
        )

        return responses

    def _submit_batch_requests(self,
                             requests: List[Optional[RequestT]],
                             batch_id: str) -> tuple[List[Future], int, TokenEstimate]:
        """Submit all requests in a batch and return futures, unique request count, and token estimate.

        Args:
            requests: List of requests to submit
            batch_id: Batch identifier for tracking

        Returns:
            Tuple of (request_futures, num_unique_requests, total_token_estimate)
        """
        request_futures: List[Future] = []
        current_thread_id = threading.get_ident()
        unique_futures: Dict[Any, Future] = {}
        num_unique_requests = 0
        total_token_estimate = TokenEstimate()

        # Submit all requests with progress indicator
        with tqdm(
                total=len(requests),
                desc=f"Submitting requests for batch: {batch_id} (model: {self.model})",
                unit="req",
        ) as pbar:
            for request in requests:
                # Check for exceptions from the event loop thread
                self._maybe_raise_thread_exception()

                # Eagerly handle empty requests
                if request is None:
                    req_future = Future()
                    request_futures.append(req_future)
                    req_future.set_result(None)
                    pbar.update(1)
                    pbar.set_postfix(
                        estimated_input_tokens=total_token_estimate.input_tokens,
                        estimated_output_tokens=total_token_estimate.output_tokens,
                    )
                    continue

                req_future, estimated_tokens = self._get_or_create_request_future(
                    unique_futures, request
                )
                request_futures.append(req_future)

                # Only enqueue if this is a new, unique request
                if estimated_tokens is not None:
                    num_unique_requests += 1
                    total_token_estimate += estimated_tokens
                    queue_item = QueueItem(
                        thread_id=current_thread_id,
                        request=request,
                        future=req_future,
                        estimated_tokens=estimated_tokens,
                        batch_id=batch_id,
                    )
                    enqueue_future: Future = asyncio.run_coroutine_threadsafe(
                        self._enqueue_request(queue_item),
                        ModelClientManager().event_loop,
                    )
                    enqueue_future.result()

                pbar.update(1)
                pbar.set_postfix(
                    estimated_input_tokens=total_token_estimate.input_tokens,
                    estimated_output_tokens=total_token_estimate.output_tokens,
                )

        return request_futures, num_unique_requests, total_token_estimate

    def _collect_batch_responses(self, request_futures: List[Future], batch_id: str) -> List[ResponseT]:
        """Collect responses from all request futures with progress tracking.

        Args:
            request_futures: List of futures to wait for
            batch_id: Batch identifier for logging

        Returns:
            List of responses in same order as input futures
        """
        responses = []
        with tqdm(
                total=len(request_futures),
                desc=f"Awaiting responses for batch {batch_id} (model: {self.model})",
                unit="res",
        ) as pbar:
            for req_future in request_futures:
                responses.append(req_future.result())
                pbar.update(1)

        return responses

    #
    # Consumer methods (run on the shared asyncio event loop)
    #
    async def _process_queue(self):
        """Continuously processes requests from the request and retry queues. This method runs on the shared asyncio event loop."""
        try:
            while True:
                # Prioritize the retry queue if it has items, otherwise get from the main request queue
                if not self.pending_requests:
                    queue_items = await self._get_queued_requests()
                    if not queue_items:
                        logger.debug(f"Worker for model {self.model} shutting down")
                        return
                    self.pending_requests = queue_items

                # Iterate through pending requests and process those with available capacity
                for queue_item in self.pending_requests:
                    if self._check_and_consume_rate_limit(queue_item.estimated_tokens):
                        task = asyncio.create_task(
                            self._process_single_request(queue_item)
                        )
                        self._track_inflight_task(task)
                        self.pending_requests.remove(queue_item)
                    else:
                        # Sleep for a short duration to wait for rate limit to refill to avoid busy-waiting
                        await asyncio.sleep(MILLISECOND_IN_SECONDS)

                    await self._maybe_backoff()
        except asyncio.CancelledError:
            logger.debug(f"Worker for model {self.model} was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in worker for model {self.model}: {e}", exc_info=True)

    async def _process_single_request(self, queue_item: QueueItem[RequestT]):
        """Process a single request from the queues.

        Args:
            queue_item: The queue item to process.
        """
        try:
            try:
                maybe_response = await asyncio.wait_for(
                    self.make_single_request(queue_item.request),
                    timeout=120.0,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Request for model {self.model} in batch {queue_item.batch_id} timed out. Retrying.")
                await self.retry_queue.put(queue_item)
                return

            await self._handle_response(queue_item, maybe_response)
        except asyncio.CancelledError:
            logger.debug(f"Request {queue_item.request} was cancelled")
            self._register_thread_exception(queue_item, asyncio.CancelledError)
            raise
        except Exception as e:
            self._register_thread_exception(queue_item, e)
            raise

    async def _handle_response(
            self,
            queue_item: QueueItem[RequestT],
            maybe_response: Union[None, ResponseT, TransientException, FatalException],
    ):
        """Handle the response from a request, including retrying if necessary.

        Args:
            queue_item: The queue item associated with the request.
            maybe_response: The response or exception from the request.
        """
        if isinstance(maybe_response, TransientException):
            if self.num_backoffs >= self.max_backoffs:
                self._register_thread_exception(
                    queue_item,
                    Exception(
                        f"Exceeded maximum number of retries for model {self.model}. If you're sharing quota with other users, reduce your TPM/RPM for this client.",
                        maybe_response.exception,
                    ),
                )
            else:
                await self.retry_queue.put(queue_item)
                current_time = time.time()
                self.last_transient_exception_time = current_time
        elif isinstance(maybe_response, FatalException):
            logger.error(
                f"Fatal error encountered for model {self.model}: {maybe_response.exception}. Request failed."
            )
            self._register_thread_exception(queue_item, maybe_response.exception)
        else:
            if not queue_item.future.done():
                queue_item.future.set_result(maybe_response)

    async def _maybe_backoff(self):
        """Manages the backoff period after encountering a transient exception."""
        if self.last_transient_exception_time <= 0:
            return

        now = time.time()
        backoff_time = self._calculate_backoff_time(self.num_backoffs)
        time_since_last_transient_exception = now - self.last_transient_exception_time

        if time_since_last_transient_exception < backoff_time:
            logger.warning(
                f"Backing off model {self.model} for {backoff_time - time_since_last_transient_exception:.2f} seconds before retrying requests due to rate limits."
            )
            await asyncio.sleep(backoff_time - time_since_last_transient_exception)
            self.num_backoffs += 1
            self.last_transient_exception_time = 0
            self.rate_limit_strategy.backoff(time.time())


    async def _get_queued_requests(self) -> List[QueueItem[RequestT]]:
        """Asynchronously retrieves items from the retry queue or the request queue,
        prioritizing the retry queue. Returns None if a shutdown is signaled.

        Returns:
            A list of queue items, or None if a shutdown is signaled.
        """
        get_request_task = asyncio.create_task(self.request_queue.get())
        get_retry_task = asyncio.create_task(self.retry_queue.get())
        shutdown_task = asyncio.create_task(self.shutdown_event.wait())

        done, pending = await asyncio.wait(
            [get_request_task, get_retry_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        if shutdown_task in done:
            return []

        queue_items: List[QueueItem[RequestT]] = []
        if get_retry_task in done:
            queue_items.append(get_retry_task.result())
        if get_request_task in done:
            queue_items.append(get_request_task.result())

        return queue_items

    def _track_inflight_task(self, task: asyncio.Task):
        """Adds a task to the set of inflight requests and removes it upon completion.

        Args:
            task: The task to track.
        """
        self.inflight_requests.add(task)
        task.add_done_callback(lambda _: self.inflight_requests.discard(task))

    def _register_thread_exception(
            self, queue_item: QueueItem[RequestT], exception: Exception
    ):
        """Registers an exception that occurred on the event loop to be raised in the originating thread.

        Args:
            queue_item: The queue item associated with the exception.
            exception: The exception that occurred.
        """
        if not queue_item.future.done():
            queue_item.future.set_exception(exception)

        with self.thread_exceptions_lock:
            self.thread_exceptions[queue_item.thread_id] = exception

    async def _cancel_in_flight_requests(self):
        """Cancels all inflight tasks and gathers their results."""
        for task in self.inflight_requests:
            task.cancel()
        await asyncio.gather(*self.inflight_requests, return_exceptions=True)


class ModelClientManager:
    """Manages a shared asyncio event loop for multiple ModelClient instances.
    Ensures that all clients run on the same loop for efficient resource management.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelClientManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Initializes the ModelClientManager, creating the event loop if it doesn't exist."""
        self.registered_clients: Set[ModelClient] = set()
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.background_thread: Optional[threading.Thread] = None
        self._maybe_create_event_loop()

    def register_client(self, client: ModelClient):
        """Registers a ModelClient with the manager and starts its processing task on the shared event loop.

        Args:
            client: The client to register.
        """
        with self._lock:
            self._maybe_create_event_loop()
            self.registered_clients.add(client)
            asyncio.run_coroutine_threadsafe(client._process_queue(), self.event_loop)

    def unregister_client(self, client: ModelClient):
        """Unregisters a ModelClient from the manager and shuts down the event loop if no clients remain.

        Args:
            client: The client to unregister.
        """
        loop_to_shutdown = None
        thread_to_join = None
        with self._lock:
            self.registered_clients.discard(client)
            if (
                    not self.registered_clients
                    and self.event_loop
                    and self.event_loop.is_running()
            ):
                loop_to_shutdown = self.event_loop
                thread_to_join = self.background_thread
                self.event_loop = None
                self.background_thread = None

        if loop_to_shutdown:
            cancel_future = asyncio.run_coroutine_threadsafe(
                _cancel_event_loop_tasks(loop_to_shutdown), loop_to_shutdown
            )
            cancel_future.result()
            loop_to_shutdown.call_soon_threadsafe(loop_to_shutdown.stop)
            if thread_to_join and thread_to_join.is_alive():
                thread_to_join.join()
            loop_to_shutdown.close()

    def _maybe_create_event_loop(self):
        """Creates and starts a dedicated event loop in a background thread if one doesn't exist."""
        if self.event_loop is None or self.event_loop.is_closed():
            self.event_loop = asyncio.new_event_loop()
            self.background_thread = threading.Thread(
                target=self.event_loop.run_forever, daemon=True
            )
            self.background_thread.start()


async def _cancel_event_loop_tasks(loop: asyncio.AbstractEventLoop):
    """Cancels all pending tasks in the given asyncio event loop, except the current task.

    Args:
        loop: The event loop to cancel tasks for.
    """
    asyncio.set_event_loop(loop)
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task(loop)]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
