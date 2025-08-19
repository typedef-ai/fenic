import asyncio
import logging
import random
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Iterable

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_MULTIPLIER = 10
HARD_BUFFER_LIMIT = 50_000
DEFAULT_PENDING_MULTIPLIER = 3
HARD_PENDING_LIMIT = 1_000


class AsyncUDFSyncStream:
    """
    Async UDF execution engine with bounded concurrency, retries, and ordered results.

    High-level intuition:

    - Goal: Run an async function on many input items while:
        * Limiting concurrency (max_pending tasks in-flight)
        * Handling retries and timeouts per item
        * Yielding results in input order
        * Enforcing bounded memory usage (max_buffer_size)

    - Key idea: Use two “buckets” of state:
        1. pending: tasks currently running
        2. results: completed tasks that are waiting to be yielded

    - Flow:
        1. Schedule tasks up to max_pending / max_buffer_size
        2. Wait for tasks to finish
           - Usually wait for any task
           - If results buffer full, wait specifically for the next expected result
        3. Store completed results in a dict keyed by input index
        4. Yield results in input order (next_to_yield)
        5. Schedule more tasks if room exists
        6. Repeat until all items are processed

    - Guarantees:
        * Ordered output
        * Memory bounded by max_buffer_size
        * In-flight tasks bounded by max_pending
        * Retries and timeout handled per item
        * Safe cleanup of remaining tasks if iteration stops early
    """

    def __init__(
        self,
        fn: Callable[[Dict[str, Any]], Awaitable[Any]],
        *,
        loop: asyncio.AbstractEventLoop,
        max_concurrency: int,
        num_retries: int,
        timeout: float,
    ):
        self.fn = fn
        self.loop = loop
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_concurrency = max_concurrency
        self.num_retries = num_retries
        self.timeout = timeout

        # Maximum number of results that can be buffered before forcing yield
        self.max_buffer_size = min(HARD_BUFFER_LIMIT, DEFAULT_BUFFER_MULTIPLIER * max_concurrency)
        # Maximum number of tasks allowed to be in-flight
        self.max_pending = min(HARD_PENDING_LIMIT, DEFAULT_PENDING_MULTIPLIER * max_concurrency)

    async def _call(self, item: Any) -> Any:
        """Execute a single async call with retries, timeout, and concurrency control."""
        async with self.semaphore:
            last_err = None
            for attempt in range(self.num_retries + 1):
                try:
                    return await asyncio.wait_for(self.fn(item), timeout=self.timeout)
                except Exception as e:
                    last_err = e
                    msg = "Timeout" if isinstance(e, asyncio.TimeoutError) else f"Failure: {e}"
                    logger.warning(f"AsyncUDFStream: {msg} (attempt {attempt+1}/{self.num_retries+1})")

                    # Exponential backoff with jitter for retries
                    if attempt < self.num_retries:
                        # trunk-ignore(bandit/B311): pseudo random is safe
                        await asyncio.sleep(2**attempt + random.uniform(0, 2**attempt * 0.5))
            raise last_err  # All retries exhausted

    async def _call_batch_async(self, items: Iterable[Dict[str, Any]]) -> AsyncGenerator[Any, None]:
        """
        Async generator yielding results in input order.

        Key properties:
        - Bounded memory: results buffer never exceeds max_buffer_size
        - Bounded concurrency: in-flight tasks never exceed max_pending
        - Ordered output: results are yielded in input order
        - Retry and timeout logic per item
        """
        items_iter = enumerate(items)
        pending: dict[int, asyncio.Task] = {}  # index -> in-flight task
        results: dict[int, Any] = {}           # index -> result or exception
        next_to_yield = 0                       # index of the next result to yield
        exhausted = False                        # whether input iterator is exhausted

        # Wrap each task to always return (index, result_or_exception)
        async def call_with_index(idx: int, item: Any):
            try:
                res = await self._call(item)
                return idx, res
            except Exception as e:
                return idx, e

        def can_schedule_more():
            """Check if we can schedule more tasks without exceeding limits."""
            return not exhausted and len(pending) < self.max_pending and len(results) < self.max_buffer_size

        # Initial scheduling: fill up the pending tasks as much as allowed
        while can_schedule_more():
            try:
                idx, item = next(items_iter)
                pending[idx] = asyncio.create_task(call_with_index(idx, item))
            except StopIteration:
                exhausted = True
                break

        try:
            while pending or results:
                # Yield ready results in strict order
                while next_to_yield in results:
                    yield results.pop(next_to_yield)
                    next_to_yield += 1

                if not pending:
                    break  # no more tasks, we are done

                # Decide which tasks to wait for:
                # - If the buffer is full, wait specifically for the next needed result
                # - Otherwise, wait for any task to complete
                tasks_to_wait = {pending[next_to_yield]} if len(results) >= self.max_buffer_size else set(pending.values())
                done, _ = await asyncio.wait(tasks_to_wait, return_when=asyncio.FIRST_COMPLETED)

                # Collect all completed tasks
                for t in done:
                    idx, res = await t   # always returns (index, result_or_exception)
                    results[idx] = res
                    del pending[idx]

                # Schedule more tasks if we have room
                while can_schedule_more():
                    try:
                        idx, item = next(items_iter)
                        pending[idx] = asyncio.create_task(call_with_index(idx, item))
                    except StopIteration:
                        exhausted = True
                        break
        finally:
            # Clean up remaining tasks if the generator is closed early
            for t in pending.values():
                t.cancel()

    def call(self, items: Iterable[Dict[str, Any]]) -> Iterable[Any]:
        """
        Synchronous interface to the async UDF engine.

        - Yields results in order
        - Blocks on the existing event loop
        - Exceptions are returned as-is in the results
        """
        async_gen = self._call_batch_async(items)
        try:
            while True:
                fut = asyncio.run_coroutine_threadsafe(async_gen.__anext__(), self.loop)
                yield fut.result()
        except (StopIteration, StopAsyncIteration):
            return
