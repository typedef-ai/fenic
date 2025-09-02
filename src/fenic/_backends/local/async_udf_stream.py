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

        self.max_buffer_size = min(HARD_BUFFER_LIMIT, DEFAULT_BUFFER_MULTIPLIER * max_concurrency)
        self.max_pending = min(HARD_PENDING_LIMIT, DEFAULT_PENDING_MULTIPLIER * max_concurrency)

        self._pending: Dict[int, asyncio.Task] = {}
        self._results: Dict[int, Any] = {}
        self._next_idx_to_yield: int = 0
        self._exhausted: bool = False

    def call(self, items: Iterable[Dict[str, Any]]) -> Iterable[Any]:
        """Synchronous interface: yields results in order, blocks on loop."""
        async_gen = self._call_batch_async(items)
        try:
            while True:
                fut = asyncio.run_coroutine_threadsafe(async_gen.__anext__(), self.loop)
                yield fut.result()
        except (StopIteration, StopAsyncIteration):
            return

    async def _call_batch_async(self, items: Iterable[Dict[str, Any]]) -> AsyncGenerator[Any, None]:
        """
        Async generator yielding results in input order.
        """
        items_iter = enumerate(items)

        async def call_with_index(idx: int, item: Any):
            try:
                res = await self._call(item)
                return idx, res
            except Exception as e:
                return idx, e

        def can_schedule_more():
            return not self._exhausted and len(self._pending) < self.max_pending and len(self._results) < self.max_buffer_size

        # Initial scheduling
        while can_schedule_more():
            try:
                idx, item = next(items_iter)
                task = asyncio.create_task(call_with_index(idx, item))
                self._pending[idx] = task
            except StopIteration:
                self._exhausted = True
                break

        try:
            while self._pending or self._results:
                # Yield ready results in order
                while self._next_idx_to_yield in self._results:
                    yield self._results.pop(self._next_idx_to_yield)
                    self._next_idx_to_yield += 1

                if not self._pending:
                    break

                # Wait logic: either wait for specific next result if buffer is full, else any task
                tasks_to_wait = {self._pending[self._next_idx_to_yield]} if len(self._results) >= self.max_buffer_size else set(self._pending.values())
                done, _ = await asyncio.wait(tasks_to_wait, return_when=asyncio.FIRST_COMPLETED)

                # Process completed tasks
                for t in done:
                    idx, res = t.result()
                    self._results[idx] = res
                    self._pending.pop(idx, None)

                # Schedule more tasks if room
                while can_schedule_more():
                    try:
                        idx, item = next(items_iter)
                        task = asyncio.create_task(call_with_index(idx, item))
                        self._pending[idx] = task
                    except StopIteration:
                        self._exhausted = True
                        break
        finally:
            # Cancel remaining tasks if generator exits early
            self._cancel_pending_tasks()

    async def _call(self, item: Any) -> Any:
        """Execute a single async call with retries, timeout, and concurrency control."""
        async with self.semaphore:
            last_err = None
            for attempt in range(self.num_retries + 1):
                try:
                    return await asyncio.wait_for(self.fn(item), timeout=self.timeout)
                except Exception as e:
                    last_err = e
                    exc_msg = str(e) or "<no error message>"
                    if attempt < self.num_retries:
                        # trunk-ignore(bandit/B311): pseudo random is safe
                        backoff_delay = 2**attempt + random.uniform(0, 2**attempt * 0.5)
                        logger.info(
                            "Attempt %d/%d failed for input=%r "
                            "with %s: %s. Retrying in %.2fs...",
                            attempt + 1,
                            self.num_retries + 1,
                            item,
                            type(e).__name__,
                            exc_msg,
                            backoff_delay,
                        )
                        await asyncio.sleep(backoff_delay)
            logger.warning(
                "All %d attempt(s) failed for input=%r. "
                "Last error was %s: %s",
                self.num_retries + 1,
                item,
                type(last_err).__name__,
                str(last_err) or "<no error message>",
            )
            raise last_err

    def _cancel_pending_tasks(self):
        """Externally cancel all currently pending tasks."""
        for t in self._pending.values():
            t.cancel()
        self._pending.clear()
