"""Async UDF execution engine for Fenic."""

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


class AsyncUDFSyncStream:
    """Executes async UDFs with bounded concurrency, retries, and ordered results.

    This class bridges async execution with sync iteration, allowing async UDFs
    to be used in Polars map_batches operations while maintaining result ordering
    and providing robust error handling.
    """

    def __init__(
        self,
        fn: Callable[[Dict[str, Any]], Awaitable[Any]],
        *,
        loop: asyncio.AbstractEventLoop,
        max_concurrency: int = 10,
        num_retries: int = 0,
        timeout: Optional[float] = None,  # per-item timeout in seconds
        max_buffer_size: int = 1000,  # limit memory used for results
    ):
        """Initialize the async UDF stream processor.

        Args:
            fn: Async function to execute for each item
            loop: Event loop to run async operations on
            max_concurrency: Maximum number of concurrent executions
            num_retries: Number of retries for failed items
            timeout: Per-item timeout in seconds
            max_buffer_size: Maximum number of results to buffer

        Raises:
            TypeError: If fn is not an async function
        """
        if not asyncio.iscoroutinefunction(fn):
            print(fn)
            raise TypeError("fn must be async")
        self.fn = fn
        self.loop = loop
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_concurrency = max_concurrency
        self.num_retries = num_retries
        self.timeout = timeout
        self.max_buffer_size = max_buffer_size
        self.pending_tasks = set()  # Track for cancellation

    async def _call(self, item: Any) -> Any:
        """Execute single async call with semaphore, retries, and timeout.

        Args:
            item: Input item to process

        Returns:
            Result from the async function

        Raises:
            Last exception if all retries fail
        """
        async with self.semaphore:
            last_err = None
            for attempt in range(self.num_retries + 1):
                try:
                    if self.timeout is not None:
                        return await asyncio.wait_for(self.fn(item), timeout=self.timeout)
                    else:
                        return await self.fn(item)
                except asyncio.TimeoutError as e:
                    last_err = e
                    logger.warning(f"Item timed out (attempt {attempt+1}/{self.num_retries+1})")
                except Exception as e:
                    last_err = e
                    logger.warning(f"Item failed (attempt {attempt+1}/{self.num_retries+1}): {e}")

                # Simple fixed delay with small jitter
                # Exponential backoff doesn't help much with high concurrency
                if attempt < self.num_retries:
                    delay = 1.0 + random.uniform(0, 0.5)  # 1-1.5 seconds
                    await asyncio.sleep(delay)
            raise last_err

    async def _call_batch_async(self, items: Iterable[Dict[str, Any]]) -> Iterable[Any]:
        """Async generator yielding results in input order with bounded buffer.

        This method manages concurrent execution while maintaining result ordering
        and limiting memory usage through a bounded buffer.

        Args:
            items: Iterable of items to process

        Yields:
            Results in the same order as input items (exceptions included)
        """
        items_iter = enumerate(items)
        results_buffer = {}
        next_index_to_yield = 0

        class IndexedTask:
            """Wrapper to track task index for ordering."""
            def __init__(self, idx, coro):
                self.idx = idx
                self.task = asyncio.create_task(coro)

        async def schedule_task(idx, item):
            """Schedule a task and set up its completion callback."""
            task_wrapper = IndexedTask(idx, self._call(item))
            self.pending_tasks.add(task_wrapper)

            def _done_callback(tw: IndexedTask):
                self.pending_tasks.discard(tw)
                try:
                    results_buffer[tw.idx] = tw.task.result()
                except Exception as e:
                    # Store exception as result - let caller decide how to handle
                    results_buffer[tw.idx] = e
                    logger.error(f"Item at index {tw.idx} failed permanently: {e}")

            task_wrapper.task.add_done_callback(lambda t: _done_callback(task_wrapper))

        # Initially schedule up to max_concurrency tasks
        for _ in range(self.max_concurrency):
            try:
                idx, item = next(items_iter)
                await schedule_task(idx, item)
            except StopIteration:
                break

        while self.pending_tasks or results_buffer:
            # Yield in order if available
            while next_index_to_yield in results_buffer:
                res = results_buffer.pop(next_index_to_yield)
                next_index_to_yield += 1

                # Always yield the result (success or exception)
                # Let the caller decide how to handle exceptions
                yield res

                # Schedule next items lazily but respect max_buffer_size
                try:
                    while len(results_buffer) < self.max_buffer_size:
                        idx, item = next(items_iter)
                        await schedule_task(idx, item)
                except StopIteration:
                    break

            if self.pending_tasks and next_index_to_yield not in results_buffer:
                # Wait for at least one task to complete
                await asyncio.wait(
                    [tw.task for tw in self.pending_tasks],
                    return_when=asyncio.FIRST_COMPLETED
                )

    def cancel_pending_tasks(self):
        """Cancel all pending tasks on fatal error.

        This method is called when a fatal error occurs (e.g., type mismatch)
        to prevent orphaned tasks from continuing execution.
        """
        for tw in self.pending_tasks:
            tw.task.cancel()
        logger.info(f"Cancelled {len(self.pending_tasks)} pending tasks")

    def call(self, items: Iterable[Dict[str, Any]]) -> Iterable[Any]:
        """Sync streaming entrypoint: yields results in order, blocking on existing loop.

        This method provides a synchronous interface to the async execution engine,
        allowing it to be used in Polars map_batches operations.

        Args:
            items: Iterable of items to process

        Yields:
            Results in the same order as input items

        Raises:
            Exception: Re-raises any exceptions that occur during processing
        """
        self.pending_tasks = set()  # Reset for new batch
        async_gen = self._call_batch_async(items)
        try:
            while True:
                coro = async_gen.__anext__()
                fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
                yield fut.result()
        except (StopIteration, StopAsyncIteration):
            return
        except Exception:
            # Cancel any remaining tasks on fatal error
            self.cancel_pending_tasks()
            raise
