"""Shared async utilities for Fenic backend operations."""

import asyncio
import atexit
import logging
import threading
from contextlib import contextmanager
from typing import Generator, Optional

logger = logging.getLogger(__name__)


class EventLoopManager:
    """Singleton managing shared event loop for all async operations.

    This class provides a centralized event loop that runs on a background thread,
    allowing multiple components (ModelClient, AsyncUDF, etc.) to share the same
    async infrastructure efficiently.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the manager with no active loop."""
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._manager_lock = threading.Lock()
        self._client_count = 0
        # Required to clean up event loop at exit if user doesn't call session.stop()
        self._shutdown_handler = None

    def get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Get existing loop or create new one on background thread.

        Returns:
            The shared event loop instance.
        """
        with self._manager_lock:
            if self.loop is None or self.loop.is_closed():
                self._create_event_loop()
            self._client_count += 1
            return self.loop

    def release_loop(self):
        """Decrement client count and shutdown loop if no clients remain."""
        loop_to_shutdown = None
        thread_to_join = None

        with self._manager_lock:
            self._client_count -= 1
            if self._client_count <= 0 and self.loop and self.loop.is_running():
                loop_to_shutdown = self.loop
                thread_to_join = self.thread
                self.loop = None
                self.thread = None
                self._client_count = 0
                if self._shutdown_handler:
                    atexit.unregister(self._shutdown_handler)
                    self._shutdown_handler = None

        # Shutdown outside lock to avoid deadlock
        if loop_to_shutdown:
            self._shutdown_loop(loop_to_shutdown, thread_to_join)

    def _create_event_loop(self):
        """Create and start event loop on background thread.

        Must be called while holding _manager_lock.
        """
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_event_loop,
            args=(self.loop,),
            daemon=True,
            name="EventLoopManager-Thread"
        )
        self.thread.start()

        # Wait for loop to start
        while not self.loop.is_running():
            pass

        self._shutdown_handler = atexit.register(self._shutdown_loop, self.loop, self.thread)
        logger.info("Created new event loop on background thread")

    def _run_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Run the event loop in the background thread."""
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            loop.close()

    def _shutdown_loop(self, loop: asyncio.AbstractEventLoop, thread: Optional[threading.Thread]):
        """Shutdown event loop and join thread."""
        try:
            # Cancel all tasks
            cancel_future = asyncio.run_coroutine_threadsafe(
                _cancel_event_loop_tasks(loop), loop
            )
            cancel_future.result()
        except Exception as e:
            logger.warning(f"Error cancelling tasks: {e}")

        # Stop the loop
        loop.call_soon_threadsafe(loop.stop)

        # Join thread
        if thread and thread.is_alive():
            thread.join()

        logger.info("Event loop shutdown complete")

    @contextmanager
    def loop_context(self) -> Generator[asyncio.AbstractEventLoop, None, None]:
        """Context manager for temporary event loop usage.

        Automatically acquires and releases the event loop reference.
        Perfect for short-lived operations like async UDF execution.

        Usage:
            with EventLoopManager().loop_context() as loop:
                # Use loop for async operations
                asyncio.run_coroutine_threadsafe(coro, loop)
        """
        loop = self.get_or_create_loop()
        try:
            yield loop
        finally:
            self.release_loop()


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
